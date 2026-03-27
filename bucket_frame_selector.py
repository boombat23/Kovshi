#!/usr/bin/env python3
"""Select one best frame for each passing elevator bucket from a video."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# =========================
# Tunable parameters
# =========================
VIDEO_PATH = Path(r"kovshi_video\3.mp4")
OUTPUT_DIR = Path("output_frames")

ROI = (100, 100, 650, 900)  # (x1, y1, x2, y2) in full-frame coordinates
MIN_AREA = 1200
MAX_AREA = 120000
MIN_BBOX_W = 25
MAX_BBOX_W = 1000
MIN_BBOX_H = 25
MAX_BBOX_H = 1000
ALPHA = 0.9
MAX_MISSING_FRAMES = 6
KERNEL_SIZE = 5
MIN_FRAMES_BETWEEN_BUCKETS = 2
CENTER_GATE_PX = 120

SHOW_PREVIEW = True
SAVE_DEBUG_VIDEO = False
DEBUG_VIDEO_PATH = Path("debug_preview.mp4")


@dataclass
class TrackState:
    active: bool = False
    missing_frames: int = 0
    best_score: float = float("-inf")
    best_frame: Optional[np.ndarray] = None
    best_frame_idx: int = -1
    best_bbox: Optional[Tuple[int, int, int, int]] = None
    last_seen_frame_idx: int = -1
    bucket_index: int = 0


@dataclass
class RuntimeConfig:
    video_path: Path = VIDEO_PATH
    output_dir: Path = OUTPUT_DIR
    roi: Tuple[int, int, int, int] = ROI
    min_area: int = MIN_AREA
    max_area: int = MAX_AREA
    min_bbox_w: int = MIN_BBOX_W
    max_bbox_w: int = MAX_BBOX_W
    min_bbox_h: int = MIN_BBOX_H
    max_bbox_h: int = MAX_BBOX_H
    alpha: float = ALPHA
    max_missing_frames: int = MAX_MISSING_FRAMES
    kernel_size: int = KERNEL_SIZE
    min_frames_between_buckets: int = MIN_FRAMES_BETWEEN_BUCKETS
    center_gate_px: int = CENTER_GATE_PX
    show_preview: bool = SHOW_PREVIEW
    save_debug_video: bool = SAVE_DEBUG_VIDEO
    debug_video_path: Path = DEBUG_VIDEO_PATH


@dataclass
class VideoContext:
    fps: float
    frame_size: Tuple[int, int]


def validate_roi(roi: Tuple[int, int, int, int], frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    x1 = max(0, min(x1, frame_w - 1))
    x2 = max(1, min(x2, frame_w))
    y1 = max(0, min(y1, frame_h - 1))
    y2 = max(1, min(y2, frame_h))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid ROI after clipping: {(x1, y1, x2, y2)}")

    return x1, y1, x2, y2




def validate_config(config: RuntimeConfig) -> RuntimeConfig:
    if config.min_area <= 0 or config.max_area <= 0 or config.min_area >= config.max_area:
        raise ValueError("Area thresholds must satisfy 0 < min_area < max_area")

    if config.min_bbox_w <= 0 or config.min_bbox_h <= 0:
        raise ValueError("Minimum bbox dimensions must be positive")

    if config.max_bbox_w < config.min_bbox_w or config.max_bbox_h < config.min_bbox_h:
        raise ValueError("Maximum bbox dimensions must be >= minimum bbox dimensions")

    if config.kernel_size < 1:
        raise ValueError("kernel_size must be >= 1")

    if config.kernel_size % 2 == 0:
        config.kernel_size += 1

    if config.max_missing_frames < 0 or config.min_frames_between_buckets < 0:
        raise ValueError("Frame-gap parameters must be non-negative")

    if config.center_gate_px < 0:
        raise ValueError("center_gate_px must be non-negative")

    return config

def create_debug_writer(path: Path, ctx: VideoContext) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, max(ctx.fps, 1.0), ctx.frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create debug video writer: {path}")
    return writer


def preprocess_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def select_best_contour(mask: np.ndarray, config: RuntimeConfig) -> Optional[Tuple[int, int, int, int, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    _, roi_w = mask.shape[:2]
    roi_center_x = roi_w / 2.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < config.min_area or area > config.max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if not (config.min_bbox_w <= w <= config.max_bbox_w):
            continue
        if not (config.min_bbox_h <= h <= config.max_bbox_h):
            continue

        center_x = x + w / 2.0
        distance_to_center = abs(center_x - roi_center_x)
        if distance_to_center > config.center_gate_px:
            continue

        candidates.append((x, y, w, h, distance_to_center, area))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[4], -item[5]))
    x, y, w, h, dist, _ = candidates[0]
    return x, y, w, h, dist


def laplacian_sharpness(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def score_candidate(
    roi_frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    distance_to_center: float,
    alpha: float,
) -> Tuple[float, float]:
    x, y, w, h = bbox
    crop = roi_frame[y : y + h, x : x + w]
    if crop.size == 0:
        return float("-inf"), 0.0

    sharpness = laplacian_sharpness(crop)
    score = sharpness - alpha * distance_to_center
    return score, sharpness


def draw_debug(
    frame: np.ndarray,
    roi: Tuple[int, int, int, int],
    bbox_roi: Optional[Tuple[int, int, int, int]],
    score: Optional[float],
    sharpness: Optional[float],
    bucket_idx: int,
    active: bool,
) -> np.ndarray:
    x1, y1, x2, y2 = roi
    vis = frame.copy()

    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
    center_x = (x1 + x2) // 2
    cv2.line(vis, (center_x, y1), (center_x, y2), (255, 255, 0), 2)

    if bbox_roi is not None:
        bx, by, bw, bh = bbox_roi
        cv2.rectangle(vis, (x1 + bx, y1 + by), (x1 + bx + bw, y1 + by + bh), (0, 255, 0), 2)

    status = "TRACKING" if active else "IDLE"
    lines = [f"Bucket: {bucket_idx + (1 if active else 0)}", f"State: {status}"]
    if score is not None and sharpness is not None:
        lines.append(f"Score: {score:.2f}")
        lines.append(f"Sharpness: {sharpness:.2f}")

    y_text = 30
    for line in lines:
        cv2.putText(
            vis,
            line,
            (20, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y_text += 30

    return vis


def finalize_bucket(track: TrackState, output_dir: Path) -> None:
    if track.best_frame is None:
        return

    track.bucket_index += 1
    output_path = output_dir / f"bucket_{track.bucket_index:05d}.jpg"
    cv2.imwrite(str(output_path), track.best_frame)


def reset_track(track: TrackState) -> None:
    track.active = False
    track.missing_frames = 0
    track.best_score = float("-inf")
    track.best_frame = None
    track.best_frame_idx = -1
    track.best_bbox = None


def run(config: RuntimeConfig) -> None:
    config = validate_config(config)

    if not config.video_path.exists():
        raise FileNotFoundError(f"Video not found: {config.video_path}")

    cap = cv2.VideoCapture(str(config.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {config.video_path}")

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise RuntimeError("Video is empty or first frame cannot be read")

    frame_h, frame_w = first_frame.shape[:2]
    config.roi = validate_roi(config.roi, frame_w, frame_h)
    x1, y1, x2, y2 = config.roi

    config.output_dir.mkdir(parents=True, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    ctx = VideoContext(fps=fps if fps > 0 else 25.0, frame_size=(frame_w, frame_h))
    debug_writer = create_debug_writer(config.debug_video_path, ctx) if config.save_debug_video else None

    back_sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=20, detectShadows=False)

    track = TrackState()
    frame_idx = 0

    def process_frame(frame: np.ndarray, current_idx: int) -> np.ndarray:
        roi_frame = frame[y1:y2, x1:x2]
        fg_mask = back_sub.apply(roi_frame)
        cleaned_mask = preprocess_mask(fg_mask, config.kernel_size)

        detection = select_best_contour(cleaned_mask, config)
        score = None
        sharpness = None
        bbox = None

        if detection is not None:
            bx, by, bw, bh, dist = detection
            bbox = (bx, by, bw, bh)
            cand_score, cand_sharp = score_candidate(roi_frame, bbox, dist, config.alpha)
            score = cand_score
            sharpness = cand_sharp

            if not track.active and current_idx - track.last_seen_frame_idx >= config.min_frames_between_buckets:
                track.active = True
                track.missing_frames = 0
                track.best_score = float("-inf")
                track.best_frame = None

            if track.active:
                track.missing_frames = 0
                track.last_seen_frame_idx = current_idx

                if cand_score > track.best_score:
                    track.best_score = cand_score
                    track.best_frame = frame.copy()
                    track.best_frame_idx = current_idx
                    track.best_bbox = bbox

        elif track.active:
            track.missing_frames += 1
            if track.missing_frames > config.max_missing_frames:
                finalize_bucket(track, config.output_dir)
                reset_track(track)

        return draw_debug(
            frame=frame,
            roi=config.roi,
            bbox_roi=bbox,
            score=score,
            sharpness=sharpness,
            bucket_idx=track.bucket_index,
            active=track.active,
        )

    debug_frame = process_frame(first_frame, frame_idx)

    if config.show_preview:
        cv2.imshow("Bucket Selector Debug", debug_frame)
    if debug_writer is not None:
        debug_writer.write(debug_frame)

    frame_idx += 1

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        debug_frame = process_frame(frame, frame_idx)

        if config.show_preview:
            cv2.imshow("Bucket Selector Debug", debug_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        if debug_writer is not None:
            debug_writer.write(debug_frame)

        frame_idx += 1

    if track.active:
        finalize_bucket(track, config.output_dir)
        reset_track(track)

    cap.release()
    if debug_writer is not None:
        debug_writer.release()
    if config.show_preview:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cfg = RuntimeConfig()

    try:
        run(cfg)
        print("Done. Best frames are saved to:", cfg.output_dir)
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)
