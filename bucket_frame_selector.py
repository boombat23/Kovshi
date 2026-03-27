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
MIN_GAP_AREA = 400
MAX_GAP_AREA = 80000
MIN_GAP_BBOX_W = 20
MAX_GAP_BBOX_W = 1000
MIN_GAP_BBOX_H = 8
MAX_GAP_BBOX_H = 400
MIN_GAP_WIDTH_RATIO = 0.2
SYMMETRY_TOLERANCE_PX = 12
MAX_SYMMETRY_FRACTION = 0.12
MIN_GAP_ASPECT_RATIO = 3.0
MIN_GAP_BLACKNESS = 8.0
MIN_GAP_CONTRAST = 6.0
MIN_GAP_X_OVERLAP_RATIO = 0.4
MIN_GAP_WIDTH_SIMILARITY = 0.6
MAX_MISSING_FRAMES = 6
KERNEL_SIZE = 5
MIN_FRAMES_BETWEEN_BUCKETS = 2
CENTER_GATE_PX = 120

SHOW_PREVIEW = True
SAVE_DEBUG_VIDEO = False
DEBUG_VIDEO_PATH = Path("debug_preview.mp4")
DRAW_SELECTED_OVERLAY = True


@dataclass
class TrackState:
    active: bool = False
    missing_frames: int = 0
    best_score: float = float("-inf")
    best_frame: Optional[np.ndarray] = None
    best_frame_idx: int = -1
    best_gap_bboxes: Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]] = None
    last_seen_frame_idx: int = -1
    bucket_index: int = 0


@dataclass
class RuntimeConfig:
    video_path: Path = VIDEO_PATH
    output_dir: Path = OUTPUT_DIR
    roi: Tuple[int, int, int, int] = ROI
    min_gap_area: int = MIN_GAP_AREA
    max_gap_area: int = MAX_GAP_AREA
    min_gap_bbox_w: int = MIN_GAP_BBOX_W
    max_gap_bbox_w: int = MAX_GAP_BBOX_W
    min_gap_bbox_h: int = MIN_GAP_BBOX_H
    max_gap_bbox_h: int = MAX_GAP_BBOX_H
    min_gap_width_ratio: float = MIN_GAP_WIDTH_RATIO
    symmetry_tolerance_px: int = SYMMETRY_TOLERANCE_PX
    max_symmetry_fraction: float = MAX_SYMMETRY_FRACTION
    min_gap_aspect_ratio: float = MIN_GAP_ASPECT_RATIO
    min_gap_blackness: float = MIN_GAP_BLACKNESS
    min_gap_contrast: float = MIN_GAP_CONTRAST
    min_gap_x_overlap_ratio: float = MIN_GAP_X_OVERLAP_RATIO
    min_gap_width_similarity: float = MIN_GAP_WIDTH_SIMILARITY
    max_missing_frames: int = MAX_MISSING_FRAMES
    kernel_size: int = KERNEL_SIZE
    min_frames_between_buckets: int = MIN_FRAMES_BETWEEN_BUCKETS
    center_gate_px: int = CENTER_GATE_PX
    show_preview: bool = SHOW_PREVIEW
    save_debug_video: bool = SAVE_DEBUG_VIDEO
    debug_video_path: Path = DEBUG_VIDEO_PATH
    draw_selected_overlay: bool = DRAW_SELECTED_OVERLAY


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
    if config.min_gap_area <= 0 or config.max_gap_area <= 0 or config.min_gap_area >= config.max_gap_area:
        raise ValueError("Gap area thresholds must satisfy 0 < min_gap_area < max_gap_area")

    if config.min_gap_bbox_w <= 0 or config.min_gap_bbox_h <= 0:
        raise ValueError("Minimum gap bbox dimensions must be positive")

    if config.max_gap_bbox_w < config.min_gap_bbox_w or config.max_gap_bbox_h < config.min_gap_bbox_h:
        raise ValueError("Maximum gap bbox dimensions must be >= minimum gap bbox dimensions")

    if not (0.0 < config.min_gap_width_ratio <= 1.0):
        raise ValueError("min_gap_width_ratio must be in (0, 1]")

    if config.min_gap_aspect_ratio <= 0:
        raise ValueError("min_gap_aspect_ratio must be > 0")

    if config.min_gap_blackness < 0 or config.min_gap_contrast < 0:
        raise ValueError("Darkness constraints must be non-negative")

    if not (0.0 < config.max_symmetry_fraction <= 1.0):
        raise ValueError("max_symmetry_fraction must be in (0, 1]")

    if not (0.0 <= config.min_gap_x_overlap_ratio <= 1.0):
        raise ValueError("min_gap_x_overlap_ratio must be in [0, 1]")

    if not (0.0 < config.min_gap_width_similarity <= 1.0):
        raise ValueError("min_gap_width_similarity must be in (0, 1]")

    if config.kernel_size < 1:
        raise ValueError("kernel_size must be >= 1")

    if config.kernel_size % 2 == 0:
        config.kernel_size += 1

    if config.max_missing_frames < 0 or config.min_frames_between_buckets < 0:
        raise ValueError("Frame-gap parameters must be non-negative")

    if config.center_gate_px < 0 or config.symmetry_tolerance_px < 0:
        raise ValueError("center_gate_px and symmetry_tolerance_px must be non-negative")

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


def extract_dark_mask(roi_frame: np.ndarray, kernel_size: int) -> np.ndarray:
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    dark_threshold = int(np.clip(np.percentile(blur, 32), 10, 140))
    _, dark_mask = cv2.threshold(blur, dark_threshold, 255, cv2.THRESH_BINARY_INV)

    # Dark horizontal slits between buckets stand out after a horizontal black-hat.
    horiz_kernel_w = max(15, ((blur.shape[1] // 8) // 2) * 2 + 1)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_kernel_w, 3))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, horiz_kernel)
    line_threshold = int(np.clip(np.percentile(blackhat, 82), 8, 120))
    _, line_mask = cv2.threshold(blackhat, line_threshold, 255, cv2.THRESH_BINARY)

    combined = cv2.bitwise_or(dark_mask, line_mask)
    combined = preprocess_mask(combined, kernel_size)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, kernel_size * 4), max(3, kernel_size - 1)))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel)
    return combined


def x_overlap_ratio(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, _, aw, _ = a
    bx1, _, bw, _ = b
    ax2 = ax1 + aw
    bx2 = bx1 + bw
    overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
    min_w = max(1, min(aw, bw))
    return overlap / float(min_w)


def select_gap_pair(
    mask: np.ndarray,
    roi_frame: np.ndarray,
    config: RuntimeConfig,
) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[Tuple[int, int, int, int, float, float, float]] = []

    roi_h, roi_w = mask.shape[:2]
    roi_center_y = roi_h / 2.0
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

    roi_mean_gray = float(np.mean(gray))

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < config.min_gap_area or area > config.max_gap_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if not (config.min_gap_bbox_w <= w <= config.max_gap_bbox_w):
            continue
        if not (config.min_gap_bbox_h <= h <= config.max_gap_bbox_h):
            continue
        if w < roi_w * config.min_gap_width_ratio:
            continue
        if (w / max(h, 1)) < config.min_gap_aspect_ratio:
            continue

        center_y = y + h / 2.0
        distance_to_horizon = abs(center_y - roi_center_y)
        if distance_to_horizon > config.center_gate_px:
            continue

        pad = max(1, int(round(h * 0.6)))
        y0 = max(0, y)
        y1 = min(roi_h, y + h)
        ext0 = max(0, y - pad)
        ext1 = min(roi_h, y + h + pad)
        band = gray[y0:y1, x : x + w]
        surround = gray[ext0:ext1, x : x + w]
        if band.size == 0 or surround.size == 0:
            continue
        band_mean = float(np.mean(band))
        surround_mean = float(np.mean(surround))
        contrast = surround_mean - band_mean
        if band_mean > (roi_mean_gray - config.min_gap_blackness):
            continue
        if band_mean > (surround_mean - 1.0):
            continue
        if contrast < config.min_gap_contrast:
            continue

        candidates.append((x, y, w, h, center_y, area, contrast))

    if len(candidates) < 2:
        return None

    upper = [c for c in candidates if c[4] < roi_center_y]
    lower = [c for c in candidates if c[4] >= roi_center_y]
    if not upper or not lower:
        return None

    best_pair = None
    best_symmetry_error = float("inf")
    best_total_area = float("-inf")
    best_pair_contrast = float("-inf")
    max_allowed_symmetry = max(config.symmetry_tolerance_px, roi_h * config.max_symmetry_fraction)

    for up in upper:
        for low in lower:
            dist_up = roi_center_y - up[4]
            dist_low = low[4] - roi_center_y
            symmetry_error = abs(dist_up - dist_low)
            if symmetry_error > max_allowed_symmetry:
                continue

            up_box = (up[0], up[1], up[2], up[3])
            low_box = (low[0], low[1], low[2], low[3])
            if x_overlap_ratio(up_box, low_box) < config.min_gap_x_overlap_ratio:
                continue

            width_similarity = min(up[2], low[2]) / float(max(up[2], low[2]))
            if width_similarity < config.min_gap_width_similarity:
                continue

            total_area = up[5] + low[5]
            pair_contrast = up[6] + low[6]
            if symmetry_error < best_symmetry_error or (
                symmetry_error == best_symmetry_error
                and (pair_contrast > best_pair_contrast or (pair_contrast == best_pair_contrast and total_area > best_total_area))
            ):
                best_symmetry_error = symmetry_error
                best_pair_contrast = pair_contrast
                best_total_area = total_area
                best_pair = (up_box, low_box)

    if best_pair is None:
        return None

    return best_pair[0], best_pair[1], best_symmetry_error


def laplacian_sharpness(frame_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def score_gap_pair(
    roi_frame: np.ndarray,
    gap_pair: Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]],
    symmetry_error: float,
) -> Tuple[float, float]:
    (x1, y1, w1, h1), (x2, y2, w2, h2) = gap_pair
    crop_up = roi_frame[y1 : y1 + h1, x1 : x1 + w1]
    crop_down = roi_frame[y2 : y2 + h2, x2 : x2 + w2]
    if crop_up.size == 0 or crop_down.size == 0:
        return float("-inf"), 0.0

    sharpness = (laplacian_sharpness(crop_up) + laplacian_sharpness(crop_down)) / 2.0
    score = -symmetry_error + 0.01 * sharpness
    return score, sharpness


def draw_debug(
    frame: np.ndarray,
    roi: Tuple[int, int, int, int],
    gap_bboxes_roi: Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]],
    score: Optional[float],
    sharpness: Optional[float],
    symmetry_error: Optional[float],
    bucket_idx: int,
    active: bool,
) -> np.ndarray:
    x1, y1, x2, y2 = roi
    vis = frame.copy()

    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
    center_y = (y1 + y2) // 2
    cv2.line(vis, (x1, center_y), (x2, center_y), (255, 255, 0), 2)

    if gap_bboxes_roi is not None:
        for bx, by, bw, bh in gap_bboxes_roi:
            cv2.rectangle(vis, (x1 + bx, y1 + by), (x1 + bx + bw, y1 + by + bh), (0, 255, 0), 2)

    status = "TRACKING" if active else "IDLE"
    lines = [f"Bucket: {bucket_idx + (1 if active else 0)}", f"State: {status}"]
    if score is not None and sharpness is not None:
        lines.append(f"Score: {score:.2f}")
        lines.append(f"Sharpness: {sharpness:.2f}")
    if symmetry_error is not None:
        lines.append(f"Symmetry err: {symmetry_error:.2f}")

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
    track.best_gap_bboxes = None


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

    track = TrackState()
    frame_idx = 0

    def process_frame(frame: np.ndarray, current_idx: int) -> np.ndarray:
        roi_frame = frame[y1:y2, x1:x2]
        dark_mask = extract_dark_mask(roi_frame, config.kernel_size)
        detection = select_gap_pair(dark_mask, roi_frame, config)
        score = None
        sharpness = None
        symmetry_error = None
        gap_bboxes = None

        if detection is not None:
            up_gap, down_gap, sym_error = detection
            gap_bboxes = (up_gap, down_gap)
            cand_score, cand_sharp = score_gap_pair(roi_frame, gap_bboxes, sym_error)
            score = cand_score
            sharpness = cand_sharp
            symmetry_error = sym_error

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
                    if config.draw_selected_overlay:
                        track.best_frame = draw_debug(
                            frame=frame,
                            roi=config.roi,
                            gap_bboxes_roi=gap_bboxes,
                            score=score,
                            sharpness=sharpness,
                            symmetry_error=symmetry_error,
                            bucket_idx=track.bucket_index,
                            active=track.active,
                        )
                    else:
                        track.best_frame = frame.copy()
                    track.best_frame_idx = current_idx
                    track.best_gap_bboxes = gap_bboxes

        elif track.active:
            track.missing_frames += 1
            if track.missing_frames > config.max_missing_frames:
                finalize_bucket(track, config.output_dir)
                reset_track(track)

        return draw_debug(
            frame=frame,
            roi=config.roi,
            gap_bboxes_roi=gap_bboxes,
            score=score,
            sharpness=sharpness,
            symmetry_error=symmetry_error,
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
