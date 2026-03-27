#!/usr/bin/env python3
"""Extract every Nth frame from selected videos into separate folders."""

from __future__ import annotations

from pathlib import Path

import cv2

VIDEO_DIR = Path("kovshi_video")
VIDEO_NAMES = ("4.mp4", "5.mp4")
FRAME_STEP = 5
OUTPUT_ROOT = Path("extracted_frames")


def extract_every_nth_frame(video_path: Path, output_dir: Path, step: int) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    saved_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % step == 0:
            out_file = output_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(out_file), frame)
            saved_count += 1

        frame_index += 1

    cap.release()
    return frame_index, saved_count


def main() -> None:
    if FRAME_STEP <= 0:
        raise ValueError("FRAME_STEP должен быть больше 0")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for video_name in VIDEO_NAMES:
        video_path = VIDEO_DIR / video_name
        if not video_path.exists():
            print(f"[SKIP] Файл не найден: {video_path}")
            continue

        video_output_dir = OUTPUT_ROOT / video_path.stem
        total_frames, saved_frames = extract_every_nth_frame(video_path, video_output_dir, FRAME_STEP)
        print(
            f"[OK] {video_name}: всего кадров {total_frames}, сохранено {saved_frames} в {video_output_dir}"
        )


if __name__ == "__main__":
    main()
