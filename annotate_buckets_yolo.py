#!/usr/bin/env python3
"""Интерактивная разметка кадров bounding boxes класса `bucket` в YOLO-формате.

Скрипт открывает изображения по очереди, позволяет поставить несколько bbox на кадр,
и сохраняет аннотации в txt, совместимый с дообучением YOLO.

Пример:
    python annotate_buckets_yolo.py \
        --image-dirs extracted_frames/4 extracted_frames/5 \
        --labels-root extracted_frames/labels
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_ID_BUCKET = 0


@dataclass
class AppState:
    image_paths: List[Path]
    labels_root: Path
    class_id: int
    index: int = 0
    boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    drawing: bool = False
    start_point: Tuple[int, int] | None = None
    temp_end: Tuple[int, int] | None = None


class YoloBucketAnnotator:
    def __init__(self, state: AppState, window_name: str = "YOLO Bucket Annotator") -> None:
        self.state = state
        self.window_name = window_name
        self.current_image = None

    def labels_path_for(self, image_path: Path) -> Path:
        """Сопоставляет `extracted_frames/X/img.jpg` -> `labels_root/X/img.txt`."""
        try:
            relative = image_path.relative_to(image_path.parents[1])
            # relative == '<dir_name>/<filename>' when input is extracted_frames/4/*.jpg
        except ValueError:
            relative = image_path.name
        return (self.state.labels_root / relative).with_suffix(".txt")

    def load_boxes(self, image_path: Path, width: int, height: int) -> List[Tuple[int, int, int, int]]:
        label_path = self.labels_path_for(image_path)
        if not label_path.exists():
            return []

        boxes: List[Tuple[int, int, int, int]] = []
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cls_id, xc, yc, bw, bh = map(float, parts)
            except ValueError:
                continue
            if int(cls_id) != self.state.class_id:
                continue
            x1 = int((xc - bw / 2) * width)
            y1 = int((yc - bh / 2) * height)
            x2 = int((xc + bw / 2) * width)
            y2 = int((yc + bh / 2) * height)
            boxes.append(self._clip_box((x1, y1, x2, y2), width, height))
        return boxes

    def save_boxes(self, image_path: Path, boxes: List[Tuple[int, int, int, int]], width: int, height: int) -> None:
        label_path = self.labels_path_for(image_path)
        label_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        for x1, y1, x2, y2 in boxes:
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            xc = x1 + bw / 2
            yc = y1 + bh / 2
            lines.append(
                f"{self.state.class_id} {xc / width:.6f} {yc / height:.6f} {bw / width:.6f} {bh / height:.6f}"
            )

        if lines:
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif label_path.exists():
            label_path.unlink()

    @staticmethod
    def _clip_box(box: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width - 1, x2))))
        y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height - 1, y2))))
        return x1, y1, x2, y2

    def _render(self) -> None:
        canvas = self.current_image.copy()

        for idx, (x1, y1, x2, y2) in enumerate(self.state.boxes, start=1):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(
                canvas,
                f"bucket#{idx}",
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (50, 255, 50),
                1,
                cv2.LINE_AA,
            )

        if self.state.drawing and self.state.start_point and self.state.temp_end:
            cv2.rectangle(canvas, self.state.start_point, self.state.temp_end, (0, 255, 255), 1)

        help_lines = [
            "LMB: draw box | U: undo | C: clear",
            "S/ENTER: save+next | N: next(no save) | P: previous",
            "Q/ESC: quit",
        ]
        y = 20
        for line in help_lines:
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            y += 24

        cv2.imshow(self.window_name, canvas)

    def _mouse_handler(self, event: int, x: int, y: int, _flags: int, _userdata) -> None:
        h, w = self.current_image.shape[:2]
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.state.drawing = True
            self.state.start_point = (x, y)
            self.state.temp_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.state.drawing:
            self.state.temp_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.state.drawing and self.state.start_point:
            self.state.drawing = False
            x1, y1 = self.state.start_point
            x2, y2 = x, y
            box = self._clip_box((x1, y1, x2, y2), w, h)
            if abs(box[2] - box[0]) > 3 and abs(box[3] - box[1]) > 3:
                self.state.boxes.append(box)
            self.state.start_point = None
            self.state.temp_end = None

    def _open_current(self) -> bool:
        if not (0 <= self.state.index < len(self.state.image_paths)):
            return False

        image_path = self.state.image_paths[self.state.index]
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Не удалось открыть: {image_path}")
            return False

        self.current_image = image
        h, w = image.shape[:2]
        self.state.boxes = self.load_boxes(image_path, w, h)
        print(
            f"[{self.state.index + 1}/{len(self.state.image_paths)}] {image_path} | "
            f"loaded boxes: {len(self.state.boxes)}"
        )
        return True

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_handler)

        while 0 <= self.state.index < len(self.state.image_paths):
            if not self._open_current():
                self.state.index += 1
                continue

            while True:
                self._render()
                key = cv2.waitKey(20) & 0xFF

                if key in (ord("q"), 27):
                    cv2.destroyAllWindows()
                    return
                if key == ord("u"):
                    if self.state.boxes:
                        self.state.boxes.pop()
                elif key == ord("c"):
                    self.state.boxes.clear()
                elif key in (ord("s"), 13):
                    image_path = self.state.image_paths[self.state.index]
                    h, w = self.current_image.shape[:2]
                    self.save_boxes(image_path, self.state.boxes, w, h)
                    print(f"[SAVE] {image_path.name}: {len(self.state.boxes)} box(es)")
                    self.state.index += 1
                    break
                elif key == ord("n"):
                    self.state.index += 1
                    break
                elif key == ord("p"):
                    self.state.index = max(0, self.state.index - 1)
                    break

        cv2.destroyAllWindows()


def collect_images(image_dirs: List[Path]) -> List[Path]:
    images: List[Path] = []
    for directory in image_dirs:
        if not directory.exists():
            print(f"[WARN] Папка не найдена: {directory}")
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                images.append(path)
    return images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Разметка bucket bbox в YOLO txt формате")
    parser.add_argument(
        "--image-dirs",
        nargs="+",
        type=Path,
        default=[Path("extracted_frames/4"), Path("extracted_frames/5")],
        help="Список директорий с кадрами (по умолчанию extracted_frames/4 extracted_frames/5)",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=Path("extracted_frames/labels"),
        help="Корень для txt-аннотаций YOLO",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=CLASS_ID_BUCKET,
        help="ID класса bucket (обычно 0)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_paths = collect_images(args.image_dirs)
    if not image_paths:
        raise SystemExit("Не найдено изображений для разметки.")

    print(f"Найдено изображений: {len(image_paths)}")
    print("Формат сохранения: <class_id> <x_center> <y_center> <width> <height> (нормализованные)")

    state = AppState(image_paths=image_paths, labels_root=args.labels_root, class_id=args.class_id)
    YoloBucketAnnotator(state).run()


if __name__ == "__main__":
    main()
