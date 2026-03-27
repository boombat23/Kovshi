#!/usr/bin/env python3
"""Дообучение YOLO26/Ultralytics YOLO на размеченных bucket-кадрах.

Сценарий под ваш кейс:
- есть размеченные кадры в extracted_frames/4 и extracted_frames/5
- метки YOLO лежат в extracted_frames/labels/4 и extracted_frames/labels/5
- хотите использовать все размеченные кадры для обучения

Скрипт автоматически:
1) находит только кадры, у которых есть .txt метка;
2) формирует train/val списки (в режиме --use-all-for-train val = train);
3) генерирует dataset YAML;
4) запускает обучение и сохраняет best/last веса в runs/<project>/<name>/weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Дообучение YOLO26 на bucket-кадрах")
    parser.add_argument(
        "--images-dirs",
        nargs="+",
        type=Path,
        default=[Path("extracted_frames/4"), Path("extracted_frames/5")],
        help="Папки с изображениями",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=Path("extracted_frames/labels"),
        help="Корень папки с YOLO txt метками",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("training_artifacts/bucket_yolo26"),
        help="Куда сохранять train.txt/val.txt/data.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n.pt",
        help="Базовая модель для дообучения (например yolo26n.pt или yolo11n.pt)",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Количество эпох")
    parser.add_argument("--imgsz", type=int, default=960, help="Размер картинки для обучения")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cpu", help="CUDA device, либо cpu")
    parser.add_argument("--project", type=str, default="runs/train", help="Папка проекта в runs")
    parser.add_argument("--name", type=str, default="bucket_yolo26_finetune", help="Имя запуска")
    parser.add_argument(
        "--use-all-for-train",
        action="store_true",
        default=True,
        help="Использовать все размеченные кадры и для train, и для val",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для воспроизводимости",
    )
    return parser.parse_args()


def iter_images(image_dir: Path) -> Iterable[Path]:
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in sorted(image_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in valid_ext:
            yield p


def label_path_for(image_path: Path, labels_root: Path) -> Path:
    # extracted_frames/4/frame_001.jpg -> extracted_frames/labels/4/frame_001.txt
    relative = image_path.parent.name
    return (labels_root / relative / image_path.stem).with_suffix(".txt")


def collect_labeled_images(images_dirs: List[Path], labels_root: Path) -> List[Path]:
    labeled: List[Path] = []
    for images_dir in images_dirs:
        if not images_dir.exists():
            print(f"[WARN] Не найдена папка изображений: {images_dir}")
            continue

        for image_path in iter_images(images_dir):
            if label_path_for(image_path, labels_root).exists():
                labeled.append(image_path.resolve())

    return labeled


def write_list_file(path: Path, image_paths: List[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(p) for p in image_paths) + "\n", encoding="utf-8")


def write_dataset_yaml(path: Path, train_txt: Path, val_txt: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = (
        f"path: {Path('.').resolve()}\n"
        f"train: {train_txt.resolve()}\n"
        f"val: {val_txt.resolve()}\n"
        "names:\n"
        "  0: bucket\n"
        "nc: 1\n"
    )
    path.write_text(yaml_text, encoding="utf-8")

from ultralytics import YOLO

def run_training(args: argparse.Namespace, data_yaml: Path, ultralytics=None) -> None:


    model = YOLO(args.model)
    model.train(
        data=str(data_yaml.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        pretrained=True,
        cache=False,
    )



def main() -> None:
    args = parse_args()

    labeled_images = collect_labeled_images(args.images_dirs, args.labels_root)
    if not labeled_images:
        raise SystemExit(
            "Не найдено размеченных кадров. Проверьте, что есть пары image + label.txt в extracted_frames/labels/<subdir>."
        )

    print(f"Найдено размеченных кадров: {len(labeled_images)}")
    args.workdir.mkdir(parents=True, exist_ok=True)

    train_txt = args.workdir / "train.txt"
    val_txt = args.workdir / "val.txt"
    data_yaml = args.workdir / "bucket_data.yaml"

    write_list_file(train_txt, labeled_images)

    if args.use_all_for_train:
        write_list_file(val_txt, labeled_images)
        print("[INFO] Используем все кадры и для train, и для val (как вы запросили).")
    else:
        # fallback: даже если отключат флаг, без split-логики сохраняем текущее поведение безопасно
        write_list_file(val_txt, labeled_images)
        print("[INFO] val set совпадает с train set.")

    write_dataset_yaml(data_yaml, train_txt, val_txt)
    print(f"[INFO] Dataset YAML: {data_yaml}")

    run_training(args, data_yaml)

    print("\nГотово. Веса обычно сохраняются сюда:")
    print(f"- {args.project}/{args.name}/weights/best.pt")
    print(f"- {args.project}/{args.name}/weights/last.pt")


if __name__ == "__main__":
    main()
