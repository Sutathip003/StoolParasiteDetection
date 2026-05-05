import argparse
import logging
import random
from pathlib import Path
from shutil import copy2
from typing import List, Tuple

from ..utils.config import DEFAULT_SPLIT_CONFIG, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def build_split_paths(base_dir: Path, split_name: str) -> Tuple[Path, Path]:
    images_dir = base_dir / split_name / "images"
    labels_dir = base_dir / split_name / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def split_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> None:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    split_config = (train_ratio, val_ratio, test_ratio)
    if abs(sum(split_config) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    files = [path for path in images_dir.glob("*") if path.is_file()]
    random.Random(random_seed).shuffle(files)

    total = len(files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    splits = {
        "train": files[:train_count],
        "val": files[train_count : train_count + val_count],
        "test": files[train_count + val_count :],
    }

    for split_name, image_paths in splits.items():
        split_images_dir, split_labels_dir = build_split_paths(output_dir, split_name)
        logging.info("Creating %s split with %d image(s)", split_name, len(image_paths))

        for image_path in image_paths:
            copy2(image_path, split_images_dir / image_path.name)
            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                copy2(label_path, split_labels_dir / label_path.name)
            else:
                logging.warning("Missing label for image: %s", image_path.name)

    logging.info("Split dataset into train/val/test under %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split processed dataset into train/val/test sets.")
    parser.add_argument("--images-dir", default=str(PROCESSED_DIR / "images"), help="Directory containing processed images")
    parser.add_argument("--labels-dir", default=str(PROCESSED_DIR / "labels"), help="Directory containing processed labels")
    parser.add_argument("--output-dir", default=str(PROCESSED_DIR), help="Directory to store the split dataset")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT_CONFIG.train_ratio, help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_SPLIT_CONFIG.val_ratio, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_SPLIT_CONFIG.test_ratio, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_CONFIG.random_seed, help="Random seed for reproducible splits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dataset(
        images_dir=Path(args.images_dir),
        labels_dir=Path(args.labels_dir),
        output_dir=Path(args.output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
