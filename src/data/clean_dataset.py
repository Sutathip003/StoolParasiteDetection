import argparse
import logging
from pathlib import Path
from shutil import copy2
from typing import Iterable

from ..utils.config import LABELED_DIR, PROCESSED_DIR, RAW_DIR
from ..utils.image_utils import is_valid_image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PROCESSED_IMAGES_DIR = PROCESSED_DIR / "images"
PROCESSED_LABELS_DIR = PROCESSED_DIR / "labels"


def scan_raw_images(raw_dir: Path = RAW_DIR) -> Iterable[Path]:
    if not raw_dir.exists():
        logging.warning("Raw data directory does not exist: %s", raw_dir)
        return []
    return [path for path in raw_dir.rglob("*") if path.is_file()]


def clean_raw_images(raw_dir: Path = RAW_DIR, output_dir: Path = PROCESSED_IMAGES_DIR) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned = 0

    for image_path in scan_raw_images(raw_dir):
        if is_valid_image(image_path):
            target = output_dir / image_path.name
            copy2(image_path, target)
            cleaned += 1
            logging.info("Validated image: %s", image_path)
        else:
            logging.warning("Removed invalid image: %s", image_path)

    logging.info("Cleaned %d images into %s", cleaned, output_dir)
    return cleaned


def sync_labels(
    label_dir: Path = LABELED_DIR,
    image_dir: Path = PROCESSED_IMAGES_DIR,
    output_dir: Path = PROCESSED_LABELS_DIR,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    for image_path in image_dir.glob("*"):
        expected_label = label_dir / f"{image_path.stem}.txt"
        if expected_label.exists():
            copy2(expected_label, output_dir / expected_label.name)
            copied += 1
            logging.info("Copied label: %s", expected_label)
        else:
            logging.warning("Label not found for image: %s", image_path.name)

    logging.info("Copied %d label(s) into %s", copied, output_dir)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw dataset images and sync YOLO labels.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR), help="Directory containing raw images")
    parser.add_argument("--labeled-dir", default=str(LABELED_DIR), help="Directory containing existing labels")
    parser.add_argument("--output-dir", default=str(PROCESSED_DIR), help="Directory for processed output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cleaned = clean_raw_images(Path(args.raw_dir), PROCESSED_IMAGES_DIR)
    if cleaned > 0:
        sync_labels(Path(args.labeled_dir), PROCESSED_IMAGES_DIR, PROCESSED_LABELS_DIR)


if __name__ == "__main__":
    main()
