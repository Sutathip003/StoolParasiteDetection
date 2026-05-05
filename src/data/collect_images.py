import argparse
import logging
from shutil import copy2
from typing import Iterable

from pathlib import Path

from ..utils.config import RAW_DIR
from ..utils.image_utils import is_valid_image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def collect_images(sources: Iterable[Path], destination: Path = RAW_DIR) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0

    for source in sources:
        if source.is_file():
            if is_valid_image(source):
                target = destination / source.name
                copy2(source, target)
                copied += 1
                logging.info("Copied image: %s", source)
            else:
                logging.warning("Skipped invalid image: %s", source)
            continue

        if not source.exists() or not source.is_dir():
            logging.warning("Source path does not exist or is not a directory: %s", source)
            continue

        for image_path in source.rglob("*"):
            if image_path.is_file() and is_valid_image(image_path):
                target = destination / image_path.name
                copy2(image_path, target)
                copied += 1
                logging.info("Copied image: %s", image_path)

    logging.info("Collected %d valid image(s) into %s", copied, destination)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect raw parasite images into the data/raw directory.")
    parser.add_argument("sources", nargs="+", help="Directories or image files to collect")
    parser.add_argument("--destination", default=str(RAW_DIR), help="Path to the raw data destination directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [Path(source) for source in args.sources]
    destination = Path(args.destination)
    collect_images(sources, destination)


if __name__ == "__main__":
    main()
