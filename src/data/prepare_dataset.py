import argparse
import json
import logging
from pathlib import Path
from shutil import copy2
from typing import Dict, Iterable, List, Optional, Tuple

from src.utils.config import EXTERNAL_DIR, PROCESSED_DIR, RAW_DIR
from src.utils.image_utils import box_to_yolo, format_yolo_annotation, is_valid_image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_CLASS_MAP: Dict[str, int] = {
    "AL": 0,
    "HD": 1,
    "FB": 2,
    "Tn": 3,
}

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_image_files(root_dir: Path) -> List[Path]:
    return [path for path in root_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]


def normalize_category_name(raw_name: str) -> str:
    return raw_name.strip().replace(" ", "_").replace("-", "_").upper()


def copy_images_to_raw(image_paths: Iterable[Path], raw_dir: Path) -> int:
    raw_dir = ensure_dir(raw_dir)
    copied = 0
    for image_path in image_paths:
        if not is_valid_image(image_path):
            logging.warning("Skipping invalid image: %s", image_path)
            continue

        destination = raw_dir / image_path.name
        if destination.exists():
            destination = raw_dir / f"{image_path.stem}_{image_path.suffix.lstrip('.')}"

        copy2(image_path, destination)
        copied += 1
        logging.info("Copied raw image: %s", destination)

    logging.info("Collected %d raw images into %s", copied, raw_dir)
    return copied


def create_full_image_yolo_label(image_path: Path, label_dir: Path, class_id: int) -> None:
    image_size = (image_path.stat().st_size, image_path.stat().st_size)
    label_path = label_dir / f"{image_path.stem}.txt"
    label_path.write_text(format_yolo_annotation(class_id, (0.5, 0.5, 1.0, 1.0)))


def combine_class_folder_dataset(
    source_dir: Path,
    raw_dir: Path = RAW_DIR,
    label_dir: Path = PROCESSED_DIR / "labels",
    class_map: Optional[Dict[str, int]] = None,
    full_image_bbox: bool = True,
) -> int:
    class_map = class_map or DEFAULT_CLASS_MAP.copy()
    raw_dir = ensure_dir(raw_dir)
    label_dir = ensure_dir(label_dir)
    copied = 0

    for class_folder in source_dir.iterdir():
        if not class_folder.is_dir():
            continue

        class_name = normalize_category_name(class_folder.name)
        class_id = class_map.get(class_name, len(class_map))
        if class_name not in class_map:
            class_map[class_name] = class_id
            logging.info("Added class mapping: %s -> %d", class_name, class_id)

        for image_path in find_image_files(class_folder):
            if not is_valid_image(image_path):
                continue

            destination = raw_dir / f"{class_name}_{image_path.name}"
            copy2(image_path, destination)
            copied += 1

            if full_image_bbox:
                label_path = label_dir / f"{destination.stem}.txt"
                label_path.write_text(format_yolo_annotation(class_id, (0.5, 0.5, 1.0, 1.0)))
            logging.info("Merged class image: %s (class=%s)", destination.name, class_name)

    logging.info("Combined %d class-labeled images from %s", copied, source_dir)
    return copied


def find_coco_annotation_file(root_dir: Path) -> Optional[Path]:
    for candidate in root_dir.rglob("*.json"):
        name = candidate.name.lower()
        if "instances" in name or "annotation" in name or "coco" in name:
            return candidate
    return None


def convert_coco_to_yolo(
    coco_json_path: Path,
    images_dir: Path,
    labels_dir: Path = PROCESSED_DIR / "labels",
    class_map: Optional[Dict[str, int]] = None,
) -> Tuple[int, int]:
    labels_dir = ensure_dir(labels_dir)
    with coco_json_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
    class_map = class_map or {}
    next_id = max(class_map.values(), default=-1) + 1

    for category_name in sorted(set(categories.values())):
        if category_name not in class_map:
            class_map[category_name] = next_id
            next_id += 1
            logging.info("Mapped category to class id: %s -> %d", category_name, class_map[category_name])

    image_map = {image["id"]: image for image in coco.get("images", [])}
    annotations_by_image: Dict[int, List[Dict]] = {}
    for annotation in coco.get("annotations", []):
        image_id = annotation["image_id"]
        annotations_by_image.setdefault(image_id, []).append(annotation)

    written_count = 0
    image_count = 0

    for image_id, image_info in image_map.items():
        image_file = images_dir / image_info.get("file_name", "")
        if not image_file.exists() or not is_valid_image(image_file):
            logging.warning("Skipping missing or invalid image: %s", image_file)
            continue

        image_count += 1
        annotations = annotations_by_image.get(image_id, [])
        if not annotations:
            continue

        size = (image_info.get("width", 0), image_info.get("height", 0))
        lines: List[str] = []
        for annotation in annotations:
            bbox = annotation.get("bbox", [])
            if len(bbox) != 4:
                continue
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height
            yolo_box = box_to_yolo(size, (x_min, y_min, x_max, y_max))
            class_id = class_map[categories[annotation["category_id"]]]
            lines.append(format_yolo_annotation(class_id, yolo_box).strip())

        if lines:
            label_path = labels_dir / f"{Path(image_info['file_name']).stem}.txt"
            label_path.write_text("\n".join(lines) + "\n")
            written_count += 1
            logging.info("Wrote YOLO label: %s", label_path.name)

    logging.info("Converted %d images and wrote %d YOLO label files", image_count, written_count)
    return image_count, written_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare stool parasite datasets and convert annotations to YOLO format.")
    parser.add_argument("--mode", choices=["combine", "convert_coco"], required=True, help="Operation mode")
    parser.add_argument("--source", type=Path, help="Source dataset directory for combine or root directory for COCO annotation search")
    parser.add_argument("--images-dir", type=Path, help="Directory containing COCO images")
    parser.add_argument("--coco-json", type=Path, help="COCO annotation JSON file")
    parser.add_argument("--raw-out", type=Path, default=RAW_DIR, help="Destination for merged raw images")
    parser.add_argument("--labels-out", type=Path, default=PROCESSED_DIR / "labels", help="Destination for YOLO label files")
    parser.add_argument("--full-image-bbox", action="store_true", help="Create full-image YOLO boxes for class-folder datasets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "combine":
        if not args.source:
            raise ValueError("--source is required for combine mode")
        combine_class_folder_dataset(
            source_dir=args.source,
            raw_dir=args.raw_out,
            label_dir=args.labels_out,
            full_image_bbox=args.full_image_bbox,
        )
    elif args.mode == "convert_coco":
        if args.coco_json is None or args.images_dir is None:
            raise ValueError("--coco-json and --images-dir are required for convert_coco mode")
        convert_coco_to_yolo(
            coco_json_path=args.coco_json,
            images_dir=args.images_dir,
            labels_dir=args.labels_out,
        )


if __name__ == "__main__":
    main()
