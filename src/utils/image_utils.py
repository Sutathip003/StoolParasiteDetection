import logging
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

ImageSize = Tuple[int, int]
BBox = Tuple[float, float, float, float]

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_valid_image(path: Path) -> bool:
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        logging.debug("Skipping unsupported file type: %s", path)
        return False

    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception as exc:
        logging.warning("Invalid image detected: %s (%s)", path, exc)
        return False


def get_image_size(path: Path) -> Optional[ImageSize]:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception as exc:
        logging.warning("Cannot read image size: %s (%s)", path, exc)
        return None


def box_to_yolo(
    image_size: ImageSize,
    box: BBox,
) -> BBox:
    width, height = image_size
    x_min, y_min, x_max, y_max = box

    x_center = ((x_min + x_max) / 2.0) / width
    y_center = ((y_min + y_max) / 2.0) / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height

    return x_center, y_center, box_width, box_height


def format_yolo_annotation(class_id: int, box: BBox) -> str:
    return f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n"
