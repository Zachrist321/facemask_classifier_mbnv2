"""Face-mask crop dataset utilities for MAKS (Kaggle) Pascal VOC annotations."""

from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

CLASS_NAMES = ("with_mask", "mask_weared_incorrect", "without_mask")
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class FaceSample:
    image_path: Path
    label: str
    bbox: tuple[int, int, int, int]  # xmin, ymin, xmax, ymax


def parse_annotations(data_dir: str | Path) -> list[FaceSample]:
    """Parse VOC XML files and return one sample per annotated face."""
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    ann_dir = data_dir / "annotations"
    samples: list[FaceSample] = []

    for xml_path in sorted(ann_dir.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename")
        if not filename:
            continue
        image_path = images_dir / filename
        if not image_path.exists():
            continue

        for obj in root.findall("object"):
            label = obj.findtext("name")
            if label not in CLASS_TO_INDEX:
                continue
            box = obj.find("bndbox")
            if box is None:
                continue
            xmin = int(float(box.findtext("xmin", "0")))
            ymin = int(float(box.findtext("ymin", "0")))
            xmax = int(float(box.findtext("xmax", "0")))
            ymax = int(float(box.findtext("ymax", "0")))
            if xmax <= xmin or ymax <= ymin:
                continue
            samples.append(
                FaceSample(
                    image_path=image_path,
                    label=label,
                    bbox=(xmin, ymin, xmax, ymax),
                )
            )
    return samples


def class_distribution(samples: Iterable[FaceSample]) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for sample in samples:
        counts[sample.label] += 1
    return counts


def stratified_split(
    samples: list[FaceSample],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[FaceSample], list[FaceSample]]:
    """Split per class so each label keeps roughly val_ratio in validation."""
    rng = random.Random(seed)
    by_class: dict[str, list[FaceSample]] = {name: [] for name in CLASS_NAMES}
    for sample in samples:
        by_class[sample.label].append(sample)

    train: list[FaceSample] = []
    val: list[FaceSample] = []
    for class_samples in by_class.values():
        rng.shuffle(class_samples)
        n_val = max(1, int(len(class_samples) * val_ratio))
        val.extend(class_samples[:n_val])
        train.extend(class_samples[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def crop_face(
    sample: FaceSample,
    img_size: tuple[int, int] = (224, 224),
    margin: float = 0.15,
) -> np.ndarray:
    """Crop a face region with optional margin and resize to img_size."""
    image = Image.open(sample.image_path).convert("RGB")
    xmin, ymin, xmax, ymax = sample.bbox
    width, height = image.size

    box_w = xmax - xmin
    box_h = ymax - ymin
    pad_x = int(box_w * margin)
    pad_y = int(box_h * margin)
    xmin = max(0, xmin - pad_x)
    ymin = max(0, ymin - pad_y)
    xmax = min(width, xmax + pad_x)
    ymax = min(height, ymax + pad_y)

    crop = image.crop((xmin, ymin, xmax, ymax))
    crop = crop.resize(img_size, Image.BILINEAR)
    return np.asarray(crop, dtype=np.uint8)


def export_split_metadata(
    train: list[FaceSample],
    val: list[FaceSample],
    output_path: str | Path,
) -> None:
    """Save split metadata as JSON for reproducibility."""

    def serialize(samples: list[FaceSample]) -> list[dict]:
        return [
            {
                "image_path": str(s.image_path),
                "label": s.label,
                "bbox": list(s.bbox),
            }
            for s in samples
        ]

    payload = {
        "class_names": list(CLASS_NAMES),
        "train": serialize(train),
        "val": serialize(val),
        "train_distribution": class_distribution(train),
        "val_distribution": class_distribution(val),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
