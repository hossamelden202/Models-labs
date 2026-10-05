from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from modellab.core.datasets import DatasetAdapter, DatasetSample
from modellab.core.errors import DatasetError
from modellab.evaluation.preprocessing import PreprocessConfig


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _find_yolo_dirs(root: Path) -> tuple[Path, Path]:
    candidates = [
        (root / "images", root / "labels"),
        (root / "train" / "images", root / "train" / "labels"),
        (root / "val" / "images", root / "val" / "labels"),
        (root / "valid" / "images", root / "valid" / "labels"),
        (root / "validation" / "images", root / "validation" / "labels"),
        (root / "test" / "images", root / "test" / "labels"),
    ]

    for images, labels in candidates:
        if images.is_dir() and labels.is_dir():
            return images, labels

    raise DatasetError(
        f"could not find a YOLO detection layout under {root}; "
        "expected images/ and labels/ directories"
    )


def _image_files(root: Path) -> list[Path]:
    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    ]
    files.sort()
    if not files:
        raise DatasetError(f"no images found in YOLO image directory: {root}")
    return files


def _read_labels(path: Path) -> list[list[float]]:
    if not path.exists():
        return []

    rows: list[list[float]] = []

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(f"could not read label file {path}: {exc}") from exc

    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            raise DatasetError(
                f"invalid YOLO annotation {path}:{line_number}; "
                "expected 5 values: class x_center y_center width height"
            )

        try:
            values = [float(x) for x in parts]
        except ValueError as exc:
            raise DatasetError(
                f"invalid numeric value in YOLO annotation {path}:{line_number}"
            ) from exc

        cls = values[0]
        if cls < 0 or cls != int(cls):
            raise DatasetError(
                f"invalid class id in YOLO annotation {path}:{line_number}"
            )

        x, y, w, h = values[1:]
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
            raise DatasetError(
                f"YOLO coordinates must be normalized to [0,1] in "
                f"{path}:{line_number}"
            )

        if w <= 0 or h <= 0:
            raise DatasetError(
                f"YOLO width/height must be positive in {path}:{line_number}"
            )

        rows.append(values)

    return rows


def _make_transform(preprocess: PreprocessConfig | None):
    if preprocess is None:
        return None

    try:
        from modellab.evaluation.preprocessing import build_transform

        return build_transform(preprocess)
    except ImportError:
        return None


class YOLODetectionDataset(DatasetAdapter):
    """
    YOLO-format object-detection dataset.

    Expected layout:

        root/
          images/
          labels/

    or:

        root/
          train/images/
          train/labels/

    Each label file contains:

        class_id x_center y_center width height

    Coordinates are normalized to [0, 1].
    """

    def __init__(
        self,
        root: str | Path,
        class_names: list[str] | None,
        dataset_id: str,
        preprocess: PreprocessConfig | None = None,
    ):
        self.root = Path(root).resolve()
        self._dataset_id = dataset_id
        self.class_names = list(class_names or [])
        self.preprocess = preprocess

        self.images_root, self.labels_root = _find_yolo_dirs(self.root)
        self._images = _image_files(self.images_root)
        self._transform = _make_transform(preprocess)

        self._labels: list[list[list[float]]] = []

        for image_path in self._images:
            relative = image_path.relative_to(self.images_root)
            label_path = self.labels_root / relative.with_suffix(".txt")
            self._labels.append(_read_labels(label_path))

        if self.class_names:
            k = len(self.class_names)
            for image_path, rows in zip(self._images, self._labels, strict=True):
                for row in rows:
                    if int(row[0]) >= k:
                        raise DatasetError(
                            f"class id {int(row[0])} in {image_path} is outside "
                            f"the model class range 0..{k - 1}"
                        )

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    def __len__(self) -> int:
        return len(self._images)

    def _load_image(self, index: int) -> torch.Tensor:
        path = self._images[index]

        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise DatasetError(f"could not open image {path}: {exc}") from exc

        if self._transform is not None:
            value = self._transform(image)
        else:
            import numpy as np

            array = np.asarray(image, dtype=np.float32) / 255.0
            value = torch.from_numpy(array).permute(2, 0, 1)

        if not isinstance(value, torch.Tensor):
            raise DatasetError(
                f"preprocessing for {path} did not produce a torch.Tensor"
            )

        return value

    def get_sample(self, index: int) -> DatasetSample:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        image_path = self._images[index]
        labels = self._labels[index]

        return DatasetSample(
            sample_id=image_path.relative_to(self.images_root).as_posix(),
            input=self._load_image(index),
            label=None,
            metadata={
                "path": str(image_path),
                "boxes": [
                    {
                        "class_id": int(row[0]),
                        "x_center": row[1],
                        "y_center": row[2],
                        "width": row[3],
                        "height": row[4],
                    }
                    for row in labels
                ],
            },
        )

    def get_label(self, index: int) -> int | None:
        return None

    def targets(self, index: int) -> list[dict[str, Any]]:
        return self.get_sample(index).metadata["boxes"]
