import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from modellab.core.datasets import DatasetAdapter, DatasetSample
from modellab.core.errors import DatasetError
from modellab.evaluation.preprocessing import PreprocessConfig, build_transform
from modellab.utils import get_logger

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
)

CANONICAL_CLASS_ORDER = ("gore", "blood", "neutral")

log = get_logger("dataset")


@dataclass(frozen=True)
class _Record:
    sample_id: str
    path: Path
    label: int


def _sorted_labels(labels: Iterable[str]) -> list[str]:
    labels = set(labels)

    # Project-specific class order.
    if labels == set(CANONICAL_CLASS_ORDER):
        return list(CANONICAL_CLASS_ORDER)

    # Numeric names must sort as numbers or "10" would land before "2".
    if labels and all(x.isascii() and x.isdigit() for x in labels):
        return sorted(labels, key=int)

    return sorted(labels)


def _resolve_class_names(
    found: list[str],
    given: Sequence[str] | None,
) -> list[str]:
    if given is None:
        return found

    names = list(given)
    if len(set(names)) != len(names):
        raise DatasetError("class_names contains duplicates")

    unknown = sorted(set(found) - set(names))
    if unknown:
        raise DatasetError(f"classes not present in class_names: {unknown}")

    return names


def _sorted_sample_folders(folders: Iterable[str]) -> list[str]:
    folders = list(folders)

    # Sample traversal remains deterministic and lexical for normal
    # textual class folders. Label IDs are controlled separately by
    # class_names / _sorted_labels.
    if folders and all(x.isascii() and x.isdigit() for x in folders):
        return sorted(folders, key=int)

    return sorted(folders)


class ImageClassificationDataset(DatasetAdapter):
    def __init__(
        self,
        records: Sequence[_Record],
        class_names: Sequence[str],
        dataset_id: str,
        preprocess: PreprocessConfig | None = None,
    ):
        if not records:
            raise DatasetError("dataset has no images")

        ids = [r.sample_id for r in records]
        if len(set(ids)) != len(ids):
            raise DatasetError("duplicate sample ids in dataset")

        self._records = list(records)
        self._class_names = list(class_names)
        self._dataset_id = dataset_id
        self._color_mode = preprocess.color_mode if preprocess is not None else "RGB"
        self._transform = (
            build_transform(preprocess) if preprocess is not None else None
        )

    @classmethod
    def from_folder(
        cls,
        root: str | Path,
        class_names: Sequence[str] | None = None,
        dataset_id: str | None = None,
        preprocess: PreprocessConfig | None = None,
    ) -> "ImageClassificationDataset":
        root = Path(root)

        if not root.is_dir():
            raise DatasetError(f"dataset directory not found: {root}")

        folders = [
            p.name
            for p in root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ]

        names = _resolve_class_names(_sorted_labels(folders), class_names)
        index = {name: i for i, name in enumerate(names)}

        records = []
        skipped = 0

        for name in _sorted_sample_folders(folders):
            for path in sorted((root / name).rglob("*")):
                rel = path.relative_to(root)

                if not path.is_file() or any(
                    part.startswith(".") for part in rel.parts
                ):
                    continue

                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    skipped += 1
                    continue

                records.append(
                    _Record(
                        rel.as_posix(),
                        path,
                        index[name],
                    )
                )

        if skipped:
            log.warning(
                "skipped %d non-image files under %s",
                skipped,
                root,
            )

        if not records:
            raise DatasetError(f"no images found under {root}")

        return cls(
            records,
            names,
            dataset_id or root.name,
            preprocess,
        )

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        root: str | Path | None = None,
        path_column: str = "path",
        label_column: str = "label",
        class_names: Sequence[str] | None = None,
        dataset_id: str | None = None,
        preprocess: PreprocessConfig | None = None,
    ) -> "ImageClassificationDataset":
        csv_path = Path(csv_path)

        if not csv_path.is_file():
            raise DatasetError(f"csv file not found: {csv_path}")

        root = Path(root) if root is not None else csv_path.parent

        rows = []

        try:
            with csv_path.open(newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames or []

                for column in (path_column, label_column):
                    if column not in fields:
                        raise DatasetError(
                            f"column '{column}' not found in {csv_path}"
                        )

                for line, row in enumerate(reader, start=2):
                    rel = (row.get(path_column) or "").strip()
                    label = (row.get(label_column) or "").strip()

                    if not rel or not label:
                        raise DatasetError(
                            f"{csv_path}: line {line} has an empty path or label"
                        )

                    rows.append((rel, label))

        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            raise DatasetError(f"cannot read {csv_path}: {exc}") from exc

        missing = [
            rel for rel, _ in rows
            if not (root / rel).is_file()
        ]

        if missing:
            raise DatasetError(
                f"{len(missing)} files listed in {csv_path} do not exist, "
                f"first: {missing[:5]}"
            )

        names = _resolve_class_names(
            _sorted_labels(label for _, label in rows),
            class_names,
        )

        index = {name: i for i, name in enumerate(names)}

        records = [
            _Record(rel, root / rel, index[label])
            for rel, label in rows
        ]

        return cls(
            records,
            names,
            dataset_id or csv_path.stem,
            preprocess,
        )

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    def __len__(self) -> int:
        return len(self._records)

    def _record(self, index: int) -> _Record:
        if not 0 <= index < len(self._records):
            raise IndexError(index)
        return self._records[index]

    def _metadata(self, record: _Record) -> dict[str, Any]:
        return {
            "path": str(record.path),
            "class_name": self._class_names[record.label],
        }

    def _load(self, record: _Record) -> Image.Image:
        try:
            with Image.open(record.path) as img:
                img.load()
                return img.convert(self._color_mode)
        except Exception as exc:
            # Pillow raises OSError, SyntaxError, bomb errors and more
            # depending on the damage.
            raise DatasetError(
                f"cannot read image {record.path}: {exc}"
            ) from exc

    def get_sample(self, index: int) -> DatasetSample:
        record = self._record(index)
        image = self._load(record)

        data = (
            self._transform(image)
            if self._transform is not None
            else image
        )

        return DatasetSample(
            sample_id=record.sample_id,
            input=data,
            label=record.label,
            metadata=self._metadata(record),
        )

    def get_label(self, index: int) -> int:
        return self._record(index).label

    def get_metadata(self, index: int) -> dict[str, Any]:
        return self._metadata(self._record(index))

    def validate_images(self) -> list[tuple[str, str]]:
        bad = []

        for record in self._records:
            try:
                self._load(record)
            except DatasetError as exc:
                bad.append((record.sample_id, str(exc)))

        return bad
