import csv
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modellab.audit.config import AuditConfig
from modellab.core.errors import DatasetError
from modellab.evaluation.image_dataset import IMAGE_EXTENSIONS

SPLIT_NAMES = frozenset(
    {"train", "training", "val", "valid", "validation", "dev", "test", "testing"}
)
INVALID_PATH = "<invalid>"
_DRIVE = re.compile(r"^[A-Za-z]:")


@dataclass
class Record:
    sample_id: str
    path: str
    split: str | None
    label: str | None
    label_name: str | None = None
    row: int = 0
    status: str = "pending"
    extension: str = ""
    file_size: int | None = None
    sha256: str | None = None
    format: str | None = None
    mode: str | None = None
    width: int | None = None
    height: int | None = None
    phash: int | None = None
    brightness: float | None = None
    contrast: float | None = None
    saturation: float | None = None
    error_stage: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class ScanResult:
    kind: str
    root: Path
    dataset_id: str
    layout: str
    records: list[Record]
    exclusions: list[dict[str, str]]
    declared_classes: list[str]
    declared_splits: list[str] | None


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def canonical_path(raw: str) -> str | None:
    text = _nfc(raw.strip()).replace("\\", "/")
    if not text or text.startswith("/") or _DRIVE.match(text):
        return None
    parts = [p for p in text.split("/") if p not in ("", ".")]
    if not parts or ".." in parts:
        return None
    return "/".join(parts)


def _split_like(names: list[str]) -> bool:
    return bool(names) and all(n.casefold() in SPLIT_NAMES for n in names)


def scan_folder(root, cfg: AuditConfig, dataset_id: str | None = None) -> ScanResult:
    root = Path(root)
    if not root.is_dir():
        raise DatasetError(f"dataset directory not found: {root}")

    top = [p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    layout = cfg.layout
    if layout == "auto":
        layout = "split_class" if _split_like(top) else "class_only"
    class_depth = 2 if layout == "split_class" else 1

    records: list[Record] = []
    exclusions: list[dict[str, str]] = []
    classes: set[str] = set()
    splits: set[str] = set()

    for current, dirs, files in os.walk(root):
        rel_dir = Path(current).relative_to(root)
        kept = []
        for name in sorted(dirs):
            if name.startswith("."):
                path = (rel_dir / name).as_posix()
                exclusions.append({"path": path, "reason": "hidden_directory"})
            else:
                kept.append(name)
        dirs[:] = kept

        depth = len(rel_dir.parts)
        if layout == "split_class" and depth == 1:
            splits.add(rel_dir.parts[0])
        if depth == class_depth:
            classes.add(rel_dir.parts[-1])

        for name in sorted(files):
            rel = rel_dir / name
            rel_str = rel.as_posix()
            if name.startswith("."):
                exclusions.append({"path": rel_str, "reason": "hidden_file"})
                continue
            if len(rel.parts) <= class_depth:
                exclusions.append({"path": rel_str, "reason": "outside_class_directory"})
                continue
            if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
                exclusions.append({"path": rel_str, "reason": "unsupported_extension"})
                continue
            records.append(
                Record(
                    sample_id=_nfc(rel_str),
                    path=rel_str,
                    split=rel.parts[0] if layout == "split_class" else None,
                    label=rel.parts[class_depth - 1],
                    row=len(records),
                )
            )

    return ScanResult(
        kind="folder",
        root=root,
        dataset_id=dataset_id or root.resolve().name,
        layout=layout,
        records=records,
        exclusions=exclusions,
        declared_classes=sorted(classes),
        declared_splits=sorted(splits) if layout == "split_class" else None,
    )


def scan_csv(
    csv_path, cfg: AuditConfig, root=None, dataset_id: str | None = None
) -> ScanResult:
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise DatasetError(f"csv file not found: {csv_path}")
    root = Path(root) if root is not None else csv_path.parent
    if not root.is_dir():
        raise DatasetError(f"image root not found: {root}")

    records: list[Record] = []
    exclusions: list[dict[str, str]] = []
    try:
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            for column in (cfg.path_column, cfg.label_column):
                if column not in fields:
                    raise DatasetError(f"column '{column}' not found in {csv_path.name}")
            for column in (cfg.label_name_column, cfg.id_column):
                if column is not None and column not in fields:
                    raise DatasetError(f"column '{column}' not found in {csv_path.name}")
            has_split = cfg.split_column is not None and cfg.split_column in fields

            for line, row in enumerate(reader, start=2):
                raw_path = (row.get(cfg.path_column) or "").strip()
                if not raw_path:
                    exclusions.append({"path": f"row {line}", "reason": "empty_path"})
                    continue
                canon = canonical_path(raw_path)
                explicit_id = _clean(row.get(cfg.id_column)) if cfg.id_column else None
                if canon is None:
                    path = INVALID_PATH
                    sample_id = explicit_id or f"row{line}"
                else:
                    path = canon
                    sample_id = explicit_id or canon
                records.append(
                    Record(
                        sample_id=sample_id,
                        path=path,
                        split=_clean(row.get(cfg.split_column)) if has_split else None,
                        label=_clean(row.get(cfg.label_column)),
                        label_name=(
                            _clean(row.get(cfg.label_name_column))
                            if cfg.label_name_column
                            else None
                        ),
                        row=line,
                    )
                )
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise DatasetError(f"cannot read {csv_path.name}: {exc}") from exc

    return ScanResult(
        kind="csv",
        root=root,
        dataset_id=dataset_id or csv_path.stem,
        layout="csv",
        records=records,
        exclusions=exclusions,
        declared_classes=list(cfg.class_names or []),
        declared_splits=None,
    )


def scan_adapter(dataset, root, cfg: AuditConfig, dataset_id: str | None = None) -> ScanResult:
    root = Path(root)
    names = getattr(dataset, "class_names", None)
    root_abs = os.path.abspath(root)
    records: list[Record] = []
    for index in range(len(dataset)):
        meta = dataset.get_metadata(index)
        value = dataset.get_label(index)
        if value is None:
            label = None
        elif names is not None and isinstance(value, int) and 0 <= value < len(names):
            label = str(names[value])
        else:
            label = str(value)

        canon = None
        raw = meta.get("path")
        if raw:
            try:
                rel = Path(os.path.abspath(raw)).relative_to(root_abs)
                canon = canonical_path(rel.as_posix())
            except ValueError:
                canon = None
        records.append(
            Record(
                sample_id=str(meta.get("sample_id") or canon or f"index{index}"),
                path=canon or INVALID_PATH,
                split=_clean(meta.get("split")),
                label=label,
                row=index,
            )
        )

    return ScanResult(
        kind="adapter",
        root=root,
        dataset_id=dataset_id or getattr(dataset, "dataset_id", root.resolve().name),
        layout="adapter",
        records=records,
        exclusions=[],
        declared_classes=[str(n) for n in names] if names else [],
        declared_splits=None,
    )
