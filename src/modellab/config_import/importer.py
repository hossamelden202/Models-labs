from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


class ConfigImportError(ValueError):
    """Raised when a model configuration cannot be imported or normalized."""


@dataclass
class NormalizedModelConfig:
    """
    Framework-independent ModelLab model metadata.

    Fields are optional because framework configuration files do not
    necessarily contain every piece of information ModelLab can use.
    """

    num_classes: int | None = None
    class_names: list[str] | None = None
    input_width: int | None = None
    input_height: int | None = None

    def to_spec_fields(self) -> dict[str, Any]:
        """Convert available values into the existing ModelLab spec shape."""
        result: dict[str, Any] = {}

        if self.num_classes is not None:
            result["num_classes"] = self.num_classes

        if self.class_names is not None:
            result["class_names"] = self.class_names

        if self.input_width is not None and self.input_height is not None:
            result["input_size"] = [
                self.input_width,
                self.input_height,
            ]

        return result


def _as_int(value: Any, field: str) -> int | None:
    if value is None:
        return None

    if isinstance(value, bool):
        raise ConfigImportError(f"{field} must be an integer")

    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigImportError(
            f"{field} must be an integer, got {value!r}"
        ) from exc

    if value <= 0:
        raise ConfigImportError(f"{field} must be greater than zero")

    return value


def _normalize_names(value: Any) -> list[str] | None:
    if value is None:
        return None

    if isinstance(value, dict):
        # Hugging Face style:
        # {"0": "cat", "1": "dog"}
        try:
            items = sorted(value.items(), key=lambda item: int(item[0]))
        except (TypeError, ValueError):
            items = list(value.items())

        value = [item[1] for item in items]

    if isinstance(value, (tuple, list)):
        names = [str(item).strip() for item in value]
    else:
        raise ConfigImportError(
            "class names must be a list or mapping"
        )

    if not names or any(not name for name in names):
        raise ConfigImportError("class names cannot contain empty names")

    return names


def _extract_size(data: dict[str, Any]) -> tuple[int | None, int | None]:
    """
    Extract image/model input size from common configuration conventions.
    """

    # Explicit ModelLab format.
    width = data.get("input_width")
    height = data.get("input_height")

    if width is not None or height is not None:
        return _as_int(width, "input_width"), _as_int(height, "input_height")

    # Common image_size / input_size conventions.
    for key in ("input_size", "image_size", "img_size", "imgsz"):
        value = data.get(key)

        if value is None:
            continue

        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                size = _as_int(value[0], key)
                return size, size

            if len(value) >= 2:
                return (
                    _as_int(value[0], f"{key}[0]"),
                    _as_int(value[1], f"{key}[1]"),
                )

        if isinstance(value, (int, float, str)):
            size = _as_int(value, key)
            return size, size

    return None, None


def _normalize_generic(data: dict[str, Any]) -> NormalizedModelConfig:
    num_classes = data.get("num_classes")

    if num_classes is None:
        num_classes = data.get("num_labels")

    class_names = data.get("class_names")

    if class_names is None:
        class_names = data.get("names")

    if class_names is None:
        class_names = data.get("id2label")

    width, height = _extract_size(data)

    return NormalizedModelConfig(
        num_classes=_as_int(num_classes, "num_classes")
        if num_classes is not None
        else None,
        class_names=_normalize_names(class_names),
        input_width=width,
        input_height=height,
    )


def _normalize_ultralytics(data: dict[str, Any]) -> NormalizedModelConfig:
    names = data.get("names")

    num_classes = data.get("nc")

    if num_classes is None and names is not None:
        if isinstance(names, (list, tuple, dict)):
            num_classes = len(names)

    width, height = _extract_size(data)

    return NormalizedModelConfig(
        num_classes=_as_int(num_classes, "nc")
        if num_classes is not None
        else None,
        class_names=_normalize_names(names),
        input_width=width,
        input_height=height,
    )


def _normalize_huggingface(data: dict[str, Any]) -> NormalizedModelConfig:
    num_classes = data.get("num_labels")

    if num_classes is None:
        num_classes = data.get("num_classes")

    labels = data.get("id2label")

    if labels is None:
        labels = data.get("label2id")

        if isinstance(labels, dict):
            labels = {
                str(index): label
                for label, index in labels.items()
            }

    width, height = _extract_size(data)

    return NormalizedModelConfig(
        num_classes=_as_int(num_classes, "num_labels")
        if num_classes is not None
        else None,
        class_names=_normalize_names(labels),
        input_width=width,
        input_height=height,
    )


def _detect_format(data: dict[str, Any]) -> str:
    """
    Detect known config families.

    Detection is intentionally conservative. Unknown files fall back
    to the generic normalizer so future schemas can still work when
    they expose familiar field names.
    """

    if "nc" in data and "names" in data:
        return "ultralytics"

    if (
        "id2label" in data
        or "label2id" in data
        or "num_labels" in data
    ):
        return "huggingface"

    return "generic"


def normalize_config(data: dict[str, Any]) -> NormalizedModelConfig:
    if not isinstance(data, dict):
        raise ConfigImportError("configuration root must be an object/mapping")

    format_name = _detect_format(data)

    if format_name == "ultralytics":
        result = _normalize_ultralytics(data)
    elif format_name == "huggingface":
        result = _normalize_huggingface(data)
    else:
        result = _normalize_generic(data)

    if (
        result.num_classes is not None
        and result.class_names is not None
        and result.num_classes != len(result.class_names)
    ):
        raise ConfigImportError(
            f"num_classes={result.num_classes} does not match "
            f"{len(result.class_names)} class names"
        )

    return result


def import_model_config(path: str | Path) -> NormalizedModelConfig:
    """
    Load JSON/YAML configuration and normalize it.

    Supported:
      - generic ModelLab JSON/YAML
      - Ultralytics data.yaml
      - Hugging Face config.json
      - future compatible configs using recognized fields
    """

    path = Path(path)

    if not path.is_file():
        raise ConfigImportError(f"config file not found: {path}")

    suffix = path.suffix.lower()

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigImportError(
            f"cannot read config file: {path}"
        ) from exc

    try:
        if suffix == ".json":
            data = json.loads(raw)

        elif suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise ConfigImportError(
                    "PyYAML is required to import YAML configuration files"
                )
            data = yaml.safe_load(raw)

        else:
            raise ConfigImportError(
                f"unsupported config extension: {suffix!r}; "
                "expected .json, .yaml, or .yml"
            )

    except ConfigImportError:
        raise

    except Exception as exc:
        raise ConfigImportError(
            f"invalid {suffix.lstrip('.').upper()} configuration: {exc}"
        ) from exc

    return normalize_config(data)
