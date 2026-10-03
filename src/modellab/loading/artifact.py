import pickle
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, Field

MAX_ITEMS = 200
Format = Literal["torchscript", "torch_archive", "safetensors", "onnx", "pickle_object", "unknown", "missing"]


class StateDictSummary(BaseModel):
    key_path: str | None
    num_tensors: int
    num_parameters: int
    top_level_prefixes: list[str]
    sample_keys: list[str]
    common_prefix: str | None = None
    last_matrix: dict[str, Any] | None = None


class ArtifactInfo(BaseModel):
    file_name: str
    size_bytes: int = 0
    format: Format
    self_contained: bool | None = None
    state_dicts: list[StateDictSummary] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    hints: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _looks_safetensors(path: Path, head: bytes) -> bool:
    if len(head) < 8:
        return False
    header_len = int.from_bytes(head[:8], "little")
    if header_len <= 0 or header_len + 8 > path.stat().st_size:
        return False
    with open(path, "rb") as handle:
        handle.seek(8)
        return handle.read(1) == b"{"


def sniff_format(path: Path) -> str:
    if not path.is_file():
        return "missing"
    if path.suffix.lower() == ".onnx":
        return "onnx"
    with open(path, "rb") as handle:
        head = handle.read(16)
    if path.suffix.lower() == ".safetensors" or _looks_safetensors(path, head):
        return "safetensors"
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        if any(n.endswith("constants.pkl") or "/code/" in n for n in names):
            return "torchscript"
        if any(n.endswith("data.pkl") for n in names):
            return "torch_archive"
        return "unknown"
    return "torch_archive" if head[:1] == b"\x80" else "unknown"


def load_tensors_only(path: Path, fmt: str):
    if fmt == "safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except pickle.UnpicklingError:
        raise
    except RuntimeError:
        return torch.load(path, map_location="cpu", weights_only=True)


def _is_state_dict(value) -> bool:
    if not isinstance(value, Mapping) or not value or not all(isinstance(k, str) for k in value):
        return False
    tensors = sum(isinstance(v, torch.Tensor) for v in value.values())
    return tensors >= 1 and tensors >= 0.8 * len(value)


def find_state_dicts(obj) -> list[tuple[str | None, Mapping]]:
    if _is_state_dict(obj):
        return [(None, obj)]
    if isinstance(obj, Mapping):
        return [(k, v) for k, v in obj.items() if isinstance(k, str) and _is_state_dict(v)]
    return []


def summarize_state(key_path: str | None, state: Mapping) -> StateDictSummary:
    tensors = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    keys = list(tensors)
    prefixes: list[str] = []
    for key in keys:
        head = key.split(".")[0]
        if head not in prefixes:
            prefixes.append(head)
    last = None
    for key in reversed(keys):
        if key.endswith("weight") and tensors[key].ndim == 2:
            last = {"key": key, "shape": list(tensors[key].shape)}
            break
    return StateDictSummary(
        key_path=key_path, num_tensors=len(tensors),
        num_parameters=int(sum(t.numel() for t in tensors.values())),
        top_level_prefixes=prefixes[:12], sample_keys=keys[:8],
        common_prefix="module." if keys and all(k.startswith("module.") for k in keys) else None,
        last_matrix=last,
    )


def _json_safe(value, depth: int = 0):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item(), depth)
    if depth >= 3:
        return None
    if isinstance(value, (list, tuple)) and len(value) <= MAX_ITEMS:
        items = [_json_safe(v, depth + 1) for v in value]
        return items if all(i is not None or v is None for i, v in zip(items, value, strict=True)) else None
    if isinstance(value, Mapping) and len(value) <= MAX_ITEMS:
        out = {}
        for k, v in value.items():
            if isinstance(k, (str, int)):
                safe = _json_safe(v, depth + 1)
                if safe is not None or v is None:
                    out[str(k)] = safe
        return out
    return None


def _names_from(value):
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        return value
    if isinstance(value, dict) and value:
        try:
            items = sorted(((int(k), v) for k, v in value.items()), key=lambda kv: kv[0])
        except (TypeError, ValueError):
            return None
        if all(isinstance(v, str) for _, v in items) and [i for i, _ in items] == list(range(len(items))):
            return [v for _, v in items]
    return None


def _names_from_mapping(value):
    if isinstance(value, dict) and value and all(isinstance(v, int) and not isinstance(v, bool) for v in value.values()):
        ids = sorted(value.values())
        if ids == list(range(len(ids))):
            return [k for k, _ in sorted(value.items(), key=lambda kv: kv[1])]
    return None


def _size_from(value):
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return [value, value]
    if isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) and v > 0 for v in value):
        return list(value)
    return None


def _triple(value):
    if isinstance(value, list) and len(value) in (1, 3) and all(isinstance(v, (int, float)) for v in value):
        return [float(v) for v in value]
    return None


def extract_hints(meta: dict) -> list[dict]:
    lowered = {k.lower(): (k, v) for k, v in meta.items()}
    rules = [
        ("class_names", ("class_names", "classes", "class_labels", "labels", "idx_to_class", "id2label"), _names_from),
        ("class_names", ("class_to_id", "class_to_idx", "label2id"), _names_from_mapping),
        ("num_classes", ("num_classes", "n_classes", "nb_classes"),
         lambda v: v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else None),
        ("input_size", ("image_size", "input_size", "img_size", "imgsz"), _size_from),
        ("preprocess.mean", ("image_mean", "mean"), _triple),
        ("preprocess.std", ("image_std", "std"), _triple),
        ("architecture_hint", ("architecture", "arch", "model_type", "model_name"),
         lambda v: v if isinstance(v, str) and v else None),
    ]
    hints = []
    for field, aliases, convert in rules:
        for alias in aliases:
            if alias in lowered:
                original, raw = lowered[alias]
                value = convert(raw)
                if value is not None:
                    hints.append({"field": field, "value": value, "source": f"checkpoint key '{original}'"})
                    break
    return hints


def inspect_artifact(path) -> ArtifactInfo:
    path = Path(path)
    fmt = sniff_format(path)
    info = ArtifactInfo(file_name=path.name, size_bytes=path.stat().st_size if path.is_file() else 0, format=fmt)
    if fmt in ("missing", "onnx", "unknown"):
        return info
    if fmt == "torchscript":
        info.self_contained = True
        return info
    try:
        obj = load_tensors_only(path, fmt)
    except pickle.UnpicklingError:
        info.format = "pickle_object"
        info.warnings.append("the file contains Python objects; loading it executes code stored in the file")
        return info
    except ImportError as exc:
        info.warnings.append(f"cannot read this format: {exc}")
        info.format = "unknown"
        return info
    except Exception as exc:
        info.warnings.append(f"could not read the file safely: {type(exc).__name__}: {exc}")
        info.format = "unknown"
        return info
    info.self_contained = False
    found = find_state_dicts(obj)
    info.state_dicts = [summarize_state(key, state) for key, state in found]
    if isinstance(obj, Mapping) and fmt != "safetensors":
        skip = {k for k, _ in found}
        meta = {}
        for key, value in obj.items():
            if isinstance(value, torch.Tensor) or key in skip:
                continue
            safe = _json_safe(value)
            if safe is not None or value is None:
                meta[str(key)] = safe
        info.metadata = meta
        info.hints = extract_hints(meta)
    return info
