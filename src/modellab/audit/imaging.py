import hashlib
import os
import stat
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from modellab.audit.config import AuditConfig
from modellab.audit.scan import INVALID_PATH, Record

_CHUNK = 1 << 20


class _InvalidDimensions(Exception):
    pass


def _roots(root) -> list[str]:
    variants = {os.path.abspath(root), str(Path(root).resolve())}
    return sorted((v for v in variants if v not in ("", os.sep)), key=len, reverse=True)


def clean_message(text: str, roots: list[str]) -> str:
    for variant in roots:
        text = text.replace(variant, "<root>")
    return " ".join(text.split())[:300]


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash(gray: Image.Image) -> int:
    small = gray.resize((9, 8), Image.Resampling.LANCZOS)
    arr = np.asarray(small, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _derive(rec: Record, img: Image.Image, cfg: AuditConfig) -> None:
    rgb = img.convert("RGB")
    rec.phash = dhash(rgb.convert("L"))
    if not cfg.compute_stats:
        return
    small = np.asarray(rgb.resize((32, 32), Image.Resampling.BOX), dtype=np.float64) / 255.0
    luma = small @ np.array([0.299, 0.587, 0.114])
    peak = small.max(axis=2)
    spread = peak - small.min(axis=2)
    saturation = np.where(peak > 0, spread / np.where(peak > 0, peak, 1.0), 0.0)
    rec.brightness = round(float(luma.mean()), 6)
    rec.contrast = round(float(luma.std()), 6)
    rec.saturation = round(float(saturation.mean()), 6)


def _fail(rec: Record, status: str, stage: str, error_type: str, message: str) -> None:
    rec.status = status
    rec.error_stage = stage
    rec.error_type = error_type
    rec.error_message = message


def inspect_record(rec: Record, root, cfg: AuditConfig, roots: list[str]) -> None:
    if rec.path == INVALID_PATH:
        _fail(
            rec,
            "invalid_path",
            "path",
            "InvalidPath",
            "path is absolute, escapes the dataset root or is malformed; not opened",
        )
        return

    rec.extension = PurePosixPath(rec.path).suffix.lower()
    full = Path(root) / rec.path
    try:
        info = full.stat()
    except FileNotFoundError:
        _fail(rec, "missing", "stat", "FileNotFoundError", "file does not exist")
        return
    except OSError as exc:
        _fail(rec, "unreadable", "stat", type(exc).__name__, clean_message(str(exc), roots))
        return
    if not stat.S_ISREG(info.st_mode):
        _fail(rec, "unreadable", "stat", "NotAFile", "path is not a regular file")
        return

    rec.file_size = int(info.st_size)
    try:
        rec.sha256 = sha256_file(full)
    except OSError as exc:
        _fail(rec, "unreadable", "read", type(exc).__name__, clean_message(str(exc), roots))
        return

    try:
        with Image.open(full) as img:
            rec.format = img.format
            rec.mode = img.mode
            rec.width, rec.height = int(img.size[0]), int(img.size[1])
            if rec.width <= 0 or rec.height <= 0:
                raise _InvalidDimensions("image has zero or negative dimensions")
            img.load()
            try:
                _derive(rec, img, cfg)
            except Exception as exc:
                rec.warnings.append(f"features_failed:{type(exc).__name__}")
    except Exception as exc:
        kind = "InvalidDimensions" if isinstance(exc, _InvalidDimensions) else type(exc).__name__
        _fail(rec, "unreadable", "decode", kind, clean_message(str(exc), roots))
        return
    rec.status = "ok"


def inspect_records(records: list[Record], root, cfg: AuditConfig) -> None:
    roots = _roots(root)
    for rec in records:
        try:
            inspect_record(rec, root, cfg, roots)
        except Exception as exc:
            _fail(rec, "unreadable", "inspect", type(exc).__name__, clean_message(str(exc), roots))
