import contextlib
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from modellab.core.errors import ArtifactError


class ArtifactStore:
    def __init__(self, root: str | Path):
        root = Path(root)
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ArtifactError(f"cannot use artifact root {root}: {exc}") from exc
        self.root = root.resolve()

    def resolve(self, relative: str | Path) -> Path:
        rel = Path(relative)
        if rel.is_absolute():
            raise ArtifactError(f"artifact path must be relative: {relative}")
        target = (self.root / rel).resolve()
        if target == self.root or not target.is_relative_to(self.root):
            raise ArtifactError(f"invalid artifact path: {relative}")
        return target

    def write_bytes(self, relative: str | Path, data: bytes) -> Path:
        target = self.resolve(relative)
        tmp = target.with_name(target.name + ".tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(data)
            os.replace(tmp, target)
        except OSError as exc:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise ArtifactError(f"could not write {target}: {exc}") from exc
        return target

    def write_text(self, relative: str | Path, text: str) -> Path:
        return self.write_bytes(relative, text.encode("utf-8"))

    def write_json(self, relative: str | Path, obj: Any) -> Path:
        if isinstance(obj, BaseModel):
            obj = obj.model_dump(mode="json")
        try:
            text = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ArtifactError(f"object is not JSON serializable: {exc}") from exc
        return self.write_text(relative, text + "\n")
