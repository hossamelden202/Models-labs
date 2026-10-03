import json
import os
import re
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from modellab.server.errors import ApiError

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_id(value, what: str) -> str:
    if not isinstance(value, str) or not _ID.match(value):
        raise ApiError(422, f"invalid {what} '{value}': use letters, digits, '_', '-' or '.', at most 64 characters")
    return value


def slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-._")[:40] or "item"
    return f"{cleaned}-{uuid.uuid4().hex[:6]}"


def safe_join(root: Path, sub: str | None) -> Path:
    if not sub:
        return root
    text = sub.replace("\\", "/")
    if text.startswith("/") or ".." in PurePosixPath(text).parts:
        raise ApiError(422, "subpath must be relative and stay inside the dataset")
    target = (root / text).resolve()
    if not target.is_dir() or not target.is_relative_to(root.resolve()):
        raise ApiError(422, f"subpath '{sub}' is not a directory inside the dataset")
    return target


def extract_zip(zip_path: Path, dest: Path, max_bytes: int, max_files: int = 500_000) -> int:
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise ApiError(400, "the upload is not a valid zip archive") from exc
    root = dest.resolve()
    written = count = 0
    with archive:
        infos = archive.infolist()
        if len(infos) > max_files:
            raise ApiError(413, f"the archive has more than {max_files} entries")
        for info in infos:
            name = info.filename.replace("\\", "/")
            parts = PurePosixPath(name).parts
            if not parts or parts[0] == "__MACOSX" or parts[-1] == ".DS_Store":
                continue
            if name.startswith("/") or ".." in parts or re.match(r"^[A-Za-z]:", parts[0]):
                raise ApiError(400, f"unsafe path in archive: {info.filename}")
            if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise ApiError(400, f"symbolic links are not allowed in archives: {info.filename}")
            target = root.joinpath(*parts).resolve()
            if not target.is_relative_to(root):
                raise ApiError(400, f"unsafe path in archive: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, open(target, "wb") as out:
                while chunk := src.read(1 << 20):
                    written += len(chunk)
                    if written > max_bytes:
                        raise ApiError(413, "the archive expands beyond the configured size limit")
                    out.write(chunk)
            count += 1
    return count


def find_root(path: Path) -> Path:
    while True:
        entries = [p for p in path.iterdir() if not p.name.startswith(".")]
        if (
            len(entries) == 1
            and entries[0].is_dir()
            and any(c.is_dir() for c in entries[0].iterdir() if not c.name.startswith("."))
        ):
            path = entries[0]
            continue
        return path


class Workspace:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.models_dir = self.root / "models"
        self.datasets_dir = self.root / "datasets"
        self.jobs_dir = self.root / "jobs"
        self.artifacts_dir = self.root / "artifacts"
        self.tmp_dir = self.root / "tmp"
        self.stage_dir = self.root / "model_artifacts"
        for directory in (self.models_dir, self.datasets_dir, self.jobs_dir, self.artifacts_dir, self.tmp_dir, self.stage_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def write_json(path: Path, obj) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
        os.replace(tmp, path)

    @staticmethod
    def read_json(path: Path):
        return json.loads(path.read_text())

    def _get(self, base: Path, id_: str, what: str) -> dict:
        check_id(id_, f"{what} id")
        path = base / id_ / f"{what}.json"
        if not path.is_file():
            raise ApiError(404, f"{what} '{id_}' not found")
        return self.read_json(path)

    def get_model(self, id_: str) -> dict:
        return self._get(self.models_dir, id_, "model")

    def get_dataset(self, id_: str) -> dict:
        return self._get(self.datasets_dir, id_, "dataset")

    def get_staged(self, id_: str) -> dict:
        return self._get(self.stage_dir, id_, "artifact")

    def list_records(self, base: Path, what: str) -> list[dict]:
        return [self.read_json(p) for p in sorted(base.glob(f"*/{what}.json"))]

    def delete(self, base: Path, id_: str, what: str) -> None:
        self._get(base, id_, what)
        shutil.rmtree(base / id_)
