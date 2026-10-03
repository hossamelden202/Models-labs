import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServerSettings:
    workspace: Path
    token: str | None = None
    max_workers: int = 1
    allow_local_paths: bool = True
    max_upload_bytes: int = 20 * 1024**3
    cors_origins: tuple[str, ...] = ("*",)
    allow_custom_code: bool = True
    allow_pickle: bool = True

    @classmethod
    def from_env(cls) -> "ServerSettings":
        return cls(
            workspace=Path(os.environ.get("MODELLAB_WORKSPACE", "./workspace")),
            token=os.environ.get("MODELLAB_TOKEN") or None,
            max_workers=int(os.environ.get("MODELLAB_WORKERS", "1")),
            allow_local_paths=os.environ.get("MODELLAB_ALLOW_LOCAL_PATHS", "1") != "0",
            allow_custom_code=os.environ.get("MODELLAB_ALLOW_CUSTOM_CODE", "1") != "0",
            allow_pickle=os.environ.get("MODELLAB_ALLOW_PICKLE", "1") != "0",
        )
