from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from modellab.core.errors import ConfigurationError


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(_Section):
    name: str = "modellab"
    seed: int = Field(42, ge=0, lt=2**32)


class RuntimeConfig(_Section):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class PathsConfig(_Section):
    data: Path = Path("data")
    models: Path = Path("models")
    artifacts: Path = Path("artifacts")
    experiments: Path = Path("experiments")


class Config(_Section):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


def _describe(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        where = ".".join(str(p) for p in err["loc"])
        parts.append(f"{where}: {err['msg']}")
    return "; ".join(parts)


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        return Config()

    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(f"config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {path}: {exc}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigurationError(f"{path}: top level must be a mapping")

    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid config {path}: {_describe(exc)}") from exc
