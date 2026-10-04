from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class RuntimeRequirement:
    """
    Runtime requirement inferred from a model artifact.

    This describes a controlled ModelLab runtime, not an arbitrary
    pip dependency.
    """

    runtime_id: str
    reason: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class RuntimeDefinition:
    """
    Approved runtime definition.

    package is the package installed into the isolated environment.
    import_name is the Python module used for verification.
    """

    runtime_id: str
    display_name: str
    package: str
    import_name: str
    package_spec: str

    python_min: tuple[int, int] | None = None
    python_max: tuple[int, int] | None = None

    description: str = ""

    capabilities: tuple[str, ...] = ()

    extra_packages: tuple[str, ...] = ()

    tags: tuple[str, ...] = ()

    allow_provision: bool = True


RuntimeState = Literal[
    "available",
    "missing",
    "incompatible",
    "provisionable",
    "unknown",
]


@dataclass
class RuntimeStatus:
    runtime_id: str
    state: RuntimeState
    installed_version: str | None = None
    environment_path: str | None = None
    import_name: str | None = None
    detail: str = ""
    errors: list[str] = field(default_factory=list)
