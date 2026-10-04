from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArchitectureSource:
    """
    Describes where an architecture implementation comes from.

    Supported source types are intentionally descriptive rather
    than executable:

        registered
        self_contained
        local
        factory
        huggingface
        package
        unknown
    """

    source_type: str
    identifier: str | None = None
    path: str | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArchitectureEvidence:
    """
    Evidence discovered during artifact inspection.

    Evidence is NOT automatically promoted into an architecture
    decision.
    """

    field: str
    value: Any
    source: str
    confidence: float = 0.0


@dataclass(frozen=True)
class ArchitectureCandidate:
    """
    A possible architecture reconstruction route.
    """

    architecture: str
    source: ArchitectureSource
    confidence: float = 0.0
    reason: str = ""
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArchitectureBuildContext:
    """
    Context supplied to an architecture provider.
    """

    artifact_path: Path | None
    architecture: str | None
    source: ArchitectureSource | None
    config: dict[str, Any]
    num_classes: int | None
    input_size: tuple[int, int] | None
    class_names: list[str] | None
    state_dict_key: str | None
    strip_prefix: str | None
    trust_executable: bool
    options: dict[str, Any] = field(default_factory=dict)


class ArchitectureProvider(ABC):
    """
    Generic contract for architecture providers.

    Providers determine how an architecture can be obtained.
    They must not install arbitrary packages or execute untrusted
    model code implicitly.
    """

    provider_id: str = "unknown"
    display_name: str = "Unknown"

    @abstractmethod
    def can_handle(
        self,
        *,
        architecture: str | None,
        source: ArchitectureSource | None,
        artifact: Any | None,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Return True if this provider can handle the request."""

    def candidates(
        self,
        *,
        artifact: Any | None,
        context: dict[str, Any] | None = None,
    ) -> list[ArchitectureCandidate]:
        """Return safely inferred architecture candidates."""
        return []

    @abstractmethod
    def build(
        self,
        context: ArchitectureBuildContext,
    ) -> Any:
        """Construct the architecture without loading weights."""

    def load_weights(
        self,
        model: Any,
        *,
        state_dict: dict[str, Any],
        context: ArchitectureBuildContext,
    ) -> Any:
        """
        Default strict state-dict loader.
        """

        loader = getattr(model, "load_state_dict", None)

        if loader is None:
            raise TypeError(
                f"provider {self.provider_id!r} produced an object "
                "without load_state_dict()"
            )

        loader(state_dict, strict=True)
        return model

    def describe(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "display_name": self.display_name,
        }
