from __future__ import annotations

from typing import Any

from .base import (
    ArchitectureBuildContext,
    ArchitectureCandidate,
    ArchitectureProvider,
    ArchitectureSource,
)


class RegisteredProvider(ArchitectureProvider):
    """
    Adapter for ModelLab's existing registered architecture system.

    The architecture registry is the source of truth. This provider
    does not duplicate architecture implementations.
    """

    provider_id = "registered"
    display_name = "Registered ModelLab Architecture"

    def can_handle(
        self,
        *,
        architecture: str | None,
        source: ArchitectureSource | None,
        artifact: Any,
        context: dict[str, Any],
    ) -> bool:

        if source is not None and source.source_type != self.provider_id:
            return False

        if not architecture:
            return False

        return self._architecture_exists(architecture)

    def candidates(
        self,
        *,
        architecture: str | None,
        source: ArchitectureSource | None,
        artifact: Any,
        context: dict[str, Any],
    ) -> list[ArchitectureCandidate]:

        if not self.can_handle(
            architecture=architecture,
            source=source,
            artifact=artifact,
            context=context,
        ):
            return []

        assert architecture is not None

        resolved_source = source or ArchitectureSource(
            source_type=self.provider_id,
            identifier=architecture,
        )

        return [
            ArchitectureCandidate(
                architecture=architecture,
                source=resolved_source,
                confidence=1.0,
                reason=(
                    "architecture exists in ModelLab's "
                    "registered architecture registry"
                ),
            )
        ]

    def build(
        self,
        context: ArchitectureBuildContext,
    ) -> Any:

        if not context.architecture:
            raise ValueError(
                "registered architecture provider requires "
                "an architecture name"
            )

        from modellab.loading.architectures import build_architecture

        config = dict(context.config or {})

        return build_architecture(
            context.architecture,
            config=config,
            num_classes=context.num_classes,
        )

    @staticmethod
    def _architecture_exists(
        architecture: str,
    ) -> bool:

        from modellab.loading.architectures import get_architecture

        try:
            get_architecture(architecture)
            return True
        except Exception:
            return False

    def describe(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "display_name": self.display_name,
            "kind": "registered",
            "supports": [
                "existing ModelLab architecture registry",
            ],
        }
