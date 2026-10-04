from __future__ import annotations

from .base import (
    ArchitectureBuildContext,
    ArchitectureProvider,
)


class FactoryProvider(ArchitectureProvider):

    provider_id = "factory"
    display_name = "Explicit Model Factory"

    def can_handle(
        self,
        *,
        architecture,
        source,
        artifact,
        context=None,
    ):

        return bool(
            source is not None
            and source.source_type == "factory"
        )

    def build(self, context: ArchitectureBuildContext):

        raise NotImplementedError(
            "factory construction must be delegated to "
            "ModelLab's existing trusted factory mechanism"
        )
