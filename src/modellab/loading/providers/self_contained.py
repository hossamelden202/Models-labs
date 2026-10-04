from __future__ import annotations

from .base import (
    ArchitectureBuildContext,
    ArchitectureProvider,
)


class SelfContainedProvider(ArchitectureProvider):

    provider_id = "self_contained"
    display_name = "Self-Contained Model"

    def can_handle(
        self,
        *,
        architecture,
        source,
        artifact,
        context=None,
    ):

        if source is not None:
            return source.source_type == "self_contained"

        return bool(
            getattr(artifact, "self_contained", False)
        )

    def build(self, context: ArchitectureBuildContext):

        raise NotImplementedError(
            "self-contained construction must be delegated to "
            "ModelLab's controlled artifact loader"
        )
