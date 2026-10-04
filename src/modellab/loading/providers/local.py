from __future__ import annotations

from pathlib import Path

from .base import (
    ArchitectureBuildContext,
    ArchitectureProvider,
)


class LocalProvider(ArchitectureProvider):

    provider_id = "local"
    display_name = "Local Architecture Source"

    def can_handle(
        self,
        *,
        architecture,
        source,
        artifact,
        context=None,
    ):

        if source is None:
            return False

        if source.source_type != "local":
            return False

        if not source.path:
            return False

        return Path(source.path).exists()

    def build(self, context: ArchitectureBuildContext):

        raise NotImplementedError(
            "local architecture loading must use ModelLab's "
            "controlled runtime and trust policy"
        )
