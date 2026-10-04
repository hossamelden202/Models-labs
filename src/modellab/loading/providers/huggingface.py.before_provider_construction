from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import (
    ArchitectureBuildContext,
    ArchitectureCandidate,
    ArchitectureProvider,
    ArchitectureSource,
)


class HuggingFaceProvider(ArchitectureProvider):
    """
    Generic Hugging Face architecture provider.

    This provider deliberately does NOT contain model-specific
    implementations.

    Discovery is metadata-only:
      - Hugging Face repository identifier
      - local Hugging Face directory
      - config.json

    Actual model construction belongs to the controlled
    Transformers runtime.
    """

    provider_id = "huggingface"
    display_name = "Hugging Face Transformers"

    _SOURCE_TYPES = frozenset(
        {
            "huggingface",
            "hf",
        }
    )

    def can_handle(
        self,
        *,
        architecture: str | None,
        source: ArchitectureSource | None,
        artifact: Any,
        context: dict[str, Any],
    ) -> bool:

        if source is None:
            return False

        if source.source_type not in self._SOURCE_TYPES:
            return False

        # Explicit HF repository identifier.
        if source.identifier:
            return True

        # Explicit local HF directory.
        if source.path:
            path = Path(source.path)

            if not path.exists():
                return False

            if not path.is_dir():
                return False

            return (path / "config.json").is_file()

        return False

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

        assert source is not None

        config = self._read_config(source)

        model_type = config.get("model_type")
        architectures = config.get("architectures")

        # Architecture is a model identifier supplied by the HF
        # ecosystem, not a ModelLab registered architecture name.
        identifier = (
            architecture
            or model_type
            or source.identifier
        )

        if not identifier:
            return []

        reason_parts = [
            "source is a Hugging Face Transformers model"
        ]

        if model_type:
            reason_parts.append(
                f"config.json declares model_type={model_type!r}"
            )

        if architectures:
            reason_parts.append(
                "config.json declares Transformers architectures"
            )

        return [
            ArchitectureCandidate(
                architecture=identifier,
                source=source,
                confidence=1.0 if model_type else 0.85,
                reason="; ".join(reason_parts),
                config=config,
            )
        ]

    def build(
        self,
        context: ArchitectureBuildContext,
    ) -> Any:
        """
        Construction is intentionally delegated to the controlled
        Transformers runtime.

        The provider itself must never dynamically install packages
        or execute arbitrary remote model code.
        """

        raise RuntimeError(
            "Hugging Face model construction must run through the "
            "controlled Transformers runtime"
        )

    @staticmethod
    def _read_config(
        source: ArchitectureSource,
    ) -> dict[str, Any]:

        if source.path:
            config_path = Path(source.path) / "config.json"

            if not config_path.is_file():
                raise FileNotFoundError(
                    f"Hugging Face config.json not found: {config_path}"
                )

            try:
                data = json.loads(
                    config_path.read_text(
                        encoding="utf-8"
                    )
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid Hugging Face config.json: {config_path}"
                ) from exc

            if not isinstance(data, dict):
                raise ValueError(
                    "Hugging Face config.json must contain a JSON object"
                )

            return data

        # Remote repository identifiers cannot be fetched during
        # pure provider discovery. The controlled runtime will
        # resolve the repository later.
        return dict(source.config or {})

    def describe(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "display_name": self.display_name,
            "kind": "huggingface",
            "supports": [
                "Hugging Face repository identifiers",
                "local Hugging Face model directories",
                "config.json metadata discovery",
                "controlled Transformers runtime construction",
            ],
        }
