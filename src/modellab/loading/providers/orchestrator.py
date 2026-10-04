
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import (
    ArchitectureCandidate,
    ArchitectureProvider,
    ArchitectureSource,
)
from .registry import (
    discover_architecture_providers,
    collect_candidates,
)


@dataclass(frozen=True)
class ProviderResolution:
    """
    Result of generic architecture-provider resolution.

    This object intentionally contains no model-specific logic.
    """

    status: str
    provider_id: str | None = None
    architecture: str | None = None
    source: ArchitectureSource | None = None
    confidence: float = 0.0
    reason: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    candidates: list[ArchitectureCandidate] = field(
        default_factory=list
    )
    errors: list[str] = field(
        default_factory=list
    )


class ProviderOrchestrator:
    """
    Generic architecture-provider selection layer.

    Responsibilities:
      1. discover providers
      2. collect candidates
      3. rank candidates
      4. reject ambiguous low-confidence results
      5. return a provider-independent resolution

    It does NOT:
      - import model libraries
      - install packages
      - load model weights
      - contain model-specific branches
    """

    def resolve(
        self,
        *,
        architecture: str | None = None,
        source: ArchitectureSource | None = None,
        artifact: Any = None,
        context: dict[str, Any] | None = None,
        min_confidence: float = 0.80,
    ) -> ProviderResolution:

        context = dict(context or {})

        try:
            matched = discover_architecture_providers(
                architecture=architecture,
                source=source,
                artifact=artifact,
                context=context,
            )
        except Exception as exc:
            return ProviderResolution(
                status="error",
                reason="provider discovery failed",
                errors=[str(exc)],
            )

        if not matched:
            return ProviderResolution(
                status="needs_input",
                reason=(
                    "no registered architecture provider "
                    "can handle the supplied architecture/source"
                ),
            )

        candidates: list[ArchitectureCandidate] = []

        for provider in matched:
            try:
                found = provider.candidates(
                    architecture=architecture,
                    source=source,
                    artifact=artifact,
                    context=context,
                )

                for candidate in found:
                    candidates.append(candidate)

            except Exception as exc:
                return ProviderResolution(
                    status="error",
                    reason=(
                        f"provider '{provider.provider_id}' "
                        "failed while generating candidates"
                    ),
                    errors=[str(exc)],
                )

        if not candidates:
            return ProviderResolution(
                status="needs_input",
                reason=(
                    "matching providers were found, but "
                    "none produced an architecture candidate"
                ),
            )

        ranked = sorted(
            candidates,
            key=lambda item: float(item.confidence),
            reverse=True,
        )

        best = ranked[0]

        if len(ranked) > 1:
            second = ranked[1]

            if (
                best.architecture != second.architecture
                and best.confidence == second.confidence
                and best.confidence >= min_confidence
            ):
                return ProviderResolution(
                    status="ambiguous",
                    reason=(
                        "multiple architecture candidates have "
                        "the same confidence"
                    ),
                    candidates=ranked,
                )

        if best.confidence < min_confidence:
            return ProviderResolution(
                status="needs_input",
                reason=(
                    "the strongest architecture candidate "
                    "does not meet the minimum confidence"
                ),
                candidates=ranked,
            )

        provider_id = None

        for provider in matched:
            try:
                provider_candidates = provider.candidates(
                    architecture=architecture,
                    source=source,
                    artifact=artifact,
                    context=context,
                )

                if best in provider_candidates:
                    provider_id = provider.provider_id
                    break

            except Exception:
                continue

        if provider_id is None:
            return ProviderResolution(
                status="error",
                reason=(
                    "could not associate the selected candidate "
                    "with its provider"
                ),
                candidates=ranked,
            )

        return ProviderResolution(
            status="resolved",
            provider_id=provider_id,
            architecture=best.architecture,
            source=best.source,
            confidence=float(best.confidence),
            reason=best.reason,
            config=dict(best.config),
            candidates=ranked,
        )


def resolve_provider(
    *,
    architecture: str | None = None,
    source: ArchitectureSource | None = None,
    artifact: Any = None,
    context: dict[str, Any] | None = None,
    min_confidence: float = 0.80,
) -> ProviderResolution:

    return ProviderOrchestrator().resolve(
        architecture=architecture,
        source=source,
        artifact=artifact,
        context=context,
        min_confidence=min_confidence,
    )
