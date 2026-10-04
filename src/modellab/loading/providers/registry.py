from __future__ import annotations

from typing import Any

from .base import (
    ArchitectureCandidate,
    ArchitectureProvider,
    ArchitectureSource,
)
from .factory import FactoryProvider
from .huggingface import HuggingFaceProvider
from .local import LocalProvider
from .registered import RegisteredProvider
from .self_contained import SelfContainedProvider


_PROVIDERS: dict[str, ArchitectureProvider] = {}


def register_provider(
    provider: ArchitectureProvider,
    *,
    replace: bool = False,
) -> ArchitectureProvider:
    """
    Register an architecture provider.

    Providers are keyed by their stable provider_id.
    """

    provider_id = provider.provider_id

    if not provider_id:
        raise ValueError(
            "architecture provider must define provider_id"
        )

    if provider_id in _PROVIDERS and not replace:
        raise ValueError(
            f"architecture provider already registered: {provider_id}"
        )

    _PROVIDERS[provider_id] = provider

    return provider


def unregister_provider(
    provider_id: str,
) -> ArchitectureProvider | None:
    """
    Remove and return a registered provider.
    """

    return _PROVIDERS.pop(
        provider_id,
        None,
    )


def get_provider(
    provider_id: str,
) -> ArchitectureProvider:
    """
    Return a provider by stable provider_id.
    """

    try:
        return _PROVIDERS[provider_id]
    except KeyError as exc:
        raise KeyError(
            f"unknown architecture provider: {provider_id}"
        ) from exc


def list_providers() -> list[ArchitectureProvider]:
    """
    Return registered providers in deterministic order.
    """

    return list(_PROVIDERS.values())


def providers() -> tuple[ArchitectureProvider, ...]:
    """
    Immutable provider snapshot.
    """

    return tuple(_PROVIDERS.values())


def find_providers(
    *,
    architecture: str | None = None,
    source: ArchitectureSource | None = None,
    artifact: Any = None,
    context: dict[str, Any] | None = None,
) -> list[ArchitectureProvider]:
    """
    Return providers capable of handling the supplied request.
    """

    context = dict(context or {})

    matches: list[ArchitectureProvider] = []

    for provider in providers():
        try:
            if provider.can_handle(
                architecture=architecture,
                source=source,
                artifact=artifact,
                context=context,
            ):
                matches.append(provider)
        except Exception:
            # Provider discovery must remain defensive.
            # A broken provider must not break discovery of
            # unrelated providers.
            continue

    return matches


def discover_architecture_providers(
    *,
    architecture: str | None = None,
    source: ArchitectureSource | None = None,
    artifact: Any = None,
    context: dict[str, Any] | None = None,
) -> list[ArchitectureProvider]:
    """
    Discover providers capable of handling an architecture request.

    This is intentionally a pure discovery operation. It does not
    construct models, load weights, install packages, or execute
    arbitrary model code.
    """

    return find_providers(
        architecture=architecture,
        source=source,
        artifact=artifact,
        context=context,
    )


def collect_candidates(
    *,
    architecture: str | None = None,
    source: ArchitectureSource | None = None,
    artifact: Any = None,
    context: dict[str, Any] | None = None,
) -> list[ArchitectureCandidate]:
    """
    Ask every capable provider for architecture candidates.
    """

    context = dict(context or {})

    candidates: list[ArchitectureCandidate] = []

    for provider in find_providers(
        architecture=architecture,
        source=source,
        artifact=artifact,
        context=context,
    ):
        try:
            candidates.extend(
                provider.candidates(
                    architecture=architecture,
                    source=source,
                    artifact=artifact,
                    context=context,
                )
            )
        except Exception:
            # Candidate collection is defensive for the same reason
            # as provider discovery.
            continue

    return candidates


def register_defaults(
    *,
    replace: bool = False,
) -> None:
    """
    Register ModelLab's built-in architecture providers.

    Order is deterministic and intentionally keeps specialized
    providers before the generic local/factory fallbacks.
    """

    defaults = (
        SelfContainedProvider(),
        RegisteredProvider(),
        FactoryProvider(),
        LocalProvider(),
        HuggingFaceProvider(),
    )

    for provider in defaults:
        if provider.provider_id in _PROVIDERS:
            if replace:
                register_provider(
                    provider,
                    replace=True,
                )
            continue

        register_provider(provider)


# Populate the registry when this module is imported.
register_defaults()
