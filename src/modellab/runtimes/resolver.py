from __future__ import annotations

import sys

from .environment import import_available, imported_version
from .models import RuntimeRequirement, RuntimeStatus
from .registry import get_runtime


def _python_compatible(
    minimum: tuple[int, int] | None,
    maximum: tuple[int, int] | None,
) -> tuple[bool, str]:

    current = sys.version_info[:2]

    if minimum and current < minimum:
        return (
            False,
            f"Python {current[0]}.{current[1]} is below "
            f"minimum {minimum[0]}.{minimum[1]}",
        )

    if maximum and current > maximum:
        return (
            False,
            f"Python {current[0]}.{current[1]} is above "
            f"maximum {maximum[0]}.{maximum[1]}",
        )

    return True, ""


def resolve_runtime(
    requirement: RuntimeRequirement,
) -> RuntimeStatus:

    runtime = get_runtime(requirement.runtime_id)

    if runtime is None:
        return RuntimeStatus(
            runtime_id=requirement.runtime_id,
            state="unknown",
            detail=(
                f"runtime '{requirement.runtime_id}' is not present "
                "in the controlled ModelLab runtime registry"
            ),
            errors=[
                f"unknown runtime: {requirement.runtime_id}",
            ],
        )

    compatible, detail = _python_compatible(
        runtime.python_min,
        runtime.python_max,
    )

    if not compatible:
        return RuntimeStatus(
            runtime_id=runtime.runtime_id,
            state="incompatible",
            import_name=runtime.import_name,
            detail=detail,
            errors=[detail],
        )

    if import_available(runtime.import_name):
        return RuntimeStatus(
            runtime_id=runtime.runtime_id,
            state="available",
            installed_version=imported_version(
                runtime.import_name,
                runtime.package,
            ),
            import_name=runtime.import_name,
            detail=(
                f"{runtime.display_name} is available in the "
                "current Python environment"
            ),
        )

    if runtime.allow_provision:
        return RuntimeStatus(
            runtime_id=runtime.runtime_id,
            state="provisionable",
            import_name=runtime.import_name,
            detail=(
                f"{runtime.display_name} is not available, "
                "but a controlled runtime definition exists"
            ),
        )

    return RuntimeStatus(
        runtime_id=runtime.runtime_id,
        state="missing",
        import_name=runtime.import_name,
        detail=(
            f"{runtime.display_name} is not installed and "
            "automatic provisioning is disabled"
        ),
    )
