from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentInfo:
    python_version: tuple[int, int, int]
    python_executable: str
    platform: str
    machine: str
    system: str
    release: str
    packages: dict[str, str]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_environment(
    package_names: list[str] | tuple[str, ...] = (),
) -> EnvironmentInfo:

    packages: dict[str, str] = {}

    for package in package_names:
        version = _package_version(package)

        if version is not None:
            packages[package] = version

    return EnvironmentInfo(
        python_version=sys.version_info[:3],
        python_executable=sys.executable,
        platform=platform.platform(),
        machine=platform.machine(),
        system=platform.system(),
        release=platform.release(),
        packages=packages,
    )


def import_available(import_name: str) -> bool:
    try:
        importlib.import_module(import_name)
        return True
    except Exception:
        return False


def imported_version(
    import_name: str,
    package_name: str,
) -> str | None:

    try:
        importlib.import_module(import_name)
    except Exception:
        return None

    return _package_version(package_name)
