from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .models import RuntimeRequirement
from .registry import get_runtime


@dataclass
class ProvisionResult:
    runtime_id: str
    ok: bool
    environment_path: str | None = None
    python_executable: str | None = None
    installed_packages: list[str] | None = None
    detail: str = ""
    errors: list[str] | None = None


def _default_runtime_root() -> Path:
    configured = os.environ.get("MODELLAB_RUNTIME_ROOT")

    if configured:
        return Path(configured).expanduser()

    return Path.home() / ".modellab" / "runtimes"


def _metadata_path(environment_path: Path) -> Path:
    return environment_path / "modellab-runtime.json"


def _host_python() -> Path:
    return Path(sys.executable).resolve()


def _runtime_python_path(
    environment_path: Path,
) -> Path:
    return environment_path / "bin" / "python"


def _runtime_site_packages(
    environment_path: Path,
) -> Path:
    return environment_path / "site-packages"


def _verify_import(
    python_executable: Path,
    import_name: str,
) -> tuple[bool, str]:

    code = (
        "import importlib\n"
        f"module = importlib.import_module({import_name!r})\n"
        "print(getattr(module, '__version__', 'import-ok'))\n"
    )

    result = subprocess.run(
        [str(python_executable), "-c", code],
        text=True,
        capture_output=True,
        check=False,
    )

    if result.returncode == 0:
        return True, result.stdout.strip() or "import-ok"

    detail = (
        result.stderr.strip()
        or result.stdout.strip()
        or "runtime import failed"
    )

    return False, detail


def _write_runtime_python(
    runtime_python: Path,
    host_python: Path,
    site_packages: Path,
) -> None:

    runtime_python.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    launcher = f"""#!/usr/bin/env bash
set -e

export PYTHONPATH="{site_packages}:$PYTHONPATH"

exec "{host_python}" "$@"
"""

    runtime_python.write_text(
        launcher,
        encoding="utf-8",
    )

    runtime_python.chmod(0o755)


def provision_runtime(
    requirement: RuntimeRequirement,
    *,
    runtime_root: str | Path | None = None,
) -> ProvisionResult:

    runtime = get_runtime(requirement.runtime_id)

    if runtime is None:
        return ProvisionResult(
            runtime_id=requirement.runtime_id,
            ok=False,
            detail=f"unknown runtime: {requirement.runtime_id}",
            errors=[
                f"unknown runtime: {requirement.runtime_id}"
            ],
        )

    if not runtime.allow_provision:
        return ProvisionResult(
            runtime_id=runtime.runtime_id,
            ok=False,
            detail=(
                f"runtime '{runtime.runtime_id}' "
                "is not provisionable"
            ),
            errors=[
                (
                    f"runtime '{runtime.runtime_id}' "
                    "is not provisionable"
                )
            ],
        )

    root = (
        Path(runtime_root).expanduser()
        if runtime_root is not None
        else _default_runtime_root()
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    environment_path = root / runtime.runtime_id

    runtime_python = _runtime_python_path(
        environment_path
    )

    site_packages = _runtime_site_packages(
        environment_path
    )

    host_python = _host_python()

    # ------------------------------------------------------------
    # Remove any broken/old runtime.
    # ------------------------------------------------------------

    if environment_path.exists():

        if runtime_python.exists():

            ok, detail = _verify_import(
                runtime_python,
                runtime.import_name,
            )

            if ok:
                return ProvisionResult(
                    runtime_id=runtime.runtime_id,
                    ok=True,
                    environment_path=str(
                        environment_path
                    ),
                    python_executable=str(
                        runtime_python
                    ),
                    installed_packages=[],
                    detail=(
                        "existing controlled runtime "
                        f"verified: {detail}"
                    ),
                    errors=[],
                )

        shutil.rmtree(
            environment_path,
            ignore_errors=True,
        )

    # ------------------------------------------------------------
    # Create controlled runtime directories.
    # ------------------------------------------------------------

    site_packages.mkdir(
        parents=True,
        exist_ok=True,
    )

    runtime_python.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Create controlled Python launcher.
    # ------------------------------------------------------------

    _write_runtime_python(
        runtime_python,
        host_python,
        site_packages,
    )

    # ------------------------------------------------------------
    # Install ONLY the registered runtime package into the
    # runtime's own site-packages.
    #
    # We do NOT modify Kaggle's global Python installation.
    # ------------------------------------------------------------

    packages = [
        runtime.package_spec,
        *runtime.extra_packages,
    ]

    install_command = [
        str(host_python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        "--no-cache-dir",
        "--upgrade-strategy",
        "only-if-needed",
        "--target",
        str(site_packages),
    ]

    # Ultralytics must reuse Kaggle's existing PyTorch.
    # Installing torch/torchvision again inside the runtime would
    # be wasteful and can break the Kaggle environment.
    if runtime.runtime_id == "ultralytics":
        install_command.append("--no-deps")

    install_command.extend(packages)

    install = subprocess.run(
        install_command,
        text=True,
        capture_output=True,
        check=False,
    )

    if install.returncode != 0:

        detail = (
            install.stderr.strip()
            or install.stdout.strip()
            or "runtime installation failed"
        )

        return ProvisionResult(
            runtime_id=runtime.runtime_id,
            ok=False,
            environment_path=str(
                environment_path
            ),
            python_executable=str(
                runtime_python
            ),
            installed_packages=packages,
            detail=detail,
            errors=[detail],
        )

    # ------------------------------------------------------------
    # Verify runtime import.
    # ------------------------------------------------------------

    ok, detail = _verify_import(
        runtime_python,
        runtime.import_name,
    )

    if not ok:

        return ProvisionResult(
            runtime_id=runtime.runtime_id,
            ok=False,
            environment_path=str(
                environment_path
            ),
            python_executable=str(
                runtime_python
            ),
            installed_packages=packages,
            detail=(
                f"runtime import failed: "
                f"{runtime.import_name}: {detail}"
            ),
            errors=[
                (
                    f"runtime import failed: "
                    f"{runtime.import_name}: {detail}"
                )
            ],
        )

    # ------------------------------------------------------------
    # Verify PyTorch reuse.
    # ------------------------------------------------------------

    if runtime.runtime_id == "ultralytics":

        torch_check = subprocess.run(
            [
                str(runtime_python),
                "-c",
                (
                    "import torch; "
                    "print(torch.__version__); "
                    "print('cuda=' + "
                    "str(torch.cuda.is_available())); "
                    "print(torch.__file__)"
                ),
            ],
            text=True,
            capture_output=True,
            check=False,
        )

        if torch_check.returncode != 0:

            detail = (
                torch_check.stderr.strip()
                or torch_check.stdout.strip()
                or "PyTorch import failed"
            )

            return ProvisionResult(
                runtime_id=runtime.runtime_id,
                ok=False,
                environment_path=str(
                    environment_path
                ),
                python_executable=str(
                    runtime_python
                ),
                installed_packages=packages,
                detail=detail,
                errors=[
                    (
                        "controlled Ultralytics runtime "
                        f"cannot import PyTorch: {detail}"
                    )
                ],
            )

        torch_output = torch_check.stdout.strip()

    else:
        torch_output = None

    # ------------------------------------------------------------
    # Runtime metadata.
    # ------------------------------------------------------------

    metadata = {
        "runtime_id": runtime.runtime_id,
        "package": runtime.package,
        "package_spec": runtime.package_spec,
        "import_name": runtime.import_name,
        "environment_path": str(
            environment_path
        ),
        "python_executable": str(
            runtime_python
        ),
        "host_python": str(
            host_python
        ),
        "site_packages": str(
            site_packages
        ),
        "mode": "controlled-target-site-packages",
        "installed_packages": packages,
        "torch_check": torch_output,
    }

    _metadata_path(
        environment_path
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    return ProvisionResult(
        runtime_id=runtime.runtime_id,
        ok=True,
        environment_path=str(
            environment_path
        ),
        python_executable=str(
            runtime_python
        ),
        installed_packages=packages,
        detail=(
            "controlled runtime provisioned and "
            f"verified: {detail}"
        ),
        errors=[],
    )
