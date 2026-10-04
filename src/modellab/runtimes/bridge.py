from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modellab.loading.resolver import (
    ModelRequest,
    ResolveSettings,
    resolve_model,
)

from .models import RuntimeRequirement
from .provisioner import provision_runtime
from .registry import get_runtime
from .resolver import resolve_runtime


@dataclass
class RuntimeResolution:
    resolution: Any
    runtime_required: RuntimeRequirement | None = None
    runtime_status: Any | None = None
    runtime_provisioned: bool = False
    runtime_environment: str | None = None
    runtime_id: str | None = None
    runtime_error: str | None = None


_MISSING_MODULE_RE = re.compile(
    r"No module named ['\"]([^'\"]+)['\"]"
)


def _resolution_errors(resolution: Any) -> list[str]:
    errors = getattr(resolution, "errors", None)

    if errors is None:
        return []

    if isinstance(errors, str):
        return [errors]

    return [str(item) for item in errors]


def _missing_modules(resolution: Any) -> list[str]:
    modules: list[str] = []

    for error in _resolution_errors(resolution):
        match = _MISSING_MODULE_RE.search(error)

        if not match:
            continue

        module = match.group(1).strip()

        if module and module not in modules:
            modules.append(module)

    return modules


def _runtime_for_import(import_name: str):
    for runtime_id in (
        "ultralytics",
        "transformers",
        "timm",
        "tensorflow",
        "onnxruntime",
        "pytorch",
    ):
        runtime = get_runtime(runtime_id)

        if runtime is None:
            continue

        if import_name == runtime.import_name:
            return runtime

        if import_name.startswith(runtime.import_name + "."):
            return runtime

    return None


def detect_runtime_requirement(
    resolution: Any,
) -> RuntimeRequirement | None:

    for module in _missing_modules(resolution):
        runtime = _runtime_for_import(module)

        if runtime is None:
            continue

        return RuntimeRequirement(
            runtime_id=runtime.runtime_id,
            reason=(
                f"model resolution requires Python "
                f"package '{runtime.package}'"
            ),
            confidence=1.0,
        )

    return None


def _request_to_dict(req: ModelRequest) -> dict[str, Any]:
    """
    Serialize only the public ModelRequest contract.

    Server/API subclasses such as ResolveBody may contain transport
    fields such as ``artifact_id``. Those fields belong to the server
    layer and must never cross into the controlled runtime.

    The controlled runtime reconstructs a strict ModelRequest with
    extra="forbid", so serialization must be based on ModelRequest's
    fields rather than the concrete subclass's fields.
    """

    if hasattr(req, "model_dump"):
        raw = req.model_dump(mode="json")
    elif hasattr(req, "dict"):
        raw = req.dict()
    else:
        raw = {}

        for key, value in vars(req).items():
            if key.startswith("_"):
                continue

            if isinstance(value, Path):
                value = str(value)

            raw[key] = value

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # req may be ResolveBody (a ModelRequest subclass).
    # Therefore req.model_dump() can contain fields that are
    # NOT part of ModelRequest itself.
    #
    # Keep only fields declared by the actual ModelRequest
    # contract.
    # --------------------------------------------------------

    model_fields = getattr(ModelRequest, "model_fields", None)

    if model_fields is not None:
        allowed_fields = set(model_fields.keys())
    else:
        # Pydantic v1 compatibility.
        allowed_fields_map = getattr(ModelRequest, "__fields__", {})
        allowed_fields = set(allowed_fields_map.keys())

    result = {
        key: value
        for key, value in raw.items()
        if key in allowed_fields
    }

    return result


def _settings_to_dict(
    settings: ResolveSettings,
) -> dict[str, Any]:

    if hasattr(settings, "model_dump"):
        return settings.model_dump(mode="json")

    if hasattr(settings, "dict"):
        return settings.dict()

    return {
        "allow_custom_code": bool(
            getattr(settings, "allow_custom_code", False)
        ),
        "allow_pickle": bool(
            getattr(settings, "allow_pickle", False)
        ),
    }


def _subprocess_script() -> str:
    return r"""
import json

from modellab.loading.resolver import (
    ModelRequest,
    ResolveSettings,
    resolve_model,
)


def main():
    payload = json.load(__import__("sys").stdin)

    request = ModelRequest(**payload["request"])
    settings = ResolveSettings(**payload["settings"])

    resolution = resolve_model(
        request,
        settings=settings,
        factory_file=payload.get("factory_file"),
    )

    if hasattr(resolution, "model_dump"):
        result = resolution.model_dump(mode="json")
    elif hasattr(resolution, "dict"):
        result = resolution.dict()
    else:
        result = {
            key: value
            for key, value in vars(resolution).items()
            if not key.startswith("_")
        }

    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
"""



def _runtime_environment_marker(
    runtime_id: str | None,
    environment_path: str | Path | None,
) -> dict[str, Any]:
    """
    Return a stable, JSON-safe description of the controlled runtime.

    This is metadata only. It never imports packages and never executes
    model code in the parent process.
    """
    return {
        "runtime_id": runtime_id,
        "environment_path": (
            str(environment_path)
            if environment_path is not None
            else None
        ),
        "controlled": True,
    }


def _run_in_runtime(
    environment_path: str | Path,
    request: ModelRequest,
    settings: ResolveSettings,
    factory_file: str | None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:

    environment_path = Path(environment_path)

    python_path = environment_path / "bin" / "python"

    if not python_path.is_file():
        raise RuntimeError(
            f"runtime Python executable was not found: "
            f"{python_path}"
        )

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    repo_root = Path(repo_root)

    payload = {
        "request": _request_to_dict(request),
        "settings": _settings_to_dict(settings),
        "factory_file": factory_file,
    }

    env = os.environ.copy()

    existing_pythonpath = env.get("PYTHONPATH", "")

    paths = [str(repo_root)]

    if existing_pythonpath:
        paths.append(existing_pythonpath)

    env["PYTHONPATH"] = os.pathsep.join(paths)

    process = subprocess.run(
        [
            str(python_path),
            "-c",
            _subprocess_script(),
        ],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        env=env,
        cwd=str(repo_root),
        check=False,
    )

    if process.returncode != 0:
        stderr = process.stderr.strip()
        stdout = process.stdout.strip()

        detail = (
            stderr
            or stdout
            or (
                "runtime resolver exited with code "
                f"{process.returncode}"
            )
        )

        raise RuntimeError(detail)

    output = process.stdout.strip()

    if not output:
        raise RuntimeError(
            "runtime resolver returned no output"
        )

    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "runtime resolver returned invalid JSON: "
            + output[:2000]
        ) from exc


def resolve_model_with_runtime(
    req: ModelRequest,
    *,
    settings: ResolveSettings | None = None,
    factory_file: str | None = None,
    repo_root: str | Path | None = None,
    runtime_root: str | Path | None = None,
) -> RuntimeResolution:

    settings = settings or ResolveSettings()

    # ------------------------------------------------------------
    # First attempt in the current ModelLab environment.
    # ------------------------------------------------------------

    resolution = resolve_model(
        req,
        settings=settings,
        factory_file=factory_file,
    )

    requirement = detect_runtime_requirement(resolution)

    # No controlled runtime is required.
    if requirement is None:
        return RuntimeResolution(
            resolution=resolution,
        )

    # ------------------------------------------------------------
    # A controlled runtime IS required.
    #
    # Never return the original failed resolution here.
    # That was the previous failure mode.
    # ------------------------------------------------------------

    runtime_status = resolve_runtime(requirement)

    runtime_environment: str | None = None
    runtime_provisioned = False

    # ------------------------------------------------------------
    # If an existing runtime is reported available, determine
    # whether it actually exposes a usable environment.
    # ------------------------------------------------------------

    if runtime_status.state == "available":
        runtime_environment = getattr(
            runtime_status,
            "environment_path",
            None,
        )

        if runtime_environment:
            try:
                runtime_result = _run_in_runtime(
                    runtime_environment,
                    req,
                    settings,
                    factory_file,
                    repo_root=repo_root,
                )

                return RuntimeResolution(
                    resolution=runtime_result,
                    runtime_required=requirement,
            runtime_id=requirement.runtime_id,
                    runtime_status=runtime_status,
                    runtime_provisioned=False,
                    runtime_environment=str(
                        runtime_environment
                    ),
                )

            except Exception as exc:
                # Existing runtime is not actually usable.
                runtime_error = str(exc)
        else:
            runtime_error = (
                "runtime registry reported 'available' but "
                "did not provide an environment path"
            )

        # Fall through to rebuilding/provisioning it.

    else:
        runtime_error = getattr(
            runtime_status,
            "detail",
            "",
        ) or (
            f"runtime state is "
            f"'{runtime_status.state}'"
        )

    # ------------------------------------------------------------
    # Provision/rebuild the controlled runtime.
    # ------------------------------------------------------------

    provision = provision_runtime(
        requirement,
        runtime_root=runtime_root,
    )

    runtime_environment = provision.environment_path
    runtime_provisioned = bool(provision.ok)

    if not provision.ok:
        detail = (
            "; ".join(provision.errors or [])
            or provision.detail
            or runtime_error
            or "runtime provisioning failed"
        )

        # Preserve the original resolution but expose the real
        # runtime failure in its errors.
        try:
            if hasattr(resolution, "errors"):
                resolution.errors = list(
                    _resolution_errors(resolution)
                ) + [f"runtime provisioning failed: {detail}"]
        except Exception:
            pass

        return RuntimeResolution(
            resolution=resolution,
            runtime_required=requirement,
            runtime_id=requirement.runtime_id,
            runtime_status=runtime_status,
            runtime_provisioned=False,
            runtime_environment=runtime_environment,
            runtime_error=detail,
        )

    # ------------------------------------------------------------
    # Runtime exists. Run the ORIGINAL resolver inside it.
    # ------------------------------------------------------------

    try:
        runtime_result = _run_in_runtime(
            provision.environment_path,
            req,
            settings,
            factory_file,
            repo_root=repo_root,
        )

    except Exception as exc:
        detail = str(exc)

        try:
            if hasattr(resolution, "errors"):
                resolution.errors = list(
                    _resolution_errors(resolution)
                ) + [
                    (
                        "controlled runtime was provisioned, "
                        "but runtime resolver failed: "
                        f"{detail}"
                    )
                ]
        except Exception:
            pass

        return RuntimeResolution(
            resolution=resolution,
            runtime_required=requirement,
            runtime_id=requirement.runtime_id,
            runtime_status=runtime_status,
            runtime_provisioned=True,
            runtime_environment=runtime_environment,
            runtime_error=detail,
        )

    return RuntimeResolution(
        resolution=runtime_result,
        runtime_required=requirement,
            runtime_id=requirement.runtime_id,
        runtime_status=runtime_status,
        runtime_provisioned=True,
        runtime_environment=runtime_environment,
    )
