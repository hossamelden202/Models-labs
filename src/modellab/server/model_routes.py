from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from modellab.loading import (
    ModelRequest,
    ResolveSettings,
    inspect_artifact,
    resolve_model,
)
from modellab.server.errors import ApiError
from modellab.server.workspace import Workspace

router = APIRouter(tags=["models"])


class ResolveBody(ModelRequest):
    artifact_id: str | None = None


def _workspace(request: Any) -> Workspace:
    workspace = getattr(request.app.state, "workspace", None)
    if workspace is None:
        raise ApiError(
            status_code=500,
            code="workspace_not_configured",
            message="Server workspace is not configured.",
        )
    return workspace


def _settings(request: Any) -> ResolveSettings:
    settings = getattr(request.app.state, "settings", None)

    if settings is None:
        return ResolveSettings()

    return ResolveSettings(
        allow_custom_code=bool(
            getattr(settings, "allow_custom_code", False)
        ),
        allow_pickle=bool(
            getattr(settings, "allow_pickle", False)
        ),
    )


def _jsonable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    if hasattr(value, "__dict__"):
        return {
            key: _jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }

    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]

    return value


def _artifact_path(workspace: Workspace, artifact_id: str) -> Path:
    path = workspace.get_staged(artifact_id)

    if path is None:
        raise ApiError(
            status_code=404,
            code="artifact_not_found",
            message=f"Model artifact '{artifact_id}' was not found.",
        )

    return path


@router.get("/architectures")
def list_architectures() -> JSONResponse:
    from modellab.loading.architectures import list_architectures as _list

    return JSONResponse(
        content={
            "architectures": _jsonable(_list()),
        }
    )


@router.post("/model-artifacts")
async def upload_model_artifact(
    request: Any,
    artifact: Annotated[UploadFile | None, File()] = None,
    factory_file: Annotated[UploadFile | None, File()] = None,
    local_path: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    workspace = _workspace(request)

    if artifact is None and factory_file is None and not local_path:
        raise ApiError(
            status_code=400,
            code="missing_artifact",
            message=(
                "Provide an uploaded artifact, factory file, "
                "or local_path."
            ),
        )

    if artifact is not None:
        filename = Path(artifact.filename or "model_artifact").name
        destination = workspace.stage_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)

        data = await artifact.read()
        destination.write_bytes(data)

        return JSONResponse(
            status_code=201,
            content={
                "artifact_id": filename,
                "path": str(destination),
                "size": len(data),
            },
        )

    if factory_file is not None:
        filename = Path(
            factory_file.filename or "model_factory.py"
        ).name

        destination = workspace.stage_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)

        data = await factory_file.read()
        destination.write_bytes(data)

        return JSONResponse(
            status_code=201,
            content={
                "artifact_id": filename,
                "path": str(destination),
                "size": len(data),
            },
        )

    source = Path(local_path).expanduser()

    if not source.exists():
        raise ApiError(
            status_code=404,
            code="local_path_not_found",
            message=f"Local model path does not exist: {source}",
        )

    if not source.is_file():
        raise ApiError(
            status_code=400,
            code="local_path_not_file",
            message=f"Local model path is not a file: {source}",
        )

    return JSONResponse(
        status_code=201,
        content={
            "artifact_id": source.name,
            "path": str(source),
            "size": source.stat().st_size,
        },
    )


@router.get("/model-artifacts")
def list_model_artifacts(request: Any) -> JSONResponse:
    workspace = _workspace(request)

    root = workspace.stage_dir
    root.mkdir(parents=True, exist_ok=True)

    artifacts = []

    for path in sorted(root.iterdir()):
        if path.is_file():
            artifacts.append(
                {
                    "artifact_id": path.name,
                    "path": str(path),
                    "size": path.stat().st_size,
                }
            )

    return JSONResponse(
        content={
            "artifacts": artifacts,
        }
    )


@router.get("/model-artifacts/{artifact_id}")
def inspect_model_artifact(
    request: Any,
    artifact_id: str,
) -> JSONResponse:
    workspace = _workspace(request)
    path = _artifact_path(workspace, artifact_id)

    result = inspect_artifact(
        path,
        allow_pickle=bool(
            getattr(
                getattr(request.app.state, "settings", None),
                "allow_pickle",
                False,
            )
        ),
    )

    return JSONResponse(
        content=_jsonable(result),
    )


@router.delete("/model-artifacts/{artifact_id}")
def delete_model_artifact(
    request: Any,
    artifact_id: str,
) -> JSONResponse:
    workspace = _workspace(request)
    path = _artifact_path(workspace, artifact_id)

    if path.is_file():
        path.unlink()

    return JSONResponse(
        content={
            "deleted": True,
            "artifact_id": artifact_id,
        }
    )


@router.post("/models/resolve")
def resolve_model_endpoint(
    request: Any,
    body: ResolveBody,
) -> JSONResponse:
    workspace = _workspace(request)

    if body.artifact_id:
        artifact_path = _artifact_path(
            workspace,
            body.artifact_id,
        )
        body.artifact = str(artifact_path)

    settings = _settings(request)

    resolution = resolve_model(
        body,
        settings=settings,
    )

    return JSONResponse(
        content=_jsonable(resolution),
    )


@router.post("/models/from-artifact")
def model_from_artifact(
    request: Any,
    body: ResolveBody,
) -> JSONResponse:
    workspace = _workspace(request)

    if not body.artifact and body.artifact_id:
        body.artifact = str(
            _artifact_path(
                workspace,
                body.artifact_id,
            )
        )

    if not body.artifact:
        raise ApiError(
            status_code=400,
            code="missing_artifact",
            message="An artifact or artifact_id is required.",
        )

    settings = _settings(request)

    resolution = resolve_model(
        body,
        settings=settings,
    )

    return JSONResponse(
        content=_jsonable(resolution),
    )


@router.post("/models/inspect")
def inspect_model_path(
    request: Any,
    body: ResolveBody,
) -> JSONResponse:
    if not body.artifact:
        raise ApiError(
            status_code=400,
            code="missing_artifact",
            message="An artifact path is required.",
        )

    path = Path(body.artifact).expanduser()

    if not path.exists():
        raise ApiError(
            status_code=404,
            code="artifact_not_found",
            message=f"Artifact does not exist: {path}",
        )

    result = inspect_artifact(
        path,
        allow_pickle=bool(
            getattr(
                getattr(request.app.state, "settings", None),
                "allow_pickle",
                False,
            )
        ),
    )

    return JSONResponse(
        content=_jsonable(result),
    )


__all__ = ["router", "ResolveBody"]

def add_model_routes(api, ws, settings, cache, require_local):
    """Register model-loading routes on the application's main API router.

    The route implementations are already defined in this module.  This
    function provides the registration contract expected by server.app.
    """
    api.include_router(router)
