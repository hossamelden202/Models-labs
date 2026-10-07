from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from uuid import uuid4
from fastapi.responses import JSONResponse

from modellab.loading import (
    ModelRequest,
    ResolveSettings,
    inspect_artifact,
    resolve_model,
)
from modellab.runtimes.bridge import resolve_model_with_runtime
from modellab.server.errors import ApiError
from modellab.server.workspace import Workspace

router = APIRouter(tags=["models"])


class ResolveBody(ModelRequest):
    artifact_id: str | None = None


def _workspace(request: Any) -> Workspace:
    workspace = getattr(request.app.state, "workspace", None)
    if workspace is None:
        raise ApiError(
            status=500,
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
    record = workspace.get_staged(artifact_id)

    if not record:
        raise ApiError(
            status=404,
            message=f"Model artifact '{artifact_id}' was not found.",
        )

    # Staged artifact records store the actual model path.
    artifact_path = record.get("path")

    if not artifact_path:
        raise ApiError(
            status=500,
            message=f"Model artifact '{artifact_id}' has no stored path.",
        )

    path = Path(artifact_path)

    if not path.is_file():
        raise ApiError(
            status=404,
            message=f"Model artifact '{artifact_id}' was not found.",
        )

    return path


@router.get("/architectures")
def list_architectures() -> JSONResponse:
    from modellab.loading.architectures import list_architectures as _list

    return JSONResponse(
        content=_jsonable(_list()),
    )


@router.post("/model-artifacts")
async def upload_model_artifact(
    request: Request,
    artifact: Annotated[UploadFile | None, File()] = None,
    factory_file: Annotated[UploadFile | None, File()] = None,
    local_path: Annotated[str | None, Form()] = None,
    artifact_id: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    workspace = _workspace(request)
    settings = _settings(request)

    if artifact is None and factory_file is None and not local_path:
        raise ApiError(
            status=422,
            message=(
                "Provide an uploaded artifact, factory file, "
                "or local_path."
            ),
        )

    public_id = (
        artifact_id.strip()
        if artifact_id and artifact_id.strip()
        else None
    )

    # ---------------------------------------------------------
    # Validate artifact ID
    # ---------------------------------------------------------
    if public_id is not None:
        if (
            public_id in {".", ".."}
            or Path(public_id).name != public_id
            or not public_id
        ):
            raise ApiError(
                status=400,
                message="artifact_id must be a simple file-safe identifier.",
            )

    # ---------------------------------------------------------
    # Determine source filename
    # ---------------------------------------------------------
    if artifact is not None:
        filename = Path(
            artifact.filename or "model_artifact"
        ).name
    elif local_path:
        filename = Path(local_path).expanduser().name
    else:
        filename = Path(
            factory_file.filename or "model_factory.py"
        ).name

    if public_id is None:
        # The source filename is metadata, NOT identity.
        #
        # Multiple models/artifacts may legitimately have the same
        # filename, for example:
        #
        #   /run/a/last.pt
        #   /run/b/last.pt
        #   /run/c/last.pt
        #
        # Generate a unique staging ID when the caller does not
        # explicitly provide one.
        public_id = f"artifact_{uuid4().hex}"

    # ---------------------------------------------------------
    # One staged artifact = one directory:
    #
    # model_artifacts/
    #   ck1/
    #     artifact.json
    #     model.pt
    #
    # Factory artifacts additionally contain:
    #
    #   fk/
    #     artifact.json
    #     model.pt
    #     factory.py
    # ---------------------------------------------------------
    folder = workspace.stage_dir / public_id

    if folder.exists():
        raise ApiError(
            status=409,
            message=f"Model artifact '{public_id}' already exists.",
        )

    folder.mkdir(parents=True, exist_ok=False)

    try:
        # -----------------------------------------------------
        # Uploaded model artifact
        # -----------------------------------------------------
        if artifact is not None:
            model_filename = filename
            model_path = folder / model_filename
            data = await artifact.read()
            model_path.write_bytes(data)

            inspection = inspect_artifact(model_path)

            # Optional factory accompanying the model artifact.
            factory_path = None

            if factory_file is not None:
                factory_filename = Path(
                    factory_file.filename or "model_factory.py"
                ).name
                factory_path = folder / factory_filename
                factory_data = await factory_file.read()
                factory_path.write_bytes(factory_data)

            record = {
                "artifact_id": public_id,
                "kind": "uploaded",
                "file_name": model_filename,
                "path": str(model_path),
                "size": len(data),
            }

            if factory_path is not None:
                record["factory_file"] = str(factory_path)

            workspace.write_json(
                folder / "artifact.json",
                record,
            )

            return JSONResponse(
                status_code=201,
                content={
                    "artifact_id": public_id,
                    "artifact": {
                        k: v
                        for k, v in record.items()
                        if k != "path"
                    },
                    "inspection": _jsonable(inspection),
                },
            )

        # -----------------------------------------------------
        # Factory-only upload
        # -----------------------------------------------------
        if factory_file is not None:
            factory_filename = Path(
                factory_file.filename or "model_factory.py"
            ).name

            factory_path = folder / factory_filename
            factory_data = await factory_file.read()
            factory_path.write_bytes(factory_data)

            record = {
                "artifact_id": public_id,
                "kind": "factory",
                "file_name": factory_filename,
                "path": str(factory_path),
                "factory_file": str(factory_path),
                "size": len(factory_data),
            }

            workspace.write_json(
                folder / "artifact.json",
                record,
            )

            return JSONResponse(
                status_code=201,
                content={
                    "artifact_id": public_id,
                    "artifact": {
                        k: v
                        for k, v in record.items()
                        if k != "path"
                    },
                    "inspection": None,
                },
            )

        # -----------------------------------------------------
        # Server-local path
        # -----------------------------------------------------
        source = Path(local_path).expanduser()

        server_settings = getattr(request.app.state, "settings", None)

        if not bool(
            getattr(server_settings, "allow_local_paths", False)
        ):
            raise ApiError(
                status=403,
                message="local model paths are disabled on this server",
            )

        if not source.exists():
            raise ApiError(
                status=404,
                message=f"Local model path does not exist: {source}",
            )

        if not source.is_file():
            raise ApiError(
                status=400,
                message=f"Local model path is not a file: {source}",
            )

        # -----------------------------------------------------
        # Hugging Face local model support
        #
        # A HF model is normally a directory, not just a weights
        # file. If the supplied weights file has a sibling config.json
        # identifying a HF model, preserve the entire directory so
        # the resolver can discover the architecture automatically.
        # -----------------------------------------------------
        hf_config = source.parent / "config.json"
        is_huggingface = False

        if hf_config.is_file():
            try:
                import json

                config = json.loads(
                    hf_config.read_text(encoding="utf-8")
                )

                is_huggingface = (
                    isinstance(config, dict)
                    and (
                        bool(config.get("model_type"))
                        or bool(config.get("architectures"))
                    )
                )
            except Exception:
                is_huggingface = False

        if is_huggingface:
            import shutil

            staged_dir = folder / source.parent.name

            shutil.copytree(
                source.parent,
                staged_dir,
                dirs_exist_ok=True,
            )

            model_path = staged_dir / source.name
            data = source.read_bytes()

            inspection = inspect_artifact(model_path)

            record = {
                "artifact_id": public_id,
                "kind": "local",
                "file_name": source.name,
                "path": str(model_path),
                "size": len(data),
                "artifact_dir": str(staged_dir),
                "format": "huggingface",
            }

            workspace.write_json(
                folder / "artifact.json",
                record,
            )

            return JSONResponse(
                status_code=201,
                content={
                    "artifact_id": public_id,
                    "artifact": {
                        k: v
                        for k, v in record.items()
                        if k != "path"
                    },
                    "inspection": _jsonable(inspection),
                },
            )

        # -----------------------------------------------------
        # Normal local model file
        # -----------------------------------------------------
        model_path = folder / source.name
        data = source.read_bytes()
        model_path.write_bytes(data)

        inspection = inspect_artifact(model_path)

        record = {
            "artifact_id": public_id,
            "kind": "local",
            "file_name": source.name,
            "path": str(model_path),
            "size": len(data),
        }

        workspace.write_json(
            folder / "artifact.json",
            record,
        )

        return JSONResponse(
            status_code=201,
            content={
                "artifact_id": public_id,
                "artifact": {
                    k: v
                    for k, v in record.items()
                    if k != "path"
                },
                "inspection": _jsonable(inspection),
            },
        )

    except BaseException:
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
        raise


@router.get("/model-artifacts")
def list_model_artifacts(request: Request) -> JSONResponse:
    workspace = _workspace(request)

    records = workspace.list_records(
        workspace.stage_dir,
        "artifact",
    )

    # Never expose filesystem paths in the public artifact listing.
    public_records = []

    for record in records:
        public_records.append(
            {
                k: v
                for k, v in record.items()
                if k != "path"
            }
        )

    return JSONResponse(
        content=_jsonable(public_records),
    )


@router.get("/model-artifacts/{artifact_id}")
def inspect_model_artifact(
    request: Request,
    artifact_id: str,
) -> JSONResponse:
    workspace = _workspace(request)
    path = _artifact_path(workspace, artifact_id)

    result = inspect_artifact(path)

    return JSONResponse(
        content=_jsonable(result),
    )


@router.delete("/model-artifacts/{artifact_id}")
def delete_model_artifact(
    request: Request,
    artifact_id: str,
) -> JSONResponse:
    workspace = _workspace(request)

    # Validate that the record exists.
    workspace.get_staged(artifact_id)

    folder = workspace.stage_dir / artifact_id

    if not folder.is_dir():
        raise ApiError(
            status=404,
            message=f"Model artifact '{artifact_id}' was not found.",
        )

    import shutil
    shutil.rmtree(folder)

    return JSONResponse(
        content={
            "deleted": True,
            "artifact_id": artifact_id,
        }
    )


@router.post("/models/resolve")
def resolve_model_endpoint(
    request: Request,
    body: ResolveBody,
) -> JSONResponse:
    workspace = _workspace(request)

    factory_file = None

    if body.artifact_id:
        artifact_path = _artifact_path(
            workspace,
            body.artifact_id,
        )

        body.artifact = str(artifact_path)

        record = workspace.get_staged(body.artifact_id)

        stored_factory = record.get("factory_file")

        if stored_factory:
            factory_file = stored_factory

    settings = _settings(request)

    runtime_resolution = resolve_model_with_runtime(
        body,
        settings=settings,
        factory_file=factory_file,
        repo_root=Path(__file__).resolve().parents[2],
    )
    resolution = runtime_resolution.resolution

    response = _jsonable(resolution)

    if isinstance(response, dict):
        runtime = _jsonable(runtime_resolution)

        response["runtime"] = {
            "required": runtime.get("runtime_required"),
            "status": runtime.get("runtime_status"),
            "provisioned": bool(runtime.get("runtime_provisioned")),
            "environment": runtime.get("runtime_environment"),
            "error": runtime.get("runtime_error"),
        }

    return JSONResponse(
        content=response,
    )


@router.post("/models/from-artifact")
def model_from_artifact(
    request: Request,
    body: ResolveBody,
) -> JSONResponse:
    from datetime import datetime, timezone
    import shutil

    workspace = _workspace(request)
    settings = _settings(request)

    factory_file = None

    if not body.artifact and body.artifact_id:
        body.artifact = str(
            _artifact_path(
                workspace,
                body.artifact_id,
            )
        )

        record = workspace.get_staged(body.artifact_id)
        stored_factory = record.get("factory_file")

        if stored_factory:
            factory_file = stored_factory

    if not body.artifact:
        raise ApiError(
            status=400,
            message="An artifact or artifact_id is required.",
        )

    runtime_resolution = resolve_model_with_runtime(
        body,
        settings=settings,
        factory_file=factory_file,
        repo_root=Path(__file__).resolve().parents[2],
    )
    resolution = runtime_resolution.resolution

    # -------------------------------------------------------------
    # Resolution errors are returned unchanged to the caller.
    # -------------------------------------------------------------
    if getattr(resolution, "status", None) == "invalid":
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "resolution": _jsonable(resolution),
                }
            },
        )

    if getattr(resolution, "status", None) != "ready":
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "resolution": _jsonable(resolution),
                }
            },
        )

    model_id = body.model_id

    if not model_id:
        raise ApiError(
            status=422,
            message="model_id is required.",
        )

    # -------------------------------------------------------------
    # Prevent overwriting an existing model.
    # -------------------------------------------------------------
    try:
        workspace.get_model(model_id)
    except ApiError as exc:
        if getattr(exc, "status", None) != 404:
            raise
    else:
        raise ApiError(
            status=409,
            message=f"Model '{model_id}' already exists.",
        )

    model_dir = workspace.models_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=False)

    try:
        # ---------------------------------------------------------
        # Materialize model weights into models/<model_id>/.
        # ---------------------------------------------------------
        source_path = Path(body.artifact)

        if not source_path.is_file():
            raise ApiError(
                status=404,
                message=f"Resolved model artifact does not exist: {source_path}",
            )

        model_path = model_dir / source_path.name
        shutil.copy2(source_path, model_path)

        # ---------------------------------------------------------
        # Materialize factory sidecar when used.
        # ---------------------------------------------------------
        materialized_factory = None

        if factory_file:
            factory_source = Path(factory_file)

            if not factory_source.is_file():
                raise ApiError(
                    status=404,
                    message=f"Resolved factory file does not exist: {factory_source}",
                )

            factory_destination = model_dir / factory_source.name
            shutil.copy2(factory_source, factory_destination)
            materialized_factory = factory_destination

        # ---------------------------------------------------------
        # Build the self-contained model specification.
        # ---------------------------------------------------------
        spec = _jsonable(resolution.spec)

        # Preserve the provider selected by the resolver at the registration boundary.
        # Local Hugging Face artifacts are identified by config.json beside the
        # checkpoint. Existing registered architectures are unaffected.
        if isinstance(spec, dict):
            hf_config_path = Path(model_path).parent / "config.json"

            if (
                resolution.route == "provider_architecture"
                and hf_config_path.is_file()
            ):
                try:
                    hf_config = json.loads(hf_config_path.read_text())
                except Exception:
                    hf_config = {}

                if (
                    isinstance(hf_config, dict)
                    and (
                        hf_config.get("model_type")
                        or hf_config.get("architectures")
                    )
                ):
                    spec["provider_id"] = "huggingface"

        if isinstance(spec, dict):
            spec["path"] = str(model_path)

        if materialized_factory is not None:
            factory_ref = spec.get("factory")

            if factory_ref:
                factory_ref = str(factory_ref)

                if ":" in factory_ref:
                    callable_name = factory_ref.rsplit(":", 1)[1]
                    spec["factory"] = (
                        f"{materialized_factory}:{callable_name}"
                    )
                else:
                    spec["factory"] = str(materialized_factory)

        # ---------------------------------------------------------
        # Preserve resolver metadata required by downstream systems.
        #
        # In particular, the evaluation worker expects preprocess
        # to be present on the persisted model record.
        # ---------------------------------------------------------
        preprocess = _jsonable(
            getattr(resolution, "preprocess", {})
        )

        created_at = datetime.now(timezone.utc).isoformat()

        model_record = {
            "model_id": model_id,
            "source": "resolved",
            "created_at": created_at,
            "spec": spec,
            "preprocess": preprocess,
            "resolution": _jsonable(resolution),
        }

        workspace.write_json(
            model_dir / "model.json",
            model_record,
        )

    except BaseException:
        shutil.rmtree(model_dir, ignore_errors=True)
        raise

    return JSONResponse(
        status_code=201,
        content=_jsonable(model_record),
    )



@router.post("/models/inspect")
def inspect_model_path(
    request: Request,
    body: ResolveBody,
) -> JSONResponse:
    if not body.artifact:
        raise ApiError(
            status=400,
            message="An artifact path is required.",
        )

    path = Path(body.artifact).expanduser()

    if not path.exists():
        raise ApiError(
            status=404,
            message=f"Artifact does not exist: {path}",
        )

    result = inspect_artifact(path)

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
