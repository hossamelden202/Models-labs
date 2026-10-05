import importlib.metadata
import json
import platform
import shutil
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

import modellab
from modellab.core.errors import ArtifactError, ConfigurationError, DatasetError, ModelLabError
from modellab.core.results import ArtifactStore
from modellab.server import services
from modellab.server.errors import ApiError
from modellab.server.jobs import JobManager
from modellab.server.settings import ServerSettings
from modellab.server.workspace import Workspace, check_id, extract_zip, find_root, slug, utcnow
from modellab.utils import get_logger, setup_logging

log = get_logger("server")
SINGULAR = {
    "evaluations": "evaluation", "audits": "audit", "failure_analyses": "analysis",
    "experiments": "experiment family", "hypotheses": "hypothesis run",
    "investigations": "investigation", "repairs": "repair",
}


class _Req(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class EvaluateRequest(_Req):
    model_id: str
    dataset_id: str
    subpath: str | None = None
    evaluation_id: str | None = None
    batch_size: int = Field(32, ge=1)
    num_workers: int = Field(0, ge=0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    seed: int = Field(42, ge=0, lt=2**32)


class AuditRequest(_Req):
    dataset_id: str
    subpath: str | None = None
    audit_id: str | None = None
    config: dict = Field(default_factory=dict)


class AnalysisRequest(_Req):
    evaluation_id: str
    audit_id: str | None = None
    analysis_id: str | None = None
    config: dict = Field(default_factory=dict)


class ExperimentRequest(_Req):
    family_id: str
    baseline_evaluation_id: str | None = None
    specs: list[dict] = Field(default_factory=list)
    matrices: list[dict] = Field(default_factory=list)
    model_id: str | None = None
    dataset_id: str | None = None
    subpath: str | None = None
    device: Literal["auto", "cpu", "cuda"] = "auto"
    force: bool = False
    stop_on_failure: bool = False
    alpha: float = Field(0.05, gt=0, lt=1)


class RegisterDataset(_Req):
    dataset_id: str | None = None
    path: str
    kind: Literal["folder", "csv"] = "folder"
    csv_path: str | None = None
    path_column: str = "path"
    label_column: str = "label"


class RegisterModel(_Req):
    model_id: str | None = None
    spec: dict
    preprocess: dict = Field(default_factory=dict)
    validate_load: bool = True


def status_for(exc: Exception) -> int:
    if isinstance(exc, (ConfigurationError, DatasetError)):
        return 422
    if isinstance(exc, ArtifactError):
        return 409
    return 400


def save_upload(upload: UploadFile, dest: Path, limit: int) -> int:
    written = 0
    with open(dest, "wb") as out:
        while chunk := upload.file.read(1 << 20):
            written += len(chunk)
            if written > limit:
                raise ApiError(413, "the upload exceeds the configured size limit")
            out.write(chunk)
    return written


def parse_json(text: str, what: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ApiError(422, f"{what} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ApiError(422, f"{what} must be a JSON object")
    return value


def table_page(path: Path, offset: int, limit: int, equals: dict | None = None) -> dict:
    import pandas as pd

    if not path.is_file():
        raise ApiError(404, f"{path.name} not found")
    frame = pd.read_parquet(path)
    for column, value in (equals or {}).items():
        if column not in frame.columns:
            raise ApiError(422, f"unknown column '{column}'")
        frame = frame[frame[column] == value]
    page = frame.iloc[offset : offset + limit]
    return {
        "total": int(len(frame)), "offset": offset, "limit": limit,
        "columns": list(frame.columns), "rows": json.loads(page.to_json(orient="records")),
    }


def create_app(settings: ServerSettings) -> FastAPI:
    setup_logging("INFO")
    ws = Workspace(settings.workspace)
    jobs = JobManager(ws, settings.max_workers)
    cache = services.ModelCache()
    store = ArtifactStore(ws.artifacts_dir)

    @asynccontextmanager
    async def lifespan(app):
        yield
        jobs.shutdown()

    app = FastAPI(title="ModelLab developer API", version=modellab.__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.workspace = ws
    app.add_middleware(
        CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["*"], allow_headers=["*"]
    )

    @app.exception_handler(ApiError)
    async def _api_error(request, exc):
        return JSONResponse({"detail": exc.message}, status_code=exc.status)

    @app.exception_handler(ModelLabError)
    async def _modellab_error(request, exc):
        return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=status_for(exc))

    @app.exception_handler(ValidationError)
    async def _validation_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    def auth(request: Request):
        if settings.token and request.headers.get("authorization") != f"Bearer {settings.token}":
            raise ApiError(401, "missing or invalid bearer token")

    api = APIRouter(dependencies=[Depends(auth)])

    def require_local():
        if not settings.allow_local_paths:
            raise ApiError(403, "registering server paths is disabled on this server")

    def enqueue(kind, params, fn, wait, timeout):
        job = jobs.submit(kind, params, fn)
        if wait:
            job = jobs.wait(job["job_id"], timeout)
        return JSONResponse(job, status_code=200 if wait else 202)

    def art(kind: str, id_: str, *parts: str) -> Path:
        check_id(id_, "id")
        base = ws.artifacts_dir / kind / id_
        if not base.is_dir():
            raise ApiError(404, f"{SINGULAR[kind]} '{id_}' not found")
        return base.joinpath(*parts)

    def read(path: Path):
        if not path.is_file():
            raise ApiError(404, f"{path.name} not found")
        return json.loads(path.read_text())

    @app.get("/health")
    def health():
        return {"status": "ok", "version": modellab.__version__}

    @api.get("/system")
    def system():
        info = {
            "modellab": modellab.__version__, "python": sys.version.split()[0],
            "platform": platform.platform(), "workspace": str(ws.root),
            "max_workers": settings.max_workers, "allow_local_paths": settings.allow_local_paths,
            "token_required": bool(settings.token), "packages": {},
        }
        for pkg in ("fastapi", "pydantic", "numpy", "pandas", "scikit-learn", "scipy", "pillow"):
            try:
                info["packages"][pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                info["packages"][pkg] = None
        try:
            import torch

            info["torch"] = {
                "version": torch.__version__, "cuda": torch.cuda.is_available(),
                "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            }
        except ImportError:
            info["torch"] = None
        return info

    def finish_model(model_id, spec_d, pre_d, validate_load, folder, source, files):
        from modellab.evaluation.preprocessing import PreprocessConfig
        from modellab.evaluation.torch_model import TorchImageClassifier, TorchModelSpec

        try:
            spec = TorchModelSpec.model_validate({**spec_d, "model_id": model_id})
            pre = PreprocessConfig.model_validate(pre_d)
        except ValidationError as exc:
            raise ApiError(422, f"invalid model definition: {exc}") from exc
        if spec.path is not None and not Path(spec.path).is_file():
            raise ApiError(422, f"weights file not found: {spec.path}")
        check = {"loaded": None}
        if validate_load:
            try:
                meta = TorchImageClassifier(spec, device="cpu").metadata()
            except ModelLabError as exc:
                raise ApiError(422, f"model failed to load: {type(exc).__name__}: {exc}") from exc
            check = {"loaded": True, "metadata": meta.model_dump(mode="json")}
        record = {
            "model_id": model_id, "source": source, "spec": spec.model_dump(mode="json"),
            "preprocess": pre.model_dump(mode="json"), "files": files,
            "created_at": utcnow(), "check": check,
        }
        ws.write_json(folder / "model.json", record)
        return record

    @api.post("/models", status_code=201)
    def upload_model(
        model_id: str | None = Form(None),
        spec: str = Form(...),
        preprocess: str = Form("{}"),
        validate_load: bool = Form(True),
        weights: Annotated[UploadFile | None, File()] = None,
        factory_file: Annotated[UploadFile | None, File()] = None,
    ):
        spec_d, pre_d = parse_json(spec, "spec"), parse_json(preprocess, "preprocess")
        model_id = check_id(model_id or spec_d.get("model_id") or slug("model"), "model id")
        folder = ws.models_dir / model_id
        if folder.exists():
            raise ApiError(409, f"model '{model_id}' already exists")
        source, files = spec_d.get("source"), []
        folder.mkdir(parents=True)
        try:
            if weights is not None:
                if source not in ("torchscript", "module", "state_dict"):
                    raise ApiError(422, f"source '{source}' does not take a weights file")
                name = "weights" + (Path(weights.filename or "").suffix or ".pt")
                save_upload(weights, folder / name, settings.max_upload_bytes)
                spec_d["path"] = str(folder / name)
                files.append(name)
            elif spec_d.get("path"):
                require_local()
            if factory_file is not None:
                callable_name = spec_d.get("factory")
                if source not in ("state_dict", "factory"):
                    raise ApiError(422, f"source '{source}' does not take a factory file")
                if not callable_name or ":" in callable_name:
                    raise ApiError(422, "with factory_file, spec.factory must be the callable name only")
                save_upload(factory_file, folder / "factory.py", settings.max_upload_bytes)
                spec_d["factory"] = f"{folder / 'factory.py'}:{callable_name}"
                files.append("factory.py")
            return finish_model(model_id, spec_d, pre_d, validate_load, folder, "upload", files)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise

    @api.post("/models/register", status_code=201)
    def register_model(req: RegisterModel):
        require_local()
        model_id = check_id(req.model_id or req.spec.get("model_id") or slug("model"), "model id")
        folder = ws.models_dir / model_id
        if folder.exists():
            raise ApiError(409, f"model '{model_id}' already exists")
        folder.mkdir(parents=True)
        try:
            return finish_model(model_id, dict(req.spec), req.preprocess, req.validate_load, folder, "local", [])
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise

    @api.get("/models")
    def list_models():
        return ws.list_records(ws.models_dir, "model")

    @api.get("/models/{model_id}")
    def get_model(model_id: str):
        return ws.get_model(model_id)

    @api.delete("/models/{model_id}")
    def delete_model(model_id: str):
        ws.delete(ws.models_dir, model_id, "model")
        cache.drop(model_id)
        return {"deleted": model_id}

    def finish_dataset(dataset_id, folder, source, kind, root, csv, path_column, label_column):
        rec = {
            "dataset_id": dataset_id, "source": source, "kind": kind, "root": str(root),
            "csv": str(csv) if csv else None, "path_column": path_column,
            "label_column": label_column, "created_at": utcnow(),
        }
        summary = services.dataset_summary(ws, rec)
        if summary["num_samples"] == 0:
            raise ApiError(422, "no images found in the dataset")
        rec["summary"] = summary
        ws.write_json(folder / "dataset.json", rec)
        return rec

    @api.post("/datasets", status_code=201)
    def upload_dataset(
        archive: Annotated[UploadFile, File(...)],
        dataset_id: str | None = Form(None),
        path_column: str = Form("path"),
        label_column: str = Form("label"),
        labels_csv: Annotated[UploadFile | None, File()] = None,
    ):
        dataset_id = check_id(dataset_id or slug(Path(archive.filename or "dataset").stem), "dataset id")
        folder = ws.datasets_dir / dataset_id
        if folder.exists():
            raise ApiError(409, f"dataset '{dataset_id}' already exists")
        folder.mkdir(parents=True)
        tmp = ws.tmp_dir / f"{uuid.uuid4().hex}.zip"
        try:
            save_upload(archive, tmp, settings.max_upload_bytes)
            data = folder / "data"
            data.mkdir()
            extract_zip(tmp, data, settings.max_upload_bytes)
            csv = None
            if labels_csv is not None:
                csv = folder / "labels.csv"
                save_upload(labels_csv, csv, settings.max_upload_bytes)
            kind = "csv" if csv else "folder"
            return finish_dataset(dataset_id, folder, "upload", kind, find_root(data), csv, path_column, label_column)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        finally:
            tmp.unlink(missing_ok=True)

    @api.post("/datasets/register", status_code=201)
    def register_dataset(req: RegisterDataset):
        require_local()
        path = Path(req.path).expanduser()
        if not path.is_dir():
            raise ApiError(422, f"directory not found on the server: {req.path}")
        csv = None
        if req.kind == "csv":
            csv = Path(req.csv_path or "").expanduser()
            if not csv.is_file():
                raise ApiError(422, f"csv file not found on the server: {req.csv_path}")
        dataset_id = check_id(req.dataset_id or slug(path.name), "dataset id")
        folder = ws.datasets_dir / dataset_id
        if folder.exists():
            raise ApiError(409, f"dataset '{dataset_id}' already exists")
        folder.mkdir(parents=True)
        try:
            return finish_dataset(
                dataset_id, folder, "local", req.kind, path.resolve(), csv, req.path_column, req.label_column
            )
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise

    @api.get("/datasets")
    def list_datasets():
        return ws.list_records(ws.datasets_dir, "dataset")

    @api.get("/datasets/{dataset_id}")
    def get_dataset(dataset_id: str):
        return ws.get_dataset(dataset_id)

    @api.get("/datasets/{dataset_id}/summary")
    def dataset_summary(dataset_id: str, subpath: str | None = None):
        return services.dataset_summary(ws, ws.get_dataset(dataset_id), subpath)

    @api.delete("/datasets/{dataset_id}")
    def delete_dataset(dataset_id: str):
        ws.delete(ws.datasets_dir, dataset_id, "dataset")
        return {"deleted": dataset_id}

    @api.get("/jobs")
    def list_jobs(kind: str | None = None, limit: int = Query(100, ge=1, le=1000)):
        return jobs.list(kind, limit)

    @api.get("/jobs/{job_id}")
    def get_job(job_id: str):
        job_response = jobs.get(job_id)

        # ========================================================
        # Attach complete persisted failure-analysis report.
        #
        # jobs.get() may return a model/dataclass/object rather
        # than a plain dict, so normalize it before enrichment.
        # ========================================================

        try:
            # Normalize the Jobs store result into a mutable dict.
            if isinstance(job_response, dict):
                response_dict = dict(job_response)

            elif hasattr(job_response, "model_dump"):
                response_dict = job_response.model_dump(
                    mode="json"
                )

            elif hasattr(job_response, "dict"):
                response_dict = job_response.dict()

            elif hasattr(job_response, "__dict__"):
                response_dict = dict(vars(job_response))

            else:
                response_dict = None

            if isinstance(response_dict, dict):

                # ------------------------------------------------
                # Find analysis_id recursively.
                # ------------------------------------------------
                def _find_analysis_id(value):
                    if isinstance(value, dict):
                        for key in (
                            "analysis_id",
                            "analysisId",
                        ):
                            candidate = value.get(key)

                            if (
                                isinstance(candidate, str)
                                and candidate.strip()
                            ):
                                return candidate.strip()

                        for value_item in value.values():
                            found = _find_analysis_id(value_item)
                            if found:
                                return found

                    elif isinstance(value, list):
                        for item in value:
                            found = _find_analysis_id(item)
                            if found:
                                return found

                    return None

                analysis_id = _find_analysis_id(response_dict)

                # ------------------------------------------------
                # Load the persisted full failure-analysis report.
                #
                # Do this whenever an analysis_id is present.
                # We do NOT depend on a particular status string.
                # ------------------------------------------------
                if analysis_id:

                    from pathlib import Path
                    import json

                    report_path = (
                        Path("/kaggle/working/modellab")
                        / "workspace"
                        / "artifacts"
                        / "failure_analyses"
                        / str(analysis_id)
                        / "report.json"
                    )

                    if report_path.is_file():

                        with report_path.open(
                            "r",
                            encoding="utf-8",
                        ) as fh:
                            report = json.load(fh)

                        response_dict["analysis_id"] = analysis_id
                        response_dict["analysis_report"] = report
                        response_dict["report"] = report

                # Always return the normalized response so the
                # enriched fields actually reach the API client.
                job_response = response_dict

        except Exception:
            # Report enrichment must never break the normal Jobs API.
            # Fall back to the original job response if enrichment
            # encounters an unexpected problem.
            pass

        return job_response

    @api.get("/jobs/{job_id}/log")
    def job_log(job_id: str, tail: int | None = Query(None, ge=1)):
        path = jobs.log_path(job_id)
        text = path.read_text() if path.is_file() else ""
        if tail:
            text = "\n".join(text.splitlines()[-tail:])
        return PlainTextResponse(text)

    @api.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        return jobs.cancel(job_id)

    @api.post("/evaluations")
    def start_evaluation(req: EvaluateRequest, wait: bool = False, timeout: float = Query(600, gt=0, le=86400)):
        ws.get_model(req.model_id)
        rec = ws.get_dataset(req.dataset_id)
        if req.subpath and rec["kind"] == "folder":
            services.dataset_root(ws, rec, req.subpath)
        if req.evaluation_id:
            check_id(req.evaluation_id, "evaluation id")
            if (ws.artifacts_dir / "evaluations" / req.evaluation_id).exists():
                raise ApiError(409, f"evaluation '{req.evaluation_id}' already exists")
        params = req.model_dump()
        return enqueue("evaluation", params, lambda: services.evaluate(ws, cache, store, params), wait, timeout)

    @api.post("/audits")
    def start_audit(req: AuditRequest, wait: bool = False, timeout: float = Query(600, gt=0, le=86400)):
        from modellab.audit import AuditConfig

        rec = ws.get_dataset(req.dataset_id)
        AuditConfig.model_validate(req.config)
        if req.subpath and rec["kind"] == "folder":
            services.dataset_root(ws, rec, req.subpath)
        if req.audit_id:
            check_id(req.audit_id, "audit id")
            if (ws.artifacts_dir / "audits" / req.audit_id).exists():
                raise ApiError(409, f"audit '{req.audit_id}' already exists")
        params = req.model_dump()
        return enqueue("audit", params, lambda: services.audit(ws, store, params), wait, timeout)

    @api.post("/analyses")
    def start_analysis(req: AnalysisRequest, wait: bool = False, timeout: float = Query(600, gt=0, le=86400)):
        from modellab.analysis import FailureConfig

        art("evaluations", req.evaluation_id)
        if req.audit_id:
            art("audits", req.audit_id)
        FailureConfig.model_validate(req.config)
        if req.analysis_id:
            check_id(req.analysis_id, "analysis id")
            if (ws.artifacts_dir / "failure_analyses" / req.analysis_id).exists():
                raise ApiError(409, f"analysis '{req.analysis_id}' already exists")
        params = req.model_dump()
        return enqueue("analysis", params, lambda: services.analyze(ws, store, params), wait, timeout)

    @api.post("/experiments")
    def start_experiments(req: ExperimentRequest, wait: bool = False, timeout: float = Query(600, gt=0, le=86400)):
        from modellab.experiments import expand_matrix, parse_spec

        check_id(req.family_id, "family id")
        specs = [parse_spec(s) for s in req.specs]
        for matrix in req.matrices:
            specs.extend(expand_matrix(matrix))
        if not specs:
            raise ApiError(422, "give at least one spec or matrix")
        if req.baseline_evaluation_id:
            art("evaluations", req.baseline_evaluation_id)
        elif not (ws.artifacts_dir / "experiments" / req.family_id / "baseline.json").is_file():
            raise ApiError(422, f"family '{req.family_id}' has no baseline, give baseline_evaluation_id")
        if any(s.intervention.kind != "inference" for s in specs):
            if not req.model_id or not req.dataset_id:
                raise ApiError(422, "image and preprocessing interventions need model_id and dataset_id")
            ws.get_model(req.model_id)
            ws.get_dataset(req.dataset_id)
        params = req.model_dump()
        return enqueue("experiments", params, lambda: services.experiments(ws, cache, store, params), wait, timeout)

    @api.get("/evaluations")
    def list_evaluations(include_experiment_runs: bool = False):
        out = []
        for path in sorted((ws.artifacts_dir / "evaluations").glob("*/metadata.json")):
            meta = json.loads(path.read_text())
            if meta["evaluation_id"].startswith("exp_") and not include_experiment_runs:
                continue
            metrics_path = path.with_name("metrics.json")
            metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
            out.append({
                "evaluation_id": meta["evaluation_id"], "model_id": meta["model_id"],
                "dataset_id": meta["dataset_id"], "timestamp": meta["timestamp"],
                "num_samples": meta["num_samples"], "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
            })
        return out

    @api.get("/evaluations/{evaluation_id}")
    def get_evaluation(evaluation_id: str):
        return {
            "metadata": read(art("evaluations", evaluation_id, "metadata.json")),
            "metrics": read(art("evaluations", evaluation_id, "metrics.json")),
            "confusion_matrix": read(art("evaluations", evaluation_id, "confusion_matrix.json")),
        }

    @api.get("/evaluations/{evaluation_id}/predictions")
    def evaluation_predictions(
        evaluation_id: str, only_errors: bool = False, offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
    ):
        equals = {"correct": False} if only_errors else None
        return table_page(art("evaluations", evaluation_id, "predictions.parquet"), offset, limit, equals)

    @api.get("/audits")
    def list_audits():
        return [json.loads(p.read_text()) for p in sorted((ws.artifacts_dir / "audits").glob("*/metadata.json"))]

    @api.get("/audits/{audit_id}")
    def get_audit(audit_id: str):
        return read(art("audits", audit_id, "report.json"))

    @api.get("/audits/{audit_id}/summary")
    def audit_summary(audit_id: str):
        return PlainTextResponse(art("audits", audit_id, "summary.txt").read_text())

    @api.get("/audits/{audit_id}/records")
    def audit_records(
        audit_id: str, status: str | None = None, offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
    ):
        equals = {"status": status} if status else None
        return table_page(art("audits", audit_id, "records.parquet"), offset, limit, equals)

    @api.get("/analyses")
    def list_analyses():
        return [
            json.loads(p.read_text())
            for p in sorted((ws.artifacts_dir / "failure_analyses").glob("*/metadata.json"))
        ]

    @api.get("/analyses/{analysis_id}")
    def get_analysis(analysis_id: str):
        return read(art("failure_analyses", analysis_id, "report.json"))

    @api.get("/analyses/{analysis_id}/summary")
    def analysis_summary(analysis_id: str):
        return PlainTextResponse(art("failure_analyses", analysis_id, "summary.txt").read_text())

    @api.get("/analyses/{analysis_id}/slices")
    def analysis_slices(
        analysis_id: str, tested: bool | None = None, evidence: str | None = None,
        slice_type: str | None = None, offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000),
    ):
        equals = {}
        if tested is not None:
            equals["tested"] = tested
        if evidence:
            equals["evidence"] = evidence
        if slice_type:
            equals["slice_type"] = slice_type
        return table_page(art("failure_analyses", analysis_id, "slices.parquet"), offset, limit, equals)

    @api.get("/analyses/{analysis_id}/samples")
    def analysis_samples(
        analysis_id: str, failure_type: str | None = None, is_error: bool | None = None,
        offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000),
    ):
        equals = {}
        if failure_type:
            equals["failure_type"] = failure_type
        if is_error is not None:
            equals["is_error"] = is_error
        return table_page(art("failure_analyses", analysis_id, "samples.parquet"), offset, limit, equals)

    @api.get("/analyses/{analysis_id}/slice_samples")
    def analysis_slice_samples(analysis_id: str, slice_id: str):
        from modellab.analysis import load_failure_analysis

        art("failure_analyses", analysis_id)
        analysis = load_failure_analysis(store, analysis_id)
        try:
            ids = analysis.slice_sample_ids(slice_id)
        except KeyError as exc:
            raise ApiError(404, str(exc.args[0])) from exc
        return {"slice_id": slice_id, "count": len(ids), "sample_ids": ids}

    @api.get("/experiments")
    def list_families():
        out = []
        for path in sorted((ws.artifacts_dir / "experiments").glob("*/baseline.json")):
            doc = json.loads(path.read_text())
            out.append({
                "family_id": doc["family_id"], "baseline_evaluation_id": doc["evaluation_id"],
                "num_experiments": len(list(path.parent.glob("exp_*/result.json"))),
            })
        return out

    @api.get("/experiments/{family_id}")
    def get_family(family_id: str):
        base = art("experiments", family_id)
        experiments = []
        for path in sorted(base.glob("exp_*/result.json")):
            doc = json.loads(path.read_text())
            entry = {
                "experiment_id": doc["experiment_id"], "name": doc["spec"]["name"], "status": doc["status"],
                "intervention": doc["spec"]["intervention"],
            }
            if doc["status"] == "completed":
                entry["accuracy_difference"] = doc["statistics"]["affected"]["accuracy_difference"]
                entry["verdict"] = doc["verdict"]["label"]
            else:
                entry["error"] = doc["error"]
            experiments.append(entry)
        report_path = base / "family_report.json"
        return {
            "baseline": read(base / "baseline.json"),
            "family_report": json.loads(report_path.read_text()) if report_path.is_file() else None,
            "experiments": experiments,
        }

    @api.get("/experiments/{family_id}/{experiment_id}")
    def get_experiment(family_id: str, experiment_id: str):
        base = art("experiments", family_id)
        check_id(experiment_id, "experiment id")
        return read(base / experiment_id / "result.json")

    @api.get("/experiments/{family_id}/{experiment_id}/paired")
    def experiment_paired(
        family_id: str, experiment_id: str, changed_only: bool = False,
        offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000),
    ):
        base = art("experiments", family_id)
        check_id(experiment_id, "experiment id")
        equals = {"changed": True} if changed_only else None
        return table_page(base / experiment_id / "paired.parquet", offset, limit, equals)

    @api.get("/files")
    def files(path: str = Query(...), download: bool = False):
        text = path.replace("\\", "/")
        if text.startswith("/") or ".." in text.split("/"):
            raise ApiError(400, "path must be relative to the artifacts directory")
        target = (ws.artifacts_dir / text).resolve()
        if not target.is_relative_to(ws.artifacts_dir) or not target.exists():
            raise ApiError(404, f"'{path}' not found")
        if target.is_dir():
            entries = [
                {"name": p.name, "is_dir": p.is_dir(), "size": p.stat().st_size if p.is_file() else None}
                for p in sorted(target.iterdir())
            ]
            return {"path": path, "entries": entries}
        if download:
            return FileResponse(target, filename=target.name)
        if target.suffix == ".json":
            return JSONResponse(json.loads(target.read_text()))
        if target.suffix in (".txt", ".md", ".log", ".csv"):
            return PlainTextResponse(target.read_text())
        return FileResponse(target, filename=target.name)

    from modellab.server.pipeline_routes import add_pipeline_routes

    add_pipeline_routes(api, ws, jobs, cache, store, enqueue, art, read)
    from modellab.server.model_routes import add_model_routes

    add_model_routes(api, ws, settings, cache, require_local)
    app.include_router(api)
    return app


def serve(settings: ServerSettings, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    if host not in ("127.0.0.1", "localhost", "::1") and not settings.token:
        log.warning("serving on %s without a token; this server can run uploaded code", host)
    uvicorn.run(create_app(settings), host=host, port=port, log_level="info")
