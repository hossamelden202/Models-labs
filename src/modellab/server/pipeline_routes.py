import json
import uuid
from typing import Literal

from fastapi import Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from modellab.server import pipeline_services as ps
from modellab.server.errors import ApiError
from modellab.server.workspace import check_id


class _Req(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class HypothesisRequest(_Req):
    analysis_id: str
    audit_id: str | None = None
    run_id: str | None = None
    config: dict = Field(default_factory=dict)
    allow_image_experiments: bool = True


class InvestigationRequest(_Req):
    investigation_id: str | None = None
    model_id: str | None = None
    dataset_id: str | None = None
    subpath: str | None = None
    evaluation_id: str | None = None
    audit_id: str | None = None
    analysis_id: str | None = None
    config: dict = Field(default_factory=dict)
    device: Literal["auto", "cpu", "cuda"] = "auto"


class ReproduceRequest(_Req):
    new_investigation_id: str
    model_id: str | None = None
    dataset_id: str | None = None
    subpath: str | None = None
    device: Literal["auto", "cpu", "cuda"] = "auto"


class RepairRequest(_Req):
    investigation_id: str
    repair_id: str | None = None
    model_id: str | None = None
    dataset_id: str | None = None
    subpath: str | None = None
    config: dict = Field(default_factory=dict)
    device: Literal["auto", "cpu", "cuda"] = "auto"


def add_pipeline_routes(api, ws, jobs, cache, store, enqueue, art, read):
    root = ws.artifacts_dir

    def names(kind):
        base = root / kind
        return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []

    def metadata(kind):
        out = []
        for name in names(kind):
            path = root / kind / name / "metadata.json"
            if path.is_file():
                out.append(json.loads(path.read_text()))
        return out

    def new_id(prefix, given):
        return check_id(given or f"{prefix}-{uuid.uuid4().hex[:8]}", f"{prefix} id")

    def check_objects(model_id, dataset_id):
        if model_id:
            ws.get_model(model_id)
        if dataset_id:
            ws.get_dataset(dataset_id)

    @api.post("/hypotheses", status_code=201)
    def make_hypotheses(req: HypothesisRequest):
        from modellab.hypotheses import HypothesisConfig, generate_plan

        art("failure_analyses", req.analysis_id)
        if req.audit_id:
            art("audits", req.audit_id)
        cfg = HypothesisConfig.model_validate(req.config)
        run_id = new_id("hyp", req.run_id)
        if (root / "hypotheses" / run_id).exists():
            raise ApiError(409, f"hypothesis run '{run_id}' already exists")
        return generate_plan(store, req.analysis_id, cfg, req.audit_id, run_id, req.allow_image_experiments)

    @api.get("/hypotheses")
    def list_hypotheses():
        return names("hypotheses")

    @api.get("/hypotheses/{run_id}")
    def get_hypotheses(run_id: str):
        return read(art("hypotheses", run_id, "hypotheses.json"))

    @api.get("/hypotheses/{run_id}/plan")
    def get_plan(run_id: str):
        return read(art("hypotheses", run_id, "experiment_plan.json"))

    @api.post("/investigations")
    def start_investigation(req: InvestigationRequest, wait: bool = False, timeout: float = Query(1800, gt=0, le=86400)):
        from modellab.investigation import InvestigationConfig

        InvestigationConfig.model_validate(req.config)
        check_objects(req.model_id, req.dataset_id)
        if not req.evaluation_id and not (req.model_id and req.dataset_id):
            raise ApiError(422, "give evaluation_id, or model_id and dataset_id to run the baseline")
        if req.evaluation_id:
            art("evaluations", req.evaluation_id)
        investigation_id = new_id("inv", req.investigation_id)
        if (root / "investigations" / investigation_id).exists():
            raise ApiError(409, f"investigation '{investigation_id}' already exists")
        params = {**req.model_dump(), "investigation_id": investigation_id}
        return enqueue("investigation", params, lambda: ps.investigate(ws, cache, store, params), wait, timeout)

    @api.post("/investigations/{investigation_id}/reproduce")
    def reproduce(investigation_id: str, req: ReproduceRequest, wait: bool = False, timeout: float = Query(1800, gt=0, le=86400)):
        art("investigations", investigation_id)
        check_id(req.new_investigation_id, "investigation id")
        check_objects(req.model_id, req.dataset_id)
        if (root / "investigations" / req.new_investigation_id).exists():
            raise ApiError(409, f"investigation '{req.new_investigation_id}' already exists")
        params = {**req.model_dump(), "investigation_id": investigation_id}
        return enqueue("reproduction", params, lambda: ps.reproduce(ws, cache, store, params), wait, timeout)

    @api.get("/investigations")
    def list_investigations():
        return metadata("investigations")

    @api.get("/investigations/{investigation_id}")
    def get_investigation(investigation_id: str):
        return read(art("investigations", investigation_id, "investigation.json"))

    @api.get("/investigations/{investigation_id}/report")
    def investigation_report(investigation_id: str):
        return PlainTextResponse(art("investigations", investigation_id, "evidence_report.txt").read_text())

    @api.post("/repairs")
    def start_repair(req: RepairRequest, wait: bool = False, timeout: float = Query(1800, gt=0, le=86400)):
        from modellab.repair import RepairConfig

        art("investigations", req.investigation_id)
        RepairConfig.model_validate(req.config)
        check_objects(req.model_id, req.dataset_id)
        repair_id = new_id("rep", req.repair_id)
        if (root / "repairs" / repair_id).exists():
            raise ApiError(409, f"repair '{repair_id}' already exists")
        params = {**req.model_dump(), "repair_id": repair_id}
        return enqueue("repair", params, lambda: ps.repair(ws, cache, store, params), wait, timeout)

    @api.get("/repairs")
    def list_repairs():
        return metadata("repairs")

    @api.get("/repairs/{repair_id}")
    def get_repair(repair_id: str):
        return read(art("repairs", repair_id, "repair_results.json"))

    @api.get("/repairs/{repair_id}/report")
    def repair_report(repair_id: str):
        return PlainTextResponse(art("repairs", repair_id, "repair_report.txt").read_text())

    @api.get("/repairs/{repair_id}/version")
    def repair_version(repair_id: str):
        path = art("repairs", repair_id, "repair_version.json")
        if not path.is_file():
            raise ApiError(404, "this repair was not accepted, so there is no version")
        return read(path)
