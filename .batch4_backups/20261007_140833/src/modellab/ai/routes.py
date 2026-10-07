import importlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from modellab.ai import tools
from modellab.ai.finding import interpret
from modellab.ai.knowledge.store import HashingEmbedder, KnowledgeStore, SentenceTransformerEmbedder
from modellab.ai.llm.provider import provider_from_env
from modellab.ai.research.graph import run_research
from modellab.ai.schemas import Change
from modellab.ai.validation import validate_changes
from modellab.experiments.spec import parse_spec
from modellab.server.errors import ApiError
from modellab.server.workspace import check_id


class _Req(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ResearchRequest(_Req):
    family_id: str
    question: str = Field("Why is my detector weak?", min_length=1, max_length=500)


class ApproveRequest(_Req):
    model_id: str | None = None
    dataset_id: str | None = None
    subpath: str | None = None
    device: Literal["auto", "cpu", "cuda"] = "auto"
    force: bool = False


def default_embedder():
    if os.environ.get("MODELLAB_EMBEDDER", "minilm") == "hashing":
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder()
    except Exception:
        return HashingEmbedder()


class _Runtime:
    def __init__(self, root, llm_factory, embedder_factory):
        self.root = root
        self.llm_factory = llm_factory
        self.embedder_factory = embedder_factory
        self.lock = threading.Lock()
        self._llm = None
        self._kb = None

    def llm(self):
        if self._llm is None:
            try:
                self._llm = self.llm_factory()
            except Exception as exc:
                raise ApiError(503, f"LLM backend unavailable: {str(exc)[:300]}")
        return self._llm

    def knowledge(self):
        if self._kb is None:
            self._kb = KnowledgeStore(self.embedder_factory(), self.root / "knowledge")
        return self._kb


def add_ai_routes(api, ws, jobs, cache, store, services=None, llm_factory=None, embedder_factory=None):
    root = Path(ws.artifacts_dir)
    base = root / "ai_research"
    rt = _Runtime(root, llm_factory or provider_from_env, embedder_factory or default_embedder)
    record_lock = threading.Lock()

    def svc():
        return services or importlib.import_module("modellab.server.services")

    def save(record):
        folder = base / record["id"]
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / "research.json.tmp"
        tmp.write_text(json.dumps(record, indent=1))
        tmp.replace(folder / "research.json")

    def load(rid):
        check_id(rid, "research id")
        path = base / rid / "research.json"
        if not path.is_file():
            raise ApiError(404, f"research '{rid}' not found")
        return json.loads(path.read_text())

    def job_view(job_id):
        try:
            job = jobs.get(job_id)
        except Exception:
            return {"job_id": job_id, "status": "unknown"}
        view = {k: job[k] for k in ("job_id", "status", "error") if k in job}
        return view or {"job_id": job_id, "status": "unknown"}

    def refresh(record):
        if record["status"] != "approved" or record.get("finding"):
            return record
        approval = record["approval"]
        exp_id = record["result"]["validation"]["experiment_id"]
        path = root / "experiments" / record["family_id"] / exp_id / "result.json"
        if not path.is_file():
            return record
        if approval.get("force") and path.stat().st_mtime < approval["at_epoch"]:
            return record
        try:
            result = json.loads(path.read_text())
        except json.JSONDecodeError:
            return record
        if result.get("status") not in ("completed", "failed"):
            return record
        record["finding"] = interpret(record["result"]["hypothesis"], result)
        save(record)
        return record

    def view(record):
        out = dict(record)
        out["job"] = job_view(record["approval"]["job_id"]) if record.get("approval") else None
        return out

    @api.post("/ai/research")
    def start_research(req: ResearchRequest):
        check_id(req.family_id, "family id")
        try:
            baseline = tools.get_baseline(root, req.family_id)
        except tools.ToolError as exc:
            raise ApiError(422, str(exc))
        meta = baseline.get("metadata") or {}
        with rt.lock:
            llm = rt.llm()
            kb = rt.knowledge()
            try:
                result = run_research(root, req.family_id, req.question, llm, kb)
            except tools.ToolError as exc:
                raise ApiError(422, str(exc))
        record = {
            "id": f"res_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:6]}",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "family_id": req.family_id,
            "question": req.question,
            "status": "proposed" if result["proposal"] else "no_proposal",
            "defaults": {"model_id": meta.get("model_id"), "dataset_id": meta.get("dataset_id")},
            "result": result,
            "approval": None,
            "finding": None,
        }
        save(record)
        return view(record)

    @api.get("/ai/research")
    def list_research():
        rows = []
        for path in sorted(base.glob("*/research.json"), reverse=True):
            try:
                rec = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            proposal = rec["result"].get("proposal") or {}
            rows.append({
                "id": rec["id"], "created_at": rec["created_at"], "family_id": rec["family_id"],
                "status": rec["status"], "candidate_id": proposal.get("candidate_id"),
                "verdict": (rec.get("finding") or {}).get("verdict"),
            })
        return rows

    @api.get("/ai/research/{rid}")
    def get_research(rid: str):
        with record_lock:
            return view(refresh(load(rid)))

    @api.post("/ai/research/{rid}/reject")
    def reject_research(rid: str):
        with record_lock:
            record = load(rid)
            if record["status"] != "proposed":
                raise ApiError(409, f"research is {record['status']}, only a proposed one can be rejected")
            record["status"] = "rejected"
            save(record)
            return view(record)

    @api.post("/ai/research/{rid}/approve")
    def approve_research(rid: str, req: ApproveRequest, wait: bool = False,
                         timeout: float = Query(600, gt=0, le=86400)):
        with record_lock:
            record = load(rid)
            if record["status"] == "approved" and not req.force:
                raise ApiError(409, "already approved; pass force to run it again")
            if record["status"] not in ("proposed", "approved"):
                raise ApiError(409, f"research is {record['status']}, nothing to approve")
            proposal = record["result"]["proposal"]
            stored = record["result"]["validation"]
            if not stored or not stored["valid"]:
                raise ApiError(422, "the proposal did not pass validation")
            fresh = validate_changes([Change(**c) for c in proposal["changes"]], proposal["name"], proposal["reason"])
            if not fresh.valid or fresh.experiment_id != stored["experiment_id"]:
                raise ApiError(422, "the stored proposal no longer validates: " + "; ".join(fresh.errors))
            model_id = req.model_id or record["defaults"].get("model_id")
            dataset_id = req.dataset_id or record["defaults"].get("dataset_id")
            if not model_id or not dataset_id:
                raise ApiError(422, "model_id and dataset_id are needed")
            ws.get_model(model_id)
            ws.get_dataset(dataset_id)
            parse_spec(fresh.spec)
            params = {
                "family_id": record["family_id"], "baseline_evaluation_id": None,
                "specs": [fresh.spec], "matrices": [], "model_id": model_id, "dataset_id": dataset_id,
                "subpath": req.subpath, "device": req.device, "force": req.force,
                "stop_on_failure": True, "alpha": 0.05,
            }
            job = jobs.submit("experiments", params, lambda: svc().experiments(ws, cache, store, params))
            record["status"] = "approved"
            record["finding"] = None
            record["approval"] = {
                "job_id": job["job_id"], "model_id": model_id, "dataset_id": dataset_id,
                "force": req.force, "at_epoch": time.time(),
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            save(record)
        if wait:
            jobs.wait(job["job_id"], timeout)
        return JSONResponse({"research_id": rid, "status": "approved", "job": job_view(job["job_id"])}, status_code=202)
