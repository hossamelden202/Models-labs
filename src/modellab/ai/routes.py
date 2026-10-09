import importlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from modellab.ai import tools
from modellab.ai.analysis import normalize_per_class
from modellab.ai.chat import answer_question, build_context, route_intent
from modellab.ai.finding import interpret
from modellab.ai.knowledge.ingestion import ingest_all
from modellab.ai.knowledge.store import HashingEmbedder, KnowledgeStore, SentenceTransformerEmbedder
from modellab.ai.llm.provider import NoLLM, provider_from_env
from modellab.ai.report import build_report
from modellab.ai.research.graph import run_research
from modellab.ai.schemas import Change
from modellab.ai.validation import validate_changes
from modellab.experiments.spec import experiment_id, parse_spec
from modellab.server.errors import ApiError
from modellab.server.workspace import check_id


class _Req(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ResearchRequest(_Req):
    family_id: str
    question: str = Field("Why is my detector weak?", min_length=1, max_length=500)
    audit_id: str | None = None
    analysis_id: str | None = None


class ChatCreate(_Req):
    family_id: str
    audit_id: str | None = None
    analysis_id: str | None = None


class ChatContext(_Req):
    audit_id: str | None = None
    analysis_id: str | None = None


class ChatMessage(_Req):
    text: str = Field(min_length=1, max_length=2000)


CONTROL_SPEC = {
    "name": "advisor_control_default",
    "hypothesis": {"claim": "Default-settings control run", "expected_direction": "no_change"},
    "intervention": {"kind": "training", "changes": [{"path": "epochs", "value": 10}]},
}


class ApproveRequest(_Req):
    include_control: bool | None = None
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
    chats = root / "ai_chats"
    rt = _Runtime(root, llm_factory or provider_from_env, embedder_factory or default_embedder)
    record_lock = threading.Lock()
    chat_lock = threading.Lock()

    def svc():
        return services or importlib.import_module("modellab.server.services")

    def read_json(path):
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

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
        family_dir = root / "experiments" / record["family_id"]
        path = family_dir / record["result"]["validation"]["experiment_id"] / "result.json"
        if not path.is_file():
            return record
        if approval.get("force") and path.stat().st_mtime < approval["at_epoch"]:
            return record
        result = read_json(path)
        if not result or result.get("status") not in ("completed", "failed"):
            return record
        control = None
        if approval.get("control_experiment_id"):
            control = read_json(family_dir / approval["control_experiment_id"] / "result.json")
        record["finding"] = interpret(record["result"]["hypothesis"], result, control)
        save(record)
        return record

    def view(record):
        out = dict(record)
        out["job"] = job_view(record["approval"]["job_id"]) if record.get("approval") else None
        out["control_available"] = tools.find_control(root, record["family_id"]) is not None
        return out

    def check_context(family_id, audit_id, analysis_id):
        try:
            baseline = tools.get_baseline(root, family_id)
            if audit_id:
                tools.audit_digest(root, audit_id, 100)
            if analysis_id:
                tools.failure_per_class(root, baseline, analysis_id)
        except tools.ToolError as exc:
            raise ApiError(422, str(exc))
        return baseline

    def create_research(family_id, question, audit_id=None, analysis_id=None, exclude=(), tolerate_no_llm=False):
        check_id(family_id, "family id")
        baseline = check_context(family_id, audit_id, analysis_id)
        meta = baseline.get("metadata") or {}
        note = None
        with rt.lock:
            try:
                llm = rt.llm()
            except ApiError as exc:
                if not tolerate_no_llm:
                    raise
                llm, note = NoLLM(), str(exc.message)
            kb = rt.knowledge()
            try:
                result = run_research(root, family_id, question, llm, kb,
                                      {"audit_id": audit_id, "analysis_id": analysis_id, "exclude": list(exclude)})
            except tools.ToolError as exc:
                raise ApiError(422, str(exc))
        if note:
            result["notes"].append(note + "; the top-ranked option was chosen by rule")
        record = {
            "id": f"res_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:6]}",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "family_id": family_id,
            "question": question,
            "status": "proposed" if result["proposal"] else "no_proposal",
            "defaults": {"model_id": meta.get("model_id"), "dataset_id": meta.get("dataset_id")},
            "result": result,
            "approval": None,
            "finding": None,
        }
        save(record)
        return record

    @api.post("/ai/research")
    def start_research(req: ResearchRequest):
        return view(create_research(req.family_id, req.question, req.audit_id, req.analysis_id))

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
            existing = tools.find_control(root, record["family_id"])
            specs = [fresh.spec]
            control_id = existing
            if existing is None and req.include_control is not False:
                control = parse_spec(CONTROL_SPEC)
                specs = [control.model_dump(mode="json"), fresh.spec]
                control_id = experiment_id(control)
            params = {
                "family_id": record["family_id"], "baseline_evaluation_id": None,
                "specs": specs, "matrices": [], "model_id": model_id, "dataset_id": dataset_id,
                "subpath": req.subpath, "device": req.device, "force": req.force,
                "stop_on_failure": True, "alpha": 0.05,
            }
            job = jobs.submit("experiments", params, lambda: svc().experiments(ws, cache, store, params))
            record["status"] = "approved"
            record["finding"] = None
            record["approval"] = {
                "job_id": job["job_id"], "model_id": model_id, "dataset_id": dataset_id,
                "force": req.force, "at_epoch": time.time(),
                "control_experiment_id": control_id, "control_runs": existing is None and control_id is not None,
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            save(record)
        if wait:
            jobs.wait(job["job_id"], timeout)
        body = {"research_id": rid, "status": "approved", "job": job_view(job["job_id"]),
                "control": {"experiment_id": control_id, "runs": len(specs) == 2}}
        return JSONResponse(body, status_code=202)

    def stamp():
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def save_chat(chat):
        folder = chats / chat["id"]
        folder.mkdir(parents=True, exist_ok=True)
        tmp = folder / "chat.json.tmp"
        tmp.write_text(json.dumps(chat, indent=1))
        tmp.replace(folder / "chat.json")

    def load_chat(cid):
        check_id(cid, "chat id")
        path = chats / cid / "chat.json"
        if not path.is_file():
            raise ApiError(404, f"chat '{cid}' not found")
        return json.loads(path.read_text())

    def add(chat, role, kind="text", **fields):
        msg = {"id": f"m{len(chat['messages']) + 1}", "role": role, "kind": kind, "text": "",
               "research_id": None, "report": None, "warning": None, "created_at": stamp()}
        msg.update(fields)
        chat["messages"].append(msg)
        return msg

    def fresh(rid):
        try:
            with record_lock:
                return refresh(load(rid))
        except ApiError:
            return None

    def hydrate(chat):
        out = dict(chat)
        out["messages"] = []
        for m in chat["messages"]:
            m = dict(m)
            if m.get("research_id"):
                rec = fresh(m["research_id"])
                m["research"] = view(rec) if rec else None
            out["messages"].append(m)
        return out

    def respond(chat, text):
        baseline = check_context(chat["family_id"], chat.get("audit_id"), chat.get("analysis_id"))
        names = [r["class_name"] for r in normalize_per_class((baseline.get("metrics") or {}).get("per_class") or {})]
        rids = [m["research_id"] for m in chat["messages"] if m.get("research_id")]
        intent = route_intent(text, names)

        if intent == "report":
            rid = next((r for r in reversed(rids) if fresh(r)), None)
            if rid is None:
                add(chat, "assistant", text="There is nothing to report yet. Ask me why the detector is weak and I will analyse it first.")
            else:
                add(chat, "assistant", "report", text=f"Report for {rid}", report=build_report(fresh(rid), chat), research_id=rid)
            return

        if intent == "propose":
            excluded = []
            for rid in rids:
                rec = fresh(rid)
                if rec and rec["result"].get("proposal"):
                    excluded.append(rec["result"]["proposal"]["candidate_id"])
            record = create_research(chat["family_id"], text, chat.get("audit_id"), chat.get("analysis_id"),
                                     exclude=excluded, tolerate_no_llm=True)
            if record["status"] == "no_proposal":
                add(chat, "assistant", text="I have no further experiment to propose. " + " ".join(record["result"]["notes"]))
            else:
                add(chat, "assistant", "research", research_id=record["id"],
                    text="Here is what I found and the experiment I would run next.")
            return

        latest = fresh(rids[-1]) if rids else None
        try:
            with rt.lock:
                kb = rt.knowledge()
                ingest_all(root, kb)
                try:
                    llm = rt.llm()
                except ApiError:
                    llm = None
                context, analysis = build_context(root, chat, text, kb, latest)
                answer, _, warning = answer_question(llm, chat, text, context, analysis)
        except tools.ToolError as exc:
            raise ApiError(422, str(exc))
        add(chat, "assistant", text=answer, warning=warning)

    @api.get("/ai/context-options")
    def context_options(family_id: str | None = None):
        evaluation_id = None
        if family_id:
            check_id(family_id, "family id")
            try:
                evaluation_id = tools.get_baseline(root, family_id).get("evaluation_id")
            except tools.ToolError as exc:
                raise ApiError(422, str(exc))
        return {"families": tools.list_families(root), "audits": tools.list_audits(root),
                "analyses": tools.list_analyses(root, evaluation_id)}

    @api.post("/ai/chats")
    def create_chat(req: ChatCreate):
        check_id(req.family_id, "family id")
        check_context(req.family_id, req.audit_id, req.analysis_id)
        chat = {"id": f"chat_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:6]}",
                "created_at": stamp(), "family_id": req.family_id, "audit_id": req.audit_id,
                "analysis_id": req.analysis_id, "title": "", "messages": []}
        with chat_lock:
            save_chat(chat)
        return hydrate(chat)

    @api.get("/ai/chats")
    def list_chats():
        rows = []
        for path in sorted(chats.glob("*/chat.json"), reverse=True):
            try:
                c = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            rows.append({"id": c["id"], "family_id": c["family_id"], "title": c["title"] or "New chat",
                         "created_at": c["created_at"], "messages": len(c["messages"])})
        return rows

    @api.get("/ai/chats/{cid}")
    def get_chat(cid: str):
        return hydrate(load_chat(cid))

    @api.post("/ai/chats/{cid}/context")
    def set_chat_context(cid: str, req: ChatContext):
        with chat_lock:
            chat = load_chat(cid)
            check_context(chat["family_id"], req.audit_id, req.analysis_id)
            chat["audit_id"], chat["analysis_id"] = req.audit_id, req.analysis_id
            add(chat, "assistant", text=f"Context updated. Dataset audit: {req.audit_id or 'none'}. "
                                        f"Failure analysis: {req.analysis_id or 'latest for the baseline'}.")
            save_chat(chat)
        return hydrate(chat)

    @api.post("/ai/chats/{cid}/messages")
    def send_message(cid: str, req: ChatMessage):
        with chat_lock:
            chat = load_chat(cid)
            text = req.text.strip()
            add(chat, "user", text=text)
            respond(chat, text)
            if not chat["title"]:
                chat["title"] = text[:60]
            save_chat(chat)
        return hydrate(chat)

    @api.get("/ai/research/{rid}/report")
    def research_report(rid: str):
        with record_lock:
            record = refresh(load(rid))
        return PlainTextResponse(build_report(record), media_type="text/markdown")
