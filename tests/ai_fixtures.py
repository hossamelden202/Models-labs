import json

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from ai_client import AiClient
from modellab.ai.knowledge.store import HashingEmbedder
from modellab.ai.routes import add_ai_routes
from modellab.experiments.spec import experiment_id, parse_spec
from modellab.experiments.training_config import TrainingConfig, apply_training_changes
from modellab.server.errors import ApiError

FAMILY = "weapons-final-model"
BASE_METRICS = {
    "task": "detection", "f1": 0.16, "precision": 0.165, "recall": 0.155, "num_samples": 472,
    "per_class": {
        "handgun": {"class_id": 0, "f1": 0.01, "fn": 77, "fp": 81, "precision": 0.01, "recall": 0.013, "tp": 1},
        "knife": {"class_id": 2, "f1": 0.22, "fn": 190, "fp": 145, "precision": 0.25, "recall": 0.2, "tp": 48},
    },
}
GOOD = json.dumps({"candidate_id": "res_416", "claim": "Higher resolution should raise handgun recall",
                   "rationale": "small objects"})


class FakeWs:
    def __init__(self, root):
        self.artifacts_dir = root

    def get_model(self, model_id):
        if model_id != "weapons-final-model":
            raise ApiError(404, f"model '{model_id}' not found")

    def get_dataset(self, dataset_id):
        if dataset_id != "weapons-small":
            raise ApiError(404, f"dataset '{dataset_id}' not found")


CONTROL_TRAINED = {"f1": 0.25, "precision": 0.20, "recall": 0.30, "mean_iou": 0.85}
CANDIDATE_TRAINED = {"f1": 0.30, "precision": 0.25, "recall": 0.40, "mean_iou": 0.90}
BASELINE = {"f1": 0.16, "precision": 0.165, "recall": 0.155, "mean_iou": 0.75}


class FakeServices:
    def __init__(self, root):
        self.root = root
        self.calls = []
        self.mode = "ok"
        self.control_mode = "ok"

    def experiments(self, ws, cache, store, params):
        self.calls.append(params)
        if self.mode == "raise":
            raise RuntimeError("boom")
        for raw in params["specs"]:
            spec = parse_spec(raw)
            exp_id = experiment_id(spec)
            folder = self.root / "experiments" / params["family_id"] / exp_id
            folder.mkdir(parents=True, exist_ok=True)
            is_control = spec.name == "advisor_control_default"
            mode = self.control_mode if is_control else self.mode
            if mode == "fail":
                doc = {"status": "failed", "task": "detection", "error": {"type": "OOM", "message": "out of memory"}}
            else:
                trained = CONTROL_TRAINED if is_control else CANDIDATE_TRAINED
                config = apply_training_changes(TrainingConfig(), spec.intervention).model_dump(mode="python")
                doc = {"status": "completed", "task": "detection",
                       "training": {"config": config, "total_epochs": config["epochs"], "best_epoch": 3},
                       "comparison": {"available": True, "baseline": BASELINE, "trained": trained,
                                      "delta": {k: round(trained[k] - BASELINE[k], 6) for k in BASELINE}}}
            (folder / "result.json").write_text(json.dumps(doc))
        return {"ok": True}


class FakeJobs:
    def __init__(self):
        self.items = {}

    def submit(self, kind, params, fn):
        job = {"job_id": f"job{len(self.items) + 1}", "kind": kind, "status": "completed"}
        try:
            fn()
        except Exception as exc:
            job.update(status="failed", error=str(exc))
        self.items[job["job_id"]] = job
        return dict(job)

    def get(self, job_id):
        return dict(self.items[job_id])

    def wait(self, job_id, timeout):
        return self.get(job_id)


class Env:
    def __init__(self, root, llm):
        self.root = root
        self.ws = FakeWs(root)
        self.jobs = FakeJobs()
        self.services = FakeServices(root)
        self.llm = llm
        app = FastAPI()
        api = APIRouter()

        @app.exception_handler(ApiError)
        def handler(request, exc):
            return JSONResponse({"detail": exc.message}, status_code=exc.status)

        add_ai_routes(api, self.ws, self.jobs, None, None, services=self.services,
                      llm_factory=self.llm_factory, embedder_factory=HashingEmbedder)
        app.include_router(api)
        self.http = TestClient(app)
        self.client = AiClient("", session=self.http)

    def llm_factory(self):
        if isinstance(self.llm, Exception):
            raise self.llm
        return self.llm


def put(root, rel, data):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))


def make_env(tmp_path, replies=6):
    from modellab.ai.llm.provider import ScriptedLLM

    put(tmp_path, f"experiments/{FAMILY}/baseline.json", {
        "family_id": FAMILY, "task": "detection", "evaluation_id": "ev1", "metrics": BASE_METRICS,
        "metadata": {"model_id": "weapons-final-model", "dataset_id": "weapons-small"},
    })
    script = replies if isinstance(replies, list) else [GOOD] * replies
    return Env(tmp_path, ScriptedLLM(script))


def add_context(root):
    put(root, "audits/aud1/metadata.json", {"dataset_id": "weapons-small", "timestamp": "2026-10-05T10:00:00Z"})
    put(root, "audits/aud1/report.json", {
        "summary": {"duplicates": 12, "near_duplicates": 30, "corrupt": 0},
        "flags": [{"id": "dup_pairs", "severity": "high"}, {"id": "small_images", "severity": "low"}],
        "rows": [{"sample": i, "hash": f"h{i}"} for i in range(500)],
    })
    put(root, "failure_analyses/fa1/metadata.json", {"evaluation_id": "ev1", "task": "detection"})
    put(root, "failure_analyses/fa1/per_class.json", {
        "0": {"class": "handgun", "class_id": 0, "support": 78, "true_positives": 1, "false_positives": 81,
              "false_negatives": 77, "precision": 0.012, "recall": 0.013, "f1": 0.012},
        "2": {"class": "knife", "class_id": 2, "support": 238, "true_positives": 48, "false_positives": 145,
              "false_negatives": 190, "precision": 0.249, "recall": 0.202, "f1": 0.223}})
    put(root, "failure_analyses/fa1/report.json", {"clusters": [{"id": 1, "size": 40}], "warnings": ["small sample"]})
    put(root, "failure_analyses/other/metadata.json", {"evaluation_id": "different", "task": "detection"})
    put(root, "failure_analyses/other/per_class.json", {})
