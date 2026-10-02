import json
from pathlib import Path

from modellab.analysis import load_audit, load_evaluation, load_failure_analysis
from modellab.core.errors import ArtifactError
from modellab.core.results import ArtifactStore
from modellab.experiments.spec import canonical_json
from modellab.hypotheses.config import HypothesisConfig
from modellab.hypotheses.evidence import DISCLAIMER, update_hypotheses
from modellab.hypotheses.rules import Context, audit_flag_ids, generate_hypotheses, select_failures
from modellab.hypotheses.selection import select_experiments


def build_plan(analysis, baseline_meta: dict, audit_report, cfg: HypothesisConfig, allow_images: bool):
    failures = select_failures(analysis, cfg)
    ctx = Context(analysis.samples, (baseline_meta or {}).get("preprocessing"), audit_flag_ids(audit_report))
    hypotheses = []
    for failure in failures:
        hypotheses.extend(generate_hypotheses(failure, ctx, cfg))
    plan = select_experiments(failures, hypotheses, analysis.samples, analysis.analysis_id, cfg, allow_images)
    hypotheses = update_hypotheses(hypotheses, plan, [], None, cfg)
    return failures, hypotheses, plan


def hypotheses_document(failures, hypotheses) -> dict:
    groups = []
    for failure in failures:
        members = sorted(h.hypothesis_id for h in hypotheses if h.failure_id == failure.target.failure_id)
        if len(members) > 1:
            groups.append({"failure_id": failure.target.failure_id, "hypothesis_ids": members})
    return {
        "schema_version": 1, "disclaimer": DISCLAIMER,
        "failures": [f.target.model_dump(mode="json") for f in failures],
        "hypotheses": [h.model_dump(mode="json") for h in hypotheses],
        "competing_groups": groups,
    }


def generate_plan(store: ArtifactStore, analysis_id: str, config: HypothesisConfig | None = None,
                  audit_id: str | None = None, run_id: str | None = None, allow_images: bool = True) -> dict:
    cfg = config or HypothesisConfig()
    analysis = load_failure_analysis(store, analysis_id)
    evaluation_id = analysis.report["inputs"]["evaluation_id"]
    _, _, meta = load_evaluation(store.root, evaluation_id)
    audit_report = load_audit(store.root, audit_id)[1] if audit_id else None
    failures, hypotheses, plan = build_plan(analysis, meta, audit_report, cfg, allow_images)
    document = hypotheses_document(failures, hypotheses)
    run_id = run_id or f"{analysis_id}_hyp"
    relative = Path("hypotheses") / run_id
    directory = store.resolve(relative)
    if directory.exists():
        raise ArtifactError(f"hypothesis run '{run_id}' already exists")
    directory.mkdir(parents=True)
    store.write_json(relative / "hypotheses.json", document)
    store.write_json(relative / "experiment_plan.json", plan)
    json.loads(canonical_json(document))
    return {"run_id": run_id, "num_failures": len(failures), "num_hypotheses": len(hypotheses), "num_experiments": len(plan["selected"])}
