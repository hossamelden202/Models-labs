import hashlib
import json
import platform
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import modellab
from modellab.analysis import analyze_failures, load_audit, load_evaluation, load_failure_analysis
from modellab.audit import AuditConfig, run_audit
from modellab.core.errors import ArtifactError, ConfigurationError, EvaluationError
from modellab.core.results import ArtifactStore
from modellab.experiments import create_baseline, parse_spec, run_family
from modellab.experiments.spec import canonical_json
from modellab.hypotheses import DISCLAIMER, build_plan, hypotheses_document, update_hypotheses
from modellab.investigation.conclude import conclude
from modellab.investigation.config import InvestigationConfig
from modellab.utils import get_logger

log = get_logger("investigation")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
LIMITATIONS = [
    "Hypotheses are candidate explanations; a supported hypothesis is not a proven cause.",
    "Experiments change one factor at a time on one model and one evaluation set; results may not transfer.",
    "Slice associations are observed in this evaluation set and were not randomised.",
    "Interventions that need retraining are not available, so some explanations cannot be tested.",
]


@dataclass
class InvestigationRun:
    investigation_id: str
    directory: Path
    record: dict


def _check(*values) -> None:
    for value in values:
        if value is not None and not _ID.match(value):
            raise ConfigurationError(f"invalid id '{value}'")


def comparable(record: dict) -> dict:
    out = deepcopy(record)

    # Investigation identity and execution-specific values must not affect
    # reproducibility. Keep all substantive investigation results intact.
    ignored_keys = {
        "investigation_id",
        "family_id",
        "duration_seconds",
        "started_at",
        "finished_at",
        "timestamp",
        "created_at",
        "updated_at",
    }

    def normalize(value):
        if isinstance(value, dict):
            return {
                key: normalize(val)
                for key, val in value.items()
                if key not in ignored_keys
            }

        if isinstance(value, list):
            return [normalize(item) for item in value]

        return value

    return normalize(out)


def fingerprint(record: dict) -> str:
    return hashlib.sha256(canonical_json(comparable(record)).encode()).hexdigest()[:16]


def _experiments_summary(plan: dict, results: list) -> list:
    by_name = {r["spec"]["name"]: r for r in results}
    out = []
    for entry in plan["selected"]:
        result = by_name.get(entry["name"])
        item = {"name": entry["name"], "tests": entry["tests"], "intervention": entry["spec"]["intervention"],
                "population_kind": "sample_ids" if "sample_ids" in entry["spec"]["population"] else "slice", "cost": entry["cost"],
                "status": "not_run" if result is None else result["status"], "experiment_id": result["experiment_id"] if result else None}
        if result and result["status"] == "completed":
            stats, glob = result["statistics"]["affected"], result["comparison"]["global"]["differences"]["accuracy"]
            item["effect"] = {
                "n_affected": result["population"]["num_affected"], "accuracy_difference": stats["accuracy_difference"],
                "ci_low": stats["ci_low"], "ci_high": stats["ci_high"], "primary_p": stats["primary_p"],
                "global_accuracy_difference": glob["difference"], "label_uncorrected": result["verdict"]["label"],
            }
        elif result:
            item["error"] = result["error"]
        out.append(item)
    return out


def render_report(rec: dict) -> str:
    lines = [f"ModelLab investigation {rec['investigation_id']}", "", "1. OBSERVED RESULTS (measured; no interpretation)"]
    lines.append(f"baseline: {rec['baseline']['num_samples']} samples, accuracy {rec['baseline']['accuracy']:.4f}")
    for f in rec["failures"]:
        lines.append(f"failure {f['failure_id']}: {f['description']} (support {f['support']}, errors {f['errors']})")
    for e in rec["experiments"]:
        if "effect" in e:
            x = e["effect"]
            lines.append(f"experiment {e['name']} on {x['n_affected']} samples: accuracy {x['accuracy_difference']:+.4f} "
                         f"[{x['ci_low']:+.4f}, {x['ci_high']:+.4f}], global {x['global_accuracy_difference']:+.4f}, p={x['primary_p']:.3g}")
        else:
            lines.append(f"experiment {e['name']}: {e['status']}")
    lines += ["", "2. INFERRED ASSOCIATIONS AND HYPOTHESES (candidate explanations, not facts)"]
    for h in rec["hypotheses"]:
        lines.append(f"[{h['status']}, {h['confidence_label']} {h['evidence_strength']:.2f}] {h['hypothesis_id']} ({h['failure_id']}): {h['mechanism']}")
        for e in h["supporting_evidence"] + h["contradicting_evidence"] + h["inconclusive_evidence"]:
            lines.append(f"    {e['kind']} {e['stance']}: {e['summary']}")
        if h["note"]:
            lines.append(f"    note: {h['note']}")
    lines += ["", "3. CONCLUSIONS (only where the stated rules are met)"]
    for c in rec["conclusions"]:
        lines.append(f"failure {c['failure_id']}: {c['status']}")
        lines.append(f"    {c['statement']}")
        for k in c["rule_checks"]:
            lines.append(f"    rule {k['rule']}: {'pass' if k['passed'] else 'fail'} ({k['detail']})")
        for a in c["alternatives"]:
            lines.append(f"    alternative kept: {a['category']} {a['hypothesis_id']} {a['status']} {a['strength']:.2f}")
    lines += ["", "LIMITATIONS"] + [f"  - {t}" for t in rec["limitations"]] + [""]
    return "\n".join(lines)


def run_investigation(store: ArtifactStore, investigation_id: str, config: InvestigationConfig | None = None,
                      model=None, dataset=None, evaluation_id: str | None = None, audit_id: str | None = None,
                      scan_fn=None, analysis_id: str | None = None, family_id: str | None = None) -> InvestigationRun:
    cfg = config or InvestigationConfig()
    _check(investigation_id, evaluation_id, audit_id, analysis_id, family_id)
    relative = Path("investigations") / investigation_id
    directory = store.resolve(relative)
    if directory.exists():
        raise ArtifactError(f"investigation '{investigation_id}' already exists")
    started, clock = datetime.now(timezone.utc), time.monotonic()

    ev_id = evaluation_id or f"{investigation_id}_baseline"
    if not (store.root / "evaluations" / ev_id / "predictions.parquet").is_file():
        if model is None or dataset is None:
            raise EvaluationError("no baseline evaluation exists and no model and dataset were given")
        from modellab.evaluation.engine import EvalConfig, run_evaluation

        log.info("running baseline evaluation %s", ev_id)
        run_evaluation(model, dataset, store, EvalConfig(batch_size=cfg.batch_size, seed=cfg.seed), evaluation_id=ev_id)
    predictions, names, meta = load_evaluation(store.root, ev_id)

    au_id = audit_id or (f"{investigation_id}_audit" if scan_fn is not None else None)
    records = report = None
    if au_id:
        if not (store.root / "audits" / au_id / "records.parquet").is_file():
            if scan_fn is None:
                raise EvaluationError(f"audit '{au_id}' does not exist and no dataset scan was provided")
            log.info("running dataset audit %s", au_id)
            audit_cfg = AuditConfig.model_validate(cfg.audit)
            run_audit(scan_fn(audit_cfg), store, audit_cfg, audit_id=au_id)
        records, report = load_audit(store.root, au_id)

    an_id = analysis_id or f"{investigation_id}_analysis"
    if (store.root / "failure_analyses" / an_id / "report.json").is_file():
        analysis = load_failure_analysis(store, an_id)
        if analysis.report["inputs"].get("evaluation_id") != ev_id or analysis.report["config"] != cfg.analysis.model_dump(mode="json"):
            raise EvaluationError(f"analysis '{an_id}' exists but was built from different inputs or configuration")
    else:
        log.info("running failure analysis %s", an_id)
        analysis = analyze_failures(predictions, names, cfg.analysis, audit_records=records, audit_report=report,
                                    store=store, analysis_id=an_id, evaluation_id=ev_id, audit_id=au_id)

    allow_images = model is not None and dataset is not None and cfg.hypotheses.allow_image_experiments
    failures, hypotheses, plan = build_plan(analysis, meta, report, cfg.hypotheses, allow_images)
    log.info("%d failures, %d hypotheses, %d planned experiments", len(failures), len(hypotheses), len(plan["selected"]))

    fam = family_id or f"{investigation_id}_family"
    results, family_report = [], None
    baseline_fp = None
    if plan["selected"]:
        baseline_fp = create_baseline(store, fam, ev_id).fingerprint
        specs = [parse_spec(e["spec"]) for e in plan["selected"]]
        run = run_family(store, fam, specs, model=model if allow_images else None,
                         dataset=dataset if allow_images else None, analysis=analysis, alpha=cfg.hypotheses.alpha)
        results, family_report = [o.result for o in run.outcomes], run.report
    hypotheses = update_hypotheses(hypotheses, plan, results, family_report, cfg.hypotheses)
    conclusions = [conclude(f.target.failure_id, [h for h in hypotheses if h.failure_id == f.target.failure_id], cfg) for f in failures]

    experiments = _experiments_summary(plan, results)
    links = [
        {"failure_id": h.failure_id, "hypothesis_id": h.hypothesis_id, "experiment_id": e.experiment_id,
         "test_id": e.test_id, "stance": e.stance, "strength": e.strength, "evidence": e.summary}
        for h in hypotheses for e in h.supporting_evidence + h.contradicting_evidence + h.inconclusive_evidence if e.kind == "experiment"
    ]
    doc = hypotheses_document(failures, hypotheses)
    record = {
        "schema_version": 1, "investigation_id": investigation_id, "disclaimer": DISCLAIMER,
        "inputs": {"evaluation_id": ev_id, "audit_id": au_id, "analysis_id": an_id, "family_id": fam if plan["selected"] else None,
                   "baseline_fingerprint": baseline_fp, "analysis_fingerprint": analysis.report["inputs"]["fingerprint"],
                   "model_id": meta.get("model_id"), "dataset_id": meta.get("dataset_id")},
        "baseline": {"num_samples": analysis.report["inputs"]["n_samples"], "accuracy": analysis.report["overall"]["accuracy"]},
        "failures": [
            {**f.target.model_dump(mode="json"),
             "hypothesis_ids": sorted(h.hypothesis_id for h in hypotheses if h.failure_id == f.target.failure_id),
             "experiment_ids": sorted({
                 link["experiment_id"]
                 for link in links
                 if link["failure_id"] == f.target.failure_id
             })}
            for f in failures
        ],
        "hypotheses": doc["hypotheses"], "experiments": experiments, "links": links, "conclusions": conclusions,
        "family_report": family_report, "limitations": LIMITATIONS,
    }
    config_doc = {"config": cfg.model_dump(mode="json"), "inputs": {k: record["inputs"][k] for k in ("evaluation_id", "audit_id", "analysis_id")}}
    directory.mkdir(parents=True)
    try:
        store.write_json(relative / "investigation.json", record)
        store.write_json(relative / "hypotheses.json", doc)
        store.write_json(relative / "experiment_plan.json", plan)
        store.write_json(relative / "config.json", config_doc)
        store.write_text(relative / "evidence_report.txt", render_report(record))
        store.write_json(relative / "metadata.json", {
            "investigation_id": investigation_id, "timestamp": started.isoformat(),
            "duration_seconds": round(time.monotonic() - clock, 3), "fingerprint": fingerprint(record),
            "versions": {"modellab": modellab.__version__, "python": platform.python_version()},
        })
    except BaseException:
        import shutil

        shutil.rmtree(directory, ignore_errors=True)
        raise
    return InvestigationRun(investigation_id, directory, record)


def reproduce_investigation(store: ArtifactStore, investigation_id: str, new_id: str, model=None, dataset=None, scan_fn=None) -> dict:
    base = store.root / "investigations" / investigation_id
    if not (base / "config.json").is_file():
        raise ArtifactError(f"investigation '{investigation_id}' not found")
    stored = json.loads((base / "config.json").read_text())
    original = json.loads((base / "investigation.json").read_text())
    cfg = InvestigationConfig.model_validate(stored["config"])
    inputs = stored["inputs"]
    rerun = run_investigation(store, new_id, cfg, model, dataset, evaluation_id=inputs["evaluation_id"],
                              audit_id=inputs["audit_id"], scan_fn=scan_fn, analysis_id=inputs["analysis_id"])
    return {"reproduced": comparable(original) == comparable(rerun.record), "original_fingerprint": fingerprint(original),
            "new_fingerprint": fingerprint(rerun.record), "new_investigation_id": new_id}
