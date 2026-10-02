import hashlib
import json
import platform
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import modellab
from modellab.analysis import load_failure_analysis
from modellab.analysis.stats import benjamini_hochberg
from modellab.core.errors import ArtifactError, ConfigurationError, EvaluationError
from modellab.core.results import ArtifactStore
from modellab.experiments import create_baseline, load_baseline, parse_spec, run_family
from modellab.experiments.spec import canonical_json
from modellab.experiments.stats import paired_summary
from modellab.hypotheses import population_ids
from modellab.repair.candidates import build_candidates
from modellab.repair.config import RepairConfig
from modellab.utils import get_logger

log = get_logger("repair")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
NOTE = "The original model and the baseline evaluation were not modified."


class RepairError(EvaluationError):
    pass


@dataclass
class RepairRun:
    repair_id: str
    directory: Path
    results: dict


def load_investigation(store: ArtifactStore, investigation_id: str) -> dict:
    path = store.root / "investigations" / investigation_id / "investigation.json"
    if not path.is_file():
        raise RepairError(f"investigation '{investigation_id}' not found")
    record = json.loads(path.read_text())
    for key in ("schema_version", "inputs", "failures", "hypotheses", "conclusions"):
        if key not in record:
            raise RepairError(f"investigation record is missing '{key}'")
    for key in ("evaluation_id", "analysis_id"):
        if not record["inputs"].get(key):
            raise RepairError(f"investigation record has no inputs.{key}")
    return record


def comparable(results: dict) -> dict:
    out = json.loads(json.dumps(results))
    out.pop("repair_id", None)
    out["baseline"].pop("family_id", None)
    return out


def _evaluate(paired: pd.DataFrame, target_ids: set, names: list, important: list, cfg: RepairConfig) -> dict:
    df = paired.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    ids, y = df["sample_id"].to_numpy(), df["true_label"].to_numpy(dtype=int)
    before, after = df["baseline_correct"].to_numpy(bool), df["intervention_correct"].to_numpy(bool)
    target = np.isin(ids, list(target_ids))
    c = cfg.criteria
    stats = paired_summary(before[target], [after[target]], cfg.n_bootstrap, cfg.n_permutations, cfg.seed)
    new_errors, fixed = int((before & ~after).sum()), int((~before & after).sum())
    classes = []
    for index, name in enumerate(names):
        m = y == index
        if int(m.sum()) >= c.min_support:
            classes.append({"class": name, "n": int(m.sum()), "recall_before": float(before[m].mean()),
                            "recall_after": float(after[m].mean()), "drop": float(before[m].mean() - after[m].mean())})
    slices = []
    for slice_id, members in important:
        m = np.isin(ids, list(members))
        if int(m.sum()) >= c.min_support:
            slices.append({"slice_id": slice_id, "n": int(m.sum()), "accuracy_before": float(before[m].mean()),
                           "accuracy_after": float(after[m].mean()), "drop": float(before[m].mean() - after[m].mean())})
    return {
        "target": {"n": int(target.sum()), "accuracy_before": float(before[target].mean()), "accuracy_after": float(after[target].mean()),
                   "difference": stats["accuracy_difference"], "ci_low": stats["ci_low"], "ci_high": stats["ci_high"],
                   "p": stats["primary_p"], "test": stats["primary_test"]},
        "global": {"n": int(len(ids)), "accuracy_before": float(before.mean()), "accuracy_after": float(after.mean()),
                   "difference": float(after.mean() - before.mean()), "new_errors": new_errors, "fixed_errors": fixed,
                   "new_error_rate": new_errors / max(int(before.sum()), 1)},
        "classes": classes, "slices": slices,
    }


def _checks(ev: dict, q: float, c) -> list:
    t, g = ev["target"], ev["global"]
    worst_class = max((x["drop"] for x in ev["classes"]), default=0.0)
    worst_slice = max((x["drop"] for x in ev["slices"]), default=0.0)
    out = [
        ("target_improvement", t["difference"] >= c.min_target_improvement, t["difference"], c.min_target_improvement),
        ("global_accuracy", -g["difference"] <= c.max_global_drop, -g["difference"], c.max_global_drop),
        ("new_error_rate", g["new_error_rate"] <= c.max_new_error_rate, g["new_error_rate"], c.max_new_error_rate),
        ("class_recall", worst_class <= c.max_class_recall_drop, worst_class, c.max_class_recall_drop),
        ("slice_regression", worst_slice <= c.max_slice_regression, worst_slice, c.max_slice_regression),
    ]
    if c.require_significance:
        out.insert(1, ("target_significance", q <= c.alpha, q, c.alpha))
    return [
        {
            "name": name,
            "passed": bool(passed),
            "value": float(value),
            "limit": float(limit_value),
        }
        for name, passed, value, limit_value in out
    ]


def render_report(res: dict) -> str:
    lines = [f"ModelLab repair {res['repair_id']} for investigation {res['investigation_id']}", ""]
    lines.append(f"baseline evaluation {res['baseline']['evaluation_id']} (unchanged: {res['baseline']['unchanged']})")
    lines.append(f"decision: {res['decision']}")
    for r in res["results"]:
        lines.append("")
        lines.append(f"[{r['decision']}] {r['candidate_id']} ({r['category']}): {r['description']}")
        lines.append(f"    failure {r['failure_id']}, mechanism {r['mechanism_category']} (hypothesis {r['hypothesis_id']})")
        if r["status"] != "evaluated":
            lines.append(f"    not evaluated: {r['reason']}")
            continue
        t, g = r["target"], r["global"]
        lines.append(f"    target accuracy {t['accuracy_before']:.4f} -> {t['accuracy_after']:.4f} (q={t['q']:.3g}); "
                     f"global {g['accuracy_before']:.4f} -> {g['accuracy_after']:.4f}; new errors {g['new_errors']}, fixed {g['fixed_errors']}")
        for k in r["checks"]:
            lines.append(f"    check {k['name']}: {'pass' if k['passed'] else 'fail'} ({k['value']:.4f} against {k['limit']:.4f})")
    if res["recommendations"]:
        lines += ["", "RECOMMENDATIONS (not executed)"]
        lines += [f"  {x['category']}: {x['suggestion']} ({x['total_samples']} samples)" for x in res["recommendations"]]
    lines += ["", NOTE, ""]
    return "\n".join(lines)


def run_repair(store: ArtifactStore, repair_id: str, investigation_id: str, config: RepairConfig | None = None,
               model=None, dataset=None) -> RepairRun:
    cfg = config or RepairConfig()
    if not _ID.match(repair_id):
        raise ConfigurationError(f"invalid repair id '{repair_id}'")
    relative = Path("repairs") / repair_id
    directory = store.resolve(relative)
    if directory.exists():
        raise ArtifactError(f"repair '{repair_id}' already exists")
    started, clock = datetime.now(timezone.utc), time.monotonic()

    inv = load_investigation(store, investigation_id)
    ev_id, an_id = inv["inputs"]["evaluation_id"], inv["inputs"]["analysis_id"]
    analysis = load_failure_analysis(store, an_id)
    ids_by_failure = {f["failure_id"]: population_ids(analysis, f["population_ref"]) for f in inv["failures"]}
    candidates, recommendations = build_candidates(inv, analysis, ids_by_failure, cfg)

    family = f"{repair_id}_family"
    names, baseline_fp, unchanged = [], None, True
    results, runnable = [], []
    if candidates:
        baseline = create_baseline(store, family, ev_id)
        names, baseline_fp = baseline.class_names, baseline.fingerprint
        can_images = model is not None and dataset is not None
        for cand in candidates:
            needs_images = cand["intervention"]["kind"] in ("image_transform", "preprocessing")
            if needs_images and not can_images:
                results.append({**cand, "status": "not_evaluated", "reason": "needs a model and a dataset", "decision": "not_evaluated"})
                continue
            population = cand["population"] or {}
            spec = {
                "name": f"repair_{cand['candidate_id']}",
                "hypothesis": {"claim": f"repair for {cand['mechanism_category']} (hypothesis {cand['hypothesis_id']})", "expected_direction": "improve"},
                "intervention": cand["intervention"], "population": population, "seed": cfg.seed,
                "alpha": cfg.criteria.alpha, "practical_threshold": cfg.criteria.min_target_improvement,
                "n_bootstrap": cfg.n_bootstrap, "n_permutations": cfg.n_permutations,
            }
            runnable.append((cand, parse_spec(spec)))
    if runnable:
        log.info("evaluating %d repair candidates", len(runnable))
        run = run_family(store, family, [s for _, s in runnable], model=model if model is not None else None,
                         dataset=dataset, analysis=analysis, alpha=cfg.criteria.alpha)
        outcomes = {o.result["spec"]["name"]: o for o in run.outcomes}
        evaluated = []
        for cand, spec in runnable:
            outcome = outcomes[spec.name]
            if outcome.status == "failed":
                results.append({**cand, "status": "failed", "reason": outcome.result["error"]["message"], "decision": "rejected"})
                continue
            paired = pd.read_parquet(outcome.directory / "paired.parquet")
            failure = next(f for f in inv["failures"] if f["failure_id"] == cand["failure_id"])
            own = failure["population_ref"].get("slice_id")
            tested = analysis.slices[analysis.slices["tested"].astype(bool) & ~analysis.slices["uses_model_output"].astype(bool)]
            tested = tested[tested["slice_id"] != own].sort_values(["support", "slice_id"], ascending=[False, True])
            important = [(sid, set(analysis.slice_sample_ids(sid))) for sid in tested["slice_id"].head(cfg.important_slices)]
            ev = _evaluate(paired, set(ids_by_failure[cand["failure_id"]]), names, important, cfg)
            evaluated.append((cand, outcome, ev))
        if evaluated:
            q = benjamini_hochberg([e["target"]["p"] for _, _, e in evaluated])
            for (cand, outcome, ev), qv in zip(evaluated, q, strict=True):
                ev["target"]["q"] = float(qv)
                checks = _checks(ev, float(qv), cfg.criteria)
                failed = [k["name"] for k in checks if not k["passed"]]
                results.append({**cand, "status": "evaluated", "experiment_id": outcome.experiment_id, **ev, "checks": checks,
                                "violations": failed, "decision": "accepted" if not failed else "rejected"})
        after = load_baseline(store, family)
        unchanged = after.fingerprint == baseline_fp
    results.sort(key=lambda r: (r["category"], r["grid_index"], r["candidate_id"]))

    accepted = [r for r in results if r["decision"] == "accepted"]
    best = min(accepted, key=lambda r: (-r["target"]["difference"], r["global"]["new_error_rate"], r["grid_index"], r["candidate_id"]), default=None)
    decision = "accepted" if best else ("rejected" if any(r["status"] == "evaluated" for r in results) else "no_repair_candidates")
    document = {
        "schema_version": 1, "repair_id": repair_id, "investigation_id": investigation_id, "decision": decision,
        "criteria": cfg.criteria.model_dump(mode="json"), "baseline": {"evaluation_id": ev_id, "family_id": family if candidates else None,
        "fingerprint": baseline_fp, "unchanged": unchanged}, "results": results, "recommendations": recommendations,
        "accepted_candidate_id": best["candidate_id"] if best else None, "note": NOTE,
        "summary": {"candidates": len(candidates), "evaluated": sum(r["status"] == "evaluated" for r in results),
                    "accepted": len(accepted), "rejected": sum(r["decision"] == "rejected" for r in results)},
    }
    directory.mkdir(parents=True)
    try:
        store.write_json(relative / "repair_candidates.json", {"investigation_id": investigation_id, "candidates": candidates, "recommendations": recommendations})
        store.write_json(relative / "repair_results.json", document)
        store.write_text(relative / "repair_report.txt", render_report(document))
        if best:
            meta = json.loads((store.root / "evaluations" / ev_id / "metadata.json").read_text())
            store.write_json(relative / "accepted_repair.json", {"candidate": best, "decision": "accepted"})
            store.write_json(relative / "repair_version.json", {
                "version_id": f"{repair_id}-v1", "repair_id": repair_id, "candidate_id": best["candidate_id"],
                "base_model_id": meta.get("model_id"), "base_evaluation_id": ev_id, "rule": best["deployment_rule"],
                "intervention": best["intervention"], "status": "accepted", "note": NOTE,
            })
        store.write_json(relative / "metadata.json", {
            "repair_id": repair_id, "timestamp": started.isoformat(), "duration_seconds": round(time.monotonic() - clock, 3),
            "fingerprint": hashlib.sha256(
                canonical_json(comparable(document)).encode()
            ).hexdigest()[:16], "versions": {"modellab": modellab.__version__, "python": platform.python_version()},
        })
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return RepairRun(repair_id, directory, document)
