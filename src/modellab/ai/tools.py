import json
import re
from pathlib import Path

from modellab.ai.digest import digest_json
from modellab.ai.knowledge.ingestion import config_changes

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DELTAS = ("f1", "precision", "recall")


class ToolError(Exception):
    pass


def check_name(value, what):
    if not isinstance(value, str) or not NAME.match(value):
        raise ToolError(f"invalid {what}: {value!r}")
    return value


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def get_baseline(root, family_id):
    check_name(family_id, "family_id")
    doc = _read(Path(root) / "experiments" / family_id / "baseline.json")
    if doc is None:
        raise ToolError(f"no readable baseline for family {family_id!r}")
    task = (
        doc.get("task")
        or (doc.get("metadata") or {}).get("task")
        or ((doc.get("metadata") or {}).get("model") or {}).get("task")
    )
    if task != "detection":
        raise ToolError("the advisor supports detection families only")
    return doc


def list_experiments(root, family_id):
    check_name(family_id, "family_id")
    out = []
    for path in sorted((Path(root) / "experiments" / family_id).glob("exp_*/result.json")):
        result = _read(path)
        if result is None:
            continue
        comp = result.get("comparison") or {}
        delta = comp.get("delta") or {}
        out.append({
            "experiment_id": path.parent.name,
            "status": result.get("status"),
            "changes": config_changes((result.get("training") or {}).get("config") or {}),
            "delta": {k: delta.get(k) for k in DELTAS if k in delta},
        })
    return out


def get_experiment(root, family_id, experiment_id):
    check_name(family_id, "family_id")
    check_name(experiment_id, "experiment_id")
    doc = _read(Path(root) / "experiments" / family_id / experiment_id / "result.json")
    if doc is None:
        raise ToolError(f"no readable result for {experiment_id!r}")
    return doc


def failure_per_class(root, baseline, analysis_id=None):
    root = Path(root)
    if analysis_id:
        check_name(analysis_id, "analysis_id")
        meta = _read(root / "failure_analyses" / analysis_id / "metadata.json")
        per_class = _read(root / "failure_analyses" / analysis_id / "per_class.json")
        if not meta or not per_class:
            raise ToolError(f"no readable failure analysis {analysis_id!r}")
        if meta.get("task") != "detection":
            raise ToolError("the advisor supports detection failure analyses only")
        return f"failure_analysis:{analysis_id}", per_class

    evaluation_id = baseline.get("evaluation_id")
    found = None
    for meta_path in sorted(root.glob("failure_analyses/*/metadata.json")):
        meta = _read(meta_path)
        if meta and meta.get("evaluation_id") == evaluation_id and meta.get("task") == "detection":
            per_class = _read(meta_path.parent / "per_class.json")
            if per_class:
                found = (f"failure_analysis:{meta_path.parent.name}", per_class)
    if found:
        return found
    per_class = (baseline.get("metrics") or {}).get("per_class")
    if not per_class:
        raise ToolError("no failure analysis and no per-class baseline metrics")
    return "baseline_metrics", per_class


def list_families(root):
    out = []
    for path in sorted(Path(root).glob("experiments/*/baseline.json")):
        doc = _read(path)
        if doc and doc.get("task") == "detection":
            meta = doc.get("metadata") or {}
            out.append({
                "family_id": path.parent.name,
                "model_id": meta.get("model_id"),
                "dataset_id": meta.get("dataset_id"),
                "num_experiments": len(list(path.parent.glob("exp_*/result.json"))),
            })
    return out


def list_audits(root):
    out = []
    for folder in sorted((Path(root) / "audits").glob("*"), reverse=True):
        if folder.is_dir():
            meta = _read(folder / "metadata.json") or {}
            out.append({"audit_id": folder.name, "dataset_id": meta.get("dataset_id"), "timestamp": meta.get("timestamp")})
    return out


def list_analyses(root, evaluation_id=None):
    out = []
    for path in sorted(Path(root).glob("failure_analyses/*/metadata.json"), reverse=True):
        meta = _read(path)
        if meta and meta.get("task") == "detection":
            out.append({
                "analysis_id": path.parent.name,
                "evaluation_id": meta.get("evaluation_id"),
                "matches_baseline": bool(evaluation_id) and meta.get("evaluation_id") == evaluation_id,
            })
    out.sort(key=lambda a: not a["matches_baseline"])
    return out


def find_control(root, family_id):
    check_name(family_id, "family_id")
    for path in sorted((Path(root) / "experiments" / family_id).glob("exp_*/result.json")):
        result = _read(path)
        if result and result.get("status") == "completed":
            if not config_changes((result.get("training") or {}).get("config") or {}):
                return path.parent.name
    return None


def audit_digest(root, audit_id, max_chars=1500):
    check_name(audit_id, "audit_id")
    folder = Path(root) / "audits" / audit_id
    if not folder.is_dir():
        raise ToolError(f"no audit {audit_id!r}")
    names = ["report.json", "summary.json"] + [
        p.name for p in sorted(folder.glob("*.json")) if p.name not in ("report.json", "summary.json", "metadata.json")
    ] + ["metadata.json"]
    for name in names:
        doc = _read(folder / name)
        if doc is not None:
            return name, digest_json(doc, max_chars)
    raise ToolError(f"audit {audit_id!r} has no readable json files")


def analysis_digest(root, analysis_id, max_chars=1200):
    check_name(analysis_id, "analysis_id")
    doc = _read(Path(root) / "failure_analyses" / analysis_id / "report.json")
    return digest_json(doc, max_chars) if doc is not None else None
