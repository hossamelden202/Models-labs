import json
import re
from pathlib import Path

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
    if doc.get("task") != "detection":
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


def failure_per_class(root, baseline):
    evaluation_id = baseline.get("evaluation_id")
    found = None
    for meta_path in sorted((Path(root) / "failure_analyses").glob("*/metadata.json")):
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
