from modellab.server import services


def make_scan_fn(ws, p):
    from modellab.audit import scan_csv, scan_folder

    rec = ws.get_dataset(p["dataset_id"])
    label = services.dataset_label(p["dataset_id"], p.get("subpath"))

    def scan(audit_cfg):
        if rec["kind"] == "csv":
            return scan_csv(rec["csv"], audit_cfg, root=rec["root"], dataset_id=label)
        return scan_folder(services.dataset_root(ws, rec, p.get("subpath")), audit_cfg, dataset_id=label)

    return scan


def _objects(ws, cache, p):
    if not p.get("model_id") or not p.get("dataset_id"):
        return None, None
    model, pre, _ = cache.get(ws, p["model_id"], p["device"])
    return model, services.build_dataset(ws, p["dataset_id"], p.get("subpath"), model.spec.class_names, pre)


def investigate(ws, cache, store, p):
    from modellab.investigation import InvestigationConfig, run_investigation

    cfg = InvestigationConfig.model_validate(p["config"])
    model, dataset = _objects(ws, cache, p)
    scan = make_scan_fn(ws, p) if p.get("dataset_id") else None
    run = run_investigation(
        store, p["investigation_id"], cfg, model, dataset, evaluation_id=p.get("evaluation_id"),
        audit_id=p.get("audit_id"), scan_fn=scan, analysis_id=p.get("analysis_id"),
    )
    rec = run.record
    return {
        "investigation_id": run.investigation_id,
        "num_failures": len(rec["failures"]),
        "num_hypotheses": len(rec["hypotheses"]),
        "num_experiments": len(rec["experiments"]),
        "conclusions": [
            {"failure_id": c["failure_id"], "status": c["status"], "mechanism": c["mechanism"], "strength": c["strength"]}
            for c in rec["conclusions"]
        ],
    }


def reproduce(ws, cache, store, p):
    from modellab.investigation import reproduce_investigation

    model, dataset = _objects(ws, cache, p)
    scan = make_scan_fn(ws, p) if p.get("dataset_id") else None
    return reproduce_investigation(store, p["investigation_id"], p["new_investigation_id"], model, dataset, scan)


def repair(ws, cache, store, p):
    from modellab.repair import RepairConfig, run_repair

    model, dataset = _objects(ws, cache, p)
    run = run_repair(store, p["repair_id"], p["investigation_id"], RepairConfig.model_validate(p["config"]), model, dataset)
    res = run.results
    accepted = next((r for r in res["results"] if r["candidate_id"] == res["accepted_candidate_id"]), None)
    return {
        "repair_id": run.repair_id, "decision": res["decision"], "summary": res["summary"],
        "accepted": accepted and {
            "candidate_id": accepted["candidate_id"], "description": accepted["description"],
            "target": accepted["target"], "global": accepted["global"],
        },
    }
