METRICS = ("f1", "precision", "recall", "mean_iou")


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers, rows):
    if not rows:
        return []
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return out + [""]


def _num(value):
    return f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"


def build_report(record, chat=None):
    res = record["result"]
    a = res.get("analysis")
    ctx = res.get("context") or {}
    lines = [
        "# ModelLab AI research report", "",
        f"- Research: `{record['id']}`",
        f"- Family: `{record['family_id']}`",
        f"- Created: {record['created_at']}",
        f"- Status: {record['status']}",
        f"- Question: {record['question']}",
        f"- Failure data: `{res.get('failure_source')}`",
    ]
    if ctx.get("audit_id"):
        lines.append(f"- Dataset audit: `{ctx['audit_id']}`")
    lines.append("")

    bm = res.get("baseline_metrics") or {}
    if bm:
        lines += ["## Baseline", ""]
        lines += _table(["precision", "recall", "f1", "samples"],
                        [[_num(bm.get("precision")), _num(bm.get("recall")), _num(bm.get("f1")), bm.get("num_samples", "n/a")]])

    if not a:
        lines += ["## Result", "", "No experiment could be proposed.", ""]
        lines += [f"- {n}" for n in res.get("notes", [])]
        return "\n".join(lines) + "\n"

    lines += ["## Analysis", "", a["summary"], ""]
    lines += _table(["class", "metric", "value", "support", "note"],
                    [[w["class_name"], w["metric"], _num(w["value"]), w["support"], w["note"]] for w in a["weaknesses"]])
    if a["data_issues"]:
        lines += ["### Data notes", ""] + [f"- {i}" for i in a["data_issues"]] + [""]
    if res.get("retrieved"):
        lines += ["## Past experiments consulted", ""]
        lines += _table(["experiment", "similarity"], [[r["title"], _num(r["score"])] for r in res["retrieved"]])

    hyp, prop, val = res["hypothesis"], res["proposal"], res["validation"]
    how = "language model" if res.get("llm_used") else "rule (language model unavailable or unusable)"
    lines += ["## Hypothesis", "", hyp["claim"], "", hyp["rationale"], "",
              f"Chosen by: {how}. Evidence level: **{hyp['evidence_level']}**.", ""]
    if hyp["evidence"]:
        lines += ["Evidence:", ""] + [f"- {e}" for e in hyp["evidence"]] + [""]
    if hyp["evidence_reasons"]:
        lines += ["Why this level:", ""] + [f"- {r}" for r in hyp["evidence_reasons"]] + [""]
    lines += ["## Options considered", ""]
    lines += _table(["option", "score", "level", "why"],
                    [[r["candidate_id"], r["score"], r["level"], "; ".join(r["reasons"])] for r in res.get("ranking", [])])

    lines += ["## Proposed experiment", "", f"`{prop['name']}`: {prop['expected_effect']}.", ""]
    lines += _table(["setting", "value"], [[c["path"], c["value"]] for c in prop["changes"]])
    if val["valid"]:
        lines += [f"Validation: passed (`{val['experiment_id']}`).", ""]
    else:
        lines += ["Validation: failed. " + "; ".join(val["errors"]), ""]

    ap = record.get("approval")
    if ap:
        lines += ["## Execution", "", f"- Approved: {ap['at']}", f"- Job: `{ap['job_id']}`",
                  f"- Model: `{ap['model_id']}`, dataset: `{ap['dataset_id']}`"]
        if ap.get("control_experiment_id"):
            lines.append(f"- Control run: `{ap['control_experiment_id']}`" + (" (run with this experiment)" if ap.get("control_runs") else " (existing)"))
        lines.append("")
    elif record["status"] == "rejected":
        lines += ["## Execution", "", "The proposal was rejected and not run.", ""]

    f = record.get("finding")
    if f:
        lines += ["## Finding", "", f"**{f['verdict']}**: {f['hypothesis_status']}.", "", f["summary"], ""]
        if f.get("trained"):
            ref_name = "control" if f.get("basis") == "control" else "baseline"
            lines += _table(["metric", ref_name, "trained", "change"],
                            [[m, _num(f["reference"].get(m)), _num(f["trained"].get(m)), _num(f["delta"].get(m))] for m in METRICS])
        lines += ["Caveats:", ""] + [f"- {c}" for c in f["caveats"]] + [""]

    if res.get("notes"):
        lines += ["## Notes", ""] + [f"- {n}" for n in res["notes"]] + [""]

    if chat:
        turns = [m for m in chat["messages"] if m["kind"] == "text" and m.get("text")]
        if turns:
            lines += ["## Conversation", ""]
            for m in turns:
                who = "You" if m["role"] == "user" else "Assistant"
                lines += [f"**{who}:** {m['text']}", ""]
    return "\n".join(lines).rstrip() + "\n"
