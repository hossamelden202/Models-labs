from modellab.ai.schemas import ProblemAnalysis, Weakness


def _num(value, default=0.0):
    return float(value) if isinstance(value, (int, float)) else default


def normalize_per_class(per_class):
    rows = []
    for key, v in per_class.items():
        tp = int(v.get("true_positives", v.get("tp", 0)) or 0)
        fp = int(v.get("false_positives", v.get("fp", 0)) or 0)
        fn = int(v.get("false_negatives", v.get("fn", 0)) or 0)
        rows.append({
            "class_name": str(v.get("class") or key),
            "class_id": v.get("class_id"),
            "support": int(v.get("support", tp + fn)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": _num(v.get("precision")),
            "recall": _num(v.get("recall")),
            "f1": _num(v.get("f1")),
        })
    rows.sort(key=lambda r: (r["class_id"] is None, r["class_id"] or 0, r["class_name"]))
    return rows


def per_class_lines(rows):
    return [
        f"{r['class_name']}: support {r['support']}, precision {r['precision']:.3f}, "
        f"recall {r['recall']:.3f}, f1 {r['f1']:.3f}, tp {r['tp']}, fp {r['fp']}, fn {r['fn']}"
        for r in rows
    ]


def analyze_detection(metrics, per_class, recall_floor=0.5, rare_share=0.05):
    rows = normalize_per_class(per_class)
    present = [r for r in rows if r["support"] > 0]
    total = sum(r["support"] for r in present)

    weaknesses = []
    for r in present:
        if r["recall"] < recall_floor:
            weaknesses.append(Weakness(
                class_name=r["class_name"], metric="recall", value=r["recall"],
                support=r["support"], note=f"{r['fn']} of {r['support']} objects missed",
            ))
        if r["fp"] > r["tp"]:
            weaknesses.append(Weakness(
                class_name=r["class_name"], metric="precision", value=r["precision"],
                support=r["support"], note=f"{r['fp']} false positives against {r['tp']} true positives",
            ))
    weaknesses.sort(key=lambda w: w.value)

    issues = []
    for r in rows:
        if r["support"] == 0:
            issues.append(
                f"{r['class_name']} has no ground-truth objects in the evaluation set, so its recall is "
                f"undefined and its {r['fp']} false positives cannot be judged"
            )
        elif total and r["support"] / total < rare_share:
            issues.append(f"{r['class_name']} is rare in the evaluation set ({r['support']} of {total} objects)")

    precision = _num(metrics.get("precision"))
    recall = _num(metrics.get("recall"))
    f1 = _num(metrics.get("f1"))
    summary = (
        f"Overall precision {precision:.3f}, recall {recall:.3f}, f1 {f1:.3f} "
        f"over {metrics.get('num_samples', 'n/a')} samples."
    )
    if weaknesses:
        w = weaknesses[0]
        summary += f" Weakest: {w.class_name} {w.metric} {w.value:.3f} ({w.note})."

    return ProblemAnalysis(
        summary=summary,
        weaknesses=weaknesses,
        data_issues=issues,
        primary_target="recall" if recall <= precision else "precision",
    )
