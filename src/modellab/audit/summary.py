from modellab.audit.schema import AuditReport

NOTE = (
    "Findings describe measured properties of the data. They do not by themselves "
    "establish the cause of any model behaviour."
)


def render_summary(report: AuditReport) -> str:
    inv = report.inventory
    lines = [
        "ModelLab dataset audit",
        f"dataset: {report.dataset_id}    layout: {report.scan['layout']}"
        f"    fingerprint: {report.fingerprint}",
        f"samples: {inv['total_samples']}  ("
        + ", ".join(f"{k} {v}" for k, v in inv["by_status"].items())
        + ")",
        f"excluded from scan: {report.scan['excluded_total']}",
        "classes: " + ", ".join(f"{k} {v['count']} ({v['percent']:.1f}%)" for k, v in inv["classes"].items()),
        "splits: " + (", ".join(f"{k} {v['count']}" for k, v in inv["splits"].items()) or "none"),
    ]
    by_sev = report.summary["findings_by_severity"]
    lines.append(
        "findings: " + ", ".join(f"{s} {by_sev.get(s, 0)}" for s in ("high", "medium", "low", "info"))
    )
    lines.append("")
    for f in report.findings:
        lines.append(f"[{f.severity.value.upper()}] {f.finding_id}: {f.title} (count {f.count})")
        lines.append(f"    {f.description}")
        if f.sample_ids:
            shown = ", ".join(f.sample_ids[:5])
            more = ", ..." if len(f.sample_ids) > 5 or f.evidence_truncated else ""
            lines.append(f"    examples: {shown}{more}")
        if f.affected_splits:
            lines.append(f"    splits: {', '.join(f.affected_splits)}")
    if not report.findings:
        lines.append("no findings")
    lines += ["", NOTE, ""]
    return "\n".join(lines)
