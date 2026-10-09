import re

from modellab.ai import tools
from modellab.ai.analysis import analyze_detection, normalize_per_class, per_class_lines
from modellab.ai.llm.provider import LLMError

MAX_CONTEXT = 7000
HISTORY_TURNS = 6

REPORT = re.compile(r"\b(report|export|write[- ]?up|download)\b", re.I)
PROPOSE = re.compile(
    r"\b(propose|suggest|recommend|another|different|something else|next experiment|what should i|"
    r"improve|fix|try|investigate|analy[sz]e|diagnos\w*)\b", re.I)
WHY_GENERAL = re.compile(r"\bwhy\b.*\b(detector|model|it)\b.*\b(weak|bad|poor|low|fail\w*|wrong|struggl\w*)", re.I)

ASK_SYSTEM = (
    "You are an assistant for an object detection model. Answer in two to six short sentences using only the "
    "context. Quote numbers exactly as written in the context. If the context does not contain the answer, "
    "say what is missing. Do not invent causes; call a guess a guess."
)


def route_intent(text, class_names=()):
    if REPORT.search(text):
        return "report"
    if PROPOSE.search(text):
        return "propose"
    mentions_class = any(c.lower() in text.lower() for c in class_names)
    if WHY_GENERAL.search(text) and not mentions_class:
        return "propose"
    return "ask"


_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?%|(?<![\w.])\d+\.\d+(?!\w|\.\d)")


def _context_values(context):
    values = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)(%?)", context):
        v = float(m.group(1))
        values.append(v)
        if m.group(2):
            values.append(v / 100)
    return values


def ungrounded_numbers(answer, context):
    known = _context_values(context)
    bad = []
    for token in _NUM.findall(answer):
        value = float(token.rstrip("%")) / 100 if token.endswith("%") else float(token)
        if not any(abs(value - k) <= 0.0015 for k in known) and token not in bad:
            bad.append(token)
    return bad[:5]


def _clip(text, n):
    return text if len(text) <= n else text[: n - 3] + "..."


def build_context(root, chat, question, knowledge, latest=None):
    family = chat["family_id"]
    baseline = tools.get_baseline(root, family)
    source, per_class = tools.failure_per_class(root, baseline, chat.get("analysis_id"))
    analysis = analyze_detection(baseline.get("metrics") or {}, per_class)

    parts = [f"BASELINE ({family}, failure data: {source}):", analysis.summary]
    parts += per_class_lines(normalize_per_class(per_class))
    if analysis.data_issues:
        parts += ["DATA ISSUES:"] + [f"- {i}" for i in analysis.data_issues]

    experiments = tools.list_experiments(root, family)
    parts.append("EXPERIMENTS RUN:" + ("" if experiments else " none"))
    for e in experiments[:8]:
        changes = ", ".join(f"{k}={v}" for k, v in sorted(e["changes"].items())) or "defaults"
        delta = ", ".join(f"{k} {v:+.3f}" for k, v in e["delta"].items() if isinstance(v, (int, float)))
        parts.append(f"- {e['experiment_id']} ({e['status']}): {changes}; change vs baseline: {delta or 'n/a'}")

    hits = knowledge.search(question, top_k=3, source_type="experiment", filters={"family_id": family})
    if hits:
        parts.append("RELATED EXPERIMENT NOTES:")
        parts += [_clip(h.doc.content, 400) for h in hits]

    audit = failure_report = None
    if chat.get("audit_id"):
        audit = tools.audit_digest(root, chat["audit_id"], 1500)[1]
    if chat.get("analysis_id"):
        failure_report = tools.analysis_digest(root, chat["analysis_id"], 1200)

    if latest and latest["result"].get("hypothesis"):
        h = latest["result"]["hypothesis"]
        parts.append(f"LATEST PROPOSAL ({latest['status']}): {h['claim']} Evidence level {h['evidence_level']}. "
                     + " ".join(h["evidence_reasons"]))
        if latest.get("finding"):
            parts.append("LATEST RESULT: " + latest["finding"]["summary"])

    text = "\n".join(parts)
    extras = []
    if audit:
        extras.append("DATASET AUDIT (raw digest):\n" + audit)
    if failure_report:
        extras.append("FAILURE ANALYSIS REPORT (raw digest):\n" + failure_report)
    full = text + ("\n" + "\n".join(extras) if extras else "")
    if len(full) > MAX_CONTEXT:
        extras = [_clip(x, 700) for x in extras]
        full = _clip(text, MAX_CONTEXT - 1500) + ("\n" + "\n".join(extras) if extras else "")
    return full, analysis


def history_messages(chat):
    turns = [m for m in chat["messages"][:-1] if m["kind"] == "text" and m.get("text")]
    return [{"role": m["role"], "content": _clip(m["text"], 300)} for m in turns[-HISTORY_TURNS:]]


def fallback_answer(analysis, reason):
    lines = [f"{reason} Here is what the data shows: {analysis.summary}"]
    lines += [f"{w.class_name} {w.metric} {w.value:.3f} ({w.note})." for w in analysis.weaknesses[:3]]
    lines += analysis.data_issues[:2]
    return " ".join(lines)


def answer_question(llm, chat, question, context, analysis):
    if llm is None:
        return fallback_answer(analysis, "The language model is not available."), False, None
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    try:
        answer = llm.generate_text(ASK_SYSTEM, prompt, history_messages(chat))
    except LLMError as exc:
        return fallback_answer(analysis, f"The language model could not answer ({str(exc)[:80]})."), False, None
    bad = ungrounded_numbers(answer, context)
    warning = ("These figures do not appear in the data I was given: " + ", ".join(bad) + ". Treat them with care.") if bad else None
    return answer, True, warning
