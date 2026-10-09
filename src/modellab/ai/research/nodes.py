from modellab.ai import tools
from modellab.ai.analysis import analyze_detection
from modellab.ai.catalog import build_menu, get_candidate
from modellab.ai.knowledge.ingestion import ingest_all, tried_changes
from modellab.ai.llm.provider import LLMError
from modellab.ai.ranking import rank_candidates
from modellab.ai.schemas import Candidate, Change, ExperimentProposal, Hypothesis, LLMChoice
from modellab.ai.validation import validate_changes

TOP = 3
SYSTEM = (
    "You advise on improving an object detection model. You are given measured results, past experiments "
    "and a short ranked list of allowed experiments. Pick exactly one by its candidate_id. Prefer the first "
    "one unless the numbers argue against it. The claim must name a weak class from the results and may use "
    "only numbers that appear in them. Do not invent causes."
)


def _trace(state, name):
    return state.get("trace", []) + [name]


def _prompt(state, ranked):
    a = state["analysis"]
    lines = ["Measured results:", a["summary"], "", "Weak classes:"]
    lines += [f"- {w['class_name']} {w['metric']} {w['value']:.3f} ({w['note']})" for w in a["weaknesses"]] or ["- none"]
    lines += ["", "Data issues:"] + ([f"- {i}" for i in a["data_issues"]] or ["- none"])
    if state.get("audit_digest"):
        lines += ["", "Dataset audit (raw digest):", state["audit_digest"]]
    lines += ["", "Past experiments:"]
    for r in state["retrieved"]:
        lines.append(f"[{r['title']}]\n{r['content'][:500]}")
    if not state["retrieved"]:
        lines.append("none")
    by_id = {c["candidate_id"]: c for c in state["menu"]}
    lines += ["", "Allowed experiments, best first:"]
    for i, r in enumerate(ranked, 1):
        c = by_id[r["candidate_id"]]
        lines.append(
            f"{i}. {c['candidate_id']}: {c['title']} (targets {c['target_metric']}). "
            f"Why ranked here: {'; '.join(r['reasons'])}. Mechanism: {c['rationale']}"
        )
    lines += ["", f"Question: {state['request']}", "Return candidate_id, claim and rationale."]
    return "\n".join(lines)


def _evidence(analysis, retrieved, target):
    out = [f"{w['class_name']} {w['metric']} {w['value']:.3f} ({w['note']})" for w in analysis["weaknesses"][:3]]
    for r in retrieved:
        line = next((l for l in r["content"].splitlines() if l.startswith(f"{target}:")), None)
        if line:
            out.append(f"{r['title']} -> {line}")
    return out


def make_nodes(root, llm, knowledge):
    def collect_context(state):
        family = state["family_id"]
        baseline = tools.get_baseline(root, family)
        source, per_class = tools.failure_per_class(root, baseline, state.get("analysis_id"))
        exclude = set(state.get("exclude") or [])
        menu = [c for c in build_menu(tried_changes(root, family)) if c.candidate_id not in exclude]
        audit = tools.audit_digest(root, state["audit_id"])[1] if state.get("audit_id") else None
        ingest_all(root, knowledge)
        notes = list(state.get("notes", []))
        if not menu:
            notes.append("every option in the advisor's menu was already run for this family or already proposed in this chat")
        return {
            "audit_digest": audit,
            "baseline": baseline,
            "experiments": tools.list_experiments(root, family),
            "failure_source": source,
            "failure_per_class": per_class,
            "menu": [c.model_dump() for c in menu],
            "notes": notes,
            "trace": _trace(state, "collect_context"),
        }

    def analyze_problem(state):
        analysis = analyze_detection(state["baseline"].get("metrics") or {}, state["failure_per_class"])
        return {"analysis": analysis.model_dump(), "trace": _trace(state, "analyze_problem")}

    def retrieve_knowledge(state):
        a = state["analysis"]
        classes = " ".join(dict.fromkeys(w["class_name"] for w in a["weaknesses"][:3]))
        query = f"{a['primary_target']} low for {classes} object detection training change effect"
        hits = knowledge.search(query, top_k=3, source_type="experiment", filters={"family_id": state["family_id"]})
        retrieved = [
            {"id": h.doc.id, "title": h.doc.title, "score": round(h.score, 4), "content": h.doc.content}
            for h in hits
        ]
        return {"retrieved": retrieved, "trace": _trace(state, "retrieve_knowledge")}

    def generate_hypothesis(state):
        menu = [Candidate.model_validate(c) for c in state["menu"]]
        ranking = [r.model_dump() for r in rank_candidates(menu, state["analysis"], state["experiments"])]
        top = ranking[:TOP]
        ids = [r["candidate_id"] for r in top]
        weak = sorted({w["class_name"] for w in state["analysis"]["weaknesses"]})
        notes = list(state.get("notes", []))

        def check(choice):
            if choice.candidate_id not in ids:
                return f"candidate_id must be one of {ids}"
            if weak and not any(name.lower() in choice.claim.lower() for name in weak):
                return f"the claim must name a weak class from the results, one of {weak}"
            return None

        try:
            choice = llm.generate_structured(SYSTEM, _prompt(state, top), LLMChoice, check=check)
            used = True
        except LLMError as exc:
            cand = get_candidate(menu, ids[0])
            choice = LLMChoice(
                candidate_id=cand.candidate_id,
                claim=f"{cand.title} may improve {cand.target_metric}",
                rationale=cand.rationale,
            )
            used = False
            notes.append(f"LLM output unusable, took the top-ranked option by rule: {exc}")

        cand = get_candidate(menu, choice.candidate_id)
        entry = next(r for r in ranking if r["candidate_id"] == cand.candidate_id)
        hypothesis = Hypothesis(
            claim=choice.claim,
            rationale=choice.rationale,
            evidence=_evidence(state["analysis"], state["retrieved"], cand.target_metric),
            target_metric=cand.target_metric,
            evidence_level=entry["level"],
            evidence_reasons=entry["reasons"],
        )
        return {
            "choice": choice.model_dump(),
            "hypothesis": hypothesis.model_dump(),
            "ranking": ranking,
            "llm_used": used,
            "notes": notes,
            "trace": _trace(state, "generate_hypothesis"),
        }

    def generate_experiment(state):
        cand = get_candidate([Candidate.model_validate(c) for c in state["menu"]], state["choice"]["candidate_id"])
        proposal = ExperimentProposal(
            name=f"advisor_{cand.candidate_id}",
            candidate_id=cand.candidate_id,
            changes=[Change(path=c.path, value=c.value) for c in cand.changes],
            expected_effect=f"{cand.target_metric} should increase",
            reason=state["hypothesis"]["claim"],
        )
        return {"proposal": proposal.model_dump(), "trace": _trace(state, "generate_experiment")}

    def validate_experiment(state):
        p = state["proposal"]
        result = validate_changes([Change(**c) for c in p["changes"]], p["name"], p["reason"])
        return {"validation": result.model_dump(), "trace": _trace(state, "validate_experiment")}

    return {
        "collect_context": collect_context,
        "analyze_problem": analyze_problem,
        "retrieve_knowledge": retrieve_knowledge,
        "generate_hypothesis": generate_hypothesis,
        "generate_experiment": generate_experiment,
        "validate_experiment": validate_experiment,
    }
