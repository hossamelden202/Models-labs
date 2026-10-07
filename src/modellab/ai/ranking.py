from modellab.ai.schemas import Ranked
from modellab.experiments.training_config import TrainingConfig

DEFAULT_EPOCHS = TrainingConfig().epochs
MEDIUM_AT = 2.0


def _epochs(cand):
    return next((c.value for c in cand.changes if c.path == "epochs"), None)


def rank_candidates(menu, analysis, experiments):
    primary = analysis["primary_target"]
    done = [e for e in experiments if e["status"] == "completed"]
    longest = max((e["changes"].get("epochs", DEFAULT_EPOCHS) for e in done), default=None)

    ranked = []
    for cand in menu:
        score, reasons = 0.0, []
        if cand.target_metric == primary:
            score += 1.0
            reasons.append(f"targets {primary}, the weaker overall metric")
        epochs = _epochs(cand)
        if epochs and longest is not None and longest < epochs:
            score += 1.5
            reasons.append(f"the longest past run trained {longest} epoch{'s' if longest != 1 else ''}, this trains {epochs}")
        paths = {c.path for c in cand.changes}
        for e in done:
            gain = (e.get("delta") or {}).get(cand.target_metric)
            if paths & set(e["changes"]) and gain is not None and gain > 0:
                score += 0.5
                reasons.append(f"{e['experiment_id']} changed the same setting and moved {cand.target_metric} by {gain:+.3f}")
                break
        if cand.candidate_id == "res_512":
            score -= 0.5
            reasons.append("larger input size, so it costs more to train")
        ranked.append(Ranked(
            candidate_id=cand.candidate_id, title=cand.title, score=round(score, 2),
            level="medium" if score >= MEDIUM_AT else "low",
            reasons=reasons or ["no specific evidence for or against"],
        ))
    ranked.sort(key=lambda r: -r.score)
    return ranked
