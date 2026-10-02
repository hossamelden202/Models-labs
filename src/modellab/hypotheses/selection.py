import numpy as np

from modellab.experiments.spec import canonical_json, sha256_text

IMAGE_KINDS = ("image_transform", "preprocessing")


def control_ids(samples, ids, n):
    taken = set(ids)
    pool = sorted(samples.loc[~samples["sample_id"].isin(taken) & samples["correct"].astype(bool), "sample_id"])
    if len(pool) <= n:
        return pool
    picks = sorted(set(np.linspace(0, len(pool) - 1, n).round().astype(int).tolist()))
    return [pool[i] for i in picks]


def build_candidates(failures, hypotheses, samples, analysis_id, cfg, allow_images):
    by_failure = {f.target.failure_id: f for f in failures}
    candidates, untestable = {}, {}

    for h in hypotheses:
        kept = 0
        failure = by_failure[h.failure_id]

        for t in h.required_tests:
            if t.intervention["kind"] in IMAGE_KINDS and not allow_images:
                continue

            target = failure.target

            if t.population == "failure":
                # Slice populations must remain declarative. The experiment
                # runner resolves slice_ids + analysis_id to concrete samples.
                if target.slice_id and analysis_id:
                    population = {
                        "slice_ids": [target.slice_id],
                        "analysis_id": analysis_id,
                    }
                else:
                    population = {"sample_ids": failure.ids}

                cost = target.support

            else:
                ids = control_ids(samples, failure.ids, cfg.control_size)
                if not ids:
                    continue

                population = {"sample_ids": ids}
                cost = len(ids)

            key = sha256_text(
                canonical_json(
                    {
                        "i": t.intervention,
                        "p": population,
                    }
                )
            )[:16]

            entry = candidates.setdefault(
                key,
                {
                    "key": key,
                    "intervention": t.intervention,
                    "population": population,
                    "cost": cost,
                    "tests": [],
                    "claims": set(),
                },
            )

            entry["tests"].append(
                {
                    "hypothesis_id": h.hypothesis_id,
                    "test_id": t.test_id,
                    "expected_effect": t.expected_effect,
                    "failure_id": h.failure_id,
                }
            )
            entry["claims"].add(h.mechanism)
            kept += 1

        if not kept:
            if h.required_tests:
                untestable[h.hypothesis_id] = (
                    "its tests need a model and a dataset to re-run inference"
                )
            else:
                untestable[h.hypothesis_id] = (
                    h.note
                    or "no available intervention distinguishes this explanation"
                )

    return candidates, untestable


def _spec(candidate, cfg) -> dict:
    effects = {t["expected_effect"] for t in candidate["tests"]}
    return {
        "name": "plan_" + candidate["key"][:10],
        "hypothesis": {"claim": "; ".join(sorted(candidate["claims"])), "expected_direction": next(iter(effects)) if len(effects) == 1 else None},
        "intervention": candidate["intervention"],
        "population": candidate["population"],
        "seed": cfg.seed, "alpha": cfg.alpha, "practical_threshold": cfg.practical_threshold,
    }


def select_experiments(failures, hypotheses, samples, analysis_id, cfg, allow_images) -> dict:
    candidates, untestable = build_candidates(failures, hypotheses, samples, analysis_id, cfg, allow_images)
    pairs = sorted({(a.hypothesis_id, b) for a in hypotheses for b in a.competes_with if a.hypothesis_id < b})

    def effects(c):
        return {t["hypothesis_id"]: t["expected_effect"] for t in c["tests"]}

    def separates(c):
        eff, out = effects(c), set()
        for a, b in pairs:
            in_a, in_b = a in eff, b in eff
            if in_a != in_b or (in_a and eff[a] != eff[b]):
                out.add((a, b))
        return out

    remaining, selected = dict(candidates), []
    covered, separated = set(), set()
    while remaining and len(selected) < cfg.max_experiments:
        best = None
        for key in sorted(remaining):
            c = remaining[key]
            new_cover = set(effects(c)) - covered
            new_sep = separates(c) - separated
            gain = 2 * len(new_sep) + len(new_cover)
            score = (-gain, c["cost"], key)
            if gain > 0 and (best is None or score < best[0]):
                best = (score, key, gain, new_sep)
        if best is None:
            break
        _, key, gain, new_sep = best
        c = remaining.pop(key)
        covered |= set(effects(c))
        separated |= new_sep
        selected.append({
            "experiment_key": key, "name": "plan_" + key[:10], "spec": _spec(c, cfg),
            "tests": sorted(c["tests"], key=lambda t: (t["hypothesis_id"], t["test_id"])),
            "cost": c["cost"], "gain": gain,
            "rationale": f"tests {len(effects(c))} hypotheses and separates {len(new_sep)} previously unseparated competing pairs on {c['cost']} samples",
        })
    return {
        "plan_version": 1, "budget": cfg.max_experiments, "allow_image_experiments": allow_images,
        "competing_pairs": len(pairs), "separated_pairs": len(separated), "selected": selected,
        "not_selected": [
            {"experiment_key": k, "reason": "adds no new coverage or separation, or the budget is used"}
            for k in sorted(remaining)
        ],
        "untestable_hypotheses": [{"hypothesis_id": k, "reason": v} for k, v in sorted(untestable.items())],
    }
