import hashlib
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from modellab.analysis.config import FailureConfig
from modellab.analysis.records import MODEL_OUTPUT
from modellab.analysis.stats import benjamini_hochberg, cohens_h, fisher_p, wilson_interval

FORBIDDEN = frozenset({"true_class", "predicted_class"})
TIERS = {
    "statistically_supported": 0,
    "significant_small_effect": 1,
    "exploratory": 2,
    "not_supported": 3,
    "supported_lower_error": 4,
    "descriptive": 5,
}
SLICE_COLUMNS = [
    "slice_id", "slice_type", "conditions", "n_conditions", "uses_model_output", "support",
    "support_fraction", "errors", "error_rate", "error_rate_ci_low", "error_rate_ci_high",
    "accuracy", "macro_precision", "macro_recall", "macro_f1", "mean_confidence",
    "mean_confidence_correct", "mean_confidence_error", "parent_error_rate",
    "complement_error_rate", "risk_difference", "risk_ratio", "cohens_h", "excess_errors",
    "p_value", "q_value", "tested", "evidence", "rank", "equivalent_slices",
]


def evidence_for(p: float, q: float, risk_difference: float, cfg: FailureConfig) -> str:
    if q <= cfg.alpha:
        if risk_difference >= cfg.min_risk_difference:
            return "statistically_supported"
        if risk_difference <= -cfg.min_risk_difference:
            return "supported_lower_error"
        return "significant_small_effect"
    return "exploratory" if p <= cfg.alpha else "not_supported"


def _sid(key) -> str:
    return " & ".join(f"{f}={v}" for f, v in key)


def _f(x):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) or np.isinf(x) else round(x, 6)


def _prf(y, pred, k):
    true_c = np.bincount(y, minlength=k)
    pred_c = np.bincount(pred, minlength=k)
    tp = np.bincount(y[y == pred], minlength=k)
    present = true_c > 0
    prec = np.divide(tp, pred_c, out=np.zeros(k), where=pred_c > 0)
    rec = np.divide(tp, true_c, out=np.zeros(k), where=true_c > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros(k), where=denom > 0)
    return prec[present].mean(), rec[present].mean(), f1[present].mean()


def _metrics(mask, ctx):
    y, pred, err, conf = (ctx[name][mask] for name in ("y", "pred", "err", "conf"))
    n, support = ctx["n"], int(mask.sum())
    errors = int(err.sum())
    rest_n, rest_e = n - support, ctx["E"] - errors
    rate = errors / support
    rest_rate = rest_e / rest_n if rest_n else None
    low, high = wilson_interval(errors, support)
    prec, rec, f1 = _prf(y, pred, ctx["k"])
    risk_ratio = rate / rest_rate if rest_rate else None
    return {
        "support": support,
        "support_fraction": _f(support / n),
        "errors": errors,
        "error_rate": _f(rate),
        "error_rate_ci_low": _f(low),
        "error_rate_ci_high": _f(high),
        "accuracy": _f(1 - rate),
        "macro_precision": _f(prec),
        "macro_recall": _f(rec),
        "macro_f1": _f(f1),
        "mean_confidence": _f(conf.mean()),
        "mean_confidence_correct": _f(conf[~err].mean()) if (~err).any() else None,
        "mean_confidence_error": _f(conf[err].mean()) if err.any() else None,
        "parent_error_rate": _f(ctx["E"] / n),
        "complement_error_rate": _f(rest_rate),
        "risk_difference": _f(rate - rest_rate) if rest_rate is not None else None,
        "risk_ratio": _f(risk_ratio),
        "cohens_h": _f(cohens_h(rate, rest_rate)) if rest_rate is not None else None,
        "excess_errors": _f(errors - support * ctx["E"] / n),
    }


def _slice_type(key) -> str:
    features = {f for f, _ in key}
    if len(key) == 1:
        return "single_feature"
    if features & FORBIDDEN:
        return "class_conditional"
    return "pairwise" if len(key) == 2 else "subgroup"


def empty_slices() -> pd.DataFrame:
    return pd.DataFrame(columns=SLICE_COLUMNS)


def discover_slices(samples: pd.DataFrame, labels: pd.DataFrame, class_names, cfg: FailureConfig):
    n = len(samples)
    ctx = {
        "y": samples["true_label"].to_numpy(dtype=np.int64),
        "pred": samples["predicted_label"].to_numpy(dtype=np.int64),
        "err": samples["is_error"].to_numpy(dtype=bool),
        "conf": samples["confidence"].to_numpy(dtype=np.float64),
        "k": len(class_names),
        "n": n,
    }
    ctx["E"] = int(ctx["err"].sum())
    overall = ctx["E"] / n

    conds = []
    stats = {"generated": 0, "pruned": 0, "dropped": 0}
    for feature in sorted(labels.columns):
        column = labels[feature].to_numpy(dtype=object)
        for value in sorted(set(column)):
            mask = column == value
            size = int(mask.sum())
            if size < cfg.min_support or n - size < cfg.min_support:
                stats["pruned"] += 1
                continue
            conds.append((feature, value, mask))

    seen: dict[bytes, str] = {}
    kept: list = []
    equivalent = defaultdict(list)
    truncated = False

    def consider(key, mask):
        stats["generated"] += 1
        size = int(mask.sum())
        if size < cfg.min_support or n - size < cfg.min_support:
            stats["pruned"] += 1
            return
        signature = hashlib.blake2b(np.packbits(mask).tobytes(), digest_size=16).digest()

        # Keep explicit missing-value slices even when their mask is
        # equivalent to another feature's missing-value slice. Missingness
        # is feature-specific information that should remain inspectable.
        has_missing_value = any(str(value) == "<missing>" for _, value in key)

        if signature in seen and not has_missing_value:
            equivalent[seen[signature]].append(_sid(key))
            stats["dropped"] += 1
            return

        if signature not in seen:
            seen[signature] = _sid(key)

        kept.append((key, mask))

    for feature, value, mask in conds:
        consider(((feature, value),), mask)

    if cfg.max_depth >= 2:
        for i in range(len(conds)):
            if stats["generated"] >= cfg.max_candidates:
                truncated = True
                break
            fi, vi, mi = conds[i]
            for j in range(i + 1, len(conds)):
                fj, vj, mj = conds[j]
                if fi == fj or {fi, fj} == FORBIDDEN:
                    continue
                consider(((fi, vi), (fj, vj)), mi & mj)

    if cfg.max_depth >= 3 and kept and not truncated:
        err = ctx["err"]
        parents = [
            (int(err[m].sum()) - int(m.sum()) * overall, _sid(key), key, m)
            for key, m in kept
            if len(key) == 2
        ]
        parents.sort(key=lambda t: (-t[0], t[1]))
        for _, _, key, mask in parents[: cfg.beam_width]:
            used = {f for f, _ in key}
            for feature, value, cmask in conds:
                if feature in used or FORBIDDEN <= (used | {feature}):
                    continue
                consider(tuple(sorted(key + ((feature, value),))), mask & cmask)

    rows = []
    for key, mask in kept:
        row = _metrics(mask, ctx)
        errors, support = row["errors"], row["support"]
        rest_n, rest_e = n - support, ctx["E"] - errors
        row["p_value"] = fisher_p(errors, support - errors, rest_e, rest_n - rest_e)
        sid = _sid(key)
        row.update(
            slice_id=sid,
            slice_type=_slice_type(key),
            conditions=json.dumps([list(c) for c in key]),
            n_conditions=len(key),
            uses_model_output=any(f in MODEL_OUTPUT for f, _ in key),
            tested=True,
            equivalent_slices=json.dumps(sorted(equivalent.get(sid, []))),
        )
        rows.append(row)

    if rows:
        q = benjamini_hochberg(np.array([r["p_value"] for r in rows]))
        for row, qv in zip(rows, q, strict=True):
            row["q_value"] = float(qv)
            row["evidence"] = evidence_for(row["p_value"], row["q_value"], row["risk_difference"], cfg)

    def sort_key(r):
        if cfg.rank_by == "q_value":
            metric = r["q_value"]
        else:
            metric = -r[cfg.rank_by]
        return (TIERS[r["evidence"]], metric, r["slice_id"])

    rows.sort(key=sort_key)
    for position, row in enumerate(rows, start=1):
        row["rank"] = position

    descriptive = []
    names = list(class_names)
    pairs = Counter(zip(ctx["y"][ctx["err"]].tolist(), ctx["pred"][ctx["err"]].tolist(), strict=True))
    top = sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[: cfg.top_confusions]
    for (a, b), _ in top:
        mask = (ctx["y"] == a) & (ctx["pred"] == b)
        key = (("true_class", names[a]), ("predicted_class", names[b]))
        row = _metrics(mask, ctx)
        row.update(slice_id=f"confusion:{names[a]}->{names[b]}", slice_type="top_confusion",
                   conditions=json.dumps([list(c) for c in key]), n_conditions=2,
                   uses_model_output=True, tested=False, evidence="descriptive",
                   equivalent_slices="[]")
        descriptive.append(row)
    mask = samples["is_high_confidence_error"].to_numpy(dtype=bool)
    if mask.any():
        row = _metrics(mask, ctx)
        row.update(slice_id="high_confidence_errors", slice_type="high_confidence_error",
                   conditions=json.dumps([["is_high_confidence_error", "True"]]), n_conditions=1,
                   uses_model_output=True, tested=False, evidence="descriptive", equivalent_slices="[]")
        descriptive.append(row)

    frame = pd.DataFrame(rows + descriptive, columns=SLICE_COLUMNS) if rows or descriptive else empty_slices()
    info = {
        "num_candidates_generated": stats["generated"],
        "num_pruned_min_support": stats["pruned"],
        "num_duplicates_dropped": stats["dropped"],
        "num_tested": len(rows),
        "truncated": truncated,
        "correction": "benjamini_hochberg",
        "alpha": cfg.alpha,
        "max_depth": cfg.max_depth,
    }
    return frame, info
