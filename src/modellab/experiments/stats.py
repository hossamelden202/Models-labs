import numpy as np


def mcnemar_exact(b: int, c: int) -> float:
    from scipy.stats import binomtest

    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def bootstrap_mean_ci(d, n_boot: int, seed: int, level: float = 0.95):
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    if n == 0:
        return None, None
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    chunk = max(1, min(n_boot, 2_000_000 // n))
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(m, n))
        means[done : done + m] = d[idx].mean(axis=1)
        done += m
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)


def sign_flip_p(d, n_perm: int, seed: int) -> float:
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    chunk = max(1, min(n_perm, 2_000_000 // n))
    count, done = 0, 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(m, n))
        count += int((np.abs((signs * d).mean(axis=1)) >= observed - 1e-12).sum())
        done += m
    return float((count + 1) / (n_perm + 1))


def score(name: str, y, pred, k: int) -> float:
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    true_c = cm.sum(axis=1).astype(float)
    pred_c = cm.sum(axis=0).astype(float)
    if name == "accuracy":
        return float(tp.sum() / len(y))
    if name == "balanced_accuracy":
        present = true_c > 0
        return float((tp[present] / true_c[present]).mean())
    prec = np.divide(tp, pred_c, out=np.zeros(k), where=pred_c > 0)
    rec = np.divide(tp, true_c, out=np.zeros(k), where=true_c > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros(k), where=denom > 0)
    return float(f1.mean())


def bootstrap_diff_ci(name, y, base_pred, int_preds, k, n_boot, seed, level=0.95):
    n = len(y)
    rng = np.random.default_rng(seed + 7)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        base = score(name, yb, base_pred[idx], k)
        diffs[i] = np.mean([score(name, yb, p[idx], k) for p in int_preds]) - base
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(diffs, [tail, 100 - tail])
    return float(low), float(high)


def paired_summary(base_ok, int_oks, n_boot: int, n_perm: int, seed: int) -> dict:
    base_ok = np.asarray(base_ok, dtype=bool)
    n = len(base_ok)
    d = np.mean([np.asarray(o, dtype=float) for o in int_oks], axis=0) - base_ok.astype(float)
    mean = float(d.mean())
    low, high = bootstrap_mean_ci(d, n_boot, seed)
    std = float(d.std(ddof=1)) if n > 1 else 0.0
    out = {
        "n": int(n),
        "accuracy_difference": mean,
        "ci_low": low,
        "ci_high": high,
        "permutation_p": sign_flip_p(d, n_perm, seed + 1),
        "cohens_dz": mean / std if std > 0 else None,
        "discordant_worsened": None,
        "discordant_improved": None,
        "mcnemar_p": None,
        "odds_ratio": None,
    }
    if len(int_oks) == 1:
        ok = np.asarray(int_oks[0], dtype=bool)
        b = int((base_ok & ~ok).sum())
        c = int((~base_ok & ok).sum())
        out.update(
            discordant_worsened=b,
            discordant_improved=c,
            mcnemar_p=mcnemar_exact(b, c),
            odds_ratio=float((c + 0.5) / (b + 0.5)),
            primary_test="mcnemar_exact",
            primary_p=mcnemar_exact(b, c),
        )
    else:
        out.update(primary_test="sign_flip_permutation", primary_p=out["permutation_p"])
    return out
