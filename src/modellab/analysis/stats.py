import math

import numpy as np


def benjamini_hochberg(p) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def holm(p) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p, kind="mergesort")
    adjusted = np.maximum.accumulate(p[order] * (n - np.arange(n)))
    out = np.empty(n)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def wilson_interval(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None, None
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def cohens_h(p1: float, p2: float) -> float:
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def fisher_p(a: int, b: int, c: int, d: int) -> float:
    from scipy.stats import fisher_exact

    return float(fisher_exact([[a, b], [c, d]])[1])
