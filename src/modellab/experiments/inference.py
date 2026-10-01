import numpy as np

from modellab.experiments.errors import ExperimentConfigError


def apply_inference(iv, probs, base_pred, names):
    index = {name: i for i, name in enumerate(names)}

    def cls(name):
        if name not in index:
            raise ExperimentConfigError(f"unknown class '{name}', known classes: {list(names)}")
        return index[name]

    p = np.asarray(probs, dtype=np.float64)
    n = len(p)
    if iv.temperature is not None:
        z = np.log(np.clip(p, 1e-12, 1.0)) / iv.temperature
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        new = e / e.sum(axis=1, keepdims=True)
        return new, new.argmax(axis=1)
    if iv.class_weights is not None:
        w = np.ones(len(names))
        for name, value in iv.class_weights.items():
            w[cls(name)] = value
        new = p * w
        new /= new.sum(axis=1, keepdims=True)
        return new, new.argmax(axis=1)

    pred = np.asarray(base_pred, dtype=np.int64).copy()
    fallback = cls(iv.fallback_class)
    if iv.confidence_threshold is not None:
        low = p[np.arange(n), pred] < iv.confidence_threshold
        pred = np.where(low, fallback, pred)
    else:
        original = pred.copy()
        for name, threshold in iv.class_thresholds.items():
            c = cls(name)
            pred = np.where((original == c) & (p[:, c] < threshold), fallback, pred)
    return p, pred
