import numpy as np
from sklearn.metrics import r2_score

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))

def mean_pairwise_jaccard(feature_sets):
    # feature_sets: list[set[str]]
    n = len(feature_sets)
    if n <= 1:
        return 1.0
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(jaccard(feature_sets[i], feature_sets[j]))
    return float(np.mean(vals)) if vals else 1.0

def fidelity_r2(y_true, y_pred) -> float:
    # y_true, y_pred: arrays
    if len(y_true) < 2:
        return 0.0
    return float(r2_score(y_true, y_pred))
