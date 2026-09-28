from __future__ import annotations

import numpy as np


def _binary_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """ROC AUC from average ranks, including correctly handled probability ties."""
    if np.unique(labels).size != 2:
        return float("nan")
    order = np.argsort(probabilities, kind="mergesort")
    ranks = np.empty(probabilities.size, dtype=float)
    sorted_probabilities = probabilities[order]
    start = 0
    while start < probabilities.size:
        end = start + 1
        while end < probabilities.size and sorted_probabilities[end] == sorted_probabilities[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positives = labels == 1
    n_positive = int(positives.sum())
    n_negative = labels.size - n_positive
    return float((ranks[positives].sum() - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))


def classification_metrics(labels: list[int], probabilities: list[float]) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= 0.5).astype(int)
    true_positive = int(np.sum((y == 1) & (pred == 1)))
    false_positive = int(np.sum((y == 0) & (pred == 1)))
    false_negative = int(np.sum((y == 1) & (pred == 0)))
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "auc": _binary_auc(y, p),
        "acc": float(np.mean(y == pred)),
        "f1": float(2 * true_positive / denominator) if denominator else 0.0,
    }
