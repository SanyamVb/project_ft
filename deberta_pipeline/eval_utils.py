"""Evaluation utilities for DeBERTa pipeline."""

import numpy as np
from sklearn.metrics import cohen_kappa_score


def threshold_sweep(off_topic_probs: np.ndarray, y_true: np.ndarray):
    """Find best threshold by kappa. Returns (best_threshold, accuracy, kappa)."""
    thresholds = np.arange(0.10, 0.91, 0.01)
    best_kappa = -1
    best_t, best_acc, best_k = 0.5, 0.0, 0.0
    for t in thresholds:
        y_pred = (off_topic_probs >= t).astype(int)
        acc = (y_pred == y_true).mean()
        k = cohen_kappa_score(y_true, y_pred)
        if k > best_kappa:
            best_kappa = k
            best_t, best_acc, best_k = t, acc, k
    return best_t, best_acc, best_k
