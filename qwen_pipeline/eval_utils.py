"""Compute metrics for off-topic classification."""

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from typing import List


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, score_correct: float = 2.0, score_wrong: float = -1.0) -> dict:
    """Compute accuracy, kappa, score (2 if correct, -1 if wrong), classification report.
    Invalid predictions (NaN or -1) are excluded from metrics."""
    valid = np.isfinite(y_pred) & (y_pred >= 0) & (y_pred <= 1)
    if valid.sum() == 0:
        return {
            "accuracy": 0.0,
            "kappa": 0.0,
            "score": 0.0,
            "mean_score": 0.0,
            "classification_report": "",
            "confusion_matrix": np.array([[0, 0], [0, 0]]),
        }
    y_true_valid = y_true[valid].astype(int)
    y_pred_valid = y_pred[valid].astype(int)

    accuracy = float((y_pred_valid == y_true_valid).mean())
    kappa = float(cohen_kappa_score(y_true_valid, y_pred_valid))

    # Score: score_correct if correct, score_wrong if wrong
    correct = (y_pred_valid == y_true_valid).astype(float)
    scores = np.where(correct, score_correct, score_wrong)
    total_score = float(scores.sum())
    mean_score = float(scores.mean())

    return {
        "accuracy": accuracy,
        "kappa": kappa,
        "score": total_score,
        "mean_score": mean_score,
        "classification_report": classification_report(
            y_true_valid, y_pred_valid, target_names=["On-Topic", "Off-Topic"]
        ),
        "confusion_matrix": confusion_matrix(y_true_valid, y_pred_valid),
    }


def evaluate_predictions(df, predictions: List[dict]) -> dict:
    """Evaluate predictions against ground truth. Treats None topic_flag as wrong (0)."""
    y_true = df["topic_flag"].values
    y_pred = np.array([
        1 if p.get("topic_flag") is True else (0 if p.get("topic_flag") is False else -1)
        for p in predictions
    ])
    return compute_metrics(y_true, y_pred)
