"""Compute metrics for off-topic classification."""

import re
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from typing import List, Optional


# Regex pattern to validate response format (similar to prudhvi_topic.py)
# Matches: optional <think>...</think>, optional backticks/json, the JSON structure, optional whitespace/eos
RESPONSE_FORMAT_REGEX = re.compile(
    r'^(<think>.*?</think>)?'  # Optional thinking tags
    r'(?P<backticks>(`{3})?)'  # Optional triple backticks
    r'(json)?'  # Optional 'json' keyword
    r'\{"topic_flag":(?P<topic_flag>true|false), "conf_score":(?P<conf_score>[0-1]\.[0-9]{2})\}'  # JSON structure
    r'(?P=backticks)'  # Closing backticks (if opened)
    r'\s*'  # Optional whitespace
    r'(<\|im_end\|>|<\|endoftext\|>)?'  # Optional EOS tokens
    r'\s*$',  # End of string
    flags=re.DOTALL | re.IGNORECASE
)


def validate_response_format(response: str) -> tuple[bool, Optional[bool]]:
    """Validate if response matches expected JSON format and extract topic_flag.
    Uses regex-based validation similar to training reward function.
    
    Args:
        response: Raw response string from model
        
    Returns:
        Tuple of (format_is_valid, topic_flag_value)
        - format_is_valid: True if response has valid format
        - topic_flag_value: Extracted boolean topic_flag, or None if invalid
    """
    match = RESPONSE_FORMAT_REGEX.search(response.strip())
    
    if match:
        topic_flag_str = match.group('topic_flag').lower()
        topic_flag = True if topic_flag_str == 'true' else False
        return True, topic_flag
    else:
        return False, None


def compute_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    format_valid: Optional[np.ndarray] = None,
    score_format_correct: float = 1.5, 
    score_format_wrong: float = -1.0,
    score_answer_correct: float = 3.0,
    score_answer_wrong: float = -1.0
) -> dict:
    """Compute accuracy, kappa, score (format + answer), classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels (-1 for invalid predictions)
        format_valid: Optional boolean array indicating if format is valid (True=valid format)
        score_format_correct: Score for correct format (default: 1.5)
        score_format_wrong: Score for incorrect format (default: -1.0)
        score_answer_correct: Score for correct answer (default: 3.0)
        score_answer_wrong: Score for incorrect answer (default: -1.0)
    
    Invalid predictions (-1) are treated as wrong and receive score_answer_wrong.
    """
    if len(y_pred) == 0:
        return {
            "accuracy": 0.0,
            "kappa": 0.0,
            "format_score": 0.0,
            "answer_score": 0.0,
            "total_score": 0.0,
            "mean_score": 0.0,
            "classification_report": "",
            "confusion_matrix": np.array([[0, 0], [0, 0]]),
        }
    
    # For classification metrics, convert -1 to 0 (treat invalid as wrong prediction)
    # This ensures sklearn metrics work properly with binary classification
    y_pred_for_metrics = np.where(y_pred == -1, 0, y_pred).astype(int)
    y_true_int = y_true.astype(int)

    accuracy = float((y_pred_for_metrics == y_true_int).mean())
    kappa = float(cohen_kappa_score(y_true_int, y_pred_for_metrics))

    # Answer score: score_answer_correct if correct, score_answer_wrong if wrong or invalid
    answer_correct = (y_pred == y_true_int).astype(float)
    answer_scores = np.where(answer_correct, score_answer_correct, score_answer_wrong)
    total_answer_score = float(answer_scores.sum())
    
    # Format score: score_format_correct if valid format, score_format_wrong if invalid
    total_format_score = 0.0
    if format_valid is not None:
        format_scores = np.where(format_valid, score_format_correct, score_format_wrong)
        total_format_score = float(format_scores.sum())
    
    total_score = total_answer_score + total_format_score
    mean_score = total_score / len(y_pred)

    return {
        "accuracy": accuracy,
        "kappa": kappa,
        "format_score": total_format_score,
        "answer_score": total_answer_score,
        "total_score": total_score,
        "mean_score": mean_score,
        "classification_report": classification_report(
            y_true_int, y_pred_for_metrics, target_names=["On-Topic", "Off-Topic"]
        ),
        "confusion_matrix": confusion_matrix(y_true_int, y_pred_for_metrics),
    }


def evaluate_predictions(
    df, 
    predictions: List[dict], 
    check_format: bool = False
) -> dict:
    """Evaluate predictions against ground truth. Treats None topic_flag as wrong (0).
    
    Args:
        df: DataFrame with ground truth 'topic_flag' column
        predictions: List of prediction dicts with 'topic_flag' and 'raw' fields
        check_format: If True, validate format from 'raw' field and compute format-based scores
    """
    y_true = df["topic_flag"].values
    
    # Extract topic_flag predictions
    y_pred = np.array([
        1 if p.get("topic_flag") is True else (0 if p.get("topic_flag") is False else -1)
        for p in predictions
    ])
    
    # Extract and validate format from raw responses if requested
    format_valid = None
    if check_format:
        format_valid = np.array([
            validate_response_format(p.get("raw", ""))[0] for p in predictions
        ])
    
    return compute_metrics(y_true, y_pred, format_valid=format_valid)
