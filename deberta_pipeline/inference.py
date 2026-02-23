"""DeBERTa cross-encoder inference."""

import json
import os
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .config import MAX_LENGTH
from .data_loader import load_data, get_datasets
from .eval_utils import threshold_sweep


def load_model(checkpoint_path: str):
    """Load saved DeBERTa model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer


def predict_batch(model, tokenizer, test_hf, device=None, batch_size=32):
    """Run inference and return logits / off-topic probs.
    
    Args:
        model: The model to use for prediction
        tokenizer: The tokenizer
        test_hf: The test dataset
        device: The device to use (default: model's device)
        batch_size: Batch size for inference (default: 32, reduce for large models)
    """
    if device is None:
        device = next(model.parameters()).device
    from transformers import DataCollatorWithPadding
    collator = DataCollatorWithPadding(tokenizer)
    dl = torch.utils.data.DataLoader(test_hf, batch_size=batch_size, collate_fn=collator)
    logits_list = []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            out = model(**batch)
            logits_list.append(out.logits.cpu().numpy())
    logits = np.vstack(logits_list)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    off_topic_probs = probs[:, 1]
    return off_topic_probs


def run_eval(model_size: str, checkpoint_path: str = None, train_path: str = None, test_path: str = None):
    """Run evaluation on test set and return metrics."""
    from .config import OUTPUT_PREFIX, TRAIN_PATH, TEST_PATH

    path = checkpoint_path or f"{OUTPUT_PREFIX}_{model_size}_offtopic"
    train_path = train_path or TRAIN_PATH
    test_path = test_path or TEST_PATH

    model, tokenizer = load_model(path)
    train_df, test_df = load_data(train_path, test_path)
    _, test_hf = get_datasets(train_df, test_df, tokenizer)

    t0 = time.perf_counter()
    off_topic_probs = predict_batch(model, tokenizer, test_hf)
    n = len(test_df)
    inference_ms = (time.perf_counter() - t0) * 1000 / n if n > 0 else 0

    y_true = test_df["label"].values
    best_threshold, accuracy, kappa = threshold_sweep(off_topic_probs, y_true)

    threshold_path = os.path.join(path, "threshold_config.json")
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            cfg = json.load(f)
            best_threshold = cfg.get("best_threshold", best_threshold)

    params_M = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
    benchmark = {
        "family": "deberta",
        "model_size": model_size,
        "params_M": params_M,
        "train_time_sec": None,
        "inference_time_per_sample_ms": round(inference_ms, 2),
        "accuracy": accuracy,
        "kappa": kappa,
        "best_threshold": float(best_threshold),
        "checkpoint_path": path,
    }
    return benchmark
