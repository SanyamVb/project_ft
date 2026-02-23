"""DeBERTa cross-encoder training."""

import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from transformers.trainer_callback import EarlyStoppingCallback
from sklearn.metrics import cohen_kappa_score

from .config import (
    MODEL_OPTIONS,
    MAX_LENGTH,
    NUM_EPOCHS,
    BATCH_SIZE,
    BATCH_SIZE_PER_MODEL,
    EVAL_BATCH_SIZE,
    EVAL_BATCH_SIZE_PER_MODEL,
    LEARNING_RATE,
    OUTPUT_PREFIX,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    MAX_GRAD_NORM,
    GRADIENT_ACCUMULATION_STEPS,
    OPTIM,
    LR_SCHEDULER_TYPE,
    EARLY_STOPPING_PATIENCE,
    LABEL_SMOOTHING,
)
from .data_loader import load_data, get_datasets


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    kappa = cohen_kappa_score(labels, preds)
    return {"accuracy": float(acc), "kappa": float(kappa)}


class DeBERTaTrainerWithLabelSmoothing(Trainer):
    """Trainer that applies label smoothing to the classification loss."""

    def __init__(self, label_smoothing: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.label_smoothing <= 0:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        labels = inputs.pop("labels", None)
        outputs = model(**inputs)
        logits = outputs.logits
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            loss = loss_fct(logits, labels)
        else:
            loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else None
        return (loss, outputs) if return_outputs else loss


def run_train(
    model_size: str,
    train_path: str = None,
    test_path: str = None,
    output_dir: str = None,
):
    """Train DeBERTa cross-encoder for given model size."""
    from .config import TRAIN_PATH, TEST_PATH

    model_name = MODEL_OPTIONS[model_size]
    output_dir = output_dir or f"{OUTPUT_PREFIX}_{model_size}_offtopic"
    train_path = train_path or TRAIN_PATH
    test_path = test_path or TEST_PATH

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    print("Loading data...")
    train_df, test_df = load_data(train_path, test_path)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    train_hf, test_hf = get_datasets(train_df, test_df, tokenizer)

    print(f"Loading model: {model_name}")
    # Load in fp16 for large model to save memory
    if model_size == "large" and torch.cuda.is_available():
        print("Loading large model in fp16 mode to save memory...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    params_M = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
    print(f"Parameters: {params_M}M")
    
    # Enable gradient checkpointing for large model to save memory
    if model_size == "large":
        print("Enabling gradient checkpointing for large model...")
        model.gradient_checkpointing_enable()

    # Use model-size-specific batch size if available
    batch_size = BATCH_SIZE_PER_MODEL.get(model_size, BATCH_SIZE)
    eval_batch_size = EVAL_BATCH_SIZE_PER_MODEL.get(model_size, EVAL_BATCH_SIZE)
    # Adjust gradient accumulation to maintain effective batch size
    # For large model with smaller batch size, increase gradient accumulation
    if model_size == "large":
        grad_accum = max(4, (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) // batch_size)
    else:
        grad_accum = max(1, (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) // batch_size)
    
    print(f"Training config: batch_size={batch_size}, eval_batch_size={eval_batch_size}, "
          f"grad_accum={grad_accum} (effective_batch={batch_size * grad_accum})")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        optim=OPTIM,
        max_grad_norm=MAX_GRAD_NORM,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="kappa",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True if model_size == "large" else False,
    )

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=0.0,
        ),
    ]

    trainer_cls = DeBERTaTrainerWithLabelSmoothing if LABEL_SMOOTHING > 0 else Trainer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_hf,
        eval_dataset=test_hf,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
        callbacks=callbacks,
    )
    if LABEL_SMOOTHING > 0:
        trainer_kwargs["label_smoothing"] = LABEL_SMOOTHING

    trainer = trainer_cls(**trainer_kwargs)

    t0 = time.perf_counter()
    trainer.train()
    train_time_sec = time.perf_counter() - t0

    preds_output = trainer.predict(test_hf)
    logits = preds_output.predictions
    y_true = test_df["label"].values
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    off_topic_probs = probs[:, 1]

    best_threshold, accuracy, kappa = _threshold_sweep(off_topic_probs, y_true)

    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(os.path.join(output_dir, "threshold_config.json"), "w") as f:
        json.dump({"best_threshold": float(best_threshold)}, f, indent=2)

    inference_time_ms = _measure_inference_time(trainer, test_hf, len(test_df))

    benchmark = {
        "family": "deberta",
        "model_size": model_size,
        "params_M": params_M,
        "train_time_sec": round(train_time_sec, 2),
        "inference_time_per_sample_ms": round(inference_time_ms, 2),
        "accuracy": accuracy,
        "kappa": kappa,
        "best_threshold": float(best_threshold),
        "checkpoint_path": output_dir,
    }
    with open(os.path.join(output_dir, "benchmark_metrics.json"), "w") as f:
        json.dump(benchmark, f, indent=2)

    print(f"Model saved to {output_dir}")
    print(f"Benchmark: acc={accuracy:.4f} kappa={kappa:.4f} train={train_time_sec:.0f}s inf={inference_time_ms:.1f}ms/sample")
    return benchmark


def _threshold_sweep(off_topic_probs, y_true):
    """Find best threshold by kappa."""
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


def _measure_inference_time(trainer, test_hf, n_samples):
    """Measure average inference time per sample in ms."""
    import time
    t0 = time.perf_counter()
    _ = trainer.predict(test_hf)
    elapsed = (time.perf_counter() - t0) * 1000
    return elapsed / n_samples if n_samples > 0 else 0
