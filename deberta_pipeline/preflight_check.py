"""Pre-flight checks for DeBERTa pipeline before training/inference.

Run from project root: python -m deberta_pipeline.preflight_check [--model_size base]

Exits with code 0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys

_OK = "[OK]"
_FAIL = "[FAIL]"
_WARN = "[WARN]"


def _check(cond: bool, msg: str, fail_fatal: bool = True) -> bool:
    status = _OK if cond else (_FAIL if fail_fatal else _WARN)
    print(f"  {status} {msg}")
    return cond


def run_checks(model_size: str = "base", train_path: str = None, test_path: str = None, skip_model: bool = False) -> bool:
    """Run all pre-flight checks. Returns True if all critical checks pass."""
    from .config import (
        MODEL_OPTIONS,
        TRAIN_PATH,
        TEST_PATH,
        MAX_LENGTH,
        NUM_EPOCHS,
        BATCH_SIZE,
        LEARNING_RATE,
        MAX_GRAD_NORM,
        GRADIENT_ACCUMULATION_STEPS,
        OPTIM,
        LR_SCHEDULER_TYPE,
        LABEL_SMOOTHING,
    )

    train_path = train_path or TRAIN_PATH
    test_path = test_path or TEST_PATH
    all_ok = True
    train_df = test_df = None

    print("\n=== DeBERTa Pipeline Pre-flight Checks ===\n")

    # 1. Data files exist
    print("1. Data files")
    if not _check(os.path.isfile(train_path), f"Train file exists: {train_path}"):
        all_ok = False
    if not _check(os.path.isfile(test_path), f"Test file exists: {test_path}"):
        all_ok = False

    # 2. Data loading and columns
    print("\n2. Data loading")
    try:
        import pandas as pd
        train_df = pd.read_excel(train_path)
        test_df = pd.read_excel(test_path)
    except Exception as e:
        _check(False, f"Excel readable: {e}")
        all_ok = False
    else:
        _check(True, "Excel files readable")

        required = ["Script", "Category", "Transcription-Machine", "Final_Annotation"]
        missing_train = [c for c in required if c not in train_df.columns]
        missing_test = [c for c in required if c not in test_df.columns]
        if not _check(not missing_train, f"Train has required columns; missing: {missing_train}"):
            all_ok = False
        if not _check(not missing_test, f"Test has required columns; missing: {missing_test}"):
            all_ok = False

        if not missing_train and not missing_test:
            label_map = {"No": 0, "Yes": 1}
            train_df["label"] = train_df["Final_Annotation"].map(label_map)
            test_df["label"] = test_df["Final_Annotation"].map(label_map)
            nan_train = train_df["label"].isna().sum()
            nan_test = test_df["label"].isna().sum()
            if not _check(nan_train == 0, f"Train labels valid (Yes/No); {nan_train} invalid"):
                all_ok = False
            if not _check(nan_test == 0, f"Test labels valid (Yes/No); {nan_test} invalid"):
                all_ok = False

            if len(train_df) > 0 and len(test_df) > 0:
                _check(True, f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
            else:
                if not _check(len(train_df) > 0, "Train set not empty"):
                    all_ok = False
                if not _check(len(test_df) > 0, "Test set not empty"):
                    all_ok = False

    # 3. Config sanity
    print("\n3. Config validity")
    valid_optim = ("adamw_torch", "adamw_torch_fused", "adamw_8bit", "adafactor")
    valid_scheduler = ("linear", "cosine", "cosine_with_restarts", "polynomial", "constant")
    _check(OPTIM in valid_optim, f"Optimizer valid: {OPTIM}")
    _check(LR_SCHEDULER_TYPE in valid_scheduler, f"Scheduler valid: {LR_SCHEDULER_TYPE}")
    _check(NUM_EPOCHS > 0, f"NUM_EPOCHS > 0 ({NUM_EPOCHS})")
    _check(BATCH_SIZE > 0, f"BATCH_SIZE > 0 ({BATCH_SIZE})")
    _check(LEARNING_RATE > 0, f"LEARNING_RATE > 0 ({LEARNING_RATE})")
    _check(MAX_GRAD_NORM > 0, f"MAX_GRAD_NORM > 0 ({MAX_GRAD_NORM})")
    _check(GRADIENT_ACCUMULATION_STEPS >= 1, f"GRADIENT_ACCUMULATION_STEPS >= 1 ({GRADIENT_ACCUMULATION_STEPS})")
    _check(0 <= LABEL_SMOOTHING < 1, f"LABEL_SMOOTHING in [0, 1) ({LABEL_SMOOTHING})")
    _check(MAX_LENGTH > 0, f"MAX_LENGTH > 0 ({MAX_LENGTH})")

    # 4. Model size valid
    print("\n4. Model")
    if not _check(model_size in MODEL_OPTIONS, f"Model size '{model_size}' in {list(MODEL_OPTIONS.keys())}"):
        all_ok = False

    # 5. Tokenizer and dataset build (needs data)
    if all_ok and train_df is not None and test_df is not None and len(train_df) > 0:
        print("\n5. Tokenizer and tokenization")
        try:
            from transformers import AutoTokenizer
            model_name = MODEL_OPTIONS[model_size]
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            _check(True, f"Tokenizer loaded: {model_name}")
        except Exception as e:
            _check(False, f"Tokenizer load: {e}")
            all_ok = False
        else:
            try:
                from .data_loader import load_data, get_datasets
                tdf, testdf = load_data(train_path, test_path)
                train_hf, test_hf = get_datasets(tdf, testdf, tokenizer)
                _check(len(train_hf) > 0, f"Train HF dataset built ({len(train_hf)} samples)")
                _check(len(test_hf) > 0, f"Test HF dataset built ({len(test_hf)} samples)")
                sample = train_hf[0]
                _check("input_ids" in sample and "labels" in sample, "Sample has input_ids and labels")
            except Exception as e:
                _check(False, f"Dataset build: {e}")
                all_ok = False

        # 6. Model load (optional, can be slow)
        if not skip_model:
            print("\n6. Model load")
            try:
                from transformers import AutoModelForSequenceClassification
                model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
                n_params = sum(p.numel() for p in model.parameters())
                _check(True, f"Model loaded ({round(n_params/1e6, 1)}M params)")
            except Exception as e:
                _check(False, f"Model load: {e}")
                all_ok = False
        else:
            print("\n6. Model load (skipped with --skip-model)")

    # 7. Output dir writable
    print("\n7. Output and environment")
    out_prefix = f"models/deberta_cross_encoder_{model_size}_offtopic"
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
            test_file = os.path.join(out_dir, ".preflight_write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            _check(True, f"Output dir writable: {out_dir}")
        except Exception as e:
            _check(False, f"Output dir writable: {e}")
            all_ok = False

    # 8. CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        _check(cuda_ok, f"CUDA available: {cuda_ok}" + ("" if cuda_ok else " (training will be slow on CPU)"), fail_fatal=False)
    except ImportError:
        _check(False, "torch not installed")
        all_ok = False

    # 9. Inference prerequisites (threshold_config)
    print("\n9. Inference readiness")
    # Just check that threshold_sweep is importable
    try:
        from .eval_utils import threshold_sweep
        _check(True, "threshold_sweep (eval_utils) importable")
    except Exception as e:
        _check(False, f"eval_utils: {e}")
        all_ok = False

    print("\n" + ("All checks passed." if all_ok else "Some checks failed. Fix issues before training.") + "\n")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="DeBERTa pipeline pre-flight checks")
    parser.add_argument("--model_size", default="base", choices=["xsmall", "small", "base", "large"])
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--skip-model", action="store_true", help="Skip model load (faster, for data-only checks)")
    args = parser.parse_args()

    ok = run_checks(
        model_size=args.model_size,
        train_path=args.train_path,
        test_path=args.test_path,
        skip_model=args.skip_model,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
