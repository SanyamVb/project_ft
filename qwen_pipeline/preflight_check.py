"""Pre-flight checks for Qwen SFT pipeline before training/inference.

Run from project root: python -m qwen_pipeline.preflight_check [--model_size 4B]

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


def run_checks(config=None, model_size: str = "4B", skip_model: bool = False) -> bool:
    """Run all pre-flight checks. Returns True if all critical checks pass."""
    from .config import Config, QWEN_MODEL_OPTIONS

    cfg = config or Config()
    if model_size:
        cfg.llm_model = QWEN_MODEL_OPTIONS.get(model_size, cfg.llm_model)
    all_ok = True

    print("\n=== Qwen SFT Pipeline Pre-flight Checks ===\n")

    # 1. Data file exists
    print("1. Data file")
    if not _check(os.path.isfile(cfg.data_path), f"Data file exists: {cfg.data_path}"):
        all_ok = False

    # 2. Data loading and columns
    print("\n2. Data loading")
    df_prepared = None
    try:
        from .dataset_utils import load_and_prepare_df
        df_prepared = load_and_prepare_df(cfg)
        _check(True, f"Data loaded and prepared: {len(df_prepared)} rows")
    except FileNotFoundError as e:
        _check(False, f"Data file not found: {e}")
        all_ok = False
    except Exception as e:
        _check(False, f"Data load error: {e}")
        all_ok = False

    if df_prepared is not None:
        if len(df_prepared) == 0:
            _check(False, "Prepared dataframe is empty (no valid rows after filter)")
            all_ok = False
        else:
            # Check label balance for stratification
            counts = df_prepared["topic_flag"].value_counts()
            n_classes = len(counts)
            _check(n_classes >= 2, f"Both classes present for stratification: {counts.to_dict()}")
            if n_classes < 2:
                all_ok = False
            _check(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio == 1.0,
                   f"Split ratios sum to 1: {cfg.train_ratio}+{cfg.val_ratio}+{cfg.test_ratio}")

    # 3. Config sanity
    print("\n3. Config validity")
    valid_optim = ("adamw_torch", "adamw_torch_fused", "adamw_8bit", "adafactor")
    valid_scheduler = ("linear", "cosine", "cosine_with_restarts", "polynomial", "constant")
    _check(cfg.sft_optim in valid_optim, f"sft_optim valid: {cfg.sft_optim}")
    _check(cfg.sft_lr_scheduler_type in valid_scheduler, f"sft_lr_scheduler_type valid: {cfg.sft_lr_scheduler_type}")
    _check(cfg.sft_epochs > 0, f"sft_epochs > 0 ({cfg.sft_epochs})")
    _check(cfg.sft_batch_size > 0, f"sft_batch_size > 0 ({cfg.sft_batch_size})")
    _check(cfg.sft_grad_accum >= 1, f"sft_grad_accum >= 1 ({cfg.sft_grad_accum})")
    _check(cfg.sft_lr > 0, f"sft_lr > 0 ({cfg.sft_lr})")
    _check(cfg.sft_max_grad_norm > 0, f"sft_max_grad_norm > 0 ({cfg.sft_max_grad_norm})")
    _check(cfg.sft_early_stopping_patience >= 1, f"sft_early_stopping_patience >= 1 ({cfg.sft_early_stopping_patience})")
    _check(cfg.max_seq_length > 0, f"max_seq_length > 0 ({cfg.max_seq_length})")
    _check(cfg.lora_rank > 0, f"lora_rank > 0 ({cfg.lora_rank})")
    _check(0 < cfg.conf_score_sft <= 1, f"conf_score_sft in (0, 1]: {cfg.conf_score_sft}")

    # 4. Model size valid
    print("\n4. Model")
    if model_size and model_size not in QWEN_MODEL_OPTIONS:
        _check(False, f"Model size '{model_size}' not in {list(QWEN_MODEL_OPTIONS.keys())}")
        all_ok = False
    else:
        _check(True, f"Model size valid: {model_size or 'from config'}")

    # 5. Splits and SFT dataset build
    if all_ok and df_prepared is not None and len(df_prepared) > 0:
        print("\n5. Data splits and SFT dataset")
        try:
            from .dataset_utils import get_all_splits, build_sft_dataset
            splits = get_all_splits(cfg)
            train_df, val_df, test_df = splits["train_df"], splits["val_df"], splits["test_df"]
            _check(len(train_df) > 0, f"Train split: {len(train_df)} rows")
            _check(len(val_df) > 0, f"Val split: {len(val_df)} rows")
            _check(len(test_df) > 0, f"Test split: {len(test_df)} rows")

            sft_train = splits["sft_train"]
            sft_val = splits["sft_val"]
            _check(len(sft_train) > 0, f"SFT train dataset: {len(sft_train)} examples")
            _check(len(sft_val) > 0, f"SFT val dataset: {len(sft_val)} examples")

            sample = sft_train[0]
            _check("prompt" in sample and "completion" in sample, "SFT example has prompt and completion")
        except Exception as e:
            _check(False, f"Split/dataset build: {e}")
            all_ok = False

        # 6. Model load (optional, can be slow)
        if not skip_model:
            print("\n6. Model load (Unsloth)")
            try:
                from .train_sft import init_model_for_sft
                model, tokenizer = init_model_for_sft(cfg)
                _check(tokenizer.pad_token is not None or tokenizer.eos_token is not None,
                       "Tokenizer has pad_token or eos_token")
                n_params = sum(p.numel() for p in model.parameters())
                _check(True, f"Model loaded ({round(n_params/1e6, 1)}M params)")
            except Exception as e:
                _check(False, f"Model load: {e}")
                all_ok = False
        else:
            print("\n6. Model load (skipped with --skip-model)")

    # 7. Output dir writable
    print("\n7. Output and environment")
    out_dir = os.path.dirname(cfg.sft_output_dir) or cfg.sft_output_dir
    try:
        os.makedirs(cfg.sft_output_dir, exist_ok=True)
        test_file = os.path.join(cfg.sft_output_dir, ".preflight_write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        _check(True, f"Output dir writable: {cfg.sft_output_dir}")
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

    # 9. Inference dependencies
    print("\n9. Inference readiness")
    try:
        from .dataset_utils import _user_prompt
        from .inference import parse_answer
        _check(True, "parse_answer, _user_prompt importable")
    except Exception as e:
        _check(False, f"Inference imports: {e}")
        all_ok = False

    print("\n" + ("All checks passed." if all_ok else "Some checks failed. Fix issues before training.") + "\n")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Qwen SFT pipeline pre-flight checks")
    parser.add_argument("--model_size", default="4B", choices=["0.6B", "4B", "8B"])
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip-model", action="store_true", help="Skip model load (faster)")
    args = parser.parse_args()

    from .config import Config, QWEN_MODEL_OPTIONS

    config = Config()
    if args.data_path:
        config.data_path = args.data_path
    if args.model_size:
        config.llm_model = QWEN_MODEL_OPTIONS[args.model_size]
        config.sft_output_dir = args.output_dir or f"outputs_sft_{args.model_size}"

    ok = run_checks(config=config, model_size=args.model_size, skip_model=args.skip_model)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
