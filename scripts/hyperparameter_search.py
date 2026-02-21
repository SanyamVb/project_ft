"""Hyperparameter search for DeBERTa and Qwen training pipelines.

Uses Optuna for Bayesian optimization. Run from project root:

  python -m scripts.hyperparameter_search --pipeline deberta --n_trials 20 --output results/tune_deberta.json --seed 42
  python -m scripts.hyperparameter_search --pipeline qwen --n_trials 15 --model_size 0.6B --output results/tune_qwen.json --seed 42

For full grid search (small spaces only, expensive):

  python -m scripts.hyperparameter_search --pipeline deberta --grid --output results/tune_deberta.json

Note: Use --seed for reproducible results. Without it, results may vary between runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import random
import numpy as np

# Ensure project root is on path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    import optuna
except ImportError:
    print("Install optuna: pip install optuna")
    sys.exit(1)


def _set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _deberta_objective(trial: optuna.Trial, model_size: str, quick: bool, seed: int = None) -> float:
    """Optuna objective for DeBERTa pipeline. Returns kappa (maximize)."""
    import deberta_pipeline.config as dcfg
    from deberta_pipeline.train import run_train
    
    # Set seed for reproducibility
    if seed is not None:
        trial_seed = seed + trial.number
        _set_seed(trial_seed)
        trial.set_user_attr("seed", trial_seed)

    trial_dir = os.path.join("results", "tune_deberta", f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)

    # Sample hyperparameters
    dcfg.LEARNING_RATE = trial.suggest_float("learning_rate", 1e-5, 3e-5, log=True)
    dcfg.MAX_GRAD_NORM = trial.suggest_categorical("max_grad_norm", [0.5, 1.0, 1.5])
    dcfg.WARMUP_RATIO = trial.suggest_categorical("warmup_ratio", [0.05, 0.1, 0.15])
    dcfg.LABEL_SMOOTHING = trial.suggest_categorical("label_smoothing", [0.0, 0.1])
    dcfg.GRADIENT_ACCUMULATION_STEPS = trial.suggest_categorical(
        "gradient_accumulation_steps", [1, 2, 4]
    )
    dcfg.OPTIM = trial.suggest_categorical("optim", ["adamw_torch", "adamw_torch_fused"])
    dcfg.LR_SCHEDULER_TYPE = trial.suggest_categorical(
        "lr_scheduler_type", ["cosine", "linear"]
    )

    if quick:
        dcfg.NUM_EPOCHS = 2

    output_dir = os.path.join(trial_dir, f"deberta_{model_size}_offtopic")
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: Testing hyperparameters:")
    print(f"  LR: {dcfg.LEARNING_RATE:.2e}, Grad Norm: {dcfg.MAX_GRAD_NORM}")
    print(f"  Warmup: {dcfg.WARMUP_RATIO}, Label Smoothing: {dcfg.LABEL_SMOOTHING}")
    print(f"  Grad Accum: {dcfg.GRADIENT_ACCUMULATION_STEPS}, Optimizer: {dcfg.OPTIM}")
    print(f"  LR Scheduler: {dcfg.LR_SCHEDULER_TYPE}, Epochs: {dcfg.NUM_EPOCHS}")
    if seed is not None:
        print(f"  Seed: {trial_seed}")
    print(f"{'='*60}\n")

    try:
        benchmark = run_train(
            model_size=model_size,
            output_dir=output_dir,
        )
        kappa = benchmark.get("kappa", 0.0)
        accuracy = benchmark.get("accuracy", 0.0)
        train_time = benchmark.get("train_time_sec", 0)
        
        # Store detailed results
        trial.set_user_attr("accuracy", accuracy)
        trial.set_user_attr("train_time_sec", train_time)
        trial.set_user_attr("checkpoint_path", output_dir)
        trial.set_user_attr("best_threshold", benchmark.get("best_threshold", 0.5))
        
        print(f"\nTrial {trial.number} Results: Kappa={kappa:.4f}, Accuracy={accuracy:.4f}, Time={train_time:.0f}s")
        print(f"Checkpoint saved to: {output_dir}\n")
        
        return kappa
    except Exception as e:
        trial.set_user_attr("error", str(e))
        print(f"\nTrial {trial.number} FAILED: {str(e)}\n")
        raise optuna.TrialPruned() from e


def _qwen_objective(trial: optuna.Trial, model_size: str, quick: bool) -> float:
    """Optuna objective for Qwen pipeline. Returns kappa (maximize)."""
    from qwen_pipeline.config import Config, QWEN_MODEL_OPTIONS
    from qwen_pipeline.dataset_utils import get_all_splits
    from qwen_pipeline.train_sft import run_sft_train
    from qwen_pipeline.inference import load_trained_model, predict_batch
    from qwen_pipeline.eval_utils import evaluate_predictions

    trial_dir = os.path.join("results", "tune_qwen", f"trial_{trial.number}")
    output_dir = os.path.join(trial_dir, f"outputs_sft_{model_size}")
    os.makedirs(output_dir, exist_ok=True)

    config = Config()
    config.llm_model = QWEN_MODEL_OPTIONS[model_size]
    config.sft_output_dir = output_dir

    config.sft_lr = trial.suggest_float("sft_lr", 1e-5, 3e-5, log=True)
    config.sft_max_grad_norm = trial.suggest_categorical(
        "sft_max_grad_norm", [0.5, 1.0, 1.5]
    )
    config.sft_optim = trial.suggest_categorical(
        "sft_optim", ["adamw_torch", "adamw_8bit"]
    )
    config.sft_lr_scheduler_type = trial.suggest_categorical(
        "sft_lr_scheduler_type", ["cosine", "linear"]
    )
    config.sft_early_stopping_patience = trial.suggest_int(
        "sft_early_stopping_patience", 1, 3
    )

    if quick:
        config.sft_epochs = 1

    splits = get_all_splits(config)
    if quick:
        sft_train = splits["sft_train"].select(range(min(100, len(splits["sft_train"]))))
        sft_val = splits["sft_val"].select(range(min(30, len(splits["sft_val"]))))
    else:
        sft_train = splits["sft_train"]
        sft_val = splits["sft_val"]

    try:
        run_sft_train(sft_train, sft_val, config)
    except Exception as e:
        trial.set_user_attr("error", str(e))
        raise optuna.TrialPruned() from e

    model, tokenizer = load_trained_model(config, output_dir)
    predictions = predict_batch(splits["test_df"], model, tokenizer, config)
    metrics = evaluate_predictions(splits["test_df"], predictions)
    kappa = metrics["kappa"]
    trial.set_user_attr("accuracy", metrics["accuracy"])
    return kappa


_PARAM_TO_CONFIG = {
    "learning_rate": "LEARNING_RATE",
    "max_grad_norm": "MAX_GRAD_NORM",
    "warmup_ratio": "WARMUP_RATIO",
    "label_smoothing": "LABEL_SMOOTHING",
    "gradient_accumulation_steps": "GRADIENT_ACCUMULATION_STEPS",
    "optim": "OPTIM",
}


def _deberta_grid_objective(params: dict, model_size: str, quick: bool, seed: int = None) -> dict:
    """Run a single DeBERTa grid trial. Returns benchmark dict."""
    import deberta_pipeline.config as dcfg
    from deberta_pipeline.train import run_train
    
    # Set seed for reproducibility
    if seed is not None:
        _set_seed(seed)

    for k, v in params.items():
        config_key = _PARAM_TO_CONFIG.get(k, k.upper())
        setattr(dcfg, config_key, v)
    if quick:
        dcfg.NUM_EPOCHS = 2

    output_dir = os.path.join(
        "results", "tune_deberta_grid",
        "_".join(f"{k}={v}" for k, v in sorted(params.items()))
    )
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, f"deberta_{model_size}_offtopic")

    benchmark = run_train(model_size=model_size, output_dir=output_dir)
    return benchmark


def _run_optuna(pipeline: str, model_size: str, n_trials: int, output_path: str, quick: bool, seed: int = None):
    """Run Optuna study."""
    if seed is not None:
        print(f"\n{'='*60}")
        print(f"Setting random seed to {seed} for reproducibility")
        print(f"Each trial will use seed = {seed} + trial_number")
        print(f"{'='*60}\n")
        _set_seed(seed)
        sampler = optuna.samplers.TPESampler(seed=seed)
    else:
        print(f"\n{'='*60}")
        print(f"WARNING: No seed specified. Results will not be reproducible.")
        print(f"Use --seed <number> for reproducible results.")
        print(f"{'='*60}\n")
        sampler = None
    
    if pipeline == "deberta":
        objective = lambda t: _deberta_objective(t, model_size, quick, seed)
    else:
        objective = lambda t: _qwen_objective(t, model_size, quick)

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    
    # Calculate statistics
    valid_trials = [t for t in study.trials if t.value is not None]
    if valid_trials:
        values = [t.value for t in valid_trials]
        mean_kappa = sum(values) / len(values)
        std_kappa = (sum((v - mean_kappa)**2 for v in values) / len(values))**0.5
    else:
        mean_kappa = std_kappa = 0.0
    
    result = {
        "pipeline": pipeline,
        "model_size": model_size,
        "n_trials": n_trials,
        "seed": seed,
        "quick_mode": quick,
        "best_trial_number": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "best_trial_user_attrs": dict(best.user_attrs),
        "statistics": {
            "mean_kappa": round(mean_kappa, 4),
            "std_kappa": round(std_kappa, 4),
            "n_completed": len(valid_trials),
            "n_failed": n_trials - len(valid_trials),
        },
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "user_attrs": dict(t.user_attrs),
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Best Trial: #{best.number}")
    print(f"Best Kappa: {best.value:.4f}")
    print(f"Best Accuracy: {best.user_attrs.get('accuracy', 0):.4f}")
    if 'checkpoint_path' in best.user_attrs:
        print(f"Best Model: {best.user_attrs['checkpoint_path']}")
    print(f"\nBest Hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print(f"\nStatistics across {len(valid_trials)} trials:")
    print(f"  Mean Kappa: {mean_kappa:.4f} ± {std_kappa:.4f}")
    print(f"  Completed: {len(valid_trials)}, Failed: {n_trials - len(valid_trials)}")
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}\n")


def _run_grid(pipeline: str, model_size: str, output_path: str, quick: bool, seed: int = None):
    """Run full grid search (small space)."""
    import itertools
    
    if seed is not None:
        print(f"\nSetting random seed to {seed} for reproducibility\n")
        _set_seed(seed)

    if pipeline == "deberta":
        space = {
            "learning_rate": [1e-5, 2e-5, 3e-5],
            "max_grad_norm": [1.0, 1.5],
            "warmup_ratio": [0.1],
            "label_smoothing": [0.0, 0.1],
            "gradient_accumulation_steps": [2],
            "optim": ["adamw_torch"],
        }
        keys = list(space.keys())
        values = [space[k] for k in keys]
        results = []
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            params_lower = {k.lower(): v for k, v in params.items()}
            try:
                b = _deberta_grid_objective(params_lower, model_size, quick, seed)
                b["params"] = params
                results.append(b)
            except Exception as e:
                results.append({"params": params, "error": str(e), "kappa": -1})
    else:
        print("Grid search for Qwen not implemented (use --n_trials with optuna)")
        return

    best = max(
        (r for r in results if "kappa" in r and r["kappa"] >= 0),
        key=lambda r: r.get("kappa", -1),
        default=None,
    )

    result = {
        "pipeline": pipeline,
        "model_size": model_size,
        "mode": "grid",
        "seed": seed,
        "quick_mode": quick,
        "best_value": best["kappa"] if best else None,
        "best_params": best.get("params") if best else None,
        "all_trials": results,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"GRID SEARCH COMPLETE")
    print(f"{'='*60}")
    if best:
        print(f"Best Kappa: {best['kappa']:.4f}")
        print(f"Best Accuracy: {best.get('accuracy', 0):.4f}")
        print(f"\nBest Hyperparameters:")
        for k, v in best.get("params", {}).items():
            print(f"  {k}: {v}")
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for DeBERTa and Qwen pipelines"
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["deberta", "qwen"],
        required=True,
        help="Pipeline to tune",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="base",
        help="DeBERTa: xsmall|small|base|large. Qwen: 0.6B|4B|8B",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Number of Optuna trials (default: 20)",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Use full grid search instead of Optuna (DeBERTa only, small space)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/tune_results.json",
        help="Output JSON path for results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer epochs / subset of data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (recommended: 42)",
    )
    args = parser.parse_args()

    if args.grid:
        _run_grid(args.pipeline, args.model_size, args.output, args.quick, args.seed)
    else:
        _run_optuna(
            args.pipeline,
            args.model_size,
            args.n_trials,
            args.output,
            args.quick,
            args.seed,
        )


if __name__ == "__main__":
    main()
