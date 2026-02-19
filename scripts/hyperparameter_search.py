"""Hyperparameter search for DeBERTa and Qwen training pipelines.

Uses Optuna for Bayesian optimization. Run from project root:

  python -m scripts.hyperparameter_search --pipeline deberta --n_trials 20 --output results/tune_deberta.json
  python -m scripts.hyperparameter_search --pipeline qwen --n_trials 15 --model_size 0.6B --output results/tune_qwen.json

For full grid search (small spaces only, expensive):

  python -m scripts.hyperparameter_search --pipeline deberta --grid --output results/tune_deberta.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure project root is on path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    import optuna
except ImportError:
    print("Install optuna: pip install optuna")
    sys.exit(1)


def _deberta_objective(trial: optuna.Trial, model_size: str, quick: bool) -> float:
    """Optuna objective for DeBERTa pipeline. Returns kappa (maximize)."""
    import deberta_pipeline.config as dcfg
    from deberta_pipeline.train import run_train

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

    try:
        benchmark = run_train(
            model_size=model_size,
            output_dir=output_dir,
        )
        kappa = benchmark.get("kappa", 0.0)
        trial.set_user_attr("accuracy", benchmark.get("accuracy", 0.0))
        trial.set_user_attr("train_time_sec", benchmark.get("train_time_sec", 0))
        return kappa
    except Exception as e:
        trial.set_user_attr("error", str(e))
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


def _deberta_grid_objective(params: dict, model_size: str, quick: bool) -> dict:
    """Run a single DeBERTa grid trial. Returns benchmark dict."""
    import deberta_pipeline.config as dcfg
    from deberta_pipeline.train import run_train

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


def _run_optuna(pipeline: str, model_size: str, n_trials: int, output_path: str, quick: bool):
    """Run Optuna study."""
    if pipeline == "deberta":
        objective = lambda t: _deberta_objective(t, model_size, quick)
    else:
        objective = lambda t: _qwen_objective(t, model_size, quick)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    result = {
        "pipeline": pipeline,
        "model_size": model_size,
        "n_trials": n_trials,
        "best_value": best.value,
        "best_params": best.params,
        "best_trial_user_attrs": dict(best.user_attrs),
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "user_attrs": dict(t.user_attrs),
            }
            for t in study.trials
        ],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nBest kappa: {best.value:.4f}")
    print("Best params:", best.params)
    print(f"Results saved to {output_path}")


def _run_grid(pipeline: str, model_size: str, output_path: str, quick: bool):
    """Run full grid search (small space)."""
    import itertools

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
                b = _deberta_grid_objective(params_lower, model_size, quick)
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
        "best_value": best["kappa"] if best else None,
        "best_params": best.get("params") if best else None,
        "all_trials": results,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    if best:
        print(f"\nBest kappa: {best['kappa']:.4f}")
        print("Best params:", best.get("params"))
    print(f"Results saved to {output_path}")


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
    args = parser.parse_args()

    if args.grid:
        _run_grid(args.pipeline, args.model_size, args.output, args.quick)
    else:
        _run_optuna(
            args.pipeline,
            args.model_size,
            args.n_trials,
            args.output,
            args.quick,
        )


if __name__ == "__main__":
    main()
