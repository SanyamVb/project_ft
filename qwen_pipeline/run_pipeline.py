"""CLI entry point for SFT pipeline: process -> train -> inference -> eval."""

import argparse
import json
import os
import time

from .config import Config, QWEN_MODEL_OPTIONS
from .dataset_utils import get_all_splits
from .train_sft import run_sft_train
from .train_grpo import run_grpo_train
from .inference import load_trained_model, predict_batch
from .eval_utils import evaluate_predictions


def main():
    parser = argparse.ArgumentParser(description="SFT Phase 1 pipeline for off-topic detection")
    parser.add_argument(
        "--mode",
        choices=["full", "process_only", "train_only", "inference_only", "eval_only", "train_all"],
        default="full",
        help="Pipeline mode",
    )
    parser.add_argument("--data_path", type=str, default=None, help="Override data path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint for inference/eval")
    parser.add_argument("--output_dir", type=str, default=None, help="Override SFT output dir")
    parser.add_argument(
        "--model_size",
        type=str,
        choices=list(QWEN_MODEL_OPTIONS.keys()),
        default=None,
        help="Qwen model size: 0.6B, 4B, or 8B",
    )
    parser.add_argument(
        "--run_grpo",
        action="store_true",
        help="Run GRPO after SFT; inference/eval use GRPO checkpoint",
    )
    parser.add_argument(
        "--skip_sft",
        action="store_true",
        help="Skip SFT training and go straight to GRPO (requires existing SFT checkpoint)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from specific checkpoint path",
    )
    parser.add_argument(
        "--training_variant",
        type=str,
        choices=["completion_only", "full_finetune"],
        default="completion_only",
        help="SFT training variant: completion_only (mask prompts) or full_finetune (train on entire sequence)",
    )
    args = parser.parse_args()

    config = Config()
    if args.data_path:
        config.data_path = args.data_path
    if args.model_size:
        config.llm_model = QWEN_MODEL_OPTIONS[args.model_size]
        config.sft_output_dir = f"outputs_sft_{args.model_size}_{args.training_variant}"
        config.grpo_output_dir = f"outputs_grpo_{args.model_size}_{args.training_variant}"
    if args.output_dir:
        config.sft_output_dir = args.output_dir
    
    # Set training variant
    config.training_variant = args.training_variant

    if args.mode == "train_all":
        for size in QWEN_MODEL_OPTIONS:
            print(f"\n=== Training Qwen {size} ({args.training_variant}) ===")
            config.llm_model = QWEN_MODEL_OPTIONS[size]
            config.sft_output_dir = f"outputs_sft_{size}_{args.training_variant}"
            config.training_variant = args.training_variant
            splits = get_all_splits(config)
            t0 = time.perf_counter()
            run_sft_train(splits["sft_train"], splits["sft_val"], config)
            train_time_sec = time.perf_counter() - t0
            checkpoint = config.sft_output_dir
            model, tokenizer = load_trained_model(config, checkpoint)
            predictions, inference_time_total = predict_batch(splits["test_df"], model, tokenizer, config)
            n = len(splits["test_df"])
            inference_time_per_sample_ms = (inference_time_total / n) * 1000 if n > 0 else 0
            metrics = evaluate_predictions(splits["test_df"], predictions)
            params_M = _get_params_M(model)
            benchmark = {
                "family": "qwen",
                "model_size": size,
                "training_variant": args.training_variant,
                "params_M": params_M,
                "train_time_sec": round(train_time_sec, 2),
                "inference_time_per_sample_ms": round(inference_time_per_sample_ms, 2),
                "accuracy": metrics["accuracy"],
                "kappa": metrics["kappa"],
                "score": metrics["score"],
                "mean_score": metrics["mean_score"],
                "best_threshold": 0.5,
                "checkpoint_path": config.sft_output_dir,
            }
            os.makedirs(config.sft_output_dir, exist_ok=True)
            with open(os.path.join(config.sft_output_dir, "benchmark_metrics.json"), "w") as f:
                json.dump(benchmark, f, indent=2)
            print(f"Benchmark metrics saved to {config.sft_output_dir}/benchmark_metrics.json")
        return

    splits = None

    if args.mode in ("full", "process_only"):
        print("Loading and splitting data...")
        splits = get_all_splits(config)
        train_df, val_df, test_df = splits["train_df"], splits["val_df"], splits["test_df"]
        print(f"Train: {len(train_df)} | Test: {len(test_df)} (test set used for validation)")
        print(f"Train balance: {train_df['topic_flag'].value_counts().to_dict()}")
        if args.mode == "process_only":
            return

    train_time_sec = None
    if args.mode == "train_only" and splits is None:
        splits = get_all_splits(config)

    if args.mode in ("full", "train_only"):
        if splits is None:
            splits = get_all_splits(config)
        
        # Check if SFT should be skipped
        sft_checkpoint = None
        if args.skip_sft:
            if os.path.isdir(config.sft_output_dir) and os.path.exists(os.path.join(config.sft_output_dir, "config.json")):
                print(f"Skipping SFT training, using existing checkpoint: {config.sft_output_dir}")
                sft_checkpoint = config.sft_output_dir
            else:
                raise FileNotFoundError(f"Cannot skip SFT: checkpoint not found at {config.sft_output_dir}")
        else:
            print("Running SFT training...")
            t0 = time.perf_counter()
            run_sft_train(splits["sft_train"], splits["sft_val"], config, resume_from_checkpoint=args.resume_from_checkpoint)
            train_time_sec = time.perf_counter() - t0
            sft_checkpoint = config.sft_output_dir
        
        if args.run_grpo:
            print("Running GRPO training...")
            run_grpo_train(
                splits["grpo_train"], config, 
                sft_checkpoint_path=sft_checkpoint or config.sft_output_dir,
                resume_from_checkpoint=None  # Don't auto-resume GRPO
            )
        if args.mode == "train_only":
            return

    if args.mode in ("full", "inference_only", "eval_only"):
        if splits is None:
            splits = get_all_splits(config)
        checkpoint = (
            args.checkpoint
            or (config.grpo_output_dir if args.run_grpo else config.sft_output_dir)
        )
        if not os.path.isdir(checkpoint):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}. Run training first or pass --checkpoint /path/to/model"
            )
        print("Loading model from", checkpoint)
        model, tokenizer = load_trained_model(config, checkpoint)
        print("Running inference on test set...")
        predictions, inference_time_total = predict_batch(splits["test_df"], model, tokenizer, config)
        n = len(splits["test_df"])
        inference_time_per_sample_ms = (inference_time_total / n) * 1000 if n > 0 else 0

        if args.mode == "inference_only":
            return

        metrics = evaluate_predictions(splits["test_df"], predictions)
        print("\n--- Test Set Metrics ---")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Kappa: {metrics['kappa']:.4f}")
        print(f"Score (2 correct, -1 wrong): total={metrics['score']:.0f} mean={metrics['mean_score']:.2f}")
        print(metrics["classification_report"])
        print("Confusion matrix:")
        print(metrics["confusion_matrix"])

        output_dir = config.grpo_output_dir if args.run_grpo else config.sft_output_dir
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "eval_metrics.json")
        with open(out_path, "w") as f:
            json.dump({
                "accuracy": metrics["accuracy"],
                "kappa": metrics["kappa"],
                "score": metrics["score"],
                "mean_score": metrics["mean_score"],
                "confusion_matrix": metrics["confusion_matrix"].tolist(),
            }, f, indent=2)
        print(f"\nMetrics saved to {out_path}")

        params_M = _get_params_M(model)
        benchmark = {
            "family": "qwen",
            "model_size": args.model_size or "4B",
            "training_variant": args.training_variant,
            "params_M": params_M,
            "train_time_sec": round(train_time_sec, 2) if train_time_sec is not None else None,
            "inference_time_per_sample_ms": round(inference_time_per_sample_ms, 2),
            "accuracy": metrics["accuracy"],
            "kappa": metrics["kappa"],
            "score": metrics["score"],
            "mean_score": metrics["mean_score"],
            "best_threshold": 0.5,
            "checkpoint_path": output_dir,
        }
        benchmark_path = os.path.join(output_dir, "benchmark_metrics.json")
        with open(benchmark_path, "w") as f:
            json.dump(benchmark, f, indent=2)
        print(f"Benchmark metrics saved to {benchmark_path}")


def _get_params_M(model) -> float:
    """Get model parameter count in millions."""
    try:
        total = sum(p.numel() for p in model.parameters())
        return round(total / 1e6, 1)
    except Exception:
        return 0.0


if __name__ == "__main__":
    main()
