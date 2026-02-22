"""Automated pipeline to compare completion-only vs normal SFT across different epochs."""

import os
import json
import time
import argparse
from datetime import datetime

from qwen_pipeline.config import Config, QWEN_MODEL_OPTIONS
from qwen_pipeline.dataset_utils import get_all_splits
from qwen_pipeline.train_sft import run_sft_train
from qwen_pipeline.inference import load_trained_model, predict_batch
from qwen_pipeline.eval_utils import evaluate_predictions


def _get_params_M(model) -> float:
    """Get model parameter count in millions."""
    try:
        total = sum(p.numel() for p in model.parameters())
        return round(total / 1e6, 1)
    except Exception:
        return 0.0


def run_single_experiment(config: Config, training_variant: str, epochs: int, splits: dict, experiment_name: str):
    """Run a single training experiment with given configuration."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Training Variant: {training_variant} | Epochs: {epochs}")
    print(f"{'='*80}\n")
    
    # Update config for this experiment
    config.training_variant = training_variant
    config.sft_epochs = epochs
    config.sft_output_dir = f"outputs_sft_4B_{training_variant}_epoch{epochs}"
    
    # Training phase
    print("Starting training...")
    t0 = time.perf_counter()
    run_sft_train(splits["sft_train"], splits["sft_val"], config)
    train_time_sec = time.perf_counter() - t0
    print(f"Training completed in {train_time_sec:.2f} seconds")
    
    # Inference phase
    print("\nLoading trained model for inference...")
    checkpoint = config.sft_output_dir
    model, tokenizer = load_trained_model(config, checkpoint)
    
    print("Running inference on test set...")
    predictions, inference_time_total = predict_batch(splits["test_df"], model, tokenizer, config)
    n = len(splits["test_df"])
    inference_time_per_sample_ms = (inference_time_total / n) * 1000 if n > 0 else 0
    
    # Evaluation phase
    print("Evaluating predictions...")
    metrics = evaluate_predictions(splits["test_df"], predictions)
    
    print("\n--- Results ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Kappa: {metrics['kappa']:.4f}")
    print(f"Score: {metrics['score']:.0f} (mean: {metrics['mean_score']:.2f})")
    
    # Save metrics
    params_M = _get_params_M(model)
    benchmark = {
        "experiment_name": experiment_name,
        "family": "qwen",
        "model_size": "4B",
        "training_variant": training_variant,
        "epochs": epochs,
        "params_M": params_M,
        "train_time_sec": round(train_time_sec, 2),
        "inference_time_per_sample_ms": round(inference_time_per_sample_ms, 2),
        "accuracy": metrics["accuracy"],
        "kappa": metrics["kappa"],
        "score": metrics["score"],
        "mean_score": metrics["mean_score"],
        "best_threshold": 0.5,
        "checkpoint_path": config.sft_output_dir,
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }
    
    os.makedirs(config.sft_output_dir, exist_ok=True)
    with open(os.path.join(config.sft_output_dir, "benchmark_metrics.json"), "w") as f:
        json.dump(benchmark, f, indent=2)
    
    print(f"Metrics saved to {config.sft_output_dir}/benchmark_metrics.json")
    
    # Clean up model to free memory
    del model, tokenizer
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return benchmark


def main():
    parser = argparse.ArgumentParser(description="Compare completion-only vs normal SFT across epochs")
    parser.add_argument("--data_path", type=str, default=None, help="Override data path")
    parser.add_argument(
        "--skip_completion",
        action="store_true",
        help="Skip completion-only experiments (only run normal SFT)"
    )
    parser.add_argument(
        "--skip_normal",
        action="store_true",
        help="Skip normal SFT experiments (only run completion-only)"
    )
    args = parser.parse_args()
    
    # Initialize config
    config = Config()
    config.llm_model = QWEN_MODEL_OPTIONS["4B"]
    print(f"Using max_position_embeddings: {config.max_position_embeddings}")
    if args.data_path:
        config.data_path = args.data_path
    
    # Load data once
    print("Loading and splitting data...")
    splits = get_all_splits(config)
    print(f"Train: {len(splits['train_df'])} | Val: {len(splits['val_df'])} | Test: {len(splits['test_df'])}")
    print(f"Train balance: {splits['train_df']['topic_flag'].value_counts().to_dict()}")
    
    # Define all experiments
    experiments = []
    
    if not args.skip_completion:
        experiments.extend([
            ("completion_only", 1, "Completion-Only 1 Epoch"),
            ("completion_only", 2, "Completion-Only 2 Epochs"),
            ("completion_only", 3, "Completion-Only 3 Epochs"),
        ])
    
    if not args.skip_normal:
        experiments.extend([
            ("full_finetune", 1, "Normal SFT 1 Epoch"),
            ("full_finetune", 2, "Normal SFT 2 Epochs"),
            ("full_finetune", 3, "Normal SFT 3 Epochs"),
        ])
    
    # Run all experiments
    all_results = []
    start_time_overall = time.perf_counter()
    
    for i, (variant, epochs, name) in enumerate(experiments, 1):
        print(f"\n\n{'#'*80}")
        print(f"RUNNING EXPERIMENT {i}/{len(experiments)}")
        print(f"{'#'*80}")
        
        try:
            result = run_single_experiment(config, variant, epochs, splits, name)
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR in experiment {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_time = time.perf_counter() - start_time_overall
    
    # Save comparison report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"comparison_results_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(experiments),
        "total_time_sec": round(total_time, 2),
        "results": all_results
    }
    
    report_path = os.path.join(report_dir, "comparison_report.json")
    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'='*80}")
    print(f"\nTotal time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"\nComparison report saved to: {report_path}")
    
    # Print summary table
    print("\n\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Experiment':<30} {'Epochs':<8} {'Accuracy':<12} {'Kappa':<12} {'Mean Score':<12} {'Train Time (s)':<15}")
    print("-"*80)
    
    for result in all_results:
        print(f"{result['experiment_name']:<30} "
              f"{result['epochs']:<8} "
              f"{result['accuracy']:.4f}      "
              f"{result['kappa']:.4f}      "
              f"{result['mean_score']:>6.2f}      "
              f"{result['train_time_sec']:>10.2f}")
    
    print("="*80)
    
    # Create a markdown report as well
    md_report_path = os.path.join(report_dir, "comparison_report.md")
    with open(md_report_path, "w") as f:
        f.write("# SFT Training Comparison: Completion-Only vs Normal SFT\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Model:** Qwen 4B\n\n")
        f.write(f"**Total Experiments:** {len(all_results)}\n\n")
        f.write(f"**Total Time:** {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n\n")
        
        f.write("## Results\n\n")
        f.write("| Experiment | Epochs | Accuracy | Kappa | Mean Score | Train Time (s) | Inference Time (ms) |\n")
        f.write("|------------|--------|----------|-------|------------|----------------|---------------------|\n")
        
        for result in all_results:
            f.write(f"| {result['experiment_name']} | "
                   f"{result['epochs']} | "
                   f"{result['accuracy']:.4f} | "
                   f"{result['kappa']:.4f} | "
                   f"{result['mean_score']:.2f} | "
                   f"{result['train_time_sec']:.2f} | "
                   f"{result['inference_time_per_sample_ms']:.2f} |\n")
        
        f.write("\n## Detailed Results\n\n")
        for result in all_results:
            f.write(f"### {result['experiment_name']}\n\n")
            f.write(f"- **Training Variant:** {result['training_variant']}\n")
            f.write(f"- **Epochs:** {result['epochs']}\n")
            f.write(f"- **Accuracy:** {result['accuracy']:.4f}\n")
            f.write(f"- **Kappa:** {result['kappa']:.4f}\n")
            f.write(f"- **Score:** {result['score']} (mean: {result['mean_score']:.2f})\n")
            f.write(f"- **Train Time:** {result['train_time_sec']:.2f} seconds\n")
            f.write(f"- **Inference Time per Sample:** {result['inference_time_per_sample_ms']:.2f} ms\n")
            f.write(f"- **Checkpoint:** {result['checkpoint_path']}\n\n")
    
    print(f"\nMarkdown report saved to: {md_report_path}")


if __name__ == "__main__":
    main()
