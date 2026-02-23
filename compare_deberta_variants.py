"""Automated pipeline to compare DeBERTa small, base, and large models across different epochs."""

import os
import json
import time
import argparse
from datetime import datetime
import torch
import gc

from deberta_pipeline.config import MODEL_OPTIONS, TRAIN_PATH, TEST_PATH
from deberta_pipeline.data_loader import load_data, get_datasets
from deberta_pipeline.train import run_train
from deberta_pipeline.inference import load_model, predict_batch
from deberta_pipeline.eval_utils import threshold_sweep


def show_gpu_memory():
    """Display current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    else:
        print("No GPU available")


def run_single_experiment(
    model_size: str,
    epochs: int,
    train_df,
    test_df,
    experiment_name: str,
    eval_only: bool = False
):
    """Run a single training experiment with given configuration."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Model Size: {model_size} | Epochs: {epochs}")
    print(f"{'='*80}\n")
    
    # Show initial GPU memory
    show_gpu_memory()
    
    # Set output directory
    output_dir = f"outputs_deberta_{model_size}_epoch{epochs}"
    
    # Training phase
    if eval_only:
        print("Skipping training (--eval_only mode), loading existing checkpoint...")
        train_time_sec = 0.0
        
        # Load model and get params
        model, tokenizer = load_model(output_dir)
        params_M = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
    else:
        print("Starting training...")
        
        # Override NUM_EPOCHS by temporarily modifying config
        import deberta_pipeline.config as deberta_config
        original_epochs = deberta_config.NUM_EPOCHS
        deberta_config.NUM_EPOCHS = epochs
        
        try:
            t0 = time.perf_counter()
            benchmark = run_train(
                model_size=model_size,
                train_path=TRAIN_PATH,
                test_path=TEST_PATH,
                output_dir=output_dir,
            )
            train_time_sec = time.perf_counter() - t0
            params_M = benchmark["params_M"]
            print(f"Training completed in {train_time_sec:.2f} seconds")
        finally:
            # Restore original config
            deberta_config.NUM_EPOCHS = original_epochs
        
        # Load trained model for inference
        model, tokenizer = load_model(output_dir)
    
    # Inference phase
    print("\nRunning inference on test set...")
    _, test_hf = get_datasets(train_df, test_df, tokenizer)
    
    t0 = time.perf_counter()
    off_topic_probs = predict_batch(model, tokenizer, test_hf)
    inference_time_total = time.perf_counter() - t0
    n = len(test_df)
    inference_time_per_sample_ms = (inference_time_total / n) * 1000 if n > 0 else 0
    
    # Evaluation phase
    print("Evaluating predictions...")
    y_true = test_df["label"].values
    best_threshold, accuracy, kappa = threshold_sweep(off_topic_probs, y_true)
    
    # Compute score (same as Qwen: 2 for correct, -1 for wrong)
    y_pred = (off_topic_probs >= best_threshold).astype(int)
    correct_mask = (y_pred == y_true)
    scores = correct_mask.astype(int) * 2 + (~correct_mask).astype(int) * (-1)
    total_score = int(scores.sum())
    mean_score = float(scores.mean())
    
    print("\n--- Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Kappa: {kappa:.4f}")
    print(f"Score: {total_score} (mean: {mean_score:.2f})")
    print(f"Best Threshold: {best_threshold:.4f}")
    
    # Save metrics
    benchmark_result = {
        "experiment_name": experiment_name,
        "family": "deberta",
        "model_size": model_size,
        "epochs": epochs,
        "params_M": params_M,
        "train_time_sec": round(train_time_sec, 2),
        "inference_time_per_sample_ms": round(inference_time_per_sample_ms, 2),
        "accuracy": float(accuracy),
        "kappa": float(kappa),
        "score": total_score,
        "mean_score": mean_score,
        "best_threshold": float(best_threshold),
        "checkpoint_path": output_dir,
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "benchmark_metrics.json"), "w") as f:
        json.dump(benchmark_result, f, indent=2)
    
    print(f"Metrics saved to {output_dir}/benchmark_metrics.json")
    
    # Clean up model to free GPU memory
    print("\nCleaning up GPU memory...")
    # Move model to CPU before deletion to ensure GPU memory is released
    if torch.cuda.is_available():
        model.cpu()
    del model, tokenizer
    
    # Aggressive GPU memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Show final GPU memory state
    show_gpu_memory()
    print("GPU memory cleanup completed.")
    
    return benchmark_result


def main():
    parser = argparse.ArgumentParser(description="Compare DeBERTa small/base/large across epochs")
    parser.add_argument("--train_path", type=str, default=None, help="Override train data path")
    parser.add_argument("--test_path", type=str, default=None, help="Override test data path")
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and only run inference + eval on existing checkpoints"
    )
    parser.add_argument(
        "--skip_small",
        action="store_true",
        help="Skip small model experiments"
    )
    parser.add_argument(
        "--skip_base",
        action="store_true",
        help="Skip base model experiments"
    )
    parser.add_argument(
        "--skip_large",
        action="store_true",
        help="Skip large model experiments"
    )
    args = parser.parse_args()
    
    # Load data paths
    train_path = args.train_path or TRAIN_PATH
    test_path = args.test_path or TEST_PATH
    
    # Load data once
    print("Loading data...")
    train_df, test_df = load_data(train_path, test_path)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"Train balance: {train_df['topic_flag'].value_counts().to_dict()}")
    print(f"Test balance: {test_df['topic_flag'].value_counts().to_dict()}")
    
    # Define all experiments
    experiments = []
    
    if not args.skip_small:
        experiments.extend([
            ("small", 1, "DeBERTa-Small 1 Epoch"),
            ("small", 2, "DeBERTa-Small 2 Epochs"),
            ("small", 3, "DeBERTa-Small 3 Epochs"),
        ])
    
    if not args.skip_base:
        experiments.extend([
            ("base", 1, "DeBERTa-Base 1 Epoch"),
            ("base", 2, "DeBERTa-Base 2 Epochs"),
            ("base", 3, "DeBERTa-Base 3 Epochs"),
        ])
    
    if not args.skip_large:
        experiments.extend([
            ("large", 1, "DeBERTa-Large 1 Epoch"),
            ("large", 2, "DeBERTa-Large 2 Epochs"),
            ("large", 3, "DeBERTa-Large 3 Epochs"),
        ])
    
    # Run all experiments
    all_results = []
    start_time_overall = time.perf_counter()
    
    for i, (model_size, epochs, name) in enumerate(experiments, 1):
        print(f"\n\n{'#'*80}")
        print(f"RUNNING EXPERIMENT {i}/{len(experiments)}")
        print(f"{'#'*80}")
        
        try:
            result = run_single_experiment(
                model_size, epochs, train_df, test_df, name, eval_only=args.eval_only
            )
            all_results.append(result)
        except Exception as e:
            print(f"\nERROR in experiment {name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_time = time.perf_counter() - start_time_overall
    
    # Save comparison report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"deberta_comparison_results_{timestamp}"
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
    print("\n\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'Experiment':<30} {'Epochs':<8} {'Params (M)':<12} {'Accuracy':<12} {'Kappa':<12} {'Mean Score':<12} {'Train Time (s)':<15}")
    print("-"*100)
    
    for result in all_results:
        print(f"{result['experiment_name']:<30} "
              f"{result['epochs']:<8} "
              f"{result['params_M']:<12.1f} "
              f"{result['accuracy']:.4f}      "
              f"{result['kappa']:.4f}      "
              f"{result['mean_score']:>6.2f}      "
              f"{result['train_time_sec']:>10.2f}")
    
    print("="*100)
    
    # Create a markdown report as well
    md_report_path = os.path.join(report_dir, "comparison_report.md")
    with open(md_report_path, "w") as f:
        f.write("# DeBERTa Model Comparison: Small vs Base vs Large\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Models:** DeBERTa-v3 (small, base, large)\n\n")
        f.write(f"**Epochs:** 1, 2, 3 for each model\n\n")
        f.write(f"**Total Experiments:** {len(all_results)}\n\n")
        f.write(f"**Total Time:** {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n\n")
        
        f.write("## Results\n\n")
        f.write("| Experiment | Epochs | Params (M) | Accuracy | Kappa | Mean Score | Train Time (s) | Inference Time (ms) |\n")
        f.write("|------------|--------|------------|----------|-------|------------|----------------|---------------------|\n")
        
        for result in all_results:
            f.write(f"| {result['experiment_name']} | "
                   f"{result['epochs']} | "
                   f"{result['params_M']:.1f} | "
                   f"{result['accuracy']:.4f} | "
                   f"{result['kappa']:.4f} | "
                   f"{result['mean_score']:.2f} | "
                   f"{result['train_time_sec']:.2f} | "
                   f"{result['inference_time_per_sample_ms']:.2f} |\n")
        
        f.write("\n## Detailed Results\n\n")
        for result in all_results:
            f.write(f"### {result['experiment_name']}\n\n")
            f.write(f"- **Model Size:** {result['model_size']}\n")
            f.write(f"- **Parameters:** {result['params_M']:.1f}M\n")
            f.write(f"- **Epochs:** {result['epochs']}\n")
            f.write(f"- **Accuracy:** {result['accuracy']:.4f}\n")
            f.write(f"- **Kappa:** {result['kappa']:.4f}\n")
            f.write(f"- **Score:** {result['score']} (mean: {result['mean_score']:.2f})\n")
            f.write(f"- **Best Threshold:** {result['best_threshold']:.4f}\n")
            f.write(f"- **Train Time:** {result['train_time_sec']:.2f} seconds\n")
            f.write(f"- **Inference Time per Sample:** {result['inference_time_per_sample_ms']:.2f} ms\n")
            f.write(f"- **Checkpoint:** {result['checkpoint_path']}\n\n")
        
        # Add analysis section
        f.write("\n## Analysis\n\n")
        f.write("### Best Performance by Metric\n\n")
        
        # Find best for each metric
        best_acc = max(all_results, key=lambda x: x['accuracy'])
        best_kappa = max(all_results, key=lambda x: x['kappa'])
        best_score = max(all_results, key=lambda x: x['mean_score'])
        
        f.write(f"- **Best Accuracy:** {best_acc['experiment_name']} ({best_acc['accuracy']:.4f})\n")
        f.write(f"- **Best Kappa:** {best_kappa['experiment_name']} ({best_kappa['kappa']:.4f})\n")
        f.write(f"- **Best Mean Score:** {best_score['experiment_name']} ({best_score['mean_score']:.2f})\n\n")
        
        # Group by model size
        f.write("### Performance by Model Size\n\n")
        for model_size in ["small", "base", "large"]:
            size_results = [r for r in all_results if r['model_size'] == model_size]
            if size_results:
                f.write(f"#### {model_size.upper()}\n\n")
                for r in size_results:
                    f.write(f"- **{r['epochs']} epoch(s):** Acc={r['accuracy']:.4f}, Kappa={r['kappa']:.4f}, Score={r['mean_score']:.2f}\n")
                f.write("\n")
    
    print(f"\nMarkdown report saved to: {md_report_path}")


if __name__ == "__main__":
    main()
