"""Unified benchmark report: aggregates Qwen and DeBERTa metrics for model comparison."""

import argparse
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Generate unified benchmark comparison report")
    parser.add_argument(
        "--qwen_dir",
        type=str,
        default="outputs_sft_*",
        help="Glob pattern for Qwen output dirs (default: outputs_sft_*)",
    )
    parser.add_argument(
        "--deberta_dir",
        type=str,
        default="models",
        help="Base dir for DeBERTa models (default: models)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate scatter plot (accuracy vs inference_ms)",
    )
    args = parser.parse_args()

    results = []

    for d in glob.glob(args.qwen_dir):
        if os.path.isdir(d):
            p = os.path.join(d, "benchmark_metrics.json")
            if os.path.isfile(p):
                with open(p) as f:
                    data = json.load(f)
                    data["_dir"] = d
                    results.append(data)

    deberta_base = os.path.join(args.deberta_dir, "deberta_cross_encoder_*_offtopic")
    for d in glob.glob(deberta_base):
        if os.path.isdir(d):
            p = os.path.join(d, "benchmark_metrics.json")
            if os.path.isfile(p):
                with open(p) as f:
                    data = json.load(f)
                    data["_dir"] = d
                    results.append(data)

    if not results:
        print("No benchmark_metrics.json found. Run:")
        print("  python -m qwen_pipeline.run_pipeline --mode train_all")
        print("  python -m deberta_pipeline.run_pipeline --mode train_all")
        return

    by_acc = sorted(results, key=lambda x: x.get("accuracy", 0) or 0, reverse=True)
    by_inf = sorted(
        [r for r in results if r.get("inference_time_per_sample_ms") is not None],
        key=lambda x: x.get("inference_time_per_sample_ms", float("inf")),
    )

    print("\n--- All models (by accuracy) ---")
    for r in by_acc:
        fam = r.get("family", "?")
        sz = r.get("model_size", "?")
        variant = r.get("training_variant", "")
        variant_s = f"[{variant}]" if variant else ""
        acc = r.get("accuracy") or 0
        kap = r.get("kappa") or 0
        score = r.get("score")
        score_s = f" score={score:.0f}" if score is not None else ""
        inf = r.get("inference_time_per_sample_ms")
        inf_s = f"{inf:.1f}ms" if inf is not None else "N/A"
        params = r.get("params_M", "?")
        print(f"  {fam} {sz} {variant_s} ({params}M): acc={acc:.4f} kappa={kap:.4f}{score_s} inf={inf_s}")

    print("\n--- Fastest inference ---")
    for r in by_inf[:5]:
        fam = r.get("family", "?")
        sz = r.get("model_size", "?")
        inf = r.get("inference_time_per_sample_ms")
        acc = r.get("accuracy") or 0
        print(f"  {fam} {sz}: {inf:.1f}ms/sample, acc={acc:.4f}")

    print("\n--- Recommendation summary ---")
    if by_acc:
        best = by_acc[0]
        variant = best.get("training_variant", "")
        variant_s = f" [{variant}]" if variant else ""
        print(f"  Best accuracy: {best.get('family')} {best.get('model_size')}{variant_s} (acc={best.get('accuracy', 0):.4f})")
    if by_inf:
        fastest = by_inf[0]
        variant = fastest.get("training_variant", "")
        variant_s = f" [{variant}]" if variant else ""
        print(f"  Fastest: {fastest.get('family')} {fastest.get('model_size')}{variant_s} ({fastest.get('inference_time_per_sample_ms'):.1f}ms/sample)")
    
    # Training variant comparison (for Qwen models)
    _print_variant_comparison(results)

    _write_csv(results, args.output)
    print(f"\nResults written to {args.output}")

    if args.plot:
        _plot_results(results)


def _write_csv(results, path):
    keys = [
        "family", "model_size", "training_variant", "params_M",
        "train_time_sec", "inference_time_per_sample_ms",
        "accuracy", "kappa", "score", "mean_score", "best_threshold", "checkpoint_path",
    ]
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in results:
            row = [str(r.get(k, "")) for k in keys]
            f.write(",".join(row) + "\n")


def _plot_results(results):
    """Scatter plot: accuracy vs inference_time_per_sample_ms."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    has_inf = [r for r in results if r.get("inference_time_per_sample_ms") is not None and r.get("accuracy") is not None]
    if not has_inf:
        print("No data with both inference time and accuracy, skipping plot")
        return

    familias = list(set(r.get("family", "?") for r in has_inf))
    colors = {"qwen": "tab:blue", "deberta": "tab:orange"}
    for r in has_inf:
        fam = r.get("family", "?")
        c = colors.get(fam, "gray")
        inf = r.get("inference_time_per_sample_ms")
        acc = r.get("accuracy")
        sz = r.get("model_size", "?")
        plt.scatter(inf, acc, c=c, s=80)
        plt.annotate(f"{fam} {sz}", (inf, acc), fontsize=8, alpha=0.8)

    plt.xlabel("Inference time (ms/sample)")
    plt.ylabel("Accuracy")
    plt.title("Model comparison: accuracy vs inference latency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("benchmark_accuracy_vs_inference.png")
    print("Plot saved to benchmark_accuracy_vs_inference.png")


def _print_variant_comparison(results):
    """Print side-by-side comparison of training variants for each model size."""
    qwen_results = [r for r in results if r.get("family") == "qwen"]
    if not qwen_results:
        return
    
    # Group by model size
    by_size = {}
    for r in qwen_results:
        size = r.get("model_size", "?")
        variant = r.get("training_variant", "unknown")
        if size not in by_size:
            by_size[size] = {}
        by_size[size][variant] = r
    
    # Check if we have multiple variants to compare
    has_comparison = any(len(variants) > 1 for variants in by_size.values())
    if not has_comparison:
        return
    
    print("\n--- Training Variant Comparison (Qwen) ---")
    for size in sorted(by_size.keys()):
        variants = by_size[size]
        if len(variants) < 2:
            continue
        
        print(f"\n  Model: Qwen {size}")
        print(f"  {'Variant':<20} {'Accuracy':<12} {'Kappa':<12} {'Train Time':<15} {'Inference (ms)'}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*15} {'-'*15}")
        
        for variant in ["completion_only", "full_finetune"]:
            if variant not in variants:
                continue
            r = variants[variant]
            acc = r.get("accuracy", 0)
            kap = r.get("kappa", 0)
            train_t = r.get("train_time_sec")
            train_s = f"{train_t:.1f}s" if train_t is not None else "N/A"
            inf = r.get("inference_time_per_sample_ms")
            inf_s = f"{inf:.2f}" if inf is not None else "N/A"
            print(f"  {variant:<20} {acc:<12.4f} {kap:<12.4f} {train_s:<15} {inf_s}")
        
        # Show delta if both variants exist
        if "completion_only" in variants and "full_finetune" in variants:
            co = variants["completion_only"]
            ff = variants["full_finetune"]
            acc_delta = (co.get("accuracy", 0) - ff.get("accuracy", 0)) * 100
            kap_delta = (co.get("kappa", 0) - ff.get("kappa", 0))
            
            delta_sign = "+" if acc_delta >= 0 else ""
            print(f"  {'Delta (CO - FF)':<20} {delta_sign}{acc_delta:.2f}%{'pts':<7} {kap_delta:+.4f}")
            
            if abs(acc_delta) < 0.5:
                print("  → Variants perform similarly, completion_only recommended (faster training)")
            elif acc_delta > 0:
                print("  → Completion-only is better")
            else:
                print("  → Full-finetune is better")


if __name__ == "__main__":
    main()
