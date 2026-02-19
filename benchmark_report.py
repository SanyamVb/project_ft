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
        acc = r.get("accuracy") or 0
        kap = r.get("kappa") or 0
        score = r.get("score")
        score_s = f" score={score:.0f}" if score is not None else ""
        inf = r.get("inference_time_per_sample_ms")
        inf_s = f"{inf:.1f}ms" if inf is not None else "N/A"
        params = r.get("params_M", "?")
        print(f"  {fam} {sz} ({params}M): acc={acc:.4f} kappa={kap:.4f}{score_s} inf={inf_s}")

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
        print(f"  Best accuracy: {best.get('family')} {best.get('model_size')} (acc={best.get('accuracy', 0):.4f})")
    if by_inf:
        fastest = by_inf[0]
        print(f"  Fastest: {fastest.get('family')} {fastest.get('model_size')} ({fastest.get('inference_time_per_sample_ms'):.1f}ms/sample)")

    _write_csv(results, args.output)
    print(f"\nResults written to {args.output}")

    if args.plot:
        _plot_results(results)


def _write_csv(results, path):
    keys = [
        "family", "model_size", "params_M",
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


if __name__ == "__main__":
    main()
