"""CLI for DeBERTa cross-encoder pipeline."""

import argparse
import glob
import json
import os

from .config import MODEL_OPTIONS, TRAIN_PATH, TEST_PATH, OUTPUT_PREFIX
from .train import run_train
from .inference import run_eval


def main():
    parser = argparse.ArgumentParser(description="DeBERTa cross-encoder pipeline for off-topic detection")
    parser.add_argument(
        "--mode",
        choices=["train", "train_all", "eval", "report"],
        default="train",
        help="Pipeline mode",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=list(MODEL_OPTIONS.keys()),
        default="base",
        help="DeBERTa model size",
    )
    parser.add_argument("--train_path", type=str, default=None, help="Override train data path")
    parser.add_argument("--test_path", type=str, default=None, help="Override test data path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for eval")
    parser.add_argument("--qwen_dir", type=str, default=None, help="Qwen output dir pattern for report")
    parser.add_argument("--deberta_dir", type=str, default=None, help="DeBERTa models dir for report")
    args = parser.parse_args()

    if args.mode == "train":
        run_train(
            model_size=args.model_size,
            train_path=args.train_path or TRAIN_PATH,
            test_path=args.test_path or TEST_PATH,
        )
        return

    if args.mode == "train_all":
        for size in MODEL_OPTIONS:
            print(f"\n=== Training DeBERTa {size} ===")
            run_train(
                model_size=size,
                train_path=args.train_path or TRAIN_PATH,
                test_path=args.test_path or TEST_PATH,
            )
        return

    if args.mode == "eval":
        benchmark = run_eval(
            model_size=args.model_size,
            checkpoint_path=args.checkpoint,
            train_path=args.train_path or TRAIN_PATH,
            test_path=args.test_path or TEST_PATH,
        )
        path = args.checkpoint or f"{OUTPUT_PREFIX}_{args.model_size}_offtopic"
        out = os.path.join(path, "benchmark_metrics.json")
        os.makedirs(path, exist_ok=True)
        with open(out, "w") as f:
            json.dump(benchmark, f, indent=2)
        print(f"Benchmark saved to {out}")
        return

    if args.mode == "report":
        _run_report(args.qwen_dir, args.deberta_dir)
        return


def _run_report(qwen_dir=None, deberta_dir=None):
    """Aggregate benchmark_metrics.json from both pipelines and print/CSV."""
    results = []

    qwen_pattern = qwen_dir or "outputs_sft_*"
    for d in glob.glob(qwen_pattern):
        p = os.path.join(d, "benchmark_metrics.json")
        if os.path.isfile(p):
            with open(p) as f:
                data = json.load(f)
                data["_dir"] = d
                results.append(data)

    deberta_base = deberta_dir or "models"
    for d in glob.glob(os.path.join(deberta_base, "deberta_cross_encoder_*_offtopic")):
        p = os.path.join(d, "benchmark_metrics.json")
        if os.path.isfile(p):
            with open(p) as f:
                data = json.load(f)
                data["_dir"] = d
                results.append(data)

    if not results:
        print("No benchmark_metrics.json found. Run train or train_all first.")
        return

    by_acc = sorted(results, key=lambda x: x.get("accuracy", 0), reverse=True)
    by_inf = sorted(
        [r for r in results if r.get("inference_time_per_sample_ms")],
        key=lambda x: x.get("inference_time_per_sample_ms", float("inf")),
    )

    print("\n--- All models (by accuracy) ---")
    for r in by_acc:
        fam = r.get("family", "?")
        sz = r.get("model_size", "?")
        acc = r.get("accuracy", 0)
        kap = r.get("kappa", 0)
        inf = r.get("inference_time_per_sample_ms")
        inf_s = f"{inf:.1f}ms" if inf is not None else "N/A"
        print(f"  {fam} {sz}: acc={acc:.4f} kappa={kap:.4f} inf={inf_s}")

    print("\n--- Fastest inference ---")
    for r in by_inf[:5]:
        fam = r.get("family", "?")
        sz = r.get("model_size", "?")
        inf = r.get("inference_time_per_sample_ms")
        acc = r.get("accuracy", 0)
        print(f"  {fam} {sz}: {inf:.1f}ms/sample, acc={acc:.4f}")

    out_csv = "benchmark_results.csv"
    _write_csv(results, out_csv)
    print(f"\nResults written to {out_csv}")


def _write_csv(results, path):
    """Write results to CSV."""
    keys = ["family", "model_size", "params_M", "train_time_sec", "inference_time_per_sample_ms", "accuracy", "kappa", "best_threshold", "checkpoint_path"]
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in results:
            row = [str(r.get(k, "")) for k in keys]
            f.write(",".join(row) + "\n")


if __name__ == "__main__":
    main()
