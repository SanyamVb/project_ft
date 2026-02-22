"""Analyze and visualize comparison results from SFT experiments."""

import json
import os
import argparse
from pathlib import Path


def load_latest_comparison():
    """Load the most recent comparison report."""
    # Find all comparison_results directories
    comparison_dirs = [d for d in os.listdir('.') if d.startswith('comparison_results_')]
    
    if not comparison_dirs:
        print("No comparison results found. Run compare_sft_variants.py first.")
        return None
    
    # Get the most recent one
    latest_dir = max(comparison_dirs)
    report_path = os.path.join(latest_dir, 'comparison_report.json')
    
    if not os.path.exists(report_path):
        print(f"Report file not found: {report_path}")
        return None
    
    with open(report_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded comparison from: {latest_dir}")
    return data, latest_dir


def analyze_results(data):
    """Analyze and print insights from the comparison."""
    results = data['results']
    
    if not results:
        print("No results to analyze.")
        return
    
    print("\n" + "="*80)
    print("ANALYSIS: Completion-Only vs Normal SFT")
    print("="*80)
    
    # Group results by training variant
    completion_only = [r for r in results if r['training_variant'] == 'completion_only']
    full_finetune = [r for r in results if r['training_variant'] == 'full_finetune']
    
    if completion_only:
        print("\n### Completion-Only Training")
        print("-" * 80)
        for r in sorted(completion_only, key=lambda x: x['epochs']):
            print(f"Epoch {r['epochs']}: Accuracy={r['accuracy']:.4f}, Kappa={r['kappa']:.4f}, "
                  f"Mean Score={r['mean_score']:.2f}, Train Time={r['train_time_sec']:.2f}s")
    
    if full_finetune:
        print("\n### Normal SFT (Full Fine-tune)")
        print("-" * 80)
        for r in sorted(full_finetune, key=lambda x: x['epochs']):
            print(f"Epoch {r['epochs']}: Accuracy={r['accuracy']:.4f}, Kappa={r['kappa']:.4f}, "
                  f"Mean Score={r['mean_score']:.2f}, Train Time={r['train_time_sec']:.2f}s")
    
    # Find best performers
    print("\n### Best Performers")
    print("-" * 80)
    
    best_accuracy = max(results, key=lambda x: x['accuracy'])
    best_kappa = max(results, key=lambda x: x['kappa'])
    best_score = max(results, key=lambda x: x['mean_score'])
    fastest = min(results, key=lambda x: x['train_time_sec'])
    
    print(f"Best Accuracy: {best_accuracy['experiment_name']} ({best_accuracy['accuracy']:.4f})")
    print(f"Best Kappa: {best_kappa['experiment_name']} ({best_kappa['kappa']:.4f})")
    print(f"Best Mean Score: {best_score['experiment_name']} ({best_score['mean_score']:.2f})")
    print(f"Fastest Training: {fastest['experiment_name']} ({fastest['train_time_sec']:.2f}s)")
    
    # Epoch impact analysis
    if completion_only and len(completion_only) > 1:
        print("\n### Completion-Only: Impact of Epochs")
        print("-" * 80)
        completion_only_sorted = sorted(completion_only, key=lambda x: x['epochs'])
        for i in range(1, len(completion_only_sorted)):
            prev = completion_only_sorted[i-1]
            curr = completion_only_sorted[i]
            acc_delta = curr['accuracy'] - prev['accuracy']
            kappa_delta = curr['kappa'] - prev['kappa']
            time_delta = curr['train_time_sec'] - prev['train_time_sec']
            print(f"Epoch {prev['epochs']} → {curr['epochs']}: "
                  f"Accuracy Δ={acc_delta:+.4f}, Kappa Δ={kappa_delta:+.4f}, "
                  f"Time Δ={time_delta:+.2f}s")
    
    if full_finetune and len(full_finetune) > 1:
        print("\n### Normal SFT: Impact of Epochs")
        print("-" * 80)
        full_finetune_sorted = sorted(full_finetune, key=lambda x: x['epochs'])
        for i in range(1, len(full_finetune_sorted)):
            prev = full_finetune_sorted[i-1]
            curr = full_finetune_sorted[i]
            acc_delta = curr['accuracy'] - prev['accuracy']
            kappa_delta = curr['kappa'] - prev['kappa']
            time_delta = curr['train_time_sec'] - prev['train_time_sec']
            print(f"Epoch {prev['epochs']} → {curr['epochs']}: "
                  f"Accuracy Δ={acc_delta:+.4f}, Kappa Δ={kappa_delta:+.4f}, "
                  f"Time Δ={time_delta:+.2f}s")
    
    # Variant comparison at same epoch counts
    print("\n### Completion-Only vs Normal SFT (Same Epochs)")
    print("-" * 80)
    for epoch in [1, 2, 3]:
        comp = next((r for r in completion_only if r['epochs'] == epoch), None)
        full = next((r for r in full_finetune if r['epochs'] == epoch), None)
        
        if comp and full:
            acc_diff = full['accuracy'] - comp['accuracy']
            kappa_diff = full['kappa'] - comp['kappa']
            time_diff = full['train_time_sec'] - comp['train_time_sec']
            
            winner_acc = "Normal SFT" if acc_diff > 0 else "Completion-Only"
            winner_kappa = "Normal SFT" if kappa_diff > 0 else "Completion-Only"
            
            print(f"\nEpoch {epoch}:")
            print(f"  Accuracy: Completion={comp['accuracy']:.4f}, Normal={full['accuracy']:.4f} "
                  f"(Δ={acc_diff:+.4f}, Winner: {winner_acc})")
            print(f"  Kappa: Completion={comp['kappa']:.4f}, Normal={full['kappa']:.4f} "
                  f"(Δ={kappa_diff:+.4f}, Winner: {winner_kappa})")
            print(f"  Train Time: Completion={comp['train_time_sec']:.2f}s, Normal={full['train_time_sec']:.2f}s "
                  f"(Δ={time_diff:+.2f}s)")
    
    print("\n" + "="*80)


def create_csv_export(data, output_dir):
    """Export results to CSV for further analysis."""
    import csv
    
    csv_path = os.path.join(output_dir, 'comparison_results.csv')
    
    if not data['results']:
        print("No results to export.")
        return
    
    fieldnames = [
        'experiment_name', 'training_variant', 'epochs', 'accuracy', 
        'kappa', 'score', 'mean_score', 'train_time_sec', 
        'inference_time_per_sample_ms', 'params_M', 'checkpoint_path'
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in data['results']:
            row = {k: result.get(k, '') for k in fieldnames}
            writer.writerow(row)
    
    print(f"\nCSV export saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze SFT comparison results")
    parser.add_argument(
        "--report_dir",
        type=str,
        default=None,
        help="Specific comparison results directory to analyze (default: latest)"
    )
    parser.add_argument(
        "--export_csv",
        action="store_true",
        help="Export results to CSV"
    )
    args = parser.parse_args()
    
    if args.report_dir:
        report_path = os.path.join(args.report_dir, 'comparison_report.json')
        if not os.path.exists(report_path):
            print(f"Report file not found: {report_path}")
            return
        with open(report_path, 'r') as f:
            data = json.load(f)
        output_dir = args.report_dir
        print(f"Loaded comparison from: {args.report_dir}")
    else:
        result = load_latest_comparison()
        if result is None:
            return
        data, output_dir = result
    
    analyze_results(data)
    
    if args.export_csv:
        create_csv_export(data, output_dir)


if __name__ == "__main__":
    main()
