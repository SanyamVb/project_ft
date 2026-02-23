"""Analyze and visualize comparison results from DeBERTa experiments."""

import json
import os
import argparse
from pathlib import Path


def load_latest_comparison():
    """Load the most recent comparison report."""
    # Find all deberta_comparison_results directories
    comparison_dirs = [d for d in os.listdir('.') if d.startswith('deberta_comparison_results_')]
    
    if not comparison_dirs:
        print("No DeBERTa comparison results found. Run compare_deberta_variants.py first.")
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
    print("ANALYSIS: DeBERTa Small vs Base vs Large")
    print("="*80)
    
    # Group results by model size
    small_results = [r for r in results if r['model_size'] == 'small']
    base_results = [r for r in results if r['model_size'] == 'base']
    large_results = [r for r in results if r['model_size'] == 'large']
    
    if small_results:
        print("\n### DeBERTa-Small")
        print("-" * 80)
        for r in sorted(small_results, key=lambda x: x['epochs']):
            print(f"Epoch {r['epochs']}: Accuracy={r['accuracy']:.4f}, Kappa={r['kappa']:.4f}, "
                  f"Mean Score={r['mean_score']:.2f}, Train Time={r['train_time_sec']:.2f}s, "
                  f"Params={r['params_M']:.1f}M")
    
    if base_results:
        print("\n### DeBERTa-Base")
        print("-" * 80)
        for r in sorted(base_results, key=lambda x: x['epochs']):
            print(f"Epoch {r['epochs']}: Accuracy={r['accuracy']:.4f}, Kappa={r['kappa']:.4f}, "
                  f"Mean Score={r['mean_score']:.2f}, Train Time={r['train_time_sec']:.2f}s, "
                  f"Params={r['params_M']:.1f}M")
    
    if large_results:
        print("\n### DeBERTa-Large")
        print("-" * 80)
        for r in sorted(large_results, key=lambda x: x['epochs']):
            print(f"Epoch {r['epochs']}: Accuracy={r['accuracy']:.4f}, Kappa={r['kappa']:.4f}, "
                  f"Mean Score={r['mean_score']:.2f}, Train Time={r['train_time_sec']:.2f}s, "
                  f"Params={r['params_M']:.1f}M")
    
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
    
    # Epoch impact analysis for each model size
    for model_size, model_results in [
        ('small', small_results),
        ('base', base_results),
        ('large', large_results)
    ]:
        if model_results and len(model_results) > 1:
            print(f"\n### {model_size.upper()}: Impact of Epochs")
            print("-" * 80)
            sorted_results = sorted(model_results, key=lambda x: x['epochs'])
            for i in range(1, len(sorted_results)):
                prev = sorted_results[i-1]
                curr = sorted_results[i]
                acc_delta = curr['accuracy'] - prev['accuracy']
                kappa_delta = curr['kappa'] - prev['kappa']
                time_delta = curr['train_time_sec'] - prev['train_time_sec']
                print(f"Epoch {prev['epochs']} → {curr['epochs']}: "
                      f"Accuracy Δ={acc_delta:+.4f}, Kappa Δ={kappa_delta:+.4f}, "
                      f"Time Δ={time_delta:+.2f}s")
    
    # Model size comparison at same epoch counts
    print("\n### Model Size Comparison (Same Epochs)")
    print("-" * 80)
    for epoch in [1, 2, 3]:
        print(f"\n** Epoch {epoch} **")
        epoch_results = [r for r in results if r['epochs'] == epoch]
        
        if not epoch_results:
            continue
        
        # Sort by accuracy
        epoch_results_by_acc = sorted(epoch_results, key=lambda x: x['accuracy'], reverse=True)
        epoch_results_by_kappa = sorted(epoch_results, key=lambda x: x['kappa'], reverse=True)
        
        print("\nBy Accuracy:")
        for r in epoch_results_by_acc:
            print(f"  {r['model_size']:>6}: Accuracy={r['accuracy']:.4f}, Params={r['params_M']:.1f}M")
        
        print("\nBy Kappa:")
        for r in epoch_results_by_kappa:
            print(f"  {r['model_size']:>6}: Kappa={r['kappa']:.4f}, Params={r['params_M']:.1f}M")
    
    # Performance vs Model Size Analysis
    print("\n### Performance vs Model Size")
    print("-" * 80)
    
    # For each epoch, compare small vs base vs large
    for epoch in [1, 2, 3]:
        small = next((r for r in small_results if r['epochs'] == epoch), None)
        base = next((r for r in base_results if r['epochs'] == epoch), None)
        large = next((r for r in large_results if r['epochs'] == epoch), None)
        
        if small and base:
            acc_gain_base = base['accuracy'] - small['accuracy']
            kappa_gain_base = base['kappa'] - small['kappa']
            time_cost_base = base['train_time_sec'] - small['train_time_sec']
            
            print(f"\nEpoch {epoch} - Small → Base:")
            print(f"  Accuracy gain: {acc_gain_base:+.4f}")
            print(f"  Kappa gain: {kappa_gain_base:+.4f}")
            print(f"  Time cost: {time_cost_base:+.2f}s")
            print(f"  Params increase: {base['params_M'] - small['params_M']:.1f}M")
        
        if base and large:
            acc_gain_large = large['accuracy'] - base['accuracy']
            kappa_gain_large = large['kappa'] - base['kappa']
            time_cost_large = large['train_time_sec'] - base['train_time_sec']
            
            print(f"\nEpoch {epoch} - Base → Large:")
            print(f"  Accuracy gain: {acc_gain_large:+.4f}")
            print(f"  Kappa gain: {kappa_gain_large:+.4f}")
            print(f"  Time cost: {time_cost_large:+.2f}s")
            print(f"  Params increase: {large['params_M'] - base['params_M']:.1f}M")
    
    # Value for money analysis
    print("\n### Efficiency Analysis (Kappa per Training Minute)")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x['kappa'] / (x['train_time_sec'] / 60), reverse=True):
        efficiency = r['kappa'] / (r['train_time_sec'] / 60) if r['train_time_sec'] > 0 else 0
        print(f"{r['experiment_name']:<30}: {efficiency:.4f} kappa/min "
              f"(Kappa={r['kappa']:.4f}, Time={r['train_time_sec']/60:.2f}min)")
    
    print("\n" + "="*80)


def create_csv_export(data, output_dir):
    """Export results to CSV for further analysis."""
    import csv
    
    csv_path = os.path.join(output_dir, 'comparison_results.csv')
    
    if not data['results']:
        print("No results to export.")
        return
    
    fieldnames = [
        'experiment_name', 'model_size', 'epochs', 'params_M',
        'accuracy', 'kappa', 'score', 'mean_score', 'best_threshold',
        'train_time_sec', 'inference_time_per_sample_ms', 'checkpoint_path'
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in data['results']:
            row = {k: result.get(k, '') for k in fieldnames}
            writer.writerow(row)
    
    print(f"\nCSV export saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze DeBERTa comparison results")
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
