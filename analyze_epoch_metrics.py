import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import sys

# Configuration
MODEL_PATH = "models/qwen3_cross_encoder_sft_10epochs_final"
TRAINING_HISTORY_FILE = f"{MODEL_PATH}/training_history.csv"

def load_training_history():
    """Load training history from CSV file."""
    if not os.path.exists(TRAINING_HISTORY_FILE):
        print(f"Error: Training history file not found at {TRAINING_HISTORY_FILE}")
        print("Please run the training script first to generate the training history.")
        sys.exit(1)
    
    df = pd.read_csv(TRAINING_HISTORY_FILE)
    print(f"Loaded training history with {len(df)} log entries\n")
    return df

def extract_epoch_metrics(df):
    """Extract metrics for each epoch from training history."""
    # Filter rows that have epoch information (these are evaluation logs)
    epoch_logs = df[df['epoch'].notna()].copy()
    
    # Filter for rows that have evaluation metrics
    eval_logs = epoch_logs[epoch_logs['eval_loss'].notna()].copy()
    
    if len(eval_logs) == 0:
        print("Warning: No evaluation logs found in training history")
        return None
    
    # Sort by epoch
    eval_logs = eval_logs.sort_values('epoch')
    
    return eval_logs

def display_epoch_metrics(eval_logs):
    """Display metrics for each epoch in a formatted table."""
    print("="*100)
    print("EPOCH-BY-EPOCH METRICS")
    print("="*100)
    
    # Select relevant columns
    columns_to_display = ['epoch', 'eval_loss', 'eval_accuracy', 'eval_kappa']
    
    # Add optional columns if they exist
    if 'eval_on_topic_recall' in eval_logs.columns:
        columns_to_display.append('eval_on_topic_recall')
    if 'eval_off_topic_recall' in eval_logs.columns:
        columns_to_display.append('eval_off_topic_recall')
    
    # Filter to only include columns that exist
    available_columns = [col for col in columns_to_display if col in eval_logs.columns]
    
    display_df = eval_logs[available_columns].copy()
    display_df['epoch'] = display_df['epoch'].astype(int)
    
    print(display_df.to_string(index=False))
    print()
    
    return display_df

def find_best_epoch(eval_logs):
    """Find the epoch with best performance."""
    print("="*100)
    print("BEST EPOCH ANALYSIS")
    print("="*100)
    
    # Find best epoch by different metrics
    best_by_accuracy = eval_logs.loc[eval_logs['eval_accuracy'].idxmax()]
    best_by_kappa = eval_logs.loc[eval_logs['eval_kappa'].idxmax()]
    best_by_loss = eval_logs.loc[eval_logs['eval_loss'].idxmin()]
    
    print(f"\nBest Epoch by Accuracy: Epoch {int(best_by_accuracy['epoch'])}")
    print(f"  Accuracy: {best_by_accuracy['eval_accuracy']:.4f}")
    print(f"  Kappa: {best_by_accuracy['eval_kappa']:.4f}")
    print(f"  Loss: {best_by_accuracy['eval_loss']:.4f}")
    
    print(f"\nBest Epoch by Kappa: Epoch {int(best_by_kappa['epoch'])}")
    print(f"  Accuracy: {best_by_kappa['eval_accuracy']:.4f}")
    print(f"  Kappa: {best_by_kappa['eval_kappa']:.4f}")
    print(f"  Loss: {best_by_kappa['eval_loss']:.4f}")
    
    print(f"\nBest Epoch by Loss (lowest): Epoch {int(best_by_loss['epoch'])}")
    print(f"  Accuracy: {best_by_loss['eval_accuracy']:.4f}")
    print(f"  Kappa: {best_by_loss['eval_kappa']:.4f}")
    print(f"  Loss: {best_by_loss['eval_loss']:.4f}")
    print()
    
    return {
        'best_accuracy_epoch': int(best_by_accuracy['epoch']),
        'best_kappa_epoch': int(best_by_kappa['epoch']),
        'best_loss_epoch': int(best_by_loss['epoch'])
    }

def plot_metrics(eval_logs):
    """Create visualization of metrics across epochs."""
    epochs = eval_logs['epoch'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Accuracy
    ax1 = axes[0, 0]
    ax1.plot(epochs, eval_logs['eval_accuracy'], 'o-', linewidth=2, markersize=6, color='#2E86AB')
    best_acc_idx = eval_logs['eval_accuracy'].idxmax()
    best_acc_epoch = eval_logs.loc[best_acc_idx, 'epoch']
    best_acc_value = eval_logs.loc[best_acc_idx, 'eval_accuracy']
    ax1.axvline(best_acc_epoch, color='red', linestyle='--', alpha=0.5, 
                label=f'Best: Epoch {int(best_acc_epoch)}')
    ax1.scatter([best_acc_epoch], [best_acc_value], color='red', s=100, zorder=5)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Accuracy vs Epoch', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)
    
    # Plot 2: Kappa
    ax2 = axes[0, 1]
    ax2.plot(epochs, eval_logs['eval_kappa'], 'o-', linewidth=2, markersize=6, color='#A23B72')
    best_kappa_idx = eval_logs['eval_kappa'].idxmax()
    best_kappa_epoch = eval_logs.loc[best_kappa_idx, 'epoch']
    best_kappa_value = eval_logs.loc[best_kappa_idx, 'eval_kappa']
    ax2.axvline(best_kappa_epoch, color='red', linestyle='--', alpha=0.5,
                label=f'Best: Epoch {int(best_kappa_epoch)}')
    ax2.scatter([best_kappa_epoch], [best_kappa_value], color='red', s=100, zorder=5)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel("Cohen's Kappa", fontsize=11)
    ax2.set_title("Kappa vs Epoch", fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    
    # Plot 3: Loss
    ax3 = axes[1, 0]
    ax3.plot(epochs, eval_logs['eval_loss'], 'o-', linewidth=2, markersize=6, color='#F18F01')
    best_loss_idx = eval_logs['eval_loss'].idxmin()
    best_loss_epoch = eval_logs.loc[best_loss_idx, 'epoch']
    best_loss_value = eval_logs.loc[best_loss_idx, 'eval_loss']
    ax3.axvline(best_loss_epoch, color='red', linestyle='--', alpha=0.5,
                label=f'Best: Epoch {int(best_loss_epoch)}')
    ax3.scatter([best_loss_epoch], [best_loss_value], color='red', s=100, zorder=5)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Evaluation Loss', fontsize=11)
    ax3.set_title('Loss vs Epoch', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(epochs)
    
    # Plot 4: Recall metrics (if available)
    ax4 = axes[1, 1]
    if 'eval_on_topic_recall' in eval_logs.columns and 'eval_off_topic_recall' in eval_logs.columns:
        ax4.plot(epochs, eval_logs['eval_on_topic_recall'], 'o-', linewidth=2, markersize=6, 
                color='#06A77D', label='On-Topic Recall')
        ax4.plot(epochs, eval_logs['eval_off_topic_recall'], 's-', linewidth=2, markersize=6, 
                color='#D62246', label='Off-Topic Recall')
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('Recall', fontsize=11)
        ax4.set_title('Recall Metrics vs Epoch', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(epochs)
    else:
        # If recall metrics not available, plot accuracy and kappa together
        ax4.plot(epochs, eval_logs['eval_accuracy'], 'o-', linewidth=2, markersize=6, 
                color='#2E86AB', label='Accuracy')
        ax4.plot(epochs, eval_logs['eval_kappa'], 's-', linewidth=2, markersize=6, 
                color='#A23B72', label='Kappa')
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('Score', fontsize=11)
        ax4.set_title('Accuracy & Kappa vs Epoch', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(epochs)
    
    plt.tight_layout()
    
    # Save plot
    output_path = f"{MODEL_PATH}/epoch_metrics_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Metrics plot saved to: {output_path}")
    
    return output_path

def calculate_improvement_stats(eval_logs):
    """Calculate improvement statistics from first to best epoch."""
    print("="*100)
    print("IMPROVEMENT STATISTICS")
    print("="*100)
    
    first_epoch = eval_logs.iloc[0]
    best_kappa_idx = eval_logs['eval_kappa'].idxmax()
    best_epoch = eval_logs.loc[best_kappa_idx]
    
    acc_improvement = best_epoch['eval_accuracy'] - first_epoch['eval_accuracy']
    kappa_improvement = best_epoch['eval_kappa'] - first_epoch['eval_kappa']
    loss_improvement = first_epoch['eval_loss'] - best_epoch['eval_loss']
    
    print(f"\nFrom Epoch {int(first_epoch['epoch'])} to Epoch {int(best_epoch['epoch'])} (best by kappa):")
    print(f"  Accuracy:  {first_epoch['eval_accuracy']:.4f} → {best_epoch['eval_accuracy']:.4f} ({acc_improvement:+.4f})")
    print(f"  Kappa:     {first_epoch['eval_kappa']:.4f} → {best_epoch['eval_kappa']:.4f} ({kappa_improvement:+.4f})")
    print(f"  Loss:      {first_epoch['eval_loss']:.4f} → {best_epoch['eval_loss']:.4f} ({-loss_improvement:+.4f})")
    print()
    
    return {
        'first_epoch': int(first_epoch['epoch']),
        'best_epoch': int(best_epoch['epoch']),
        'accuracy_improvement': float(acc_improvement),
        'kappa_improvement': float(kappa_improvement),
        'loss_improvement': float(loss_improvement)
    }

def save_analysis_summary(eval_logs, best_epochs, improvement_stats):
    """Save analysis summary to JSON file."""
    summary = {
        'total_epochs': int(eval_logs['epoch'].max()),
        'best_epochs': best_epochs,
        'improvement_stats': improvement_stats,
        'final_epoch_metrics': {
            'epoch': int(eval_logs.iloc[-1]['epoch']),
            'accuracy': float(eval_logs.iloc[-1]['eval_accuracy']),
            'kappa': float(eval_logs.iloc[-1]['eval_kappa']),
            'loss': float(eval_logs.iloc[-1]['eval_loss'])
        }
    }
    
    output_path = f"{MODEL_PATH}/epoch_analysis_summary.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Analysis summary saved to: {output_path}")

def main():
    print("\n" + "="*100)
    print("EPOCH METRICS ANALYSIS")
    print("="*100 + "\n")
    
    # Load training history
    df = load_training_history()
    
    # Extract epoch metrics
    eval_logs = extract_epoch_metrics(df)
    
    if eval_logs is None or len(eval_logs) == 0:
        print("No evaluation metrics found in training history.")
        return
    
    # Display metrics
    display_df = display_epoch_metrics(eval_logs)
    
    # Find best epochs
    best_epochs = find_best_epoch(eval_logs)
    
    # Calculate improvements
    improvement_stats = calculate_improvement_stats(eval_logs)
    
    # Create visualizations
    plot_metrics(eval_logs)
    
    # Save summary
    save_analysis_summary(eval_logs, best_epochs, improvement_stats)
    
    # Export epoch metrics to CSV
    output_csv = f"{MODEL_PATH}/epoch_metrics_summary.csv"
    display_df.to_csv(output_csv, index=False)
    print(f"Epoch metrics exported to: {output_csv}")
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE!")
    print("="*100 + "\n")

if __name__ == "__main__":
    main()
