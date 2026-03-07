"""
Comprehensive Checkpoint Analysis Script
Evaluates all 20 epoch checkpoints and generates detailed analysis for each.
"""

import pandas as pd
import numpy as np
import torch
import os
import json
import time
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from peft import PeftModel
from datasets import Dataset
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    cohen_kappa_score,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_NAME = "Qwen/Qwen3-4B"
MAX_LENGTH = 4096
# Two separate checkpoint directories
CHECKPOINT_DIR_EPOCHS_1_10 = "./qwen3_cross_encoder_sft_10epochs_v3"  # First 10 epochs
CHECKPOINT_DIR_EPOCHS_11_20 = "./qwen3_cross_encoder_sft_20epochs_v1"  # Next 10 epochs
RESULTS_DIR = "./checkpoint_analysis_results"
TEST_DATA_PATH = "data/test_0216.xlsx"

# System prompt (same as training)
SYSTEM_PROMPT = """You are an expert at detecting whether the test-taker's response is strictly associated to the given prompt topic, prompt sub-topic and prompt script or not. Being off topic refers to the test-taker's response being either totally or atleast majorly unrelated to the prompt topic, prompt sub-topic and prompt script given to the test-taker. The test-taker is in a setting, where to pass, they have to genuinely provide a response based on the prompt topic, prompt sub-topic and prompt script given to them.
GENERIC GUIDELINE FOR EVALUATION:
Focus on the overall essence of the response rather than rigid criteria. Filler words and disfluencies are not any criteria for awarding off topic flag as true.
There are two proficiency levels for which testing is being done. These two proficiency levels (along with their short forms in brackets) are given in descending order of strictness - 1) Advanced (A), 2) Intermediate (I). The more strict the testing proficiency level is the more stricter are the criteria for evaluating the off-topic flag.
Consider the context: Does the test-taker's response feel naturally connected to the specific prompt topic, prompt sub-topic and prompt script? Doesthe test-taker's response seems to be answered accordingly to the prompt topic, prompt sub-topic and prompt script? Or does it feel like the test-taker is trying to just provide a response which has no logical connection or majorly related to the prompt topic, prompt sub-topic and prompt script?
Use logical reasoning based on the given prompt topic, prompt sub-topic, prompt script, test-taker's response, to guide your judgment."""

# Checkpoint to Epoch mapping (361 steps per epoch)
CHECKPOINT_MAPPING = {
    361: 1, 722: 2, 1083: 3, 1444: 4, 1805: 5,
    2166: 6, 2527: 7, 2888: 8, 3249: 9, 3610: 10,
    3971: 11, 4332: 12, 4693: 13, 5054: 14, 5415: 15,
    5776: 16, 6137: 17, 6498: 18, 6859: 19, 7220: 20
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'='*100}")
print(f"COMPREHENSIVE CHECKPOINT ANALYSIS")
print(f"{'='*100}")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"PyTorch version: {torch.__version__}")
print(f"Base Model: {MODEL_NAME}")
print(f"Checkpoints to analyze: {len(CHECKPOINT_MAPPING)}")
print(f"Checkpoint directories:")
print(f"  - Epochs 1-10: {CHECKPOINT_DIR_EPOCHS_1_10}")
print(f"  - Epochs 11-20: {CHECKPOINT_DIR_EPOCHS_11_20}")
print(f"{'='*100}\n")

def get_checkpoint_directory(epoch):
    """Determine which checkpoint directory to use based on epoch number."""
    if epoch <= 10:
        return CHECKPOINT_DIR_EPOCHS_1_10
    else:
        return CHECKPOINT_DIR_EPOCHS_11_20

def build_cross_encoder_inputs(examples, tokenizer):
    """Build inputs for Qwen3 sequence classification."""
    formatted_texts = []
    for i in range(len(examples["Prompt Script"])):
        user_content = f"""You are now given the below data:
Prompt Topic: {examples['Prompt Topic'][i]}
Prompt Sub-Topic: {examples['Prompt Sub Topic'][i]}
Prompt Script: {examples['Prompt Script'][i]}
Test-taker's response: {examples['Machine Transciption'][i]}
Testing Proficiency Level: {examples['Level'][i]}

Based on your expertise, determine if the test-taker's response is off-topic or not."""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False
        )
        formatted_texts.append(formatted_text)
    
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    
    return tokenized

def load_test_data():
    """Load and prepare test dataset."""
    print("Loading test data...")
    test_df = pd.read_excel(TEST_DATA_PATH)
    test_df["label"] = test_df["Human_Combined"]
    print(f"Test set: {len(test_df)} samples")
    print(f"Label distribution: {test_df['label'].value_counts().to_dict()}")
    return test_df

def prepare_test_dataset(test_df, tokenizer):
    """Prepare test dataset for inference."""
    cols_to_keep = ["Prompt Topic", "Prompt Sub Topic", "Prompt Script", "Machine Transciption", "Level", "label"]
    test_hf = Dataset.from_pandas(test_df[cols_to_keep], preserve_index=False)
    
    # Tokenize
    test_hf = test_hf.map(
        lambda x: build_cross_encoder_inputs(x, tokenizer),
        batched=True,
        remove_columns=["Prompt Topic", "Prompt Sub Topic", "Prompt Script", "Machine Transciption", "Level"]
    )
    test_hf = test_hf.rename_column("label", "labels")
    
    return test_hf

def run_inference(model, tokenizer, test_hf, batch_size=8):
    """Run inference on test set and return predictions and metrics."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    from torch.utils.data import DataLoader
    data_collator = DataCollatorWithPadding(tokenizer, padding=True)
    dataloader = DataLoader(test_hf, batch_size=batch_size, collate_fn=data_collator)
    
    print(f"Running inference on {len(test_hf)} samples...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            # Move batch to device
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels']
            
            # Get predictions
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (off-topic)
    
    inference_time = time.time() - start_time
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probs), inference_time

def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate comprehensive metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Precision, Recall, F1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1]
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    class_report = classification_report(
        y_true, y_pred, 
        target_names=["On-Topic (0)", "Off-Topic (1)"],
        digits=4
    )
    
    metrics = {
        "accuracy": float(accuracy),
        "kappa": float(kappa),
        "on_topic_precision": float(precision[0]),
        "on_topic_recall": float(recall[0]),
        "on_topic_f1": float(f1[0]),
        "off_topic_precision": float(precision[1]),
        "off_topic_recall": float(recall[1]),
        "off_topic_f1": float(f1[1]),
        "confusion_matrix": cm.tolist(),
        "classification_report": class_report,
        "mean_confidence": float(np.mean(y_probs)),
        "std_confidence": float(np.std(y_probs))
    }
    
    return metrics

def save_epoch_results(epoch, checkpoint_num, metrics, predictions, labels, probs, test_df, inference_time):
    """Save detailed results for a specific epoch."""
    epoch_dir = os.path.join(RESULTS_DIR, f"epoch_{epoch:02d}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Save metrics JSON
    metrics_with_meta = {
        "epoch": epoch,
        "checkpoint": checkpoint_num,
        "inference_time_sec": inference_time,
        "inference_time_per_sample_ms": (inference_time / len(predictions)) * 1000,
        **metrics
    }
    
    with open(os.path.join(epoch_dir, "metrics.json"), "w") as f:
        json.dump(metrics_with_meta, f, indent=2)
    
    # Save detailed predictions CSV
    results_df = test_df.copy()
    results_df["predicted_label"] = predictions
    results_df["true_label"] = labels
    results_df["prediction_probability"] = probs
    results_df["correct"] = (predictions == labels)
    results_df.to_csv(os.path.join(epoch_dir, "detailed_predictions.csv"), index=False)
    
    # Save confusion matrix visualization
    plt.figure(figsize=(8, 6))
    cm = np.array(metrics["confusion_matrix"])
    im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im)
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black' if cm[i, j] < cm.max()/2 else 'white', fontsize=14)
    
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["On-Topic", "Off-Topic"])
    plt.yticks(tick_marks, ["On-Topic", "Off-Topic"])
    plt.title(f"Confusion Matrix - Epoch {epoch}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(epoch_dir, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save classification report to text file
    with open(os.path.join(epoch_dir, "classification_report.txt"), "w") as f:
        f.write(f"Epoch {epoch} - Checkpoint {checkpoint_num}\n")
        f.write("="*80 + "\n\n")
        f.write(metrics["classification_report"])
        f.write("\n\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Cohen's Kappa: {metrics['kappa']:.4f}\n")
        f.write(f"Inference Time: {inference_time:.2f}s ({metrics_with_meta['inference_time_per_sample_ms']:.2f}ms/sample)\n")
    
    print(f"  ✓ Results saved to {epoch_dir}")

def create_summary_report(all_results):
    """Create comprehensive summary comparing all epochs."""
    summary_dir = RESULTS_DIR
    
    # Create summary DataFrame
    summary_data = []
    for result in all_results:
        summary_data.append({
            "Epoch": result["epoch"],
            "Checkpoint": result["checkpoint"],
            "Accuracy": result["accuracy"],
            "Kappa": result["kappa"],
            "On-Topic F1": result["on_topic_f1"],
            "Off-Topic F1": result["off_topic_f1"],
            "On-Topic Recall": result["on_topic_recall"],
            "Off-Topic Recall": result["off_topic_recall"],
            "Inference Time (s)": result["inference_time_sec"],
            "Time/Sample (ms)": result["inference_time_per_sample_ms"]
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(summary_dir, "epoch_comparison_summary.csv"), index=False)
    
    # Find best epochs
    best_accuracy_epoch = summary_df.loc[summary_df["Accuracy"].idxmax()]
    best_kappa_epoch = summary_df.loc[summary_df["Kappa"].idxmax()]
    best_off_topic_f1_epoch = summary_df.loc[summary_df["Off-Topic F1"].idxmax()]
    
    # Create summary text report
    with open(os.path.join(summary_dir, "ANALYSIS_SUMMARY.txt"), "w") as f:
        f.write("="*100 + "\n")
        f.write("COMPREHENSIVE CHECKPOINT ANALYSIS SUMMARY\n")
        f.write("="*100 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Epochs Analyzed: {len(all_results)}\n")
        f.write(f"Base Model: {MODEL_NAME}\n\n")
        
        f.write("="*100 + "\n")
        f.write("BEST PERFORMING EPOCHS\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Best by Accuracy: Epoch {best_accuracy_epoch['Epoch']}\n")
        f.write(f"  Accuracy: {best_accuracy_epoch['Accuracy']:.4f}\n")
        f.write(f"  Kappa: {best_accuracy_epoch['Kappa']:.4f}\n")
        f.write(f"  Off-Topic F1: {best_accuracy_epoch['Off-Topic F1']:.4f}\n\n")
        
        f.write(f"Best by Kappa: Epoch {best_kappa_epoch['Epoch']}\n")
        f.write(f"  Accuracy: {best_kappa_epoch['Accuracy']:.4f}\n")
        f.write(f"  Kappa: {best_kappa_epoch['Kappa']:.4f}\n")
        f.write(f"  Off-Topic F1: {best_kappa_epoch['Off-Topic F1']:.4f}\n\n")
        
        f.write(f"Best by Off-Topic F1: Epoch {best_off_topic_f1_epoch['Epoch']}\n")
        f.write(f"  Accuracy: {best_off_topic_f1_epoch['Accuracy']:.4f}\n")
        f.write(f"  Kappa: {best_off_topic_f1_epoch['Kappa']:.4f}\n")
        f.write(f"  Off-Topic F1: {best_off_topic_f1_epoch['Off-Topic F1']:.4f}\n\n")
        
        f.write("="*100 + "\n")
        f.write("EPOCH-BY-EPOCH COMPARISON\n")
        f.write("="*100 + "\n\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")
        
        # Training progression analysis
        first_epoch = summary_df.iloc[0]
        last_epoch = summary_df.iloc[-1]
        
        f.write("="*100 + "\n")
        f.write("TRAINING PROGRESSION (First vs Last Epoch)\n")
        f.write("="*100 + "\n\n")
        f.write(f"Accuracy Improvement: {first_epoch['Accuracy']:.4f} → {last_epoch['Accuracy']:.4f} ({last_epoch['Accuracy'] - first_epoch['Accuracy']:+.4f})\n")
        f.write(f"Kappa Improvement: {first_epoch['Kappa']:.4f} → {last_epoch['Kappa']:.4f} ({last_epoch['Kappa'] - first_epoch['Kappa']:+.4f})\n")
        f.write(f"Off-Topic F1 Improvement: {first_epoch['Off-Topic F1']:.4f} → {last_epoch['Off-Topic F1']:.4f} ({last_epoch['Off-Topic F1'] - first_epoch['Off-Topic F1']:+.4f})\n")
    
    print(f"\n{'='*100}")
    print(f"Summary report saved to {os.path.join(summary_dir, 'ANALYSIS_SUMMARY.txt')}")
    print(f"{'='*100}\n")
    
    # Create visualizations
    create_comparison_plots(summary_df, summary_dir)

def create_comparison_plots(summary_df, output_dir):
    """Create comprehensive visualizations comparing all epochs."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    epochs = summary_df["Epoch"].values
    
    # Plot 1: Accuracy
    ax = axes[0, 0]
    ax.plot(epochs, summary_df["Accuracy"], 'o-', linewidth=2, markersize=8, color='#2E86AB')
    best_idx = summary_df["Accuracy"].idxmax()
    ax.scatter([summary_df.loc[best_idx, "Epoch"]], [summary_df.loc[best_idx, "Accuracy"]], 
               color='red', s=200, zorder=5, marker='*', label=f'Best: Epoch {summary_df.loc[best_idx, "Epoch"]}')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs Epoch', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Kappa
    ax = axes[0, 1]
    ax.plot(epochs, summary_df["Kappa"], 'o-', linewidth=2, markersize=8, color='#A23B72')
    best_idx = summary_df["Kappa"].idxmax()
    ax.scatter([summary_df.loc[best_idx, "Epoch"]], [summary_df.loc[best_idx, "Kappa"]], 
               color='red', s=200, zorder=5, marker='*', label=f'Best: Epoch {summary_df.loc[best_idx, "Epoch"]}')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel("Cohen's Kappa", fontsize=12)
    ax.set_title("Kappa vs Epoch", fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: F1 Scores
    ax = axes[0, 2]
    ax.plot(epochs, summary_df["On-Topic F1"], 'o-', linewidth=2, markersize=6, 
            color='#06A77D', label='On-Topic F1')
    ax.plot(epochs, summary_df["Off-Topic F1"], 's-', linewidth=2, markersize=6, 
            color='#D62246', label='Off-Topic F1')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 Scores vs Epoch', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Recall Metrics
    ax = axes[1, 0]
    ax.plot(epochs, summary_df["On-Topic Recall"], 'o-', linewidth=2, markersize=6, 
            color='#06A77D', label='On-Topic Recall')
    ax.plot(epochs, summary_df["Off-Topic Recall"], 's-', linewidth=2, markersize=6, 
            color='#D62246', label='Off-Topic Recall')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Recall Metrics vs Epoch', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Inference Time
    ax = axes[1, 1]
    ax.plot(epochs, summary_df["Time/Sample (ms)"], 'o-', linewidth=2, markersize=8, color='#F18F01')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Time per Sample (ms)', fontsize=12)
    ax.set_title('Inference Speed vs Epoch', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Accuracy & Kappa Combined
    ax = axes[1, 2]
    ax2 = ax.twinx()
    line1 = ax.plot(epochs, summary_df["Accuracy"], 'o-', linewidth=2, markersize=6, 
                    color='#2E86AB', label='Accuracy')
    line2 = ax2.plot(epochs, summary_df["Kappa"], 's-', linewidth=2, markersize=6, 
                     color='#A23B72', label='Kappa')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12, color='#2E86AB')
    ax2.set_ylabel("Cohen's Kappa", fontsize=12, color='#A23B72')
    ax.set_title('Accuracy & Kappa Combined', fontsize=13, fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#2E86AB')
    ax2.tick_params(axis='y', labelcolor='#A23B72')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, "comprehensive_metrics_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path}")

def main():
    """Main execution function."""
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load test data
    test_df = load_test_data()
    
    # Load tokenizer (only once)
    print(f"\nLoading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Set padding token for Qwen3 (required for batch processing)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    # Prepare test dataset
    test_hf = prepare_test_dataset(test_df, tokenizer)
    
    all_results = []
    
    print(f"\n{'='*100}")
    print(f"Starting checkpoint analysis...")
    print(f"{'='*100}\n")
    
    # Process each checkpoint
    for checkpoint_num, epoch in sorted(CHECKPOINT_MAPPING.items()):
        # Determine correct checkpoint directory based on epoch
        checkpoint_dir = get_checkpoint_directory(epoch)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{checkpoint_num}")
        
        print(f"\n{'─'*100}")
        print(f"Epoch {epoch:02d} | Checkpoint {checkpoint_num} | {checkpoint_path}")
        print(f"{'─'*100}")
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"  ✗ Checkpoint not found: {checkpoint_path}")
            print(f"  Skipping epoch {epoch}...")
            continue
        
        try:
            # Load model
            print(f"  Loading model...")
            base_model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=2,
                torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            )
            
            # Set pad_token_id in model config to match tokenizer
            base_model.config.pad_token_id = tokenizer.pad_token_id
            
            # Load LoRA weights
            model = PeftModel.from_pretrained(base_model, checkpoint_path)
            model = model.to(device)
            model.eval()
            
            # Run inference
            predictions, labels, probs, inference_time = run_inference(model, tokenizer, test_hf)
            
            # Calculate metrics
            print(f"  Calculating metrics...")
            metrics = calculate_metrics(labels, predictions, probs)
            
            # Display key metrics
            print(f"  Accuracy: {metrics['accuracy']:.4f} | Kappa: {metrics['kappa']:.4f} | Off-Topic F1: {metrics['off_topic_f1']:.4f}")
            print(f"  Inference: {inference_time:.2f}s ({(inference_time/len(predictions))*1000:.2f}ms/sample)")
            
            # Save results
            save_epoch_results(
                epoch, checkpoint_num, metrics, 
                predictions, labels, probs, test_df, inference_time
            )
            
            # Store for summary
            all_results.append({
                "epoch": epoch,
                "checkpoint": checkpoint_num,
                "inference_time_sec": inference_time,
                "inference_time_per_sample_ms": (inference_time / len(predictions)) * 1000,
                **metrics
            })
            
            # Cleanup to free memory
            del model, base_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ✗ Error processing checkpoint: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create summary report
    if all_results:
        print(f"\n{'='*100}")
        print(f"Creating summary report...")
        print(f"{'='*100}")
        create_summary_report(all_results)
        
        print(f"\n{'='*100}")
        print(f"ANALYSIS COMPLETE!")
        print(f"{'='*100}")
        print(f"Total epochs analyzed: {len(all_results)}")
        print(f"Results directory: {RESULTS_DIR}")
        print(f"{'='*100}\n")
    else:
        print(f"\n✗ No checkpoints were successfully analyzed.")

if __name__ == "__main__":
    main()
