import pandas as pd
import numpy as np
import torch
import os
import time
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import PeftModel
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

# Using Qwen3 instead of DeBERTa
# Available models: "Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B"
# Set MODEL_SIZE to either "0.6B" or "4B"
MODEL_SIZE = "4B"  # Change this to "0.6B" or "4B"
MODEL_NAME = f"Qwen/Qwen3-{MODEL_SIZE}"
MAX_LENGTH = 4096

# System prompt from qwen_pipeline
SYSTEM_PROMPT = """You are an expert at detecting whether the test-taker's response is strictly associated to the given prompt topic, prompt sub-topic and prompt script or not. Being off topic refers to the test-taker's response being either totally or atleast majorly unrelated to the prompt topic, prompt sub-topic and prompt script given to the test-taker. The test-taker is in a setting, where to pass, they have to genuinely provide a response based on the prompt topic, prompt sub-topic and prompt script given to them.
GENERIC GUIDELINE FOR EVALUATION:
Focus on the overall essence of the response rather than rigid criteria. Filler words and disfluencies are not any criteria for awarding off topic flag as true.
There are two proficiency levels for which testing is being done. These two proficiency levels (along with their short forms in brackets) are given in descending order of strictness - 1) Advanced (A), 2) Intermediate (I). The more strict the testing proficiency level is the more stricter are the criteria for evaluating the off-topic flag.
Consider the context: Does the test-taker's response feel naturally connected to the specific prompt topic, prompt sub-topic and prompt script? Doesthe test-taker's response seems to be answered accordingly to the prompt topic, prompt sub-topic and prompt script? Or does it feel like the test-taker is trying to just provide a response which has no logical connection or majorly related to the prompt topic, prompt sub-topic and prompt script?
Use logical reasoning based on the given prompt topic, prompt sub-topic, prompt script, test-taker's response, to guide your judgment."""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"PyTorch version: {torch.__version__}")
print(f"\n{'='*80}")
print(f"RUNNING ARGMAX INFERENCE FOR MODEL: {MODEL_NAME}")
print(f"{'='*80}\n")

# Load data
test_df = pd.read_excel("data/test_0216.xlsx")

label_map = {0: 0, 1: 1}  # Human_Combined is already 0/1
test_df["label"] = test_df["Human_Combined"]

print(f"Test:  {len(test_df)} rows — {test_df['label'].value_counts().to_dict()}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Important: Qwen3 models need a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

def build_cross_encoder_inputs(examples):
    """
    Build inputs for Qwen3 sequence classification using qwen_pipeline prompt format.
    Combines SYSTEM_PROMPT + user prompt as text_a, and response as text_b.
    """
    prompts = []
    for i in range(len(examples["Prompt Script"])):
        user_prompt = f"""You are now given the below data:
Prompt Topic: {examples['Prompt Topic'][i]}
Prompt Sub-Topic: {examples['Prompt Sub Topic'][i]}
Prompt Script: {examples['Prompt Script'][i]}
Test-taker's response: {examples['Machine Transciption'][i]}
Testing Proficiency Level: {examples['Level'][i]}

Based on your expertise, determine if the test-taker's response is off-topic or not."""
        # Combine system prompt + user prompt
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        prompts.append(full_prompt)
    
    # Tokenize with truncation and padding
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,  # Will be handled by DataCollator
    )
    
    return tokenized

# Prepare datasets with qwen_pipeline columns
cols_to_keep = ["Prompt Topic", "Prompt Sub Topic", "Prompt Script", "Machine Transciption", "Level", "label"]
test_hf = Dataset.from_pandas(test_df[cols_to_keep], preserve_index=False)

test_hf = test_hf.map(
    build_cross_encoder_inputs, 
    batched=True, 
    remove_columns=["Prompt Topic", "Prompt Sub Topic", "Prompt Script", "Machine Transciption", "Level"]
)

print(f"Test features: {test_hf.features}")
print(f"Example token count: {len(test_hf[0]['input_ids'])}")

# Store results for all epoch configurations
all_epoch_results = {}

# Loop through different epoch configurations (load trained models)
for num_epochs in [1, 2, 3, 4, 5, 6]:
    print(f"\n{'='*80}")
    print(f"{'='*80}")
    print(f"LOADING TRAINED MODEL: {num_epochs} EPOCH(S)")
    print(f"{'='*80}")
    print(f"{'='*80}\n")
    
    model_path = f"models/qwen3_cross_encoder_offtopic/{num_epochs}_epoch"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}, skipping...")
        continue
    
    # Load the trained model with LoRA weights
    print(f"Loading model from {model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"Model loaded successfully")

    # Evaluation with timing
    print(f"\n=== Running Inference on Test Set ({num_epochs} epoch(s)) ===")
    
    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer, padding=True)
    
    # Create trainer for inference
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    # Time the inference
    inference_start = time.time()
    preds_output = trainer.predict(test_hf)
    inference_end = time.time()
    
    inference_time = inference_end - inference_start
    num_samples = len(test_hf)
    avg_inference_time_per_sample = inference_time / num_samples
    
    print(f"Total inference time: {inference_time:.2f} seconds")
    print(f"Number of samples: {num_samples}")
    print(f"Average inference time per sample: {avg_inference_time_per_sample*1000:.2f} ms")
    
    logits = preds_output.predictions
    y_true = test_df["label"].values

    # Use argmax for predictions (no threshold optimization)
    y_pred_argmax = np.argmax(logits, axis=-1)

    print(f"\n=== Results with Argmax - {num_epochs} epoch(s) ===")
    print(classification_report(y_true, y_pred_argmax, target_names=["On-Topic", "Off-Topic"]))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred_argmax)
    print(cm)

    kappa_argmax = cohen_kappa_score(y_true, y_pred_argmax)
    acc_argmax = (y_pred_argmax == y_true).mean()
    print(f"Accuracy: {acc_argmax:.4f}")
    print(f"Kappa: {kappa_argmax:.4f}")
    
    # Calculate recall metrics
    on_topic_recall = cm[0][0] / cm[0].sum() if cm[0].sum() > 0 else 0
    off_topic_recall = cm[1][1] / cm[1].sum() if cm[1].sum() > 0 else 0
    print(f"On-Topic Recall: {on_topic_recall:.4f}")
    print(f"Off-Topic Recall: {off_topic_recall:.4f}")
    
    # Store all metrics for this epoch configuration
    all_epoch_results[f"{num_epochs}_epoch"] = {
        "num_epochs": num_epochs,
        "model_name": MODEL_NAME,
        "model_size": MODEL_SIZE,
        "model_path": model_path,
        "inference_metrics": {
            "total_inference_time": float(inference_time),
            "num_samples": num_samples,
            "avg_inference_time_per_sample_ms": float(avg_inference_time_per_sample * 1000),
        },
        "argmax_metrics": {
            "accuracy": float(acc_argmax),
            "kappa": float(kappa_argmax),
            "on_topic_recall": float(on_topic_recall),
            "off_topic_recall": float(off_topic_recall),
            "confusion_matrix": cm.tolist(),
        }
    }
    
    # Save individual epoch metrics
    os.makedirs(f"{model_path}/argmax_results", exist_ok=True)
    with open(f"{model_path}/argmax_results/metrics.json", "w") as f:
        json.dump(all_epoch_results[f"{num_epochs}_epoch"], f, indent=2)
    
    print(f"\nArgmax metrics saved to {model_path}/argmax_results/metrics.json")
    
    # Clear GPU memory between runs
    if device.type == "cuda":
        del model, trainer
        torch.cuda.empty_cache()
        print("GPU memory cleared")

# Save comparison of all epoch results
print(f"\n{'='*80}")
print("SUMMARY: Argmax Results Across All Epochs")
print(f"{'='*80}\n")

comparison_df = pd.DataFrame([
    {
        "Epochs": result["num_epochs"],
        "Inf Time (s)": result["inference_metrics"]["total_inference_time"],
        "Avg Inf/Sample (ms)": result["inference_metrics"]["avg_inference_time_per_sample_ms"],
        "Argmax Acc": result["argmax_metrics"]["accuracy"],
        "Argmax Kappa": result["argmax_metrics"]["kappa"],
        "On-Topic Recall": result["argmax_metrics"]["on_topic_recall"],
        "Off-Topic Recall": result["argmax_metrics"]["off_topic_recall"],
    }
    for result in all_epoch_results.values()
])

print(comparison_df.to_string(index=False))
print()

# Save comparison
os.makedirs("models/qwen3_cross_encoder_offtopic/argmax_results", exist_ok=True)
comparison_df.to_csv("models/qwen3_cross_encoder_offtopic/argmax_results/epoch_comparison.csv", index=False)
with open("models/qwen3_cross_encoder_offtopic/argmax_results/all_epoch_results.json", "w") as f:
    json.dump(all_epoch_results, f, indent=2)

print("Argmax comparison saved to models/qwen3_cross_encoder_offtopic/argmax_results/epoch_comparison.csv")
print("All argmax results saved to models/qwen3_cross_encoder_offtopic/argmax_results/all_epoch_results.json")

# Generate comparison visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Kappa vs Epochs
ax1 = axes[0, 0]
ax1.plot(comparison_df["Epochs"], comparison_df["Argmax Kappa"], 'o-', color='darkblue', linewidth=2, markersize=8)
ax1.set_xlabel("Number of Epochs")
ax1.set_ylabel("Cohen's Kappa")
ax1.set_title("Argmax Kappa Score vs Training Epochs")
ax1.grid(True, alpha=0.3)
ax1.set_xticks(comparison_df["Epochs"].values)

# Plot 2: Accuracy vs Epochs
ax2 = axes[0, 1]
ax2.plot(comparison_df["Epochs"], comparison_df["Argmax Acc"], 'o-', color='darkgreen', linewidth=2, markersize=8)
ax2.set_xlabel("Number of Epochs")
ax2.set_ylabel("Accuracy")
ax2.set_title("Argmax Accuracy vs Training Epochs")
ax2.grid(True, alpha=0.3)
ax2.set_xticks(comparison_df["Epochs"].values)

# Plot 3: Recall Comparison vs Epochs
ax3 = axes[1, 0]
ax3.plot(comparison_df["Epochs"], comparison_df["On-Topic Recall"], 'o-', label="On-Topic Recall", linewidth=2, markersize=8)
ax3.plot(comparison_df["Epochs"], comparison_df["Off-Topic Recall"], 's-', label="Off-Topic Recall", linewidth=2, markersize=8)
ax3.set_xlabel("Number of Epochs")
ax3.set_ylabel("Recall")
ax3.set_title("Recall Scores vs Training Epochs")
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xticks(comparison_df["Epochs"].values)

# Plot 4: Average Inference Time per Sample vs Epochs
ax4 = axes[1, 1]
ax4.plot(comparison_df["Epochs"], comparison_df["Avg Inf/Sample (ms)"], 'o-', color='green', linewidth=2, markersize=8)
ax4.set_xlabel("Number of Epochs")
ax4.set_ylabel("Avg Inference Time per Sample (ms)")
ax4.set_title("Inference Time per Sample vs Epochs")
ax4.grid(True, alpha=0.3)
ax4.set_xticks(comparison_df["Epochs"].values)

plt.tight_layout()
plt.savefig("models/qwen3_cross_encoder_offtopic/argmax_results/epoch_comparison.png", dpi=300)
print("\nArgmax comparison plot saved to models/qwen3_cross_encoder_offtopic/argmax_results/epoch_comparison.png")
plt.show()

print("\n" + "="*80)
print("ALL ARGMAX INFERENCE COMPLETED!")
print("="*80)
