"""
Quick test script for qwen3_cross_encoder_offtopic.py
Uses minimal data (2 train, 2 test samples) to validate all code paths quickly.
"""

import pandas as pd
import numpy as np
import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("QUICK TEST MODE - Using minimal data to validate code")
print("="*80)

# Using smaller model for faster testing
MODEL_NAME = "Qwen/Qwen3-0.6B"  # Smaller model
MAX_LENGTH = 512  # Reduced for speed

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

# Load data and take only 2 samples from each
print("\nLoading data (taking only 2 samples for quick test)...")
train_df_full = pd.read_excel("data/train_0216.xlsx")
test_df_full = pd.read_excel("data/test_0216.xlsx")

# Take 2 samples: 1 on-topic, 1 off-topic if possible
label_map = {"No": 0, "Yes": 1}
train_df_full["label"] = train_df_full["Final_Annotation"].map(label_map)
test_df_full["label"] = test_df_full["Final_Annotation"].map(label_map)

# Try to get balanced samples
train_on = train_df_full[train_df_full["label"] == 0].head(1)
train_off = train_df_full[train_df_full["label"] == 1].head(1)
train_df = pd.concat([train_on, train_off]).reset_index(drop=True)

test_on = test_df_full[test_df_full["label"] == 0].head(1)
test_off = test_df_full[test_df_full["label"] == 1].head(1)
test_df = pd.concat([test_on, test_off]).reset_index(drop=True)

print(f"Train: {len(train_df)} rows — {train_df['label'].value_counts().to_dict()}")
print(f"Test:  {len(test_df)} rows — {test_df['label'].value_counts().to_dict()}")

# Load tokenizer
print(f"\nLoading tokenizer for {MODEL_NAME}...")
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
print("\nPreparing datasets...")
cols_to_keep = ["Prompt Topic", "Prompt Sub Topic", "Prompt Script", "Machine Transciption", "Level", "label"]
train_hf = Dataset.from_pandas(train_df[cols_to_keep], preserve_index=False)
test_hf = Dataset.from_pandas(test_df[cols_to_keep], preserve_index=False)

train_hf = train_hf.map(
    build_cross_encoder_inputs, 
    batched=True, 
    remove_columns=["Prompt Topic", "Prompt Sub Topic", "Prompt Script", "Machine Transciption", "Level"]
)
test_hf = test_hf.map(
    build_cross_encoder_inputs, 
    batched=True, 
    remove_columns=["Prompt Topic", "Prompt Sub Topic", "Prompt Script", "Machine Transciption", "Level"]
)

print(f"Train features: {train_hf.features}")
print(f"Example token count: {len(train_hf[0]['input_ids'])}")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    kappa = cohen_kappa_score(labels, preds)
    return {"accuracy": float(acc), "kappa": float(kappa)}

# Create output directory
os.makedirs("models/qwen3_cross_encoder_test", exist_ok=True)

# Store results for all epoch configurations
all_epoch_results = {}

# Loop through different epoch configurations - just test with 1 epoch
for num_epochs in [1]:  # Only 1 epoch for quick test
    print(f"\n{'='*80}")
    print(f"TRAINING WITH {num_epochs} EPOCH(S) - QUICK TEST")
    print(f"{'='*80}\n")
    
    # Load model
    print(f"Loading {MODEL_NAME} for sequence classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.0f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.0f}M)")
    
    # Training arguments - adjusted for quick test
    training_args = TrainingArguments(
        output_dir=f"./qwen3_cross_encoder_test/{num_epochs}_epoch",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="kappa",
        greater_is_better=True,
        logging_steps=1,
        report_to="none",
        fp16=True if device.type == "cuda" else False,
        bf16=False,
        gradient_checkpointing=False,  # Disabled for small model
    )
    
    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer, padding=True)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf,
        eval_dataset=test_hf,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print(f"\n=== Starting Training ({num_epochs} epoch(s)) ===")
    train_result = trainer.train()
    
    # Store training metrics
    training_metrics = {
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "train_loss": train_result.metrics.get("train_loss"),
        "epoch": train_result.metrics.get("epoch"),
    }
    
    print(f"\nTraining completed in {training_metrics['train_runtime']:.2f} seconds")
    print(f"Train loss: {training_metrics['train_loss']:.4f}")

    # Evaluation
    print(f"\n=== Evaluating on Test Set ({num_epochs} epoch(s)) ===")
    preds_output = trainer.predict(test_hf)
    logits = preds_output.predictions
    y_true = test_df["label"].values

    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    off_topic_probs = probs[:, 1]

    y_pred_default = (off_topic_probs >= 0.5).astype(int)

    print(f"\n=== Results with Default Threshold (0.5) - {num_epochs} epoch(s) ===")
    print(f"True labels: {y_true}")
    print(f"Predictions: {y_pred_default}")
    print(f"Probabilities (off-topic): {off_topic_probs}")
    
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred_default)) > 1:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_default, target_names=["On-Topic", "Off-Topic"], zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred_default))
        kappa_default = cohen_kappa_score(y_true, y_pred_default)
    else:
        print("\nNote: Not enough class diversity for classification report (expected in quick test)")
        kappa_default = 0.0
    
    acc_default = (y_pred_default == y_true).mean()
    print(f"Accuracy: {acc_default:.4f}")
    print(f"Kappa: {kappa_default:.4f}")

    # Threshold optimization (simplified for quick test)
    print(f"\n=== Optimizing Threshold ({num_epochs} epoch(s)) ===")
    thresholds = np.arange(0.10, 0.91, 0.05)  # Fewer thresholds for speed
    results = []

    for t in thresholds:
        y_pred_t = (off_topic_probs >= t).astype(int)
        acc_t = (y_pred_t == y_true).mean()
        try:
            kappa_t = cohen_kappa_score(y_true, y_pred_t)
        except:
            kappa_t = 0.0
        cm_t = confusion_matrix(y_true, y_pred_t)
        
        # Handle edge cases
        on_recall = cm_t[0][0] / cm_t[0].sum() if cm_t[0].sum() > 0 else 0
        off_recall = cm_t[1][1] / cm_t[1].sum() if len(cm_t) > 1 and cm_t[1].sum() > 0 else 0
        
        results.append({
            "threshold": t,
            "accuracy": acc_t,
            "kappa": kappa_t,
            "on_topic_recall": on_recall,
            "off_topic_recall": off_recall,
        })

    results_df = pd.DataFrame(results)
    best_idx = results_df["kappa"].idxmax()
    best = results_df.iloc[best_idx]

    print(f"\n=== Optimized Results ({num_epochs} epoch(s)) ===")
    print(f"Best threshold: {best['threshold']:.2f}")
    print(f"  Accuracy:         {best['accuracy']:.4f}")
    print(f"  Kappa:            {best['kappa']:.4f}")
    print(f"  On-Topic Recall:  {best['on_topic_recall']:.4f}")
    print(f"  Off-Topic Recall: {best['off_topic_recall']:.4f}")

    # Save model
    save_path = f"models/qwen3_cross_encoder_test/{num_epochs}_epoch"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Store all metrics for this epoch configuration
    all_epoch_results[f"{num_epochs}_epoch"] = {
        "num_epochs": num_epochs,
        "training_metrics": training_metrics,
        "default_threshold_metrics": {
            "threshold": 0.5,
            "accuracy": float(acc_default),
            "kappa": float(kappa_default),
        },
        "optimized_threshold_metrics": {
            "threshold": float(best['threshold']),
            "accuracy": float(best['accuracy']),
            "kappa": float(best['kappa']),
            "on_topic_recall": float(best['on_topic_recall']),
            "off_topic_recall": float(best['off_topic_recall']),
        },
        "improvement": {
            "kappa_improvement": float(best['kappa'] - kappa_default),
            "accuracy_improvement": float(best['accuracy'] - acc_default),
        }
    }
    
    # Save individual epoch metrics
    with open(f"{save_path}/metrics.json", "w") as f:
        json.dump(all_epoch_results[f"{num_epochs}_epoch"], f, indent=2)
    
    print(f"\nModel and metrics saved to {save_path}")

# Save comparison of all epoch results
print(f"\n{'='*80}")
print("SUMMARY: Quick Test Results")
print(f"{'='*80}\n")

comparison_df = pd.DataFrame([
    {
        "Epochs": result["num_epochs"],
        "Train Loss": result["training_metrics"]["train_loss"],
        "Train Time (s)": result["training_metrics"]["train_runtime"],
        "Default Acc": result["default_threshold_metrics"]["accuracy"],
        "Default Kappa": result["default_threshold_metrics"]["kappa"],
        "Best Threshold": result["optimized_threshold_metrics"]["threshold"],
        "Best Acc": result["optimized_threshold_metrics"]["accuracy"],
        "Best Kappa": result["optimized_threshold_metrics"]["kappa"],
        "Kappa Gain": result["improvement"]["kappa_improvement"],
    }
    for result in all_epoch_results.values()
])

print(comparison_df.to_string(index=False))
print()

# Save comparison
comparison_df.to_csv("models/qwen3_cross_encoder_test/epoch_comparison.csv", index=False)
with open("models/qwen3_cross_encoder_test/all_epoch_results.json", "w") as f:
    json.dump(all_epoch_results, f, indent=2)

print("Comparison saved to models/qwen3_cross_encoder_test/epoch_comparison.csv")
print("All results saved to models/qwen3_cross_encoder_test/all_epoch_results.json")

# Generate simple comparison visualization
try:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    x_pos = np.arange(len(comparison_df))
    width = 0.35
    
    ax.bar(x_pos - width/2, comparison_df["Default Kappa"], width, label="Default (0.5)", alpha=0.8)
    ax.bar(x_pos + width/2, comparison_df["Best Kappa"], width, label="Optimized", alpha=0.8)
    
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Cohen's Kappa")
    ax.set_title("Quick Test: Kappa Score Comparison")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(comparison_df["Epochs"])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig("models/qwen3_cross_encoder_test/quick_test_results.png", dpi=150)
    print("\nPlot saved to models/qwen3_cross_encoder_test/quick_test_results.png")
    plt.close()
except Exception as e:
    print(f"\nNote: Could not generate plot: {e}")

print("\n" + "="*80)
print("QUICK TEST COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nAll code paths validated. You can now run the full script with confidence.")
print(f"Total runtime: {training_metrics['train_runtime']:.2f}s")
