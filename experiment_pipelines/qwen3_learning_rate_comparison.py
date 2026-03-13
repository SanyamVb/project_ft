import pandas as pd
import numpy as np
import torch
import os
import time
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
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
print(f"RUNNING PIPELINE FOR MODEL: {MODEL_NAME}")
print(f"EXPERIMENT: Learning Rate Comparison (4 variants)")
print(f"{'='*80}\n")

# Load data
train_df = pd.read_excel("data/train_0216.xlsx")
test_df = pd.read_excel("data/test_0216.xlsx")

label_map = {0: 0, 1: 1}  # Human_Combined is already 0/1
train_df["label"] = train_df["Human_Combined"]
test_df["label"] = test_df["Human_Combined"]

print(f"Train: {len(train_df)} rows — {train_df['label'].value_counts().to_dict()}")
print(f"Test:  {len(test_df)} rows — {test_df['label'].value_counts().to_dict()}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def build_cross_encoder_inputs(examples):
    """
    Build inputs for Qwen3 sequence classification using proper chat template.
    Formats messages with system + user roles, then applies Qwen's ChatML template.
    """
    formatted_texts = []
    for i in range(len(examples["Prompt Script"])):
        user_content = f"""You are now given the below data:
Prompt Topic: {examples['Prompt Topic'][i]}
Prompt Sub-Topic: {examples['Prompt Sub Topic'][i]}
Prompt Script: {examples['Prompt Script'][i]}
Test-taker's response: {examples['Machine Transciption'][i]}
Testing Proficiency Level: {examples['Level'][i]}

Based on your expertise, determine if the test-taker's response is off-topic or not."""
        
        # Structure as proper chat messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        # Apply Qwen chat template with enable_thinking=False for classification
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # No generation needed for classification
            enable_thinking=False  # Disable thinking mode for efficiency
        )
        formatted_texts.append(formatted_text)
    
    # Tokenize the formatted texts
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,  # Will be handled by DataCollator
    )
    
    return tokenized

# Prepare datasets with qwen_pipeline columns
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

# Rename 'label' to 'labels' (required by trainer)
train_hf = train_hf.rename_column("label", "labels")
test_hf = test_hf.rename_column("label", "labels")

print(f"Train features: {train_hf.features}")
print(f"Example token count: {len(train_hf[0]['input_ids'])}")

def compute_metrics(eval_pred):
    """Compute metrics for evaluation during training."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    kappa = cohen_kappa_score(labels, preds)
    
    # Calculate confusion matrix for additional metrics
    cm = confusion_matrix(labels, preds)
    
    # Calculate recall for each class if confusion matrix is valid
    metrics = {
        "accuracy": float(acc),
        "kappa": float(kappa),
    }
    
    # Add per-class metrics if both classes are present
    if cm.shape == (2, 2):
        on_topic_recall = cm[0][0] / cm[0].sum() if cm[0].sum() > 0 else 0.0
        off_topic_recall = cm[1][1] / cm[1].sum() if cm[1].sum() > 0 else 0.0
        metrics["on_topic_recall"] = float(on_topic_recall)
        metrics["off_topic_recall"] = float(off_topic_recall)
    
    return metrics

# Set random seed for reproducibility
SEED = 3407
set_seed(SEED)

# Learning rates to compare
LEARNING_RATES = [2e-6, 5e-5, 2e-5, 2e-4]

# Store all results
all_lr_results = {}

for lr in LEARNING_RATES:
    lr_str = f"{lr:.0e}".replace("e-0", "e-")  # Format like "2e-6"
    print(f"\n{'='*80}")
    print(f"Training with Learning Rate: {lr} ({lr_str})")
    print(f"{'='*80}\n")
    
    # Load fresh model for each learning rate
    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    
    # Apply LoRA for parameter-efficient fine-tuning
    print("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Clear CUDA cache to free up memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Training arguments with specific learning rate
    output_dir = f"./qwen3_lr_{lr_str}_20epochs"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=20,
        per_device_train_batch_size=1,  # Reduced to 1 for memory
        gradient_accumulation_steps=8,  # Increased to maintain effective batch size of 8
        per_device_eval_batch_size=4,  # Reduced for memory
        learning_rate=lr,  # DIFFERENT FOR EACH VARIANT
        weight_decay=0.001,
        warmup_ratio=0.5,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_strategy="epoch",
        save_steps=500,
        save_total_limit=None,
        report_to="none",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="kappa",
        greater_is_better=True,
        fp16=False,
        bf16=True,
        gradient_checkpointing=False,
        ignore_data_skip=False,
    )
    
    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer, padding=True)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf,
        eval_dataset=test_hf,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Training
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"Pre-training GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")
    
    train_result = trainer.train()
    
    print(f"\nTraining Completed for LR={lr}")
    print(f"Total training time: {train_result.metrics.get('train_runtime', 0.0):.2f} seconds")
    
    # Save model
    final_model_path = f"models/qwen3_lr_{lr_str}_20epochs"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # Post-training analysis
    print(f"\nPost-Training Analysis for LR={lr}")
    
    # Get predictions on test set
    inference_start = time.time()
    preds_output = trainer.predict(test_hf)
    inference_end = time.time()
    
    inference_time = inference_end - inference_start
    num_samples = len(test_hf)
    avg_inference_time_per_sample = inference_time / num_samples
    
    logits = preds_output.predictions
    y_true = test_df["label"].values
    
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    off_topic_probs = probs[:, 1]
    
    y_pred_default = (off_topic_probs >= 0.5).astype(int)
    
    kappa_default = cohen_kappa_score(y_true, y_pred_default)
    acc_default = (y_pred_default == y_true).mean()
    
    print(f"Default Threshold (0.5) - Accuracy: {acc_default:.4f}, Kappa: {kappa_default:.4f}")
    
    # Threshold optimization
    thresholds = np.arange(0.10, 0.91, 0.01)
    results = []
    
    for t in thresholds:
        y_pred_t = (off_topic_probs >= t).astype(int)
        acc_t = (y_pred_t == y_true).mean()
        kappa_t = cohen_kappa_score(y_true, y_pred_t)
        cm_t = confusion_matrix(y_true, y_pred_t)
        results.append({
            "threshold": t,
            "accuracy": acc_t,
            "kappa": kappa_t,
            "on_topic_recall": cm_t[0][0] / cm_t[0].sum() if cm_t[0].sum() > 0 else 0,
            "off_topic_recall": cm_t[1][1] / cm_t[1].sum() if cm_t[1].sum() > 0 else 0,
        })
    
    results_df = pd.DataFrame(results)
    best_idx = results_df["kappa"].idxmax()
    best = results_df.iloc[best_idx]
    
    print(f"Optimized Threshold ({best['threshold']:.2f}) - Accuracy: {best['accuracy']:.4f}, Kappa: {best['kappa']:.4f}")
    
    # Save threshold optimization results
    os.makedirs(final_model_path, exist_ok=True)
    results_df.to_csv(f"{final_model_path}/threshold_optimization.csv", index=False)
    
    # Save metrics
    lr_metrics = {
        "model_name": MODEL_NAME,
        "model_size": MODEL_SIZE,
        "learning_rate": float(lr),
        "experiment": f"lr_comparison_{lr_str}_20epochs",
        "total_training_time": float(train_result.metrics.get('train_runtime', 0.0)),
        "inference_metrics": {
            "total_inference_time": float(inference_time),
            "num_samples": num_samples,
            "avg_inference_time_per_sample_ms": float(avg_inference_time_per_sample * 1000),
        },
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
    
    with open(f"{final_model_path}/final_metrics.json", "w") as f:
        json.dump(lr_metrics, f, indent=2)
    
    # Generate threshold optimization plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.plot(results_df["threshold"], results_df["kappa"], 'o-', linewidth=2, markersize=4)
    ax1.axvline(best['threshold'], color='red', linestyle='--', label=f"Best: {best['threshold']:.2f}")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Cohen's Kappa")
    ax1.set_title(f"Kappa vs Threshold (LR={lr})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(results_df["threshold"], results_df["accuracy"], 'o-', linewidth=2, markersize=4)
    ax2.axvline(best['threshold'], color='red', linestyle='--', label=f"Best: {best['threshold']:.2f}")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Accuracy vs Threshold (LR={lr})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{final_model_path}/threshold_optimization.png", dpi=300)
    
    # Save training history
    if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
        history_df = pd.DataFrame(trainer.state.log_history)
        history_df.to_csv(f"{final_model_path}/training_history.csv", index=False)
    
    # Store results for comparison
    all_lr_results[lr_str] = {
        "learning_rate": float(lr),
        "best_kappa": float(best['kappa']),
        "best_accuracy": float(best['accuracy']),
        "best_threshold": float(best['threshold']),
        "training_time": float(train_result.metrics.get('train_runtime', 0.0)),
    }
    
    # Clean up model to free memory
    del model
    del trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    print(f"\nCompleted LR={lr}\n")

# Save comparison summary
print(f"\n{'='*80}")
print("LEARNING RATE COMPARISON SUMMARY")
print(f"{'='*80}\n")

comparison_df = pd.DataFrame.from_dict(all_lr_results, orient='index')
comparison_df = comparison_df.sort_values('learning_rate')
print(comparison_df.to_string())

# Save comparison results
os.makedirs("models/lr_comparison_summary", exist_ok=True)
comparison_df.to_csv("models/lr_comparison_summary/lr_comparison_results.csv")

with open("models/lr_comparison_summary/lr_comparison_summary.json", "w") as f:
    json.dump(all_lr_results, f, indent=2)

# Create comparison plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

lr_values = [all_lr_results[k]['learning_rate'] for k in sorted(all_lr_results.keys(), key=lambda x: all_lr_results[x]['learning_rate'])]
kappas = [all_lr_results[k]['best_kappa'] for k in sorted(all_lr_results.keys(), key=lambda x: all_lr_results[x]['learning_rate'])]
accuracies = [all_lr_results[k]['best_accuracy'] for k in sorted(all_lr_results.keys(), key=lambda x: all_lr_results[x]['learning_rate'])]

ax1 = axes[0]
ax1.plot(lr_values, kappas, 'o-', linewidth=2, markersize=8)
ax1.set_xlabel("Learning Rate")
ax1.set_ylabel("Best Kappa Score")
ax1.set_title("Learning Rate vs Kappa")
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(lr_values, accuracies, 'o-', linewidth=2, markersize=8)
ax2.set_xlabel("Learning Rate")
ax2.set_ylabel("Best Accuracy")
ax2.set_title("Learning Rate vs Accuracy")
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("models/lr_comparison_summary/lr_comparison_plot.png", dpi=300)

print(f"\nComparison summary saved to models/lr_comparison_summary/")
print("\n" + "="*80)
print("ALL LEARNING RATE EXPERIMENTS COMPLETED!")
print("="*80)
