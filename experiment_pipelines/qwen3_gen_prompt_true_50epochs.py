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
print(f"EXPERIMENT: add_generation_prompt=True, 50 Epochs")
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
        # MODIFICATION: add_generation_prompt=True
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # CHANGED TO TRUE
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

# Load model
print(f"\nLoading {MODEL_NAME} for training...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
)

# Set random seed for reproducibility
SEED = 3407
set_seed(SEED)

# Apply LoRA for parameter-efficient fine-tuning
print("\nApplying LoRA configuration...")
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

# Training arguments
training_args = TrainingArguments(
    output_dir="./qwen3_gen_prompt_true_50epochs",
    num_train_epochs=50,  # 50 EPOCHS
    per_device_train_batch_size=1,  # Reduced to 1 for memory
    gradient_accumulation_steps=8,  # Increased to maintain effective batch size of 8
    per_device_eval_batch_size=4,  # Reduced for memory
    learning_rate=2e-5,
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
print(f"\n{'='*80}")
print("Starting Training (50 Epochs, add_generation_prompt=True)")
print(f"{'='*80}\n")

if device.type == "cuda":
    torch.cuda.empty_cache()
    print(f"Pre-training GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")

train_result = trainer.train()

print(f"\n{'='*80}")
print("Training Completed!")
print(f"{'='*80}\n")
print(f"Total training time: {train_result.metrics.get('train_runtime', 0.0):.2f} seconds")

# Save final model
final_model_path = "models/qwen3_gen_prompt_true_50epochs"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"\nFinal model saved to {final_model_path}")

# Post-training analysis: Threshold optimization and detailed evaluation
print(f"\n{'='*80}")
print("Post-Training Analysis: Threshold Optimization")
print(f"{'='*80}\n")

# Get predictions on test set
inference_start = time.time()
preds_output = trainer.predict(test_hf)
inference_end = time.time()

inference_time = inference_end - inference_start
num_samples = len(test_hf)
avg_inference_time_per_sample = inference_time / num_samples

print(f"Total inference time: {inference_time:.2f} seconds")
print(f"Number of samples: {num_samples}")
print(f"Average inference time per sample: {avg_inference_time_per_sample*1000:.2f} ms\n")

logits = preds_output.predictions
y_true = test_df["label"].values

probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
off_topic_probs = probs[:, 1]

y_pred_default = (off_topic_probs >= 0.5).astype(int)

print("=== Results with Default Threshold (0.5) ===")
print(classification_report(y_true, y_pred_default, target_names=["On-Topic", "Off-Topic"]))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred_default))
kappa_default = cohen_kappa_score(y_true, y_pred_default)
acc_default = (y_pred_default == y_true).mean()
print(f"Accuracy: {acc_default:.4f}")
print(f"Kappa: {kappa_default:.4f}")

# Threshold optimization
print("\n=== Optimizing Threshold ===")
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

print(f"\n=== Optimized Results ===")
print(f"Best threshold: {best['threshold']:.2f}")
print(f"  Accuracy:         {best['accuracy']:.4f}")
print(f"  Kappa:            {best['kappa']:.4f}")
print(f"  On-Topic Recall:  {best['on_topic_recall']:.4f}")
print(f"  Off-Topic Recall: {best['off_topic_recall']:.4f}")
print()
print(f"Improvement over default (0.5):")
print(f"  Kappa: {kappa_default:.4f} -> {best['kappa']:.4f} ({best['kappa'] - kappa_default:+.4f})")
print(f"  Accuracy: {acc_default:.4f} -> {best['accuracy']:.4f} ({best['accuracy'] - acc_default:+.4f})")

# Save threshold optimization results
os.makedirs(final_model_path, exist_ok=True)
results_df.to_csv(f"{final_model_path}/threshold_optimization.csv", index=False)

# Save final metrics
final_metrics = {
    "model_name": MODEL_NAME,
    "model_size": MODEL_SIZE,
    "experiment": "add_generation_prompt_true_50epochs",
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
    json.dump(final_metrics, f, indent=2)

print(f"\nThreshold optimization and metrics saved to {final_model_path}")

# Generate threshold optimization plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.plot(results_df["threshold"], results_df["kappa"], 'o-', linewidth=2, markersize=4)
ax1.axvline(best['threshold'], color='red', linestyle='--', label=f"Best: {best['threshold']:.2f}")
ax1.set_xlabel("Threshold")
ax1.set_ylabel("Cohen's Kappa")
ax1.set_title("Kappa vs Threshold")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(results_df["threshold"], results_df["accuracy"], 'o-', linewidth=2, markersize=4)
ax2.axvline(best['threshold'], color='red', linestyle='--', label=f"Best: {best['threshold']:.2f}")
ax2.set_xlabel("Threshold")
ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy vs Threshold")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{final_model_path}/threshold_optimization.png", dpi=300)
print(f"Threshold optimization plot saved to {final_model_path}/threshold_optimization.png")

# Load training history from trainer logs if available
print(f"\n{'='*80}")
print("Training History Summary")
print(f"{'='*80}\n")

if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
    history_df = pd.DataFrame(trainer.state.log_history)
    if 'epoch' in history_df.columns:
        print("Training metrics by epoch:")
        epoch_metrics = history_df[history_df['epoch'].notna()].copy()
        if len(epoch_metrics) > 0:
            print(epoch_metrics[['epoch', 'loss', 'eval_accuracy', 'eval_kappa']].to_string(index=False))
            
            # Save training history
            history_df.to_csv(f"{final_model_path}/training_history.csv", index=False)
            print(f"\nTraining history saved to {final_model_path}/training_history.csv")
else:
    print("Training history not available in this trainer state.")

print("\n" + "="*80)
print("EXPERIMENT COMPLETED!")
print("="*80)
