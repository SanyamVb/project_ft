"""
Mock pipeline test: mimics the full qwen3_cross_encoder_offtopic.py pipeline
with only 4 samples (2 per class) to verify everything works end-to-end.
"""
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

MODEL_SIZE = "4B"
MODEL_NAME = f"Qwen/Qwen3-{MODEL_SIZE}"
MAX_LENGTH = 4096

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
print(f"[MOCK TEST] RUNNING PIPELINE FOR MODEL: {MODEL_NAME}")
print(f"{'='*80}\n")

# ---- Create mock data: 2 on-topic (label=0) + 2 off-topic (label=1) ----
mock_data = {
    "Prompt Topic": [
        "Travel",
        "Technology",
        "Travel",
        "Technology",
    ],
    "Prompt Sub Topic": [
        "Vacation Planning",
        "Smartphones",
        "Vacation Planning",
        "Smartphones",
    ],
    "Prompt Script": [
        "Describe your ideal vacation destination and why you would like to visit there.",
        "Talk about how smartphones have changed the way people communicate.",
        "Describe your ideal vacation destination and why you would like to visit there.",
        "Talk about how smartphones have changed the way people communicate.",
    ],
    "Machine Transciption": [
        "I would love to visit Japan because of the culture and the food is amazing there.",
        "Smartphones have changed communication by making it instant through messaging apps.",
        "I think cooking pasta is very easy you just need water and salt and some sauce.",
        "My favorite color is blue and I like to play football on weekends with my friends.",
    ],
    "Level": ["I", "A", "I", "A"],
    "Human_Combined": [0, 0, 1, 1],
}

train_df = pd.DataFrame(mock_data)
test_df = pd.DataFrame(mock_data)  # Use same data for test in mock

train_df["label"] = train_df["Human_Combined"]
test_df["label"] = test_df["Human_Combined"]

print(f"Train: {len(train_df)} rows — {train_df['label'].value_counts().to_dict()}")
print(f"Test:  {len(test_df)} rows — {test_df['label'].value_counts().to_dict()}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def build_cross_encoder_inputs(examples):
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

# Prepare datasets
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

train_hf = train_hf.rename_column("label", "labels")
test_hf = test_hf.rename_column("label", "labels")

print(f"Train features: {train_hf.features}")
print(f"Example token count: {len(train_hf[0]['input_ids'])}")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    kappa = cohen_kappa_score(labels, preds)

    cm = confusion_matrix(labels, preds)

    metrics = {
        "accuracy": float(acc),
        "kappa": float(kappa),
    }

    if cm.shape == (2, 2):
        on_topic_recall = cm[0][0] / cm[0].sum() if cm[0].sum() > 0 else 0.0
        off_topic_recall = cm[1][1] / cm[1].sum() if cm[1].sum() > 0 else 0.0
        metrics["on_topic_recall"] = float(on_topic_recall)
        metrics["off_topic_recall"] = float(off_topic_recall)

    return metrics

# Create output directory
MOCK_OUTPUT_DIR = "./mock_test_output"
MOCK_MODEL_PATH = "models/mock_test"
os.makedirs(MOCK_MODEL_PATH, exist_ok=True)

# Load model
print(f"\nLoading {MODEL_NAME} for mock test...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
)

# Set random seed
SEED = 3407
set_seed(SEED)

# Apply LoRA
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

if device.type == "cuda":
    torch.cuda.empty_cache()
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

# Training arguments — minimal for mock test
training_args = TrainingArguments(
    output_dir=MOCK_OUTPUT_DIR,
    num_train_epochs=2,  # Just 2 epochs for quick test
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,  # No accumulation needed with 4 samples
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.001,
    warmup_ratio=0.5,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_strategy="epoch",
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
)

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

print(f"\n{'='*80}")
print("[MOCK] Starting Training for 2 Epochs")
print(f"{'='*80}\n")

if device.type == "cuda":
    torch.cuda.empty_cache()
    print(f"Pre-training GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")

train_result = trainer.train()

print(f"\n{'='*80}")
print("[MOCK] Training Completed!")
print(f"{'='*80}\n")
print(f"Total training time: {train_result.metrics.get('train_runtime', 0.0):.2f} seconds")

# Save final model
trainer.save_model(MOCK_MODEL_PATH)
tokenizer.save_pretrained(MOCK_MODEL_PATH)
print(f"\nFinal model saved to {MOCK_MODEL_PATH}")

# Post-training analysis
print(f"\n{'='*80}")
print("[MOCK] Post-Training Analysis: Threshold Optimization")
print(f"{'='*80}\n")

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
results_df.to_csv(f"{MOCK_MODEL_PATH}/threshold_optimization.csv", index=False)

# Save final metrics
final_metrics = {
    "model_name": MODEL_NAME,
    "model_size": MODEL_SIZE,
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

with open(f"{MOCK_MODEL_PATH}/final_metrics.json", "w") as f:
    json.dump(final_metrics, f, indent=2)

print(f"\nThreshold optimization and metrics saved to {MOCK_MODEL_PATH}")

# Generate threshold optimization plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.plot(results_df["threshold"], results_df["kappa"], 'o-', linewidth=2, markersize=4)
ax1.axvline(best['threshold'], color='red', linestyle='--', label=f"Best: {best['threshold']:.2f}")
ax1.set_xlabel("Threshold")
ax1.set_ylabel("Cohen's Kappa")
ax1.set_title("[MOCK] Kappa vs Threshold")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(results_df["threshold"], results_df["accuracy"], 'o-', linewidth=2, markersize=4)
ax2.axvline(best['threshold'], color='red', linestyle='--', label=f"Best: {best['threshold']:.2f}")
ax2.set_xlabel("Threshold")
ax2.set_ylabel("Accuracy")
ax2.set_title("[MOCK] Accuracy vs Threshold")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{MOCK_MODEL_PATH}/threshold_optimization.png", dpi=300)
print(f"Threshold optimization plot saved to {MOCK_MODEL_PATH}/threshold_optimization.png")

# Training history
print(f"\n{'='*80}")
print("[MOCK] Training History Summary")
print(f"{'='*80}\n")

if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
    history_df = pd.DataFrame(trainer.state.log_history)
    if 'epoch' in history_df.columns:
        print("Training metrics by epoch:")
        epoch_metrics = history_df[history_df['epoch'].notna()].copy()
        if len(epoch_metrics) > 0:
            print(epoch_metrics[['epoch', 'loss', 'eval_accuracy', 'eval_kappa']].to_string(index=False))

            history_df.to_csv(f"{MOCK_MODEL_PATH}/training_history.csv", index=False)
            print(f"\nTraining history saved to {MOCK_MODEL_PATH}/training_history.csv")
else:
    print("Training history not available in this trainer state.")

print("\n" + "="*80)
print("[MOCK] ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
print("="*80)
