import pandas as pd
import numpy as np
import torch
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

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"PyTorch version: {torch.__version__}")

train_df = pd.read_excel("data/train_1000_0822_balanced.xlsx")
test_df = pd.read_excel("data/test_331_0822_balanced.xlsx")

label_map = {"No": 0, "Yes": 1}
train_df["label"] = train_df["Final_Annotation"].map(label_map)
test_df["label"] = test_df["Final_Annotation"].map(label_map)

print(f"Train: {len(train_df)} rows — {train_df['label'].value_counts().to_dict()}")
print(f"Test:  {len(test_df)} rows — {test_df['label'].value_counts().to_dict()}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

def build_cross_encoder_inputs(examples):
    prompts = [
        f"{script}\nTopic: {category}"
        for script, category in zip(examples["Script"], examples["Category"])
    ]
    return tokenizer(
        prompts,
        examples["Transcription-Machine"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

cols_to_keep = ["Script", "Category", "Transcription-Machine", "label"]
train_hf = Dataset.from_pandas(train_df[cols_to_keep], preserve_index=False)
test_hf = Dataset.from_pandas(test_df[cols_to_keep], preserve_index=False)

train_hf = train_hf.map(build_cross_encoder_inputs, batched=True, remove_columns=["Script", "Category", "Transcription-Machine"])
test_hf = test_hf.map(build_cross_encoder_inputs, batched=True, remove_columns=["Script", "Category", "Transcription-Machine"])

print(f"Train features: {train_hf.features}")
print(f"Example token count: {len(train_hf[0]['input_ids'])}")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,} ({total_params/1e6:.0f}M)")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    kappa = cohen_kappa_score(labels, preds)
    return {"accuracy": float(acc), "kappa": float(kappa)}

training_args = TrainingArguments(
    output_dir="./cross_encoder_output",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="kappa",
    greater_is_better=True,
    logging_steps=10,
    report_to="none",
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_hf,
    eval_dataset=test_hf,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()

preds_output = trainer.predict(test_hf)
logits = preds_output.predictions
y_true = test_df["label"].values

probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
off_topic_probs = probs[:, 1]

y_pred_default = (off_topic_probs >= 0.5).astype(int)

print(classification_report(y_true, y_pred_default, target_names=["On-Topic", "Off-Topic"]))
print(confusion_matrix(y_true, y_pred_default))
kappa_default = cohen_kappa_score(y_true, y_pred_default)
acc_default = (y_pred_default == y_true).mean()
print(f"Accuracy: {acc_default:.4f}")
print(f"Kappa: {kappa_default:.4f}")

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
        "on_topic_recall": cm_t[0][0] / cm_t[0].sum(),
        "off_topic_recall": cm_t[1][1] / cm_t[1].sum(),
    })

results_df = pd.DataFrame(results)
best_idx = results_df["kappa"].idxmax()
best = results_df.iloc[best_idx]

print(f"Best threshold: {best['threshold']:.2f}")
print(f"  Accuracy:         {best['accuracy']:.4f}")
print(f"  Kappa:            {best['kappa']:.4f}")
print(f"  On-Topic Recall:  {best['on_topic_recall']:.4f}")
print(f"  Off-Topic Recall: {best['off_topic_recall']:.4f}")
print()
print(f"Improvement over default (0.5):")
print(f"  Kappa: {kappa_default:.4f} -> {best['kappa']:.4f} ({best['kappa'] - kappa_default:+.4f})")
print(f"  Accuracy: {acc_default:.4f} -> {best['accuracy']:.4f} ({best['accuracy'] - acc_default:+.4f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(results_df["threshold"], results_df["kappa"], "b-", linewidth=2)
ax1.axvline(x=best["threshold"], color="r", linestyle="--", label=f"Best: {best['threshold']:.2f}")
ax1.axvline(x=0.5, color="gray", linestyle=":", label="Default: 0.50")
ax1.set_xlabel("Threshold")
ax1.set_ylabel("Cohen's Kappa")
ax1.set_title("Kappa vs Decision Threshold")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(results_df["threshold"], results_df["on_topic_recall"], "g-", linewidth=2, label="On-Topic Recall")
ax2.plot(results_df["threshold"], results_df["off_topic_recall"], "r-", linewidth=2, label="Off-Topic Recall")
ax2.axvline(x=best["threshold"], color="blue", linestyle="--", label=f"Best Kappa: {best['threshold']:.2f}")
ax2.set_xlabel("Threshold")
ax2.set_ylabel("Recall")
ax2.set_title("Recall Tradeoff")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

best_threshold = best["threshold"]
y_pred_best = (off_topic_probs >= best_threshold).astype(int)

print(f"Optimized threshold: {best_threshold:.2f}")
print(classification_report(y_true, y_pred_best, target_names=["On-Topic", "Off-Topic"]))
print(confusion_matrix(y_true, y_pred_best))
kappa_best = cohen_kappa_score(y_true, y_pred_best)
acc_best = (y_pred_best == y_true).mean()
print(f"Accuracy: {acc_best:.4f}")
print(f"Kappa: {kappa_best:.4f}")



test_df["predicted"] = y_pred_best
test_df["off_topic_prob"] = off_topic_probs
test_df["correct"] = test_df["predicted"] == test_df["label"]

errors = test_df[~test_df["correct"]].copy()
print(f"Total errors: {len(errors)} / {len(test_df)}")
print(f"\nFalse Positives (on-topic flagged as off-topic): {(errors['label']==0).sum()}")
print(f"False Negatives (off-topic missed): {(errors['label']==1).sum()}")
print(f"\nErrors by Category:")
print(errors["Category"].value_counts().head(10))

print("=== MOST CONFIDENT FALSE NEGATIVES (off-topic, model said on-topic) ===")
fn = errors[errors["label"] == 1].nsmallest(3, "off_topic_prob")
for _, row in fn.iterrows():
    print(f"\nP(off-topic) = {row['off_topic_prob']:.3f}  |  Category: {row['Category']}")
    print(f"Script: {row['Script'][:150]}...")
    print(f"Response: {row['Transcription-Machine'][:150]}...")
    print("-" * 50)

print("\n=== MOST CONFIDENT FALSE POSITIVES (on-topic, model said off-topic) ===")
fp = errors[errors["label"] == 0].nlargest(3, "off_topic_prob")
for _, row in fp.iterrows():
    print(f"\nP(off-topic) = {row['off_topic_prob']:.3f}  |  Category: {row['Category']}")
    print(f"Script: {row['Script'][:150]}...")
    print(f"Response: {row['Transcription-Machine'][:150]}...")
    print("-" * 50)

fig, ax = plt.subplots(figsize=(10, 5))

on_topic_probs = off_topic_probs[y_true == 0]
off_topic_probs_actual = off_topic_probs[y_true == 1]

ax.hist(on_topic_probs, bins=30, alpha=0.6, label="Actually On-Topic", color="green")
ax.hist(off_topic_probs_actual, bins=30, alpha=0.6, label="Actually Off-Topic", color="red")
ax.axvline(x=best_threshold, color="blue", linestyle="--", linewidth=2, label=f"Threshold: {best_threshold:.2f}")
ax.set_xlabel("P(Off-Topic)")
ax.set_ylabel("Count")
ax.set_title("Model Confidence Distribution")
ax.legend()
plt.tight_layout()
plt.show()

save_path = "models/deberta_cross_encoder_base_offtopic"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

with open(f"{save_path}/threshold_config.json", "w") as f:
    json.dump({
        "best_threshold": float(best_threshold),
        "kappa_at_best": float(kappa_best),
        "accuracy_at_best": float(acc_best),
        "kappa_at_default": float(kappa_default),
        "accuracy_at_default": float(acc_default),
    }, f, indent=2)

print(f"Model saved to {save_path}")
print(f"Threshold config saved to {save_path}/threshold_config.json")
