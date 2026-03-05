"""Quick test script to validate SFT training pipeline with 2 examples."""

import pandas as pd
import numpy as np
import torch
import os
import time
import json
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("QUICK TEST: SFT Training Pipeline Validation")
print("="*80)

# Minimal config for testing
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

# Load data and take only 2 examples from each
print("\nLoading data (2 train, 2 test examples)...")
train_df = pd.read_excel("data/train_0216.xlsx").head(2)
test_df = pd.read_excel("data/test_0216.xlsx").head(2)

train_df["label"] = train_df["Human_Combined"]
test_df["label"] = test_df["Human_Combined"]

print(f"Train: {len(train_df)} rows — {train_df['label'].value_counts().to_dict()}")
print(f"Test:  {len(test_df)} rows — {test_df['label'].value_counts().to_dict()}")

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

if tokenizer.chat_template is None:
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'user' %}"
        "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'assistant' %}"
        "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{% endif %}"
    )

def build_cross_encoder_inputs(examples):
    """Build inputs for Qwen3 sequence classification using chat template."""
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

# Rename 'label' to 'labels' (required by trainer)
train_hf = train_hf.rename_column("label", "labels")
test_hf = test_hf.rename_column("label", "labels")

print(f"Train features: {train_hf.features}")
print(f"Example token count: {len(train_hf[0]['input_ids'])}")

# Load model
print(f"\nLoading {MODEL_NAME} for sequence classification...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
)
model.config.pad_token_id = tokenizer.pad_token_id

# Apply LoRA
print("\nApplying LoRA configuration...")
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    kappa = cohen_kappa_score(labels, preds)
    return {"accuracy": float(acc), "kappa": float(kappa)}

# Minimal SFT config for testing
print("\nConfiguring SFT trainer...")
training_args = SFTConfig(
    output_dir="./test_output_temp",
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Keep small for testing
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=1,
    learning_rate=2e-6,
    weight_decay=0.001,
    warmup_steps=1,
    lr_scheduler_type="linear",
    eval_strategy="no",
    save_strategy="no",
    logging_steps=1,
    report_to="none",
    fp16=False,
    bf16=True if device.type == "cuda" else False,
    gradient_checkpointing=False,
    optim="adamw_8bit",
    dataset_text_field=None,
    dataset_kwargs={"skip_prepare_dataset": True},
    packing=False,
    completion_only_loss=False,
)

# Create data collator
data_collator = DataCollatorWithPadding(tokenizer, padding=True)

# Create SFT trainer
print("\nCreating SFT trainer...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_hf,
    eval_dataset=test_hf,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Training
print("\n" + "="*80)
print("STARTING TRAINING (1 epoch, 2 examples)")
print("="*80)
train_start = time.time()
train_result = trainer.train()
train_time = time.time() - train_start

print(f"\nTraining completed in {train_time:.2f} seconds")
print(f"Train loss: {train_result.metrics.get('train_loss', 'N/A')}")

# Evaluation
print("\n" + "="*80)
print("RUNNING EVALUATION (2 examples)")
print("="*80)
eval_start = time.time()
preds_output = trainer.predict(test_hf)
eval_time = time.time() - eval_start

print(f"\nEvaluation completed in {eval_time:.2f} seconds")

logits = preds_output.predictions
y_true = test_df["label"].values

probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
off_topic_probs = probs[:, 1]

y_pred = (off_topic_probs >= 0.5).astype(int)

print("\n=== Results ===")
print(f"Ground truth: {y_true}")
print(f"Predictions: {y_pred}")
print(f"Off-topic probabilities: {off_topic_probs}")

if len(y_true) > 0:
    acc = (y_pred == y_true).mean()
    print(f"\nAccuracy: {acc:.4f}")
    if len(set(y_true)) > 1 and len(set(y_pred)) > 1:
        kappa = cohen_kappa_score(y_true, y_pred)
        print(f"Kappa: {kappa:.4f}")
    else:
        print("Kappa: N/A (need varied predictions)")

# Mock result saving
print("\n" + "="*80)
print("MOCKING RESULT SAVING (not writing to disk)")
print("="*80)

mock_results = {
    "model_name": MODEL_NAME,
    "model_size": MODEL_SIZE,
    "training_metrics": {
        "train_runtime": train_time,
        "train_loss": train_result.metrics.get("train_loss"),
    },
    "inference_metrics": {
        "total_inference_time": eval_time,
        "num_samples": len(test_hf),
    },
    "test_results": {
        "accuracy": float((y_pred == y_true).mean()),
        "ground_truth": y_true.tolist(),
        "predictions": y_pred.tolist(),
        "probabilities": off_topic_probs.tolist(),
    }
}

print("\nMocked results:")
print(json.dumps(mock_results, indent=2))

print("\n" + "="*80)
print("✓ TEST COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nAll pipeline components validated:")
print("  ✓ Data loading and preprocessing")
print("  ✓ Tokenization with chat template")
print("  ✓ Model loading with LoRA")
print("  ✓ SFT trainer initialization")
print("  ✓ Training loop")
print("  ✓ Evaluation and metrics")
print("  ✓ Result collection")
print("\nYou can now run the full script with confidence!")
