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

# TEST VERSION: Using minimal data (1 sample per class)
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
print(f"TEST RUN - MINIMAL DATA (1 sample per class)")
print(f"RUNNING PIPELINE FOR MODEL: {MODEL_NAME}")
print(f"EXPERIMENT: add_generation_prompt=True, 50 Epochs")
print(f"{'='*80}\n")

# Load data
train_df = pd.read_excel("data/train_0216.xlsx")
test_df = pd.read_excel("data/test_0216.xlsx")

label_map = {0: 0, 1: 1}
train_df["label"] = train_df["Human_Combined"]
test_df["label"] = test_df["Human_Combined"]

# TEST: Use only 1 sample per class
train_class_0 = train_df[train_df["label"] == 0].head(1)
train_class_1 = train_df[train_df["label"] == 1].head(1)
train_df = pd.concat([train_class_0, train_class_1]).reset_index(drop=True)

test_class_0 = test_df[test_df["label"] == 0].head(1)
test_class_1 = test_df[test_df["label"] == 1].head(1)
test_df = pd.concat([test_class_0, test_class_1]).reset_index(drop=True)

print(f"TEST - Train: {len(train_df)} rows — {train_df['label'].value_counts().to_dict()}")
print(f"TEST - Test:  {len(test_df)} rows — {test_df['label'].value_counts().to_dict()}")

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
            add_generation_prompt=True,
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

print(f"\nLoading {MODEL_NAME} for training...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
)

SEED = 3407
set_seed(SEED)

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

# TEST: Reduced epochs
training_args = TrainingArguments(
    output_dir="./test_qwen3_gen_prompt_true_50epochs",
    num_train_epochs=2,  # TEST: Only 2 epochs instead of 50
    per_device_train_batch_size=1,  # TEST: Batch size 1
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=2,
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
print("Starting TEST Training (2 Epochs, add_generation_prompt=True)")
print(f"{'='*80}\n")

if device.type == "cuda":
    torch.cuda.empty_cache()
    print(f"Pre-training GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated")

train_result = trainer.train()

print(f"\n{'='*80}")
print("TEST Training Completed!")
print(f"{'='*80}\n")
print(f"Total training time: {train_result.metrics.get('train_runtime', 0.0):.2f} seconds")

final_model_path = "models/test_qwen3_gen_prompt_true_50epochs"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"\nTest model saved to {final_model_path}")

print(f"\n{'='*80}")
print("TEST EXPERIMENT COMPLETED SUCCESSFULLY!")
print(f"{'='*80}")

# Clear GPU memory
if device.type == "cuda":
    del model
    del trainer
    torch.cuda.empty_cache()
    print(f"\nGPU memory cleared")
