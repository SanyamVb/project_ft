# needs to be at the top for patching
from unsloth import FastLanguageModel

import warnings
warnings.filterwarnings("ignore")

from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from transformers.trainer_callback import EarlyStoppingCallback

from .config import Config


def init_model_for_sft(config: Config):
    """Load Unsloth model with LoRA for SFT (no vLLM)."""
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.llm_model,
        max_seq_length=config.max_seq_length,
        load_in_4bit=False,
        fast_inference=False,  # No vLLM for SFT
        max_lora_rank=config.lora_rank,
    )
    model = FastLanguageModel.get_peft_model(
        base_model,
        r=config.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=config.lora_rank * 2,
        use_gradient_checkpointing="unsloth",
        random_state=config.random_state,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def formatting_func(examples):
    """Format prompt + completion into chat template for SFTTrainer."""
    texts = []
    for prompt, completion in zip(examples["prompt"], examples["completion"]):
        # Combine prompt and completion messages
        messages = prompt + completion
        texts.append(messages)
    return texts


def run_sft_train(sft_train: Dataset, sft_val: Dataset, config: Config):
    """Run SFT training with Unsloth + TRL SFTTrainer."""
    model, tokenizer = init_model_for_sft(config)

    sft_config = SFTConfig(
        output_dir=config.sft_output_dir,
        num_train_epochs=config.sft_epochs,
        per_device_train_batch_size=config.sft_batch_size,
        gradient_accumulation_steps=config.sft_grad_accum,
        learning_rate=config.sft_lr,
        max_length=config.max_seq_length,
        max_grad_norm=config.sft_max_grad_norm,
        optim=config.sft_optim,
        lr_scheduler_type=config.sft_lr_scheduler_type,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        logging_steps=10,
        weight_decay=0.01,
        warmup_ratio=0.1,
        completion_only_loss=False,
    )

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=config.sft_early_stopping_patience,
            early_stopping_threshold=0.0,
        ),
    ]

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=sft_train,
        eval_dataset=sft_val,
        formatting_func=formatting_func,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(config.sft_output_dir)
    return model
