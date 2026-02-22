# needs to be at the top for patching
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

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
    
    # Set default chat template for Qwen models if not present
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
    
    return model, tokenizer


def run_sft_train(sft_train: Dataset, sft_val: Dataset, config: Config, resume_from_checkpoint: str = None):
    """Run SFT training with optional checkpoint resume."""
    import os
    
    # Auto-detect checkpoint if not provided
    if resume_from_checkpoint is None and os.path.isdir(config.sft_output_dir):
        checkpoints = [d for d in os.listdir(config.sft_output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            resume_from_checkpoint = os.path.join(config.sft_output_dir, latest_checkpoint)
            print(f"Found existing SFT checkpoint, resuming from: {resume_from_checkpoint}")
    
    model, tokenizer = init_model_for_sft(config)
    
    def formatting_func(examples):
        """Format prompt + completion for Unsloth SFTTrainer with enable_thinking=False."""
        # Handle both single example (dict) and batched examples
        if isinstance(examples["prompt"][0], dict):
            # Single example: examples["prompt"] is a list of message dicts
            messages = examples["prompt"] + examples["completion"]
            chat_template_kwargs = examples.get("chat_template_kwargs", {"enable_thinking": False})
            return [tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False,
                **chat_template_kwargs
            )]
        else:
            # Batched examples: examples["prompt"] is a list of conversations
            results = []
            for i in range(len(examples["prompt"])):
                messages = examples["prompt"][i] + examples["completion"][i]
                chat_template_kwargs = examples.get("chat_template_kwargs", [{"enable_thinking": False}] * len(examples["prompt"]))[i]
                results.append(
                    tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=False,
                        **chat_template_kwargs
                    )
                )
            return results

    sft_config = SFTConfig(
        output_dir=config.sft_output_dir,
        num_train_epochs=config.sft_epochs,
        per_device_train_batch_size=config.sft_batch_size,
        gradient_accumulation_steps=config.sft_grad_accum,
        learning_rate=config.sft_lr,
        max_seq_length=config.max_seq_length,
        max_grad_norm=config.sft_max_grad_norm,
        optim=config.sft_optim,
        lr_scheduler_type=config.sft_lr_scheduler_type,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=10,
        report_to="none",
        logging_steps=1,
        weight_decay=0.01,
        warmup_ratio=0.1,
        packing=False,
        
        # For optional training + evaluation
        # eval_strategy="epoch",
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,
        # fp16_full_eval = True,
        # per_device_eval_batch_size = 4,
        # eval_accumulation_steps = 1,
    )

    # Early stopping requires evaluation to be enabled
    # callbacks = [
    #     EarlyStoppingCallback(
    #         early_stopping_patience=config.sft_early_stopping_patience,
    #         early_stopping_threshold=0.0,
    #     ),
    # ]

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=sft_train,
        eval_dataset=sft_val,
        formatting_func=formatting_func,
        tokenizer=tokenizer,
        # callbacks=callbacks,
    )
    
    # Apply Unsloth's completion-only training if selected
    if config.training_variant == "completion_only":
        print("Using completion-only training (masking prompt tokens with Unsloth)")
        # Wrap trainer to only train on assistant responses (after <|im_start|>assistant)
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
    else:
        print("Using full-finetune training (training on entire sequence)")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(config.sft_output_dir)
    return model
