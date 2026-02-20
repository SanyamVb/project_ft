# needs to be at the top for patching
from unsloth import FastLanguageModel

import json
import re
import warnings
from typing import Optional

from datasets import Dataset
from vllm import SamplingParams
from trl import GRPOConfig, GRPOTrainer

from .config import Config

warnings.filterwarnings("ignore")


def _parse_answer(answer: str) -> Optional[bool]:
    """Extract topic_flag from model output."""
    if not answer or not isinstance(answer, str):
        return None
    text = (
        answer.removeprefix("<think>")
        .split("</think>")[-1]
        .strip()
        .strip("`")
        .removeprefix("json")
        .strip()
    )
    try:
        result = json.loads(text)
        val = result.get("topic_flag")
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes", "y")
        return None
    except (json.JSONDecodeError, TypeError):
        return None


def _make_check_format(tokenizer):
    eos = getattr(tokenizer, "eos_token", None) or ""
    response_format_regex = re.compile(
        rf"^(<think>.*?</think>)?(?P<backticks>(`{{3}})?)(json)?{{\"topic_flag\":(?i:true|false), \"conf_score\":[0-1]\.[0-9][0-9]}}(?P=backticks)[\s]{{0,}}(?:{re.escape(eos)})?[\s]{{0,}}$",
        flags=re.DOTALL,
    )

    def check_format(completions, **kwargs):
        return [
            3.0 if response_format_regex.search(c[0]["content"]) is not None else 0
            for c in completions
        ]

    return check_format


def _make_check_answer(tokenizer):
    def check_answer(completions, ground_truth, **kwargs):
        return [
            3.0 if (r := _parse_answer(c[0]["content"])) is not None and r == gt else 0
            for c, gt in zip(completions, ground_truth)
        ]

    return check_answer


def load_model_for_grpo(config: Config, checkpoint_path: str):
    """Load SFT checkpoint for GRPO with vLLM (fast_inference=True)."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=config.max_seq_length,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=config.lora_rank,
        gpu_memory_utilization=config.grpo_gpu_memory_utilization,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run_grpo_train(grpo_train: Dataset, config: Config, sft_checkpoint_path: str, resume_from_checkpoint: str = None):
    """Run GRPO training on SFT checkpoint with optional resume."""
    model, tokenizer = load_model_for_grpo(config, sft_checkpoint_path)

    vllm_sampling_params = SamplingParams(
        min_p=0.01,
        top_p=0.8,
        top_k=20,
        seed=config.random_state,
        stop=[tokenizer.eos_token] if tokenizer.eos_token else [],
        include_stop_str_in_output=True,
    )

    training_args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=config.grpo_temperature,
        learning_rate=config.grpo_learning_rate,
        weight_decay=config.grpo_weight_decay,
        warmup_ratio=config.grpo_warmup_ratio,
        lr_scheduler_type=config.grpo_lr_scheduler_type,
        optim=config.grpo_optim,
        logging_steps=1,
        per_device_train_batch_size=config.grpo_batch_size,
        gradient_accumulation_steps=config.grpo_grad_accum,
        num_generations=config.grpo_num_generations,
        max_prompt_length=config.max_seq_length,
        max_completion_length=config.grpo_max_completion_length,
        num_train_epochs=config.grpo_epochs,
        save_steps=config.grpo_save_steps,
        save_total_limit=config.grpo_save_total_limit,
        report_to="none",
        output_dir=config.grpo_output_dir,
        use_vllm=True,
        mask_truncated_completions=True,
        log_completions=True,
    )

    reward_funcs = [_make_check_format(tokenizer), _make_check_answer(tokenizer)]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=grpo_train,
    )
    trainer.train()
    trainer.save_model(config.grpo_output_dir)
    return model
