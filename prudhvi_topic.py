# needs to be at the top for patching
from unsloth import FastLanguageModel

import re
import os
import pandas as pd
import torch
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

from time import time
from typing import List, Dict, Tuple
from vllm import SamplingParams
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
from transformers.trainer_callback import EarlyStoppingCallback
from dataclasses import dataclass
from datasets import Dataset
from sklearn.model_selection import train_test_split
import gc


def normalize_topic_flag(value) -> int:
    """Normalize Human_Combined to 0/1. Handles Yes/No, True/False, 1/0."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in ("yes", "true", "1", "y"):
        return 1
    if s in ("no", "false", "0", "n"):
        return 0
    try:
        v = int(float(value))
        return 1 if v == 1 else 0
    except (TypeError, ValueError):
        return np.nan


SYSTEM_PROMPT = """You are an expert at detecting whether the test-taker's response is strictly associated to the given prompt topic, prompt sub-topic and prompt script or not. Being off topic refers to the test-taker's response being either totally or atleast majorly unrelated to the prompt topic, prompt sub-topic and prompt script given to the test-taker. The test-taker is in a setting, where to pass, they have to genuinely provide a response based on the prompt topic, prompt sub-topic and prompt script given to them.
GENERIC GUIDELINE FOR EVALUATION:
Focus on the overall essence of the response rather than rigid criteria. Filler words and disfluencies are not any criteria for awarding off topic flag as true.
There are two proficiency levels for which testing is being done. These two proficiency levels (along with their short forms in brackets) are given in descending order of strictness - 1) Advanced (A), 2) Intermediate (I). The more strict the testing proficiency level is the more stricter are the criteria for evaluating the off-topic flag.
Consider the context: Does the test-taker's response feel naturally connected to the specific prompt topic, prompt sub-topic and prompt script? Doesthe test-taker's response seems to be answered accordingly to the prompt topic, prompt sub-topic and prompt script? Or does it feel like the test-taker is trying to just provide a response which has no logical connection or majorly related to the prompt topic, prompt sub-topic and prompt script?
Use logical reasoning based on the given prompt topic, prompt sub-topic, prompt script, test-taker's response, to guide your judgment."""


@dataclass
class Config:
    LORA_RANK = 32
    LLM_MODEL = "Qwen/Qwen3-4B"
    RANDOM_STATE = 42
    DATA_PATH = "data/MC OPIC - Off Topic Annotations - Dec-Jan 2025 - Combined.xlsx"
    TRAIN_RATIO = 0.8
    MAX_SEQ_LENGTH = 4096

    # SFT config
    SFT_OUTPUT_DIR = "outputs_sft"
    SFT_EPOCHS = 2
    SFT_LR = 2e-5
    SFT_BATCH_SIZE = 4
    SFT_GRAD_ACCUM = 4
    SFT_MAX_GRAD_NORM = 1.0
    SFT_OPTIM = "adamw_torch"
    SFT_LR_SCHEDULER_TYPE = "cosine"
    SFT_EARLY_STOPPING_PATIENCE = 2
    SFT_VAL_RATIO = 0.1  # fraction of train set held for validation
    CONF_SCORE_SFT = 0.95

    # GRPO config
    GRPO_OUTPUT_DIR = "outputs_td_e1"
    GPU_MEMORY_UTILIZATION = 0.8

class RMTrainingPipeline:
    """Main pipeline for rehearsed material detection"""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _user_prompt(self, input_data: pd.Series) -> str:
        """Build the user prompt from a data row."""
        return f"""
You are now given the below data:
Prompt Topic: {input_data['prompt_topic']}
Prompt Sub-Topic: {input_data['prompt_sub_topic']}
Prompt Script: {input_data['prompt_script']}
Test-taker's response: {input_data['transcription']}
Testing Proficiency Level: {input_data['prompt_level']}

Based on your expertise, determine if the test-taker's response is off-topic or not.
Provide your answer in this format as a json only:
{{"topic_flag": the flag is a boolean donoting if based on the given data, the test-taker's response is off-topic or not, "conf_score": a floating number between 0 and 1 denoting how confident are you in deciding the topic_flag value for the give data}}
"""

    def create_sft_messages(self, input_data: pd.Series) -> dict:
        """Create SFT example with system + user + assistant (ground-truth completion)."""
        topic_flag = bool(int(input_data["topic_flag"]))
        completion = json.dumps({"topic_flag": topic_flag, "conf_score": self.config.CONF_SCORE_SFT})
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._user_prompt(input_data)},
            {"role": "assistant", "content": completion},
        ]
        return {"messages": messages}

    # ── SFT model loading ──────────────────────────────────────────────

    def init_sft_model(self):
        """Load model with LoRA for SFT (no vLLM)."""
        base_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.LLM_MODEL,
            max_seq_length=self.config.MAX_SEQ_LENGTH,
            load_in_4bit=False,
            fast_inference=False,  # no vLLM for SFT
            max_lora_rank=self.config.LORA_RANK,
        )
        self.model = FastLanguageModel.get_peft_model(
            base_model,
            r=self.config.LORA_RANK,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=self.config.LORA_RANK * 2,
            use_gradient_checkpointing="unsloth",
            random_state=self.config.RANDOM_STATE,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Set chat template for Qwen models if missing
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = (
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

    # ── SFT training ───────────────────────────────────────────────────

    def train_sft(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Phase 1: Supervised fine-tuning."""
        print("\n" + "="*60)
        print("PHASE 1: SFT Training")
        print("="*60)

        self.init_sft_model()

        # Build datasets
        sft_train = Dataset.from_list(
            [self.create_sft_messages(row) for _, row in train_df.iterrows()]
        )
        sft_val = Dataset.from_list(
            [self.create_sft_messages(row) for _, row in val_df.iterrows()]
        )
        print(f"SFT train: {len(sft_train)} | SFT val: {len(sft_val)}")

        tokenizer = self.tokenizer  # capture for closure

        def formatting_func(examples):
            """Format messages for SFTTrainer via chat template."""
            if isinstance(examples["messages"][0], dict):
                return [tokenizer.apply_chat_template(
                    examples["messages"], tokenize=False, add_generation_prompt=False
                )]
            else:
                return [
                    tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                    for msgs in examples["messages"]
                ]

        sft_config = SFTConfig(
            output_dir=self.config.SFT_OUTPUT_DIR,
            num_train_epochs=self.config.SFT_EPOCHS,
            per_device_train_batch_size=self.config.SFT_BATCH_SIZE,
            gradient_accumulation_steps=self.config.SFT_GRAD_ACCUM,
            learning_rate=self.config.SFT_LR,
            max_seq_length=self.config.MAX_SEQ_LENGTH,
            max_grad_norm=self.config.SFT_MAX_GRAD_NORM,
            optim=self.config.SFT_OPTIM,
            lr_scheduler_type=self.config.SFT_LR_SCHEDULER_TYPE,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            logging_steps=10,
            weight_decay=0.01,
            warmup_ratio=0.1,
            packing=False,
        )

        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=self.config.SFT_EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=0.0,
            ),
        ]

        trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=sft_train,
            eval_dataset=sft_val,
            formatting_func=formatting_func,
            tokenizer=self.tokenizer,
            callbacks=callbacks,
        )
        trainer.train()
        trainer.save_model(self.config.SFT_OUTPUT_DIR)
        print(f"SFT model saved to {self.config.SFT_OUTPUT_DIR}")

        # Free GPU memory before GRPO phase
        del trainer, self.model, self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # ── GRPO model loading ─────────────────────────────────────────────

    def init_model(self, checkpoint_path: str = None):
        """Load model for GRPO. If checkpoint_path is given, load from SFT checkpoint."""
        model_name = checkpoint_path or self.config.LLM_MODEL

        base_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = self.config.MAX_SEQ_LENGTH,
            load_in_4bit = False, # False for LoRA 16bit
            fast_inference = True, # Enable vLLM fast inference
            max_lora_rank = self.config.LORA_RANK,
            gpu_memory_utilization = self.config.GPU_MEMORY_UTILIZATION,
        )

        self.model = FastLanguageModel.get_peft_model(
            base_model,
            r = self.config.LORA_RANK,
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha = self.config.LORA_RANK*2,
            use_gradient_checkpointing = "unsloth",
            random_state = self.config.RANDOM_STATE,
        )

        self.vllm_sampling_params = SamplingParams(
            min_p = 0.01,
            top_p = 0.8,
            top_k = 20,
            seed = self.config.RANDOM_STATE,
            stop = [self.tokenizer.eos_token],
            include_stop_str_in_output = True,
        )

        self.training_args = GRPOConfig(
            vllm_sampling_params = self.vllm_sampling_params,
            temperature = 0.7,
            learning_rate = 5e-6,
            weight_decay = 0.001,
            warmup_ratio = 0.1,
            lr_scheduler_type = "linear",
            optim = "adamw_8bit",
            logging_steps = 1,
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 4, # Increase to 4 for smoother training
            num_generations = 4, # Decrease if out of memory
            max_prompt_length = 4096,
            max_completion_length = 64,
            num_train_epochs = 1, # Set to 1 for a full training run
            #max_steps = 100,
            save_steps = 500,
            save_total_limit = 10,
            report_to = "none",
            output_dir = self.config.GRPO_OUTPUT_DIR,
            chat_template_kwargs = {"enable_thinking":False},
            use_vllm = True,
            #vllm_mode = "colocate"
            # Below enables GSPO:
            #importance_sampling_level = "sequence",
            mask_truncated_completions = True,
            log_completions = True,
            #loss_type = "dr_grpo",

            # For optional training + evaluation
            # fp16_full_eval = True,
            # per_device_eval_batch_size = 4,
            # eval_accumulation_steps = 1,
            # eval_strategy = "steps",
            # eval_steps = 1,
        )

        self.response_format_regex = re.compile(rf"^(<think>.*?</think>)?(?P<backticks>(`{{3}})?)(json)?{{\"topic_flag\":(?i:true|false), \"conf_score\":[0-1]\.[0-9][0-9]}}(?P=backticks)[\s]{{0,}}(?:{re.escape(self.tokenizer.eos_token)})?[\s]{{0,}}$" ,flags = re.DOTALL)

    def create_messages(self, input_data: pd.Series) -> str:
        """Create GRPO prompt (system + user, no assistant) with ground_truth."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": self._user_prompt(input_data)})
        return { "prompt" : messages, "ground_truth": input_data["topic_flag"]}
#"reasoning": "reasoning based on which you decided the rm_flag"
    def check_format(self, completions, **kwargs):
        return [3.0 if self.response_format_regex.search(completion[0]["content"]) is not None else 0 for completion in completions]

    def check_answer(self, completions, ground_truth, **kwargs):
        return [3.0 if (r:=self.parse_answer(completion[0]["content"])) is not None and r == gt else 0 for completion, gt in zip(completions, ground_truth)]

    def parse_answer(self, answer):
        answer = answer.removeprefix("<think>").split("</think>")[-1].strip().strip("`").removeprefix("json").strip()
        try:
            result = json.loads(answer)["topic_flag"]
        except Exception as e:
            result = None
        return result

    def train_grpo(self, input_data: pd.DataFrame, checkpoint_path: str = None) -> Dict:
        """Phase 2: GRPO reward-based training."""
        print("\n" + "="*60)
        print("PHASE 2: GRPO Training")
        print("="*60)

        def get_messages():
            yield from (self.create_messages(data) for _, data in input_data.iterrows())

        dataset = Dataset.from_generator(get_messages)
        self.init_model(checkpoint_path=checkpoint_path)
        trainer = GRPOTrainer(
            model = self.model,
            processing_class = self.tokenizer,
            reward_funcs = [
                self.check_format,
                self.check_answer,
            ],
            args = self.training_args,
            train_dataset = dataset,

            # For optional training + evaluation
            # train_dataset = new_dataset["train"],
            # eval_dataset = new_dataset["test"],
        )
        trainer.train()

    def save_model(self, path: str = "final_td_e1") -> None:
        self.model.save_lora(path)

def main():
    """Main entry point"""
    config = Config()

    # Load data
    print("Loading data from", config.DATA_PATH)
    df = pd.read_excel(config.DATA_PATH)

    # Rename columns to match expected format
    column_mapping = {
        'Prompt Topic': 'prompt_topic',
        'Prompt Sub Topic': 'prompt_sub_topic',
        'Prompt Script': 'prompt_script',
        'Prompt Level': 'prompt_level',
        'Machine Transciption': 'transcription',
        'Human_Combined': 'topic_flag'
    }
    df.rename(columns=column_mapping, inplace=True)

    # Normalize topic_flag to 0/1 (handles Yes/No, True/False, 1/0)
    df["topic_flag"] = df["topic_flag"].apply(normalize_topic_flag)

    # Drop rows with NaN in required columns
    required = ["topic_flag", "transcription", "prompt_topic", "prompt_sub_topic", "prompt_script", "prompt_level"]
    before = len(df)
    df = df.dropna(subset=required)
    df["topic_flag"] = df["topic_flag"].astype(int)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing values in required columns")

    # Stratified 80/20 train-test split on full dataset
    train_df, test_df = train_test_split(
        df,
        train_size=config.TRAIN_RATIO,
        stratify=df["topic_flag"],
        random_state=config.RANDOM_STATE,
    )
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    print(f"Train class balance: {train_df['topic_flag'].value_counts().to_dict()}")

    # Carve a validation set from train for SFT early stopping
    sft_train_df, sft_val_df = train_test_split(
        train_df,
        test_size=config.SFT_VAL_RATIO,
        stratify=train_df["topic_flag"],
        random_state=config.RANDOM_STATE,
    )
    print(f"SFT split: {len(sft_train_df)} train | {len(sft_val_df)} val")

    pipeline = RMTrainingPipeline(config)

    # Phase 1: SFT
    pipeline.train_sft(sft_train_df, sft_val_df)

    # Phase 2: GRPO (loads from SFT checkpoint)
    pipeline.train_grpo(train_df, checkpoint_path=config.SFT_OUTPUT_DIR)

    pipeline.save_model("final_td_e1")


if __name__ == "__main__":
    main()