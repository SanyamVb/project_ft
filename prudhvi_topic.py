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
from trl import GRPOConfig, GRPOTrainer
from dataclasses import dataclass
from datasets import Dataset
from sklearn.model_selection import train_test_split


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


@dataclass
class Config:
    LORA_RANK = 32
    LLM_MODEL = "Qwen/Qwen3-4B"
    RANDOM_STATE = 42
    DATA_PATH = "data/MC OPIC - Off Topic Annotations - Dec-Jan 2025 - Combined.xlsx"
    TRAIN_RATIO = 0.8

class RMTrainingPipeline:
    """Main pipeline for rehearsed material detection"""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def init_model(self):

        llm_model = self.config.LLM_MODEL

        base_model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = llm_model,
            max_seq_length = 4096,
            load_in_4bit = False, # False for LoRA 16bit
            fast_inference = True, # Enable vLLM fast inference
            max_lora_rank = self.config.LORA_RANK,
            gpu_memory_utilization = 0.8, # Reduce if out of memory
        )

        self.model = FastLanguageModel.get_peft_model(
            base_model,
            r = self.config.LORA_RANK, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha = self.config.LORA_RANK*2, # * 2 speeds up training
            use_gradient_checkpointing = "unsloth", # Reduces memory usage
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
            report_to = "none", # Can use Weights & Biases
            output_dir = "outputs_td_e1",
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
        """Create few-shot prompt with context examples"""
        messages = [{"role": "system", "content":"""You are an expert at detecting whether the test-taker's response is strictly associated to the given prompt topic, prompt sub-topic and prompt script or not. Being off topic refers to the test-taker's response being either totally or atleast majorly unrelated to the prompt topic, prompt sub-topic and prompt script given to the test-taker. The test-taker is in a setting, where to pass, they have to genuinely provide a response based on the prompt topic, prompt sub-topic and prompt script given to them.
GENERIC GUIDELINE FOR EVALUATION:
Focus on the overall essence of the response rather than rigid criteria. Filler words and disfluencies are not any criteria for awarding off topic flag as true.
There are two proficiency levels for which testing is being done. These two proficiency levels (along with their short forms in brackets) are given in descending order of strictness - 1) Advanced (A), 2) Intermediate (I). The more strict the testing proficiency level is the more stricter are the criteria for evaluating the off-topic flag.
Consider the context: Does the test-taker's response feel naturally connected to the specific prompt topic, prompt sub-topic and prompt script? Doesthe test-taker's response seems to be answered accordingly to the prompt topic, prompt sub-topic and prompt script? Or does it feel like the test-taker is trying to just provide a response which has no logical connection or majorly related to the prompt topic, prompt sub-topic and prompt script?
Use logical reasoning based on the given prompt topic, prompt sub-topic, prompt script, test-taker's response, to guide your judgment.
                     """}]
        # Add current case
        prompt = f"""
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
        messages.append({"role":"user", "content": prompt})
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

    def train(self, input_data: pd.DataFrame) -> Dict:
        """Predict if input is rehearsed material"""
        #messages = input_data.apply(lambda x: self.create_messages(x, context[x.name]), axis=1).to_list()
        def get_messages():
            yield from (self.create_messages(data) for _, data in input_data.iterrows())

        dataset = Dataset.from_generator(get_messages)
        self.init_model()
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

    def save_model(self) -> None:
        self.model.save_lora("final_td_e1")

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

    trainer = RMTrainingPipeline(config)

    trainer.train(train_df)

    trainer.save_model()


if __name__ == "__main__":
    main()