"""Load excel, split, and build SFT/GRPO datasets."""

import json
import pandas as pd
import numpy as np
from typing import Tuple
from datasets import Dataset
from sklearn.model_selection import train_test_split

from .config import Config, SYSTEM_PROMPT


def normalize_topic_flag(value) -> float:
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
        return 1.0 if v == 1 else 0.0
    except (TypeError, ValueError):
        return np.nan


def load_and_prepare_df(config: Config) -> pd.DataFrame:
    """Load excel, rename columns, normalize labels, drop invalid rows."""
    df = pd.read_excel(config.data_path)
    df.rename(columns=config.column_mapping, inplace=True)

    df["topic_flag"] = df["topic_flag"].apply(normalize_topic_flag)

    required = ["topic_flag", "transcription", "prompt_topic", "prompt_sub_topic", "prompt_script", "prompt_level"]
    df = df.dropna(subset=required)
    df["topic_flag"] = df["topic_flag"].astype(int)
    return df.reset_index(drop=True)


def train_val_test_split(
    df: pd.DataFrame, config: Config
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 train/val/test split."""
    train_val, test_df = train_test_split(
        df,
        test_size=config.test_ratio,
        stratify=df["topic_flag"],
        random_state=config.random_state,
    )
    val_ratio_adjusted = config.val_ratio / (config.train_ratio + config.val_ratio)
    train_df, val_df = train_test_split(
        train_val,
        test_size=val_ratio_adjusted,
        stratify=train_val["topic_flag"],
        random_state=config.random_state,
    )
    return train_df, val_df, test_df


def _user_prompt(row: pd.Series) -> str:
    return f"""
You are now given the below data:
Prompt Topic: {row['prompt_topic']}
Prompt Sub-Topic: {row['prompt_sub_topic']}
Prompt Script: {row['prompt_script']}
Test-taker's response: {row['transcription']}
Testing Proficiency Level: {row['prompt_level']}

Based on your expertise, determine if the test-taker's response is off-topic or not.
Provide your answer in this format as a json only:
{{"topic_flag": the flag is a boolean donoting if based on the given data, the test-taker's response is off-topic or not, "conf_score": a floating number between 0 and 1 denoting how confident are you in deciding the topic_flag value for the give data}}
"""


def create_sft_example(row: pd.Series, config: Config) -> dict:
    """Create SFT example with prompt + completion for TRL SFTTrainer."""
    topic_flag = bool(int(row["topic_flag"]))
    completion_content = json.dumps(
        {"topic_flag": topic_flag, "conf_score": config.conf_score_sft}
    )
    return {
        "prompt": [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": _user_prompt(row)},
        ],
        "completion": [{"role": "assistant", "content": completion_content}],
    }


def create_grpo_example(row: pd.Series, config: Config) -> dict:
    """Create GRPO example with prompt and ground_truth (for Phase 2)."""
    return {
        "prompt": [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": _user_prompt(row)},
        ],
        "ground_truth": int(row["topic_flag"]),
    }


def build_sft_dataset(df: pd.DataFrame, config: Config) -> Dataset:
    """Build HuggingFace Dataset for SFT (prompt + completion)."""
    examples = [create_sft_example(row, config) for _, row in df.iterrows()]
    return Dataset.from_list(examples)


def build_grpo_dataset(df: pd.DataFrame, config: Config) -> Dataset:
    """Build HuggingFace Dataset for GRPO (prompt + ground_truth)."""
    examples = [create_grpo_example(row, config) for _, row in df.iterrows()]
    return Dataset.from_list(examples)


def get_all_splits(config: Config) -> dict:
    """Load data, split, and build SFT/GRPO datasets."""
    df = load_and_prepare_df(config)
    train_df, val_df, test_df = train_val_test_split(df, config)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "sft_train": build_sft_dataset(train_df, config),
        "sft_val": build_sft_dataset(val_df, config),
        "grpo_train": build_grpo_dataset(train_df, config),
        "grpo_val": build_grpo_dataset(val_df, config),
    }
