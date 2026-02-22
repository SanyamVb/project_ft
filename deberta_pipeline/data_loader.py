"""Load DeBERTa train/test data and build cross-encoder datasets.

Assumes the unified schema used by the Qwen pipeline: separate train and
test Excel files with Human_Combined and Prompt* columns, where the
Human_Combined/topic_flag column is already 0/1.
"""

import pandas as pd
from datasets import Dataset

from .config import TRAIN_PATH, TEST_PATH, MAX_LENGTH


def load_data(train_path: str = TRAIN_PATH, test_path: str = TEST_PATH):
    """Load train and test data for DeBERTa cross-encoder.

    Expects both train and test Excel files to follow the same schema as
    on_off_topic_combined: Human_Combined label and Prompt* columns.
    """
    def _load_one(path: str) -> pd.DataFrame:
        df = pd.read_excel(path)

        # Mirror Qwen column mapping
        column_mapping = {
            "Prompt Topic": "prompt_topic",
            "Prompt Sub Topic": "prompt_sub_topic",
            "Prompt Script": "prompt_script",
            "Prompt Level": "prompt_level",
            "Machine Transciption": "transcription",
            "Machine Transcription": "transcription",
            "Human_Combined": "topic_flag",
        }
        df.rename(columns=column_mapping, inplace=True)

        required = [
            "topic_flag",
            "transcription",
            "prompt_topic",
            "prompt_sub_topic",
            "prompt_script",
            "prompt_level",
        ]
        df = df.dropna(subset=required)
        df["topic_flag"] = df["topic_flag"].astype(int)
        df["label"] = df["topic_flag"]
        return df.reset_index(drop=True)

    train_df = _load_one(train_path)
    test_df = _load_one(test_path)
    return train_df, test_df


def build_cross_encoder_inputs(examples, tokenizer):
    """Build tokenized inputs for cross-encoder using Qwen-style prompt.

    Mirrors qwen_pipeline.dataset_utils._user_prompt so that the text seen
    by DeBERTa matches the Qwen SFT/GRPO setup.
    """
    prompts = []
    for topic, sub_topic, script, transcription, level in zip(
        examples["prompt_topic"],
        examples["prompt_sub_topic"],
        examples["prompt_script"],
        examples["transcription"],
        examples["prompt_level"],
    ):
        prompt = f"""
You are now given the below data:
Prompt Topic: {topic}
Prompt Sub-Topic: {sub_topic}
Prompt Script: {script}
Test-taker's response: {transcription}
Testing Proficiency Level: {level}

Based on your expertise, determine if the test-taker's response is off-topic or not.
"""
        prompts.append(prompt)

    # Second sequence left empty so the model effectively does single-sequence classification
    return tokenizer(
        prompts,
        [""] * len(prompts),
        truncation=True,
        max_length=MAX_LENGTH,
    )


def get_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame, tokenizer):
    """Build HuggingFace datasets with tokenized inputs.

    Assumes unified prompt_* schema (same as Qwen).
    """
    cols_to_keep = [
        "prompt_topic",
        "prompt_sub_topic",
        "prompt_script",
        "prompt_level",
        "transcription",
        "label",
    ]
    remove_columns = [
        "prompt_topic",
        "prompt_sub_topic",
        "prompt_script",
        "prompt_level",
        "transcription",
    ]

    def _map_fn(examples):
        tok = build_cross_encoder_inputs(examples, tokenizer)
        tok["labels"] = examples["label"]
        return tok

    train_hf = Dataset.from_pandas(train_df[cols_to_keep], preserve_index=False)
    test_hf = Dataset.from_pandas(test_df[cols_to_keep], preserve_index=False)

    train_hf = train_hf.map(_map_fn, batched=True, remove_columns=remove_columns)
    test_hf = test_hf.map(_map_fn, batched=True, remove_columns=remove_columns)
    return train_hf, test_hf
