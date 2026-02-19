"""Load DeBERTa train/test data and build cross-encoder datasets."""

import pandas as pd
from datasets import Dataset

from .config import TRAIN_PATH, TEST_PATH, MAX_LENGTH


def load_data(train_path: str = TRAIN_PATH, test_path: str = TEST_PATH):
    """Load Excel files and map labels to 0/1."""
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)
    label_map = {"No": 0, "Yes": 1}
    train_df["label"] = train_df["Final_Annotation"].map(label_map)
    test_df["label"] = test_df["Final_Annotation"].map(label_map)
    return train_df, test_df


def build_cross_encoder_inputs(examples, tokenizer):
    """Build tokenized inputs for cross-encoder: (script + topic) vs transcription."""
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


def get_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame, tokenizer):
    """Build HuggingFace datasets with tokenized inputs."""
    cols_to_keep = ["Script", "Category", "Transcription-Machine", "label"]

    def _map_fn(examples):
        tok = build_cross_encoder_inputs(examples, tokenizer)
        tok["labels"] = examples["label"]
        return tok

    train_hf = Dataset.from_pandas(train_df[cols_to_keep], preserve_index=False)
    test_hf = Dataset.from_pandas(test_df[cols_to_keep], preserve_index=False)

    train_hf = train_hf.map(
        _map_fn,
        batched=True,
        remove_columns=["Script", "Category", "Transcription-Machine"],
    )
    test_hf = test_hf.map(
        _map_fn,
        batched=True,
        remove_columns=["Script", "Category", "Transcription-Machine"],
    )
    return train_hf, test_hf
