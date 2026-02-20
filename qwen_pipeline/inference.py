"""Run trained model inference on a dataset."""

import json
from typing import List, Optional

import pandas as pd
import torch

from .config import Config
from .dataset_utils import _user_prompt


def parse_answer(answer: str) -> Optional[bool]:
    """Extract topic_flag from model output. Handles <think>, backticks, json prefix."""
    if not answer or not isinstance(answer, str):
        return None
    text = answer.removeprefix("<think>").split("</think>")[-1].strip().strip("`").removeprefix("json").strip()
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


def load_trained_model(config: Config, checkpoint_path: Optional[str] = None):
    """Load trained SFT model from checkpoint."""
    path = checkpoint_path or config.sft_output_dir
    model, tokenizer = __load_unsloth_model(path, config)
    return model, tokenizer


def __load_unsloth_model(path: str, config: Config):
    """Load saved SFT model. Uses PEFT when path contains adapter, else Unsloth."""
    import os
    from transformers import AutoTokenizer

    adapter_config = os.path.join(path, "adapter_config.json")
    if os.path.exists(adapter_config):
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
        return model, tokenizer

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path,
        max_seq_length=config.max_seq_length,
        load_in_4bit=False,
        dtype=None,
    )
    # Enable inference mode properly
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _messages_for_row(row: pd.Series, config: Config) -> List[dict]:
    """Build chat messages for a single row (system + user)."""
    return [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": _user_prompt(row)},
    ]


def predict_single(model, tokenizer, messages: List[dict], max_new_tokens: int = 128) -> str:
    """Run inference for a single prompt."""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.3,
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return generated


def predict_batch(
    df: pd.DataFrame,
    model,
    tokenizer,
    config: Config,
    max_new_tokens: int = 128,
) -> List[dict]:
    """Run inference on each row. Returns list of {topic_flag, conf_score, raw}."""
    results = []
    for _, row in df.iterrows():
        messages = _messages_for_row(row, config)
        raw = predict_single(model, tokenizer, messages, max_new_tokens)
        topic_flag = parse_answer(raw)
        conf_score = 0.5
        try:
            text = raw.removeprefix("<think>").split("</think>")[-1].strip().strip("`").removeprefix("json").strip()
            parsed = json.loads(text)
            conf_score = float(parsed.get("conf_score", 0.5))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        results.append({
            "topic_flag": topic_flag,
            "conf_score": conf_score,
            "raw": raw,
        })
    return results
