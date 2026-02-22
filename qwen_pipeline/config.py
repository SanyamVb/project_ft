"""Config and column mapping for SFT pipeline."""

from dataclasses import dataclass, field


QWEN_MODEL_OPTIONS = {
    "0.6B": "unsloth/Qwen3-0.6B-Base",
    "4B": "Qwen/Qwen3-4B",
}


SYSTEM_PROMPT = """You are an expert at detecting whether the test-taker's response is strictly associated to the given prompt topic, prompt sub-topic and prompt script or not. Being off topic refers to the test-taker's response being either totally or atleast majorly unrelated to the prompt topic, prompt sub-topic and prompt script given to the test-taker. The test-taker is in a setting, where to pass, they have to genuinely provide a response based on the prompt topic, prompt sub-topic and prompt script given to them.
GENERIC GUIDELINE FOR EVALUATION:
Focus on the overall essence of the response rather than rigid criteria. Filler words and disfluencies are not any criteria for awarding off topic flag as true.
There are two proficiency levels for which testing is being done. These two proficiency levels (along with their short forms in brackets) are given in descending order of strictness - 1) Advanced (A), 2) Intermediate (I). The more strict the testing proficiency level is the more stricter are the criteria for evaluating the off-topic flag.
Consider the context: Does the test-taker's response feel naturally connected to the specific prompt topic, prompt sub-topic and prompt script? Doesthe test-taker's response seems to be answered accordingly to the prompt topic, prompt sub-topic and prompt script? Or does it feel like the test-taker is trying to just provide a response which has no logical connection or majorly related to the prompt topic, prompt sub-topic and prompt script?
Use logical reasoning based on the given prompt topic, prompt sub-topic, prompt script, test-taker's response, to guide your judgment."""


def _default_column_mapping():
    return {
        "Prompt Topic": "prompt_topic",
        "Prompt Sub Topic": "prompt_sub_topic",
        "Prompt Script": "prompt_script",
        "Prompt Level": "prompt_level",
        "Machine Transciption": "transcription",
        "Machine Transcription": "transcription",  # fallback for correct spelling
        "Human_Combined": "topic_flag",
    }


@dataclass
class Config:
    data_path: str = "data/on_off_topic_combined.xlsx"
    column_mapping: dict = field(default_factory=_default_column_mapping)
    train_ratio: float = 0.8
    test_ratio: float = 0.2
    random_state: int = 42
    llm_model: str = "Qwen/Qwen3-4B"
    lora_rank: int = 32
    sft_epochs: int = 2
    sft_lr: float = 2e-5
    sft_output_dir: str = "outputs_sft"
    sft_batch_size: int = 4
    sft_grad_accum: int = 4
    sft_max_grad_norm: float = 1.0
    sft_optim: str = "adamw_torch"
    sft_lr_scheduler_type: str = "cosine"
    sft_early_stopping_patience: int = 2
    max_position_embeddings: int = 4096
    conf_score_sft: float = 0.95
    training_variant: str = "completion_only"  # "completion_only" or "full_finetune"
    system_prompt: str = field(default_factory=lambda: SYSTEM_PROMPT)

    # GRPO (after SFT)
    grpo_output_dir: str = "outputs_grpo"
    grpo_learning_rate: float = 5e-6
    grpo_epochs: int = 1
    grpo_batch_size: int = 4
    grpo_grad_accum: int = 4
    grpo_num_generations: int = 4
    grpo_max_completion_length: int = 64
    grpo_temperature: float = 0.7
    grpo_weight_decay: float = 0.001
    grpo_warmup_ratio: float = 0.1
    grpo_lr_scheduler_type: str = "linear"
    grpo_optim: str = "adamw_8bit"
    grpo_save_steps: int = 500
    grpo_save_total_limit: int = 10
    grpo_gpu_memory_utilization: float = 0.8

    def __post_init__(self):
        """Validate configuration values."""
        valid_variants = ["completion_only", "full_finetune"]
        if self.training_variant not in valid_variants:
            raise ValueError(
                f"training_variant must be one of {valid_variants}, got: {self.training_variant}"
            )
