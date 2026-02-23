"""DeBERTa pipeline config."""

MODEL_OPTIONS = {
    "xsmall": "microsoft/deberta-v3-xsmall",
    "small": "microsoft/deberta-v3-small",
    "base": "microsoft/deberta-v3-base",
    "large": "microsoft/deberta-v3-large",
}

SYSTEM_PROMPT = """You are an expert at detecting whether the test-taker's response is strictly associated to the given prompt topic, prompt sub-topic and prompt script or not. Being off topic refers to the test-taker's response being either totally or atleast majorly unrelated to the prompt topic, prompt sub-topic and prompt script given to the test-taker. The test-taker is in a setting, where to pass, they have to genuinely provide a response based on the prompt topic, prompt sub-topic and prompt script given to them.
GENERIC GUIDELINE FOR EVALUATION:
Focus on the overall essence of the response rather than rigid criteria. Filler words and disfluencies are not any criteria for awarding off topic flag as true.
There are two proficiency levels for which testing is being done. These two proficiency levels (along with their short forms in brackets) are given in descending order of strictness - 1) Advanced (A), 2) Intermediate (I). The more strict the testing proficiency level is the more stricter are the criteria for evaluating the off-topic flag.
Consider the context: Does the test-taker's response feel naturally connected to the specific prompt topic, prompt sub-topic and prompt script? Doesthe test-taker's response seems to be answered accordingly to the prompt topic, prompt sub-topic and prompt script? Or does it feel like the test-taker is trying to just provide a response which has no logical connection or majorly related to the prompt topic, prompt sub-topic and prompt script?
Use logical reasoning based on the given prompt topic, prompt sub-topic, prompt script, test-taker's response, to guide your judgment."""

# DeBERTa uses the same unified schema as the Qwen pipeline,
# with separate train/test Excel files.
TRAIN_PATH = "data/train_0216.xlsx"
TEST_PATH = "data/test_0216.xlsx"
MAX_LENGTH = 1024  # Token limit for (script + topic) vs transcription; increase if inputs still truncated
NUM_EPOCHS = 5
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32

# Model-size-specific batch sizes (for large models with memory constraints)
BATCH_SIZE_PER_MODEL = {
    "xsmall": 32,
    "small": 32,
    "base": 16,
    "large": 4,  # Significantly reduced for memory efficiency
}

# Evaluation batch sizes (can be larger than training)
EVAL_BATCH_SIZE_PER_MODEL = {
    "xsmall": 64,
    "small": 64,
    "base": 32,
    "large": 8,  # Reduced for large model
}
LEARNING_RATE = 2.77701682113078e-05  # Best from hyperparameter tuning: Trial #7, Kappa=0.8303, Acc=0.9152
OUTPUT_PREFIX = "models/deberta_cross_encoder"
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.15  # Best from hyperparameter tuning: Trial #7

# Training optimization (best hyperparameters from 20-trial search with seed=42)
# Statistics: Mean Kappa=0.7373±0.0719 across 20 trials
MAX_GRAD_NORM = 1.5  # Best from hyperparameter tuning: Trial #7
GRADIENT_ACCUMULATION_STEPS = 1  # Best from hyperparameter tuning: Trial #7
OPTIM = "adamw_torch_fused"  # Best from hyperparameter tuning: Trial #7
LR_SCHEDULER_TYPE = "linear"  # Best from hyperparameter tuning: Trial #7
EARLY_STOPPING_PATIENCE = 2
LABEL_SMOOTHING = 0.1  # Best from hyperparameter tuning: Trial #7
