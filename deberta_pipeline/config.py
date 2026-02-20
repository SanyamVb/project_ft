"""DeBERTa pipeline config."""

MODEL_OPTIONS = {
    "xsmall": "microsoft/deberta-v3-xsmall",
    "small": "microsoft/deberta-v3-small",
    "base": "microsoft/deberta-v3-base",
    "large": "microsoft/deberta-v3-large",
}

TRAIN_PATH = "data/train_1000_0822_balanced (3).xlsx"
TEST_PATH = "data/test_331_0822_balanced (2).xlsx"
MAX_LENGTH = 1024  # Token limit for (script + topic) vs transcription; increase if inputs still truncated
NUM_EPOCHS = 5
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
OUTPUT_PREFIX = "models/deberta_cross_encoder"
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# Training optimization (tunable for grid search)
MAX_GRAD_NORM = 1.0
GRADIENT_ACCUMULATION_STEPS = 2
OPTIM = "adamw_torch"
LR_SCHEDULER_TYPE = "cosine"
EARLY_STOPPING_PATIENCE = 2
LABEL_SMOOTHING = 0.1
