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

# Model-size-specific batch sizes (for large models with memory constraints)
BATCH_SIZE_PER_MODEL = {
    "xsmall": 32,
    "small": 32,
    "base": 16,
    "large": 8,  # Reduced for memory efficiency
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
