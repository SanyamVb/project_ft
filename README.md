# Off-Topic Detection Pipelines

LLM-based and cross-encoder off-topic detection for spoken test-taker responses. Includes Qwen SFT, DeBERTa cross-encoder, and GRPO pipelines.

## Project Structure

```
project_ft/
  scripts/
    __init__.py
    hyperparameter_search.py  # Optuna hyperparameter tuning with reproducibility
  qwen_pipeline/         # Qwen SFT pipeline (0.6B, 4B, 8B)
    preflight_check.py
    config.py
    run_pipeline.py
    train_sft.py![1771678209117](image/README/1771678209117.png)![1771678212773](image/README/1771678212773.png)![1771678223679](image/README/1771678223679.png)![1771678227568](image/README/1771678227568.png)
    inference.py
    dataset_utils.py
    eval_utils.py
  deberta_pipeline/      # DeBERTa cross-encoder (xsmall, small, base, large)
    preflight_check.py
    config.py
    data_loader.py
    train.py
    inference.py
    eval_utils.py
    run_pipeline.py
  benchmark_report.py    # Unified Qwen vs DeBERTa comparison
  prudhvi_topic.py       # GRPO-only pipeline
  cross_encoder_offtopic.py  # Legacy DeBERTa standalone script
  data/
  models/                # DeBERTa checkpoints
  outputs_sft_*/         # Qwen checkpoints per size
```

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for training)
- Dependencies: `unsloth`, `trl`, `transformers`, `datasets`, `peft`, `pandas`, `openpyxl`, `sklearn`, `torch`, `vllm` (for GRPO)

Install:

```bash
pip install unsloth trl transformers datasets peft pandas openpyxl scikit-learn torch
# For GRPO (prudhvi_topic.py): pip install vllm
# For hyperparameter search: pip install optuna
```

## Data

**Qwen SFT pipeline** uses `data/on_off_topic_combined.xlsx` with columns:

- Prompt Topic, Prompt Sub Topic, Prompt Script, Prompt Level
- Machine Transciption / Machine Transcription
- Human_Combined (topic_flag: 0/1, Yes/No)

Data is automatically split **80/20** (train/test). The test set is used for both validation during training and final evaluation.

**DeBERTa pipeline** uses separate train/test Excel files:

- `data/train_1000_0822_balanced.xlsx`, `data/test_331_0822_balanced.xlsx`
- Columns: Script, Category, Transcription-Machine, Final_Annotation (Yes/No)

---

## Pre-flight Checks (before training)

Run checks to catch data/config issues before training starts:

```bash
python -m deberta_pipeline.preflight_check --model_size base
python -m deberta_pipeline.preflight_check --skip-model   # Faster, skips model load

python -m qwen_pipeline.preflight_check --model_size 4B
python -m qwen_pipeline.preflight_check --skip-model
```

Exits 0 if all checks pass, 1 otherwise.

---

## Qwen SFT Pipeline

### Full run (single model size)

```bash
python -m qwen_pipeline.run_pipeline --mode full
# With model size (0.6B, 4B, 8B):
python -m qwen_pipeline.run_pipeline --mode full --model_size 4B
```

### Step-by-step

```bash
python -m qwen_pipeline.run_pipeline --mode process_only
python -m qwen_pipeline.run_pipeline --mode train_only --model_size 4B
python -m qwen_pipeline.run_pipeline --mode inference_only --checkpoint outputs_sft_4B
python -m qwen_pipeline.run_pipeline --mode eval_only --checkpoint outputs_sft_4B
```

### Train all sizes (for benchmark)

```bash
python -m qwen_pipeline.run_pipeline --mode train_all
```

Saves to `outputs_sft_0.6B/`, `outputs_sft_4B/`, `outputs_sft_8B/` with `benchmark_metrics.json`.

### GRPO after SFT

To run GRPO training after SFT (SFT -> GRPO -> inference -> eval):

```bash
python -m qwen_pipeline.run_pipeline --mode full --run_grpo --model_size 4B
python -m qwen_pipeline.run_pipeline --mode train_only --run_grpo --model_size 4B
```

Requires `vllm`. GRPO checkpoints save to `outputs_grpo_{model_size}/`.

---

## Training Variant Comparison (Qwen)

Compare **completion-only** (trains only on assistant responses) vs **full-finetune** (trains on entire conversation) to determine which approach works better for your dataset.

### Run both variants

```bash
# Train with completion-only (default, masks prompt tokens)
python -m qwen_pipeline.run_pipeline --mode full --model_size 4B --training_variant completion_only

# Train with full-finetune (trains on entire sequence)
python -m qwen_pipeline.run_pipeline --mode full --model_size 4B --training_variant full_finetune
```

Output directories will be variant-specific:
- `outputs_sft_4B_completion_only/`
- `outputs_sft_4B_full_finetune/`

### Compare results

```bash
python benchmark_report.py
```

The report shows:
- **Side-by-side comparison table** with accuracy, kappa, and training time for each variant
- **Delta metrics** (completion-only vs full-finetune)
- **Recommendation** based on performance differences

### Interpreting results

- **Similar accuracy (<0.5% difference)**: Use completion-only (faster training, more focused learning)
- **Completion-only wins**: Better generalization by not overfitting to prompt patterns
- **Full-finetune wins**: Dataset benefits from learning the full conversation structure

**Typical expectations:**
- Completion-only: Faster convergence, better format compliance, less overfitting
- Full-finetune: May learn prompt patterns, potentially higher training loss initially

---

## DeBERTa Cross-Encoder Pipeline

### Train single size

```bash
python -m deberta_pipeline.run_pipeline --mode train --model_size base
```

Model sizes: `xsmall`, `small`, `base`, `large`.

### Train all sizes (for benchmark)

```bash
python -m deberta_pipeline.run_pipeline --mode train_all
```

Saves to `models/deberta_cross_encoder_{size}_offtopic/`.

### Eval only

```bash
python -m deberta_pipeline.run_pipeline --mode eval --model_size base
```

---

## Unified Benchmark Comparison

After training Qwen and/or DeBERTa models, generate a comparison report:

```bash
python benchmark_report.py
python benchmark_report.py --plot   # Also save scatter plot
```

Output:

- `benchmark_results.csv` - all models in one table
- Console summary sorted by accuracy and inference speed
- Optional: `benchmark_accuracy_vs_inference.png`

### Full comparison workflow

```bash
python -m deberta_pipeline.run_pipeline --mode train_all
python -m qwen_pipeline.run_pipeline --mode train_all
python benchmark_report.py --plot
```

Use the report to choose: best accuracy, fastest inference, or best accuracy-per-latency tradeoff.

---

## GRPO Pipeline

```bash
python prudhvi_topic.py
```

Reads from `data/MC OPIC - Off Topic Annotations - Dec-Jan 2025 - Combined.xlsx`, 80/20 train-test split. Requires `vllm`.

---

## Hyperparameter Search

Run Optuna-based hyperparameter search to find optimal training hyperparameters for better accuracy/kappa.

### Installation

```bash
pip install optuna
```

### Basic Usage

**DeBERTa hyperparameter tuning** (recommended with seed for reproducibility):

```bash
python -m scripts.hyperparameter_search --pipeline deberta --model_size base --n_trials 20 --seed 42 --output results/tune_deberta.json
```

**Qwen hyperparameter tuning**:

```bash
python -m scripts.hyperparameter_search --pipeline qwen --model_size 0.6B --n_trials 15 --seed 42 --output results/tune_qwen.json
```

### Options

- `--pipeline` - `deberta` or `qwen` (required)
- `--model_size` - Model size: DeBERTa (`xsmall`, `small`, `base`, `large`), Qwen (`0.6B`, `4B`, `8B`)
- `--n_trials` - Number of Optuna trials (default: 20)
- `--seed` - Random seed for reproducibility (recommended: 42)
- `--quick` - Quick mode: fewer epochs/subset of data for faster testing
- `--grid` - Use full grid search instead of Optuna (DeBERTa only, exhaustive but slow)
- `--output` - Output JSON path for results (default: `results/tune_results.json`)

### Hyperparameters Tuned (DeBERTa)

- **Learning Rate**: 1e-5 to 3e-5 (log scale)
- **Max Gradient Norm**: [0.5, 1.0, 1.5]
- **Warmup Ratio**: [0.05, 0.1, 0.15]
- **Label Smoothing**: [0.0, 0.1]
- **Gradient Accumulation Steps**: [1, 2, 4]
- **Optimizer**: ["adamw_torch", "adamw_torch_fused"]
- **LR Scheduler**: ["cosine", "linear"]

### Reproducibility

**Important**: Always use `--seed` for reproducible results. Without it, results will vary between runs due to random initialization.

Each trial uses `seed + trial_number`, ensuring:
- Same hyperparameter suggestions from Optuna
- Same weight initialization
- Same data shuffling
- Deterministic CUDA operations

### Output

Results are saved to JSON with:
- Best hyperparameters and kappa score
- Best trial checkpoint path
- Statistics: mean kappa ± std across all trials
- All trial results for analysis
- Trial state (completed/failed)

### Example Workflow

1. **Quick test** (10 trials, 2 epochs):
```bash
python -m scripts.hyperparameter_search --pipeline deberta --model_size base --n_trials 10 --quick --seed 42 --output results/tune_quick.json
```

2. **Full search** (20 trials, full training):
```bash
python -m scripts.hyperparameter_search --pipeline deberta --model_size base --n_trials 20 --seed 42 --output results/tune_deberta_base.json
```

3. **Apply best parameters** to `deberta_pipeline/config.py` and verify:
```bash
# Update config.py with best hyperparameters from JSON
python -m deberta_pipeline.run_pipeline --mode train --model_size base
```

4. **Compare results**: Check if kappa matches tuning results. Note: Some variance is normal due to early stopping and different random seeds.

### Grid Search (DeBERTa only)

For exhaustive search over a smaller predefined space:

```bash
python -m scripts.hyperparameter_search --pipeline deberta --grid --seed 42 --output results/tune_grid.json
```

Grid search tests all combinations but is slower than Optuna's Bayesian optimization.

---

## Config

**Qwen** (`qwen_pipeline/config.py`): `data_path`, `llm_model`, `sft_epochs`, `sft_batch_size`, `max_position_embeddings`, `sft_max_grad_norm`, `sft_optim`, `sft_lr_scheduler_type`, `sft_early_stopping_patience`.

**DeBERTa** (`deberta_pipeline/config.py`): `TRAIN_PATH`, `TEST_PATH`, `MODEL_OPTIONS`, `NUM_EPOCHS`, `BATCH_SIZE`, `MAX_GRAD_NORM`, `GRADIENT_ACCUMULATION_STEPS`, `OPTIM`, `LR_SCHEDULER_TYPE`, `EARLY_STOPPING_PATIENCE`, `LABEL_SMOOTHING`.

---

## Troubleshooting

**Hyperparameter search results not reproducible**

- Always use `--seed` parameter: `python -m scripts.hyperparameter_search --pipeline deberta --seed 42 ...`
- Without seed, random initialization causes different results each run
- Even with seed, some variance is expected due to early stopping and GPU non-determinism

**Hyperparameter tuning results don't match when re-training**

- Tuning may have used `--quick` mode (2 epochs) while validation uses full epochs
- Check the `quick_mode` field in the JSON results
- Optuna optimizes threshold-adjusted kappa, which may differ from eval metrics during training
- Consider the mean kappa ± std from the statistics section, not just the best single trial

**Checkpoint not found**

- Use `--checkpoint` to point to the output directory.
- Run training first: `python -m qwen_pipeline.run_pipeline --mode train_only` or `python -m deberta_pipeline.run_pipeline --mode train`.

**Out of memory**

- Qwen: Lower `sft_batch_size` or `sft_grad_accum` in `qwen_pipeline/config.py`.
- DeBERTa: Lower `BATCH_SIZE` in `deberta_pipeline/config.py`.

**Module not found**

- Install: `pip install unsloth trl transformers datasets peft pandas openpyxl scikit-learn torch`.
