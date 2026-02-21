# Off-Topic Detection Pipelines

LLM-based and cross-encoder off-topic detection for spoken test-taker responses. Includes Qwen SFT, DeBERTa cross-encoder, and GRPO pipelines.

## Project Structure

```
project_ft/
  scripts/
    hyperparameter_search.py  # Optuna hyperparameter tuning
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

Run Optuna-based hyperparameter search to tune training for better accuracy/kappa:

```bash
pip install optuna
python -m scripts.hyperparameter_search --pipeline deberta --n_trials 20 --output results/tune_deberta.json
python -m scripts.hyperparameter_search --pipeline qwen --n_trials 15 --model_size 0.6B --output results/tune_qwen.json
```

Use `--quick` for faster iterations (fewer epochs / subset of data). Use `--grid` for full grid search on DeBERTa (small search space).

---

## Config

**Qwen** (`qwen_pipeline/config.py`): `data_path`, `llm_model`, `sft_epochs`, `sft_batch_size`, `max_seq_length`, `sft_max_grad_norm`, `sft_optim`, `sft_lr_scheduler_type`, `sft_early_stopping_patience`.

**DeBERTa** (`deberta_pipeline/config.py`): `TRAIN_PATH`, `TEST_PATH`, `MODEL_OPTIONS`, `NUM_EPOCHS`, `BATCH_SIZE`, `MAX_GRAD_NORM`, `GRADIENT_ACCUMULATION_STEPS`, `OPTIM`, `LR_SCHEDULER_TYPE`, `EARLY_STOPPING_PATIENCE`, `LABEL_SMOOTHING`.

---

## Troubleshooting

**Checkpoint not found**

- Use `--checkpoint` to point to the output directory.
- Run training first: `python -m qwen_pipeline.run_pipeline --mode train_only` or `python -m deberta_pipeline.run_pipeline --mode train`.

**Out of memory**

- Qwen: Lower `sft_batch_size` or `sft_grad_accum` in `qwen_pipeline/config.py`.
- DeBERTa: Lower `BATCH_SIZE` in `deberta_pipeline/config.py`.

**Module not found**

- Install: `pip install unsloth trl transformers datasets peft pandas openpyxl scikit-learn torch`.
