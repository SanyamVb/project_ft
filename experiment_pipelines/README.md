# Experiment Pipelines

This folder contains separate pipeline files for running different training experiments with the Qwen3 cross-encoder model.

## Quick Start

### Run Test Scripts (Fast - Validates Everything Works)
```bash
# PowerShell (Windows)
.\experiment_pipelines\run_all_tests.ps1

# Bash (Linux/Mac/WSL)
bash experiment_pipelines/run_all_tests.sh
```

### Run Main Experiments (Long - Full Training)
```bash
# PowerShell (Windows)
.\experiment_pipelines\run_all_main_experiments.ps1

# Bash (Linux/Mac/WSL)
bash experiment_pipelines/run_all_main_experiments.sh
```

## Main Pipelines

### 1. `qwen3_gen_prompt_true_20epochs.py`
**Modification:** Set `add_generation_prompt=True` (instead of False)
**Training:** 20 epochs
**Output Directory:** `./qwen3_gen_prompt_true_20epochs`
**Model Saved To:** `models/qwen3_gen_prompt_true_20epochs`

This pipeline trains the model with generation prompt enabled for 20 epochs.

**Run:**
```bash
python experiment_pipelines/qwen3_gen_prompt_true_20epochs.py
```

**Run Individual Test:**
```bash
python experiment_pipelines/test_qwen3_gen_prompt_true_20epochs.py
```

---

### 2. `qwen3_learning_rate_comparison.py`
**Modification:** Compare 4 different learning rates
**Learning Rates:** 2e-6, 5e-5, 2e-5, 2e-4
**Training:** 20 epochs per learning rate (4 models total)
**Output Directories:** 
- `./qwen3_lr_2e-6_20epochs`
- `./qwen3_lr_5e-5_20epochs`
- `./qwen3_lr_2e-5_20epochs`
- `./qwen3_lr_2e-4_20epochs`

**Models Saved To:** 
- `models/qwen3_lr_2e-6_20epochs`
- `models/qwen3_lr_5e-5_20epochs`
- `models/qwen3_lr_2e-5_20epochs`
- `models/qwen3_lr_2e-4_20epochs`

**Comparison Summary:** `models/lr_comparison_summary/`

This pipeline sequentially trains 4 models with different learning rates and generates a comparison report.

**Run:**
```bash
python experiment_pipelines/qwen3_learning_rate_comparison.py
```

**Run Individual Test:**
```bash
python experiment_pipelines/test_qwen3_learning_rate_comparison.py
```

---

### 3. `qwen3_gen_prompt_true_50epochs.py`
**Modification:** Set `add_generation_prompt=True` (instead of False)
**Training:** 50 epochs
**Output Directory:** `./qwen3_gen_prompt_true_50epochs`
**Model Saved To:** `models/qwen3_gen_prompt_true_50epochs`

This pipeline trains the model with generation prompt enabled for 50 epochs.

**Run:**
```bash
python experiment_pipelines/qwen3_gen_prompt_true_50epochs.py
```

**Run Individual Test:**
```bash
python experiment_pipelines/test_qwen3_gen_prompt_true_50epochs.py
```

---

## Test Pipelines (Validate Everything Works)

### 1. `test_qwen3_gen_prompt_true_20epochs.py`
**Quick test version** of the 20-epoch pipeline
- Uses only 1 sample per class (2 total)
- Trains for 2 epochs
- Output: `models/test_qwen3_gen_prompt_true_20epochs`

### 2. `test_qwen3_learning_rate_comparison.py`
**Quick test version** of learning rate comparison
- Uses only 1 sample per class (2 total)
- Tests only 2 learning rates: 2e-5, 5e-5
- Trains for 2 epochs each
- Output: `models/test_qwen3_lr_*_20epochs`

### 3. `test_qwen3_gen_prompt_true_50epochs.py`
**Quick test version** of the 50-epoch pipeline
- Uses only 1 sample per class (2 total)
- Trains for 2 epochs
- Output: `models/test_qwen3_gen_prompt_true_50epochs`

---

## Shell Scripts

### `run_all_tests.sh` / `run_all_tests.ps1`
Runs all 3 test scripts sequentially with GPU memory clearing between each.
- Fast execution (~minutes)
- Validates all pipelines work correctly
- Use before running main experiments
- **Logs saved to**: `experiment_pipelines/logs/test_*.log`

### `run_all_main_experiments.sh` / `run_all_main_experiments.ps1`
Runs all 3 main experiment scripts sequentially with GPU memory clearing between each.
- Very long execution (~many hours)
- Trains all models with full data
- Includes timestamps for tracking progress
- **Logs saved to**: `experiment_pipelines/logs/*.log`

---

## Important Notes

1. **No Pipeline Overlap:** Each pipeline saves to unique directories, so no models will be overwritten.

2. **Training Order:** You can run these pipelines in any order or in parallel (if you have multiple GPUs).

3. **Model Outputs:**
   - Each pipeline saves the final model to a unique `models/` subdirectory
   - Training checkpoints are saved to working directories (e.g., `./qwen3_gen_prompt_true_20epochs/`)
   - Threshold optimization results and plots are saved with each model
   - Training history is saved as CSV files

4. **Memory Management:** 
   - The learning rate comparison pipeline (`qwen3_learning_rate_comparison.py`) clears GPU memory between each learning rate experiment
   - Consider running this pipeline when you have sufficient GPU memory available

5. **Results:**
   - Each model directory contains:
     - `final_metrics.json` - Complete metrics
     - `threshold_optimization.csv` - Threshold analysis
     - `threshold_optimization.png` - Visualization
     - `training_history.csv` - Epoch-by-epoch metrics

6. **Training Logs:**
   - All shell scripts automatically save logs to `experiment_pipelines/logs/`
   - Log files include timestamps in filename
   - Both stdout and stderr are captured
   - Logs remain even after scripts complete

## Expected Duration

- 20 epoch pipelines: Several hours (depending on GPU)
- 50 epoch pipeline: ~2.5x longer than 20 epoch pipeline
- Learning rate comparison: ~4x the time of a single 20 epoch run

## Requirements

All pipelines require the same dependencies as the main project:
- transformers
- torch
- peft
- datasets
- pandas
- numpy
- scikit-learn
- matplotlib
