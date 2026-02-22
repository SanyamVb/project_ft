# SFT Training Comparison Guide

## Overview

This guide helps you run automated comparison experiments between **Completion-Only** and **Normal SFT** training across different epoch counts (1, 2, and 3 epochs).

## Files Created

1. **`compare_sft_variants.py`** - Main automation script that runs all 6 experiments
2. **`analyze_comparison.py`** - Analysis and visualization of results
3. **`SFT_COMPARISON_GUIDE.md`** - This guide

## Quick Start

### Run All 6 Experiments (Recommended)

```powershell
python compare_sft_variants.py
```

This will run:
- ✓ Completion-Only: 1, 2, 3 epochs
- ✓ Normal SFT: 1, 2, 3 epochs

### Run Only Completion-Only Experiments

```powershell
python compare_sft_variants.py --skip_normal
```

### Run Only Normal SFT Experiments

```powershell
python compare_sft_variants.py --skip_completion
```

### Use Custom Data Path

```powershell
python compare_sft_variants.py --data_path "path/to/your/data.xlsx"
```

## What Each Experiment Does

### Completion-Only Training
- Trains only on the assistant's response tokens
- Masks the prompt/instruction tokens during training
- Faster training, potentially better for instruction-following

### Normal SFT (Full Fine-tune)
- Trains on the entire sequence (prompt + response)
- No masking applied
- May learn prompt patterns better

## Output Structure

After running, you'll get a directory like `comparison_results_20260222_143000/` containing:

```
comparison_results_20260222_143000/
├── comparison_report.json      # Detailed JSON results
├── comparison_report.md        # Human-readable markdown report
└── comparison_results.csv      # CSV export (if requested)
```

Each experiment also creates its own output directory:
```
outputs_sft_4B_completion_only_epoch1/
outputs_sft_4B_completion_only_epoch2/
outputs_sft_4B_completion_only_epoch3/
outputs_sft_4B_full_finetune_epoch1/
outputs_sft_4B_full_finetune_epoch2/
outputs_sft_4B_full_finetune_epoch3/
```

## Analyzing Results

### View Analysis of Latest Results

```powershell
python analyze_comparison.py
```

### Analyze Specific Results Directory

```powershell
python analyze_comparison.py --report_dir comparison_results_20260222_143000
```

### Export to CSV

```powershell
python analyze_comparison.py --export_csv
```

## Understanding the Metrics

- **Accuracy**: Overall classification accuracy
- **Kappa**: Cohen's Kappa (agreement metric, accounts for chance)
- **Mean Score**: Custom scoring (2 points for correct, -1 for wrong)
- **Train Time**: Total training time in seconds
- **Inference Time**: Average time per sample in milliseconds

## Expected Timeline

Each experiment typically takes:
- 1 epoch: ~X minutes
- 2 epochs: ~Y minutes  
- 3 epochs: ~Z minutes

**Total for all 6 experiments**: Approximately N hours

(Times vary based on hardware and dataset size)

## Troubleshooting

### Out of Memory Error
- Reduce `sft_batch_size` in `qwen_pipeline/config.py`
- Reduce `sft_grad_accum` in `qwen_pipeline/config.py`

### Experiment Fails Midway
- Check the error message in console
- Individual experiment results are still saved
- You can analyze partial results with `analyze_comparison.py`

### Resume Failed Experiment
The script saves each experiment independently, so you can:
1. Note which experiments completed
2. Run only the missing ones using `--skip_completion` or `--skip_normal`
3. Manually combine results if needed

## Tips for Best Results

1. **Monitor GPU usage**: Watch `nvidia-smi` to ensure optimal GPU utilization
2. **Check first experiment**: Let the first experiment complete before leaving it unattended
3. **Save logs**: Consider redirecting output to a log file:
   ```powershell
   python compare_sft_variants.py *>&1 | Tee-Object comparison.log
   ```
4. **Review before running**: Ensure you have enough disk space for 6 model checkpoints

## Next Steps After Completion

1. Run `python analyze_comparison.py` to see detailed analysis
2. Review the markdown report for easy comparison
3. Identify the best performing configuration
4. Use that configuration's checkpoint for production/further experiments

## Example Output

```
SUMMARY TABLE
================================================================================
Experiment                     Epochs   Accuracy     Kappa        Mean Score   Train Time (s)
--------------------------------------------------------------------------------
Completion-Only 1 Epoch        1        0.8523      0.7045          1.23           245.12
Completion-Only 2 Epochs       2        0.8712      0.7424          1.45           489.34
Completion-Only 3 Epochs       3        0.8834      0.7668          1.58           734.56
Normal SFT 1 Epoch            1        0.8456      0.6911          1.15           267.89
Normal SFT 2 Epochs           2        0.8689      0.7378          1.42           535.23
Normal SFT 3 Epochs           3        0.8801      0.7602          1.54           802.67
================================================================================
```

## Advanced Usage

### Modify Training Parameters

Edit `qwen_pipeline/config.py` to adjust:
- Learning rate (`sft_lr`)
- Batch size (`sft_batch_size`)
- Max sequence length (`max_position_embeddings`)
- LoRA rank (`lora_rank`)

### Add More Epoch Variations

Edit `compare_sft_variants.py` and add to the `experiments` list:
```python
experiments.extend([
    ("completion_only", 4, "Completion-Only 4 Epochs"),
    ("full_finetune", 4, "Normal SFT 4 Epochs"),
])
```

## Support

For issues or questions:
1. Check error messages carefully
2. Review the configuration in `qwen_pipeline/config.py`
3. Ensure all dependencies are installed
4. Check GPU memory availability
