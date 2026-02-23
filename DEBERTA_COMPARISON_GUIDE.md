# DeBERTa Model Comparison Guide

## Overview

This guide helps you run automated comparison experiments between **DeBERTa-Small**, **DeBERTa-Base**, and **DeBERTa-Large** models across different epoch counts (1, 2, and 3 epochs).

## Files Created

1. **`compare_deberta_variants.py`** - Main automation script that runs all 9 experiments (3 models × 3 epochs)
2. **`analyze_deberta_comparison.py`** - Analysis and visualization of results
3. **`DEBERTA_COMPARISON_GUIDE.md`** - This guide

## Quick Start

### Run All 9 Experiments (Recommended)

```powershell
python compare_deberta_variants.py
```

This will run:
- ✓ DeBERTa-Small: 1, 2, 3 epochs
- ✓ DeBERTa-Base: 1, 2, 3 epochs
- ✓ DeBERTa-Large: 1, 2, 3 epochs

### Run Only Specific Model Sizes

```powershell
# Only small model
python compare_deberta_variants.py --skip_base --skip_large

# Only base model
python compare_deberta_variants.py --skip_small --skip_large

# Only large model
python compare_deberta_variants.py --skip_small --skip_base

# Skip large model (run small and base only)
python compare_deberta_variants.py --skip_large
```

### Use Custom Data Paths

```powershell
python compare_deberta_variants.py --train_path "path/to/train.xlsx" --test_path "path/to/test.xlsx"
```

### Evaluation Only Mode

If you've already trained the models and just want to re-run evaluation:

```powershell
python compare_deberta_variants.py --eval_only
```

## What Each Experiment Does

### Model Sizes
- **DeBERTa-Small (~40M params)**: Fast training, good baseline performance
- **DeBERTa-Base (~86M params)**: Balanced performance and speed
- **DeBERTa-Large (~304M params)**: Best performance, slower training

### Training Epochs
Each model is trained for 1, 2, and 3 epochs to evaluate:
- Performance improvement vs training time
- Optimal stopping point for each model size
- Diminishing returns with additional epochs

## Output Structure

After running, you'll get a directory like `deberta_comparison_results_20260223_143000/` containing:

```
deberta_comparison_results_20260223_143000/
├── comparison_report.json      # Detailed JSON results
├── comparison_report.md        # Human-readable markdown report
└── comparison_results.csv      # CSV export (if requested)
```

Each experiment also creates its own output directory:
```
outputs_deberta_small_epoch1/
outputs_deberta_small_epoch2/
outputs_deberta_small_epoch3/
outputs_deberta_base_epoch1/
outputs_deberta_base_epoch2/
outputs_deberta_base_epoch3/
outputs_deberta_large_epoch1/
outputs_deberta_large_epoch2/
outputs_deberta_large_epoch3/
```

## Analyzing Results

### View Analysis of Latest Results

```powershell
python analyze_deberta_comparison.py
```

This will show:
- Performance summary for each model size
- Best performers by metric
- Impact of adding more epochs
- Model size comparison at same epochs
- Efficiency analysis (performance per training minute)

### Analyze Specific Results Directory

```powershell
python analyze_deberta_comparison.py --report_dir deberta_comparison_results_20260223_143000
```

### Export to CSV

```powershell
python analyze_deberta_comparison.py --export_csv
```

## Understanding the Metrics

- **Accuracy**: Overall classification accuracy
- **Kappa**: Cohen's Kappa (agreement metric, accounts for chance)
- **Mean Score**: Custom scoring (2 points for correct, -1 for wrong)
- **Best Threshold**: Optimal probability threshold found by kappa sweep
- **Train Time**: Total training time in seconds
- **Inference Time**: Average time per sample in milliseconds
- **Params (M)**: Model parameters in millions

## Expected Timeline

Approximate times (varies by hardware):

| Model Size | 1 Epoch | 2 Epochs | 3 Epochs |
|------------|---------|----------|----------|
| Small      | ~5 min  | ~10 min  | ~15 min  |
| Base       | ~8 min  | ~16 min  | ~24 min  |
| Large      | ~15 min | ~30 min  | ~45 min  |

**Total for all 9 experiments**: Approximately 2-3 hours

## Key Improvements vs Previous Version

### 1. System Prompt Integration
The DeBERTa pipeline now uses the **same system prompt** as the Qwen pipeline, ensuring consistent evaluation criteria across both model families. The system prompt provides:
- Clear guidelines for off-topic detection
- Proficiency level considerations (Advanced vs Intermediate)
- Context-aware evaluation instructions

### 2. Unified Input Format
The prompt format for DeBERTa now matches Qwen's structure:
```
[SYSTEM_PROMPT]

You are now given the below data:
Prompt Topic: [topic]
Prompt Sub-Topic: [sub_topic]
Prompt Script: [script]
Test-taker's response: [transcription]
Testing Proficiency Level: [level]

Based on your expertise, determine if the test-taker's response is off-topic or not.
```

## Troubleshooting

### Out of Memory Error
- Try skipping the large model: `--skip_large`
- Reduce batch size in `deberta_pipeline/config.py` (BATCH_SIZE_PER_MODEL)
- Use gradient accumulation for larger effective batch sizes

### Experiment Fails Midway
- Check the error message in console
- Individual experiment results are still saved
- You can analyze partial results with `analyze_deberta_comparison.py`
- Resume with specific model sizes using `--skip_*` flags

### CUDA Out of Memory on Large Model
The large model requires significant GPU memory. If you encounter OOM errors:
1. Ensure no other processes are using GPU
2. The batch size is already reduced for large model (8 by default)
3. Consider running large model separately with `--skip_small --skip_base`

## Tips for Best Results

1. **Monitor GPU usage**: Watch `nvidia-smi` to ensure optimal GPU utilization
2. **Start with small model**: Let one experiment complete to verify setup
3. **Save logs**: Consider redirecting output to a log file:
   ```powershell
   python compare_deberta_variants.py *>&1 | Tee-Object deberta_comparison.log
   ```
4. **Check disk space**: Ensure you have enough space for 9 model checkpoints (~5GB total)

## Comparison with Qwen Pipeline

| Aspect | Qwen SFT | DeBERTa Cross-Encoder |
|--------|----------|----------------------|
| **Architecture** | Generative (causal LM) | Discriminative (sequence classification) |
| **Training** | SFT with LoRA | Full fine-tuning with early stopping |
| **Comparison Focus** | Training variant (completion-only vs full) | Model size (small vs base vs large) |
| **Output** | JSON response with confidence | Binary classification with probability |
| **Use Case** | Flexible, instruction-following | Fast, specialized classification |

## Next Steps After Completion

1. Run `python analyze_deberta_comparison.py` to see detailed analysis
2. Review the markdown report for easy comparison
3. Identify the best performing model size and epoch count
4. Compare with Qwen results to choose the best model family for your use case
5. Use the best checkpoint for production deployment

## Example Output

```
SUMMARY TABLE
====================================================================================================
Experiment                     Epochs   Params (M)   Accuracy     Kappa        Mean Score   Train Time (s)
----------------------------------------------------------------------------------------------------
DeBERTa-Small 1 Epoch          1        40.6         0.8823      0.7646          1.45           289.12
DeBERTa-Small 2 Epochs         2        40.6         0.8945      0.7890          1.58           578.34
DeBERTa-Small 3 Epochs         3        40.6         0.9012      0.8023          1.64           867.56
DeBERTa-Base 1 Epoch           1        86.1         0.8934      0.7868          1.56           445.89
DeBERTa-Base 2 Epochs          2        86.1         0.9067      0.8134          1.71           891.23
DeBERTa-Base 3 Epochs          3        86.1         0.9123      0.8245          1.78          1336.67
DeBERTa-Large 1 Epoch          1        304.3        0.9045      0.8090          1.69           823.45
DeBERTa-Large 2 Epochs         2        304.3        0.9156      0.8312          1.83          1646.23
DeBERTa-Large 3 Epochs         3        304.3        0.9201      0.8401          1.88          2469.67
====================================================================================================
```

## Advanced Usage

### Modify Training Parameters

Edit `deberta_pipeline/config.py` to adjust:
- Learning rate (`LEARNING_RATE`)
- Batch sizes (`BATCH_SIZE_PER_MODEL`)
- Max sequence length (`MAX_LENGTH`)
- Warmup ratio (`WARMUP_RATIO`)
- Early stopping patience (`EARLY_STOPPING_PATIENCE`)

### Add More Epoch Variations

Edit `compare_deberta_variants.py` and add to the `experiments` list:
```python
experiments.extend([
    ("small", 4, "DeBERTa-Small 4 Epochs"),
    ("base", 4, "DeBERTa-Base 4 Epochs"),
    ("large", 4, "DeBERTa-Large 4 Epochs"),
])
```

### Compare with XSmall Model

The config includes an xsmall variant. To test it:
```python
# In compare_deberta_variants.py, add:
if not args.skip_xsmall:
    experiments.extend([
        ("xsmall", 1, "DeBERTa-XSmall 1 Epoch"),
        ("xsmall", 2, "DeBERTa-XSmall 2 Epochs"),
        ("xsmall", 3, "DeBERTa-XSmall 3 Epochs"),
    ])
```

## Support

For issues or questions:
1. Check error messages carefully
2. Review the configuration in `deberta_pipeline/config.py`
3. Ensure all dependencies are installed
4. Check GPU memory availability with `nvidia-smi`
5. Compare with the working Qwen pipeline setup

## Summary

This automated comparison framework helps you:
- ✓ Compare DeBERTa model sizes systematically
- ✓ Find the optimal epoch count for each model
- ✓ Understand performance vs efficiency trade-offs
- ✓ Make data-driven decisions for production deployment
- ✓ Ensure consistency with Qwen pipeline through unified prompts
