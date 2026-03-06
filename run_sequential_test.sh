#!/usr/bin/env bash
set -euo pipefail

# Sequential TEST training script
# Runs 2-epoch training on small dataset, then continues for 2 more epochs (total 4)

echo "========================================"
echo "Starting Sequential TEST Pipeline"
echo "Using 2 samples per class (4 total)"
echo "========================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Training 1: 2 epochs (TEST)
echo "========================================="
echo "Step 1/2: Training with 2 epochs (TEST)"
echo "Started at: $(date)"
echo "========================================="
python test_qwen_2epochs.py 2>&1 | tee logs/test_training_2epochs_$(date +%Y%m%d_%H%M%S).log
echo ""
echo "2-epoch TEST training completed at: $(date)"
echo ""

# Clear GPU memory before next training
echo "========================================="
echo "Clearing GPU memory..."
echo "========================================="
python -c "
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    print(f'GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('No GPU detected, skipping cache clear')
"
echo ""
sleep 2
echo ""

# Training 2: Resume from 2 epochs, train to 4 total (TEST)
echo "========================================="
echo "Step 2/2: Resuming from 2-epoch checkpoint"
echo "Training to 4 total epochs (TEST)"
echo "Started at: $(date)"
echo "========================================="
python test_qwen_4epochs.py 2>&1 | tee logs/test_training_4epochs_$(date +%Y%m%d_%H%M%S).log
echo ""
echo "4-epoch TEST training completed at: $(date)"
echo ""

echo "========================================"
echo "All TEST Training Completed!"
echo "Finished at: $(date)"
echo "========================================"
echo ""
echo "Summary:"
echo "- 2-epoch model: ./models/test_qwen3_cross_encoder_sft_2epochs"
echo "- 4-epoch model: ./models/test_qwen3_cross_encoder_sft_4epochs"
echo "- Training logs: ./logs/"
