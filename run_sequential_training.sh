#!/usr/bin/env bash
set -euo pipefail

# Sequential training script
# Runs 10-epoch training followed by 20-epoch training

echo "========================================"
echo "Starting Sequential Training Pipeline"
echo "========================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create logs directory if it doesn't exist
mkdir -p logs

# Training 1: 10 epochs
echo "========================================="
echo "Step 1/2: Training with 10 epochs"
echo "Started at: $(date)"
echo "========================================="
python qwen3_cross_encoder_offtopic.py 2>&1 | tee logs/training_10epochs_$(date +%Y%m%d_%H%M%S).log
echo ""
echo "10-epoch training completed at: $(date)"
echo ""

# Training 2: 20 epochs
echo "========================================="
echo "Step 2/2: Training with 20 epochs"
echo "Started at: $(date)"
echo "========================================="
python qwen3_cross_encoder_offtopic_20.py 2>&1 | tee logs/training_20epochs_$(date +%Y%m%d_%H%M%S).log
echo ""
echo "20-epoch training completed at: $(date)"
echo ""

echo "========================================"
echo "All Training Completed!"
echo "Finished at: $(date)"
echo "========================================"
