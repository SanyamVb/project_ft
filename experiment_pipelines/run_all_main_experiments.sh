#!/bin/bash

# Run all main experiment scripts sequentially
# WARNING: This will take a very long time (many hours)

# Create logs directory
mkdir -p experiment_pipelines/logs

echo "=========================================="
echo "Starting All Main Experiments"
echo "WARNING: This will take many hours!"
echo "=========================================="
echo ""

# Experiment 1: add_generation_prompt=True, 20 epochs
echo "Running Experiment 1: add_generation_prompt=True, 20 epochs"
echo "Started at: $(date)"
LOGFILE1="experiment_pipelines/logs/gen_prompt_true_20epochs_$(date +%Y%m%d_%H%M%S).log"
python experiment_pipelines/qwen3_gen_prompt_true_20epochs.py 2>&1 | tee "$LOGFILE1"
if [ $? -ne 0 ]; then
    echo "ERROR: Experiment 1 failed!"
    exit 1
fi
echo ""
echo "Experiment 1 completed successfully at: $(date)"
echo ""
sleep 10

# Experiment 2: Learning rate comparison (4 models, 20 epochs each)
echo "Running Experiment 2: Learning Rate Comparison"
echo "WARNING: This trains 4 separate models!"
echo "Started at: $(date)"
LOGFILE2="experiment_pipelines/logs/learning_rate_comparison_$(date +%Y%m%d_%H%M%S).log"
python experiment_pipelines/qwen3_learning_rate_comparison.py 2>&1 | tee "$LOGFILE2"
if [ $? -ne 0 ]; then
    echo "ERROR: Experiment 2 failed!"
    exit 1
fi
echo ""
echo "Experiment 2 completed successfully at: $(date)"
echo ""
sleep 10

# Experiment 3: add_generation_prompt=True, 50 epochs
echo "Running Experiment 3: add_generation_prompt=True, 50 epochs"
echo "WARNING: This is the longest experiment (50 epochs)!"
echo "Started at: $(date)"
LOGFILE3="experiment_pipelines/logs/gen_prompt_true_50epochs_$(date +%Y%m%d_%H%M%S).log"
python experiment_pipelines/qwen3_gen_prompt_true_50epochs.py 2>&1 | tee "$LOGFILE3"
if [ $? -ne 0 ]; then
    echo "ERROR: Experiment 3 failed!"
    exit 1
fi
echo ""
echo "Experiment 3 completed successfully at: $(date)"
echo ""

echo "=========================================="
echo "All Main Experiments Completed Successfully!"
echo "Finished at: $(date)"
echo "Logs saved in experiment_pipelines/logs/"
echo "=========================================="
