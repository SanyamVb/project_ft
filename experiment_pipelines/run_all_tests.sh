#!/bin/bash

# Run all test experiment scripts sequentially
# Each script uses minimal data (1 sample per class) and 2 epochs

# Create logs directory
mkdir -p experiment_pipelines/logs

echo "=========================================="
echo "Starting All Test Experiments"
echo "=========================================="
echo ""

# Test 1: add_generation_prompt=True, 20 epochs (2 epochs in test)
echo "Running Test 1: add_generation_prompt=True (20 epochs config)"
LOGFILE1="experiment_pipelines/logs/test_gen_prompt_true_20epochs_$(date +%Y%m%d_%H%M%S).log"
python experiment_pipelines/test_qwen3_gen_prompt_true_20epochs.py 2>&1 | tee "$LOGFILE1"
if [ $? -ne 0 ]; then
    echo "ERROR: Test 1 failed!"
    exit 1
fi
echo ""
echo "Test 1 completed successfully!"
echo ""
sleep 5

# Test 2: Learning rate comparison (2 LRs in test)
echo "Running Test 2: Learning Rate Comparison"
LOGFILE2="experiment_pipelines/logs/test_learning_rate_comparison_$(date +%Y%m%d_%H%M%S).log"
python experiment_pipelines/test_qwen3_learning_rate_comparison.py 2>&1 | tee "$LOGFILE2"
if [ $? -ne 0 ]; then
    echo "ERROR: Test 2 failed!"
    exit 1
fi
echo ""
echo "Test 2 completed successfully!"
echo ""
sleep 5

# Test 3: add_generation_prompt=True, 50 epochs (2 epochs in test)
echo "Running Test 3: add_generation_prompt=True (50 epochs config)"
LOGFILE3="experiment_pipelines/logs/test_gen_prompt_true_50epochs_$(date +%Y%m%d_%H%M%S).log"
python experiment_pipelines/test_qwen3_gen_prompt_true_50epochs.py 2>&1 | tee "$LOGFILE3"
if [ $? -ne 0 ]; then
    echo "ERROR: Test 3 failed!"
    exit 1
fi
echo ""
echo "Test 3 completed successfully!"
echo ""

echo "=========================================="
echo "All Test Experiments Completed Successfully!"
echo "Logs saved in experiment_pipelines/logs/"
echo "=========================================="
