# Run all test experiment scripts sequentially
# Each script uses minimal data (1 sample per class) and 2 epochs

# Create logs directory
New-Item -Path "experiment_pipelines/logs" -ItemType Directory -Force | Out-Null

Write-Host "=========================================="
Write-Host "Starting All Test Experiments"
Write-Host "=========================================="
Write-Host ""

# Test 1: add_generation_prompt=True, 20 epochs (2 epochs in test)
Write-Host "Running Test 1: add_generation_prompt=True (20 epochs config)"
$logFile1 = "experiment_pipelines/logs/test_gen_prompt_true_20epochs_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
python experiment_pipelines/test_qwen3_gen_prompt_true_20epochs.py 2>&1 | Tee-Object -FilePath $logFile1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Test 1 failed!" -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "Test 1 completed successfully!" -ForegroundColor Green
Write-Host ""
Start-Sleep -Seconds 5

# Clear GPU memory between tests
Write-Host "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; print('GPU memory cleared')"
Start-Sleep -Seconds 3

# Test 2: Learning rate comparison (2 LRs in test)
Write-Host "Running Test 2: Learning Rate Comparison"
$logFile2 = "experiment_pipelines/logs/test_learning_rate_comparison_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
python experiment_pipelines/test_qwen3_learning_rate_comparison.py 2>&1 | Tee-Object -FilePath $logFile2
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Test 2 failed!" -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "Test 2 completed successfully!" -ForegroundColor Green
Write-Host ""
Start-Sleep -Seconds 5

# Clear GPU memory between tests
Write-Host "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; print('GPU memory cleared')"
Start-Sleep -Seconds 3

# Test 3: add_generation_prompt=True, 50 epochs (2 epochs in test)
Write-Host "Running Test 3: add_generation_prompt=True (50 epochs config)"
$logFile3 = "experiment_pipelines/logs/test_gen_prompt_true_50epochs_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
python experiment_pipelines/test_qwen3_gen_prompt_true_50epochs.py 2>&1 | Tee-Object -FilePath $logFile3
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Test 3 failed!" -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "Test 3 completed successfully!" -ForegroundColor Green
Write-Host ""

Write-Host "=========================================="
Write-Host "All Test Experiments Completed Successfully!" -ForegroundColor Green
Write-Host "Logs saved in experiment_pipelines/logs/" -ForegroundColor Cyan
Write-Host "=========================================="
