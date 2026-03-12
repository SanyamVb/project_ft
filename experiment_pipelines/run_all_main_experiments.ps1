# Run all main experiment scripts sequentially
# WARNING: This will take a very long time (many hours)

# Create logs directory
New-Item -Path "experiment_pipelines/logs" -ItemType Directory -Force | Out-Null

Write-Host "=========================================="
Write-Host "Starting All Main Experiments"
Write-Host "=========================================="
Write-Host ""

# Experiment 1: add_generation_prompt=True, 20 epochs
Write-Host "Running Experiment 1: add_generation_prompt=True, 20 epochs"
Write-Host "Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$logFile1 = "experiment_pipelines/logs/gen_prompt_true_20epochs_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
python experiment_pipelines/qwen3_gen_prompt_true_20epochs.py 2>&1 | Tee-Object -FilePath $logFile1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Experiment 1 failed!" -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "Experiment 1 completed successfully at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host ""
Start-Sleep -Seconds 10

# Clear GPU memory between experiments
Write-Host "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; print('GPU memory cleared')"
Start-Sleep -Seconds 5

# Experiment 2: Learning rate comparison (4 models, 20 epochs each)
Write-Host "Running Experiment 2: Learning Rate Comparison"
Write-Host "WARNING: This trains 4 separate models!" -ForegroundColor Yellow
Write-Host "Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$logFile2 = "experiment_pipelines/logs/learning_rate_comparison_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
python experiment_pipelines/qwen3_learning_rate_comparison.py 2>&1 | Tee-Object -FilePath $logFile2
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Experiment 2 failed!" -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "Experiment 2 completed successfully at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host ""
Start-Sleep -Seconds 10

# Clear GPU memory between experiments
Write-Host "Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None; print('GPU memory cleared')"
Start-Sleep -Seconds 5

# Experiment 3: add_generation_prompt=True, 50 epochs
Write-Host "Running Experiment 3: add_generation_prompt=True, 50 epochs"
Write-Host "WARNING: This is the longest experiment (50 epochs)!" -ForegroundColor Yellow
Write-Host "Started at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$logFile3 = "experiment_pipelines/logs/gen_prompt_true_50epochs_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
python experiment_pipelines/qwen3_gen_prompt_true_50epochs.py 2>&1 | Tee-Object -FilePath $logFile3
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Experiment 3 failed!" -ForegroundColor Red
    exit 1
}
Write-Host ""
Write-Host "Experiment 3 completed successfully at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host ""

Write-Host "=========================================="
Write-Host "All Main Experiments Completed Successfully!" -ForegroundColor Green
Write-Host "Finished at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "Logs saved in experiment_pipelines/logs/" -ForegroundColor Cyan
Write-Host "=========================================="
