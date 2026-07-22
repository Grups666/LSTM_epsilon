param(
    [string]$Python = "D:\SSH\conda_envs\hydro\python.exe",
    [string]$Config = "paper_repo\configs\epsilon_experiment_pure_gcin_1950_2019.yaml",
    [string]$RunLabel = "crossfit_1990"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot
$environmentRoot = Split-Path -Parent $Python
$env:PATH = "$environmentRoot\Library\bin;$environmentRoot\Scripts;$env:PATH"

$runRoot = Join-Path $repoRoot "_private\results\epsilon_pure_gcin_1950_2019\$RunLabel"
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

for ($fold = 0; $fold -lt 5; $fold++) {
    $foldDir = Join-Path $runRoot "fold_$fold"
    $metadata = Join-Path $foldDir "run_metadata.json"
    $model = Join-Path $foldDir "best_model.pt"
    if ((Test-Path $metadata) -and (Test-Path $model)) {
        Write-Host "fold $fold already complete; skipping"
        continue
    }

    New-Item -ItemType Directory -Force -Path $foldDir | Out-Null
    $log = Join-Path $foldDir "training.log"
    & $Python "paper_repo\src\epsilon_model\train_epsilon_model.py" `
        --config $Config `
        --fold $fold `
        --run-label $RunLabel 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for fold $fold with exit code $LASTEXITCODE"
    }
}

Set-Content -Path (Join-Path $runRoot "training_complete.txt") -Value (Get-Date -Format o)
Write-Host "All crossfit folds completed."
