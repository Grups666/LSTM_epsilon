param(
    [string]$Python = "D:\SSH\conda_envs\hydro\python.exe",
    [string]$Config = "paper_repo\configs\epsilon_experiment_pure_gcin_1950_2019.yaml",
    [string]$RunLabel = "temporal_crossfit_1990"
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
    $model = Join-Path $foldDir "final_model.pt"
    if ((Test-Path $metadata) -and (Test-Path $model)) {
        Write-Host "fold $fold already complete; skipping"
    } else {
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

    $summary = Join-Path $foldDir "heldout_epsilon_change_summary.parquet"
    $simulation = Join-Path $foldDir "recession_day_simulations.parquet"
    $skill = Join-Path $foldDir "heldout_skill_summary.csv"
    if (-not ((Test-Path $summary) -and (Test-Path $simulation) -and (Test-Path $skill))) {
        Write-Host "Running held-out inference for fold $fold"
        & $Python "paper_repo\src\epsilon_model\infer_epsilon_change_summary.py" `
            --config $Config `
            --fold $fold `
            --run-label $RunLabel
        if ($LASTEXITCODE -ne 0) {
            throw "Held-out inference failed for fold $fold with exit code $LASTEXITCODE"
        }
    }

    Import-Csv -LiteralPath $skill | ForEach-Object {
        Write-Host ("fold {0} held-out {1} NSE: median={2:N3}, mean={3:N3}, pooled={4:N3}, basins={5}" -f `
            $fold, $_.period, [double]$_.median_catchment_nse, [double]$_.mean_catchment_nse, `
            [double]$_.pooled_nse, $_.n_catchments)
    }
}

Set-Content -Path (Join-Path $runRoot "training_complete.txt") -Value (Get-Date -Format o)
Write-Host "All crossfit folds completed."
