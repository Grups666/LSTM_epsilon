# Technical Methods

## Active Experiment

The maintained production workflow is the pure-GCIN paired temporal cross-fit:

```text
run label: temporal_crossfit_1990
config:    paper_repo/configs/epsilon_experiment_pure_gcin_1950_2019.yaml
status:    completed and audited
```

The scientific target is the within-catchment change in inferred recession behavior around 1990. The split therefore tests unseen time blocks in the same gauged catchments. It is not an ungauged-basin regionalization experiment.

## Data Contract

The model-ready dataset contains 2,511 GCIN catchments and daily records from 1950 through 2019.

```text
identifier:   GCIN
pre period:   1950-01-01 to 1990-12-31
post period:  1991-01-01 to 2019-12-31
record type:  catchment-day time series
```

Private inputs are restricted to:

```text
_private/processed/epsilon_training_daily_pure_gcin_1950_2019_parquet/
_private/processed/epsilon_physics_daily_pure_gcin_1950_2019_parquet/
_private/processed/epsilon_model_inputs_pure_gcin_1950_2019/
```

Observed streamflow and boundary identity use the original GCIN identifier. Private daily data and checkpoints are not published.

## Temporal Cross-Fitting

The pre and post periods are independently divided into five contiguous calendar blocks. Fold `k` holds out pre block `k` and post block `k` together.

| Fold | Pre test block | Post test block |
|---:|:---|:---|
| 0 | 1950-1958 | 1991-1996 |
| 1 | 1959-1966 | 1997-2002 |
| 2 | 1967-1974 | 2003-2008 |
| 3 | 1975-1982 | 2009-2014 |
| 4 | 1983-1990 | 2015-2019 |

Every fold uses all 2,511 catchments. Approximately 80% of calendar years train the model and 20% are tested. There is no validation set.

Each training example contains 365 warm-up days followed by a 365-day target sequence. A training example is eligible only if its complete 730-day span belongs to training years. Consequently, no held-out test date can appear in a training target or in a training example's model-input context.

Dynamic normalization statistics are estimated from training years only. The reference workflow's Q-derived `low_high_ratio` static attribute is recomputed from each fold's training years before static normalization. Test-period Q therefore cannot enter model fitting through that attribute.

## Recession Selection

Training and inference retain the reference-compatible recession rules:

```text
minimum decline length: 4 days
drop first decline day: true
decreasing-rate filter: true
remove temperature_C <= 0 deg C: true
```

Flow regimes are calculated within each catchment from observed out-of-fold recession-day streamflow:

```text
low flow:  Qobs <= catchment Q10
mid flow:  catchment Q10 < Qobs < catchment Q90
high flow: Qobs >= catchment Q90
```

## Model

The model follows the reference physics-informed LSTM-epsilon formulation and directly infers daily `epsilon` inside:

```text
dQ/dt = -epsilon * Q^2 - epsilon * alpha * AET * Q
```

The network receives dynamic forcing and state variables plus static catchment attributes. It predicts epsilon, reset flow, alpha, LP, and gamma. The reference state-reset integration, physical bounds, and four-term optimization objective are retained.

```text
L = 25.0 * L_path
  + 10.0 * L_rhs
  +  0.1 * L_smooth
  +  5.0 * L_q0
```

## Training

```text
epochs:              30 fixed
batch size:          512
learning rate:       1e-4
hidden size:         256
LSTM layers:         1
dropout:             0.4
mixture components:  10
target sequence:     365 days
warm-up context:     365 days
checkpoint:          final epoch
```

There is no early stopping or test-guided checkpoint selection. The epoch count and all architecture and loss settings are fixed before test inference. Reducing the reference maximum from 150 to 30 epochs is a fixed computational setting, not a fold-specific choice.

Each completed fold immediately reports held-out catchment NSE for progress monitoring. These reports are evaluation-only and are never used to change the fixed epoch count, model, or hyperparameters.

## Out-of-Fold Inference

Inference uses the same 365-day context plus at most 365 target days used in training. Each held-out pre/post block is tiled into non-overlapping target windows; every eligible recession day is emitted once by the model that did not train on its temporal fold.

The five fold outputs are concatenated by `GCIN` and date. For every catchment, pre-period and post-period NSE are calculated once from the concatenated out-of-fold streamflow predictions. NSE is not calculated as the arithmetic mean of five fold NSE values because each block has a different observed-flow variance. Fold-level NSE mean and standard deviation are retained as supplementary stability diagnostics.

NSE and KGE assess the physics-constrained streamflow reconstruction. They are indirect reliability diagnostics for latent epsilon, not observations of epsilon itself. The scientific contrast is:

```text
delta epsilon = mean out-of-fold epsilon in 1991-2019
              - mean out-of-fold epsilon in 1950-1990
```

The public explorer retains all catchments with valid contrasts and applies the user-selected NSE or KGE threshold in the browser. Its default reliability rule is that both pre-period and post-period NSE exceed 0.5.

## Audited Results

```text
OOF recession days:       9,192,715
catchments evaluated:     2,511
valid epsilon contrasts:  2,297
median catchment NSE:     0.581 (p10-p90: 0.128 to 0.775)
pooled NSE:               0.577
median catchment KGE:     0.642
pre median NSE / KGE:     0.555 / 0.622
post median NSE / KGE:    0.626 / 0.666
both-period NSE > 0.5:    1,304
both-period KGE > 0.5:    1,447
```

Among the 1,304 catchments passing NSE 0.5 in both periods, mean delta epsilon is -0.0192 (catchment-bootstrap 95% CI -0.0348 to -0.00521), while median delta epsilon is +0.00230 (95% CI +0.00151 to +0.00324). The opposite signs show a strongly skewed change distribution: negative outliers move the mean, whereas the typical catchment has a small positive shift. Both summaries must be reported.

## Outputs

```text
_private/results/epsilon_pure_gcin_1950_2019/temporal_crossfit_1990/
  fold_<k>/
    final_model.pt
    metrics.csv
    run_metadata.json
    heldout_epsilon_change_summary.parquet
    heldout_skill_summary.csv
    recession_day_simulations.parquet
  analysis/
    temporal_crossfit_epsilon_change_summary.parquet
    temporal_fold_catchment_skill.csv
    epsilon_change_inference.csv
  production_audit.csv
```

## Reproduction

```powershell
conda run -n hydro python paper_repo\src\epsilon_model\prepare_experiment_inputs.py `
  --config paper_repo\configs\epsilon_experiment_pure_gcin_1950_2019.yaml

powershell -ExecutionPolicy Bypass -File paper_repo\scripts\run_crossfit_training.ps1 `
  -RunLabel temporal_crossfit_1990

conda run -n hydro python paper_repo\src\epsilon_model\run_full_postprocess.py `
  --config paper_repo\configs\epsilon_experiment_pure_gcin_1950_2019.yaml `
  --run-label temporal_crossfit_1990 `
  --figures-dir paper_repo\docs\assets\epsilon_pure_gcin_1950_2019
```
