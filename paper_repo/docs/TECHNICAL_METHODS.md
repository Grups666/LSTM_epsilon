# Technical Methods

## Active Experiment

This document records the single maintained pure-GCIN LSTM-epsilon workflow.

```text
run label: crossfit_1990
config:    paper_repo/configs/epsilon_experiment_pure_gcin_1950_2019.yaml
status:    complete; all five held-out folds passed the production audit
```

The scientific target is the change in inferred catchment recession behavior around 1990. This is an explanatory comparison, not a future-flow forecasting experiment. Time is therefore not divided into training, validation, and test blocks.

## Data Contract

The active dataset contains 2,511 model-ready GCIN catchments and daily records from 1950 through 2019.

```text
identifier:   GCIN
pre period:   1950-01-01 to 1990-12-31
post period:  1991-01-01 to 2019-12-31
record type:  catchment-day time series
```

The nominal period does not imply complete observations in every basin. Counting only finite daily Q values, the median basin has 16,132 valid days: 7,472 before the breakpoint and 9,204 after it. Of the 2,511 model-ready catchments, 2,304 have at least one valid Q day in both periods, 2,220 have at least two years in each period, and 2,133 have at least five years in each period.

Private inputs are restricted to:

```text
_private/processed/epsilon_training_daily_pure_gcin_1950_2019_parquet/
_private/processed/epsilon_physics_daily_pure_gcin_1950_2019_parquet/
_private/processed/epsilon_model_inputs_pure_gcin_1950_2019/
```

Observed streamflow and catchment identity use the original GCIN identifier. Meteorological forcing and land-state variables are catchment-reduced daily series aligned to the same GCIN records. Large daily files and checkpoints are not published.

## Cross-Fitting Design

Five outer folds partition catchments, not years. For each outer fold, one complete fold is the independent test set and half of the next fold is the validation set. The remaining catchments are used for training.

| Fold | Train | Validation | Test |
|---:|---:|---:|---:|
| 0 | 1,757 | 251 | 503 |
| 1 | 1,758 | 251 | 502 |
| 2 | 1,758 | 251 | 502 |
| 3 | 1,758 | 251 | 502 |
| 4 | 1,757 | 252 | 502 |

This is approximately a 70/10/20 catchment split in every fold. Each catchment appears in the test role exactly once. Training, validation, and test catchments all retain the full 1950-2019 record, so both climate periods are represented in every role.

Fold assignment is deterministic and stratified using latitude, aridity, area, precipitation, and temperature. Dynamic and static normalization statistics are estimated from training catchments only and then applied unchanged to validation and test catchments.

The test designation means that a catchment is excluded from neural-network parameter fitting and checkpoint selection. It is not a strict ungauged-basin experiment: observed Q in each held-out catchment is still required to identify recession days, define Q10/Q90 flow regimes, evaluate NSE, and provide the reference implementation's Q-derived `low_high_ratio` static attribute. Results must therefore be described as basin-held-out cross-fitting for a gauged-catchment attribution study.

## Recession Selection

Training and epsilon inference use recession sequences selected with the reference-compatible rules:

```text
minimum decline length: 4 days
drop first decline day: true
decreasing-rate filter: true
remove temperature_C <= 0 deg C: true
```

Flow regimes are defined separately within each catchment from observed recession-day streamflow:

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

The network receives a 365-day dynamic context plus static catchment attributes and predicts epsilon, reset flow, alpha, LP, and gamma. The reference state-reset integration and physical bounds are retained.

## Training

```text
maximum epochs:       30
batch size:           512
learning rate:        1e-4
hidden size:          256
LSTM layers:          1
dropout:              0.4
mixture components:   10
target sequence:      365 days
warm-up context:      365 days
validation batches:   8 deterministic batches per epoch
early-stop minimum:   15 epochs
early-stop patience:  5 epochs
NSE improvement:      0.001
```

The unchanged physics-informed optimization objective is:

```text
L = 25.0 * L_path
  + 10.0 * L_rhs
  +  0.1 * L_smooth
  +  5.0 * L_q0
```

`L_path` constrains the integrated recession path, `L_rhs` constrains the differential-equation tendency, `L_smooth` limits excessive epsilon curvature, and `L_q0` constrains recession resets.

Loss is used for gradient optimization. For interpretable validation, every epoch also computes NSE in original streamflow units on validation recession days. Window statistics are accumulated by catchment before scoring. The primary validation value is median catchment NSE; the checkpoint with the highest value is retained. Pooled validation NSE and the component losses are diagnostics.

The retained checkpoints were:

| Fold | Trained epochs | Best epoch | Best validation median NSE |
|---:|---:|---:|---:|
| 0 | 23 | 18 | 0.513 |
| 1 | 30 | 25 | 0.501 |
| 2 | 26 | 21 | 0.530 |
| 3 | 30 | 28 | 0.496 |
| 4 | 19 | 14 | 0.476 |

## Independent Evaluation

Each retained checkpoint is applied only to that fold's held-out test catchments. Test simulations cover the complete available 1950-2019 record. The primary skill report is median catchment NSE, calculated for:

```text
all years
1950-1990
1991-2019
```

Following the reference `test_main.py`, inference is run one catchment at a time over its complete available sequence. The model's static `alpha`, `LP`, and `gamma` terms are obtained from the final recurrent state of that full sequence and then used for the reconstructed recession path. This is a retrospective full-record parameter inference design for attribution, not a causal or future-only forecast.

KGE and pooled NSE/KGE are supplementary. Period-specific NSE is also exported for the public reliability filter. The browser retains all evaluated catchments and lets the user choose NSE or KGE and change the threshold; the default is NSE > 0.5 in both periods.

The final production audit covered all 2,511 held-out catchments and 9,192,715 recession-day simulations. No test GCIN was missing, duplicated, or assigned to the wrong fold, and all observed and simulated Q values were finite.

```text
all-period median catchment NSE:       0.327
all-period NSE p10-p90:               -1.045 to 0.651
all-period pooled NSE:                 0.343
all-period median catchment KGE:       0.494
all-period KGE p10-p90:               -0.168 to 0.770
all-period pooled KGE:                 0.498
pre-period median NSE / KGE:           0.302 / 0.478
post-period median NSE / KGE:          0.388 / 0.520
valid pre/post epsilon contrasts:      2,297
pre/post NSE > 0.5 in both periods:      566
pre/post KGE > 0.5 in both periods:    1,003
```

The substantial negative NSE tail and the gap between validation-window NSE and complete-record held-out NSE show that cross-catchment generalization is limited. Full-cohort epsilon shifts are therefore descriptive. The primary reliability subset contains the 566 catchments with NSE > 0.5 in both periods; its mean and median post-minus-pre epsilon changes are 0.01982 and 0.00977, and 21.6% of these catchments have negative change.

Epsilon has no direct daily observation. NSE evaluates the physics-constrained streamflow reconstruction, so even the reliability-filtered result is indirect evidence about a latent coefficient. It should be described as a change in model-inferred epsilon among catchments with adequate reconstruction skill, not as direct observation of true epsilon change. The fixed eight-batch validation diagnostic also samples less of the record than final complete-sequence testing; broader validation coverage is a priority for a future model-improvement experiment.

## Outputs

```text
_private/results/epsilon_pure_gcin_1950_2019/crossfit_1990/
  fold_<k>/
    best_model.pt
    metrics.csv
    run_metadata.json
    heldout_epsilon_change_summary.parquet
    recession_day_simulations.parquet
  analysis/
  production_audit.csv
```

Public outputs contain only derived figures, tables, documentation, and the browser JSON. Private daily data and model checkpoints remain under `_private/`.

## Reproduction

```powershell
conda run -n hydro python paper_repo\src\epsilon_model\prepare_experiment_inputs.py `
  --config paper_repo\configs\epsilon_experiment_pure_gcin_1950_2019.yaml

powershell -ExecutionPolicy Bypass -File paper_repo\scripts\run_crossfit_training.ps1 `
  -RunLabel crossfit_1990

conda run -n hydro python paper_repo\src\epsilon_model\run_full_postprocess.py `
  --config paper_repo\configs\epsilon_experiment_pure_gcin_1950_2019.yaml `
  --run-label crossfit_1990 `
  --figures-dir _private\results\paper_figures_crossfit_1990
```
