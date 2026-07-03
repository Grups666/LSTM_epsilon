# Technical Methods

## Current Status

This document describes the active pure GCIN production experiment. The previous legacy/GridCode/Catchment_ID mixed experiment is not valid for production analysis because identifier and boundary provenance were not reliably matched.

The active remote run is:

```text
run label: full_pure_gcin_1950_2019
config:    paper_repo/configs/epsilon_experiment_pure_gcin_1950_2019.yaml
status:    five-fold training, inference aggregation, and production audit complete
```

The production audit, figure CSV tables, PNG figures, and GitHub Pages JSON export are complete for the pure GCIN run.

## Data

The current production input is the pure GCIN dataset. It combines GCIN-indexed observed streamflow with ERA5-Land catchment-reduced meteorological and land-state variables.

The model-ready files are private and are not part of the public repository:

```text
train-ready yearly daily series:
_private/processed/epsilon_training_daily_pure_gcin_1950_2019_parquet/

model daily input:
_private/processed/epsilon_training_daily_pure_gcin_1950_2019_parquet/

physics daily input:
_private/processed/epsilon_physics_daily_pure_gcin_1950_2019_parquet/

static attributes and AET parameter priors:
_private/processed/epsilon_model_inputs_pure_gcin_1950_2019/static_attributes.parquet
_private/processed/epsilon_model_inputs_pure_gcin_1950_2019/lp_gamma_fit_summary.parquet
```

The final train-ready daily product contains:

```text
years:      1950-2019
catchments: 2,511
recession simulation rows: 9,192,715
```

Each physics daily row is a catchment-day record, not a raster cell. Core fields are:

```text
GCIN, date, precipitation_mmd, temperature_C, pet_mmd,
SM_%, streamflow_mmd, observed_AET_mm
```

Important identifier rule:

```text
GCIN in the current model files is the original GCIN catchment identifier.
It must not be joined by numeric equality to GridCode or legacy force-code
products.
```

## Periods

The climate contrast is fixed as:

```text
pre-change:  1950-01-01 to 1990-12-31
post-change: 1991-01-01 to 2019-12-31
```

For flow-regime analyses, thresholds are computed separately for each catchment from its own observed recession-day flow distribution:

```text
low flow:  Qobs <= catchment Q10
mid flow:  catchment Q10 < Qobs < catchment Q90
high flow: Qobs >= catchment Q90
```

## Recession Filtering

Training and epsilon inference are restricted to recession days. The active configuration uses:

```text
minimum decline length: 4 days
drop first decline day: true
decreasing-rate filter: true
cold-temperature filter: temperature_C <= 0.0 deg C removed
```

The cold-temperature filter is a temperature-based snowmelt proxy. It removes recession days whose daily mean temperature is at or below 0 deg C.

Prepared experiment inputs report:

```text
total recession days: 6,224,673
```

## Model

The model follows the Ara-style physics-informed `LSTM-epsilon` formulation. It directly infers daily `epsilon` inside the recession equation rather than predicting `GQ` first and dividing by `Q`.

For each catchment and day, the model uses a 365-day context window of dynamic inputs and static attributes:

```text
dynamic inputs:
precipitation_mmd, temperature_C, pet_mmd, SM_%

static attributes:
longitude, latitude, area_km2, Prec_mm, Temp_C, PET_mm, AET_mm,
P_AET_mm, Aridity, elevation_mean_m, mean_slope_degree,
Median_DepthToBedrock_cm, max_soil_moisture, Porosity,
Seasonality_of_Moisture_Index, low_high_ratio, wet_days_ratio_1mm,
wet_days_ratio_5mm, high_prec_freq_10mm, high_prec_dur_10mm,
low_prec_freq_1mm, low_prec_dur_1mm
```

The network outputs:

```text
epsilon_t, q_base_t, alpha, LP, gamma
```

The recession equation is:

```text
dQ/dt = -epsilon * Q^2 - epsilon * alpha * AET * Q
```

AET is computed inside the model from PET and soil moisture using bounded `LP` and `gamma` parameters. Streamflow is solved with a closed-form state-reset recession path and compared against observed streamflow on recession days.

## Training

The active full training run uses:

```text
folds: 5
epochs: 150
batch size: 256
learning rate: 1e-4
hidden size: 256
layers: 1
dropout: 0.4
n_mul: 10
sequence length: 365
mixed precision: false
```

The loss is a physics-informed epsilon-core objective:

```text
L = lambda_path * L_path
  + lambda_rhs * L_rhs
  + lambda_smooth * L_smooth
  + lambda_q0 * L_q0
```

with:

```text
lambda_path: 25.0
lambda_rhs: 10.0
lambda_smooth: 0.1
lambda_q0: 5.0
huber_delta: 0.5
```

`L_path` compares the integrated recession streamflow path against observed streamflow. `L_rhs` constrains the differential-equation tendency. `L_smooth` penalizes excessive daily epsilon curvature. `L_q0` aligns the reset initial flow with observed streamflow at recession starts.

## Outputs

The active private output layout is:

```text
_private/results/epsilon_pure_gcin_1950_2019/
  full_pure_gcin_1950_2019/
    fold_<k>/
      best_model.pt
      metrics.csv
      heldout_epsilon_change_summary.parquet
      recession_day_simulations.parquet
      run_metadata.json
    production_audit.csv
_private/results/epsilon_pure_gcin_1950_2019/analysis/
  crossfit_epsilon_change_summary.parquet
  crossfit_epsilon_change_summary.csv
  crossfit_training_metrics.csv
  crossfit_delta_epsilon_stats.csv
```

Figure and public-summary outputs are generated after training:

```text
_private/results/paper_figures_pure_gcin/
paper_repo/docs/SUMMARY.md
_submission/LSTM_epsilon_publish/public/modules/epsilon-change/data/epsilon-catchment-distributions.json
```

Current production audit summary:

```text
all-period pooled NSE:           0.574
all-period pooled KGE:           0.707
all-period median catchment NSE: 0.466
all-period median catchment KGE: 0.663
post-period median catchment NSE: 0.485
pre-period median catchment NSE:  0.457

valid catchments for delta epsilon: 2,297
mean pre epsilon:                   0.715
mean post epsilon:                  0.757
mean delta epsilon:                 0.080
median delta epsilon:               0.019
```

## Reproduction Commands

Run all Python commands in the project-level `hydro` conda environment.

Prepare experiment inputs:

```powershell
conda run -n hydro python paper_repo\src\epsilon_model\prepare_experiment_inputs.py `
  --config paper_repo\configs\epsilon_experiment_pure_gcin_1950_2019.yaml
```

Train one fold:

```powershell
conda run -n hydro python paper_repo\src\epsilon_model\train_epsilon_model.py `
  --config paper_repo\configs\epsilon_experiment_pure_gcin_1950_2019.yaml `
  --fold 0 `
  --run-label full_pure_gcin_1950_2019
```

After all folds finish, run the full postprocess:

```powershell
conda run -n hydro python paper_repo\src\epsilon_model\run_full_postprocess.py `
  --config paper_repo\configs\epsilon_experiment_pure_gcin_1950_2019.yaml `
  --run-label full_pure_gcin_1950_2019 `
  --figures-dir _private\results\paper_figures_pure_gcin `
  --summary-md paper_repo\docs\SUMMARY.md `
  --github-pages-out _submission\LSTM_epsilon_publish\public\modules\epsilon-change\data\epsilon-catchment-distributions.json
```

The postprocess command runs fold inference where missing, aggregates cross-fitted outputs, computes the production audit including NSE/KGE, generates figures, updates the public summary, and exports GitHub Pages data.
