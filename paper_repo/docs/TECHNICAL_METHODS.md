# Technical Methods

## Data

The production experiment uses the ERA5-Land plus legacy streamflow dataset for 1950-2019. The forcing and model inputs are stored locally under `_private` and are not part of the public repository.

```text
daily training data:
_private/processed/epsilon_training_daily_era5land_legacy_qobs_parquet/

physics daily data:
_private/processed/epsilon_physics_daily_era5land_legacy_qobs_parquet/

static attributes:
_private/processed/epsilon_model_inputs_era5land_legacy/static_attributes.parquet
```

Each physics daily row is catchment-level, not gridded. The core fields are:

```text
GCIN, date, precipitation_mmd, temperature_C, pet_mmd, SM_%,
streamflow_mmd, observed_AET_mm
```

Observed streamflow is from the legacy Event_Typology streamflow source. ERA5-Land supplies the meteorological and land-state variables; PET is derived from the available meteorological fields during preprocessing.

## Periods

The climate contrast is defined as:

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

Training and epsilon inference are restricted to recession days. The current run uses:

```text
minimum decline length: 4 days
drop first decline day: true
decreasing-rate filter: true
cold-temperature filter: temperature_C <= 0.0 deg C removed
```

The cold-temperature filter is a temperature-based snow proxy. It removes recession days whose daily mean temperature is at or below 0 deg C.

## Model

The model follows the Ara-style physics-informed `LSTM-epsilon` formulation. It directly infers daily `epsilon` inside the recession equation rather than predicting `GQ` first and dividing by `Q`.

For each catchment and day, the model uses a 365-day context window of dynamic inputs and static attributes:

```text
dynamic inputs:
precipitation_mmd, temperature_C, pet_mmd, SM_%

static attributes:
longitude, latitude, area_km2, Prec_mm, Temp_C, PET_mm, AET_mm,
P_AET_mm, Aridity, mean_sm_rootzone, max_soil_moisture, Porosity,
Annual_Mean_Moisture_Index, Seasonality_of_Moisture_Index,
mean_net_radiation_mj_m2_day, swvl1_mean, swvl2_mean, swvl3_mean,
swvl4_mean, wet_days_ratio_1mm, wet_days_ratio_5mm,
high_prec_freq_10mm, high_prec_dur_10mm, low_prec_freq_1mm,
low_prec_dur_1mm
```

The network outputs:

```text
epsilon_t, q_base_t, alpha, LP, gamma
```

The recession equation is:

```text
dQ/dt = -epsilon * Q^2 - epsilon * alpha * AET * Q
```

AET is computed inside the model from PET and soil moisture using bounded `LP` and `gamma` parameters. Streamflow is then solved with a closed-form state-reset recession path and compared against observed streamflow.

## Training

The production run is:

```text
run label: full_crossfit_era5land_legacy_1950_2019
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

The main private run outputs are:

```text
_private/results/epsilon_era5land_legacy_1950_2019/
  full_crossfit_era5land_legacy_1950_2019/
    fold_<k>/
      best_model.pt
      metrics.csv
      heldout_epsilon_change_summary.parquet
      recession_day_simulations.parquet
      run_metadata.json
    production_audit.csv
  paper_figures/
    result_summary.csv
    model_skill_summary.csv
    model_skill_by_catchment.csv
    epsilon_change_by_catchment.csv
    epsilon_change_by_flow_regime.csv
    figure_01_training_loss.png/svg
    figure_02_delta_distribution.png/svg
    figure_03_hydroclimate_gradients.png/svg
    figure_04_spatial_delta.png/svg
```

The public documentation assets are copied to:

```text
paper_repo/docs/assets/epsilon_era5land_legacy_1950_2019/
```

## Audit

The production audit verifies that all five folds contain model checkpoints, training metrics, held-out epsilon summaries, recession-day simulations, and required columns. It also computes supplementary pooled NSE/KGE and primary catchment-level NSE/KGE summaries.

The audit passed after all five folds completed. Fold-level median catchment NSE values were:

```text
fold 0: -0.002
fold 1: -0.001
fold 2:  0.011
fold 3: -0.010
fold 4: -0.011
```

The final all-period model-skill summary is:

```text
pooled NSE: 0.301
pooled KGE: 0.449
median catchment NSE: -0.005
median catchment KGE: 0.094
p10-p90 catchment NSE: -0.442 to 0.213
p10-p90 catchment KGE: -0.301 to 0.403
```

Pooled metrics are supplementary because they stack all recession-day predictions before scoring. Catchment-level metrics are the primary diagnostic because each catchment contributes one score.

## Epsilon Contrast

The final cross-fitted epsilon contrast covers:

```text
catchments with valid delta: 1,149
recession-day simulations: 5,012,615
mean delta epsilon: -7.658e-03
median delta epsilon: -1.916e-03
share with negative delta: 60.5%
```

Flow-regime mean and median deltas are:

```text
low flow:  mean -5.936e-02, median -8.941e-03
mid flow:  mean -1.322e-02, median -4.148e-03
high flow: mean  1.856e-02, median  1.743e-03
```

## Reproduction Commands

Run all Python commands in the project-level `hydro` conda environment.

```powershell
conda run -n hydro python paper_repo\src\epsilon_model\audit_production_run.py `
  --config paper_repo\configs\epsilon_experiment_era5land_legacy_1950_2019.yaml `
  --run-label full_crossfit_era5land_legacy_1950_2019

conda run -n hydro python paper_repo\src\epsilon_model\make_paper_figures.py `
  --config paper_repo\configs\epsilon_experiment_era5land_legacy_1950_2019.yaml `
  --run-label full_crossfit_era5land_legacy_1950_2019 `
  --out-dir _private\results\epsilon_era5land_legacy_1950_2019\paper_figures

conda run -n hydro python paper_repo\src\epsilon_model\update_summary_from_results.py `
  --config paper_repo\configs\epsilon_experiment_era5land_legacy_1950_2019.yaml `
  --figures-dir paper_repo\docs\assets\epsilon_era5land_legacy_1950_2019 `
  --summary-md paper_repo\docs\SUMMARY.md `
  --run-label full_crossfit_era5land_legacy_1950_2019
```
