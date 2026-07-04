# Technical Methods

## Scope

This document records the active data, model, training, evaluation, and public-output workflow for the GCIN-indexed LSTM-epsilon experiment.

```text
run label: full_pure_gcin_1950_2019
config:    paper_repo/configs/epsilon_experiment_pure_gcin_1950_2019.yaml
status:    five-fold training, inference aggregation, audit, figures, and public JSON export complete
```

All Python commands for this project should be run in the conda environment `hydro`.

## Data Products

The model-ready daily series combines observed streamflow and catchment-reduced meteorological drivers:

```text
identifier: GCIN
years:      1950-2019
catchments: 2,511
record type: catchment-day time series
```

Private model inputs:

```text
_private/processed/epsilon_training_daily_pure_gcin_1950_2019_parquet/
_private/processed/epsilon_physics_daily_pure_gcin_1950_2019_parquet/
_private/processed/epsilon_model_inputs_pure_gcin_1950_2019/static_attributes.parquet
_private/processed/epsilon_model_inputs_pure_gcin_1950_2019/lp_gamma_fit_summary.parquet
```

Core daily fields:

```text
GCIN, date, precipitation_mmd, temperature_C, pet_mmd,
SM_%, streamflow_mmd, observed_AET_mm
```

The public repository includes only derived summaries, figures, and visualization JSON. Large private time-series parquet files and model checkpoints are not published.

## Static Attributes

The model uses static attributes aligned to GCIN catchments:

```text
longitude, latitude, area_km2, Prec_mm, Temp_C, PET_mm, AET_mm,
P_AET_mm, Aridity, elevation_mean_m, mean_slope_degree,
Median_DepthToBedrock_cm, max_soil_moisture, Porosity,
Seasonality_of_Moisture_Index, low_high_ratio, wet_days_ratio_1mm,
wet_days_ratio_5mm, high_prec_freq_10mm, high_prec_dur_10mm,
low_prec_freq_1mm, low_prec_dur_1mm
```

PET is derived from available meteorological variables and is used in the model's AET calculation. `LP` and `gamma` recession/AET parameters are computed from the prepared daily series and stored in `lp_gamma_fit_summary.parquet`.

## Recession Selection

Training and epsilon inference are performed on recession days. The active recession filter:

```text
minimum decline length: 4 days
drop first decline day: true
decreasing-rate filter: true
cold-temperature filter: remove temperature_C <= 0 deg C
```

The prepared experiment inputs contain:

```text
total recession days: 6,224,673
recession simulation days after cross-fitted inference: 9,192,715
```

Flow-regime summaries are computed within each catchment:

```text
low flow:  Qobs <= catchment recession-day Q10
mid flow:  catchment Q10 < Qobs < catchment Q90
high flow: Qobs >= catchment recession-day Q90
```

## Model

The model is a physics-informed LSTM-epsilon model. It directly infers `epsilon_t` in the recession equation:

```text
dQ/dt = -epsilon * Q^2 - epsilon * alpha * AET * Q
```

For each target recession day, the model reads a 365-day dynamic context window and static attributes. The network outputs:

```text
epsilon_t, q_base_t, alpha, LP, gamma
```

AET is computed internally from PET and soil moisture using bounded `LP` and `gamma`. The recession path is solved with a state-reset formulation and compared against observed streamflow.

## Training Configuration

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

The objective is:

```text
L = lambda_path * L_path
  + lambda_rhs * L_rhs
  + lambda_smooth * L_smooth
  + lambda_q0 * L_q0
```

with:

```text
lambda_path:   25.0
lambda_rhs:    10.0
lambda_smooth:  0.1
lambda_q0:      5.0
huber_delta:    0.5
```

`L_path` compares solved streamflow against observed streamflow. `L_rhs` constrains the differential-equation tendency. `L_smooth` penalizes excessive daily epsilon curvature. `L_q0` aligns recession-start flow reset with observed streamflow.

## Evaluation

Model skill is evaluated with catchment-level NSE and KGE, reported separately for pre-change and post-change periods and also as aggregate summaries.

Current production audit:

```text
all-period pooled NSE:            0.574
all-period pooled KGE:            0.707
all-period median catchment NSE:  0.466
all-period median catchment KGE:  0.663
pre-period median catchment NSE:  0.457
post-period median catchment NSE: 0.485
```

The public map applies a visualization-only reliability filter:

```text
pre_nse > 0.5 and post_nse > 0.5
displayed catchments: 846
```

All evaluated catchments remain in the underlying JSON and tabular outputs.

## Spatial Skill Diagnostic

The current diagnostic evaluates why the reliability-filtered map retains many European catchments and relatively few CONUS catchments. It combines model skill, catchment location, GEE-vs-original-forcing precipitation agreement, Qobs variability, Q90/Q10 flow contrast, aridity, area, and temperature.

Diagnostic files:

```text
_private/audits/nse_filter_geography/nse_filter_diagnostic_by_catchment.csv
_private/audits/nse_filter_geography/nse_filter_region_summary.csv
_private/audits/nse_filter_geography/nse_filter_diagnostic_summary.json
```

Main diagnostic conclusion: the spatial skill contrast is associated with hydroclimate and flow-regime structure. The retained subset tends to be cooler, less arid, and less extreme in low-to-high recession flow contrast. Regional precipitation agreement also matters, but it does not fully explain within-region skill differences.

## Outputs

Private model outputs:

```text
_private/results/epsilon_pure_gcin_1950_2019/full_pure_gcin_1950_2019/
  fold_<k>/
    best_model.pt
    metrics.csv
    heldout_epsilon_change_summary.parquet
    recession_day_simulations.parquet
    run_metadata.json
  production_audit.csv
```

Analysis and figure outputs:

```text
_private/results/paper_figures_pure_gcin/
paper_repo/docs/assets/epsilon_pure_gcin_1950_2019/
_submission/LSTM_epsilon_publish/public/modules/epsilon-change/data/epsilon-catchment-distributions.json
```

## Reproduction Commands

Prepare inputs:

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

Run full postprocess:

```powershell
conda run -n hydro python paper_repo\src\epsilon_model\run_full_postprocess.py `
  --config paper_repo\configs\epsilon_experiment_pure_gcin_1950_2019.yaml `
  --run-label full_pure_gcin_1950_2019 `
  --figures-dir _private\results\paper_figures_pure_gcin `
  --summary-md paper_repo\docs\SUMMARY.md `
  --github-pages-out _submission\LSTM_epsilon_publish\public\modules\epsilon-change\data\epsilon-catchment-distributions.json
```
