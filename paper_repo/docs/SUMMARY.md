# Catchment Epsilon Change Around 1990

## Introduction

This study evaluates whether catchment recession behavior changed before and after 1990. The target variable is `epsilon`, a daily latent coefficient inferred inside a physics-informed recession equation. Larger or smaller `epsilon` reflects changes in the inferred recession response for a given observed streamflow state, rather than a direct temperature or precipitation variable.

The current analysis uses two periods:

```text
pre-change:  1950-01-01 to 1990-12-31
post-change: 1991-01-01 to 2019-12-31
```

The main question is whether post-1990 catchment `epsilon` shifts relative to pre-1991 conditions, and whether the shift differs between low-flow and high-flow recession days.

## Data Resources

The current production dataset is the pure GCIN daily time series dataset. It combines GCIN-indexed observed streamflow records with ERA5-Land catchment-reduced meteorological and land-state variables.

```text
period:     1950-2019
catchments: 2,511
recession simulation days: 9,192,715
format:     yearly parquet
```

The train-ready product is built only from catchments whose boundary and streamflow identifier are both GCIN. The previous legacy/GridCode/Catchment_ID mixed product is excluded because its identifiers and boundaries were not reliably matched.

```text
active daily input:
_private/processed/epsilon_training_daily_pure_gcin_1950_2019_parquet/

active physics input:
_private/processed/epsilon_physics_daily_pure_gcin_1950_2019_parquet/

active static and recession-parameter inputs:
_private/processed/epsilon_model_inputs_pure_gcin_1950_2019/
```

Important identifier rule: in the current production run, `GCIN` is the original GCIN catchment identifier. GridCode-based or mixed legacy products are not used.

## Method

The model follows the Ara-style physics-informed `LSTM-epsilon` formulation. It directly predicts daily `epsilon` inside the recession equation; it does not first predict `GQ` and then divide by observed streamflow.

For each catchment, the model reads a 365-day context window of dynamic inputs plus static attributes:

```text
dynamic inputs:
precipitation_mmd, temperature_C, pet_mmd, SM_%

static attributes:
location, area, long-term hydroclimate summaries, soil moisture summaries,
precipitation frequency/duration metrics, aridity, PET, AET, and related indices
```

Training and inference are restricted to recession days. The active filter keeps declining-flow sequences and removes cold days as a simple snowmelt proxy:

```text
minimum decline length: 4 days
drop first decline day: true
decreasing-rate filter: true
remove days with daily mean temperature <= 0 deg C
```

The core recession equation is:

```text
dQ/dt = -epsilon * Q^2 - epsilon * alpha * AET * Q
```

The network outputs daily `epsilon`, a base flow state, and bounded AET parameters. Streamflow is then solved through the state-reset recession equation and supervised against observed streamflow on recession days.

## Flow Regimes

Flow-regime analyses use each catchment's own observed recession-day flow distribution:

```text
low flow:  Qobs <= catchment Q10
mid flow:  catchment Q10 < Qobs < catchment Q90
high flow: Qobs >= catchment Q90
```

This means the low-flow and high-flow definitions are local to each catchment, not global thresholds pooled across all catchments.

## Current Run

The active production run is:

```text
run label: full_pure_gcin_1950_2019
folds:     5
epochs:    150
model:     physics-informed LSTM-epsilon
```

The full five-fold training completed on the Tsinghua RTX 3060 server. NSE/KGE, epsilon pre/post contrasts, flow-regime summaries, figures, and GitHub Pages data have been regenerated and checked for the pure GCIN run.

## Current Results

The pure GCIN five-fold run completed for all folds. Median catchment-level model skill across all recession simulations is:

```text
median catchment NSE: 0.466
median catchment KGE: 0.663
pooled NSE:           0.574
pooled KGE:           0.707
```

The all-recession epsilon contrast indicates a positive post-1990 shift:

```text
valid catchments for delta epsilon: 2,297
mean pre epsilon:                   0.715
mean post epsilon:                  0.757
mean delta epsilon:                 0.080
median delta epsilon:               0.019
negative-delta catchment share:     23.3%
```

The figure tables, PNG figures, production audit, and GitHub Pages JSON export have been generated for the pure GCIN run.

## Figures

The current pure GCIN run is summarized by these generated figures:

- [Model skill distribution](assets/epsilon_pure_gcin_1950_2019/figure_01_model_skill.png)
- [Epsilon change distribution](assets/epsilon_pure_gcin_1950_2019/figure_02_delta_distribution.png)
- [Hydroclimate gradients](assets/epsilon_pure_gcin_1950_2019/figure_03_hydroclimate_gradients.png)
- [Spatial epsilon change](assets/epsilon_pure_gcin_1950_2019/figure_04_spatial_delta.png)

The technical details and reproduction commands are maintained in:

```text
paper_repo/docs/TECHNICAL_METHODS.md
```
