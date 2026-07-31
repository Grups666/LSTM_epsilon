# Catchment Epsilon Change Around 1990

## Introduction

This study asks whether catchment recession behavior changed across the 1990 transition. We use `epsilon` as a daily latent coefficient in a physics-informed recession equation, inferred directly by the model for each recession day.

The analysis is organized around two periods:

```text
pre-change:  1950-01-01 to 1990-12-31
post-change: 1991-01-01 to 2019-12-31
```

The main scientific question is:

```text
Did catchment epsilon shift between 1950-1990 and 1991-2019, and is that shift structured by flow regime and hydroclimate?
```

## Resources

The current analysis uses:

```text
ERA5-Land catchment daily forcing and state variables
Event_Typology observed streamflow
catchment static attributes
LP/gamma AET prior bounds
Ara-style physics-informed epsilon-core LSTM
```

The model-ready daily series are stored as yearly parquet files under:

```text
_private/processed/epsilon_physics_daily_pure_gcin_1950_2019_parquet/
```

Each record contains:

```text
GCIN, date, precipitation_mmd, temperature_C, pet_mmd,
SM_%, streamflow_mmd, observed_AET_mm
```

Observed-Q duration is based on valid daily values rather than nominal table bounds:

```text
median valid Q record:                         16,132 days
median valid Q days in 1950-1990:             7,472
median valid Q days in 1991-2019:             9,204
catchments with any valid Q in both periods:   2,304
catchments with >= 2 years in both periods:    2,220
catchments with >= 5 years in both periods:    2,133
```

In the current pure GCIN run, `GCIN` is the original GCIN catchment identifier. Legacy/GridCode/Catchment_ID mixed products are excluded from production analysis because their numeric identifiers cannot be assumed to reference the same catchment boundaries.

## Current Run

The production run is:

```text
run label: temporal_crossfit_1990
cluster/fold count: 5
batch_size: 512
maximum epochs: 30
time roles per outer fold: 80% train / 20% test; no validation set
```

The model and physics-informed objective follow the reference `LSTM-epsilon` structure. The pre and post periods are each divided into five contiguous calendar blocks. In fold `k`, pre block `k` and post block `k` are held out together, while the other four pre blocks and four post blocks train one shared model. All catchments participate in every fold, but no test date can enter a training target, training context window, or normalization estimate.

There is no validation split and no checkpoint selection from NSE. Architecture, physics-loss weights, learning rate, and epoch count are fixed before testing; every fold retains its epoch-30 model. The Q-derived `low_high_ratio` static attribute is recomputed from training years only in each fold.

Across five rotations, every eligible date receives one out-of-fold prediction from a model that did not train on its temporal block. The resulting out-of-fold records are concatenated before calculating each catchment's pre-period NSE, post-period NSE, and epsilon contrast. Fold-level NSE means and dispersion are retained only as stability diagnostics.

Cold-temperature filtering is enabled:

```text
remove recession days with daily mean temperature <= 0 deg C
```

The cold-temperature filter threshold is `0.0 deg C`; this removes recession days where the daily mean temperature is at or below the threshold.

## Results

The final cross-fitted analysis covers `2,297` catchments and `9,192,715` recession-day simulations.

### Model Skill

![Training loss across temporal folds](assets/epsilon_pure_gcin_1950_2019/figure_01_training_loss.png)

The model is evaluated only on out-of-fold recession days. Inference uses the same 365-day warm-up plus 365-day target-window geometry used during training. Catchment-level NSE is an indirect reliability metric for latent epsilon, not a tuning target. Scores are summarized by the median across basins so large or long-record basins do not dominate the diagnostic; KGE is retained as a supplementary robustness metric.

```text
median catchment NSE: 0.581
catchment NSE p10-p90: 0.128 to 0.775
median catchment KGE: 0.642
catchment KGE p10-p90: 0.152 to 0.837
pre-period median NSE / KGE: 0.555 / 0.622
post-period median NSE / KGE: 0.626 / 0.666
pooled NSE, supplementary: 0.577
pooled KGE, supplementary: 0.586
```

For each catchment and period, the primary NSE is calculated once after concatenating all five folds' out-of-fold predictions. It is not the arithmetic mean of five fold NSE values, because each block has a different observed-flow variance. Fold NSE averages remain supplementary diagnostics.

The public explorer retains all evaluated catchments in its JSON. Its Overview panel applies the reliability filter in the browser: users can switch between NSE and KGE and change the threshold. At the default threshold of 0.5, `1,304` catchments pass NSE in both periods and `1,447` pass KGE in both periods.

The full-cohort epsilon shift below is descriptive when the out-of-fold NSE distribution has a substantial low-skill tail. For the primary reliability subset, both pre-period and post-period catchment NSE must exceed 0.5. All `1,304` catchments passing that rule have a valid pre/post epsilon contrast:

```text
reliability-subset mean delta epsilon: -1.917e-02
reliability-subset median delta epsilon: 2.296e-03
reliability-subset share with negative delta epsilon: 42.0%
bootstrap 95% CI for mean delta: -3.477e-02 to -5.214e-03
bootstrap 95% CI for median delta: 1.514e-03 to 3.239e-03
```

The intervals resample catchments and quantify cross-catchment sampling uncertainty; they do not account for spatial dependence or model structural uncertainty. This filter does not validate epsilon against a direct observation: epsilon remains latent, and NSE measures the skill of the physics-constrained streamflow reconstruction. The subset result should therefore be interpreted as a change in model-inferred epsilon among catchments with adequate indirect reconstruction skill.

### Epsilon Shift

![Epsilon delta distribution by all days and flow regime](assets/epsilon_pure_gcin_1950_2019/figure_02_delta_distribution.png)

For each catchment, epsilon change is defined as the post-change mean minus the pre-change mean:

```text
delta epsilon = mean epsilon in 1991-2019 - mean epsilon in 1950-1990
```

Across all recession days:

```text
mean pre-change epsilon: 8.111e-01
mean post-change epsilon: 7.059e-01
mean delta epsilon: -5.875e-02
median delta epsilon: 1.574e-03
catchment share with negative delta epsilon: 44.8%
```

The mean and median have opposite signs, showing that the catchment-level change distribution is strongly skewed. Large-magnitude negative catchments move the mean, while the typical catchment represented by the median has a small positive shift; neither direction should be reported alone.

The mean, median, and negative-share statistics describe the central tendency and sign balance of the catchment-level epsilon shift. They should be interpreted together: the mean is sensitive to large-magnitude catchments, while the median is the more robust summary of the typical catchment.

Flow-regime summaries use basin-specific observed-flow thresholds:

```text
low-flow epsilon:  recession days with observed Q <= each catchment's Q10
high-flow epsilon: recession days with observed Q >= each catchment's Q90
mid-flow epsilon:  Q10 < observed Q < Q90
```

- `low` flow: mean delta epsilon = -2.438e-01; median delta epsilon = -3.154e-03; mean relative delta = 3.1%.
- `mid` flow: mean delta epsilon = -7.448e-02; median delta epsilon = -1.004e-04; mean relative delta = 1.8%.
- `high` flow: mean delta epsilon = 5.267e-03; median delta epsilon = 2.772e-03; mean relative delta = 8.6%.

Low-flow and high-flow epsilon are evaluated separately because recession behavior under the tails of the flow distribution can reflect different storage-release controls. Their mean relative changes are `3.1%` for low flow and `8.6%` for high flow. These flow-regime summaries should be read together with the median and quartile structure in the table, because outlier catchments can move the mean.

### GQ / Q Component Attribution

The retained out-of-fold table contains daily effective epsilon and simulated Q
for every evaluated recession day. Effective GQ is reconstructed as
`epsilon_effective * Qsim`, then pre/post changes are compared in log space:

```text
delta log epsilon = delta log GQ - delta log Qsim
```

The public map preserves epsilon change as the point fill and adds an outer
component ring. In the bivariate view, the left semicircle represents low flow
and the right semicircle represents high flow. Ring classes distinguish
GQ-dominant, Q-dominant, combined, and offsetting changes. These classes explain
how the epsilon ratio changed; they do not establish climate causality or
statistical significance.

### Hydroclimate Structure

![Hydroclimate gradients of epsilon change](assets/epsilon_pure_gcin_1950_2019/figure_03_hydroclimate_gradients.png)

The hydroclimate-gradient figure bins catchments into quartiles of precipitation, temperature, and aridity, then compares mean and median epsilon change within each bin. This checks whether the epsilon shift is a spatially random artifact or whether it aligns with background catchment climate.

The hydroclimate gradients should be read as descriptive evidence, not causal attribution. They show whether epsilon shifts align with background precipitation, temperature, and aridity structure, and they identify where more formal regression or hierarchical testing would be useful.

### Spatial Pattern

![Spatial distribution of epsilon change](assets/epsilon_pure_gcin_1950_2019/figure_04_spatial_delta.png)

The spatial map shows catchment-level epsilon change as point locations. It is designed to reveal regional clustering that is hidden in the histogram and boxplot. Blue and red points mark opposite signs of epsilon change, so the map should be interpreted together with the catchment-level delta table:

```text
catchment-level table: assets/epsilon_pure_gcin_1950_2019/epsilon_change_by_catchment.csv
flow-regime table:    assets/epsilon_pure_gcin_1950_2019/epsilon_change_by_flow_regime.csv
```

The map highlights where post-1990 changes cluster spatially. The cross-catchment median change is `1.574e-03`.

## Method Summary

For each catchment, the model reads a 365-day context window of dynamic inputs plus static attributes. It predicts daily `epsilon_t`, `q_base_t`, and bounded AET parameters `alpha`, `LP`, and `gamma`. AET is computed inside the model from PET, soil moisture, LP, and gamma. Streamflow is then solved through the closed-form state-reset recession equation and supervised against observed streamflow on recession days.

The main differential equation is:

```text
dQ/dt = -epsilon * Q^2 - epsilon * alpha * AET * Q
```

The model is therefore an epsilon-core physics-informed LSTM that infers daily epsilon directly inside the recession equation.
