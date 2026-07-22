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
run label: crossfit_1990
cluster/fold count: 5
batch_size: 512
maximum epochs: 30
basin roles per outer fold: approximately 70% train / 10% validation / 20% test
```

The model and physics-informed objective follow the reference `LSTM-epsilon` structure. For each outer fold, the held-out test basins never participate in fitting, normalization, validation, or checkpoint selection. Normalization is estimated from training basins only, and the best checkpoint is selected by median catchment NSE on separate validation basins. Each catchment is inferred exactly once by the model for which it belongs to the held-out test fold.

This is basin-held-out cross-fitting for a gauged-catchment attribution study, not strict ungauged prediction. Observed Q in held-out catchments is still used to identify recession days, define local Q10/Q90 regimes, evaluate NSE, and supply the reference workflow's Q-derived `low_high_ratio` static attribute.

The 1990 transition is a prespecified scientific comparison point, not a temporal train/test boundary. All basin roles use the full 1950-2019 record because the goal is cross-catchment epsilon estimation rather than future-flow forecasting. Data-driven breakpoint or transition-interval estimation is reserved for a later robustness analysis and is not used to tune the present model.

Cold-temperature filtering is enabled:

```text
remove recession days with daily mean temperature <= 0 deg C
```

The cold-temperature filter threshold is `0.0 deg C`; this removes recession days where the daily mean temperature is at or below the threshold.

## Results

The final cross-fitted analysis covers `2,297` catchments and `9,192,715` recession-day simulations.

### Model Skill

![Validation NSE and training loss](assets/epsilon_pure_gcin_1950_2019/figure_01_training_loss.png)

The model was evaluated only on held-out outer-fold basins using recession-day streamflow. Following the reference testing workflow, each held-out basin is inferred retrospectively over its complete available record; this is not a future-only forecast. Catchment-level NSE is the primary skill metric and checkpoint-selection criterion. Scores are summarized by the median across basins so large high-flow basins do not dominate the diagnostic; KGE is retained as a supplementary robustness metric.

```text
median catchment NSE: 0.327
catchment NSE p10-p90: -1.045 to 0.651
median catchment KGE: 0.494
catchment KGE p10-p90: -0.168 to 0.770
pre-period median NSE / KGE: 0.302 / 0.478
post-period median NSE / KGE: 0.388 / 0.520
pooled NSE, supplementary: 0.343
pooled KGE, supplementary: 0.498
```

Median catchment NSE is the primary reported diagnostic because each catchment contributes one score. Catchment KGE and pooled NSE/KGE are supplementary; pooled scores stack all recession-day records, so long-record or high-flow catchments can dominate the value.

The public explorer retains all evaluated catchments in its JSON. Its Overview panel applies the reliability filter in the browser: users can switch between NSE and KGE and change the threshold. At the default threshold of 0.5, `566` catchments pass NSE in both periods and `1,003` pass KGE in both periods.

The full-cohort epsilon shift below is descriptive because the held-out NSE distribution has a substantial low-skill tail. For the primary reliability subset, both pre-period and post-period catchment NSE must exceed 0.5. All `566` catchments passing that rule have a valid pre/post epsilon contrast:

```text
reliability-subset mean delta epsilon: 1.982e-02
reliability-subset median delta epsilon: 9.771e-03
reliability-subset share with negative delta epsilon: 21.6%
```

This filter does not validate epsilon against a direct observation: epsilon remains latent, and NSE measures the skill of the physics-constrained streamflow reconstruction. The subset result should therefore be interpreted as a change in model-inferred epsilon among catchments with adequate indirect reconstruction skill.

### Epsilon Shift

![Epsilon delta distribution by all days and flow regime](assets/epsilon_pure_gcin_1950_2019/figure_02_delta_distribution.png)

For each catchment, epsilon change is defined as the post-change mean minus the pre-change mean:

```text
delta epsilon = mean epsilon in 1991-2019 - mean epsilon in 1950-1990
```

Across all recession days:

```text
mean pre-change epsilon: 4.259e-01
mean post-change epsilon: 4.490e-01
mean delta epsilon: 3.777e-02
median delta epsilon: 8.809e-03
catchment share with negative delta epsilon: 28.3%
```

The mean, median, and negative-share statistics describe the central tendency and sign balance of the catchment-level epsilon shift. They should be interpreted together: the mean is sensitive to large-magnitude catchments, while the median is the more robust summary of the typical catchment.

Flow-regime summaries use basin-specific observed-flow thresholds:

```text
low-flow epsilon:  recession days with observed Q <= each catchment's Q10
high-flow epsilon: recession days with observed Q >= each catchment's Q90
mid-flow epsilon:  Q10 < observed Q < Q90
```

- `low` flow: mean delta epsilon = 3.341e-02; median delta epsilon = 4.151e-03; mean relative delta = 6.5%.
- `mid` flow: mean delta epsilon = 2.569e-02; median delta epsilon = 4.677e-03; mean relative delta = 6.2%.
- `high` flow: mean delta epsilon = 1.619e-02; median delta epsilon = 2.194e-03; mean relative delta = 7.4%.

Low-flow and high-flow epsilon are evaluated separately because recession behavior under the tails of the flow distribution can reflect different storage-release controls. Their mean relative changes are `6.5%` for low flow and `7.4%` for high flow. These flow-regime summaries should be read together with the median and quartile structure in the table, because outlier catchments can move the mean.

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

The map highlights where post-1990 changes cluster spatially. The cross-catchment median change is `8.809e-03`.

## Method Summary

For each catchment, the model reads a 365-day context window of dynamic inputs plus static attributes. It predicts daily `epsilon_t`, `q_base_t`, and bounded AET parameters `alpha`, `LP`, and `gamma`. AET is computed inside the model from PET, soil moisture, LP, and gamma. Streamflow is then solved through the closed-form state-reset recession equation and supervised against observed streamflow on recession days.

The main differential equation is:

```text
dQ/dt = -epsilon * Q^2 - epsilon * alpha * AET * Q
```

The model is therefore an epsilon-core physics-informed LSTM that infers daily epsilon directly inside the recession equation.
