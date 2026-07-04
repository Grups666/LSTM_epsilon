# Catchment Epsilon Change Around 1990

## Introduction

This study examines whether catchment recession behavior changed after 1990. The central quantity is `epsilon`, a daily latent coefficient inferred by a physics-informed LSTM inside a recession-flow differential equation. It is not temperature, precipitation, or a post-hoc ratio from a standard LSTM. It represents the model-inferred recession response of streamflow under the observed hydroclimatic state.

The analysis compares two periods:

```text
pre-change:  1950-01-01 to 1990-12-31
post-change: 1991-01-01 to 2019-12-31
```

The primary question is whether daily recession-period epsilon shifts after 1990, and whether that shift differs between low-flow and high-flow recession conditions.

## Data Resources

The active dataset is a pure GCIN catchment-day product. Observed streamflow is indexed by `GCIN`, and all meteorological and land-state drivers are reduced over the corresponding GCIN catchment boundaries.

```text
period:                    1950-2019
model-ready catchments:     2,511
recession simulation days:  9,192,715
format:                    yearly parquet
```

The dynamic inputs include catchment-level precipitation, temperature, PET, soil moisture, streamflow, and AET-related variables. Static attributes include location, area, long-term hydroclimate summaries, aridity, soil and topographic properties, precipitation frequency/duration metrics, and moisture indices.

Daily rows are catchment time-series records rather than raster grid cells.

## Method

The model follows the Ara-style physics-informed `LSTM-epsilon` formulation. It directly predicts daily epsilon inside the recession equation:

```text
dQ/dt = -epsilon * Q^2 - epsilon * alpha * AET * Q
```

The model does not first predict `GQ` and then divide by observed `Q`. For each target day, it reads a 365-day dynamic context window plus static attributes. The network outputs daily epsilon and the bounded parameters needed to solve the recession path. The integrated streamflow path is supervised against observed streamflow on recession days.

Training and inference are restricted to hydrologically interpretable recession sequences:

```text
minimum decline length: 4 days
first decline day:      removed
decreasing-rate filter: enabled
cold-day filter:        temperature_C <= 0 deg C removed
```

Low-flow and high-flow analyses use each catchment's own observed recession-day flow distribution:

```text
low flow:  Qobs <= catchment Q10
mid flow:  catchment Q10 < Qobs < catchment Q90
high flow: Qobs >= catchment Q90
```

These are local thresholds, not global discharge cutoffs.

## Model Skill

The production run used five cross-fitted folds. Median catchment-level skill across the recession simulations is:

```text
median catchment NSE: 0.466
median catchment KGE: 0.663
pooled NSE:           0.574
pooled KGE:           0.707
```

Period-specific catchment skill is:

```text
pre-period median NSE:  0.457
post-period median NSE: 0.485
pre-period median KGE:  0.660
post-period median KGE: 0.667
```

For visual exploration, the GitHub Pages map displays the high-skill subset where both pre-period and post-period catchment NSE are greater than 0.5:

```text
catchments with both-period NSE > 0.5: 846
evaluated catchments with pre/post contrast: 2,297
```

The underlying result files retain all evaluated catchments; the filter is only applied in the public map.

## Epsilon Change

Across evaluated catchments, the all-recession epsilon contrast shows a positive post-1990 shift:

```text
valid catchments for delta epsilon: 2,297
mean pre epsilon:                   0.715
mean post epsilon:                  0.757
mean delta epsilon:                 0.080
median delta epsilon:               0.019
negative-delta catchment share:     23.3%
```

The flow-regime view is essential because the same catchment can show different epsilon shifts under low-flow and high-flow recession conditions. The public explorer therefore classifies or colors catchments using low-flow and high-flow relative epsilon changes rather than only long-term mean epsilon.

## Spatial Skill Pattern

The `NSE > 0.5` display filter strongly changes the spatial sample. Europe retains many catchments, while CONUS, Australia, and Africa retain far fewer:

| Region | Evaluated | Kept | Kept % | Median pre NSE | Median post NSE | Median precip corr | Median Q90/Q10 | Median aridity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Europe | 1,283 | 724 | 56.4 | 0.561 | 0.609 | 0.823 | 8.06 | 0.634 |
| CONUS | 312 | 28 | 9.0 | 0.360 | 0.293 | 0.704 | 31.48 | 1.026 |
| Australia | 222 | 28 | 12.6 | 0.310 | 0.250 | 0.349 | 38.91 | 1.125 |
| Africa | 241 | 17 | 7.1 | 0.258 | 0.235 | 0.811 | 22.64 | 1.317 |
| South America | 138 | 37 | 26.8 | 0.407 | 0.290 | 0.698 | 6.79 | 0.707 |

The contrast is not explained by simple daily flow variability alone. The retained catchments tend to be cooler, less arid, smaller, and less extreme in low-to-high flow contrast. Across catchments, `min(pre NSE, post NSE)` is negatively associated with aridity and temperature:

```text
Spearman(min NSE, aridity):     -0.469
Spearman(min NSE, temperature): -0.467
Spearman(min NSE, Q90/Q10):     -0.205
Spearman(min NSE, precip corr):  0.182
```

The precipitation-matching hypothesis is partly supported at the regional scale: Europe has higher GEE-vs-original-forcing precipitation agreement than CONUS. However, within CONUS alone, precipitation correlation does not fully separate retained from removed catchments. The current interpretation is that model skill is controlled by a combination of forcing agreement, aridity, thermal regime, flow intermittency, and recession-regime contrast.

## Figures

The current pure GCIN run is summarized by:

- [Model skill distribution](assets/epsilon_pure_gcin_1950_2019/figure_01_model_skill.png)
- [Epsilon change distribution](assets/epsilon_pure_gcin_1950_2019/figure_02_delta_distribution.png)
- [Hydroclimate gradients](assets/epsilon_pure_gcin_1950_2019/figure_03_hydroclimate_gradients.png)
- [Spatial epsilon change](assets/epsilon_pure_gcin_1950_2019/figure_04_spatial_delta.png)

The companion technical record is:

```text
paper_repo/docs/TECHNICAL_METHODS.md
```
