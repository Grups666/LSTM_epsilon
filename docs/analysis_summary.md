# Epsilon Change Analysis Summary

Version: `v0.1.0`

## Research Question

This analysis asks whether catchment-scale recession epsilon changed around 1990. Epsilon is inferred directly as a daily latent coefficient in a physics-informed recession equation.

The public module visualizes cross-fitted epsilon changes for `4,290` catchments with valid pre/post all-recession summaries.

## Periods

- Pre-change period: 1982-1990
- Post-change period: 1991-2019

## Flow Regimes

Flow regimes are defined separately within each catchment from observed recession-day streamflow:

- All recession days
- Low flow: `Q_obs <= catchment Q10`
- Mid flow: `catchment Q10 < Q_obs < catchment Q90`
- High flow: `Q_obs >= catchment Q90`

## Main Results

Across all valid catchments, epsilon increased on average after 1990, while the median shift was much smaller:

- Mean delta epsilon: `7.395e-02`
- Median delta epsilon: `7.832e-03`
- Share of catchments with negative all-recession delta: `27.7%`

Flow-regime summaries show positive mean shifts in all three regimes:

- Low-flow mean delta epsilon: `5.704e-02`
- Mid-flow mean delta epsilon: `6.871e-02`
- High-flow mean delta epsilon: `3.729e-02`

The relative mean changes are `10.4%` for low flow and `11.7%` for high flow. The spatial and distribution views show a heterogeneous response: many catchments shift only slightly, while a subset contributes larger positive or negative changes.

## Visualization Module

The Tereon module provides:

- a global catchment point layer colored by all-recession relative epsilon change;
- a right-side catchment inspector after clicking a catchment;
- pre/post density and CDF previews for all-, low-, mid-, and high-flow regimes;
- an expanded density/CDF modal with synchronized cursor readouts.

The module data are stored in:

`public/modules/epsilon-change/data/epsilon-catchment-distributions.json`

## Notes

This repository contains public-facing visualization and summary material only. Internal meeting records, private processing notes, model checkpoints, and temporary analysis outputs are not included.
