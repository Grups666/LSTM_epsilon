# Epsilon Change Analysis Summary

Version: `v0.1.0`

## Research Question

This analysis asks whether catchment-scale epsilon changed around the 1990 transition window. Epsilon is treated as a daily modeled indicator of catchment hydrologic partitioning, defined conceptually as `epsilon = GQ / Q`.

The current public module visualizes catchment-level inferred epsilon changes across 4,831 gauged catchments.

## Periods

- Pre-change period: 1982-1990
- Post-change period: 1991-2019

## Flow Regimes

For each catchment, epsilon distributions are summarized under three regimes:

- All days
- Low flow: days with observed streamflow at or below the catchment-specific 10th percentile
- High flow: days with observed streamflow at or above the catchment-specific 90th percentile

## Main Results

Across all catchments, mean all-flow epsilon declined after 1990:

- Mean delta epsilon: `-2.29e-4`
- Median delta epsilon: `-5.68e-5`
- Share of catchments with negative all-flow delta: `79.3%`

Flow-percentile summaries show weaker shifts in the most extreme flow states:

- Low-flow mean delta epsilon: `-4.13e-5`
- High-flow mean delta epsilon: `3.13e-5`

The all-flow decline is therefore more consistent with a broad shift in daily hydrologic behavior than with a change isolated only to the lowest-flow or highest-flow days.

## Visualization Module

The Tereon module provides:

- a global catchment point layer colored by all-flow relative epsilon change;
- a right-side catchment inspector after clicking a catchment;
- pre/post density and CDF previews for all-, low-, and high-flow regimes;
- an expanded density/CDF modal with synchronized cursor readouts.

The module data are stored in:

`public/modules/epsilon-change/data/epsilon-catchment-distributions.json`

## Notes

This repository contains public-facing visualization and summary material only. Internal meeting records, private processing notes, model checkpoints, and temporary analysis outputs are not included.
