from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("paper_repo/configs/epsilon_experiment_pure_gcin_1950_2019.yaml"))
    parser.add_argument("--figures-dir", type=Path, default=Path("_private/results/paper_figures_temporal_crossfit_1990"))
    parser.add_argument("--summary-md", type=Path, default=Path("paper_repo/docs/SUMMARY.md"))
    parser.add_argument("--run-label", type=str, default="temporal_crossfit_1990")
    return parser.parse_args()


def scalar_table(path: Path) -> dict[str, float]:
    df = pd.read_csv(path, index_col=0)
    return {str(k): float(v) for k, v in df["value"].items()}


def fmt_sci(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.3e}"


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{100.0 * value:.1f}%"


def markdown_relative_path(source_md: Path, target: Path) -> str:
    source_parent = source_md.resolve().parent
    target_resolved = target.resolve()
    return Path(os.path.relpath(target_resolved, source_parent)).as_posix()


def regime_line(regime: pd.DataFrame, name: str) -> str:
    row = regime[regime["regime"] == name]
    if row.empty:
        return f"- `{name}`: not available"
    row = row.iloc[0]
    return (
        f"- `{name}` flow: mean delta epsilon = {fmt_sci(row['mean_delta_mean'])}; "
        f"median delta epsilon = {fmt_sci(row['median_delta_mean'])}; "
        f"mean relative delta = {fmt_pct(row['mean_relative_delta_mean'])}."
    )


def year(value: object) -> str:
    return str(value)[:4]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    pre_start, pre_end = cfg["data"]["pre_window"]
    post_start, post_end = cfg["data"]["post_window"]
    physics_daily_dir = cfg["paths"]["physics_daily_dir"]
    batch_size = int(cfg["training"]["batch_size"])
    epochs = int(cfg["training"]["epochs"])
    n_folds = int(cfg["splits"]["n_folds"])
    cold_filter = cfg["recession"].get("cold_temperature_filter_C")
    fig = args.figures_dir
    run_root = Path(cfg["paths"]["output_dir"]) / args.run_label
    shift_path = run_root / "analysis" / "prepost_shifts_long.parquet"
    sensitivity_path = run_root / "analysis" / "prepost_shift_sensitivity_summary.csv"
    inference_path = fig / "epsilon_change_inference.csv"
    if not inference_path.exists():
        inference_path = run_root / "analysis" / "epsilon_change_inference.csv"
    required = [
        fig / "result_summary.csv",
        fig / "model_skill_summary.csv",
        fig / "model_skill_by_catchment.csv",
        fig / "epsilon_change_by_catchment.csv",
        fig / "epsilon_change_by_flow_regime.csv",
        inference_path,
        shift_path,
        sensitivity_path,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required result files:\n" + "\n".join(missing))

    result = scalar_table(fig / "result_summary.csv")
    skill = pd.read_csv(fig / "model_skill_summary.csv")
    skill_by_catchment = pd.read_csv(fig / "model_skill_by_catchment.csv")
    epsilon_by_catchment = pd.read_csv(fig / "epsilon_change_by_catchment.csv").set_index("GCIN")
    regime = pd.read_csv(fig / "epsilon_change_by_flow_regime.csv")
    inference = pd.read_csv(inference_path).set_index("subset")
    shifts = pd.read_parquet(shift_path)
    sensitivity = pd.read_csv(sensitivity_path)
    qobs_inventory_path = Path(cfg["paths"]["output_dir"]) / "inputs" / "qobs_inventory.parquet"
    qobs_inventory = pd.read_parquet(qobs_inventory_path)
    median_valid_q_days = int(qobs_inventory["valid_q_days"].median())
    median_pre_valid_q_days = int(qobs_inventory["pre_valid_q_days"].median())
    median_post_valid_q_days = int(qobs_inventory["post_valid_q_days"].median())
    both_periods_any = int(
        ((qobs_inventory["pre_valid_q_days"] > 0) & (qobs_inventory["post_valid_q_days"] > 0)).sum()
    )
    both_periods_two_years = int(
        ((qobs_inventory["pre_valid_q_days"] >= 730) & (qobs_inventory["post_valid_q_days"] >= 730)).sum()
    )
    both_periods_five_years = int(
        ((qobs_inventory["pre_valid_q_days"] >= 1825) & (qobs_inventory["post_valid_q_days"] >= 1825)).sum()
    )
    regime_summary = (
        regime.groupby("regime", observed=True)
        .agg(
            mean_delta_mean=("delta_mean", "mean"),
            median_delta_mean=("delta_mean", "median"),
            mean_relative_delta_mean=("relative_delta_mean", "mean"),
            n_catchments=("GCIN", "nunique"),
        )
        .reset_index()
    )
    skill_all = skill[skill["period"] == "all"].iloc[0]
    skill_pre = skill[skill["period"] == "pre"].iloc[0]
    skill_post = skill[skill["period"] == "post"].iloc[0]
    skill_wide = skill_by_catchment.pivot(index="GCIN", columns="period", values=["nse", "kge"])
    both_nse_05 = int(((skill_wide[("nse", "pre")] > 0.5) & (skill_wide[("nse", "post")] > 0.5)).sum())
    both_kge_05 = int(((skill_wide[("kge", "pre")] > 0.5) & (skill_wide[("kge", "post")] > 0.5)).sum())
    reliable_nse_gcins = skill_wide.index[
        (skill_wide[("nse", "pre")] > 0.5) & (skill_wide[("nse", "post")] > 0.5)
    ]
    reliable_shifts = shifts[shifts["GCIN"].isin(reliable_nse_gcins)].copy()
    reliable_shift_counts = (
        reliable_shifts.groupby(["regime", "shift_class"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    shift_wide = reliable_shifts.pivot(index="GCIN", columns="regime", values="shift_class")
    bivariate_eligible = int(
        (
            shift_wide.get("low", pd.Series(index=shift_wide.index, dtype=object)).isin(
                ["increase", "decrease", "unresolved"]
            )
            & shift_wide.get("high", pd.Series(index=shift_wide.index, dtype=object)).isin(
                ["increase", "decrease", "unresolved"]
            )
        ).sum()
    )

    def shift_count(regime_name: str, class_name: str) -> int:
        if regime_name not in reliable_shift_counts.index or class_name not in reliable_shift_counts.columns:
            return 0
        return int(reliable_shift_counts.loc[regime_name, class_name])

    breakpoint_sensitivity = sensitivity[sensitivity["scenario"].isin(["break_1985", "break_1995"])]
    reliable_nse_delta = epsilon_by_catchment.loc[
        epsilon_by_catchment.index.intersection(reliable_nse_gcins), "delta_epsilon_mean"
    ].dropna()
    reliable_nse_mean_delta = float(reliable_nse_delta.mean())
    reliable_nse_median_delta = float(reliable_nse_delta.median())
    reliable_nse_negative_share = float((reliable_nse_delta < 0).mean())
    reliable_key = "both_period_nse_gt_0.5"
    reliable_inference = inference.reindex([reliable_key]).iloc[0]
    low = regime_summary[regime_summary["regime"] == "low"].iloc[0]
    high = regime_summary[regime_summary["regime"] == "high"].iloc[0]
    fig_rel = markdown_relative_path(args.summary_md, fig)
    mean_delta = float(result["mean_delta_epsilon"])
    median_delta = float(result["median_delta_epsilon"])
    if np.sign(mean_delta) != np.sign(median_delta):
        direction_note = (
            "The mean and median have opposite signs, showing that the catchment-level change distribution is "
            "strongly skewed. Large-magnitude negative catchments move the mean, while the typical catchment "
            "represented by the median has a small positive shift; neither direction should be reported alone."
        )
    else:
        direction_note = "The mean and median have the same sign, but their magnitude difference still reflects distribution skew."

    text = f"""# Catchment Epsilon Change Around 1990

## Introduction

This study asks whether catchment recession behavior changed across the 1990 transition. We use `epsilon` as a daily latent coefficient in a physics-informed recession equation, inferred directly by the model for each recession day.

The analysis is organized around two periods:

```text
pre-change:  {pre_start} to {pre_end}
post-change: {post_start} to {post_end}
```

The main scientific question is:

```text
Did catchment epsilon shift between {year(pre_start)}-{year(pre_end)} and {year(post_start)}-{year(post_end)}, and is that shift structured by flow regime and hydroclimate?
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
{physics_daily_dir}/
```

Each record contains:

```text
GCIN, date, precipitation_mmd, temperature_C, pet_mmd,
SM_%, streamflow_mmd, observed_AET_mm
```

Observed-Q duration is based on valid daily values rather than nominal table bounds:

```text
median valid Q record:                         {median_valid_q_days:,} days
median valid Q days in {year(pre_start)}-{year(pre_end)}:             {median_pre_valid_q_days:,}
median valid Q days in {year(post_start)}-{year(post_end)}:             {median_post_valid_q_days:,}
catchments with any valid Q in both periods:   {both_periods_any:,}
catchments with >= 2 years in both periods:    {both_periods_two_years:,}
catchments with >= 5 years in both periods:    {both_periods_five_years:,}
```

In the current pure GCIN run, `GCIN` is the original GCIN catchment identifier. Legacy/GridCode/Catchment_ID mixed products are excluded from production analysis because their numeric identifiers cannot be assumed to reference the same catchment boundaries.

## Current Run

The production run is:

```text
run label: {args.run_label}
cluster/fold count: {n_folds}
batch_size: {batch_size}
maximum epochs: {epochs}
time roles per outer fold: 80% train / 20% test; no validation set
```

The model and physics-informed objective follow the reference `LSTM-epsilon` structure. The pre and post periods are each divided into five contiguous calendar blocks. In fold `k`, pre block `k` and post block `k` are held out together, while the other four pre blocks and four post blocks train one shared model. All catchments participate in every fold, but no test date can enter a training target, training context window, or normalization estimate.

There is no validation split and no checkpoint selection from NSE. Architecture, physics-loss weights, learning rate, and epoch count are fixed before testing; every fold retains its epoch-{epochs} model. The Q-derived `low_high_ratio` static attribute is recomputed from training years only in each fold.

Across five rotations, every eligible date receives one out-of-fold prediction from a model that did not train on its temporal block. The resulting out-of-fold records are concatenated before calculating each catchment's pre-period NSE, post-period NSE, and epsilon contrast. Fold-level NSE means and dispersion are retained only as stability diagnostics.

Cold-temperature filtering is enabled:

```text
remove recession days with daily mean temperature <= 0 deg C
```

The cold-temperature filter threshold is `{cold_filter} deg C`; this removes recession days where the daily mean temperature is at or below the threshold.

## Results

The final cross-fitted analysis covers `{int(result['n_valid_delta']):,}` catchments and `{int(result['n_recession_simulation_days']):,}` recession-day simulations.

### Model Skill

![Training loss across temporal folds]({fig_rel}/figure_01_training_loss.png)

The model is evaluated only on out-of-fold recession days. Inference uses the same 365-day warm-up plus 365-day target-window geometry used during training. Catchment-level NSE is an indirect reliability metric for latent epsilon, not a tuning target. Scores are summarized by the median across basins so large or long-record basins do not dominate the diagnostic; KGE is retained as a supplementary robustness metric.

```text
median catchment NSE: {skill_all['median_catchment_nse']:.3f}
catchment NSE p10-p90: {skill_all['p10_catchment_nse']:.3f} to {skill_all['p90_catchment_nse']:.3f}
median catchment KGE: {skill_all['median_catchment_kge']:.3f}
catchment KGE p10-p90: {skill_all['p10_catchment_kge']:.3f} to {skill_all['p90_catchment_kge']:.3f}
pre-period median NSE / KGE: {skill_pre['median_catchment_nse']:.3f} / {skill_pre['median_catchment_kge']:.3f}
post-period median NSE / KGE: {skill_post['median_catchment_nse']:.3f} / {skill_post['median_catchment_kge']:.3f}
pooled NSE, supplementary: {skill_all['pooled_nse']:.3f}
pooled KGE, supplementary: {skill_all['pooled_kge']:.3f}
```

For each catchment and period, the primary NSE is calculated once after concatenating all five folds' out-of-fold predictions. It is not the arithmetic mean of five fold NSE values, because each block has a different observed-flow variance. Fold NSE averages remain supplementary diagnostics.

The public explorer retains all evaluated catchments in its JSON. Its Overview panel applies the reliability filter in the browser: users can switch between NSE and KGE and change the threshold. At the default threshold of 0.5, `{both_nse_05:,}` catchments pass NSE in both periods and `{both_kge_05:,}` pass KGE in both periods.

### Primary Fold-Adjusted Era Shift

The primary scientific estimand is one post-1990 coefficient for each catchment and flow regime. Daily out-of-fold epsilon is reduced to an annual median when at least three recession days are available. For each catchment and regime, the model is:

```text
log(annual epsilon) = fold fixed effect + beta_post * I(year >= 1991) + error
era shift (%) = 100 * (exp(beta_post) - 1)
```

Fold fixed effects prevent differences among the five trained OOF models from being mistaken for a climate-era shift. A test requires at least 10 valid years in each era and at least five pre and five post years inside folds that cover both eras. A one-year Newey-West HAC covariance allows residual heteroskedasticity and serial dependence. Benjamini-Hochberg FDR correction is applied across all statistically eligible catchments separately by regime. Increase and Decrease require `q < 0.05`; otherwise the result is Unresolved, which is not evidence of stability.

At the default both-period `NSE > 0.5` display threshold:

```text
low-flow eligible:          {shift_count('low', 'increase') + shift_count('low', 'decrease') + shift_count('low', 'unresolved'):,}
  increase / decrease:      {shift_count('low', 'increase'):,} / {shift_count('low', 'decrease'):,}
  unresolved:               {shift_count('low', 'unresolved'):,}
high-flow eligible:         {shift_count('high', 'increase') + shift_count('high', 'decrease') + shift_count('high', 'unresolved'):,}
  increase / decrease:      {shift_count('high', 'increase'):,} / {shift_count('high', 'decrease'):,}
  unresolved:               {shift_count('high', 'unresolved'):,}
low/high eligible overlap:  {bivariate_eligible:,}
```

All-recession, low-flow, and high-flow analyses retain their independent eligible samples. The low/high overlap is reported as a coverage diagnostic and does not restrict either single-regime result. The FDR family is fixed before applying the interactive reliability display filter, so changing the website threshold cannot redefine statistical significance.

The annual-support sensitivity checks use one, three, and five recession days per annual median. Alternative 1985 and 1995 breakpoints are also evaluated without selecting the most significant result. Across overlapping catchments, effect correlations with the 1990 primary analysis range from `{breakpoint_sensitivity['effect_correlation_with_primary'].min():.3f}` to `{breakpoint_sensitivity['effect_correlation_with_primary'].max():.3f}` for the breakpoint checks.

### Component Attribution and Trend Sensitivity

For interpretation, `GQ = epsilon * Qsim` gives the exact descriptive identity `delta log epsilon = delta log GQ - delta log Qsim`. GQ-dominant, Q-dominant, Combined, and Offsetting labels describe how the pre/post ratio is composed; they are not causal climate attribution.

Continuous fold-centered Theil-Sen slopes and prewhitened Kendall tests are retained as a secondary robustness check. They ask whether change is monotonic through time, whereas the primary model asks whether the two predefined climate eras differ. Continuous trends never determine the map class.

The raw daily-mean contrast below is retained only as a descriptive distribution summary. For the default reliability subset, both pre-period and post-period catchment NSE exceed 0.5:

```text
reliability-subset mean delta epsilon: {fmt_sci(reliable_nse_mean_delta)}
reliability-subset median delta epsilon: {fmt_sci(reliable_nse_median_delta)}
reliability-subset share with negative delta epsilon: {fmt_pct(reliable_nse_negative_share)}
bootstrap 95% CI for mean delta: {fmt_sci(reliable_inference['mean_bootstrap_ci025'])} to {fmt_sci(reliable_inference['mean_bootstrap_ci975'])}
bootstrap 95% CI for median delta: {fmt_sci(reliable_inference['median_bootstrap_ci025'])} to {fmt_sci(reliable_inference['median_bootstrap_ci975'])}
```

The intervals resample catchments and quantify cross-catchment sampling uncertainty; they do not account for spatial dependence or model structural uncertainty. This filter does not validate epsilon against a direct observation: epsilon remains latent, and NSE measures the skill of the physics-constrained streamflow reconstruction. The subset result should therefore be interpreted as a change in model-inferred epsilon among catchments with adequate indirect reconstruction skill.

### Descriptive Daily-Mean Contrast

![Epsilon delta distribution by all days and flow regime]({fig_rel}/figure_02_delta_distribution.png)

For each catchment, epsilon change is defined as the post-change mean minus the pre-change mean:

```text
delta epsilon = mean epsilon in {year(post_start)}-{year(post_end)} - mean epsilon in {year(pre_start)}-{year(pre_end)}
```

Across all recession days:

```text
mean pre-change epsilon: {fmt_sci(result['mean_pre_epsilon'])}
mean post-change epsilon: {fmt_sci(result['mean_post_epsilon'])}
mean delta epsilon: {fmt_sci(result['mean_delta_epsilon'])}
median delta epsilon: {fmt_sci(result['median_delta_epsilon'])}
catchment share with negative delta epsilon: {fmt_pct(result['negative_delta_share'])}
```

{direction_note}

These values describe the unadjusted daily distribution and support the interactive CDF panels. They do not determine the primary era-shift class because unequal numbers of recession days and OOF model scale can affect raw daily means.

Flow-regime summaries use basin-specific observed-flow thresholds:

```text
low-flow epsilon:  recession days with observed Q <= each catchment's Q10
high-flow epsilon: recession days with observed Q >= each catchment's Q90
mid-flow epsilon:  Q10 < observed Q < Q90
```

{regime_line(regime_summary, "low")}
{regime_line(regime_summary, "mid")}
{regime_line(regime_summary, "high")}

Low-flow and high-flow epsilon are evaluated separately because recession behavior under the tails of the flow distribution can reflect different storage-release controls. Their mean relative changes are `{fmt_pct(low['mean_relative_delta_mean'])}` for low flow and `{fmt_pct(high['mean_relative_delta_mean'])}` for high flow. These flow-regime summaries should be read together with the median and quartile structure in the table, because outlier catchments can move the mean.

### Hydroclimate Structure

![Hydroclimate gradients of epsilon change]({fig_rel}/figure_03_hydroclimate_gradients.png)

The hydroclimate-gradient figure bins catchments into quartiles of precipitation, temperature, and aridity, then compares mean and median epsilon change within each bin. This checks whether the epsilon shift is a spatially random artifact or whether it aligns with background catchment climate.

The hydroclimate gradients should be read as descriptive evidence, not causal attribution. They show whether epsilon shifts align with background precipitation, temperature, and aridity structure, and they identify where more formal regression or hierarchical testing would be useful.

### Spatial Pattern

![Spatial distribution of epsilon change]({fig_rel}/figure_04_spatial_delta.png)

The spatial map shows catchment-level epsilon change as point locations. It is designed to reveal regional clustering that is hidden in the histogram and boxplot. Blue and red points mark opposite signs of epsilon change, so the map should be interpreted together with the catchment-level delta table:

```text
catchment-level table: {fig_rel}/epsilon_change_by_catchment.csv
flow-regime table:    {fig_rel}/epsilon_change_by_flow_regime.csv
```

The map highlights where post-1990 changes cluster spatially. The cross-catchment median change is `{fmt_sci(result['median_delta_epsilon'])}`.

## Method Summary

For each catchment, the model reads a 365-day context window of dynamic inputs plus static attributes. It predicts daily `epsilon_t`, `q_base_t`, and bounded AET parameters `alpha`, `LP`, and `gamma`. AET is computed inside the model from PET, soil moisture, LP, and gamma. Streamflow is then solved through the closed-form state-reset recession equation and supervised against observed streamflow on recession days.

The main differential equation is:

```text
dQ/dt = -epsilon * Q^2 - epsilon * alpha * AET * Q
```

The model is therefore an epsilon-core physics-informed LSTM that infers daily epsilon directly inside the recession equation.
"""
    args.summary_md.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.summary_md}")


if __name__ == "__main__":
    main()
