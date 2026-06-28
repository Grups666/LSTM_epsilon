from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("paper_repo/configs/epsilon_experiment_era5land_legacy_1950_2019.yaml"))
    parser.add_argument("--figures-dir", type=Path, default=Path("_private/results/paper_figures"))
    parser.add_argument("--summary-md", type=Path, default=Path("paper_repo/docs/SUMMARY.md"))
    parser.add_argument("--run-label", type=str, default="full_crossfit_era5land_legacy_1950_2019")
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
    return Path(target_resolved).relative_to(source_parent).as_posix()


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
    required = [
        fig / "result_summary.csv",
        fig / "model_skill_summary.csv",
        fig / "epsilon_change_by_flow_regime.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required result files:\n" + "\n".join(missing))

    result = scalar_table(fig / "result_summary.csv")
    skill = pd.read_csv(fig / "model_skill_summary.csv")
    regime = pd.read_csv(fig / "epsilon_change_by_flow_regime.csv")
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
    low = regime_summary[regime_summary["regime"] == "low"].iloc[0]
    high = regime_summary[regime_summary["regime"] == "high"].iloc[0]
    fig_rel = markdown_relative_path(args.summary_md, fig)

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
GCIN, date, precipitation_mmd, temperature_C, pet_mmd, SM_%,
streamflow_mmd, observed_AET_mm
```

## Current Run

The production run is:

```text
run label: {args.run_label}
cluster/fold count: {n_folds}
batch_size: {batch_size}
epochs: {epochs}
```

The training follows Ara's `LSTM-epsilon` model structure. It trains one model per catchment cluster, then infers daily recession epsilon for the same cluster. The five cluster outputs are aggregated after all runs finish.

Cold-temperature filtering is enabled:

```text
remove recession days with daily mean temperature <= 0 deg C
```

The cold-temperature filter threshold is `{cold_filter} deg C`; this removes recession days where the daily mean temperature is at or below the threshold.

## Results

The final cross-fitted analysis covers `{int(result['n_valid_delta']):,}` catchments and `{int(result['n_recession_simulation_days']):,}` recession-day simulations.

### Model Skill

![Training loss]({fig_rel}/figure_01_training_loss.png)

The model was trained separately for five catchment clusters and evaluated on recession-day streamflow. Model skill is summarized by catchment-level NSE/KGE first, then by the median across basins, so large high-flow basins do not dominate the diagnostic.

```text
median catchment NSE: {skill_all['median_catchment_nse']:.3f}
median catchment KGE: {skill_all['median_catchment_kge']:.3f}
pooled NSE, supplementary: {skill_all['pooled_nse']:.3f}
pooled KGE, supplementary: {skill_all['pooled_kge']:.3f}
```

The catchment-median NSE is close to zero, while the pooled NSE is higher because all recession-day records are stacked before scoring. This gap indicates that the inferred epsilon contrast is more stable as a cross-fitted recession-parameter analysis than as a basin-by-basin streamflow simulator.

### Epsilon Shift

![Epsilon delta distribution by all days and flow regime]({fig_rel}/figure_02_delta_distribution.png)

For each catchment, epsilon change is defined as the post-change mean minus the pre-change mean:

```text
delta epsilon = mean epsilon in {year(post_start)}-{year(post_end)} - mean epsilon in {year(pre_start)}-{year(pre_end)}
```

Across all recession days:

```text
mean delta epsilon: {fmt_sci(result['mean_delta_epsilon'])}
median delta epsilon: {fmt_sci(result['median_delta_epsilon'])}
catchment share with negative delta epsilon: {fmt_pct(result['negative_delta_share'])}
```

The all-catchment mean is negative and the median shift is also slightly negative. The absolute median is small, so the central tendency is a weak downward shift rather than a large regime-wide displacement. The distribution remains heterogeneous: `{fmt_pct(result['negative_delta_share'])}` of valid catchments show negative epsilon change, while a smaller set of catchments has positive shifts.

Flow-regime summaries use basin-specific observed-flow thresholds:

```text
low-flow epsilon:  recession days with observed Q <= each catchment's Q10
high-flow epsilon: recession days with observed Q >= each catchment's Q90
mid-flow epsilon:  Q10 < observed Q < Q90
```

{regime_line(regime_summary, "low")}
{regime_line(regime_summary, "mid")}
{regime_line(regime_summary, "high")}

Low-flow and high-flow epsilon are evaluated separately because recession behavior under the tails of the flow distribution can reflect different storage-release controls. Their mean relative changes are `{fmt_pct(low['mean_relative_delta_mean'])}` for low flow and `{fmt_pct(high['mean_relative_delta_mean'])}` for high flow. These flow-regime summaries should be read together with the median and quartile structure in the table, because outlier basins can move the mean.

### Hydroclimate Structure

![Hydroclimate gradients of epsilon change]({fig_rel}/figure_03_hydroclimate_gradients.png)

The hydroclimate-gradient figure bins catchments into quartiles of precipitation, temperature, and aridity, then compares mean and median epsilon change within each bin. This checks whether the epsilon shift is a spatially random artifact or whether it aligns with background catchment climate.

The current result should be read as a first-order gradient analysis rather than a causal attribution test. The median changes remain close to zero compared with the mean changes, so the hydroclimate signal is likely influenced by a subset of catchments with large positive or negative deltas. The next statistical step is to test these gradients with robust regression or hierarchical models rather than relying on quartile plots alone.

### Spatial Pattern

![Spatial distribution of epsilon change]({fig_rel}/figure_04_spatial_delta.png)

The spatial map shows catchment-level epsilon change as point locations. It is designed to reveal regional clustering that is hidden in the histogram and boxplot. Blue and red points mark opposite signs of epsilon change, so the map should be interpreted together with the catchment-level delta table:

```text
catchment-level table: {fig_rel}/epsilon_change_by_catchment.csv
flow-regime table:    {fig_rel}/epsilon_change_by_flow_regime.csv
```

The map highlights heterogeneous post-1990 changes: the median change is small and negative, about `{fmt_sci(result['median_delta_epsilon'])}`.

## Method Summary

For each catchment, the model reads a 365-day context window of dynamic inputs plus static attributes. It predicts daily `epsilon_t`, `q_base_t`, and bounded AET parameters `alpha`, `LP`, and `gamma`. AET is computed inside the model from PET, soil moisture, LP, and gamma. Streamflow is then solved through the closed-form state-reset recession equation and supervised against observed streamflow on recession days.

The main differential equation is:

```text
dQ/dt = -epsilon * Q^2 - epsilon * alpha * AET * Q
```

The model is therefore an epsilon-core physics-informed LSTM that infers daily epsilon directly inside the recession equation.
"""
    args.summary_md.write_text(text, encoding="utf-8")
    print(f"wrote {args.summary_md}")


if __name__ == "__main__":
    main()
