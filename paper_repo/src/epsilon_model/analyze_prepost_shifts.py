from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from analyze_gq_attribution import label_simulations
from config import load_config, output_dir


REGIMES = ("all", "low", "high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("paper_repo/configs/epsilon_experiment_pure_gcin_1950_2019.yaml"),
    )
    parser.add_argument("--run-label", default="temporal_crossfit_1990")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--break-year", type=int, default=1990)
    parser.add_argument("--min-days-per-year", type=int, default=3)
    parser.add_argument("--min-years-per-period", type=int, default=10)
    parser.add_argument("--min-paired-folds", type=int, default=1)
    parser.add_argument("--min-identifying-years-per-period", type=int, default=5)
    parser.add_argument("--hac-lag-years", type=int, default=1)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    return parser.parse_args()


def bh_fdr(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return pd.Series(adjusted, index=p_values.index)
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def annual_epsilon(sim: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for regime in REGIMES:
        data = sim if regime == "all" else sim[sim["regime"] == regime]
        annual = (
            data.groupby(["GCIN", "year", "fold"], observed=True)
            .agg(n_days=("epsilon_effective", "size"), value=("epsilon_effective", "median"))
            .reset_index()
        )
        annual["regime"] = regime
        frames.append(annual)
    return pd.concat(frames, ignore_index=True)


def fold_design(folds: np.ndarray, post: np.ndarray) -> np.ndarray:
    unique_folds = np.unique(folds)
    dummies = [(folds == fold).astype(float) for fold in unique_folds[1:]]
    return np.column_stack([np.ones(len(folds), dtype=float), post.astype(float), *dummies])


def hac_covariance(
    design: np.ndarray,
    residuals: np.ndarray,
    years: np.ndarray,
    max_lag_years: int,
) -> np.ndarray:
    n_obs, n_params = design.shape
    bread = np.linalg.pinv(design.T @ design)
    scores = design * residuals[:, None]
    meat = scores.T @ scores
    by_year: dict[int, list[int]] = {}
    for index, year in enumerate(years):
        by_year.setdefault(int(year), []).append(index)
    for lag in range(1, max_lag_years + 1):
        weight = 1.0 - lag / float(max_lag_years + 1)
        for year, left_indices in by_year.items():
            right_indices = by_year.get(year + lag, [])
            for left in left_indices:
                for right in right_indices:
                    meat += weight * (
                        np.outer(scores[left], scores[right])
                        + np.outer(scores[right], scores[left])
                    )
    correction = n_obs / max(n_obs - n_params, 1)
    return correction * bread @ meat @ bread


def fit_shift(
    series: pd.DataFrame,
    break_year: int,
    min_years_per_period: int,
    min_paired_folds: int,
    min_identifying_years_per_period: int,
    hac_lag_years: int,
) -> dict[str, float | int]:
    series = series.sort_values(["year", "fold"])
    years = series["year"].to_numpy(int)
    folds = series["fold"].to_numpy(int)
    values = series["value"].to_numpy(float)
    n_days = series["n_days"].to_numpy(int)
    valid = np.isfinite(values) & (values > 0)
    years = years[valid]
    folds = folds[valid]
    values = values[valid]
    n_days = n_days[valid]
    post = (years > break_year).astype(float)
    pre_years = int(np.unique(years[post == 0]).size)
    post_years = int(np.unique(years[post == 1]).size)
    paired_folds = int(
        sum(np.unique(post[folds == fold]).size == 2 for fold in np.unique(folds))
    )
    paired_fold_ids = {
        fold for fold in np.unique(folds) if np.unique(post[folds == fold]).size == 2
    }
    identifying = np.isin(folds, list(paired_fold_ids))
    identifying_pre_years = int(np.unique(years[identifying & (post == 0)]).size)
    identifying_post_years = int(np.unique(years[identifying & (post == 1)]).size)
    result: dict[str, float | int] = {
        "pre_years": pre_years,
        "post_years": post_years,
        "paired_folds": paired_folds,
        "identifying_pre_years": identifying_pre_years,
        "identifying_post_years": identifying_post_years,
        "n_years": int(np.unique(years).size),
        "n_recession_days": int(n_days.sum()),
        "shift_log": np.nan,
        "shift_pct": np.nan,
        "shift_se_log": np.nan,
        "shift_ci_low_pct": np.nan,
        "shift_ci_high_pct": np.nan,
        "t_statistic": np.nan,
        "p_value": np.nan,
    }
    if (
        pre_years < min_years_per_period
        or post_years < min_years_per_period
        or paired_folds < min_paired_folds
        or identifying_pre_years < min_identifying_years_per_period
        or identifying_post_years < min_identifying_years_per_period
    ):
        return result

    design = fold_design(folds, post)
    n_obs, n_params = design.shape
    if n_obs <= n_params or np.linalg.matrix_rank(design) < n_params:
        return result
    response = np.log(values)
    bread = np.linalg.pinv(design.T @ design)
    coefficients = bread @ design.T @ response
    residuals = response - design @ coefficients
    covariance = hac_covariance(design, residuals, years, hac_lag_years)
    variance = float(covariance[1, 1])
    if not np.isfinite(variance) or variance <= 0:
        return result

    shift_log = float(coefficients[1])
    standard_error = float(np.sqrt(variance))
    degrees_freedom = int(n_obs - n_params)
    t_statistic = shift_log / standard_error
    p_value = float(2.0 * student_t.sf(abs(t_statistic), degrees_freedom))
    critical = float(student_t.ppf(0.975, degrees_freedom))
    ci_low = shift_log - critical * standard_error
    ci_high = shift_log + critical * standard_error
    result.update(
        {
            "shift_log": shift_log,
            "shift_pct": float(100.0 * np.expm1(shift_log)),
            "shift_se_log": standard_error,
            "shift_ci_low_pct": float(100.0 * np.expm1(ci_low)),
            "shift_ci_high_pct": float(100.0 * np.expm1(ci_high)),
            "t_statistic": float(t_statistic),
            "p_value": p_value,
        }
    )
    return result


def summarize_shifts(
    annual: pd.DataFrame,
    break_year: int,
    min_days_per_year: int,
    min_years_per_period: int,
    min_paired_folds: int,
    min_identifying_years_per_period: int,
    hac_lag_years: int,
    fdr_alpha: float,
) -> pd.DataFrame:
    eligible_years = annual[annual["n_days"] >= min_days_per_year]
    rows: list[dict[str, object]] = []
    for (gcin, regime), group in eligible_years.groupby(
        ["GCIN", "regime"], observed=True, sort=True
    ):
        row: dict[str, object] = {"GCIN": int(gcin), "regime": str(regime)}
        row.update(
            fit_shift(
                group,
                break_year,
                min_years_per_period,
                min_paired_folds,
                min_identifying_years_per_period,
                hac_lag_years,
            )
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    result["q_value"] = np.nan
    for _regime, indices in result.groupby("regime", observed=True).groups.items():
        result.loc[indices, "q_value"] = bh_fdr(result.loc[indices, "p_value"])
    result["significant"] = result["q_value"] < fdr_alpha
    result["shift_class"] = "unresolved"
    result.loc[result["significant"] & (result["shift_log"] > 0), "shift_class"] = "increase"
    result.loc[result["significant"] & (result["shift_log"] < 0), "shift_class"] = "decrease"
    result.loc[result["p_value"].isna(), "shift_class"] = "insufficient"
    result["break_year"] = int(break_year)
    result["min_days_per_year"] = int(min_days_per_year)
    result["min_years_per_period"] = int(min_years_per_period)
    result["min_paired_folds"] = int(min_paired_folds)
    result["min_identifying_years_per_period"] = int(min_identifying_years_per_period)
    result["hac_lag_years"] = int(hac_lag_years)
    return result


def build_wide(long: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in long.columns:
        if column in {"GCIN", "regime"}:
            continue
        renamed[column] = (
            f"epsilon_{column}" if column.startswith("shift_") else f"epsilon_shift_{column}"
        )
    return long.rename(columns=renamed)


def sensitivity_summary(
    annual: pd.DataFrame,
    args: argparse.Namespace,
    primary: pd.DataFrame,
) -> pd.DataFrame:
    scenarios = (
        ("primary_1990_d3", args.break_year, args.min_days_per_year),
        ("days_min_1", args.break_year, 1),
        ("days_min_5", args.break_year, 5),
        ("break_1985", 1985, args.min_days_per_year),
        ("break_1995", 1995, args.min_days_per_year),
    )
    primary_values = primary[["GCIN", "regime", "shift_pct", "shift_class"]].rename(
        columns={"shift_pct": "primary_shift_pct", "shift_class": "primary_shift_class"}
    )
    rows: list[dict[str, object]] = []
    for scenario, break_year, min_days in scenarios:
        result = primary if scenario == "primary_1990_d3" else summarize_shifts(
            annual,
            break_year,
            min_days,
            args.min_years_per_period,
            args.min_paired_folds,
            args.min_identifying_years_per_period,
            args.hac_lag_years,
            args.fdr_alpha,
        )
        joined = result.merge(primary_values, on=["GCIN", "regime"], how="left")
        for regime, group in joined.groupby("regime", observed=True):
            eligible = group[group["p_value"].notna()].copy()
            overlap = eligible[
                eligible["primary_shift_pct"].notna()
                & np.isfinite(eligible["shift_pct"])
            ]
            correlation = np.nan
            sign_agreement = np.nan
            if len(overlap) >= 2:
                correlation = float(
                    np.corrcoef(overlap["shift_pct"], overlap["primary_shift_pct"])[0, 1]
                )
                sign_agreement = float(
                    np.mean(np.sign(overlap["shift_pct"]) == np.sign(overlap["primary_shift_pct"]))
                )
            counts = eligible["shift_class"].value_counts()
            rows.append(
                {
                    "scenario": scenario,
                    "break_year": break_year,
                    "min_days_per_year": min_days,
                    "min_years_per_period": args.min_years_per_period,
                    "min_paired_folds": args.min_paired_folds,
                    "min_identifying_years_per_period": args.min_identifying_years_per_period,
                    "regime": regime,
                    "eligible": len(eligible),
                    "increase": int(counts.get("increase", 0)),
                    "decrease": int(counts.get("decrease", 0)),
                    "unresolved": int(counts.get("unresolved", 0)),
                    "median_shift_pct": float(eligible["shift_pct"].median()) if len(eligible) else np.nan,
                    "primary_overlap": len(overlap),
                    "effect_correlation_with_primary": correlation,
                    "direction_agreement_with_primary": sign_agreement,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_root = args.run_root or (output_dir(cfg) / args.run_label)
    analysis_dir = run_root / "analysis"
    sim_path = analysis_dir / "oof_recession_day_simulations.parquet"
    sim = pd.read_parquet(
        sim_path,
        columns=["GCIN", "date", "observed_Q_mmd", "simulated_Q_mmd", "epsilon_effective", "fold"],
    )
    sim = label_simulations(sim, cfg)
    sim["year"] = sim["date"].dt.year.astype(int)
    annual = annual_epsilon(sim)
    primary = summarize_shifts(
        annual,
        args.break_year,
        args.min_days_per_year,
        args.min_years_per_period,
        args.min_paired_folds,
        args.min_identifying_years_per_period,
        args.hac_lag_years,
        args.fdr_alpha,
    )
    wide = build_wide(primary)
    sensitivity = sensitivity_summary(annual, args, primary)

    primary.to_parquet(analysis_dir / "prepost_shifts_long.parquet", index=False)
    primary.to_csv(analysis_dir / "prepost_shifts_long.csv", index=False)
    wide.to_parquet(analysis_dir / "prepost_shifts_by_catchment.parquet", index=False)
    wide.to_csv(analysis_dir / "prepost_shifts_by_catchment.csv", index=False)
    sensitivity.to_csv(analysis_dir / "prepost_shift_sensitivity_summary.csv", index=False)

    print("primary fold-adjusted pre/post epsilon shift", flush=True)
    print(
        primary.groupby(["regime", "shift_class"], observed=True)
        .size()
        .unstack(fill_value=0)
        .to_string(),
        flush=True,
    )
    print("sensitivity summary", flush=True)
    print(sensitivity.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
