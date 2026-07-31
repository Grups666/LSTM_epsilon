from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, theilslopes

from analyze_gq_attribution import label_simulations
from config import load_config, output_dir


REGIMES = ("all", "low", "high")
VARIABLES = {
    "epsilon": "epsilon_effective",
    "gq": "gq_effective",
    "qsim": "simulated_Q_mmd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("paper_repo/configs/epsilon_experiment_pure_gcin_1950_2019.yaml"),
    )
    parser.add_argument("--run-label", default="temporal_crossfit_1990")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--min-days-per-year", type=int, default=5)
    parser.add_argument("--min-years", type=int, default=20)
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


def fold_center(values: np.ndarray, folds: np.ndarray) -> np.ndarray:
    centered = values.copy()
    overall = float(np.mean(values))
    for fold in np.unique(folds):
        mask = folds == fold
        centered[mask] = values[mask] - float(np.mean(values[mask])) + overall
    return centered


def lag1_correlation(values: np.ndarray) -> float:
    if len(values) < 3:
        return np.nan
    lag_x = values[:-1] - float(np.mean(values[:-1]))
    lag_y = values[1:] - float(np.mean(values[1:]))
    denominator = float(np.sqrt(np.sum(lag_x * lag_x) * np.sum(lag_y * lag_y)))
    return float(np.sum(lag_x * lag_y) / denominator) if denominator > 0 else np.nan


def trend_stats(series: pd.DataFrame, min_years: int) -> dict[str, float | int]:
    series = series.sort_values("year")
    years = series["year"].to_numpy(float)
    values = series["value"].to_numpy(float)
    folds = series["fold"].to_numpy(int)
    valid = np.isfinite(years) & np.isfinite(values) & (values > 0)
    years = years[valid]
    values = values[valid]
    folds = folds[valid]
    result: dict[str, float | int] = {
        "n_years": int(len(years)),
        "start_year": int(years.min()) if len(years) else np.nan,
        "end_year": int(years.max()) if len(years) else np.nan,
        "median_annual_value": float(np.median(values)) if len(values) else np.nan,
        "raw_slope_pct_decade": np.nan,
        "slope_pct_decade": np.nan,
        "slope_log_per_year": np.nan,
        "slope_ci_low_pct_decade": np.nan,
        "slope_ci_high_pct_decade": np.nan,
        "kendall_tau": np.nan,
        "raw_kendall_tau": np.nan,
        "raw_p_value": np.nan,
        "p_value": np.nan,
        "lag1_autocorrelation": np.nan,
    }
    if len(years) < min_years or len(np.unique(years)) < min_years:
        return result

    logs = np.log(values)
    adjusted = fold_center(logs, folds)
    slope, _intercept, low, high = theilslopes(adjusted, years, alpha=0.95)
    raw_slope = float(theilslopes(logs, years, alpha=0.95)[0])
    raw_tau, raw_p_value = kendalltau(years, adjusted, nan_policy="omit")

    detrended = adjusted - slope * years
    lag1 = lag1_correlation(detrended)
    if np.isfinite(lag1) and len(years) >= 4:
        prewhitened = detrended[1:] - lag1 * detrended[:-1] + slope * years[1:]
        test_years = years[1:]
    else:
        prewhitened = adjusted
        test_years = years
    tau, p_value = kendalltau(test_years, prewhitened, nan_policy="omit")
    result.update(
        {
            "raw_slope_pct_decade": float(100.0 * np.expm1(raw_slope * 10.0)),
            "slope_pct_decade": float(100.0 * np.expm1(slope * 10.0)),
            "slope_log_per_year": float(slope),
            "slope_ci_low_pct_decade": float(100.0 * np.expm1(low * 10.0)),
            "slope_ci_high_pct_decade": float(100.0 * np.expm1(high * 10.0)),
            "kendall_tau": float(tau),
            "raw_kendall_tau": float(raw_tau),
            "raw_p_value": float(raw_p_value),
            "p_value": float(p_value),
            "lag1_autocorrelation": lag1,
        }
    )
    return result


def annual_series(sim: pd.DataFrame, min_days: int) -> pd.DataFrame:
    frames = []
    for regime in REGIMES:
        data = sim if regime == "all" else sim[sim["regime"] == regime]
        annual = (
            data.groupby(["GCIN", "year", "fold"], observed=True)
            .agg(
                n_days=("epsilon_effective", "size"),
                epsilon=("epsilon_effective", "median"),
                gq=("gq_effective", "median"),
                qsim=("simulated_Q_mmd", "median"),
            )
            .reset_index()
        )
        annual = annual[annual["n_days"] >= min_days].copy()
        annual = annual.melt(
            id_vars=["GCIN", "year", "fold", "n_days"],
            value_vars=list(VARIABLES),
            var_name="variable",
            value_name="value",
        )
        annual["regime"] = regime
        frames.append(annual)
    return pd.concat(frames, ignore_index=True)


def summarize_trends(annual: pd.DataFrame, min_years: int, fdr_alpha: float) -> pd.DataFrame:
    rows = []
    for (gcin, regime, variable), group in annual.groupby(
        ["GCIN", "regime", "variable"], observed=True, sort=True
    ):
        row: dict[str, object] = {
            "GCIN": int(gcin),
            "regime": str(regime),
            "variable": str(variable),
        }
        row.update(trend_stats(group, min_years))
        rows.append(row)
    long = pd.DataFrame(rows)
    long["q_value"] = np.nan
    for (_regime, _variable), indices in long.groupby(["regime", "variable"], observed=True).groups.items():
        long.loc[indices, "q_value"] = bh_fdr(long.loc[indices, "p_value"])
    long["significant"] = long["q_value"] < fdr_alpha
    long["trend_class"] = "no_significant_trend"
    long.loc[long["significant"] & (long["slope_log_per_year"] > 0), "trend_class"] = "increase"
    long.loc[long["significant"] & (long["slope_log_per_year"] < 0), "trend_class"] = "decrease"
    long.loc[long["n_years"] < min_years, "trend_class"] = "insufficient"
    return long


def build_wide(long: pd.DataFrame, fdr_alpha: float) -> pd.DataFrame:
    fields = [
        "n_years",
        "start_year",
        "end_year",
        "median_annual_value",
        "raw_slope_pct_decade",
        "slope_pct_decade",
        "slope_log_per_year",
        "slope_ci_low_pct_decade",
        "slope_ci_high_pct_decade",
        "kendall_tau",
        "raw_kendall_tau",
        "raw_p_value",
        "p_value",
        "q_value",
        "lag1_autocorrelation",
        "significant",
        "trend_class",
    ]
    wide = long.pivot(index=["GCIN", "regime"], columns="variable", values=fields)
    wide.columns = [f"{variable}_{field}" for field, variable in wide.columns]
    wide = wide.reset_index()

    def driver(row: pd.Series) -> str:
        epsilon_q = float(row.get("epsilon_q_value", np.nan))
        if not np.isfinite(epsilon_q) or epsilon_q >= fdr_alpha:
            return "nonsignificant"
        gq_sig = bool(row.get("gq_significant", False))
        q_sig = bool(row.get("qsim_significant", False))
        if gq_sig and q_sig:
            return "combined"
        if gq_sig:
            return "gq"
        if q_sig:
            return "q"
        return "unresolved"

    wide["trend_driver"] = wide.apply(driver, axis=1)
    return wide


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_root = args.run_root or (output_dir(cfg) / args.run_label)
    sim_path = run_root / "analysis" / "oof_recession_day_simulations.parquet"
    sim = pd.read_parquet(
        sim_path,
        columns=["GCIN", "date", "observed_Q_mmd", "simulated_Q_mmd", "epsilon_effective", "fold"],
    )
    sim = label_simulations(sim, cfg)
    sim["year"] = sim["date"].dt.year.astype(int)
    annual = annual_series(sim, args.min_days_per_year)
    long = summarize_trends(annual, args.min_years, args.fdr_alpha)
    wide = build_wide(long, args.fdr_alpha)

    analysis_dir = run_root / "analysis"
    annual.to_parquet(analysis_dir / "annual_epsilon_gq_q_by_regime.parquet", index=False)
    long.to_csv(analysis_dir / "continuous_trends_long.csv", index=False)
    long.to_parquet(analysis_dir / "continuous_trends_long.parquet", index=False)
    wide.to_csv(analysis_dir / "continuous_trends_by_catchment.csv", index=False)
    wide.to_parquet(analysis_dir / "continuous_trends_by_catchment.parquet", index=False)

    epsilon = long[long["variable"] == "epsilon"]
    print(f"annual catchment-regime-variable rows: {len(annual):,}")
    print(f"trend series: {len(long):,}")
    print(epsilon.groupby(["regime", "trend_class"], observed=True).size().unstack(fill_value=0).to_string())
    print("significance-based attribution")
    print(wide.groupby(["regime", "trend_driver"], observed=True).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
