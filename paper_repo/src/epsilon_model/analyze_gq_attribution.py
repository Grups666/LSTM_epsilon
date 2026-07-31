from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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
    return parser.parse_args()


def contribution_driver(gq_component: float, q_component: float) -> str:
    if not np.isfinite(gq_component) or not np.isfinite(q_component):
        return "insufficient"
    if gq_component * q_component < 0:
        return "offsetting"
    total = abs(gq_component) + abs(q_component)
    if total <= np.finfo(float).eps:
        return "combined"
    gq_share = abs(gq_component) / total
    if gq_share >= 2.0 / 3.0:
        return "gq"
    if gq_share <= 1.0 / 3.0:
        return "q"
    return "combined"


def geometric_summary(values: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(values, errors="coerce").to_numpy(float)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return np.nan, np.nan
    mean_log = float(np.mean(np.log(values)))
    return float(np.exp(mean_log)), mean_log


def label_simulations(sim: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    sim = sim.copy()
    sim["date"] = pd.to_datetime(sim["date"])
    pre_start, pre_end = pd.to_datetime(cfg["data"]["pre_window"])
    post_start, post_end = pd.to_datetime(cfg["data"]["post_window"])
    sim["period"] = pd.NA
    sim.loc[sim["date"].between(pre_start, pre_end), "period"] = "pre"
    sim.loc[sim["date"].between(post_start, post_end), "period"] = "post"
    sim = sim[sim["period"].isin(["pre", "post"])].copy()

    qtiles = sim.groupby("GCIN", observed=True)["observed_Q_mmd"].quantile([0.1, 0.9]).unstack()
    qtiles.columns = ["q10", "q90"]
    sim = sim.merge(qtiles, on="GCIN", how="left")
    sim["regime"] = "mid"
    sim.loc[sim["observed_Q_mmd"] <= sim["q10"], "regime"] = "low"
    sim.loc[sim["observed_Q_mmd"] >= sim["q90"], "regime"] = "high"
    sim["gq_effective"] = sim["epsilon_effective"] * sim["simulated_Q_mmd"]
    return sim


def summarize(sim: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gcin, basin in sim.groupby("GCIN", observed=True, sort=True):
        for regime in REGIMES:
            data = basin if regime == "all" else basin[basin["regime"] == regime]
            row: dict[str, object] = {"GCIN": int(gcin), "regime": regime}
            period_logs: dict[str, dict[str, float]] = {}
            for period in ("pre", "post"):
                part = data[data["period"] == period]
                period_logs[period] = {}
                row[f"{period}_n_days"] = int(len(part))
                for variable, column in (
                    ("epsilon", "epsilon_effective"),
                    ("gq", "gq_effective"),
                    ("qsim", "simulated_Q_mmd"),
                ):
                    geometric_mean, mean_log = geometric_summary(part[column])
                    row[f"{period}_{variable}_geomean"] = geometric_mean
                    row[f"{period}_{variable}_median"] = float(part[column].median()) if len(part) else np.nan
                    period_logs[period][variable] = mean_log

            for variable in ("epsilon", "gq", "qsim"):
                delta_log = period_logs["post"][variable] - period_logs["pre"][variable]
                row[f"{variable}_delta_log"] = delta_log
                row[f"{variable}_change_pct"] = float(100.0 * np.expm1(delta_log))

            gq_component = float(row["gq_delta_log"])
            q_component = -float(row["qsim_delta_log"])
            epsilon_change = float(row["epsilon_delta_log"])
            closure_error = epsilon_change - (gq_component + q_component)
            total_component = abs(gq_component) + abs(q_component)
            row["gq_component_log"] = gq_component
            row["q_component_log"] = q_component
            row["closure_error_log"] = closure_error
            row["gq_absolute_share"] = abs(gq_component) / total_component if total_component > 0 else 0.5
            row["offsetting"] = bool(gq_component * q_component < 0)
            row["driver"] = contribution_driver(gq_component, q_component)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_root = args.run_root or (output_dir(cfg) / args.run_label)
    sim_path = run_root / "analysis" / "oof_recession_day_simulations.parquet"
    columns = [
        "GCIN",
        "date",
        "observed_Q_mmd",
        "simulated_Q_mmd",
        "epsilon_effective",
        "fold",
    ]
    sim = pd.read_parquet(sim_path, columns=columns)
    sim = label_simulations(sim, cfg)
    result = summarize(sim)
    analysis_dir = run_root / "analysis"
    csv_path = analysis_dir / "gq_q_attribution_by_catchment.csv"
    parquet_path = analysis_dir / "gq_q_attribution_by_catchment.parquet"
    result.to_csv(csv_path, index=False)
    result.to_parquet(parquet_path, index=False)

    finite_closure = result["closure_error_log"].abs().dropna()
    print(f"daily recession rows: {len(sim):,}")
    print(f"catchment-regime summaries: {len(result):,}")
    print(f"maximum log-space closure error: {finite_closure.max():.3e}")
    print(result.groupby(["regime", "driver"], observed=True).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
