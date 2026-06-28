from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config, output_dir


EXPECTED_SUMMARY_COLUMNS = {
    "GCIN",
    "fold",
    "pre_epsilon_mean",
    "post_epsilon_mean",
    "delta_epsilon_mean",
    "pre_qobs_valid_days",
    "post_qobs_valid_days",
    "pre_n_recession_days_predicted",
    "post_n_recession_days_predicted",
}

EXPECTED_SIM_COLUMNS = {
    "GCIN",
    "date",
    "observed_Q_mmd",
    "simulated_Q_mmd",
    "epsilon_effective",
    "simulated_AET_mm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-label", type=str, default="full_crossfit_era5land_legacy_1950_2019")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() < 2:
        return np.nan
    obs = obs[valid]
    pred = pred[valid]
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom <= 0:
        return np.nan
    return float(1.0 - np.sum((obs - pred) ** 2) / denom)


def kge(obs: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() < 2:
        return np.nan
    obs = obs[valid]
    pred = pred[valid]
    obs_std = obs.std(ddof=1)
    pred_std = pred.std(ddof=1)
    obs_mean = obs.mean()
    pred_mean = pred.mean()
    if obs_std <= 0 or pred_std <= 0 or obs_mean == 0:
        return np.nan
    r = float(((obs - obs_mean) * (pred - pred_mean)).sum() / ((len(obs) - 1) * obs_std * pred_std))
    alpha = pred_std / obs_std
    beta = pred_mean / obs_mean
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def audit_fold(run_root: Path, fold: int) -> dict[str, object]:
    fold_dir = run_root / f"fold_{fold}"
    row: dict[str, object] = {"fold": fold, "fold_dir": str(fold_dir)}
    required = {
        "best_model": fold_dir / "best_model.pt",
        "metrics": fold_dir / "metrics.csv",
        "summary": fold_dir / "heldout_epsilon_change_summary.parquet",
        "simulation": fold_dir / "recession_day_simulations.parquet",
        "metadata": fold_dir / "run_metadata.json",
    }
    for key, path in required.items():
        row[f"has_{key}"] = path.exists()
        row[f"{key}_size_mb"] = round(path.stat().st_size / 1024 / 1024, 3) if path.exists() else np.nan
    if not all(path.exists() for path in required.values()):
        row["status"] = "incomplete"
        row["summary_missing_columns"] = "summary_file_missing"
        row["simulation_missing_columns"] = "simulation_file_missing"
        return row

    summary = pd.read_parquet(required["summary"])
    sim = pd.read_parquet(required["simulation"])
    metrics = pd.read_csv(required["metrics"])

    row["status"] = "complete"
    row["summary_rows"] = len(summary)
    row["summary_gcins"] = summary["GCIN"].nunique() if "GCIN" in summary else np.nan
    row["simulation_rows"] = len(sim)
    row["simulation_gcins"] = sim["GCIN"].nunique() if "GCIN" in sim else np.nan
    row["metrics_epochs"] = metrics["epoch"].nunique() if "epoch" in metrics else np.nan
    row["summary_missing_columns"] = ",".join(sorted(EXPECTED_SUMMARY_COLUMNS - set(summary.columns)))
    row["simulation_missing_columns"] = ",".join(sorted(EXPECTED_SIM_COLUMNS - set(sim.columns)))
    row["summary_any_nan_delta"] = bool(summary["delta_epsilon_mean"].isna().any()) if "delta_epsilon_mean" in summary else True
    row["simulation_any_nan_q"] = bool(sim[["observed_Q_mmd", "simulated_Q_mmd"]].isna().any().any())
    row["mean_delta_epsilon"] = float(summary["delta_epsilon_mean"].mean()) if "delta_epsilon_mean" in summary else np.nan
    row["median_delta_epsilon"] = float(summary["delta_epsilon_mean"].median()) if "delta_epsilon_mean" in summary else np.nan
    row["pooled_nse"] = nse(sim["observed_Q_mmd"].to_numpy(float), sim["simulated_Q_mmd"].to_numpy(float))
    row["pooled_kge"] = kge(sim["observed_Q_mmd"].to_numpy(float), sim["simulated_Q_mmd"].to_numpy(float))
    by_catchment = []
    for gcin, group in sim.groupby("GCIN", observed=True):
        obs = group["observed_Q_mmd"].to_numpy(float)
        pred = group["simulated_Q_mmd"].to_numpy(float)
        by_catchment.append({"GCIN": int(gcin), "nse": nse(obs, pred), "kge": kge(obs, pred)})
    catchment_skill = pd.DataFrame(by_catchment)
    row["mean_catchment_nse"] = float(catchment_skill["nse"].mean())
    row["median_catchment_nse"] = float(catchment_skill["nse"].median())
    row["p10_catchment_nse"] = float(catchment_skill["nse"].quantile(0.10))
    row["p90_catchment_nse"] = float(catchment_skill["nse"].quantile(0.90))
    row["mean_catchment_kge"] = float(catchment_skill["kge"].mean())
    row["median_catchment_kge"] = float(catchment_skill["kge"].median())
    row["p10_catchment_kge"] = float(catchment_skill["kge"].quantile(0.10))
    row["p90_catchment_kge"] = float(catchment_skill["kge"].quantile(0.90))
    return row


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_root = output_dir(cfg) / args.run_label
    rows = []
    for fold in range(int(cfg["splits"]["n_folds"])):
        row = audit_fold(run_root, fold)
        rows.append(row)
        print(
            f"audited fold {fold}: status={row.get('status')} "
            f"median_nse={row.get('median_catchment_nse', np.nan)}",
            flush=True,
        )
    audit = pd.DataFrame(rows)

    out = args.out or (output_dir(cfg) / args.run_label / "production_audit.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out, index=False)

    print(audit.to_string(index=False))
    complete = (audit["status"] == "complete").all()
    no_missing_cols = (audit["summary_missing_columns"].fillna("") == "").all() and (
        audit["simulation_missing_columns"].fillna("") == ""
    ).all()
    if not complete or not no_missing_cols:
        raise SystemExit(1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
