from __future__ import annotations

import argparse
import hashlib
import json
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
    parser.add_argument("--run-label", type=str, default="crossfit_1990")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def verify_run_manifest(run_root: Path) -> None:
    manifest_path = run_root / "production_run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing production run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    mismatches = []
    for name, record in manifest.get("files", {}).items():
        path = Path(record["path"])
        if not path.exists():
            mismatches.append(f"{name}: missing {path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        if digest != str(record["sha256"]).upper():
            mismatches.append(f"{name}: expected {record['sha256']} got {digest}")
    if mismatches:
        raise RuntimeError("Production inputs changed during the run:\n" + "\n".join(mismatches))


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


def add_period_skill(row: dict[str, object], sim: pd.DataFrame, period: str) -> None:
    row[f"{period}_pooled_nse"] = nse(
        sim["observed_Q_mmd"].to_numpy(float), sim["simulated_Q_mmd"].to_numpy(float)
    )
    row[f"{period}_pooled_kge"] = kge(
        sim["observed_Q_mmd"].to_numpy(float), sim["simulated_Q_mmd"].to_numpy(float)
    )
    catchment_skill = []
    for gcin, group in sim.groupby("GCIN", observed=True):
        obs = group["observed_Q_mmd"].to_numpy(float)
        pred = group["simulated_Q_mmd"].to_numpy(float)
        catchment_skill.append({"GCIN": int(gcin), "nse": nse(obs, pred), "kge": kge(obs, pred)})
    skill = pd.DataFrame(catchment_skill, columns=["GCIN", "nse", "kge"])
    for metric in ("nse", "kge"):
        row[f"{period}_mean_catchment_{metric}"] = float(skill[metric].mean())
        row[f"{period}_median_catchment_{metric}"] = float(skill[metric].median())
        row[f"{period}_p10_catchment_{metric}"] = float(skill[metric].quantile(0.10))
        row[f"{period}_p90_catchment_{metric}"] = float(skill[metric].quantile(0.90))


def audit_fold(
    run_root: Path,
    fold: int,
    expected_test_gcins: set[int],
    pre_window: tuple[pd.Timestamp, pd.Timestamp],
    post_window: tuple[pd.Timestamp, pd.Timestamp],
) -> dict[str, object]:
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
    validation_columns = {"validation_total", "validation_median_nse", "validation_pooled_nse"}
    row["metrics_has_validation"] = validation_columns.issubset(metrics.columns)
    row["best_validation_median_nse"] = (
        float(metrics["validation_median_nse"].max()) if "validation_median_nse" in metrics else np.nan
    )
    row["summary_missing_columns"] = ",".join(sorted(EXPECTED_SUMMARY_COLUMNS - set(summary.columns)))
    row["simulation_missing_columns"] = ",".join(sorted(EXPECTED_SIM_COLUMNS - set(sim.columns)))
    summary_gcins = set(summary["GCIN"].dropna().astype(int))
    simulation_gcins = set(sim["GCIN"].dropna().astype(int))
    row["expected_test_gcins"] = len(expected_test_gcins)
    row["missing_summary_gcins"] = len(expected_test_gcins - summary_gcins)
    row["missing_simulation_gcins"] = len(expected_test_gcins - simulation_gcins)
    row["unexpected_summary_gcins"] = len(summary_gcins - expected_test_gcins)
    row["unexpected_simulation_gcins"] = len(simulation_gcins - expected_test_gcins)
    if "delta_epsilon_mean" in summary:
        delta_values = pd.to_numeric(summary["delta_epsilon_mean"], errors="coerce").to_numpy(float)
        row["summary_valid_delta_count"] = int(np.isfinite(delta_values).sum())
        row["summary_any_infinite_delta"] = bool(np.isinf(delta_values).any())
    else:
        row["summary_valid_delta_count"] = 0
        row["summary_any_infinite_delta"] = True
    q_values = sim[["observed_Q_mmd", "simulated_Q_mmd"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    row["simulation_any_nonfinite_q"] = bool(~np.isfinite(q_values).all())
    row["mean_delta_epsilon"] = float(summary["delta_epsilon_mean"].mean()) if "delta_epsilon_mean" in summary else np.nan
    row["median_delta_epsilon"] = float(summary["delta_epsilon_mean"].median()) if "delta_epsilon_mean" in summary else np.nan
    sim["date"] = pd.to_datetime(sim["date"])
    add_period_skill(row, sim, "all")
    add_period_skill(row, sim.loc[sim["date"].between(*pre_window)], "pre")
    add_period_skill(row, sim.loc[sim["date"].between(*post_window)], "post")
    return row


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_root = output_dir(cfg) / args.run_label
    verify_run_manifest(run_root)
    pre_window = tuple(pd.to_datetime(cfg["data"]["pre_window"]))
    post_window = tuple(pd.to_datetime(cfg["data"]["post_window"]))
    assignments = pd.read_parquet(output_dir(cfg) / "inputs" / "fold_assignment.parquet")
    rows = []
    for fold in range(int(cfg["splits"]["n_folds"])):
        role_column = f"role_fold_{fold}"
        expected_test_gcins = set(assignments.loc[assignments[role_column] == "test", "GCIN"].astype(int))
        row = audit_fold(run_root, fold, expected_test_gcins, pre_window, post_window)
        rows.append(row)
        print(
            f"audited fold {fold}: status={row.get('status')} "
            f"all_median_nse={row.get('all_median_catchment_nse', np.nan)} "
            f"pre_median_nse={row.get('pre_median_catchment_nse', np.nan)} "
            f"post_median_nse={row.get('post_median_catchment_nse', np.nan)}",
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
    role_safe = (
        (audit["missing_summary_gcins"].fillna(1) == 0).all()
        and (audit["missing_simulation_gcins"].fillna(1) == 0).all()
        and (audit["unexpected_summary_gcins"].fillna(1) == 0).all()
        and (audit["unexpected_simulation_gcins"].fillna(1) == 0).all()
    )
    validation_recorded = audit["metrics_has_validation"].fillna(False).all()
    numeric_safe = (
        (audit["summary_valid_delta_count"].fillna(0) > 0).all()
        and (~audit["summary_any_infinite_delta"].fillna(True)).all()
        and (~audit["simulation_any_nonfinite_q"].fillna(True)).all()
    )
    if not complete or not no_missing_cols or not role_safe or not validation_recorded or not numeric_safe:
        raise SystemExit(1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
