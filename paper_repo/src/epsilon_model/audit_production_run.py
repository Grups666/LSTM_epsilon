from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config, output_dir
from temporal_split import test_years, train_years


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-label", type=str, default="temporal_crossfit_1990")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() < 2:
        return np.nan
    obs = obs[valid]
    pred = pred[valid]
    denominator = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - np.sum((pred - obs) ** 2) / denominator) if denominator > 0 else np.nan


def audit_fold(run_root: Path, fold: int, cfg: dict) -> tuple[dict[str, object], pd.DataFrame]:
    fold_dir = run_root / f"fold_{fold}"
    required = {
        "model": fold_dir / "final_model.pt",
        "metrics": fold_dir / "metrics.csv",
        "metadata": fold_dir / "run_metadata.json",
        "summary": fold_dir / "heldout_epsilon_change_summary.parquet",
        "simulation": fold_dir / "recession_day_simulations.parquet",
        "skill": fold_dir / "heldout_skill_summary.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        return {"fold": fold, "status": "incomplete", "missing": ";".join(missing)}, pd.DataFrame()

    metadata = json.loads(required["metadata"].read_text(encoding="utf-8"))
    metrics = pd.read_csv(required["metrics"])
    sim = pd.read_parquet(required["simulation"])
    skill = pd.read_csv(required["skill"]).set_index("period")
    sim["date"] = pd.to_datetime(sim["date"])
    expected_years = test_years(cfg, fold)
    expected_train_years = train_years(cfg, fold)
    metadata_test_years = set(int(year) for year in metadata.get("test_years", []))
    metadata_train_years = set(int(year) for year in metadata.get("train_years", []))
    expected_epochs = int(
        cfg["smoke"]["epochs"] if bool(metadata.get("smoke", False)) else cfg["training"]["epochs"]
    )
    wrong_years = int((~sim["date"].dt.year.isin(expected_years)).sum())
    duplicates = int(sim.duplicated(["GCIN", "date"]).sum())
    q = sim[["observed_Q_mmd", "simulated_Q_mmd"]].to_numpy(float)
    finite_q = bool(np.isfinite(q).all())
    catchment_nse = [
        nse(group["observed_Q_mmd"].to_numpy(float), group["simulated_Q_mmd"].to_numpy(float))
        for _, group in sim.groupby("GCIN", observed=True)
    ]
    no_validation_columns = not any(column.startswith("validation_") for column in metrics.columns)
    reported_median_nse = float(skill.loc["all", "median_catchment_nse"])
    calculated_median_nse = float(np.nanmedian(catchment_nse))
    final_epoch = int(metadata.get("final_epoch", -1))
    row = {
        "fold": fold,
        "status": "complete",
        "test_years": ",".join(str(year) for year in sorted(expected_years)),
        "simulation_rows": len(sim),
        "simulation_gcins": sim["GCIN"].nunique(),
        "wrong_test_year_rows": wrong_years,
        "metadata_test_years_match": metadata_test_years == expected_years,
        "metadata_train_years_match": metadata_train_years == expected_train_years,
        "metadata_year_overlap": len(metadata_test_years & metadata_train_years),
        "duplicate_catchment_dates": duplicates,
        "all_q_finite": finite_q,
        "metrics_epochs": metrics["epoch"].nunique(),
        "final_epoch": final_epoch,
        "expected_epochs": expected_epochs,
        "no_validation_columns": no_validation_columns,
        "checkpoint_selection": metadata.get("checkpoint_selection"),
        "median_fold_catchment_nse": calculated_median_nse,
        "reported_median_fold_catchment_nse": reported_median_nse,
        "skill_report_matches": bool(np.isclose(calculated_median_nse, reported_median_nse, equal_nan=True)),
    }
    return row, sim


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_root = output_dir(cfg) / args.run_label
    rows = []
    simulations = []
    for fold in range(int(cfg["splits"]["n_folds"])):
        row, sim = audit_fold(run_root, fold, cfg)
        rows.append(row)
        if not sim.empty:
            simulations.append(sim.assign(fold=fold))
        print(
            f"fold={fold} status={row['status']} rows={row.get('simulation_rows', 0)} "
            f"median_nse={row.get('median_fold_catchment_nse', np.nan):.3f}",
            flush=True,
        )

    audit = pd.DataFrame(rows)
    if simulations:
        combined = pd.concat(simulations, ignore_index=True)
        cross_fold_duplicates = int(combined.duplicated(["GCIN", "date"]).sum())
    else:
        cross_fold_duplicates = -1
    audit["cross_fold_duplicate_catchment_dates"] = cross_fold_duplicates
    out = args.out or (run_root / "production_audit.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out, index=False)

    checks = (
        (audit["status"] == "complete").all()
        and (audit["wrong_test_year_rows"] == 0).all()
        and audit["metadata_test_years_match"].all()
        and audit["metadata_train_years_match"].all()
        and (audit["metadata_year_overlap"] == 0).all()
        and (audit["duplicate_catchment_dates"] == 0).all()
        and audit["all_q_finite"].all()
        and audit["no_validation_columns"].all()
        and audit["skill_report_matches"].all()
        and (audit["checkpoint_selection"] == "fixed final epoch; no validation or test tuning").all()
        and (audit["metrics_epochs"] == audit["final_epoch"]).all()
        and (audit["final_epoch"] == audit["expected_epochs"]).all()
        and cross_fold_duplicates == 0
    )
    if not checks:
        print(audit.to_string(index=False))
        raise SystemExit(1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
