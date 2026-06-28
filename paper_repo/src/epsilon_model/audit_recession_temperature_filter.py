from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config
from train_epsilon_model import detect_recession_paper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("_private/processed/qobs_inventory/recession_temperature_filter_audit.csv"))
    parser.add_argument("--thresholds", nargs="+", type=float, default=[-2.0, 0.0, 1.0, 2.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    daily_dir = Path(cfg["paths"]["physics_daily_dir"])
    years = range(int(cfg["data"]["start_year"]), int(cfg["data"]["end_year"]) + 1)
    frames = []
    for year in years:
        path = daily_dir / f"epsilon_physics_daily_{year}.parquet"
        frames.append(pd.read_parquet(path, columns=["GCIN", "date", "temperature_C", "streamflow_mmd"]))
    df = pd.concat(frames, ignore_index=True).sort_values(["GCIN", "date"])

    rows = []
    threshold_counts = {thr: 0 for thr in args.thresholds}
    threshold_catchments = {thr: [] for thr in args.thresholds}
    total_recession = 0
    total_days = 0
    basin_rows = []

    for gcin, group in df.groupby("GCIN", sort=False, observed=True):
        q = group["streamflow_mmd"].to_numpy("float64")
        temp = group["temperature_C"].to_numpy("float64")
        rec = detect_recession_paper(
            q,
            min_len=int(cfg["recession"]["min_decline_days"]),
            drop_first=1 if bool(cfg["recession"].get("drop_first_decline_day", True)) else 0,
            decreasing_rate=bool(cfg["recession"].get("decreasing_rate", True)),
        )
        rec_n = int(np.sum(rec))
        total_recession += rec_n
        total_days += len(group)
        basin_row = {"GCIN": int(gcin), "total_days": len(group), "recession_days": rec_n}
        for thr in args.thresholds:
            cold_n = int(np.sum(rec & np.isfinite(temp) & (temp <= thr)))
            basin_row[f"recession_days_temp_le_{thr:g}C"] = cold_n
            basin_row[f"recession_removed_frac_temp_le_{thr:g}C"] = cold_n / rec_n if rec_n else np.nan
            threshold_counts[thr] += cold_n
            if cold_n > 0:
                threshold_catchments[thr].append(int(gcin))
        basin_rows.append(basin_row)

    for thr in args.thresholds:
        removed = threshold_counts[thr]
        rows.append(
            {
                "threshold_C": thr,
                "total_days": total_days,
                "total_recession_days": total_recession,
                "removed_recession_days": removed,
                "removed_frac_of_recession_days": removed / total_recession if total_recession else np.nan,
                "catchments_with_removed_days": len(threshold_catchments[thr]),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    basin = pd.DataFrame(basin_rows)
    summary.to_csv(args.out, index=False)
    basin.to_csv(args.out.with_name(args.out.stem + "_by_gcin.csv"), index=False)
    print(summary.to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
