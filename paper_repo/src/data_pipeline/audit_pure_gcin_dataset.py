from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


STATIC_COLUMNS = [
    "longitude",
    "latitude",
    "area_km2",
    "elevation_mean_m",
    "mean_slope_degree",
    "Median_DepthToBedrock_cm",
    "Prec_mm",
    "Temp_C",
    "PET_mm",
    "AET_mm",
    "P_AET_mm",
    "Aridity",
    "max_soil_moisture",
    "Porosity",
    "Seasonality_of_Moisture_Index",
    "low_high_ratio",
    "wet_days_ratio_1mm",
    "wet_days_ratio_5mm",
    "high_prec_freq_10mm",
    "high_prec_dur_10mm",
    "low_prec_freq_1mm",
    "low_prec_dur_1mm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--sample-year", type=int, default=1980)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed = args.processed_dir
    static_path = processed / "epsilon_model_inputs_pure_gcin_1950_2019" / "static_attributes.parquet"
    lp_path = processed / "epsilon_model_inputs_pure_gcin_1950_2019" / "lp_gamma_fit_summary.parquet"
    physics_path = (
        processed
        / "epsilon_physics_daily_pure_gcin_1950_2019_parquet"
        / f"epsilon_physics_daily_{args.sample_year}.parquet"
    )
    daily_path = (
        processed
        / "epsilon_training_daily_pure_gcin_1950_2019_parquet"
        / f"epsilon_training_daily_{args.sample_year}.parquet"
    )

    static = pd.read_parquet(static_path)
    lp = pd.read_parquet(lp_path)
    physics = pd.read_parquet(physics_path)
    daily = pd.read_parquet(daily_path)

    audit = {
        "static_rows": int(len(static)),
        "static_gcins": int(static["GCIN"].nunique()),
        "static_duplicate_gcins": int(static["GCIN"].duplicated().sum()),
        "lp_rows": int(len(lp)),
        "lp_gcins": int(lp["GCIN"].nunique()),
        "lp_duplicate_gcins": int(lp["GCIN"].duplicated().sum()),
        "missing_static_columns": [c for c in STATIC_COLUMNS if c not in static.columns],
        "static_missing_values": {c: int(static[c].isna().sum()) for c in STATIC_COLUMNS if c in static.columns},
        "sample_year": int(args.sample_year),
        "physics_shape": [int(physics.shape[0]), int(physics.shape[1])],
        "physics_gcins": int(physics["GCIN"].nunique()),
        "physics_duplicate_gcin_date": int(physics.duplicated(["GCIN", "date"]).sum()),
        "daily_shape": [int(daily.shape[0]), int(daily.shape[1])],
        "daily_gcins": int(daily["GCIN"].nunique()),
        "daily_columns": daily.columns.tolist(),
        "daily_duplicate_gcin_date": int(daily.duplicated(["GCIN", "date"]).sum()),
        "daily_valid_q_days": int(daily["qobs_streamflow_mm"].notna().sum()),
        "gcin_overlap_static_lp": int(len(set(static["GCIN"]).intersection(set(lp["GCIN"])))),
        "gcin_overlap_static_physics_sample": int(len(set(static["GCIN"]).intersection(set(physics["GCIN"])))),
    }
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
