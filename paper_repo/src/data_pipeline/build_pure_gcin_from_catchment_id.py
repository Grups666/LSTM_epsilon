from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TRAINING_COLS = [
    "GCIN",
    "catchment_id",
    "date",
    "qobs_streamflow_mm",
    "precipitation_mmd",
    "t2m",
    "d2m",
    "u10",
    "v10",
    "sp",
    "skt",
    "swvl1",
    "swvl2",
    "swvl3",
    "swvl4",
    "lai_hv",
    "lai_lv",
    "tp",
    "ssr",
    "str",
    "aet",
]

PHYSICS_COLS = [
    "GCIN",
    "date",
    "precipitation_mmd",
    "temperature_C",
    "pet_mmd",
    "SM_%",
    "streamflow_mmd",
    "observed_AET_mm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a pure original-GCIN training dataset from the Catchment_ID production dataset."
    )
    parser.add_argument("--training-in-dir", type=Path)
    parser.add_argument("--physics-in-dir", type=Path, required=True)
    parser.add_argument("--static-in", type=Path, required=True)
    parser.add_argument("--lp-gamma-in", type=Path, required=True)
    parser.add_argument("--gcin-metadata", type=Path, required=True)
    parser.add_argument("--training-out-dir", type=Path, required=True)
    parser.add_argument("--physics-out-dir", type=Path, required=True)
    parser.add_argument("--inputs-out-dir", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=1950)
    parser.add_argument("--end-year", type=int, default=2019)
    return parser.parse_args()


def read_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path)
    required = ["GCIN", "mean_elevation_m", "mean_slope_degree", "Median DepthToBedrock"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise KeyError(f"Missing columns in GCIN metadata: {missing}")
    out = meta[
        [
            "GCIN",
            "mean_elevation_m",
            "mean_slope_degree",
            "Median DepthToBedrock",
            "AI",
            "P-AET",
            "Area",
            "Urban",
            "Forest",
            "Cropland",
            "Bare",
            "Median TWI",
            "First Geo Dominant Class",
            "Second Geo Dominant Class",
        ]
    ].copy()
    out = out.rename(
        columns={
            "mean_elevation_m": "elevation_mean_m",
            "Median DepthToBedrock": "Median_DepthToBedrock_cm",
            "AI": "metadata_AI",
            "P-AET": "metadata_P_AET",
            "Area": "metadata_area_km2",
            "Median TWI": "Median_TWI",
            "First Geo Dominant Class": "First_Geo_Dominant_Class",
            "Second Geo Dominant Class": "Second_Geo_Dominant_Class",
        }
    )
    out["GCIN"] = pd.to_numeric(out["GCIN"], errors="raise").astype("int64")
    return out.drop_duplicates("GCIN")


def build_maps(static_in: Path) -> tuple[pd.DataFrame, dict[int, int]]:
    static = pd.read_parquet(static_in)
    gcin = static.loc[static["source"].eq("GCIN")].copy()
    gcin["original_GCIN"] = pd.to_numeric(gcin["source_id"], errors="raise").astype("int64")
    if gcin["original_GCIN"].duplicated().any():
        dup = gcin.loc[gcin["original_GCIN"].duplicated(), "original_GCIN"].head(10).tolist()
        raise ValueError(f"Duplicate original GCIN values in source mapping: {dup}")
    id_map = dict(zip(gcin["GCIN"].astype("int64"), gcin["original_GCIN"]))
    return gcin, id_map


def map_gcin_column(df: pd.DataFrame, id_map: dict[int, int]) -> pd.DataFrame:
    out = df[df["GCIN"].isin(id_map)].copy()
    out["internal_catchment_key"] = out["GCIN"].astype("int64")
    out["GCIN"] = out["internal_catchment_key"].map(id_map).astype("int64")
    return out


def compute_low_high_ratio(training_out_dir: Path, start_year: int, end_year: int) -> pd.DataFrame:
    pieces = []
    for year in range(start_year, end_year + 1):
        path = training_out_dir / f"epsilon_training_daily_{year}.parquet"
        df = pd.read_parquet(path, columns=["GCIN", "qobs_streamflow_mm"])
        pieces.append(df)
    q = pd.concat(pieces, ignore_index=True)
    q["qobs_streamflow_mm"] = pd.to_numeric(q["qobs_streamflow_mm"], errors="coerce")
    q = q[np.isfinite(q["qobs_streamflow_mm"]) & (q["qobs_streamflow_mm"] >= 0)].copy()
    stats = q.groupby("GCIN", observed=True)["qobs_streamflow_mm"].quantile([0.10, 0.90]).unstack()
    stats.columns = ["q10_streamflow_mmd", "q90_streamflow_mmd"]
    stats["low_high_ratio"] = stats["q10_streamflow_mmd"] / stats["q90_streamflow_mmd"].replace(0, np.nan)
    stats["low_high_ratio"] = stats["low_high_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return stats.reset_index()


def build_lightweight_training_frame(phys: pd.DataFrame) -> pd.DataFrame:
    out = phys[["GCIN", "date", "precipitation_mmd", "streamflow_mmd", "internal_catchment_key"]].copy()
    out = out.rename(
        columns={
            "precipitation_mmd": "tp",
            "streamflow_mmd": "qobs_streamflow_mm",
        }
    )
    return out


def main() -> None:
    args = parse_args()
    args.training_out_dir.mkdir(parents=True, exist_ok=True)
    args.physics_out_dir.mkdir(parents=True, exist_ok=True)
    args.inputs_out_dir.mkdir(parents=True, exist_ok=True)

    source_static, id_map = build_maps(args.static_in)
    meta = read_metadata(args.gcin_metadata)

    yearly_rows: list[dict[str, object]] = []
    for year in range(args.start_year, args.end_year + 1):
        phys_in = args.physics_in_dir / f"epsilon_physics_daily_{year}.parquet"
        phys = pd.read_parquet(phys_in)

        phys = map_gcin_column(phys, id_map)

        if args.training_in_dir is not None:
            train_in = args.training_in_dir / f"epsilon_training_daily_{year}.parquet"
            train = pd.read_parquet(train_in)
            train = map_gcin_column(train, id_map)
            train = train[[c for c in TRAINING_COLS if c in train.columns] + ["internal_catchment_key"]]
        else:
            train = build_lightweight_training_frame(phys)
        phys = phys[[c for c in PHYSICS_COLS if c in phys.columns] + ["internal_catchment_key"]]

        train_out = args.training_out_dir / f"epsilon_training_daily_{year}.parquet"
        phys_out = args.physics_out_dir / f"epsilon_physics_daily_{year}.parquet"
        train.to_parquet(train_out, index=False)
        phys.to_parquet(phys_out, index=False)
        yearly_rows.append(
            {
                "year": year,
                "training_rows": len(train),
                "training_gcins": int(train["GCIN"].nunique()),
                "physics_rows": len(phys),
                "physics_gcins": int(phys["GCIN"].nunique()),
            }
        )
        print(f"{year}: rows={len(train)} gcins={train['GCIN'].nunique()}", flush=True)

    low_high = compute_low_high_ratio(args.training_out_dir, args.start_year, args.end_year)

    static = source_static.copy()
    static["internal_catchment_key"] = static["GCIN"].astype("int64")
    static["GCIN"] = static["original_GCIN"].astype("int64")
    static = static.merge(meta, on="GCIN", how="left")
    static = static.merge(low_high, on="GCIN", how="left")
    static["low_high_ratio"] = static["low_high_ratio"].fillna(0.0)
    static["attribute_version"] = "pure_gcin_gee_forcing_metadata_v1"
    static.to_parquet(args.inputs_out_dir / "static_attributes.parquet", index=False)
    static.to_csv(args.inputs_out_dir / "static_attributes.csv", index=False)

    lp = pd.read_parquet(args.lp_gamma_in)
    lp = map_gcin_column(lp, id_map)
    lp = lp.drop(columns=["internal_catchment_key"], errors="ignore")
    lp.to_parquet(args.inputs_out_dir / "lp_gamma_fit_summary.parquet", index=False)
    lp.to_csv(args.inputs_out_dir / "lp_gamma_fit_summary.csv", index=False)

    audit = {
        "start_year": args.start_year,
        "end_year": args.end_year,
        "source_static_rows_gcin": int(len(source_static)),
        "final_static_rows": int(len(static)),
        "metadata_matched_rows": int(static["elevation_mean_m"].notna().sum()),
        "lp_gamma_rows": int(len(lp)),
        "training_rows_total": int(sum(r["training_rows"] for r in yearly_rows)),
        "physics_rows_total": int(sum(r["physics_rows"] for r in yearly_rows)),
        "yearly": yearly_rows,
        "notes": [
            "Only source == GCIN catchments are retained.",
            "Model GCIN is remapped from internal Catchment_ID numeric key to original GCIN/source_id.",
            "Meteorological and land-state variables are inherited from GEE-derived Catchment_ID products.",
            "Observed streamflow is inherited from GCIN forcing records.",
        ],
    }
    (args.inputs_out_dir / "pure_gcin_build_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "yearly"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
