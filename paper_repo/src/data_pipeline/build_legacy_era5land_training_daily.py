"""Merge legacy streamflow with ERA5-Land forcing for epsilon training.

Streamflow is the only variable taken from the legacy matched timeseries.
All meteorological and land-surface variables come from the GEE ERA5-Land
catchment product.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forcing-dir", type=Path, default=Path("_private/processed/gee_era5land_daily_yearly"))
    parser.add_argument(
        "--qobs-dir",
        type=Path,
        default=Path("_private/processed/legacy_forcings/timeseries_matched_extended_qobs_yearly_parquet"),
    )
    parser.add_argument(
        "--crosswalk",
        type=Path,
        default=Path("_private/processed/legacy_forcings/timeseries_matched_forcing_code_to_current_gcin_crosswalk.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("_private/processed/epsilon_training_daily_era5land_legacy_qobs_parquet"))
    parser.add_argument("--start-year", type=int, default=1950)
    parser.add_argument("--end-year", type=int, default=2019)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    crosswalk = pd.read_csv(args.crosswalk)
    if "force_code" in crosswalk.columns:
        force_col = "force_code"
    elif "forcing_code" in crosswalk.columns:
        force_col = "forcing_code"
    else:
        raise ValueError(f"crosswalk lacks force_code/forcing_code: {crosswalk.columns.tolist()}")
    if "GCIN" not in crosswalk.columns:
        raise ValueError(f"crosswalk lacks GCIN: {crosswalk.columns.tolist()}")
    force_by_gcin = (
        crosswalk[["GCIN", force_col]]
        .rename(columns={force_col: "force_code"})
        .dropna()
        .drop_duplicates("GCIN")
        .assign(GCIN=lambda x: x["GCIN"].astype("int32"), force_code=lambda x: x["force_code"].astype("int32"))
    )
    total_rows = 0
    total_q = 0
    for year in range(args.start_year, args.end_year + 1):
        out = args.out_dir / f"epsilon_training_daily_{year}.parquet"
        if out.exists() and not args.overwrite:
            print(f"skip existing {out}", flush=True)
            continue
        forcing_path = args.forcing_dir / f"era5land_legacy_daily_{year}.parquet"
        qobs_path = args.qobs_dir / f"timeseries_matched_extended_qobs_{year}.parquet"
        if not forcing_path.exists():
            raise FileNotFoundError(forcing_path)
        if not qobs_path.exists():
            raise FileNotFoundError(qobs_path)

        forcing = pd.read_parquet(forcing_path)
        forcing["force_code"] = forcing["force_code"].astype("int32")
        forcing["cur_gcin"] = forcing["cur_gcin"].astype("int32")
        forcing["date"] = pd.to_datetime(forcing["date"], errors="raise")
        forcing = forcing.rename(columns={"cur_gcin": "GCIN"})

        qobs = pd.read_parquet(qobs_path, columns=["GCIN", "legacy_forcing_code", "date", "qobs_streamflow_mm", "qobs_source"])
        qobs["GCIN"] = qobs["GCIN"].astype("int32")
        qobs = qobs.merge(force_by_gcin, on="GCIN", how="left", validate="many_to_one")
        qobs = qobs[qobs["force_code"].notna()].copy()
        has_legacy_code = qobs["legacy_forcing_code"].notna()
        qobs.loc[has_legacy_code, "force_code"] = qobs.loc[has_legacy_code, "legacy_forcing_code"].astype("int32")
        if qobs["force_code"].isna().any():
            missing = sorted(qobs.loc[qobs["force_code"].isna(), "GCIN"].unique().tolist())[:10]
            raise ValueError(f"missing force_code for GCIN examples: {missing}")
        qobs["force_code"] = qobs["force_code"].astype("int32")
        qobs["date"] = pd.to_datetime(qobs["date"], errors="raise")
        qobs = qobs.drop(columns=["legacy_forcing_code"])

        merged = forcing.merge(qobs, on=["GCIN", "force_code", "date"], how="left", validate="one_to_one")
        merged = merged.sort_values(["GCIN", "date"])
        merged["qobs_streamflow_mm"] = merged["qobs_streamflow_mm"].astype("float32")
        merged.to_parquet(out, index=False, compression="zstd")
        rows = len(merged)
        valid_q = int(merged["qobs_streamflow_mm"].notna().sum())
        total_rows += rows
        total_q += valid_q
        print(f"wrote {out} rows={rows} valid_q={valid_q}", flush=True)
    print(f"done rows={total_rows} valid_q={total_q}", flush=True)


if __name__ == "__main__":
    main()
