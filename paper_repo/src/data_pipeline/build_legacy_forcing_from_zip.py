"""Build mapped legacy forcing/Q parquet files from zipped per-catchment CSVs."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("_private/processed/legacy_forcings"))
    parser.add_argument("--min-year", type=int, default=1950)
    parser.add_argument("--max-year", type=int, default=2019)
    return parser.parse_args()


def file_code(name: str) -> int | None:
    match = re.search(r"([^/\\]+)\.csv$", name)
    if not match or not match.group(1).isdigit():
        return None
    return int(match.group(1))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    forcing_out = args.out_dir / "legacy_forcing_daily_parquet"
    qobs_out = args.out_dir / "legacy_qobs_yearly_parquet"
    forcing_out.mkdir(parents=True, exist_ok=True)
    qobs_out.mkdir(parents=True, exist_ok=True)

    mapping = pd.read_csv(args.mapping)
    mapping = mapping[mapping["match_method"].isin(["geometry", "source_id"])].copy()
    mapping["forcing_code"] = mapping["forcing_code"].astype(int)
    mapping["GCIN"] = pd.to_numeric(mapping["project_gcin_final"], errors="coerce").astype("Int64")
    mapping = mapping[mapping["GCIN"].notna()].copy()
    for col in ["iou", "source_coverage", "target_coverage"]:
        if col in mapping.columns:
            mapping[col] = pd.to_numeric(mapping[col], errors="coerce").fillna(0.0)
    mapping.sort_values(
        ["GCIN", "iou", "source_coverage", "target_coverage"],
        ascending=[True, False, False, False],
        inplace=True,
    )
    mapping = mapping.drop_duplicates("GCIN", keep="first")
    code_to_gcin = dict(zip(mapping["forcing_code"], mapping["GCIN"].astype(int)))

    by_year: dict[int, list[pd.DataFrame]] = {year: [] for year in range(args.min_year, args.max_year + 1)}
    skipped = []
    with zipfile.ZipFile(args.zip) as zf:
      names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
      for idx, name in enumerate(names, start=1):
          code = file_code(name)
          gcin = code_to_gcin.get(code)
          if gcin is None:
              skipped.append({"name": name, "forcing_code": code, "reason": "no_project_gcin_mapping"})
              continue
          with zf.open(name) as handle:
              df = pd.read_csv(handle, usecols=["date", "precipitation_mmd", "streamflow_mmd"])
          df["date"] = pd.to_datetime(df["date"], errors="raise")
          df = df[(df["date"].dt.year >= args.min_year) & (df["date"].dt.year <= args.max_year)].copy()
          if df.empty:
              continue
          df["GCIN"] = gcin
          df["forcing_code"] = code
          df["precipitation_mmd"] = pd.to_numeric(df["precipitation_mmd"], errors="coerce").astype("float32")
          df["streamflow_mmd"] = pd.to_numeric(df["streamflow_mmd"], errors="coerce").astype("float32")
          df["year"] = df["date"].dt.year.astype("int16")
          df["GCIN"] = df["GCIN"].astype("int32")
          df["forcing_code"] = df["forcing_code"].astype("int32")
          for year, part in df.groupby("year", sort=False):
              by_year[int(year)].append(part[["GCIN", "forcing_code", "date", "precipitation_mmd", "streamflow_mmd"]])
          if idx % 100 == 0:
              print(f"processed {idx}/{len(names)} csv files")

    inventory_rows = []
    for year, pieces in by_year.items():
        if not pieces:
            continue
        out = pd.concat(pieces, ignore_index=True)
        out.sort_values(["GCIN", "date"], inplace=True)
        target = forcing_out / f"legacy_forcing_daily_{year}.parquet"
        out.to_parquet(target, index=False, compression="zstd")

        qobs = out[["GCIN", "forcing_code", "date", "streamflow_mmd"]].copy()
        qobs.rename(columns={"streamflow_mmd": "qobs_streamflow_mm"}, inplace=True)
        qobs_target = qobs_out / f"legacy_qobs_streamflow_daily_{year}.parquet"
        qobs.to_parquet(qobs_target, index=False, compression="zstd")
        inventory_rows.append({
            "year": year,
            "rows": len(out),
            "gcins": out["GCIN"].nunique(),
            "valid_precip": int(out["precipitation_mmd"].notna().sum()),
            "valid_streamflow": int(out["streamflow_mmd"].notna().sum()),
            "forcing_path": str(target),
            "qobs_path": str(qobs_target),
        })
        print(f"wrote {target} rows={len(out)} valid_q={inventory_rows[-1]['valid_streamflow']}")

    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(args.out_dir / "legacy_forcing_yearly_inventory.csv", index=False)
    pd.DataFrame(skipped).to_csv(args.out_dir / "legacy_forcing_skipped_files.csv", index=False)

    coverage_parts = []
    for path in sorted(qobs_out.glob("legacy_qobs_streamflow_daily_*.parquet")):
        df = pd.read_parquet(path, columns=["GCIN", "date", "qobs_streamflow_mm"])
        valid = df[df["qobs_streamflow_mm"].notna()]
        if valid.empty:
            continue
        coverage_parts.append(valid.groupby("GCIN").agg(
            first_valid_date=("date", "min"),
            last_valid_date=("date", "max"),
            valid_days=("qobs_streamflow_mm", "size"),
        ).reset_index())
    if coverage_parts:
        coverage = pd.concat(coverage_parts, ignore_index=True)
        coverage = coverage.groupby("GCIN").agg(
            first_valid_date=("first_valid_date", "min"),
            last_valid_date=("last_valid_date", "max"),
            valid_days=("valid_days", "sum"),
        ).reset_index()
        coverage.to_parquet(args.out_dir / "legacy_qobs_coverage_by_gcin.parquet", index=False)
        coverage.to_csv(args.out_dir / "legacy_qobs_coverage_by_gcin.csv", index=False)
        print(f"wrote coverage rows={len(coverage)}")


if __name__ == "__main__":
    main()
