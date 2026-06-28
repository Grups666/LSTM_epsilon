"""Build Qobs extension from streamflow-only legacy/current matches."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-zip", type=Path, required=True)
    parser.add_argument("--match", type=Path, default=Path("_private/processed/legacy_forcings/legacy_current_streamflow_timeseries_match.csv"))
    parser.add_argument("--current-dir", type=Path, default=Path("_private/processed/qobs_yearly_parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("_private/processed/legacy_forcings/timeseries_matched_extended_qobs_yearly_parquet"))
    parser.add_argument("--coverage-out", type=Path, default=Path("_private/processed/legacy_forcings/timeseries_matched_extended_qobs_coverage_by_gcin.csv"))
    parser.add_argument("--crosswalk-out", type=Path, default=Path("_private/processed/legacy_forcings/timeseries_matched_forcing_code_to_current_gcin_crosswalk.csv"))
    parser.add_argument("--start-year", type=int, default=1950)
    parser.add_argument("--current-start-year", type=int, default=1982)
    parser.add_argument("--end-year", type=int, default=2019)
    return parser.parse_args()


def file_code(name: str) -> int | None:
    match = re.search(r"([^/\\]+)\.csv$", name)
    if not match or not match.group(1).isdigit():
        return None
    return int(match.group(1))


def primary_crosswalk(match_path: Path) -> pd.DataFrame:
    matches = pd.read_csv(match_path)
    passed = matches[matches["passes_streamflow_match"].eq(True)].copy()
    passed["best_current_gcin"] = passed["best_current_gcin"].astype("int32")
    passed["forcing_code"] = passed["forcing_code"].astype("int32")
    passed.sort_values(
        ["best_current_gcin", "log_corr_nonzero", "both_nonzero_days", "median_ratio_fold"],
        ascending=[True, False, False, True],
        inplace=True,
    )
    primary = passed.drop_duplicates("best_current_gcin", keep="first").copy()
    primary.rename(columns={"best_current_gcin": "GCIN"}, inplace=True)
    return primary


def write_pre_current_from_zip(args: argparse.Namespace, crosswalk: pd.DataFrame) -> list[pd.DataFrame]:
    code_to_gcin = dict(zip(crosswalk["forcing_code"].astype(int), crosswalk["GCIN"].astype(int)))
    by_year: dict[int, list[pd.DataFrame]] = {year: [] for year in range(args.start_year, args.current_start_year)}
    with zipfile.ZipFile(args.legacy_zip) as zf:
        for name in sorted(n for n in zf.namelist() if n.lower().endswith(".csv")):
            code = file_code(name)
            if code not in code_to_gcin:
                continue
            with zf.open(name) as handle:
                df = pd.read_csv(handle, usecols=["date", "streamflow_mmd"])
            df["date"] = pd.to_datetime(df["date"], errors="raise")
            df = df[(df["date"].dt.year >= args.start_year) & (df["date"].dt.year < args.current_start_year)].copy()
            if df.empty:
                continue
            df["qobs_streamflow_mm"] = pd.to_numeric(df["streamflow_mmd"], errors="coerce").astype("float32")
            df = df[df["qobs_streamflow_mm"].notna()].copy()
            if df.empty:
                continue
            df["GCIN"] = int(code_to_gcin[code])
            df["legacy_forcing_code"] = int(code)
            df["qobs_source"] = "legacy_pre1982_timeseries_matched"
            df["year"] = df["date"].dt.year.astype("int16")
            for year, part in df.groupby("year", sort=False):
                by_year[int(year)].append(part[["GCIN", "legacy_forcing_code", "date", "qobs_streamflow_mm", "qobs_source"]])

    coverage_parts = []
    for year, pieces in by_year.items():
        if not pieces:
            continue
        out = pd.concat(pieces, ignore_index=True)
        out.sort_values(["GCIN", "date"], inplace=True)
        path = args.out_dir / f"timeseries_matched_extended_qobs_{year}.parquet"
        out.to_parquet(path, index=False, compression="zstd")
        coverage_parts.append(out[["GCIN", "date", "qobs_streamflow_mm", "qobs_source"]])
        print(f"wrote {path} rows={len(out)} gcins={out['GCIN'].nunique()}")
    return coverage_parts


def write_current_years(args: argparse.Namespace) -> list[pd.DataFrame]:
    coverage_parts = []
    for year in range(args.current_start_year, args.end_year + 1):
        path = args.current_dir / f"qobs_streamflow_daily_{year}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df[df["qobs_streamflow_mm"].notna()].copy()
        df["legacy_forcing_code"] = pd.NA
        df["qobs_source"] = "current"
        df = df[["GCIN", "legacy_forcing_code", "date", "qobs_streamflow_mm", "qobs_source"]]
        out_path = args.out_dir / f"timeseries_matched_extended_qobs_{year}.parquet"
        df.to_parquet(out_path, index=False, compression="zstd")
        coverage_parts.append(df[["GCIN", "date", "qobs_streamflow_mm", "qobs_source"]])
        print(f"wrote {out_path} rows={len(df)} gcins={df['GCIN'].nunique()}")
    return coverage_parts


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.coverage_out.parent.mkdir(parents=True, exist_ok=True)
    crosswalk = primary_crosswalk(args.match)
    crosswalk.to_csv(args.crosswalk_out, index=False)
    print(f"wrote {args.crosswalk_out} rows={len(crosswalk)}")
    print(f"accepted forcing codes={crosswalk['forcing_code'].nunique()} current GCINs={crosswalk['GCIN'].nunique()}")

    coverage_parts = []
    coverage_parts.extend(write_pre_current_from_zip(args, crosswalk))
    coverage_parts.extend(write_current_years(args))

    if coverage_parts:
        coverage = pd.concat(coverage_parts, ignore_index=True)
        summary = coverage.groupby("GCIN").agg(
            first_valid_date=("date", "min"),
            last_valid_date=("date", "max"),
            valid_days=("qobs_streamflow_mm", "size"),
            sources=("qobs_source", lambda x: ",".join(sorted(set(x)))),
        ).reset_index()
        summary.to_csv(args.coverage_out, index=False)
        summary.to_parquet(args.coverage_out.with_suffix(".parquet"), index=False, compression="zstd")
        print(f"wrote {args.coverage_out} rows={len(summary)}")


if __name__ == "__main__":
    main()
