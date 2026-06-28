"""Inventory months with valid legacy streamflow for GEE export filtering."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-zip", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, default=Path("_private/processed/legacy_forcings/timeseries_matched_forcing_code_to_current_gcin_crosswalk.csv"))
    parser.add_argument("--out", type=Path, default=Path("_private/processed/legacy_forcings/timeseries_matched_valid_streamflow_months.csv"))
    parser.add_argument("--start-year", type=int, default=1950)
    parser.add_argument("--end-year", type=int, default=2019)
    return parser.parse_args()


def file_code(name: str) -> int | None:
    match = re.search(r"([^/\\]+)\.csv$", name)
    if not match or not match.group(1).isdigit():
        return None
    return int(match.group(1))


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cross = pd.read_csv(args.crosswalk)
    codes = set(cross["forcing_code"].astype(int))
    code_to_gcin = dict(zip(cross["forcing_code"].astype(int), cross["GCIN"].astype(int)))

    rows = []
    with zipfile.ZipFile(args.legacy_zip) as zf:
        for idx, name in enumerate(sorted(n for n in zf.namelist() if n.lower().endswith(".csv")), start=1):
            code = file_code(name)
            if code not in codes:
                continue
            with zf.open(name) as handle:
                df = pd.read_csv(handle, usecols=["date", "streamflow_mmd"])
            df["date"] = pd.to_datetime(df["date"], errors="raise")
            df = df[(df["date"].dt.year >= args.start_year) & (df["date"].dt.year <= args.end_year)].copy()
            df["streamflow_mmd"] = pd.to_numeric(df["streamflow_mmd"], errors="coerce")
            valid = df[df["streamflow_mmd"].notna()]
            if valid.empty:
                continue
            monthly = valid.groupby([valid["date"].dt.year, valid["date"].dt.month]).size()
            for (year, month), valid_days in monthly.items():
                rows.append(
                    {
                        "year": int(year),
                        "month": int(month),
                        "force_code": int(code),
                        "cur_gcin": int(code_to_gcin[code]),
                        "valid_streamflow_days": int(valid_days),
                    }
                )
            if idx % 100 == 0:
                print(f"processed {idx} zip entries")

    out = pd.DataFrame(rows).sort_values(["year", "month", "force_code"])
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} rows={len(out)}")
    print(out.groupby(["year", "month"])["force_code"].nunique().describe().to_string())


if __name__ == "__main__":
    main()
