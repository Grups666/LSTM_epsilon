"""Match legacy forcing codes to current GCINs using streamflow only.

This script intentionally does not use boundary geometries or source ids. It
searches for the current Qobs series that best matches each legacy forcing
streamflow series over the overlapping daily period.
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-zip", type=Path, required=True)
    parser.add_argument("--current-dir", type=Path, default=Path("_private/processed/qobs_yearly_parquet"))
    parser.add_argument("--out", type=Path, default=Path("_private/processed/legacy_forcings/legacy_current_streamflow_timeseries_match.csv"))
    parser.add_argument("--start-date", default="1982-01-01")
    parser.add_argument("--end-date", default="2019-12-31")
    parser.add_argument("--legacy-chunk-size", type=int, default=32)
    parser.add_argument("--min-overlap-days", type=int, default=365)
    parser.add_argument("--zero-threshold", type=float, default=1e-6)
    parser.add_argument("--pass-log-corr", type=float, default=0.70)
    parser.add_argument("--pass-ratio-fold", type=float, default=2.0)
    return parser.parse_args()


def file_code(name: str) -> int | None:
    match = re.search(r"([^/\\]+)\.csv$", name)
    if not match or not match.group(1).isdigit():
        return None
    return int(match.group(1))


def load_current_matrix(current_dir: Path, dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    gcins = set()
    year_paths = []
    for year in range(dates[0].year, dates[-1].year + 1):
        path = current_dir / f"qobs_streamflow_daily_{year}.parquet"
        if path.exists():
            year_paths.append((year, path))
            df = pd.read_parquet(path, columns=["GCIN"])
            gcins.update(df["GCIN"].dropna().astype("int32").unique().tolist())
    gcin_values = np.array(sorted(gcins), dtype=np.int32)
    gcin_index = {int(g): i for i, g in enumerate(gcin_values)}
    date_index = {d: i for i, d in enumerate(dates)}
    matrix = np.full((len(dates), len(gcin_values)), np.nan, dtype=np.float32)

    for year, path in year_paths:
        df = pd.read_parquet(path, columns=["GCIN", "date", "qobs_streamflow_mm"])
        df = df[df["qobs_streamflow_mm"].notna()].copy()
        df["date"] = pd.to_datetime(df["date"], errors="raise")
        df = df[(df["date"] >= dates[0]) & (df["date"] <= dates[-1])]
        if df.empty:
            continue
        rows = df["date"].map(date_index).to_numpy("int32")
        cols = df["GCIN"].map(gcin_index).to_numpy("int32")
        matrix[rows, cols] = df["qobs_streamflow_mm"].to_numpy("float32")
        print(f"loaded current year={year} rows={len(df)}")
    return gcin_values, matrix


def load_legacy_matrix(zip_path: Path, dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    codes = []
    arrays = []
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
        for idx, name in enumerate(names, start=1):
            code = file_code(name)
            if code is None:
                continue
            with zf.open(name) as handle:
                df = pd.read_csv(handle, usecols=["date", "streamflow_mmd"])
            df["date"] = pd.to_datetime(df["date"], errors="raise")
            df = df[(df["date"] >= dates[0]) & (df["date"] <= dates[-1])]
            series = pd.Series(pd.to_numeric(df["streamflow_mmd"], errors="coerce").to_numpy("float32"), index=df["date"])
            arr = series.reindex(dates).to_numpy("float32")
            codes.append(code)
            arrays.append(arr)
            if idx % 100 == 0:
                print(f"loaded legacy csv {idx}/{len(names)}")
    return np.array(codes, dtype=np.int32), np.stack(arrays, axis=1).astype("float32")


def prepare_log_nonzero(q: np.ndarray, zero_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(q) & (q > zero_threshold)
    x = np.zeros(q.shape, dtype=np.float32)
    x[mask] = np.log1p(q[mask]).astype("float32")
    return x, mask.astype("float32")


def pairwise_corr_best(
    current_q: np.ndarray,
    legacy_q: np.ndarray,
    min_overlap: int,
    zero_threshold: float,
    legacy_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_x, current_m = prepare_log_nonzero(current_q, zero_threshold)
    current_x2 = current_x * current_x
    n_current = current_q.shape[1]
    n_legacy = legacy_q.shape[1]
    best_idx = np.full(n_legacy, -1, dtype=np.int32)
    best_corr = np.full(n_legacy, np.nan, dtype=np.float32)
    best_n = np.zeros(n_legacy, dtype=np.int32)

    cx_t = current_x.T.astype("float64", copy=False)
    cm_t = current_m.T.astype("float64", copy=False)
    cx2_t = current_x2.T.astype("float64", copy=False)

    for start in range(0, n_legacy, legacy_chunk_size):
        stop = min(start + legacy_chunk_size, n_legacy)
        legacy_x, legacy_m = prepare_log_nonzero(legacy_q[:, start:stop], zero_threshold)
        legacy_x2 = legacy_x * legacy_x

        lx = legacy_x.astype("float64", copy=False)
        lm = legacy_m.astype("float64", copy=False)
        lx2 = legacy_x2.astype("float64", copy=False)

        n = cm_t @ lm
        sum_x = cx_t @ lm
        sum_y = cm_t @ lx
        sum_xy = cx_t @ lx
        sum_x2 = cx2_t @ lm
        sum_y2 = cm_t @ lx2

        numerator = sum_xy - (sum_x * sum_y / np.maximum(n, 1.0))
        var_x = sum_x2 - (sum_x * sum_x / np.maximum(n, 1.0))
        var_y = sum_y2 - (sum_y * sum_y / np.maximum(n, 1.0))
        denom = np.sqrt(np.maximum(var_x, 0.0) * np.maximum(var_y, 0.0))
        corr = np.full_like(numerator, np.nan, dtype="float64")
        valid = (n >= min_overlap) & (denom > 0)
        corr[valid] = numerator[valid] / denom[valid]

        local_best = np.nanargmax(np.where(np.isfinite(corr), corr, -np.inf), axis=0)
        local_corr = corr[local_best, np.arange(stop - start)]
        local_n = n[local_best, np.arange(stop - start)]
        best_idx[start:stop] = local_best.astype("int32")
        best_corr[start:stop] = local_corr.astype("float32")
        best_n[start:stop] = local_n.astype("int32")
        print(f"matched legacy chunk {start}-{stop - 1}")

    return best_idx, best_corr, best_n


def pair_details(
    current_q: np.ndarray,
    legacy_q: np.ndarray,
    current_idx: int,
    legacy_idx: int,
    zero_threshold: float,
) -> dict[str, float | int]:
    cur = current_q[:, current_idx].astype("float64")
    leg = legacy_q[:, legacy_idx].astype("float64")
    mask = np.isfinite(cur) & np.isfinite(leg)
    both_nonzero = mask & (cur > zero_threshold) & (leg > zero_threshold)
    cur_nz = cur[both_nonzero]
    leg_nz = leg[both_nonzero]
    if len(cur_nz) >= 3 and np.nanstd(cur_nz) > 0 and np.nanstd(leg_nz) > 0:
        raw_corr = float(np.corrcoef(cur_nz, leg_nz)[0, 1])
        log_corr = float(np.corrcoef(np.log1p(cur_nz), np.log1p(leg_nz))[0, 1])
    else:
        raw_corr = np.nan
        log_corr = np.nan
    ratio = cur_nz / leg_nz if len(cur_nz) else np.array([])
    median_ratio = float(np.nanmedian(ratio)) if len(ratio) else np.nan
    ratio_fold = max(median_ratio, 1 / median_ratio) if median_ratio and np.isfinite(median_ratio) else np.nan
    return {
        "overlap_days_any": int(mask.sum()),
        "both_nonzero_days": int(both_nonzero.sum()),
        "current_nonzero_days": int((mask & (cur > zero_threshold)).sum()),
        "legacy_nonzero_days": int((mask & (leg > zero_threshold)).sum()),
        "raw_corr_nonzero": raw_corr,
        "log_corr_nonzero": log_corr,
        "median_current_mm": float(np.nanmedian(cur_nz)) if len(cur_nz) else np.nan,
        "median_legacy_mm": float(np.nanmedian(leg_nz)) if len(leg_nz) else np.nan,
        "median_current_over_legacy": median_ratio,
        "median_ratio_fold": ratio_fold,
        "mae_nonzero": float(np.nanmean(np.abs(cur_nz - leg_nz))) if len(cur_nz) else np.nan,
    }


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(args.start_date, args.end_date, freq="D")
    current_gcins, current_q = load_current_matrix(args.current_dir, dates)
    legacy_codes, legacy_q = load_legacy_matrix(args.legacy_zip, dates)
    print(f"current matrix={current_q.shape} legacy matrix={legacy_q.shape}")

    best_idx, best_corr, best_n = pairwise_corr_best(
        current_q=current_q,
        legacy_q=legacy_q,
        min_overlap=args.min_overlap_days,
        zero_threshold=args.zero_threshold,
        legacy_chunk_size=args.legacy_chunk_size,
    )

    rows = []
    for legacy_i, current_i in enumerate(best_idx):
        if current_i < 0:
            rows.append({
                "forcing_code": int(legacy_codes[legacy_i]),
                "best_current_gcin": pd.NA,
                "best_log_corr_nonzero": np.nan,
                "best_both_nonzero_days": 0,
                "passes_streamflow_match": False,
            })
            continue
        details = pair_details(current_q, legacy_q, int(current_i), legacy_i, args.zero_threshold)
        passes = (
            details["both_nonzero_days"] >= args.min_overlap_days
            and np.isfinite(details["log_corr_nonzero"])
            and details["log_corr_nonzero"] >= args.pass_log_corr
            and np.isfinite(details["median_ratio_fold"])
            and details["median_ratio_fold"] <= args.pass_ratio_fold
        )
        rows.append({
            "forcing_code": int(legacy_codes[legacy_i]),
            "best_current_gcin": int(current_gcins[current_i]),
            "best_log_corr_nonzero": float(best_corr[legacy_i]),
            "best_both_nonzero_days": int(best_n[legacy_i]),
            **details,
            "passes_streamflow_match": passes,
        })

    out = pd.DataFrame(rows).sort_values(["passes_streamflow_match", "log_corr_nonzero"], ascending=[False, False])
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} rows={len(out)}")
    print(out["passes_streamflow_match"].value_counts(dropna=False).to_string())
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
