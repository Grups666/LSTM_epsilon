"""Fit GCIN-level LP/gamma priors for the epsilon model AET term."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from soil_moisture import ROOT_ZONE_COLUMNS, root_zone_soil_moisture


EPS = 1e-6
VARIABLES = ["t2m", "d2m", "u10", "v10", "sp", "ssr", "str", "aet", *ROOT_ZONE_COLUMNS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--forcing-dir",
        type=Path,
        default=Path("_private/processed/era5land_catchment_daily_parquet_training"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_private/processed/epsilon_model_inputs"),
    )
    parser.add_argument("--lp-min", type=float, default=0.05)
    parser.add_argument("--lp-max", type=float, default=1.50)
    parser.add_argument("--gamma-min", type=float, default=0.10)
    parser.add_argument("--gamma-max", type=float, default=5.00)
    return parser.parse_args()


def saturation_vapor_pressure_kpa(temp_c: pd.Series) -> pd.Series:
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def penman_monteith_pet_mm_day(df: pd.DataFrame) -> pd.Series:
    temp_c = df["t2m"] - 273.15
    dew_c = df["d2m"] - 273.15
    pressure_kpa = df["sp"] / 1000.0
    wind10 = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
    wind2 = wind10 * 4.87 / np.log(67.8 * 10.0 - 5.42)
    es = saturation_vapor_pressure_kpa(temp_c)
    ea = saturation_vapor_pressure_kpa(dew_c)
    delta = 4098.0 * es / ((temp_c + 237.3) ** 2)
    psychrometric = 0.000665 * pressure_kpa
    rn_mj = (df["ssr"] + df["str"]) / 1_000_000.0
    numerator = 0.408 * delta * rn_mj + psychrometric * (900.0 / (temp_c + 273.0)) * wind2 * (es - ea)
    denominator = delta + psychrometric * (1.0 + 0.34 * wind2)
    return (numerator / denominator).clip(lower=0.0)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.forcing_dir.glob("era5land_catchment_daily_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No forcing Parquet files found in {args.forcing_dir}")

    stats: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    for path in files:
        df = pd.read_parquet(path, columns=["GCIN", *VARIABLES])
        df["GCIN"] = df["GCIN"].astype(str)
        df["pet_mmd"] = penman_monteith_pet_mm_day(df)
        df["sm_rootzone"] = root_zone_soil_moisture(df)
        valid = (df["pet_mmd"] > EPS) & (df["aet"] > EPS) & (df["sm_rootzone"] > EPS)
        d = df.loc[valid, ["GCIN", "pet_mmd", "aet", "sm_rootzone"]].copy()
        d["x"] = np.log(d["sm_rootzone"].clip(lower=EPS))
        d["y"] = np.log((d["aet"] / d["pet_mmd"]).clip(lower=EPS))
        d["x2"] = d["x"] ** 2
        d["y2"] = d["y"] ** 2
        d["xy"] = d["x"] * d["y"]
        for gcin, grp in d.groupby("GCIN", sort=False):
            s = stats[gcin]
            s["n"] += len(grp)
            s["sum_x"] += float(grp["x"].sum())
            s["sum_y"] += float(grp["y"].sum())
            s["sum_xx"] += float(grp["x2"].sum())
            s["sum_yy"] += float(grp["y2"].sum())
            s["sum_xy"] += float(grp["xy"].sum())

    rows = []
    for gcin in sorted(stats, key=lambda x: int(x) if x.isdigit() else x):
        s = stats[gcin]
        n = int(s["n"])
        if n < 10:
            rows.append(
                {
                    "GCIN": gcin,
                    "Lp": np.nan,
                    "Lp_lower_CI": np.nan,
                    "Lp_upper_CI": np.nan,
                    "gamma": np.nan,
                    "gamma_low": np.nan,
                    "gamma_high": np.nan,
                    "R2": np.nan,
                    "n_fit_days": n,
                    "fit_scope": "all_days_no_streamflow_recession_filter",
                }
            )
            continue
        xbar = s["sum_x"] / n
        ybar = s["sum_y"] / n
        ssx = s["sum_xx"] - n * xbar**2
        ssy = s["sum_yy"] - n * ybar**2
        sxy = s["sum_xy"] - n * xbar * ybar
        gamma = sxy / ssx if ssx > 0 else np.nan
        intercept = ybar - gamma * xbar if np.isfinite(gamma) else np.nan
        sse = ssy - gamma * sxy if np.isfinite(gamma) else np.nan
        r2 = 1.0 - (sse / ssy) if ssy > 0 and np.isfinite(sse) else np.nan
        sigma2 = max(sse / max(n - 2, 1), 0.0) if np.isfinite(sse) else np.nan
        se_gamma = np.sqrt(sigma2 / ssx) if ssx > 0 and np.isfinite(sigma2) else np.nan
        gamma_low = gamma - 1.96 * se_gamma if np.isfinite(se_gamma) else np.nan
        gamma_high = gamma + 1.96 * se_gamma if np.isfinite(se_gamma) else np.nan
        if np.isfinite(gamma) and abs(gamma) > EPS:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                lp = np.exp(-intercept / gamma)
                lp_low = np.exp(-intercept / gamma_high) if np.isfinite(gamma_high) and abs(gamma_high) > EPS else np.nan
                lp_high = np.exp(-intercept / gamma_low) if np.isfinite(gamma_low) and abs(gamma_low) > EPS else np.nan
        else:
            lp = lp_low = lp_high = np.nan
        if np.isfinite(lp) and (not np.isfinite(lp_low) or not np.isfinite(lp_high)):
            lp_low = lp_high = lp

        rows.append(
            {
                "GCIN": gcin,
                "Lp": float(np.clip(lp, args.lp_min, args.lp_max)) if np.isfinite(lp) else np.nan,
                "Lp_lower_CI": float(np.clip(min(lp_low, lp_high), args.lp_min, args.lp_max))
                if np.isfinite(lp_low) and np.isfinite(lp_high)
                else np.nan,
                "Lp_upper_CI": float(np.clip(max(lp_low, lp_high), args.lp_min, args.lp_max))
                if np.isfinite(lp_low) and np.isfinite(lp_high)
                else np.nan,
                "gamma": float(np.clip(gamma, args.gamma_min, args.gamma_max)) if np.isfinite(gamma) else np.nan,
                "gamma_low": float(np.clip(gamma_low, args.gamma_min, args.gamma_max)) if np.isfinite(gamma_low) else np.nan,
                "gamma_high": float(np.clip(gamma_high, args.gamma_min, args.gamma_max)) if np.isfinite(gamma_high) else np.nan,
                "R2": float(r2) if np.isfinite(r2) else np.nan,
                "n_fit_days": n,
                "fit_scope": "all_days_no_streamflow_recession_filter",
            }
        )

    out = pd.DataFrame(rows)
    target_parquet = args.output_dir / "lp_gamma_fit_summary.parquet"
    target_csv = args.output_dir / "lp_gamma_fit_summary.csv"
    out.to_parquet(target_parquet, index=False)
    out.to_csv(target_csv, index=False)
    print(f"wrote {target_parquet} rows={len(out)}")
    print(f"wrote {target_csv} rows={len(out)}")


if __name__ == "__main__":
    main()
