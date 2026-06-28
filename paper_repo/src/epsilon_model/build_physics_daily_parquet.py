from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-dir", type=Path, default=Path("_private/processed/epsilon_training_daily_parquet"))
    parser.add_argument("--out-dir", type=Path, default=Path("_private/processed/epsilon_physics_daily_parquet"))
    parser.add_argument("--start-year", type=int, default=1982)
    parser.add_argument("--end-year", type=int, default=2019)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def saturation_vapor_pressure_kpa(temp_c: np.ndarray) -> np.ndarray:
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def daily_pet_fao56_pm(df: pd.DataFrame) -> np.ndarray:
    """Approximate daily FAO56 Penman-Monteith PET from ERA5-Land daily fields.

    Inputs use the units currently present in the project parquet files:
    t2m/d2m K, u10/v10 m s-1, sp Pa, ssr/str J m-2 day-1.
    """

    temp_c = df["t2m"].to_numpy("float64") - 273.15
    dew_c = df["d2m"].to_numpy("float64") - 273.15
    pressure_kpa = df["sp"].to_numpy("float64") / 1000.0
    wind10 = np.sqrt(df["u10"].to_numpy("float64") ** 2 + df["v10"].to_numpy("float64") ** 2)
    wind2 = wind10 * 4.87 / np.log(67.8 * 10.0 - 5.42)

    rn_mj_m2_day = (df["ssr"].to_numpy("float64") + df["str"].to_numpy("float64")) / 1_000_000.0
    rn_mj_m2_day = np.maximum(rn_mj_m2_day, 0.0)

    es = saturation_vapor_pressure_kpa(temp_c)
    ea = saturation_vapor_pressure_kpa(dew_c)
    vpd = np.maximum(es - ea, 0.0)
    delta = 4098.0 * es / ((temp_c + 237.3) ** 2)
    gamma = 0.000665 * pressure_kpa

    numerator = 0.408 * delta * rn_mj_m2_day + gamma * (900.0 / (temp_c + 273.0)) * wind2 * vpd
    denominator = delta + gamma * (1.0 + 0.34 * wind2)
    pet = numerator / np.maximum(denominator, 1e-6)
    return np.maximum(np.nan_to_num(pet, nan=0.0, posinf=0.0, neginf=0.0), 0.0).astype("float32")


def rootzone_sm(df: pd.DataFrame) -> np.ndarray:
    """Weighted root-zone soil moisture from ERA5-Land volumetric layers."""

    weights = np.array([0.07, 0.21, 0.72, 1.89], dtype="float64")
    vals = df[["swvl1", "swvl2", "swvl3", "swvl4"]].to_numpy("float64")
    sm = np.average(vals, axis=1, weights=weights)
    return np.clip(np.nan_to_num(sm, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype("float32")


def convert_year(path: Path, out_path: Path) -> tuple[int, int]:
    cols = [
        "GCIN",
        "date",
        "t2m",
        "d2m",
        "u10",
        "v10",
        "sp",
        "swvl1",
        "swvl2",
        "swvl3",
        "swvl4",
        "tp",
        "ssr",
        "str",
        "aet",
        "qobs_streamflow_mm",
    ]
    df = pd.read_parquet(path, columns=cols)
    out = pd.DataFrame(
        {
            "GCIN": df["GCIN"].astype("int32"),
            "date": df["date"],
            "precipitation_mmd": df["tp"].clip(lower=0.0).astype("float32"),
            "temperature_C": (df["t2m"].astype("float64") - 273.15).astype("float32"),
            "pet_mmd": daily_pet_fao56_pm(df),
            "SM_%": rootzone_sm(df),
            "streamflow_mmd": df["qobs_streamflow_mm"].clip(lower=0.0).astype("float32"),
            "observed_AET_mm": df["aet"].astype("float32"),
        }
    )
    out.to_parquet(out_path, index=False)
    return len(out), int(out["streamflow_mmd"].notna().sum())


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    total_q = 0
    for year in range(args.start_year, args.end_year + 1):
        src = args.daily_dir / f"epsilon_training_daily_{year}.parquet"
        dst = args.out_dir / f"epsilon_physics_daily_{year}.parquet"
        if dst.exists() and not args.overwrite:
            print(f"skip existing {dst}")
            continue
        rows, valid_q = convert_year(src, dst)
        total_rows += rows
        total_q += valid_q
        print(f"wrote {dst} rows={rows} valid_q={valid_q}")
    print(f"done rows={total_rows} valid_q={total_q}")


if __name__ == "__main__":
    main()
