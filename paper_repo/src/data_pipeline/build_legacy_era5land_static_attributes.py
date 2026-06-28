"""Build static attributes for the legacy-matched ERA5-Land training set."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forcing-dir", type=Path, default=Path("_private/processed/gee_era5land_daily_yearly"))
    parser.add_argument("--boundaries", type=Path, default=Path("D:/ClimateEpsilon/data/Gauged_Catchments_Boundaries.gpkg"))
    parser.add_argument(
        "--crosswalk",
        type=Path,
        default=Path("D:/ClimateEpsilon/data/timeseries_matched_forcing_code_to_current_gcin_crosswalk.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("_private/processed/epsilon_model_inputs_era5land_legacy"))
    return parser.parse_args()


def saturation_vapor_pressure_kpa(temp_c: pd.Series) -> pd.Series:
    return 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))


def pet_pm(df: pd.DataFrame) -> pd.Series:
    temp_c = df["t2m"] - 273.15
    dew_c = df["d2m"] - 273.15
    pressure_kpa = df["sp"] / 1000.0
    wind10 = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
    wind2 = wind10 * 4.87 / np.log(67.8 * 10.0 - 5.42)
    es = saturation_vapor_pressure_kpa(temp_c)
    ea = saturation_vapor_pressure_kpa(dew_c)
    vpd = (es - ea).clip(lower=0.0)
    delta = 4098.0 * es / ((temp_c + 237.3) ** 2)
    gamma = 0.000665 * pressure_kpa
    rn = ((df["ssr"] + df["str"]) / 1_000_000.0).clip(lower=0.0)
    pet = (0.408 * delta * rn + gamma * (900.0 / (temp_c + 273.0)) * wind2 * vpd) / (
        delta + gamma * (1.0 + 0.34 * wind2)
    )
    return pet.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)


def spell_stats(values: pd.Series, threshold: float, wet: bool) -> tuple[int, float]:
    flags = values > threshold if wet else values <= threshold
    if flags.empty:
        return 0, 0.0
    starts = flags & ~flags.shift(fill_value=False)
    spell_ids = starts.cumsum()
    lengths = flags.groupby(spell_ids).sum()
    lengths = lengths[lengths > 0]
    if lengths.empty:
        return 0, 0.0
    return int(len(lengths)), float(lengths.mean())


def boundary_attributes(boundaries: Path, crosswalk: Path) -> pd.DataFrame:
    cross = pd.read_csv(crosswalk)
    if "forcing_code" in cross.columns:
        force_col = "forcing_code"
    elif "force_code" in cross.columns:
        force_col = "force_code"
    else:
        raise ValueError(f"crosswalk lacks forcing_code/force_code: {cross.columns.tolist()}")
    if "GCIN" not in cross.columns:
        raise ValueError(f"crosswalk lacks GCIN: {cross.columns.tolist()}")
    cross = cross[[force_col, "GCIN"]].rename(columns={force_col: "force_code", "GCIN": "cur_gcin"})
    cross["force_code"] = cross["force_code"].astype("int32")
    cross["cur_gcin"] = cross["cur_gcin"].astype("int32")

    gdf = gpd.read_file(boundaries)
    if "GCIN" not in gdf.columns:
        raise ValueError(f"boundary file lacks GCIN column: {boundaries}")
    gdf["force_code"] = pd.to_numeric(gdf["GCIN"], errors="coerce").astype("Int64")
    gdf = gdf[gdf["force_code"].notna()].copy()
    gdf["force_code"] = gdf["force_code"].astype("int32")
    gdf = gdf.merge(cross, on="force_code", how="inner", validate="one_to_one")
    centroids = gdf.to_crs(6933).geometry.centroid.to_crs(4326)
    area_km2 = gdf.to_crs(6933).area / 1e6
    out = pd.DataFrame(
        {
            "GCIN": gdf["cur_gcin"].astype("int32"),
            "force_code": gdf["force_code"].astype("int32"),
            "longitude": centroids.x.to_numpy(),
            "latitude": centroids.y.to_numpy(),
            "area_km2": area_km2.to_numpy(),
        }
    )
    out["Source ID"] = "legacy_forcing_matched"
    out["country"] = ""
    out["precipitation_source"] = "ERA5-Land GEE reduceRegions"
    out["streamflow_source"] = "legacy_pre1982_timeseries_matched"
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.forcing_dir.glob("era5land_legacy_daily_*.parquet"))
    if not files:
        raise FileNotFoundError(args.forcing_dir)

    sums: dict[int, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: defaultdict[int, int] = defaultdict(int)
    precip_series: dict[int, list[pd.Series]] = defaultdict(list)
    monthly: dict[tuple[int, int], defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    monthly_counts: defaultdict[tuple[int, int], int] = defaultdict(int)

    for path in files:
        df = pd.read_parquet(path)
        df = df.rename(columns={"cur_gcin": "GCIN"})
        df["GCIN"] = df["GCIN"].astype("int32")
        df["date"] = pd.to_datetime(df["date"])
        df["temp_c"] = df["t2m"] - 273.15
        df["pet_mmd"] = pet_pm(df)
        df["sm_rootzone"] = df[["swvl1", "swvl2", "swvl3"]].mean(axis=1)
        df["net_radiation_mj_m2_day"] = (df["ssr"] + df["str"]) / 1_000_000.0
        for gcin, grp in df.groupby("GCIN", sort=False):
            gcin = int(gcin)
            counts[gcin] += len(grp)
            precip_series[gcin].append(grp.sort_values("date")["tp"].reset_index(drop=True))
            for source, target in [
                ("tp", "precip_mm"),
                ("temp_c", "temp_c"),
                ("pet_mmd", "pet_mm"),
                ("aet", "aet_mm"),
                ("sm_rootzone", "sm_rootzone"),
                ("net_radiation_mj_m2_day", "net_radiation_mj_m2_day"),
                ("swvl1", "swvl1"),
                ("swvl2", "swvl2"),
                ("swvl3", "swvl3"),
                ("swvl4", "swvl4"),
            ]:
                sums[gcin][target] += float(grp[source].sum(skipna=True))
            sums[gcin]["wet_days_1mm"] += float((grp["tp"] >= 1.0).sum())
            sums[gcin]["wet_days_5mm"] += float((grp["tp"] >= 5.0).sum())
            sums[gcin]["high_prec_days_10mm"] += float((grp["tp"] >= 10.0).sum())
            sums[gcin]["dry_days_1mm"] += float((grp["tp"] < 1.0).sum())
            max_sm = float(grp["sm_rootzone"].max(skipna=True)) if grp["sm_rootzone"].notna().any() else 0.0
            sums[gcin]["max_soil_moisture"] = max(sums[gcin]["max_soil_moisture"], max_sm)
        df["month"] = df["date"].dt.month
        for (gcin, month), grp in df.groupby(["GCIN", "month"], sort=False):
            key = (int(gcin), int(month))
            monthly_counts[key] += len(grp)
            monthly[key]["precip_mm"] += float(grp["tp"].sum(skipna=True))
            monthly[key]["pet_mm"] += float(grp["pet_mmd"].sum(skipna=True))

    rows = []
    for gcin in sorted(counts):
        n = counts[gcin]
        p_mean = sums[gcin]["precip_mm"] / n
        pet_mean = sums[gcin]["pet_mm"] / n
        aet_mean = sums[gcin]["aet_mm"] / n
        monthly_p = []
        monthly_pet = []
        for month in range(1, 13):
            key = (gcin, month)
            c = max(monthly_counts[key], 1)
            monthly_p.append(monthly[key]["precip_mm"] / c)
            monthly_pet.append(monthly[key]["pet_mm"] / c)
        moisture_index = np.array(monthly_p) - np.array(monthly_pet)
        p_all = pd.concat(precip_series[gcin], ignore_index=True)
        high_freq, high_dur = spell_stats(p_all, 10.0, wet=True)
        low_freq, low_dur = spell_stats(p_all, 1.0, wet=False)
        rows.append(
            {
                "GCIN": gcin,
                "n_days": n,
                "Prec_mm": p_mean,
                "Temp_C": sums[gcin]["temp_c"] / n,
                "PET_mm": pet_mean,
                "AET_mm": aet_mean,
                "P_AET_mm": p_mean - aet_mean,
                "Aridity": pet_mean / p_mean if p_mean > 0 else np.nan,
                "mean_sm_rootzone": sums[gcin]["sm_rootzone"] / n,
                "max_soil_moisture": sums[gcin]["max_soil_moisture"],
                "Porosity": sums[gcin]["max_soil_moisture"],
                "Annual_Mean_Moisture_Index": float(np.mean(moisture_index)),
                "Seasonality_of_Moisture_Index": float(np.std(moisture_index)),
                "mean_net_radiation_mj_m2_day": sums[gcin]["net_radiation_mj_m2_day"] / n,
                "swvl1_mean": sums[gcin]["swvl1"] / n,
                "swvl2_mean": sums[gcin]["swvl2"] / n,
                "swvl3_mean": sums[gcin]["swvl3"] / n,
                "swvl4_mean": sums[gcin]["swvl4"] / n,
                "wet_days_ratio_1mm": sums[gcin]["wet_days_1mm"] / n,
                "wet_days_ratio_5mm": sums[gcin]["wet_days_5mm"] / n,
                "high_prec_freq_10mm": high_freq / (n / 365.25),
                "high_prec_dur_10mm": high_dur,
                "low_prec_freq_1mm": low_freq / (n / 365.25),
                "low_prec_dur_1mm": low_dur,
            }
        )

    attrs = boundary_attributes(args.boundaries, args.crosswalk).merge(pd.DataFrame(rows), on="GCIN", how="inner")
    attrs["attribute_version"] = "gee_era5land_legacy_1950_2019_v1"
    attrs.to_parquet(args.out_dir / "static_attributes.parquet", index=False)
    attrs.to_csv(args.out_dir / "static_attributes.csv", index=False)
    print(f"wrote {args.out_dir / 'static_attributes.parquet'} rows={len(attrs)}", flush=True)


if __name__ == "__main__":
    main()
