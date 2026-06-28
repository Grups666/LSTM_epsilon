"""Prepare legacy catchments for GEE export.

The asset uses legacy GPKG geometries, but carries both identifiers:
- force_code: legacy forcing CSV code
- cur_gcin: current Qobs GCIN found by streamflow time-series matching
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-boundaries", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, default=Path("_private/processed/legacy_forcings/timeseries_matched_forcing_code_to_current_gcin_crosswalk.csv"))
    parser.add_argument("--out", type=Path, default=Path("_private/gee/legacy_streamflow_matched_catchments_gee.zip"))
    parser.add_argument("--simplify-tolerance", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    work = out.with_suffix("")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    cross = pd.read_csv(args.crosswalk)
    cross = cross[["forcing_code", "GCIN"]].copy()
    cross["forcing_code"] = cross["forcing_code"].astype("int32")
    cross["GCIN"] = cross["GCIN"].astype("int32")

    gdf = gpd.read_file(args.legacy_boundaries)
    gdf["forcing_code"] = pd.to_numeric(gdf["GCIN"], errors="coerce").astype("Int64")
    gdf = gdf[gdf["forcing_code"].notna()].copy()
    gdf["forcing_code"] = gdf["forcing_code"].astype("int32")
    gdf = gdf.merge(cross, on="forcing_code", how="inner", validate="one_to_one")
    gdf = gdf[["forcing_code", "GCIN_y", "geometry"]].rename(
        columns={"forcing_code": "force_code", "GCIN_y": "cur_gcin"}
    )
    gdf = gdf.dropna(subset=["force_code", "cur_gcin", "geometry"]).copy()
    gdf["force_code"] = gdf["force_code"].astype("int32")
    gdf["cur_gcin"] = gdf["cur_gcin"].astype("int32")
    gdf = gdf.to_crs("EPSG:4326")
    gdf["geometry"] = gdf.geometry.make_valid()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    if args.simplify_tolerance > 0:
        gdf["geometry"] = gdf.geometry.simplify(args.simplify_tolerance, preserve_topology=True)

    shp = work / "legacy_streamflow_matched_catchments.shp"
    gdf.to_file(shp, driver="ESRI Shapefile", encoding="UTF-8")
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in work.iterdir():
            zf.write(path, arcname=path.name)
    print(f"Wrote {out} with {len(gdf)} catchments")


if __name__ == "__main__":
    main()
