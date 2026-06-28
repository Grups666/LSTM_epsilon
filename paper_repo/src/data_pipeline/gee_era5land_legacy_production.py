"""Production GEE pipeline for legacy streamflow-matched catchments.

This script is designed to run on the Tsinghua server. It submits monthly
ERA5-Land daily catchment exports only for catchments with observed streamflow
in that month, tracks task status in a manifest, and converts downloaded Drive
CSVs to parquet files.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import re
from dataclasses import dataclass
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Iterable

import ee
import pandas as pd
import requests
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


PROJECT = "climate-epsilon-gee-2026"
ASSET_ID = "projects/climate-epsilon-gee-2026/assets/legacy_streamflow_matched_catchments"
DRIVE_FOLDER = "era5land_legacy_matched_ts_v2"
COLLECTION_ID = "ECMWF/ERA5_LAND/DAILY_AGGR"
SCALE_METERS = 11132
DESC_PREFIX = "era5land_legacy_matched_daily"
DESC_RE = re.compile(rf"^{DESC_PREFIX}_(\d{{4}})_(\d{{2}})$")


@dataclass(frozen=True)
class Band:
    source: str
    out: str
    multiplier: float = 1.0


BANDS = [
    Band("temperature_2m", "t2m"),
    Band("dewpoint_temperature_2m", "d2m"),
    Band("u_component_of_wind_10m", "u10"),
    Band("v_component_of_wind_10m", "v10"),
    Band("surface_pressure", "sp"),
    Band("skin_temperature", "skt"),
    Band("volumetric_soil_water_layer_1", "swvl1"),
    Band("volumetric_soil_water_layer_2", "swvl2"),
    Band("volumetric_soil_water_layer_3", "swvl3"),
    Band("volumetric_soil_water_layer_4", "swvl4"),
    Band("leaf_area_index_high_vegetation", "lai_hv"),
    Band("leaf_area_index_low_vegetation", "lai_lv"),
    Band("total_precipitation_sum", "tp", 1000.0),
    Band("surface_net_solar_radiation_sum", "ssr"),
    Band("surface_net_thermal_radiation_sum", "str"),
    Band("total_evaporation_sum", "aet", -1000.0),
]

VARIABLE_COLUMNS = [band.out for band in BANDS]
SELECTORS = ["force_code", "cur_gcin", "date"] + VARIABLE_COLUMNS
MANIFEST_COLUMNS = [
    "year",
    "month",
    "description",
    "task_id",
    "state",
    "submitted_at",
    "updated_at",
    "completed_at",
    "error_message",
    "drive_folder",
    "file_prefix",
    "active_catchments",
    "expected_days",
    "expected_rows",
    "local_csv",
    "monthly_parquet",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def init_ee(project: str) -> None:
    ee.Initialize(project=project)


def month_end(year: int, month: int) -> date:
    if month == 12:
        return date(year + 1, 1, 1)
    return date(year, month + 1, 1)


def description(year: int, month: int) -> str:
    return f"{DESC_PREFIX}_{year:04d}_{month:02d}"


def parse_months(value: str) -> list[int]:
    out: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = part.split("-", 1)
            out.update(range(int(first), int(last) + 1))
        else:
            out.add(int(part))
    months = sorted(out)
    bad = [m for m in months if m < 1 or m > 12]
    if bad:
        raise ValueError(f"months out of range 1-12: {bad}")
    return months


def load_inventory(path: Path, start_year: int, end_year: int) -> pd.DataFrame:
    inventory = pd.read_csv(path)
    needed = {"year", "month", "force_code", "cur_gcin", "valid_streamflow_days"}
    missing = needed - set(inventory.columns)
    if missing:
        raise ValueError(f"inventory missing columns: {sorted(missing)}")
    inventory = inventory[(inventory["year"] >= start_year) & (inventory["year"] <= end_year)].copy()
    inventory["year"] = inventory["year"].astype(int)
    inventory["month"] = inventory["month"].astype(int)
    inventory["force_code"] = inventory["force_code"].astype(int)
    inventory["cur_gcin"] = inventory["cur_gcin"].astype(int)
    return inventory


def expected_months(inventory: pd.DataFrame, months: Iterable[int]) -> list[tuple[int, int]]:
    month_set = set(months)
    rows = (
        inventory[inventory["month"].isin(month_set)][["year", "month"]]
        .drop_duplicates()
        .sort_values(["year", "month"])
    )
    return [(int(row.year), int(row.month)) for row in rows.itertuples(index=False)]


def active_codes(inventory: pd.DataFrame, year: int, month: int) -> list[int]:
    rows = inventory[(inventory["year"] == year) & (inventory["month"] == month)]
    return sorted(rows["force_code"].astype(int).unique().tolist())


def expected_row_count(inventory: pd.DataFrame, year: int, month: int) -> tuple[int, int, int]:
    rows = inventory[(inventory["year"] == year) & (inventory["month"] == month)]
    active = int(rows["force_code"].nunique())
    days = calendar.monthrange(year, month)[1]
    return active, days, active * days


def prepared_image(image: ee.Image) -> ee.Image:
    pieces = []
    for band in BANDS:
        img = image.select(band.source).rename(band.out)
        if band.multiplier != 1.0:
            img = img.multiply(band.multiplier).rename(band.out)
        pieces.append(img)
    return ee.Image.cat(pieces).copyProperties(image, ["system:time_start"])


def period_collection(catchments: ee.FeatureCollection, start: str, end: str, tile_scale: int) -> ee.FeatureCollection:
    images = ee.ImageCollection(COLLECTION_ID).filterDate(start, end).map(prepared_image)

    def reduce_one(image: ee.Image) -> ee.FeatureCollection:
        image_date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
        reduced = image.reduceRegions(
            collection=catchments,
            reducer=ee.Reducer.mean(),
            scale=SCALE_METERS,
            crs="EPSG:4326",
            tileScale=tile_scale,
        )
        return reduced.map(lambda feature: feature.set("date", image_date))

    return ee.FeatureCollection(images.map(reduce_one).flatten())


def read_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    df = pd.read_csv(path, dtype={"task_id": "string", "error_message": "string"})
    for column in MANIFEST_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[MANIFEST_COLUMNS]


def write_manifest(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    for column in MANIFEST_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df = df[MANIFEST_COLUMNS].sort_values(["year", "month", "submitted_at"], na_position="last")
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def upsert_manifest(path: Path, rows: list[dict[str, object]]) -> pd.DataFrame:
    manifest = read_manifest(path)
    incoming = pd.DataFrame(rows)
    if incoming.empty:
        return manifest
    for column in MANIFEST_COLUMNS:
        if column not in incoming.columns:
            incoming[column] = ""
    if manifest.empty:
        out = incoming[MANIFEST_COLUMNS]
    else:
        out = pd.concat([manifest, incoming[MANIFEST_COLUMNS]], ignore_index=True)
        out = out.drop_duplicates(subset=["year", "month", "description"], keep="last")
    write_manifest(path, out)
    return out


def task_statuses(project: str) -> dict[str, dict[str, object]]:
    init_ee(project)
    statuses = {}
    for task in ee.batch.Task.list():
        status = task.status()
        desc = status.get("description")
        if desc and DESC_RE.match(desc):
            statuses[desc] = status
    return statuses


def cmd_auth_test(args: argparse.Namespace) -> None:
    init_ee(args.project)
    print(f"ee_number={ee.Number(1).getInfo()}")


def cmd_asset_test(args: argparse.Namespace) -> None:
    init_ee(args.project)
    fc = ee.FeatureCollection(args.catchments_asset)
    first = fc.first().getInfo()
    props = first.get("properties", {}) if first else {}
    count = fc.size().getInfo()
    print(f"asset={args.catchments_asset}")
    print(f"count={count}")
    print(f"properties={sorted(props.keys())}")
    for field in ["force_code", "cur_gcin"]:
        if field not in props:
            raise SystemExit(f"missing required asset field: {field}")


def cmd_submit(args: argparse.Namespace) -> None:
    init_ee(args.project)
    inventory = load_inventory(args.monthly_inventory, args.start_year, args.end_year)
    months = parse_months(args.months)
    expected = expected_months(inventory, months)
    manifest = read_manifest(args.manifest)
    status_by_desc = task_statuses(args.project)

    completed_or_active = set()
    if not manifest.empty:
        ok_states = {"READY", "RUNNING", "PENDING", "COMPLETED", "SUCCEEDED"}
        completed_or_active.update(manifest[manifest["state"].isin(ok_states)]["description"].astype(str).tolist())
    for desc, status in status_by_desc.items():
        if status.get("state") in {"READY", "RUNNING", "PENDING", "COMPLETED", "SUCCEEDED"}:
            completed_or_active.add(desc)

    active_states = {"READY", "RUNNING", "PENDING"}
    active_now = sum(1 for status in status_by_desc.values() if status.get("state") in active_states)
    submit_room = max(0, args.queue_target - active_now)
    submit_limit = min(args.max_submit, submit_room)
    print(f"expected_months={len(expected)} active_now={active_now} queue_target={args.queue_target} submit_limit={submit_limit}")
    if submit_limit <= 0:
        return

    catchment_asset = ee.FeatureCollection(args.catchments_asset)
    rows = []
    submitted = 0
    for year, month in expected:
        desc = description(year, month)
        if desc in completed_or_active:
            continue
        codes = active_codes(inventory, year, month)
        if not codes:
            continue
        active, days, expected_rows = expected_row_count(inventory, year, month)
        catchments = catchment_asset.filter(ee.Filter.inList("force_code", codes))
        start = date(year, month, 1)
        table = period_collection(catchments, start.isoformat(), month_end(year, month).isoformat(), args.tile_scale)
        task = ee.batch.Export.table.toDrive(
            collection=table,
            description=desc,
            folder=args.drive_folder,
            fileNamePrefix=desc,
            fileFormat="CSV",
            selectors=SELECTORS,
        )
        if args.dry_run:
            task_id = "DRY_RUN"
            state = "DRY_RUN"
        else:
            task.start()
            task_id = task.id
            state = "READY"
        row = {
            "year": year,
            "month": month,
            "description": desc,
            "task_id": task_id,
            "state": state,
            "submitted_at": utc_now(),
            "updated_at": utc_now(),
            "completed_at": "",
            "error_message": "",
            "drive_folder": args.drive_folder,
            "file_prefix": desc,
            "active_catchments": active,
            "expected_days": days,
            "expected_rows": expected_rows,
            "local_csv": "",
            "monthly_parquet": "",
        }
        rows.append(row)
        submitted += 1
        print(f"submitted {desc}: catchments={active} expected_rows={expected_rows} task={task_id}")
        if submitted >= submit_limit:
            break
    upsert_manifest(args.manifest, rows)
    print(f"newly_submitted={submitted}")


def cmd_status(args: argparse.Namespace) -> None:
    inventory = load_inventory(args.monthly_inventory, args.start_year, args.end_year)
    months = parse_months(args.months)
    expected = {description(year, month) for year, month in expected_months(inventory, months)}
    manifest = read_manifest(args.manifest)
    status_by_desc = task_statuses(args.project)
    rows = []
    for _, row in manifest.iterrows():
        desc = str(row["description"])
        status = status_by_desc.get(desc)
        new = row.to_dict()
        if status:
            state = str(status.get("state", ""))
            new["state"] = state
            new["updated_at"] = utc_now()
            new["error_message"] = status.get("error_message", "") or status.get("error", "") or ""
            if state in {"COMPLETED", "SUCCEEDED", "FAILED", "CANCELLED"} and not str(new.get("completed_at") or ""):
                new["completed_at"] = utc_now()
        rows.append(new)
    if rows:
        write_manifest(args.manifest, pd.DataFrame(rows))
        manifest = read_manifest(args.manifest)
    counts = manifest[manifest["description"].isin(expected)]["state"].value_counts(dropna=False).to_dict()
    completed = int(sum(counts.get(s, 0) for s in ["COMPLETED", "SUCCEEDED"]))
    active = int(sum(counts.get(s, 0) for s in ["READY", "RUNNING", "PENDING"]))
    failed = int(sum(counts.get(s, 0) for s in ["FAILED", "CANCELLED"]))
    submitted = int(manifest[manifest["description"].isin(expected)]["description"].nunique())
    remaining = len(expected) - completed - failed
    print(f"expected_months={len(expected)} submitted={submitted} completed={completed} active={active} failed={failed} remaining={remaining}")
    print("states=" + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    if not manifest.empty:
        latest = manifest.sort_values(["updated_at", "submitted_at"], na_position="first").tail(10)
        print(latest[["year", "month", "state", "description", "error_message"]].to_string(index=False))


def cmd_convert_csv(args: argparse.Namespace) -> None:
    csv_dir = args.csv_dir
    out_dir = args.monthly_parquet_dir
    csv_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.manifest)
    rows = []
    for csv_path in sorted(csv_dir.glob(f"{DESC_PREFIX}_*.csv")):
        match = DESC_RE.match(csv_path.stem)
        if not match:
            continue
        year, month = int(match.group(1)), int(match.group(2))
        pq_path = out_dir / f"era5land_legacy_daily_{year:04d}_{month:02d}.parquet"
        df = pd.read_csv(csv_path)
        missing = [col for col in SELECTORS if col not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} missing columns: {missing}")
        df = df[SELECTORS].copy()
        df["force_code"] = df["force_code"].astype("int64")
        df["cur_gcin"] = df["cur_gcin"].astype("int64")
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype("string")
        df.to_parquet(pq_path, index=False)
        if args.delete_csv:
            csv_path.unlink()
        desc = description(year, month)
        base = manifest[manifest["description"] == desc].tail(1).to_dict("records")
        row = base[0] if base else {
            "year": year,
            "month": month,
            "description": desc,
            "task_id": "",
            "state": "CSV_CONVERTED",
            "submitted_at": "",
            "updated_at": utc_now(),
            "completed_at": "",
            "error_message": "",
            "drive_folder": args.drive_folder,
            "file_prefix": desc,
            "active_catchments": "",
            "expected_days": calendar.monthrange(year, month)[1],
            "expected_rows": len(df),
        }
        row["local_csv"] = "" if args.delete_csv else str(csv_path)
        row["monthly_parquet"] = str(pq_path)
        row["updated_at"] = utc_now()
        rows.append(row)
        print(f"converted {csv_path.name}: rows={len(df)} -> {pq_path}")
    upsert_manifest(args.manifest, rows)
    print(f"converted_files={len(rows)}")


def drive_service():
    credentials = ee.data.get_persistent_credentials()
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def drive_folder_id(service, folder_name: str) -> str:
    safe_name = folder_name.replace("'", "\\'")
    query = (
        "mimeType='application/vnd.google-apps.folder' and "
        f"name='{safe_name}' and trashed=false"
    )
    response = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id,name,modifiedTime)",
        pageSize=10,
        orderBy="modifiedTime desc",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = response.get("files", [])
    if not files:
        raise FileNotFoundError(f"Drive folder not found: {folder_name}")
    return files[0]["id"]


def list_drive_csvs(service, folder_id: str) -> list[dict[str, str]]:
    query = f"'{folder_id}' in parents and trashed=false and mimeType='text/csv'"
    files: list[dict[str, str]] = []
    token = None
    while True:
        response = service.files().list(
            q=query,
            spaces="drive",
            fields="nextPageToken,files(id,name,size,modifiedTime)",
            pageSize=1000,
            pageToken=token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        files.extend(response.get("files", []))
        token = response.get("nextPageToken")
        if not token:
            return sorted(files, key=lambda item: item["name"])


def download_drive_file(service, file_id: str, out_path: Path) -> int:
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        downloader = MediaIoBaseDownload(handle, request, chunksize=32 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return out_path.stat().st_size


def cmd_download_drive(args: argparse.Namespace) -> None:
    init_ee(args.project)
    service = drive_service()
    folder_id = args.drive_folder_id or drive_folder_id(service, args.drive_folder)
    files = list_drive_csvs(service, folder_id)
    args.csv_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.manifest)
    wanted = set(manifest["file_prefix"].dropna().astype(str).tolist()) if not manifest.empty else set()
    downloaded = 0
    for item in files:
        stem = Path(item["name"]).stem
        if wanted and stem not in wanted:
            continue
        if args.year and not stem.startswith(f"{DESC_PREFIX}_{args.year:04d}_"):
            continue
        out_path = args.csv_dir / item["name"]
        if out_path.exists() and not args.overwrite:
            print(f"skip_existing {out_path}")
            continue
        size = download_drive_file(service, item["id"], out_path)
        downloaded += 1
        print(f"downloaded {item['name']} bytes={size}")
        if args.max_files and downloaded >= args.max_files:
            break
    print(f"downloaded_files={downloaded} folder_id={folder_id}")


def cmd_download_ee_table(args: argparse.Namespace) -> None:
    init_ee(args.project)
    inventory = load_inventory(args.monthly_inventory, args.start_year, args.end_year)
    args.csv_dir.mkdir(parents=True, exist_ok=True)
    catchment_asset = ee.FeatureCollection(args.catchments_asset)
    months = parse_months(args.months)
    downloaded = 0
    rows = []
    for year, month in expected_months(inventory, months):
        desc = description(year, month)
        csv_path = args.csv_dir / f"{desc}.csv"
        if csv_path.exists() and not args.overwrite:
            print(f"skip_existing {csv_path}")
            continue
        codes = active_codes(inventory, year, month)
        if not codes:
            continue
        active, days, expected_rows = expected_row_count(inventory, year, month)
        catchments = catchment_asset.filter(ee.Filter.inList("force_code", codes))
        start = date(year, month, 1)
        table = period_collection(catchments, start.isoformat(), month_end(year, month).isoformat(), args.tile_scale)
        download_id = ee.data.getTableDownloadId(
            {
                "table": table,
                "format": "CSV",
                "selectors": ",".join(SELECTORS),
                "filename": desc,
            }
        )
        url = ee.data.makeTableDownloadUrl(download_id)
        with requests.get(url, stream=True, timeout=(30, 600)) as response:
            response.raise_for_status()
            with csv_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=16 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        size = csv_path.stat().st_size
        print(f"downloaded {desc}: catchments={active} expected_rows={expected_rows} bytes={size}")
        rows.append(
            {
                "year": year,
                "month": month,
                "description": desc,
                "task_id": "EE_TABLE_DOWNLOAD",
                "state": "DOWNLOADED",
                "submitted_at": "",
                "updated_at": utc_now(),
                "completed_at": utc_now(),
                "error_message": "",
                "drive_folder": "",
                "file_prefix": desc,
                "active_catchments": active,
                "expected_days": days,
                "expected_rows": expected_rows,
                "local_csv": str(csv_path),
                "monthly_parquet": "",
            }
        )
        downloaded += 1
        if args.max_files and downloaded >= args.max_files:
            break
    upsert_manifest(args.manifest, rows)
    print(f"downloaded_files={downloaded}")


def cmd_assemble_years(args: argparse.Namespace) -> None:
    args.yearly_parquet_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args.manifest)
    for year in range(args.start_year, args.end_year + 1):
        files = sorted(args.monthly_parquet_dir.glob(f"era5land_legacy_daily_{year:04d}_*.parquet"))
        if not files:
            continue
        frames = [pd.read_parquet(path) for path in files]
        df = pd.concat(frames, ignore_index=True)
        duplicate_count = int(df.duplicated(["force_code", "cur_gcin", "date"]).sum())
        if duplicate_count:
            raise ValueError(f"{year} has duplicate force_code/cur_gcin/date rows: {duplicate_count}")
        out = args.yearly_parquet_dir / f"era5land_legacy_daily_{year:04d}.parquet"
        df.sort_values(["force_code", "cur_gcin", "date"]).to_parquet(out, index=False)
        expected = manifest[manifest["year"].astype(str) == str(year)]["expected_rows"]
        expected_rows = int(pd.to_numeric(expected, errors="coerce").fillna(0).sum())
        print(f"assembled {year}: months={len(files)} rows={len(df)} expected_rows={expected_rows} -> {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--catchments-asset", default=ASSET_ID)
    parser.add_argument("--monthly-inventory", type=Path, default=Path(r"D:\ClimateEpsilon\data\timeseries_matched_valid_streamflow_months.csv"))
    parser.add_argument("--manifest", type=Path, default=Path(r"D:\ClimateEpsilon\logs\gee_era5land_legacy_task_manifest.csv"))
    parser.add_argument("--drive-folder", default=DRIVE_FOLDER)
    parser.add_argument("--start-year", type=int, default=1950)
    parser.add_argument("--end-year", type=int, default=2019)
    parser.add_argument("--months", default="1-12")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("auth-test").set_defaults(func=cmd_auth_test)
    sub.add_parser("asset-test").set_defaults(func=cmd_asset_test)

    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--queue-target", type=int, default=20)
    p_submit.add_argument("--max-submit", type=int, default=1)
    p_submit.add_argument("--tile-scale", type=int, default=4)
    p_submit.add_argument("--dry-run", action="store_true")
    p_submit.set_defaults(func=cmd_submit)

    sub.add_parser("status").set_defaults(func=cmd_status)

    p_convert = sub.add_parser("convert-csv")
    p_convert.add_argument("--csv-dir", type=Path, default=Path(r"D:\ClimateEpsilon\downloads\gee_era5land_drive_csv"))
    p_convert.add_argument("--monthly-parquet-dir", type=Path, default=Path(r"D:\ClimateEpsilon\processed\gee_era5land_daily_monthly"))
    p_convert.add_argument("--delete-csv", action="store_true")
    p_convert.set_defaults(func=cmd_convert_csv)

    p_download = sub.add_parser("download-drive")
    p_download.add_argument("--csv-dir", type=Path, default=Path(r"D:\ClimateEpsilon\downloads\gee_era5land_drive_csv"))
    p_download.add_argument("--drive-folder-id")
    p_download.add_argument("--year", type=int)
    p_download.add_argument("--max-files", type=int, default=0)
    p_download.add_argument("--overwrite", action="store_true")
    p_download.set_defaults(func=cmd_download_drive)

    p_ee_download = sub.add_parser("download-ee-table")
    p_ee_download.add_argument("--csv-dir", type=Path, default=Path(r"D:\ClimateEpsilon\downloads\gee_era5land_direct_csv"))
    p_ee_download.add_argument("--tile-scale", type=int, default=4)
    p_ee_download.add_argument("--max-files", type=int, default=0)
    p_ee_download.add_argument("--overwrite", action="store_true")
    p_ee_download.set_defaults(func=cmd_download_ee_table)

    p_assemble = sub.add_parser("assemble-years")
    p_assemble.add_argument("--monthly-parquet-dir", type=Path, default=Path(r"D:\ClimateEpsilon\processed\gee_era5land_daily_monthly"))
    p_assemble.add_argument("--yearly-parquet-dir", type=Path, default=Path(r"D:\ClimateEpsilon\processed\gee_era5land_daily_yearly"))
    p_assemble.set_defaults(func=cmd_assemble_years)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
