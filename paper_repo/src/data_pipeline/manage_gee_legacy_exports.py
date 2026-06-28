"""Keep legacy matched GEE exports queued without duplicate submissions."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import ee
import pandas as pd

from gee_era5land_export_legacy_matched_timeseries import active_codes, init_ee, next_month, period_collection, BANDS


DESC_RE = re.compile(r"^era5land_legacy_matched_daily_(\d{4})_(\d{2})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="climate-epsilon-era5land-2026")
    parser.add_argument("--catchments-asset", default="projects/climate-epsilon-era5land-2026/assets/legacy_streamflow_matched_catchments")
    parser.add_argument("--monthly-inventory", type=Path, default=Path("_private/processed/legacy_forcings/timeseries_matched_valid_streamflow_months.csv"))
    parser.add_argument("--drive-folder", default="era5land_legacy_matched_ts")
    parser.add_argument("--tile-scale", type=int, default=4)
    parser.add_argument("--queue-target", type=int, default=120)
    parser.add_argument("--max-submit", type=int, default=96)
    parser.add_argument("--start-year", type=int, default=1950)
    parser.add_argument("--end-year", type=int, default=2019)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def existing_legacy_tasks() -> dict[tuple[int, int], str]:
    existing = {}
    for task in ee.batch.Task.list():
        status = task.status()
        desc = status.get("description", "")
        match = DESC_RE.match(desc)
        if not match:
            continue
        state = status.get("state", "")
        year, month = int(match.group(1)), int(match.group(2))
        existing[(year, month)] = state
    return existing


def print_progress(existing: dict[tuple[int, int], str], expected_months: list[tuple[int, int]]) -> None:
    counts = Counter(existing.get(month, "NOT_SUBMITTED") for month in expected_months)
    submitted = len(expected_months) - counts["NOT_SUBMITTED"]
    completed = counts["COMPLETED"] + counts["SUCCEEDED"]
    active = sum(counts[state] for state in ["READY", "RUNNING", "PENDING"])
    print(f"expected_months={len(expected_months)} submitted={submitted} completed={completed} active={active}")
    print("states=" + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    print(f"submission_progress={submitted / len(expected_months) * 100:.1f}% completion_progress={completed / len(expected_months) * 100:.1f}%")


def main() -> None:
    args = parse_args()
    init_ee(args.project)
    inventory = pd.read_csv(args.monthly_inventory)
    expected = (
        inventory[["year", "month"]]
        .drop_duplicates()
        .query("year >= @args.start_year and year <= @args.end_year")
        .sort_values(["year", "month"])
    )
    expected_months = [(int(row.year), int(row.month)) for row in expected.itertuples(index=False)]
    existing = existing_legacy_tasks()
    print_progress(existing, expected_months)

    queue_states = {"READY", "RUNNING", "PENDING"}
    queued = sum(1 for month in expected_months if existing.get(month) in queue_states)
    room = max(0, args.queue_target - queued)
    submit_limit = min(room, args.max_submit)
    print(f"queued={queued} queue_target={args.queue_target} room={room} submit_limit={submit_limit}")
    if submit_limit <= 0:
        return

    catchment_asset = ee.FeatureCollection(args.catchments_asset)
    submitted = 0
    for year, month in expected_months:
        if existing.get((year, month)) in {"READY", "RUNNING", "PENDING", "COMPLETED", "SUCCEEDED"}:
            continue
        codes = active_codes(inventory, year, month)
        if not codes:
            continue
        start = pd.Timestamp(year=year, month=month, day=1).date()
        end = next_month(year, month)
        desc = f"era5land_legacy_matched_daily_{year:04d}_{month:02d}"
        print(f"submit_candidate {desc} catchments={len(codes)}")
        if not args.dry_run:
            catchments = catchment_asset.filter(ee.Filter.inList("force_code", codes))
            table = period_collection(catchments, start.isoformat(), end.isoformat(), args.tile_scale)
            selectors = ["force_code", "cur_gcin", "date"] + [band.out for band in BANDS]
            task = ee.batch.Export.table.toDrive(
                collection=table,
                description=desc,
                folder=args.drive_folder,
                fileNamePrefix=desc,
                fileFormat="CSV",
                selectors=selectors,
            )
            task.start()
            print(f"submitted {desc} task={task.id}")
        submitted += 1
        if submitted >= submit_limit:
            break
    print(f"newly_submitted={submitted}")


if __name__ == "__main__":
    main()
