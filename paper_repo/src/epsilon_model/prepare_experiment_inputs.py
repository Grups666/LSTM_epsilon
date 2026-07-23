from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import load_config, output_dir
from temporal_split import blocks_frame, expected_fold_for_year


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-max-catchments", type=int, default=None)
    return parser.parse_args()


def build_temporal_fold_assignment(cfg: dict, inputs_dir: Path) -> pd.DataFrame:
    expected_fold_for_year(cfg)
    out = blocks_frame(cfg)
    out_path = inputs_dir / "temporal_fold_assignment.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"wrote {out_path} rows={len(out)}")
    return out


def build_qobs_inventory(cfg: dict, inputs_dir: Path) -> pd.DataFrame:
    daily_dir = Path(cfg["paths"]["daily_dir"])
    rows = []
    gap_state: dict[int, dict[str, object]] = {}
    pre_start, pre_end = pd.to_datetime(cfg["data"]["pre_window"])
    post_start, post_end = pd.to_datetime(cfg["data"]["post_window"])
    pre = []
    post = []

    for year in range(int(cfg["data"]["start_year"]), int(cfg["data"]["end_year"]) + 1):
        path = daily_dir / f"epsilon_training_daily_{year}.parquet"
        df = pd.read_parquet(path, columns=["GCIN", "date", cfg["data"]["target_column"]])
        df["date"] = pd.to_datetime(df["date"])
        df["has_q"] = df[cfg["data"]["target_column"]].notna()
        df["valid_q_date"] = df["date"].where(df["has_q"])
        agg = df.groupby("GCIN", observed=True).agg(
            rows=("has_q", "size"),
            valid_q_days=("has_q", "sum"),
            start=("valid_q_date", "min"),
            end=("valid_q_date", "max"),
        )
        agg["year"] = year
        rows.append(agg.reset_index())

        valid = df.loc[df["has_q"], ["GCIN", "date"]].drop_duplicates()
        if year <= pre_end.year:
            pre.append(valid.loc[(valid["date"] >= pre_start) & (valid["date"] <= pre_end), ["GCIN"]])
        if year >= post_start.year:
            post.append(valid.loc[(valid["date"] >= post_start) & (valid["date"] <= post_end), ["GCIN"]])

        for gcin, group in valid.groupby("GCIN", observed=True, sort=False):
            dates = group["date"].sort_values().reset_index(drop=True)
            if dates.empty:
                continue
            key = int(gcin)
            state = gap_state.setdefault(key, {"previous": None, "gaps": []})
            previous = state["previous"]
            day_deltas = dates.diff().dt.days
            if previous is not None:
                day_deltas.iloc[0] = (dates.iloc[0] - previous).days
            for index in day_deltas.index[day_deltas > 1]:
                date = dates.iloc[index]
                previous_date = previous if index == 0 else dates.iloc[index - 1]
                state["gaps"].append(
                    {
                        "start": (previous_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        "end": (date - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        "days": int(day_deltas.iloc[index] - 1),
                    }
                )
            state["previous"] = dates.iloc[-1]

    yearly = pd.concat(rows, ignore_index=True)
    inv = yearly.groupby("GCIN", observed=True).agg(
        rows=("rows", "sum"),
        valid_q_days=("valid_q_days", "sum"),
        start=("start", "min"),
        end=("end", "max"),
    ).reset_index()
    pre_counts = pd.concat(pre).value_counts("GCIN").rename("pre_valid_q_days")
    post_counts = pd.concat(post).value_counts("GCIN").rename("post_valid_q_days")
    inv = inv.merge(pre_counts, on="GCIN", how="left").merge(post_counts, on="GCIN", how="left")
    inv[["pre_valid_q_days", "post_valid_q_days"]] = inv[["pre_valid_q_days", "post_valid_q_days"]].fillna(0).astype("int32")

    inv["calendar_days"] = (inv["end"] - inv["start"]).dt.days + 1
    inv["missing_q_days"] = (inv["calendar_days"] - inv["valid_q_days"]).clip(lower=0).astype("int32")
    inv["qobs_coverage_pct"] = (100 * inv["valid_q_days"] / inv["calendar_days"]).round(4)
    inv["qobs_missing_pct"] = (100 * inv["missing_q_days"] / inv["calendar_days"]).round(4)
    inv["qobs_gap_runs"] = inv["GCIN"].map(
        lambda gcin: len(gap_state.get(int(gcin), {}).get("gaps", []))
    ).astype("int32")
    inv["qobs_long_gap_count"] = inv["GCIN"].map(
        lambda gcin: sum(
            gap["days"] >= 30
            for gap in gap_state.get(int(gcin), {}).get("gaps", [])
        )
    ).astype("int32")
    inv["qobs_long_gaps_json"] = inv["GCIN"].map(
        lambda gcin: json.dumps(
            sorted(
                (
                    gap
                    for gap in gap_state.get(int(gcin), {}).get("gaps", [])
                    if gap["days"] >= 30
                ),
                key=lambda gap: (-gap["days"], gap["start"]),
            ),
            separators=(",", ":"),
        )
    )

    out_path = inputs_dir / "qobs_inventory.parquet"
    inv.to_parquet(out_path, index=False)
    inv.to_csv(inputs_dir / "qobs_inventory.csv", index=False)
    print(f"wrote {out_path} rows={len(inv)} valid_q_total={int(inv.valid_q_days.sum())}")
    return inv


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    inputs_dir = output_dir(cfg) / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    build_temporal_fold_assignment(cfg, inputs_dir)
    build_qobs_inventory(cfg, inputs_dir)


if __name__ == "__main__":
    main()
