from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import load_config, output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-max-catchments", type=int, default=None)
    return parser.parse_args()


def make_strata(static: pd.DataFrame, columns: list[str]) -> pd.Series:
    parts: list[pd.Series] = []
    for col in columns:
        s = static[col]
        if col == "area_km2":
            s = np.log10(s.clip(lower=1e-6))
        try:
            binned = pd.qcut(s, q=4, duplicates="drop").astype(str)
            binned = binned.where(s.notna(), "missing")
        except ValueError:
            binned = pd.Series(["all"] * len(static), index=static.index)
        parts.append(binned)
    strata = parts[0]
    for p in parts[1:]:
        strata = strata + "|" + p
    counts = strata.value_counts()
    rare = strata.map(counts) < 5
    strata = strata.mask(rare, "rare")
    return strata


def build_fold_assignment(cfg: dict, inputs_dir: Path) -> pd.DataFrame:
    static = pd.read_parquet(cfg["paths"]["static_attributes"])
    static = static.sort_values("GCIN").reset_index(drop=True)
    strata = make_strata(static, cfg["splits"]["stratify_columns"])

    n_folds = int(cfg["splits"]["n_folds"])
    seed = int(cfg["seed"])
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.empty(len(static), dtype=np.int16)
    for fold_id, (_, test_idx) in enumerate(splitter.split(static, strata)):
        folds[test_idx] = fold_id

    out = static[["GCIN", *cfg["splits"]["stratify_columns"]]].copy()
    out["fold"] = folds
    out["stratum"] = strata
    out_path = inputs_dir / "fold_assignment.parquet"
    out.to_parquet(out_path, index=False)
    out.to_csv(inputs_dir / "fold_assignment.csv", index=False)
    print(f"wrote {out_path} rows={len(out)}")
    return out


def build_qobs_inventory(cfg: dict, inputs_dir: Path) -> pd.DataFrame:
    daily_dir = Path(cfg["paths"]["daily_dir"])
    rows = []
    for year in range(int(cfg["data"]["start_year"]), int(cfg["data"]["end_year"]) + 1):
        path = daily_dir / f"epsilon_training_daily_{year}.parquet"
        df = pd.read_parquet(path, columns=["GCIN", "date", cfg["data"]["target_column"]])
        df["has_q"] = df[cfg["data"]["target_column"]].notna()
        agg = df.groupby("GCIN", observed=True).agg(
            rows=("has_q", "size"),
            valid_q_days=("has_q", "sum"),
            start=("date", "min"),
            end=("date", "max"),
        )
        agg["year"] = year
        rows.append(agg.reset_index())
    yearly = pd.concat(rows, ignore_index=True)
    inv = yearly.groupby("GCIN", observed=True).agg(
        rows=("rows", "sum"),
        valid_q_days=("valid_q_days", "sum"),
        start=("start", "min"),
        end=("end", "max"),
    ).reset_index()
    pre_start, pre_end = pd.to_datetime(cfg["data"]["pre_window"])
    post_start, post_end = pd.to_datetime(cfg["data"]["post_window"])
    pre = []
    post = []
    for year in range(pre_start.year, post_end.year + 1):
        path = daily_dir / f"epsilon_training_daily_{year}.parquet"
        df = pd.read_parquet(path, columns=["GCIN", "date", cfg["data"]["target_column"]])
        date = pd.to_datetime(df["date"])
        has_q = df[cfg["data"]["target_column"]].notna()
        if year <= pre_end.year:
            pre.append(df.loc[(date >= pre_start) & (date <= pre_end) & has_q, ["GCIN"]])
        if year >= post_start.year:
            post.append(df.loc[(date >= post_start) & (date <= post_end) & has_q, ["GCIN"]])
    pre_counts = pd.concat(pre).value_counts("GCIN").rename("pre_valid_q_days")
    post_counts = pd.concat(post).value_counts("GCIN").rename("post_valid_q_days")
    inv = inv.merge(pre_counts, on="GCIN", how="left").merge(post_counts, on="GCIN", how="left")
    inv[["pre_valid_q_days", "post_valid_q_days"]] = inv[["pre_valid_q_days", "post_valid_q_days"]].fillna(0).astype("int32")
    out_path = inputs_dir / "qobs_inventory.parquet"
    inv.to_parquet(out_path, index=False)
    inv.to_csv(inputs_dir / "qobs_inventory.csv", index=False)
    print(f"wrote {out_path} rows={len(inv)} valid_q_total={int(inv.valid_q_days.sum())}")
    return inv


def recession_mask(q: np.ndarray, precip: np.ndarray, min_days: int, drop_first: bool, max_precip: float) -> np.ndarray:
    """Legacy recession helper retained for older input-preparation outputs.

    The current Ara-aligned training path uses detect_recession_paper() in
    train_epsilon_model.py, with optional snow-month masking and no precipitation
    threshold filter.
    """
    valid = np.isfinite(q)
    decline = np.zeros(len(q), dtype=bool)
    decline[1:] = valid[1:] & valid[:-1] & (q[1:] < q[:-1])
    dry = np.isfinite(precip) & (precip <= max_precip)
    candidate = decline & dry
    out = np.zeros(len(q), dtype=bool)
    i = 0
    while i < len(candidate):
        if not candidate[i]:
            i += 1
            continue
        j = i
        while j < len(candidate) and candidate[j]:
            j += 1
        if j - i >= min_days:
            start = i + 1 if drop_first else i
            out[start:j] = True
        i = j
    return out


def build_recession_days(cfg: dict, inputs_dir: Path) -> None:
    daily_dir = Path(cfg["paths"]["daily_dir"])
    rec_dir = inputs_dir / "qobs_recession_days_by_year"
    rec_dir.mkdir(parents=True, exist_ok=True)
    target = cfg["data"]["target_column"]
    params = cfg["recession"]
    total = 0
    for year in range(int(cfg["data"]["start_year"]), int(cfg["data"]["end_year"]) + 1):
        path = daily_dir / f"epsilon_training_daily_{year}.parquet"
        df = pd.read_parquet(path, columns=["GCIN", "date", "tp", target])
        pieces = []
        for gcin, group in df.groupby("GCIN", sort=False, observed=True):
            q = group[target].to_numpy(dtype="float64")
            precip = group["tp"].to_numpy(dtype="float64")
            mask = recession_mask(
                q,
                precip,
                int(params["min_decline_days"]),
                bool(params["drop_first_decline_day"]),
                float(params.get("max_precip_mm", 1.0)),
            )
            if mask.any():
                pieces.append(group.loc[mask, ["GCIN", "date", target]].assign(recession_day=True))
        out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["GCIN", "date", target, "recession_day"])
        out_path = rec_dir / f"qobs_recession_days_{year}.parquet"
        out.to_parquet(out_path, index=False)
        total += len(out)
        print(f"wrote {out_path} rows={len(out)}")
    print(f"recession_days_total={total}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    inputs_dir = output_dir(cfg) / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    build_fold_assignment(cfg, inputs_dir)
    build_qobs_inventory(cfg, inputs_dir)
    build_recession_days(cfg, inputs_dir)


if __name__ == "__main__":
    main()
