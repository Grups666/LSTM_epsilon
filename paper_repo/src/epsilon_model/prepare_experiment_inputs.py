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
    validation_offset = int(cfg["splits"].get("validation_fold_offset", 1))
    validation_fraction = float(cfg["splits"].get("validation_fraction_of_fold", 0.5))
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction_of_fold must be between 0 and 1")

    for test_fold in range(n_folds):
        role = np.full(len(out), "train", dtype=object)
        test_mask = folds == test_fold
        role[test_mask] = "test"

        validation_fold = (test_fold + validation_offset) % n_folds
        validation_candidates = np.flatnonzero(folds == validation_fold)
        rng = np.random.default_rng(seed + test_fold)
        rng.shuffle(validation_candidates)
        n_validation = max(1, int(round(validation_fraction * len(validation_candidates))))
        role[validation_candidates[:n_validation]] = "validation"
        out[f"role_fold_{test_fold}"] = role

        counts = pd.Series(role).value_counts()
        print(
            f"crossfit fold {test_fold}: "
            f"train={int(counts.get('train', 0))} "
            f"validation={int(counts.get('validation', 0))} "
            f"test={int(counts.get('test', 0))}"
        )

    role_columns = [f"role_fold_{fold}" for fold in range(n_folds)]
    allowed_roles = {"train", "validation", "test"}
    if any(set(out[col].unique()) != allowed_roles for col in role_columns):
        raise RuntimeError("Every crossfit fold must contain train, validation, and test basins")
    test_counts = (out[role_columns] == "test").sum(axis=1)
    if not bool((test_counts == 1).all()):
        raise RuntimeError("Every basin must be held out as test exactly once across the five folds")
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
        df["valid_q_date"] = pd.to_datetime(df["date"]).where(df["has_q"])
        agg = df.groupby("GCIN", observed=True).agg(
            rows=("has_q", "size"),
            valid_q_days=("has_q", "sum"),
            start=("valid_q_date", "min"),
            end=("valid_q_date", "max"),
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    inputs_dir = output_dir(cfg) / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    build_fold_assignment(cfg, inputs_dir)
    build_qobs_inventory(cfg, inputs_dir)


if __name__ == "__main__":
    main()
