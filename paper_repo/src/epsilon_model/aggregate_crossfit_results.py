from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import load_config, output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-label", type=str, default="crossfit_1990")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    run_root = out / args.run_label
    frames = []
    metrics = []
    for fold in range(int(cfg["splits"]["n_folds"])):
        fold_dir = run_root / f"fold_{fold}"
        summary = pd.read_parquet(fold_dir / "heldout_epsilon_change_summary.parquet")
        frames.append(summary)
        m = pd.read_csv(fold_dir / "metrics.csv")
        m["fold"] = fold
        metrics.append(m)
    all_summary = pd.concat(frames, ignore_index=True)
    all_metrics = pd.concat(metrics, ignore_index=True)
    assignments = pd.read_parquet(out / "inputs" / "fold_assignment.parquet").set_index("GCIN")
    if all_summary["GCIN"].duplicated().any():
        duplicates = sorted(all_summary.loc[all_summary["GCIN"].duplicated(), "GCIN"].astype(int).unique())
        raise RuntimeError(f"Catchments appear in more than one held-out fold: {duplicates[:10]}")
    expected_all = set()
    for fold in range(int(cfg["splits"]["n_folds"])):
        expected_all.update(assignments.index[assignments[f"role_fold_{fold}"] == "test"].astype(int))
    for fold, group in all_summary.groupby("fold", observed=True):
        expected = assignments.loc[group["GCIN"].astype(int), f"role_fold_{int(fold)}"]
        if not bool((expected == "test").all()):
            raise RuntimeError(f"Fold {fold} summary contains non-test catchments")
    observed_all = set(all_summary["GCIN"].dropna().astype(int))
    missing = sorted(expected_all - observed_all)
    unexpected = sorted(observed_all - expected_all)
    if missing or unexpected:
        raise RuntimeError(
            "Cross-fit summary does not match the expected held-out catchments: "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )

    out_dir = run_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summary.to_parquet(out_dir / "crossfit_epsilon_change_summary.parquet", index=False)
    all_summary.to_csv(out_dir / "crossfit_epsilon_change_summary.csv", index=False)
    all_metrics.to_csv(out_dir / "crossfit_training_metrics.csv", index=False)

    stats = all_summary["delta_epsilon_mean"].describe().to_frame("delta_epsilon_mean")
    stats.loc["n_catchments", "delta_epsilon_mean"] = len(all_summary)
    stats.loc["n_valid_delta", "delta_epsilon_mean"] = all_summary["delta_epsilon_mean"].notna().sum()
    stats.loc["mean_pre_epsilon", "delta_epsilon_mean"] = all_summary["pre_epsilon_mean"].mean()
    stats.loc["mean_post_epsilon", "delta_epsilon_mean"] = all_summary["post_epsilon_mean"].mean()
    stats.to_csv(out_dir / "crossfit_delta_epsilon_stats.csv")
    print(stats.to_string())
    print("fold delta means")
    print(all_summary.groupby("fold")["delta_epsilon_mean"].agg(["count", "mean", "std", "min", "max"]).to_string())


if __name__ == "__main__":
    main()
