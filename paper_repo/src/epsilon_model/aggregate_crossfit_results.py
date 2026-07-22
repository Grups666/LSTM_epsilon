from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from config import load_config, output_dir
from temporal_split import expected_fold_for_year


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-label", type=str, default="temporal_crossfit_1990")
    return parser.parse_args()


def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() < 2:
        return np.nan
    obs = obs[valid]
    pred = pred[valid]
    denominator = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - np.sum((pred - obs) ** 2) / denominator) if denominator > 0 else np.nan


def kge(obs: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() < 2:
        return np.nan
    obs = obs[valid]
    pred = pred[valid]
    obs_std = obs.std(ddof=1)
    pred_std = pred.std(ddof=1)
    obs_mean = obs.mean()
    pred_mean = pred.mean()
    if obs_std <= 0 or pred_std <= 0 or obs_mean == 0:
        return np.nan
    correlation = float(
        ((obs - obs_mean) * (pred - pred_mean)).sum()
        / ((len(obs) - 1) * obs_std * pred_std)
    )
    alpha = pred_std / obs_std
    beta = pred_mean / obs_mean
    return float(1.0 - np.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def summarize_oof(sim: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    pre_start, pre_end = pd.to_datetime(cfg["data"]["pre_window"])
    post_start, post_end = pd.to_datetime(cfg["data"]["post_window"])
    sim = sim.copy()
    sim["date"] = pd.to_datetime(sim["date"])
    sim["period"] = pd.NA
    sim.loc[sim["date"].between(pre_start, pre_end), "period"] = "pre"
    sim.loc[sim["date"].between(post_start, post_end), "period"] = "post"
    sim = sim[sim["period"].isin(["pre", "post"])].copy()

    fold_rows = []
    for (gcin, period, fold), group in sim.groupby(["GCIN", "period", "fold"], observed=True):
        obs = group["observed_Q_mmd"].to_numpy(float)
        pred = group["simulated_Q_mmd"].to_numpy(float)
        fold_rows.append(
            {
                "GCIN": int(gcin),
                "period": str(period),
                "fold": int(fold),
                "n_recession_days": int(np.isfinite(obs).sum()),
                "nse": nse(obs, pred),
                "kge": kge(obs, pred),
            }
        )
    fold_skill = pd.DataFrame(fold_rows)

    rows = []
    for gcin, group in sim.groupby("GCIN", observed=True, sort=True):
        row: dict[str, object] = {"GCIN": int(gcin)}
        for period in ("pre", "post"):
            part = group[group["period"] == period]
            obs = part["observed_Q_mmd"].to_numpy(float)
            pred = part["simulated_Q_mmd"].to_numpy(float)
            epsilon = part["epsilon_effective"].to_numpy(float)
            period_folds = fold_skill[(fold_skill["GCIN"] == int(gcin)) & (fold_skill["period"] == period)]
            row[f"{period}_epsilon_mean"] = float(np.nanmean(epsilon)) if np.isfinite(epsilon).any() else np.nan
            row[f"{period}_qobs_valid_days"] = int(np.isfinite(obs).sum())
            row[f"{period}_n_recession_days_predicted"] = int(np.isfinite(epsilon).sum())
            row[f"{period}_nse"] = nse(obs, pred)
            row[f"{period}_kge"] = kge(obs, pred)
            row[f"{period}_fold_nse_mean"] = float(period_folds["nse"].mean())
            row[f"{period}_fold_nse_std"] = float(period_folds["nse"].std(ddof=1))
            row[f"{period}_fold_kge_mean"] = float(period_folds["kge"].mean())
            row[f"{period}_folds_scored"] = int(period_folds["nse"].notna().sum())
        row["delta_epsilon_mean"] = row["post_epsilon_mean"] - row["pre_epsilon_mean"]
        rows.append(row)
    return pd.DataFrame(rows), fold_skill


def change_inference(summary: pd.DataFrame, seed: int, n_bootstrap: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for threshold in (None, 0.0, 0.3, 0.5, 0.6):
        subset = summary.dropna(subset=["delta_epsilon_mean"]).copy()
        label = "all"
        if threshold is not None:
            subset = subset[(subset["pre_nse"] > threshold) & (subset["post_nse"] > threshold)]
            label = f"both_period_nse_gt_{threshold:.1f}"
        values = subset["delta_epsilon_mean"].to_numpy(float)
        if len(values) == 0:
            rows.append(
                {
                    "subset": label,
                    "n_catchments": 0,
                    "mean_delta_epsilon": np.nan,
                    "mean_bootstrap_ci025": np.nan,
                    "mean_bootstrap_ci975": np.nan,
                    "median_delta_epsilon": np.nan,
                    "median_bootstrap_ci025": np.nan,
                    "median_bootstrap_ci975": np.nan,
                    "negative_share": np.nan,
                    "positive_share": np.nan,
                    "two_sided_sign_test_p": np.nan,
                    "wilcoxon_signed_rank_p": np.nan,
                }
            )
            continue
        boot_mean = np.empty(n_bootstrap, dtype="float64")
        boot_median = np.empty(n_bootstrap, dtype="float64")
        for index in range(n_bootstrap):
            sample = rng.choice(values, size=len(values), replace=True)
            boot_mean[index] = sample.mean()
            boot_median[index] = np.median(sample)
        nonzero = values[values != 0]
        positive = int((nonzero > 0).sum())
        signed_rank_p = float(wilcoxon(values, zero_method="wilcox").pvalue) if np.any(values != 0) else np.nan
        rows.append(
            {
                "subset": label,
                "n_catchments": len(values),
                "mean_delta_epsilon": float(values.mean()),
                "mean_bootstrap_ci025": float(np.quantile(boot_mean, 0.025)),
                "mean_bootstrap_ci975": float(np.quantile(boot_mean, 0.975)),
                "median_delta_epsilon": float(np.median(values)),
                "median_bootstrap_ci025": float(np.quantile(boot_median, 0.025)),
                "median_bootstrap_ci975": float(np.quantile(boot_median, 0.975)),
                "negative_share": float((values < 0).mean()),
                "positive_share": float((values > 0).mean()),
                "two_sided_sign_test_p": float(binomtest(positive, len(nonzero), 0.5).pvalue) if len(nonzero) else np.nan,
                "wilcoxon_signed_rank_p": signed_rank_p,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_root = output_dir(cfg) / args.run_label
    simulations = []
    metrics = []
    for fold in range(int(cfg["splits"]["n_folds"])):
        fold_dir = run_root / f"fold_{fold}"
        sim = pd.read_parquet(fold_dir / "recession_day_simulations.parquet")
        sim["fold"] = fold
        simulations.append(sim)
        metric = pd.read_csv(fold_dir / "metrics.csv")
        metric["fold"] = fold
        metrics.append(metric)

    all_sim = pd.concat(simulations, ignore_index=True)
    all_sim["date"] = pd.to_datetime(all_sim["date"])
    if all_sim.duplicated(["GCIN", "date"]).any():
        raise RuntimeError("A catchment date appears in more than one temporal test fold")
    fold_for_year = expected_fold_for_year(cfg)
    expected = all_sim["date"].dt.year.map(fold_for_year).astype(int)
    if not bool((expected.to_numpy() == all_sim["fold"].to_numpy()).all()):
        raise RuntimeError("At least one OOF simulation belongs to the wrong temporal fold")

    summary, fold_skill = summarize_oof(all_sim, cfg)
    inference = change_inference(summary, seed=int(cfg["seed"]))
    out_dir = run_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_sim.to_parquet(out_dir / "oof_recession_day_simulations.parquet", index=False)
    summary.to_parquet(out_dir / "temporal_crossfit_epsilon_change_summary.parquet", index=False)
    summary.to_csv(out_dir / "temporal_crossfit_epsilon_change_summary.csv", index=False)
    fold_skill.to_csv(out_dir / "temporal_fold_catchment_skill.csv", index=False)
    inference.to_csv(out_dir / "epsilon_change_inference.csv", index=False)
    pd.concat(metrics, ignore_index=True).to_csv(out_dir / "temporal_crossfit_training_metrics.csv", index=False)

    print(f"OOF recession days: {len(all_sim):,}")
    print(f"catchments with contrasts: {summary['delta_epsilon_mean'].notna().sum():,}")
    print(
        summary[["pre_nse", "post_nse", "delta_epsilon_mean"]]
        .agg(["count", "mean", "median"])
        .to_string()
    )


if __name__ == "__main__":
    main()
