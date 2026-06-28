from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from config import load_config, output_dir

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:  # pragma: no cover - optional plotting dependency
    ccrs = None
    cfeature = None


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#202433",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "blue": "#3D6DCC",
    "red": "#B65045",
    "gold": "#B99B33",
    "green": "#5C8F43",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-label", type=str, default="full_crossfit_era5land_legacy_1950_2019")
    parser.add_argument("--out-dir", type=Path, default=Path("_private/results/paper_figures"))
    return parser.parse_args()


def setup_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "axes.edgecolor": "#D7DBE7",
            "axes.labelcolor": TOKENS["ink"],
            "axes.titlecolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "grid.color": TOKENS["grid"],
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "savefig.facecolor": TOKENS["surface"],
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.06, 0.965, title, ha="left", va="top", fontsize=15, fontweight="bold", color=TOKENS["ink"])
    fig.text(0.06, 0.925, subtitle, ha="left", va="top", fontsize=9.5, color=TOKENS["muted"])


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required result is missing: {path}")
    return path


def load_run(cfg: dict, run_label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_root = output_dir(cfg) / run_label
    summaries = []
    metrics = []
    sims = []
    for fold in range(int(cfg["splits"]["n_folds"])):
        fold_dir = run_root / f"fold_{fold}"
        summary = pd.read_parquet(require(fold_dir / "heldout_epsilon_change_summary.parquet"))
        summary["fold"] = fold
        summaries.append(summary)
        metric = pd.read_csv(require(fold_dir / "metrics.csv"))
        metric["fold"] = fold
        metrics.append(metric)
        sim_path = fold_dir / "recession_day_simulations.parquet"
        if sim_path.exists():
            sim = pd.read_parquet(sim_path)
            sim["fold"] = fold
            sims.append(sim)
    return pd.concat(summaries, ignore_index=True), pd.concat(metrics, ignore_index=True), pd.concat(sims, ignore_index=True)


def compute_flow_regime_stats(sim: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    pre_start, pre_end = pd.to_datetime(cfg["data"]["pre_window"])
    post_start, post_end = pd.to_datetime(cfg["data"]["post_window"])
    sim = sim.copy()
    sim["date"] = pd.to_datetime(sim["date"])
    qtiles = sim.groupby("GCIN", observed=True)["observed_Q_mmd"].quantile([0.1, 0.9]).unstack()
    qtiles.columns = ["q10", "q90"]
    sim = sim.merge(qtiles, on="GCIN", how="left")
    sim["regime"] = "mid"
    sim.loc[sim["observed_Q_mmd"] <= sim["q10"], "regime"] = "low"
    sim.loc[sim["observed_Q_mmd"] >= sim["q90"], "regime"] = "high"
    sim["period"] = pd.NA
    sim.loc[(sim["date"] >= pre_start) & (sim["date"] <= pre_end), "period"] = "pre"
    sim.loc[(sim["date"] >= post_start) & (sim["date"] <= post_end), "period"] = "post"
    sim = sim[sim["period"].isin(["pre", "post"])].copy()

    grouped = (
        sim.groupby(["GCIN", "regime", "period"], observed=True)["epsilon_effective"]
        .agg(["mean", "median", "std", "count", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
        .reset_index()
        .rename(columns={"<lambda_0>": "q25", "<lambda_1>": "q75"})
    )
    wide = grouped.pivot(index=["GCIN", "regime"], columns="period", values=["mean", "median", "std", "q25", "q75", "count"])
    wide.columns = [f"{stat}_{period}" for stat, period in wide.columns]
    wide = wide.reset_index()
    wide["delta_mean"] = wide["mean_post"] - wide["mean_pre"]
    wide["relative_delta_mean"] = wide["delta_mean"] / wide["mean_pre"].replace(0, np.nan)
    return wide, sim


def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() < 2:
        return np.nan
    obs = obs[valid]
    pred = pred[valid]
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom <= 0:
        return np.nan
    return float(1.0 - np.sum((obs - pred) ** 2) / denom)


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
    r = float(((obs - obs_mean) * (pred - pred_mean)).sum() / ((len(obs) - 1) * obs_std * pred_std))
    alpha = pred_std / obs_std
    beta = pred_mean / obs_mean
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def compute_model_skill(sim: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    pre_start, pre_end = pd.to_datetime(cfg["data"]["pre_window"])
    post_start, post_end = pd.to_datetime(cfg["data"]["post_window"])
    work = sim.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["period"] = "other"
    work.loc[(work["date"] >= pre_start) & (work["date"] <= pre_end), "period"] = "pre"
    work.loc[(work["date"] >= post_start) & (work["date"] <= post_end), "period"] = "post"
    work = work[work["period"].isin(["pre", "post"])].copy()

    rows = []
    for (gcin, period), g in work.groupby(["GCIN", "period"], observed=True):
        obs = g["observed_Q_mmd"].to_numpy(dtype=float)
        pred = g["simulated_Q_mmd"].to_numpy(dtype=float)
        rows.append(
            {
                "GCIN": int(gcin),
                "period": period,
                "n_days": int(np.isfinite(obs).sum()),
                "nse": nse(obs, pred),
                "kge": kge(obs, pred),
                "bias_ratio": float(np.nanmean(pred) / np.nanmean(obs)) if np.nanmean(obs) != 0 else np.nan,
            }
        )
    by_catchment = pd.DataFrame(rows)

    summary_rows = []
    for period, g in work.groupby("period", observed=True):
        obs = g["observed_Q_mmd"].to_numpy(dtype=float)
        pred = g["simulated_Q_mmd"].to_numpy(dtype=float)
        catch = by_catchment[by_catchment["period"] == period]
        summary_rows.append(
            {
                "period": period,
                "n_days": int(np.isfinite(obs).sum()),
                "n_catchments": int(catch["GCIN"].nunique()),
                "pooled_nse": nse(obs, pred),
                "pooled_kge": kge(obs, pred),
                "median_catchment_nse": float(catch["nse"].median()),
                "median_catchment_kge": float(catch["kge"].median()),
                "p10_catchment_nse": float(catch["nse"].quantile(0.10)),
                "p90_catchment_nse": float(catch["nse"].quantile(0.90)),
                "p10_catchment_kge": float(catch["kge"].quantile(0.10)),
                "p90_catchment_kge": float(catch["kge"].quantile(0.90)),
                "mean_catchment_nse": float(catch["nse"].mean()),
                "mean_catchment_kge": float(catch["kge"].mean()),
            }
        )
    all_obs = work["observed_Q_mmd"].to_numpy(dtype=float)
    all_pred = work["simulated_Q_mmd"].to_numpy(dtype=float)
    summary_rows.append(
        {
            "period": "all",
            "n_days": int(np.isfinite(all_obs).sum()),
            "n_catchments": int(work["GCIN"].nunique()),
            "pooled_nse": nse(all_obs, all_pred),
            "pooled_kge": kge(all_obs, all_pred),
            "median_catchment_nse": float(by_catchment["nse"].median()),
            "median_catchment_kge": float(by_catchment["kge"].median()),
            "p10_catchment_nse": float(by_catchment["nse"].quantile(0.10)),
            "p90_catchment_nse": float(by_catchment["nse"].quantile(0.90)),
            "p10_catchment_kge": float(by_catchment["kge"].quantile(0.10)),
            "p90_catchment_kge": float(by_catchment["kge"].quantile(0.90)),
            "mean_catchment_nse": float(by_catchment["nse"].mean()),
            "mean_catchment_kge": float(by_catchment["kge"].mean()),
        }
    )
    return by_catchment, pd.DataFrame(summary_rows)


def figure_training(metrics: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    fig.subplots_adjust(top=0.78)
    header(fig, "Physics-informed training convergence", "Total loss by cluster/fold for the Ara-style epsilon-core model.")
    sns.lineplot(data=metrics, x="epoch", y="total", hue="fold", palette="tab10", linewidth=1.2, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss, log scale")
    ax.legend(title="Cluster", frameon=False, ncols=5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    save(fig, out_dir / "figure_01_training_loss")


def figure_delta_distribution(summary: pd.DataFrame, regime: pd.DataFrame, out_dir: Path) -> None:
    valid = summary.dropna(subset=["delta_epsilon_mean"]).copy()
    box = regime[regime["regime"].isin(["low", "mid", "high"])].dropna(subset=["delta_mean"]).copy()
    order = ["low", "mid", "high"]
    labels = ["Low flow\nQ <= Q10", "Mid flow\nQ10 < Q < Q90", "High flow\nQ >= Q90"]
    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    fig.subplots_adjust(top=0.80, left=0.12, right=0.96, bottom=0.13)
    header(
        fig,
        "Flow-regime epsilon change is evaluated as post-change minus pre-change",
        f"Each catchment uses its own Q10 and Q90 thresholds; n={len(valid):,} catchments with all-recession summaries.",
    )
    palette = {"low": "#87A9D9", "mid": "#D8D3B0", "high": "#D68A7A"}
    sns.violinplot(
        data=box,
        x="regime",
        y="delta_mean",
        hue="regime",
        order=order,
        hue_order=order,
        palette=palette,
        inner="quartile",
        linewidth=0.9,
        cut=0,
        density_norm="width",
        legend=False,
        ax=ax,
    )
    sample = box.sample(min(len(box), 3000), random_state=42)
    sns.stripplot(
        data=sample,
        x="regime",
        y="delta_mean",
        order=order,
        color=TOKENS["ink"],
        alpha=0.10,
        size=1.2,
        jitter=0.22,
        ax=ax,
    )
    medians = box.groupby("regime", observed=True)["delta_mean"].median().reindex(order)
    means = box.groupby("regime", observed=True)["delta_mean"].mean().reindex(order)
    for idx, regime_name in enumerate(order):
        ax.scatter(idx, medians.loc[regime_name], s=42, color=TOKENS["ink"], zorder=5, label="median" if idx == 0 else None)
        ax.scatter(idx, means.loc[regime_name], s=50, color=TOKENS["red"], marker="D", zorder=5, label="mean" if idx == 0 else None)
    ax.axhline(0, color=TOKENS["ink"], lw=1.0)
    ax.set_yscale("symlog", linthresh=1e-3, linscale=0.9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("")
    ax.set_ylabel("Delta epsilon, symmetric log scale")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: "0" if value == 0 else f"{value:.0e}"))
    ax.legend(frameon=False, loc="upper right")
    save(fig, out_dir / "figure_02_delta_distribution")


def figure_hydroclimate(summary: pd.DataFrame, static_path: Path, out_dir: Path) -> None:
    static = pd.read_parquet(static_path, columns=["GCIN", "Prec_mm", "Temp_C", "Aridity", "area_km2", "longitude", "latitude"])
    static["GCIN"] = pd.to_numeric(static["GCIN"], errors="coerce")
    df = summary.merge(static, on="GCIN", how="left").dropna(subset=["delta_epsilon_mean"]).copy()
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.4), sharey=True)
    fig.subplots_adjust(top=0.76, wspace=0.16)
    header(fig, "Hydroclimate gradients of epsilon change", "Catchments are binned into quartiles for each static hydroclimate attribute.")
    for ax, col, label in zip(axes, ["Prec_mm", "Temp_C", "Aridity"], ["Precipitation", "Temperature", "Aridity"]):
        df["bin"] = pd.qcut(df[col], 4, duplicates="drop")
        g = df.groupby("bin", observed=True)["delta_epsilon_mean"].agg(["mean", "median", "count"]).reset_index()
        g["x"] = np.arange(len(g))
        ax.plot(g["x"], g["mean"], marker="o", color=TOKENS["blue"], label="mean")
        ax.plot(g["x"], g["median"], marker="s", color=TOKENS["red"], label="median")
        ax.axhline(0, color=TOKENS["ink"], lw=0.9)
        ax.set_title(label)
        ax.set_xlabel("Quartile")
        ax.set_xticks(g["x"])
        ax.set_xticklabels([str(i + 1) for i in g["x"]])
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1e"))
    axes[0].set_ylabel("Delta epsilon")
    axes[-1].legend(frameon=False, loc="upper right")
    save(fig, out_dir / "figure_03_hydroclimate_gradients")


def figure_map(summary: pd.DataFrame, static_path: Path, out_dir: Path) -> None:
    static = pd.read_parquet(static_path, columns=["GCIN", "longitude", "latitude"])
    static["GCIN"] = pd.to_numeric(static["GCIN"], errors="coerce")
    df = summary.merge(static, on="GCIN", how="left").dropna(subset=["longitude", "latitude", "delta_epsilon_mean"]).copy()
    vmax = float(df["delta_epsilon_mean"].abs().quantile(0.98))
    if ccrs is not None:
        proj = ccrs.Robinson()
        data_proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(11.4, 5.8))
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_global()
        ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="#F4F7FA", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#ECEFF3", edgecolor="none", zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="#AEB6C4", linewidth=0.45, zorder=2)
        ax.add_feature(cfeature.BORDERS.with_scale("110m"), edgecolor="#C9CED8", linewidth=0.25, zorder=2)
        gl = ax.gridlines(
            crs=data_proj,
            linewidth=0.25,
            color="#D6DAE3",
            alpha=0.85,
            linestyle="-",
            draw_labels=False,
            zorder=1,
        )
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
        gl.ylocator = mticker.FixedLocator(np.arange(-60, 91, 30))
        scatter_kwargs = {"transform": data_proj}
    else:
        fig, ax = plt.subplots(figsize=(11.2, 5.6))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 75)
        scatter_kwargs = {}
    fig.subplots_adjust(top=0.80)
    header(fig, "Spatial distribution of epsilon change", "Representative catchment points; colors clipped at the 98th percentile.")
    sc = ax.scatter(
        df["longitude"],
        df["latitude"],
        c=df["delta_epsilon_mean"],
        cmap="PuOr",
        vmin=-vmax,
        vmax=vmax,
        s=12,
        alpha=0.86,
        linewidth=0,
        zorder=3,
        **scatter_kwargs,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Delta epsilon")
    save(fig, out_dir / "figure_04_spatial_delta")


def write_tables(
    summary: pd.DataFrame,
    regime: pd.DataFrame,
    sim: pd.DataFrame,
    skill_by_catchment: pd.DataFrame,
    skill_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "epsilon_change_by_catchment.csv", index=False)
    regime.to_csv(out_dir / "epsilon_change_by_flow_regime.csv", index=False)
    skill_by_catchment.to_csv(out_dir / "model_skill_by_catchment.csv", index=False)
    skill_summary.to_csv(out_dir / "model_skill_summary.csv", index=False)
    stats = {
        "n_catchments": len(summary),
        "n_valid_delta": int(summary["delta_epsilon_mean"].notna().sum()),
        "mean_delta_epsilon": float(summary["delta_epsilon_mean"].mean()),
        "median_delta_epsilon": float(summary["delta_epsilon_mean"].median()),
        "negative_delta_share": float((summary["delta_epsilon_mean"] < 0).mean()),
        "n_recession_simulation_days": len(sim),
        "pooled_nse_all": float(skill_summary.loc[skill_summary["period"] == "all", "pooled_nse"].iloc[0]),
        "pooled_kge_all": float(skill_summary.loc[skill_summary["period"] == "all", "pooled_kge"].iloc[0]),
        "median_catchment_nse_all": float(skill_summary.loc[skill_summary["period"] == "all", "median_catchment_nse"].iloc[0]),
        "median_catchment_kge_all": float(skill_summary.loc[skill_summary["period"] == "all", "median_catchment_kge"].iloc[0]),
        "p10_catchment_nse_all": float(skill_summary.loc[skill_summary["period"] == "all", "p10_catchment_nse"].iloc[0]),
        "p90_catchment_nse_all": float(skill_summary.loc[skill_summary["period"] == "all", "p90_catchment_nse"].iloc[0]),
        "p10_catchment_kge_all": float(skill_summary.loc[skill_summary["period"] == "all", "p10_catchment_kge"].iloc[0]),
        "p90_catchment_kge_all": float(skill_summary.loc[skill_summary["period"] == "all", "p90_catchment_kge"].iloc[0]),
    }
    pd.Series(stats).to_csv(out_dir / "result_summary.csv", header=["value"])


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    setup_style()
    print("loading run outputs", flush=True)
    summary, metrics, sim = load_run(cfg, args.run_label)
    print(f"loaded summary={summary.shape} metrics={metrics.shape} sim={sim.shape}", flush=True)
    print("computing flow-regime summaries", flush=True)
    regime, sim_labeled = compute_flow_regime_stats(sim, cfg)
    print("computing model skill", flush=True)
    skill_by_catchment, skill_summary = compute_model_skill(sim_labeled, cfg)
    print("writing tables", flush=True)
    write_tables(summary, regime, sim_labeled, skill_by_catchment, skill_summary, args.out_dir)
    print("drawing training figure", flush=True)
    figure_training(metrics, args.out_dir)
    print("drawing delta distribution", flush=True)
    figure_delta_distribution(summary, regime, args.out_dir)
    print("drawing hydroclimate figure", flush=True)
    figure_hydroclimate(summary, Path(cfg["paths"]["static_attributes"]), args.out_dir)
    print("drawing map", flush=True)
    figure_map(summary, Path(cfg["paths"]["static_attributes"]), args.out_dir)
    print(f"wrote figures and tables to {args.out_dir}")


if __name__ == "__main__":
    main()
