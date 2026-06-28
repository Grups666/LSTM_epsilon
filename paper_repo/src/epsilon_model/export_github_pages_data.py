from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config

REGIMES = ("all", "low", "high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("paper_repo/configs/epsilon_experiment_era5land_legacy_1950_2019.yaml"))
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--run-label", type=str, default="full_crossfit_era5land_legacy_1950_2019")
    parser.add_argument("--static", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=48)
    return parser.parse_args()


def finite_or_none(value: object, digits: int = 8) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return round(number, digits)


def read_simulations(run_root: Path) -> pd.DataFrame:
    frames = []
    for fold_dir in sorted(run_root.glob("fold_*")):
        path = fold_dir / "recession_day_simulations.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"No recession_day_simulations.parquet files found under {run_root}")
    sim = pd.concat(frames, ignore_index=True)
    sim["GCIN"] = pd.to_numeric(sim["GCIN"], errors="coerce").astype("Int64")
    sim["date"] = pd.to_datetime(sim["date"])
    sim = sim.dropna(subset=["GCIN", "date", "observed_Q_mmd", "epsilon_effective"]).copy()
    return sim


def label_periods(sim: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    pre_start, pre_end = pd.to_datetime(cfg["data"]["pre_window"])
    post_start, post_end = pd.to_datetime(cfg["data"]["post_window"])
    sim = sim.copy()
    sim["period"] = pd.NA
    sim.loc[(sim["date"] >= pre_start) & (sim["date"] <= pre_end), "period"] = "pre"
    sim.loc[(sim["date"] >= post_start) & (sim["date"] <= post_end), "period"] = "post"
    return sim[sim["period"].isin(["pre", "post"])].copy()


def label_regimes(sim: pd.DataFrame) -> pd.DataFrame:
    qtiles = sim.groupby("GCIN", observed=True)["observed_Q_mmd"].quantile([0.1, 0.9]).unstack()
    qtiles.columns = ["q10", "q90"]
    sim = sim.merge(qtiles, on="GCIN", how="left")
    sim["regime"] = "mid"
    sim.loc[sim["observed_Q_mmd"] <= sim["q10"], "regime"] = "low"
    sim.loc[sim["observed_Q_mmd"] >= sim["q90"], "regime"] = "high"
    return sim


def stats(values: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {"mean": None, "q25": None, "q50": None, "q75": None, "std": None, "n": 0}
    return {
        "mean": finite_or_none(values.mean()),
        "q25": finite_or_none(values.quantile(0.25)),
        "q50": finite_or_none(values.quantile(0.50)),
        "q75": finite_or_none(values.quantile(0.75)),
        "std": finite_or_none(values.std(ddof=1)),
        "n": int(values.size),
    }


def density_curve(pre: np.ndarray, post: np.ndarray, bins: int) -> dict[str, list[float | None]]:
    pre = pre[np.isfinite(pre)]
    post = post[np.isfinite(post)]
    both = np.concatenate([pre, post])
    if both.size < 4 or pre.size == 0 or post.size == 0:
        return {"x": [], "preDensity": [], "postDensity": [], "preCdf": [], "postCdf": []}
    lo = float(np.nanquantile(both, 0.005))
    hi = float(np.nanquantile(both, 0.995))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(both))
        hi = float(np.nanmax(both))
    if hi <= lo:
        hi = lo + max(abs(lo), 1.0) * 1e-6
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    pre_counts, _ = np.histogram(pre, bins=edges)
    post_counts, _ = np.histogram(post, bins=edges)
    width = np.diff(edges)
    pre_density = pre_counts / max(pre_counts.sum(), 1) / width
    post_density = post_counts / max(post_counts.sum(), 1) / width
    pre_cdf = np.cumsum(pre_counts) / max(pre_counts.sum(), 1)
    post_cdf = np.cumsum(post_counts) / max(post_counts.sum(), 1)
    return {
        "x": [finite_or_none(v) for v in centers],
        "preDensity": [finite_or_none(v) for v in pre_density],
        "postDensity": [finite_or_none(v) for v in post_density],
        "preCdf": [finite_or_none(v) for v in pre_cdf],
        "postCdf": [finite_or_none(v) for v in post_cdf],
    }


def build_payload(sim: pd.DataFrame, static_path: Path, bins: int, cfg: dict, run_label: str) -> dict[str, object]:
    static = pd.read_parquet(static_path, columns=["GCIN", "longitude", "latitude", "area_km2", "Prec_mm", "Temp_C", "Aridity"])
    static["GCIN"] = pd.to_numeric(static["GCIN"], errors="coerce").astype("Int64")
    static = static.set_index("GCIN")

    basins = []
    curves: dict[str, dict[str, object]] = {}
    grouped = sim.groupby("GCIN", observed=True, sort=True)
    for gcin, g in grouped:
        if gcin not in static.index:
            continue
        row = static.loc[gcin]
        basin: dict[str, object] = {
            "GCIN": int(gcin),
            "lon": finite_or_none(row["longitude"]),
            "lat": finite_or_none(row["latitude"]),
            "area_km2": finite_or_none(row["area_km2"]),
            "Prec_mm": finite_or_none(row["Prec_mm"]),
            "Temp_C": finite_or_none(row["Temp_C"]),
            "Aridity": finite_or_none(row["Aridity"]),
        }
        if basin["lon"] is None or basin["lat"] is None:
            continue

        q10 = finite_or_none(g["q10"].iloc[0])
        q90 = finite_or_none(g["q90"].iloc[0])
        basin_curves = {}
        all_delta_is_valid = False

        for regime in REGIMES:
            rg = g if regime == "all" else g[g["regime"] == regime]
            pre = rg.loc[rg["period"] == "pre", "epsilon_effective"]
            post = rg.loc[rg["period"] == "post", "epsilon_effective"]
            pre_stats = stats(pre)
            post_stats = stats(post)
            pre_mean = pre_stats["mean"]
            post_mean = post_stats["mean"]
            delta = None if pre_mean is None or post_mean is None else finite_or_none(post_mean - pre_mean)
            relative = None
            if delta is not None and pre_mean not in (None, 0):
                relative = finite_or_none(100.0 * delta / float(pre_mean))

            prefix = regime
            basin[f"{prefix}_pre_mean"] = pre_mean
            basin[f"{prefix}_post_mean"] = post_mean
            basin[f"{prefix}_delta_mean"] = delta
            basin[f"{prefix}_relative_delta_pct"] = relative
            basin[f"{prefix}_pre_n"] = pre_stats["n"]
            basin[f"{prefix}_post_n"] = post_stats["n"]
            basin[f"{prefix}_pre_q25"] = pre_stats["q25"]
            basin[f"{prefix}_pre_q50"] = pre_stats["q50"]
            basin[f"{prefix}_pre_q75"] = pre_stats["q75"]
            basin[f"{prefix}_post_q25"] = post_stats["q25"]
            basin[f"{prefix}_post_q50"] = post_stats["q50"]
            basin[f"{prefix}_post_q75"] = post_stats["q75"]
            basin[f"{prefix}_pre_std"] = pre_stats["std"]
            basin[f"{prefix}_post_std"] = post_stats["std"]
            basin[f"{prefix}_qobs_p10"] = q10
            basin[f"{prefix}_qobs_p90"] = q90
            basin_curves[regime] = density_curve(pre.to_numpy(float), post.to_numpy(float), bins)
            if regime == "all" and delta is not None:
                all_delta_is_valid = True

        if all_delta_is_valid:
            basins.append(basin)
            curves[str(int(gcin))] = basin_curves

    return {
        "meta": {
            "title": "Catchment epsilon distribution explorer",
            "generatedFrom": run_label,
            "periods": {
                "pre": f"{cfg['data']['pre_window'][0]} to {cfg['data']['pre_window'][1]}",
                "post": f"{cfg['data']['post_window'][0]} to {cfg['data']['post_window'][1]}",
            },
            "regimes": {
                "all": "all recession days",
                "low": "Q_obs <= catchment Q10",
                "high": "Q_obs >= catchment Q90",
            },
            "nCatchments": len(basins),
            "bins": bins,
            "module": "epsilon-change",
        },
        "basins": basins,
        "curves": curves,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_root = args.run_root or (Path(cfg["paths"]["output_dir"]) / args.run_label)
    static = args.static or Path(cfg["paths"]["static_attributes"])
    sim = read_simulations(run_root)
    sim = label_regimes(label_periods(sim, cfg))
    payload = build_payload(sim, static, args.bins, cfg, args.run_label)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"wrote {args.out} with {payload['meta']['nCatchments']:,} catchments")


if __name__ == "__main__":
    main()
