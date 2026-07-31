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
    parser.add_argument("--config", type=Path, default=Path("paper_repo/configs/epsilon_experiment_pure_gcin_1950_2019.yaml"))
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--run-label", type=str, default="temporal_crossfit_1990")
    parser.add_argument("--static", type=Path)
    parser.add_argument("--attribution", type=Path)
    parser.add_argument("--trends", type=Path)
    parser.add_argument(
        "--qobs-coverage",
        type=Path,
        default=Path("_private/results/epsilon_pure_gcin_1950_2019/inputs/qobs_inventory.parquet"),
    )
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


def nse(obs: np.ndarray, pred: np.ndarray) -> float | None:
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() < 2:
        return None
    obs = obs[valid]
    pred = pred[valid]
    denominator = np.sum((obs - obs.mean()) ** 2)
    if denominator <= 0:
        return None
    return finite_or_none(1.0 - np.sum((obs - pred) ** 2) / denominator)


def kge(obs: np.ndarray, pred: np.ndarray) -> float | None:
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() < 2:
        return None
    obs = obs[valid]
    pred = pred[valid]
    obs_std = obs.std(ddof=1)
    pred_std = pred.std(ddof=1)
    obs_mean = obs.mean()
    pred_mean = pred.mean()
    if obs_std <= 0 or pred_std <= 0 or obs_mean == 0:
        return None
    correlation = ((obs - obs_mean) * (pred - pred_mean)).sum() / ((len(obs) - 1) * obs_std * pred_std)
    alpha = pred_std / obs_std
    beta = pred_mean / obs_mean
    return finite_or_none(1.0 - np.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


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


def read_qobs_coverage(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        coverage = pd.read_parquet(path)
    else:
        coverage = pd.read_csv(path)
    if "GCIN" not in coverage.columns:
        return pd.DataFrame()
    rename = {
        "start": "first_valid_date",
        "end": "last_valid_date",
        "valid_q_days": "valid_days",
    }
    coverage = coverage.rename(columns={k: v for k, v in rename.items() if k in coverage.columns})
    coverage["GCIN"] = pd.to_numeric(coverage["GCIN"], errors="coerce").astype("Int64")
    return coverage.set_index("GCIN")


def parse_gap_ranges(value: object, limit: int = 3) -> list[dict[str, object]]:
    if value is None or pd.isna(value):
        return []
    try:
        gaps = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(gaps, list):
        return []
    clean = []
    for gap in gaps:
        if not isinstance(gap, dict):
            continue
        start = gap.get("start")
        end = gap.get("end")
        days = gap.get("days")
        if not start or not end or not isinstance(days, (int, float)):
            continue
        clean.append({"start": str(start), "end": str(end), "days": int(days)})
    return clean[:limit]


def read_attribution(path: Path) -> dict[tuple[int, str], dict[str, object]]:
    if not path.exists():
        return {}
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    required = {"GCIN", "regime", "driver"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Attribution file is missing columns: {sorted(required - set(frame.columns))}")
    fields = (
        "driver",
        "gq_change_pct",
        "qsim_change_pct",
        "epsilon_change_pct",
        "gq_component_log",
        "q_component_log",
        "gq_absolute_share",
        "offsetting",
        "closure_error_log",
        "pre_gq_geomean",
        "post_gq_geomean",
        "pre_qsim_geomean",
        "post_qsim_geomean",
    )
    records: dict[tuple[int, str], dict[str, object]] = {}
    for row in frame.itertuples(index=False):
        record = row._asdict()
        records[(int(record["GCIN"]), str(record["regime"]))] = {
            field: record.get(field) for field in fields if field in record
        }
    return records


def read_trends(path: Path) -> dict[tuple[int, str], dict[str, object]]:
    if not path.exists():
        return {}
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    required = {"GCIN", "regime", "epsilon_trend_class", "trend_driver"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Trend file is missing columns: {sorted(required - set(frame.columns))}")
    records: dict[tuple[int, str], dict[str, object]] = {}
    for row in frame.itertuples(index=False):
        record = row._asdict()
        records[(int(record["GCIN"]), str(record["regime"]))] = {
            field: value
            for field, value in record.items()
            if field not in {"GCIN", "regime"}
        }
    return records


def build_payload(
    sim: pd.DataFrame,
    static_path: Path,
    qobs_coverage_path: Path,
    attribution: dict[tuple[int, str], dict[str, object]],
    trends: dict[tuple[int, str], dict[str, object]],
    bins: int,
    cfg: dict,
    run_label: str,
) -> dict[str, object]:
    static_columns = [
        "GCIN",
        "catchment_id",
        "source",
        "source_id",
        "force_code",
        "longitude",
        "latitude",
        "area_km2",
        "Prec_mm",
        "Temp_C",
        "Aridity",
    ]
    available = set(pd.read_parquet(static_path).columns)
    static = pd.read_parquet(static_path, columns=[c for c in static_columns if c in available])
    static["GCIN"] = pd.to_numeric(static["GCIN"], errors="coerce").astype("Int64")
    static = static.set_index("GCIN")
    qobs_coverage = read_qobs_coverage(qobs_coverage_path)

    basins = []
    curves: dict[str, dict[str, object]] = {}
    grouped = sim.groupby("GCIN", observed=True, sort=True)
    for gcin, g in grouped:
        if gcin not in static.index:
            continue
        row = static.loc[gcin]
        basin: dict[str, object] = {
            "GCIN": int(gcin),
            "catchment_id": None if "catchment_id" not in static.columns or pd.isna(row["catchment_id"]) else str(row["catchment_id"]),
            "source": None if "source" not in static.columns or pd.isna(row["source"]) else str(row["source"]),
            "source_id": None if "source_id" not in static.columns or pd.isna(row["source_id"]) else int(row["source_id"]),
            "force_code": int(row["force_code"]) if "force_code" in static.columns and pd.notna(row["force_code"]) else None,
            "lon": finite_or_none(row["longitude"]),
            "lat": finite_or_none(row["latitude"]),
            "area_km2": finite_or_none(row["area_km2"]),
            "Prec_mm": finite_or_none(row["Prec_mm"]),
            "Temp_C": finite_or_none(row["Temp_C"]),
            "Aridity": finite_or_none(row["Aridity"]),
        }
        if not qobs_coverage.empty and gcin in qobs_coverage.index:
            cov = qobs_coverage.loc[gcin]
            basin["qobs_start"] = None if pd.isna(cov.get("first_valid_date")) else str(cov.get("first_valid_date"))
            basin["qobs_end"] = None if pd.isna(cov.get("last_valid_date")) else str(cov.get("last_valid_date"))
            basin["qobs_valid_days"] = int(cov.get("valid_days")) if pd.notna(cov.get("valid_days")) else None
            basin["qobs_sources"] = None if pd.isna(cov.get("sources")) else str(cov.get("sources"))
            basin["qobs_calendar_days"] = int(cov.get("calendar_days")) if pd.notna(cov.get("calendar_days")) else None
            basin["qobs_missing_days"] = int(cov.get("missing_q_days")) if pd.notna(cov.get("missing_q_days")) else None
            basin["qobs_coverage_pct"] = finite_or_none(cov.get("qobs_coverage_pct"), digits=4)
            basin["qobs_missing_pct"] = finite_or_none(cov.get("qobs_missing_pct"), digits=4)
            basin["qobs_gap_runs"] = int(cov.get("qobs_gap_runs")) if pd.notna(cov.get("qobs_gap_runs")) else None
            basin["qobs_long_gap_count"] = int(cov.get("qobs_long_gap_count")) if pd.notna(cov.get("qobs_long_gap_count")) else None
            basin["qobs_long_gaps"] = parse_gap_ranges(cov.get("qobs_long_gaps_json"))
        if basin["lon"] is None or basin["lat"] is None:
            continue

        q10 = finite_or_none(g["q10"].iloc[0])
        q90 = finite_or_none(g["q90"].iloc[0])
        basin_curves = {}
        all_delta_is_valid = False

        for period in ("pre", "post"):
            period_rows = g[g["period"] == period]
            observed = period_rows["observed_Q_mmd"].to_numpy(float)
            simulated = period_rows["simulated_Q_mmd"].to_numpy(float)
            basin[f"{period}_nse"] = nse(observed, simulated)
            basin[f"{period}_kge"] = kge(observed, simulated)

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
            attribution_row = attribution.get((int(gcin), regime), {})
            for field, value in attribution_row.items():
                key = f"{prefix}_{field}"
                if field == "driver":
                    basin[key] = str(value)
                elif field in {"offsetting"}:
                    basin[key] = bool(value)
                else:
                    basin[key] = finite_or_none(value)
            trend_row = trends.get((int(gcin), regime), {})
            for field, value in trend_row.items():
                key = f"{prefix}_{field}"
                if field in {"epsilon_trend_class", "gq_trend_class", "qsim_trend_class", "trend_driver"}:
                    basin[key] = str(value)
                elif field in {"epsilon_significant", "gq_significant", "qsim_significant"}:
                    basin[key] = bool(value)
                elif field.endswith("_n_years") or field.endswith("_start_year") or field.endswith("_end_year"):
                    basin[key] = int(value) if pd.notna(value) else None
                else:
                    basin[key] = finite_or_none(value)
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
            "evaluation": {
                "strategy": cfg["splits"]["strategy"],
                "folds": int(cfg["splits"]["n_folds"]),
                "trainBlocksPerPeriod": int(cfg["splits"]["n_folds"]) - 1,
                "testBlocksPerPeriod": 1,
                "validationSet": False,
                "checkpointSelection": cfg["training"]["checkpoint_selection"],
            },
            "attribution": {
                "definition": "GQ_effective = epsilon_effective x simulated_Q",
                "decomposition": "delta log epsilon = delta log GQ - delta log Q",
                "classification": "pre/post component dominance; not a causal climate attribution",
            },
            "continuousTrend": {
                "annualStatistic": "median of at least 5 recession days",
                "minimumYears": 20,
                "slope": "fold-centered log-space Theil-Sen slope",
                "significance": "trend-free prewhitened Kendall test with Benjamini-Hochberg FDR q < 0.05",
            },
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
    attribution_path = args.attribution or (run_root / "analysis" / "gq_q_attribution_by_catchment.parquet")
    trends_path = args.trends or (run_root / "analysis" / "continuous_trends_by_catchment.parquet")
    sim = read_simulations(run_root)
    sim = label_regimes(label_periods(sim, cfg))
    attribution = read_attribution(attribution_path)
    trends = read_trends(trends_path)
    payload = build_payload(sim, static, args.qobs_coverage, attribution, trends, args.bins, cfg, args.run_label)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"wrote {args.out} with {payload['meta']['nCatchments']:,} catchments")


if __name__ == "__main__":
    main()
