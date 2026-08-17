from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from analyze_gq_attribution import label_simulations
from analyze_prepost_shifts import fit_shift
from config import load_config


DEFAULT_CONFIG = Path("paper_repo/configs/global_story_analysis_v2.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "--phase",
        choices=("prepare", "discovery", "confirmation", "all"),
        default="all",
    )
    return parser.parse_args()


def stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def block_id(longitude: pd.Series, latitude: pd.Series, degrees: int) -> pd.Series:
    lon_cell = np.floor((pd.to_numeric(longitude, errors="coerce") + 180.0) / degrees)
    lat_cell = np.floor((pd.to_numeric(latitude, errors="coerce") + 90.0) / degrees)
    return lat_cell.astype("Int64").astype(str) + ":" + lon_cell.astype("Int64").astype(str)


def assign_discovery_blocks(
    frame: pd.DataFrame,
    degrees: int,
    fraction: float,
    seed: int,
) -> pd.DataFrame:
    result = frame.copy()
    result["assignment_block"] = block_id(result["longitude"], result["latitude"], degrees)
    scores = {
        block: stable_seed(seed, f"assignment:{block}") / float(2**32)
        for block in result["assignment_block"].dropna().unique()
    }
    result["analysis_split"] = result["assignment_block"].map(
        lambda block: "discovery" if scores.get(block, 1.0) < fraction else "confirmation"
    )
    return result


def annual_distribution(sim: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for regime in ("all", "low", "high"):
        data = sim if regime == "all" else sim[sim["regime"] == regime]
        grouped = data.groupby(["GCIN", "year", "fold"], observed=True)["epsilon_effective"]
        counts = grouped.size().rename("n_days")
        quantiles = grouped.quantile([0.25, 0.5, 0.75]).unstack()
        quantiles.columns = ["q25", "q50", "q75"]
        annual = counts.to_frame().join(quantiles).reset_index()
        annual["spread"] = annual["q75"] / annual["q25"]
        annual = annual.melt(
            id_vars=["GCIN", "year", "fold", "n_days"],
            value_vars=["q25", "q50", "q75", "spread"],
            var_name="statistic",
            value_name="value",
        )
        annual["regime"] = regime
        frames.append(annual)
    return pd.concat(frames, ignore_index=True)


def effect_table(
    annual: pd.DataFrame,
    break_year: int,
    min_days: int,
    min_years: int,
    min_identifying_years: int,
    hac_lag_years: int,
) -> pd.DataFrame:
    eligible = annual[annual["n_days"] >= min_days]
    rows: list[dict[str, object]] = []
    for (gcin, regime, statistic), group in eligible.groupby(
        ["GCIN", "regime", "statistic"], observed=True, sort=True
    ):
        row: dict[str, object] = {
            "GCIN": int(gcin),
            "regime": str(regime),
            "statistic": str(statistic),
        }
        row.update(
            fit_shift(
                group,
                break_year=break_year,
                min_years_per_period=min_years,
                min_paired_folds=1,
                min_identifying_years_per_period=min_identifying_years,
                hac_lag_years=hac_lag_years,
            )
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    result["break_year"] = int(break_year)
    result["min_days_per_year"] = int(min_days)
    return result


def climate_change_table(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    climate_cfg = cfg["climate"]
    variables = list(climate_cfg["variables"])
    minimum_years = int(climate_cfg["minimum_years_per_period"])
    daily_dir = Path(cfg["paths"]["physics_daily_dir"])
    annual_frames: list[pd.DataFrame] = []
    for path in sorted(daily_dir.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["GCIN", "date", *variables])
        frame["date"] = pd.to_datetime(frame["date"])
        frame["year"] = frame["date"].dt.year.astype(int)
        annual = frame.groupby(["GCIN", "year"], observed=True)[variables].mean().reset_index()
        annual_frames.append(annual)
    annual_climate = pd.concat(annual_frames, ignore_index=True)
    break_year = int(cfg["design"]["break_year"])
    annual_climate["period"] = np.where(annual_climate["year"] <= break_year, "pre", "post")
    rows: list[dict[str, object]] = []
    for gcin, group in annual_climate.groupby("GCIN", observed=True, sort=True):
        row: dict[str, object] = {"GCIN": int(gcin)}
        for variable, method in climate_cfg["variables"].items():
            pre = group.loc[group["period"] == "pre", variable].dropna()
            post = group.loc[group["period"] == "post", variable].dropna()
            row[f"{variable}_pre_years"] = int(len(pre))
            row[f"{variable}_post_years"] = int(len(post))
            pre_value = float(pre.mean()) if len(pre) else np.nan
            post_value = float(post.mean()) if len(post) else np.nan
            row[f"{variable}_pre"] = pre_value
            row[f"{variable}_post"] = post_value
            if len(pre) < minimum_years or len(post) < minimum_years:
                change = np.nan
            elif method == "difference":
                change = post_value - pre_value
            elif method == "log_ratio":
                change = (
                    float(np.log(post_value / pre_value))
                    if pre_value > 0 and post_value > 0
                    else np.nan
                )
            else:
                raise ValueError(f"Unsupported climate change method: {method}")
            row[f"climate_{variable}_{method}"] = change
        rows.append(row)
    return pd.DataFrame(rows), annual_climate


def cohort_flag(frame: pd.DataFrame, metric: str, threshold: float | None) -> pd.Series:
    if metric == "none":
        return pd.Series(True, index=frame.index)
    pre = pd.to_numeric(frame[f"pre_{metric}"], errors="coerce")
    post = pd.to_numeric(frame[f"post_{metric}"], errors="coerce")
    return (pre > float(threshold)) & (post > float(threshold))


def prepare(cfg: dict) -> dict[str, object]:
    audit_dir = Path(cfg["paths"]["audit_dir"])
    audit_dir.mkdir(parents=True, exist_ok=True)
    experiment_cfg = load_config(cfg["paths"]["experiment_config"])
    run_root = Path(cfg["paths"]["run_root"])
    sim_path = run_root / "analysis" / "oof_recession_day_simulations.parquet"
    sim = pd.read_parquet(
        sim_path,
        columns=[
            "GCIN",
            "date",
            "observed_Q_mmd",
            "simulated_Q_mmd",
            "epsilon_effective",
            "fold",
        ],
    )
    sim = label_simulations(sim, experiment_cfg)
    sim["year"] = sim["date"].dt.year.astype(int)
    annual = annual_distribution(sim)
    design = cfg["design"]
    effects = effect_table(
        annual,
        break_year=int(design["break_year"]),
        min_days=int(design["annual_min_days"]),
        min_years=int(design["annual_min_years_per_period"]),
        min_identifying_years=int(design["annual_min_identifying_years_per_period"]),
        hac_lag_years=int(design["hac_lag_years"]),
    )

    static = pd.read_parquet(cfg["paths"]["static_attributes"])
    static_columns = [
        "GCIN",
        "longitude",
        "latitude",
        "country",
        "area_km2",
        "Prec_mm",
        "Temp_C",
        "PET_mm",
        "Aridity",
        "mean_sm_rootzone",
        "elevation_mean_m",
        "mean_slope_degree",
    ]
    static = static[[column for column in static_columns if column in static.columns]].copy()
    static = assign_discovery_blocks(
        static,
        degrees=int(design["assignment_cell_degrees"]),
        fraction=float(design["discovery_fraction"]),
        seed=int(cfg["seed"]),
    )

    skill = pd.read_parquet(
        run_root / "analysis" / "temporal_crossfit_epsilon_change_summary.parquet"
    )
    climate, annual_climate = climate_change_table(cfg)
    dataset = effects.merge(static, on="GCIN", how="left", validate="many_to_one")
    dataset = dataset.merge(skill, on="GCIN", how="left", validate="many_to_one")
    dataset = dataset.merge(climate, on="GCIN", how="left", validate="many_to_one")
    reliability = cfg["reliability"]
    dataset["cohort_primary"] = cohort_flag(
        dataset,
        reliability["primary_metric"],
        reliability["primary_threshold"],
    )
    for cohort in reliability["sensitivity_cohorts"]:
        dataset[f"cohort_{cohort['name']}"] = cohort_flag(
            dataset, cohort["metric"], cohort.get("threshold")
        )

    annual.to_parquet(audit_dir / "annual_epsilon_distribution.parquet", index=False)
    annual_climate.to_parquet(audit_dir / "annual_climate_by_catchment.parquet", index=False)
    climate.to_parquet(audit_dir / "climate_change_by_catchment.parquet", index=False)
    effects.to_parquet(audit_dir / "global_story_effects_primary.parquet", index=False)
    dataset.to_parquet(audit_dir / "global_story_dataset.parquet", index=False)

    eligible = dataset[dataset["p_value"].notna()]
    quality = {
        "analysis_version": cfg["analysis_version"],
        "oof_rows": int(len(sim)),
        "oof_catchments": int(sim["GCIN"].nunique()),
        "oof_date_min": str(sim["date"].min().date()),
        "oof_date_max": str(sim["date"].max().date()),
        "duplicate_catchment_dates": int(sim.duplicated(["GCIN", "date"]).sum()),
        "nonpositive_epsilon_rows": int((sim["epsilon_effective"] <= 0).sum()),
        "missing_epsilon_rows": int(sim["epsilon_effective"].isna().sum()),
        "fold_values": sorted(int(value) for value in sim["fold"].dropna().unique()),
        "annual_rows": int(len(annual)),
        "effect_rows": int(len(effects)),
        "effect_catchments": int(effects["GCIN"].nunique()),
        "static_unmatched_effect_rows": int(dataset["longitude"].isna().sum()),
        "skill_unmatched_effect_rows": int(dataset["pre_nse"].isna().sum()),
        "split_catchments": dataset[["GCIN", "analysis_split"]]
        .drop_duplicates()["analysis_split"]
        .value_counts()
        .to_dict(),
        "split_blocks": dataset[["assignment_block", "analysis_split"]]
        .drop_duplicates()["analysis_split"]
        .value_counts()
        .to_dict(),
        "eligible_primary_by_regime_statistic": eligible[eligible["cohort_primary"]]
        .groupby(["regime", "statistic"], observed=True)
        .size()
        .to_dict(),
    }
    write_json(audit_dir / "data_quality.json", quality)
    return quality


def reml_meta(effect: np.ndarray, standard_error: np.ndarray) -> dict[str, float]:
    y = np.asarray(effect, dtype=float)
    se = np.asarray(standard_error, dtype=float)
    valid = np.isfinite(y) & np.isfinite(se) & (se > 0)
    y = y[valid]
    variance = se[valid] ** 2
    if len(y) < 3:
        return {"estimate": np.nan, "tau2": np.nan, "model_se": np.nan, "i2": np.nan}

    def objective(tau2: float) -> float:
        weights = 1.0 / (variance + tau2)
        mean = float(np.sum(weights * y) / np.sum(weights))
        residual = y - mean
        return 0.5 * (
            float(np.sum(np.log(variance + tau2)))
            + math.log(float(np.sum(weights)))
            + float(np.sum(weights * residual * residual))
        )

    upper = max(float(np.var(y, ddof=1) * 10.0), float(np.median(variance) * 10.0), 1e-8)
    fit = minimize_scalar(objective, bounds=(0.0, upper), method="bounded")
    tau2 = max(float(fit.x), 0.0)
    weights = 1.0 / (variance + tau2)
    estimate = float(np.sum(weights * y) / np.sum(weights))
    model_se = float(np.sqrt(1.0 / np.sum(weights)))
    typical_variance = float(np.median(variance))
    i2 = tau2 / (tau2 + typical_variance) if tau2 + typical_variance > 0 else 0.0
    return {"estimate": estimate, "tau2": tau2, "model_se": model_se, "i2": i2}


def spatial_bootstrap(
    frame: pd.DataFrame,
    estimator: Callable[[pd.DataFrame], float],
    degrees: int,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    data = frame.copy()
    data["bootstrap_block"] = block_id(data["longitude"], data["latitude"], degrees)
    data = data[data["bootstrap_block"].notna()].copy()
    blocks = data["bootstrap_block"].unique()
    indices = {
        block: data.index[data["bootstrap_block"] == block].to_numpy()
        for block in blocks
    }
    rng = np.random.default_rng(seed)
    estimates = np.full(replicates, np.nan, dtype=float)
    for index in range(replicates):
        sampled_blocks = rng.choice(blocks, size=len(blocks), replace=True)
        sampled_indices = np.concatenate([indices[block] for block in sampled_blocks])
        estimates[index] = estimator(data.loc[sampled_indices])
    estimates = estimates[np.isfinite(estimates)]
    if not len(estimates):
        return {
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "bootstrap_replicates": 0,
            "spatial_blocks": int(len(blocks)),
        }
    p_value = 2.0 * min(
        (float(np.sum(estimates <= 0)) + 1.0) / (len(estimates) + 1.0),
        (float(np.sum(estimates >= 0)) + 1.0) / (len(estimates) + 1.0),
    )
    return {
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "p_value": float(min(p_value, 1.0)),
        "bootstrap_replicates": int(len(estimates)),
        "spatial_blocks": int(len(blocks)),
    }


def meta_test(
    frame: pd.DataFrame,
    degrees: int,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    data = frame[
        frame["shift_log"].notna()
        & frame["shift_se_log"].notna()
        & (frame["shift_se_log"] > 0)
        & frame["longitude"].notna()
        & frame["latitude"].notna()
    ].copy()
    model = reml_meta(data["shift_log"].to_numpy(), data["shift_se_log"].to_numpy())

    def estimator(sample: pd.DataFrame) -> float:
        return reml_meta(
            sample["shift_log"].to_numpy(), sample["shift_se_log"].to_numpy()
        )["estimate"]

    bootstrap = spatial_bootstrap(data, estimator, degrees, replicates, seed)
    tau = math.sqrt(model["tau2"]) if np.isfinite(model["tau2"]) else np.nan
    positive_fraction = (
        float(norm.cdf(model["estimate"] / tau))
        if np.isfinite(tau) and tau > 0
        else float(model["estimate"] > 0)
    )
    block_effects = data.assign(
        story_block=block_id(data["longitude"], data["latitude"], degrees)
    ).groupby("story_block", observed=True)["shift_log"].median()
    return {
        "n_catchments": int(data["GCIN"].nunique()),
        "estimate_log": model["estimate"],
        "estimate_pct": float(100.0 * np.expm1(model["estimate"])),
        "tau_log": tau,
        "i2": model["i2"],
        "model_positive_fraction": positive_fraction,
        "median_shift_log": float(data["shift_log"].median()),
        "median_shift_pct": float(100.0 * np.expm1(data["shift_log"].median())),
        "positive_catchment_fraction": float((data["shift_log"] > 0).mean()),
        "positive_block_fraction": float((block_effects > 0).mean()),
        **bootstrap,
        "ci_low_pct": float(100.0 * np.expm1(bootstrap["ci_low"])),
        "ci_high_pct": float(100.0 * np.expm1(bootstrap["ci_high"])),
    }


def paired_frame(dataset: pd.DataFrame, statistic: str) -> pd.DataFrame:
    source = dataset[
        (dataset["statistic"] == statistic)
        & dataset["regime"].isin(["low", "high"])
    ].copy()
    values = source.pivot(index="GCIN", columns="regime", values="shift_log")
    errors = source.pivot(index="GCIN", columns="regime", values="shift_se_log")
    metadata = source.drop_duplicates("GCIN").set_index("GCIN")
    result = metadata.join(values.add_prefix("effect_"), how="inner")
    result = result.join(errors.add_prefix("se_"), how="inner")
    result = result.reset_index()
    result["shift_log"] = result["effect_low"] - result["effect_high"]
    result["shift_se_log"] = np.sqrt(result["se_low"] ** 2 + result["se_high"] ** 2)
    return result


def weighted_slope(
    frame: pd.DataFrame,
    predictor: str,
    x_mean: float,
    x_sd: float,
) -> float:
    data = frame[
        frame[predictor].notna()
        & frame["shift_log"].notna()
        & frame["shift_se_log"].notna()
        & (frame["shift_se_log"] > 0)
    ]
    if len(data) < 3 or not np.isfinite(x_sd) or x_sd <= 0:
        return np.nan
    x = (data[predictor].to_numpy(float) - x_mean) / x_sd
    y = data["shift_log"].to_numpy(float)
    se = data["shift_se_log"].to_numpy(float)
    tau2 = reml_meta(y, se)["tau2"]
    weights = 1.0 / (se * se + tau2)
    design = np.column_stack([np.ones(len(x)), x])
    bread = np.linalg.pinv(design.T @ (weights[:, None] * design))
    coefficients = bread @ design.T @ (weights * y)
    return float(coefficients[1])


def climate_test(
    frame: pd.DataFrame,
    predictor: str,
    degrees: int,
    replicates: int,
    seed: int,
    x_mean: float | None = None,
    x_sd: float | None = None,
) -> dict[str, object]:
    data = frame[
        frame[predictor].notna()
        & frame["shift_log"].notna()
        & frame["shift_se_log"].notna()
        & frame["longitude"].notna()
        & frame["latitude"].notna()
    ].copy()
    mean = float(data[predictor].mean()) if x_mean is None else float(x_mean)
    sd = float(data[predictor].std(ddof=0)) if x_sd is None else float(x_sd)
    estimate = weighted_slope(data, predictor, mean, sd)

    def estimator(sample: pd.DataFrame) -> float:
        return weighted_slope(sample, predictor, mean, sd)

    bootstrap = spatial_bootstrap(data, estimator, degrees, replicates, seed)
    return {
        "n_catchments": int(data["GCIN"].nunique()),
        "predictor_mean": mean,
        "predictor_sd": sd,
        "estimate_log_per_sd": estimate,
        "estimate_pct_per_sd": float(100.0 * np.expm1(estimate)),
        **bootstrap,
        "ci_low_pct_per_sd": float(100.0 * np.expm1(bootstrap["ci_low"])),
        "ci_high_pct_per_sd": float(100.0 * np.expm1(bootstrap["ci_high"])),
    }


def discovery(cfg: dict, config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    audit_dir = Path(cfg["paths"]["audit_dir"])
    dataset_path = audit_dir / "global_story_dataset.parquet"
    dataset = pd.read_parquet(dataset_path)
    data = dataset[
        dataset["cohort_primary"]
        & (dataset["analysis_split"] == "discovery")
        & dataset["p_value"].notna()
    ].copy()
    design = cfg["design"]
    degrees = int(design["assignment_cell_degrees"])
    replicates = int(design["discovery_bootstrap_replicates"])
    seed = int(cfg["seed"])
    selection = cfg["candidate_selection"]
    global_rows: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []

    for (regime, statistic), group in data.groupby(
        ["regime", "statistic"], observed=True, sort=True
    ):
        key = f"global:{regime}:{statistic}"
        result = meta_test(group, degrees, replicates, stable_seed(seed, key))
        row = {"key": key, "type": "global", "regime": regime, "statistic": statistic, **result}
        global_rows.append(row)
        preregistered = statistic == cfg["distribution"]["primary_statistic"]
        discovered = (
            statistic in cfg["distribution"]["discovery_statistics"]
            and result["p_value"] < float(selection["discovery_p_threshold"])
            and result["n_catchments"] >= int(selection["minimum_catchments"])
            and result["spatial_blocks"] >= int(selection["minimum_spatial_blocks"])
        )
        if preregistered or discovered:
            candidates.append(
                {
                    "key": key,
                    "type": "global",
                    "regime": regime,
                    "statistic": statistic,
                    "selection": "pre_registered_primary" if preregistered else "discovery_rule",
                    "discovery_estimate": result["estimate_log"],
                    "discovery_p_value": result["p_value"],
                }
            )

    paired = paired_frame(data, cfg["distribution"]["primary_statistic"])
    paired_result = meta_test(
        paired,
        degrees,
        replicates,
        stable_seed(seed, "paired:low_minus_high:q50"),
    )
    paired_row = {
        "key": "paired:low_minus_high:q50",
        "type": "paired",
        "regime": "low_minus_high",
        "statistic": "q50",
        **paired_result,
    }
    global_rows.append(paired_row)
    candidates.append(
        {
            "key": paired_row["key"],
            "type": "paired",
            "regime": paired_row["regime"],
            "statistic": paired_row["statistic"],
            "selection": "pre_registered_primary",
            "discovery_estimate": paired_result["estimate_log"],
            "discovery_p_value": paired_result["p_value"],
        }
    )

    predictor_columns = [
        f"climate_{variable}_{method}"
        for variable, method in cfg["climate"]["variables"].items()
    ]
    if cfg["climate"].get("include_static_aridity", False):
        predictor_columns.append("Aridity")
    climate_rows: list[dict[str, object]] = []
    for regime in cfg["distribution"]["regimes"]:
        outcome = data[
            (data["regime"] == regime)
            & (data["statistic"] == cfg["distribution"]["primary_statistic"])
        ]
        for predictor in predictor_columns:
            key = f"climate:{regime}:{predictor}"
            result = climate_test(
                outcome,
                predictor,
                degrees,
                replicates,
                stable_seed(seed, key),
            )
            row = {"key": key, "type": "climate", "regime": regime, "predictor": predictor, **result}
            climate_rows.append(row)
            if (
                result["p_value"] < float(selection["discovery_p_threshold"])
                and result["n_catchments"] >= int(selection["minimum_catchments"])
                and result["spatial_blocks"] >= int(selection["minimum_spatial_blocks"])
            ):
                candidates.append(
                    {
                        "key": key,
                        "type": "climate",
                        "regime": regime,
                        "statistic": cfg["distribution"]["primary_statistic"],
                        "predictor": predictor,
                        "predictor_mean": result["predictor_mean"],
                        "predictor_sd": result["predictor_sd"],
                        "selection": "discovery_rule",
                        "discovery_estimate": result["estimate_log_per_sd"],
                        "discovery_p_value": result["p_value"],
                    }
                )

    global_frame = pd.DataFrame(global_rows)
    climate_frame = pd.DataFrame(climate_rows)
    global_frame.to_csv(audit_dir / "discovery_global_tests.csv", index=False)
    climate_frame.to_csv(audit_dir / "discovery_climate_tests.csv", index=False)
    lock = {
        "analysis_version": cfg["analysis_version"],
        "locked": True,
        "dataset_sha256": file_sha256(dataset_path),
        "config_sha256": file_sha256(config_path),
        "selection_rule": cfg["candidate_selection"],
        "candidates": candidates,
    }
    write_json(audit_dir / "locked_confirmation_candidates.json", lock)
    return lock


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(float)
    adjusted = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return pd.Series(adjusted, index=p_values.index)
    order = valid[np.argsort(values[valid])]
    running = 0.0
    count = len(order)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def confirmation(cfg: dict, config_path: Path = DEFAULT_CONFIG) -> dict[str, object]:
    audit_dir = Path(cfg["paths"]["audit_dir"])
    dataset_path = audit_dir / "global_story_dataset.parquet"
    lock_path = audit_dir / "locked_confirmation_candidates.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not lock.get("locked"):
        raise ValueError("Confirmation candidates are not locked")
    if lock["dataset_sha256"] != file_sha256(dataset_path):
        raise ValueError("Prepared dataset changed after candidate lock")
    if lock["config_sha256"] != file_sha256(config_path):
        raise ValueError("Analysis config changed after candidate lock")
    dataset = pd.read_parquet(dataset_path)
    data = dataset[
        dataset["cohort_primary"]
        & (dataset["analysis_split"] == "confirmation")
        & dataset["p_value"].notna()
    ].copy()
    design = cfg["design"]
    degrees = int(design["assignment_cell_degrees"])
    replicates = int(design["confirmation_bootstrap_replicates"])
    seed = int(cfg["seed"])
    rows: list[dict[str, object]] = []
    for candidate in lock["candidates"]:
        key = candidate["key"]
        if candidate["type"] == "global":
            sample = data[
                (data["regime"] == candidate["regime"])
                & (data["statistic"] == candidate["statistic"])
            ]
            result = meta_test(sample, degrees, replicates, stable_seed(seed, f"confirm:{key}"))
            estimate = result["estimate_log"]
        elif candidate["type"] == "paired":
            sample = paired_frame(data, candidate["statistic"])
            result = meta_test(sample, degrees, replicates, stable_seed(seed, f"confirm:{key}"))
            estimate = result["estimate_log"]
        elif candidate["type"] == "climate":
            sample = data[
                (data["regime"] == candidate["regime"])
                & (data["statistic"] == candidate["statistic"])
            ]
            result = climate_test(
                sample,
                candidate["predictor"],
                degrees,
                replicates,
                stable_seed(seed, f"confirm:{key}"),
                x_mean=candidate["predictor_mean"],
                x_sd=candidate["predictor_sd"],
            )
            estimate = result["estimate_log_per_sd"]
        else:
            raise ValueError(f"Unknown candidate type: {candidate['type']}")
        same_direction = bool(np.sign(estimate) == np.sign(candidate["discovery_estimate"]))
        rows.append({**candidate, **result, "confirmation_estimate": estimate, "same_direction": same_direction})

    results = pd.DataFrame(rows)
    results["holm_p_value"] = holm_adjust(results["p_value"])
    alpha = float(cfg["confirmation"]["familywise_alpha"])
    results["confirmed"] = (
        (results["holm_p_value"] < alpha)
        & results["same_direction"]
        & (results["ci_low"] * results["ci_high"] > 0)
    )
    results.to_csv(audit_dir / "confirmation_tests.csv", index=False)
    payload = {
        "analysis_version": cfg["analysis_version"],
        "candidate_count": int(len(results)),
        "confirmed_count": int(results["confirmed"].sum()),
        "confirmed_keys": results.loc[results["confirmed"], "key"].tolist(),
        "results": results.to_dict(orient="records"),
    }
    write_json(audit_dir / "confirmation_results.json", payload)
    return payload


def main() -> None:
    args = parse_args()
    cfg = load_config(args.analysis_config)
    if args.phase in {"prepare", "all"}:
        quality = prepare(cfg)
        print("prepared global story dataset")
        print(json.dumps(json_ready(quality), indent=2, sort_keys=True))
    if args.phase in {"discovery", "all"}:
        lock = discovery(cfg, args.analysis_config)
        print(f"locked {len(lock['candidates'])} confirmation candidates")
    if args.phase in {"confirmation", "all"}:
        result = confirmation(cfg, args.analysis_config)
        print(f"confirmed {result['confirmed_count']} of {result['candidate_count']} candidates")


if __name__ == "__main__":
    main()
