from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_global_story import (
    block_id,
    climate_test,
    effect_table,
    meta_test,
    paired_frame,
    reml_meta,
    spatial_bootstrap,
    stable_seed,
    write_json,
)
from config import load_config


CONFIG_PATH = Path("paper_repo/configs/global_story_analysis_v2.yaml")


def effect_metadata(dataset: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "GCIN",
        "longitude",
        "latitude",
        "analysis_split",
        "assignment_block",
        "pre_nse",
        "post_nse",
        "pre_kge",
        "post_kge",
    ]
    columns.extend(column for column in dataset.columns if column.startswith("cohort_"))
    columns.extend(column for column in dataset.columns if column.startswith("climate_"))
    columns.append("Aridity")
    return dataset[[column for column in columns if column in dataset.columns]].drop_duplicates("GCIN")


def attach_metadata(effects: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    return effects.merge(metadata, on="GCIN", how="left", validate="many_to_one")


def normalized_meta_result(result: dict[str, object]) -> dict[str, object]:
    return {
        "estimate": result["estimate_log"],
        "estimate_pct": result["estimate_pct"],
        "ci_low": result["ci_low"],
        "ci_high": result["ci_high"],
        "p_value": result["p_value"],
        "n_catchments": result["n_catchments"],
        "spatial_blocks": result["spatial_blocks"],
    }


def normalized_climate_result(result: dict[str, object]) -> dict[str, object]:
    return {
        "estimate": result["estimate_log_per_sd"],
        "estimate_pct": result["estimate_pct_per_sd"],
        "ci_low": result["ci_low"],
        "ci_high": result["ci_high"],
        "p_value": result["p_value"],
        "n_catchments": result["n_catchments"],
        "spatial_blocks": result["spatial_blocks"],
    }


def weighted_block_regression(
    frame: pd.DataFrame,
    predictors: list[str],
    scales: dict[str, tuple[float, float]],
    block_degrees: int,
) -> dict[str, float]:
    required = ["shift_log", "shift_se_log", "longitude", "latitude", *predictors]
    data = frame.dropna(subset=required).copy()
    data = data[data["shift_se_log"] > 0]
    if len(data) < len(predictors) + 3:
        return {predictor: np.nan for predictor in predictors}
    y = data["shift_log"].to_numpy(float)
    se = data["shift_se_log"].to_numpy(float)
    tau2 = reml_meta(y, se)["tau2"]
    weights = 1.0 / (se * se + tau2)
    x_columns = []
    for predictor in predictors:
        mean, sd = scales[predictor]
        if not np.isfinite(sd) or sd <= 0:
            return {name: np.nan for name in predictors}
        x_columns.append((data[predictor].to_numpy(float) - mean) / sd)
    blocks = block_id(data["longitude"], data["latitude"], block_degrees)
    block_dummies = pd.get_dummies(blocks, drop_first=True, dtype=float).to_numpy()
    design = np.column_stack([np.ones(len(data)), *x_columns, block_dummies])
    bread = np.linalg.pinv(design.T @ (weights[:, None] * design))
    coefficients = bread @ design.T @ (weights * y)
    return {predictor: float(coefficients[index + 1]) for index, predictor in enumerate(predictors)}


def block_regression_test(
    frame: pd.DataFrame,
    predictors: list[str],
    scales: dict[str, tuple[float, float]],
    block_degrees: int,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, object]]:
    required = ["shift_log", "shift_se_log", "longitude", "latitude", *predictors]
    data = frame.dropna(subset=required).copy()
    data = data[data["shift_se_log"] > 0]
    estimates = weighted_block_regression(data, predictors, scales, block_degrees)
    results: dict[str, dict[str, object]] = {}
    for predictor in predictors:
        def estimator(sample: pd.DataFrame, target: str = predictor) -> float:
            return weighted_block_regression(sample, predictors, scales, block_degrees)[target]

        bootstrap = spatial_bootstrap(
            data,
            estimator,
            degrees=block_degrees,
            replicates=replicates,
            seed=stable_seed(seed, f"block-regression:{predictor}"),
        )
        estimate = estimates[predictor]
        results[predictor] = {
            "estimate": estimate,
            "estimate_pct": float(100.0 * np.expm1(estimate)),
            "ci_low": bootstrap["ci_low"],
            "ci_high": bootstrap["ci_high"],
            "p_value": bootstrap["p_value"],
            "n_catchments": int(data["GCIN"].nunique()),
            "spatial_blocks": bootstrap["spatial_blocks"],
        }
    return results


def scenario_dataset(
    cfg: dict,
    annual: pd.DataFrame,
    metadata: pd.DataFrame,
    break_year: int,
    min_days: int,
    audit_dir: Path,
) -> pd.DataFrame:
    path = audit_dir / f"effects_break{break_year}_days{min_days}.parquet"
    if path.exists():
        effects = pd.read_parquet(path)
    else:
        design = cfg["design"]
        effects = effect_table(
            annual,
            break_year=break_year,
            min_days=min_days,
            min_years=int(design["annual_min_years_per_period"]),
            min_identifying_years=int(design["annual_min_identifying_years_per_period"]),
            hac_lag_years=int(design["hac_lag_years"]),
        )
        effects.to_parquet(path, index=False)
    return attach_metadata(effects, metadata)


def candidate_sample(dataset: pd.DataFrame, candidate: pd.Series) -> pd.DataFrame:
    if candidate["type"] == "paired":
        return paired_frame(dataset, candidate["statistic"])
    return dataset[
        (dataset["regime"] == candidate["regime"])
        & (dataset["statistic"] == candidate["statistic"])
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    cfg = load_config(args.analysis_config)
    audit_dir = Path(cfg["paths"]["audit_dir"])
    dataset = pd.read_parquet(audit_dir / "global_story_dataset.parquet")
    annual = pd.read_parquet(audit_dir / "annual_epsilon_distribution.parquet")
    confirmation = pd.read_csv(audit_dir / "confirmation_tests.csv")
    confirmed = confirmation[confirmation["confirmed"]].copy()
    metadata = effect_metadata(dataset)
    seed = int(cfg["seed"])
    replicates = int(cfg["design"]["discovery_bootstrap_replicates"])
    primary_degrees = int(cfg["design"]["assignment_cell_degrees"])
    rows: list[dict[str, object]] = []

    global_candidates = confirmed[confirmed["type"].isin(["global", "paired"])]
    scenarios = {
        "break_1985": (1985, 5),
        "primary_1990_days5": (1990, 5),
        "break_1995": (1995, 5),
        "days_min_3": (1990, 3),
        "days_min_10": (1990, 10),
    }
    scenario_cache: dict[str, pd.DataFrame] = {}
    for name, (break_year, min_days) in scenarios.items():
        if name == "primary_1990_days5":
            scenario_cache[name] = dataset
        else:
            scenario_cache[name] = scenario_dataset(
                cfg, annual, metadata, break_year, min_days, audit_dir
            )

    for _, candidate in global_candidates.iterrows():
        for scenario, scenario_data in scenario_cache.items():
            eligible = scenario_data[
                scenario_data["cohort_primary"]
                & (scenario_data["analysis_split"] == "confirmation")
                & scenario_data["p_value"].notna()
            ]
            sample = candidate_sample(eligible, candidate)
            result = meta_test(
                sample,
                primary_degrees,
                replicates,
                stable_seed(seed, f"sensitivity:{candidate['key']}:{scenario}"),
            )
            rows.append(
                {
                    "key": candidate["key"],
                    "dimension": "time_support",
                    "setting": scenario,
                    **normalized_meta_result(result),
                }
            )

    cohorts = ["primary", *[item["name"] for item in cfg["reliability"]["sensitivity_cohorts"]]]
    primary_confirmation = dataset[
        (dataset["analysis_split"] == "confirmation") & dataset["p_value"].notna()
    ]
    for _, candidate in global_candidates.iterrows():
        for cohort in cohorts:
            cohort_column = "cohort_primary" if cohort == "primary" else f"cohort_{cohort}"
            sample = candidate_sample(primary_confirmation[primary_confirmation[cohort_column]], candidate)
            result = meta_test(
                sample,
                primary_degrees,
                replicates,
                stable_seed(seed, f"sensitivity:{candidate['key']}:cohort:{cohort}"),
            )
            rows.append(
                {
                    "key": candidate["key"],
                    "dimension": "reliability",
                    "setting": cohort,
                    **normalized_meta_result(result),
                }
            )
        sample = candidate_sample(primary_confirmation[primary_confirmation["cohort_primary"]], candidate)
        for degrees in cfg["design"]["bootstrap_cell_degrees"]:
            result = meta_test(
                sample,
                int(degrees),
                replicates,
                stable_seed(seed, f"sensitivity:{candidate['key']}:blocks:{degrees}"),
            )
            rows.append(
                {
                    "key": candidate["key"],
                    "dimension": "spatial_blocks",
                    "setting": f"{degrees}_degree",
                    **normalized_meta_result(result),
                }
            )

    climate_candidates = confirmed[confirmed["type"] == "climate"]
    for _, candidate in climate_candidates.iterrows():
        base = primary_confirmation[
            (primary_confirmation["regime"] == candidate["regime"])
            & (primary_confirmation["statistic"] == candidate["statistic"])
        ]
        for cohort in cohorts:
            cohort_column = "cohort_primary" if cohort == "primary" else f"cohort_{cohort}"
            result = climate_test(
                base[base[cohort_column]],
                candidate["predictor"],
                primary_degrees,
                replicates,
                stable_seed(seed, f"sensitivity:{candidate['key']}:cohort:{cohort}"),
                x_mean=candidate["predictor_mean"],
                x_sd=candidate["predictor_sd"],
            )
            rows.append(
                {
                    "key": candidate["key"],
                    "dimension": "reliability",
                    "setting": cohort,
                    **normalized_climate_result(result),
                }
            )
        for degrees in cfg["design"]["bootstrap_cell_degrees"]:
            result = climate_test(
                base[base["cohort_primary"]],
                candidate["predictor"],
                int(degrees),
                replicates,
                stable_seed(seed, f"sensitivity:{candidate['key']}:blocks:{degrees}"),
                x_mean=candidate["predictor_mean"],
                x_sd=candidate["predictor_sd"],
            )
            rows.append(
                {
                    "key": candidate["key"],
                    "dimension": "spatial_blocks",
                    "setting": f"{degrees}_degree",
                    **normalized_climate_result(result),
                }
            )

    climate_by_regime: dict[str, list[pd.Series]] = {}
    for _, candidate in climate_candidates.iterrows():
        climate_by_regime.setdefault(candidate["regime"], []).append(candidate)
    multivariable_payload: dict[str, object] = {}
    for regime, candidates in climate_by_regime.items():
        predictors = [candidate["predictor"] for candidate in candidates]
        scales = {
            candidate["predictor"]: (
                float(candidate["predictor_mean"]),
                float(candidate["predictor_sd"]),
            )
            for candidate in candidates
        }
        base = primary_confirmation[
            primary_confirmation["cohort_primary"]
            & (primary_confirmation["regime"] == regime)
            & (primary_confirmation["statistic"] == "q50")
        ]
        correlation = base[predictors].corr().to_dict()
        results = block_regression_test(
            base,
            predictors,
            scales,
            primary_degrees,
            replicates,
            stable_seed(seed, f"multivariable:{regime}"),
        )
        multivariable_payload[regime] = {
            "predictor_correlation": correlation,
            "block_fixed_effect_results": results,
        }
        for candidate in candidates:
            result = results[candidate["predictor"]]
            rows.append(
                {
                    "key": candidate["key"],
                    "dimension": "multivariable_block_fixed",
                    "setting": "precipitation_and_soil_moisture",
                    **result,
                }
            )

    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(audit_dir / "sensitivity_tests.csv", index=False)
    summaries: list[dict[str, object]] = []
    required_fraction = float(cfg["confirmation"]["require_sensitivity_direction_fraction"])
    for _, candidate in confirmed.iterrows():
        group = sensitivity[sensitivity["key"] == candidate["key"]].copy()
        direction = np.sign(float(candidate["confirmation_estimate"]))
        same = np.sign(group["estimate"].to_numpy(float)) == direction
        ci_excludes = group["ci_low"].to_numpy(float) * group["ci_high"].to_numpy(float) > 0
        mandatory = True
        if candidate["type"] == "climate":
            multivariable = group[group["dimension"] == "multivariable_block_fixed"]
            mandatory = bool(
                len(multivariable)
                and np.all(np.sign(multivariable["estimate"].to_numpy(float)) == direction)
                and np.all(
                    multivariable["ci_low"].to_numpy(float)
                    * multivariable["ci_high"].to_numpy(float)
                    > 0
                )
            )
        summaries.append(
            {
                "key": candidate["key"],
                "confirmation_estimate": float(candidate["confirmation_estimate"]),
                "sensitivity_checks": int(len(group)),
                "same_direction_fraction": float(np.mean(same)) if len(group) else np.nan,
                "ci_excludes_zero_fraction": float(np.mean(ci_excludes)) if len(group) else np.nan,
                "mandatory_multivariable_evidence": mandatory,
                "robust": bool(len(group) and np.mean(same) >= required_fraction and mandatory),
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(audit_dir / "robust_story_candidates.csv", index=False)
    payload = {
        "analysis_version": cfg["analysis_version"],
        "confirmed_before_sensitivity": int(len(confirmed)),
        "robust_after_sensitivity": int(summary["robust"].sum()),
        "robust_keys": summary.loc[summary["robust"], "key"].tolist(),
        "candidate_summary": summary.to_dict(orient="records"),
        "multivariable": multivariable_payload,
    }
    write_json(audit_dir / "robust_story_results.json", payload)
    print(json.dumps(payload["robust_keys"], indent=2))


if __name__ == "__main__":
    main()
