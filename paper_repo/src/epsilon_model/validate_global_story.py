from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_global_story import block_id, meta_test, reml_meta, stable_seed, write_json
from analyze_global_story_sensitivity import block_regression_test, weighted_block_regression
from config import load_config


DEFAULT_CONFIG = Path("paper_repo/configs/global_story_analysis_v2.yaml")


def meta_estimate(frame: pd.DataFrame) -> float:
    data = frame.dropna(subset=["shift_log", "shift_se_log"])
    data = data[data["shift_se_log"] > 0]
    return float(
        reml_meta(data["shift_log"].to_numpy(), data["shift_se_log"].to_numpy())["estimate"]
    )


def leave_one_block_meta(frame: pd.DataFrame, degrees: int) -> tuple[pd.DataFrame, dict[str, float]]:
    data = frame.dropna(subset=["shift_log", "shift_se_log", "longitude", "latitude"]).copy()
    data["block"] = block_id(data["longitude"], data["latitude"], degrees)
    rows = []
    for block in sorted(data["block"].dropna().unique()):
        sample = data[data["block"] != block]
        estimate = meta_estimate(sample)
        rows.append(
            {
                "block": block,
                "omitted_catchments": int((data["block"] == block).sum()),
                "remaining_catchments": int(len(sample)),
                "estimate": estimate,
                "estimate_pct": float(100.0 * np.expm1(estimate)),
            }
        )
    result = pd.DataFrame(rows)
    summary = {
        "blocks": int(len(result)),
        "positive_fraction": float((result["estimate"] > 0).mean()),
        "minimum_pct": float(result["estimate_pct"].min()),
        "maximum_pct": float(result["estimate_pct"].max()),
    }
    return result, summary


def leave_one_block_regression(
    frame: pd.DataFrame,
    predictors: list[str],
    scales: dict[str, tuple[float, float]],
    degrees: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    required = ["shift_log", "shift_se_log", "longitude", "latitude", *predictors]
    data = frame.dropna(subset=required).copy()
    data["block"] = block_id(data["longitude"], data["latitude"], degrees)
    rows = []
    for block in sorted(data["block"].dropna().unique()):
        sample = data[data["block"] != block]
        estimates = weighted_block_regression(sample, predictors, scales, degrees)
        for predictor, estimate in estimates.items():
            rows.append(
                {
                    "block": block,
                    "predictor": predictor,
                    "omitted_catchments": int((data["block"] == block).sum()),
                    "remaining_catchments": int(len(sample)),
                    "estimate": estimate,
                    "estimate_pct": float(100.0 * np.expm1(estimate)),
                }
            )
    result = pd.DataFrame(rows)
    summary = {}
    for predictor, group in result.groupby("predictor", observed=True):
        summary[predictor] = {
            "blocks": int(len(group)),
            "negative_fraction": float((group["estimate"] < 0).mean()),
            "minimum_pct": float(group["estimate_pct"].min()),
            "maximum_pct": float(group["estimate_pct"].max()),
        }
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.analysis_config)
    audit_dir = Path(cfg["paths"]["audit_dir"])
    dataset = pd.read_parquet(audit_dir / "global_story_dataset.parquet")
    lock = json.loads((audit_dir / "locked_confirmation_candidates.json").read_text())
    seed = int(cfg["seed"])
    replicates = int(cfg["design"]["confirmation_bootstrap_replicates"])

    all_regime = dataset[
        dataset["cohort_primary"] & (dataset["regime"] == "all")
    ].copy()
    distribution_rows = []
    influence_rows = []
    influence_summary = {}
    for statistic in ["q25", "q50", "q75", "spread"]:
        statistic_data = all_regime[all_regime["statistic"] == statistic]
        for split in ["discovery", "confirmation", "full"]:
            sample = (
                statistic_data
                if split == "full"
                else statistic_data[statistic_data["analysis_split"] == split]
            )
            result = meta_test(
                sample,
                degrees=10,
                replicates=replicates,
                seed=stable_seed(seed, f"validation:{statistic}:{split}"),
            )
            distribution_rows.append({"statistic": statistic, "split": split, **result})
        loo, summary = leave_one_block_meta(statistic_data, degrees=20)
        loo.insert(0, "statistic", statistic)
        influence_rows.append(loo)
        influence_summary[statistic] = summary

    climate_predictors = [
        "climate_precipitation_mmd_log_ratio",
        "climate_SM_%_difference",
    ]
    climate_payload = {}
    climate_influence_rows = []
    for regime in ["all", "high"]:
        sample = dataset[
            dataset["cohort_primary"]
            & (dataset["analysis_split"] == "confirmation")
            & (dataset["regime"] == regime)
            & (dataset["statistic"] == "q50")
        ].copy()
        locked = {
            candidate["predictor"]: candidate
            for candidate in lock["candidates"]
            if candidate["type"] == "climate" and candidate["regime"] == regime
        }
        scales = {
            predictor: (
                float(locked[predictor]["predictor_mean"]),
                float(locked[predictor]["predictor_sd"]),
            )
            for predictor in climate_predictors
        }
        univariate = {}
        for predictor in climate_predictors:
            result = block_regression_test(
                sample,
                [predictor],
                {predictor: scales[predictor]},
                block_degrees=10,
                replicates=replicates,
                seed=stable_seed(seed, f"validation:block-fixed:{regime}:{predictor}"),
            )[predictor]
            univariate[predictor] = result
        multivariable = block_regression_test(
            sample,
            climate_predictors,
            scales,
            block_degrees=10,
            replicates=replicates,
            seed=stable_seed(seed, f"validation:block-fixed:{regime}:joint"),
        )
        loo, loo_summary = leave_one_block_regression(
            sample, climate_predictors, scales, degrees=20
        )
        loo.insert(0, "regime", regime)
        climate_influence_rows.append(loo)
        complete = sample.dropna(
            subset=["shift_log", "shift_se_log", "longitude", "latitude", *climate_predictors]
        )
        complete = complete[complete["shift_se_log"] > 0]
        climate_payload[regime] = {
            "n_catchments": int(complete["GCIN"].nunique()),
            "predictor_correlation": float(complete[climate_predictors].corr().iloc[0, 1]),
            "univariate_block_fixed": univariate,
            "multivariable_block_fixed": multivariable,
            "leave_one_20_degree_block": loo_summary,
        }

    distribution = pd.DataFrame(distribution_rows)
    distribution.to_csv(audit_dir / "validation_distribution_tests.csv", index=False)
    pd.concat(influence_rows, ignore_index=True).to_csv(
        audit_dir / "validation_leave_one_block_distribution.csv", index=False
    )
    pd.concat(climate_influence_rows, ignore_index=True).to_csv(
        audit_dir / "validation_leave_one_block_climate.csv", index=False
    )
    payload = {
        "analysis_version": cfg["analysis_version"],
        "distribution_leave_one_20_degree_block": influence_summary,
        "climate": climate_payload,
        "interpretation_guardrails": [
            "Confirmation-split estimates remain the inferential results.",
            "Full-sample estimates and leave-one-block analyses are descriptive robustness checks.",
            "Climate regressions are associative and do not identify causal effects.",
        ],
    }
    write_json(audit_dir / "global_story_validation.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
