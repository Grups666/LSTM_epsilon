from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config


DEFAULT_CONFIG = Path("paper_repo/configs/global_story_analysis_v2.yaml")


def finite(value: object) -> float | int | None:
    if value is None or not np.isfinite(float(value)):
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def compact_result(row: pd.Series) -> dict[str, object]:
    estimate = row.get("estimate_pct")
    ci_low = row.get("ci_low_pct")
    ci_high = row.get("ci_high_pct")
    if estimate is None or pd.isna(estimate):
        estimate = row.get("estimate_pct_per_sd")
    if ci_low is None or pd.isna(ci_low):
        ci_low = row.get("ci_low_pct_per_sd")
    if ci_high is None or pd.isna(ci_high):
        ci_high = row.get("ci_high_pct_per_sd")
    return {
        "catchments": int(row["n_catchments"]),
        "estimatePct": finite(estimate),
        "ciLowPct": finite(ci_low),
        "ciHighPct": finite(ci_high),
        "pValue": finite(row["p_value"]),
        "holmPValue": finite(row.get("holm_p_value")),
        "positiveCatchmentFraction": finite(row.get("positive_catchment_fraction")),
        "positiveBlockFraction": finite(row.get("positive_block_fraction")),
        "i2": finite(row.get("i2")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.analysis_config)
    audit_dir = Path(cfg["paths"]["audit_dir"])
    distribution = pd.read_csv(audit_dir / "validation_distribution_tests.csv")
    discovery = pd.read_csv(audit_dir / "discovery_global_tests.csv")
    confirmation = pd.read_csv(audit_dir / "confirmation_tests.csv")
    robust = pd.read_csv(audit_dir / "robust_story_candidates.csv").set_index("key")
    validation = json.loads((audit_dir / "global_story_validation.json").read_text())
    dataset = pd.read_parquet(
        audit_dir / "global_story_dataset.parquet",
        columns=["GCIN", "cohort_primary", "regime", "statistic"],
    )
    reliability_catchments = int(
        dataset.loc[dataset["cohort_primary"], "GCIN"].nunique()
    )
    field_catchments = int(
        distribution.loc[
            (distribution["statistic"] == "q50")
            & (distribution["split"] == "full"),
            "n_catchments",
        ].iloc[0]
    )

    def distribution_result(statistic: str, split: str) -> dict[str, object]:
        key = f"global:all:{statistic}"
        if split == "discovery":
            return compact_result(discovery[discovery["key"] == key].iloc[0])
        if split == "confirmation":
            return compact_result(confirmation[confirmation["key"] == key].iloc[0])
        row = distribution[
            (distribution["statistic"] == statistic) & (distribution["split"] == split)
        ].iloc[0]
        return compact_result(row)

    def confirmation_result(key: str) -> dict[str, object]:
        row = confirmation[confirmation["key"] == key].iloc[0]
        return compact_result(row)

    soil_key_all = "climate:all:climate_SM_%_difference"
    soil_key_high = "climate:high:climate_SM_%_difference"
    soil_all = validation["climate"]["all"]["multivariable_block_fixed"][
        "climate_SM_%_difference"
    ]
    soil_high = validation["climate"]["high"]["multivariable_block_fixed"][
        "climate_SM_%_difference"
    ]

    def climate_result(key: str, joint: dict[str, object], regime: str) -> dict[str, object]:
        confirm = confirmation_result(key)
        return {
            "confirmation": confirm,
            "jointPrecipitationSoilMoistureBlockFixed": {
                "catchments": int(joint["n_catchments"]),
                "estimatePctPerDiscoverySd": finite(joint["estimate_pct"]),
                "ciLowPctPerDiscoverySd": finite(100.0 * np.expm1(joint["ci_low"])),
                "ciHighPctPerDiscoverySd": finite(100.0 * np.expm1(joint["ci_high"])),
                "pValue": finite(joint["p_value"]),
            },
            "leaveOne20DegreeBlock": validation["climate"][regime][
                "leave_one_20_degree_block"
            ]["climate_SM_%_difference"],
            "robust": bool(robust.loc[key, "robust"]),
        }

    payload = {
        "analysisVersion": cfg["analysis_version"],
        "design": {
            "breakYear": int(cfg["design"]["break_year"]),
            "discoveryFraction": float(cfg["design"]["discovery_fraction"]),
            "assignmentCellDegrees": int(cfg["design"]["assignment_cell_degrees"]),
            "bootstrapCellDegrees": cfg["design"]["bootstrap_cell_degrees"],
            "annualMinimumDays": int(cfg["design"]["annual_min_days"]),
            "annualMinimumYearsPerEra": int(cfg["design"]["annual_min_years_per_period"]),
            "reliability": "NSE > 0.5 in both eras",
            "confirmationCorrection": "Holm family-wise error control",
        },
        "coverage": {
            "reliabilityQualifiedCatchments": reliability_catchments,
            "fieldEligibleCatchments": field_catchments,
            "fieldCoverageFraction": field_catchments / reliability_catchments,
        },
        "fieldEvidence": {
            "distributionSpread": {
                "discovery": distribution_result("spread", "discovery"),
                "confirmation": distribution_result("spread", "confirmation"),
                "fullDescriptive": distribution_result("spread", "full"),
                "leaveOne20DegreeBlock": validation[
                    "distribution_leave_one_20_degree_block"
                ]["spread"],
                "robust": bool(robust.loc["global:all:spread", "robust"]),
            },
            "distributionMedian": {
                "discovery": distribution_result("q50", "discovery"),
                "confirmation": distribution_result("q50", "confirmation"),
                "fullDescriptive": distribution_result("q50", "full"),
                "leaveOne20DegreeBlock": validation[
                    "distribution_leave_one_20_degree_block"
                ]["q50"],
                "robust": bool(robust.loc["global:all:q50", "robust"]),
            },
        },
        "hydroclimateAssociation": {
            "soilMoistureAllRecession": climate_result(soil_key_all, soil_all, "all"),
            "soilMoistureHighFlow": climate_result(soil_key_high, soil_high, "high"),
            "predictorCorrelation": validation["climate"]["all"]["predictor_correlation"],
            "interpretation": "Associative only; precipitation loses independent support after joint adjustment.",
        },
        "nonReplicated": [
            "Low-flow and high-flow direction contrasts did not reproduce across the spatial split.",
            "Precipitation-change associations did not retain independent interval evidence after joint adjustment for soil moisture and spatial blocks.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
