from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1] / "src" / "epsilon_model"
sys.path.insert(0, str(MODULE_DIR))

from analyze_global_story import (  # noqa: E402
    annual_distribution,
    assign_discovery_blocks,
    climate_change_table,
    holm_adjust,
    reml_meta,
)
from analyze_global_story_sensitivity import block_regression_test  # noqa: E402
from export_github_pages_data import read_global_field  # noqa: E402


class GlobalStoryTests(unittest.TestCase):
    def test_reml_recovers_common_effect(self) -> None:
        result = reml_meta(
            np.array([0.19, 0.20, 0.21, 0.20]),
            np.array([0.05, 0.05, 0.05, 0.05]),
        )
        self.assertAlmostEqual(result["estimate"], 0.20, places=5)
        self.assertGreaterEqual(result["tau2"], 0.0)

    def test_holm_adjustment_is_monotone_in_rank(self) -> None:
        adjusted = holm_adjust(pd.Series([0.01, 0.04, np.nan, 0.02]))
        self.assertTrue(np.isnan(adjusted.iloc[2]))
        self.assertAlmostEqual(adjusted.iloc[0], 0.03)
        self.assertAlmostEqual(adjusted.iloc[3], 0.04)
        self.assertAlmostEqual(adjusted.iloc[1], 0.04)

    def test_spatial_block_assignment_does_not_split_a_block(self) -> None:
        frame = pd.DataFrame(
            {
                "GCIN": [1, 2, 3],
                "longitude": [1.0, 2.0, 31.0],
                "latitude": [51.0, 52.0, -10.0],
            }
        )
        result = assign_discovery_blocks(frame, degrees=10, fraction=0.4, seed=42)
        self.assertEqual(result.loc[0, "assignment_block"], result.loc[1, "assignment_block"])
        self.assertEqual(result.loc[0, "analysis_split"], result.loc[1, "analysis_split"])

    def test_annual_distribution_preserves_quantile_order(self) -> None:
        sim = pd.DataFrame(
            {
                "GCIN": [1] * 5,
                "year": [2000] * 5,
                "fold": [0] * 5,
                "regime": ["low", "low", "mid", "high", "high"],
                "epsilon_effective": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        annual = annual_distribution(sim)
        all_values = annual[annual["regime"] == "all"].set_index("statistic")["value"]
        self.assertLessEqual(all_values["q25"], all_values["q50"])
        self.assertLessEqual(all_values["q50"], all_values["q75"])
        self.assertAlmostEqual(all_values["spread"], all_values["q75"] / all_values["q25"])

    def test_climate_change_requires_minimum_years_per_period(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            daily_dir = Path(temp_dir)
            rows = []
            for year in range(1981, 1991):
                rows.append({"GCIN": 1, "date": f"{year}-01-01", "temperature_C": 1.0})
            for year in range(1991, 2000):
                rows.append({"GCIN": 1, "date": f"{year}-01-01", "temperature_C": 2.0})
            pd.DataFrame(rows).to_parquet(daily_dir / "part.parquet", index=False)
            cfg = {
                "paths": {"physics_daily_dir": str(daily_dir)},
                "design": {"break_year": 1990},
                "climate": {
                    "variables": {"temperature_C": "difference"},
                    "minimum_years_per_period": 10,
                },
            }

            change, _ = climate_change_table(cfg)

            self.assertEqual(change.loc[0, "temperature_C_pre_years"], 10)
            self.assertEqual(change.loc[0, "temperature_C_post_years"], 9)
            self.assertTrue(pd.isna(change.loc[0, "climate_temperature_C_difference"]))

    def test_block_regression_bootstrap_excludes_invalid_rows_and_blocks(self) -> None:
        frame = pd.DataFrame(
            {
                "GCIN": [1, 2, 3, 4, 5],
                "shift_log": [0.1, 0.2, 0.3, 0.4, 0.5],
                "shift_se_log": [0.1, 0.1, 0.1, 0.1, np.nan],
                "longitude": [1.0, 2.0, 21.0, 22.0, 41.0],
                "latitude": [1.0, 2.0, 1.0, 2.0, 1.0],
                "driver": [0.0, 1.0, 0.0, 1.0, 1.0],
            }
        )

        result = block_regression_test(
            frame,
            ["driver"],
            {"driver": (0.5, 0.5)},
            block_degrees=10,
            replicates=20,
            seed=7,
        )["driver"]

        self.assertEqual(result["n_catchments"], 4)
        self.assertEqual(result["spatial_blocks"], 2)

    def test_public_field_export_keeps_only_eligible_all_recession_spread(self) -> None:
        from tempfile import TemporaryDirectory

        frame = pd.DataFrame(
            {
                "GCIN": [1, 2, 3],
                "regime": ["all", "low", "all"],
                "statistic": ["spread", "spread", "spread"],
                "cohort_primary": [True, True, False],
                "analysis_split": ["confirmation", "confirmation", "discovery"],
                "shift_pct": [12.0, 99.0, -4.0],
                "shift_ci_low_pct": [1.0, 2.0, -8.0],
                "shift_ci_high_pct": [20.0, 120.0, 1.0],
                "pre_years": [20, 20, 20],
                "post_years": [18, 18, 18],
                "paired_folds": [5, 5, 5],
                "p_value": [0.03, 0.01, 0.20],
            }
        )
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "field.parquet"
            frame.to_parquet(path, index=False)
            records = read_global_field(path)

        self.assertEqual(set(records), {1})
        self.assertEqual(records[1]["field_spread_analysis_split"], "confirmation")
        self.assertEqual(records[1]["field_spread_shift_pct"], 12.0)


if __name__ == "__main__":
    unittest.main()
