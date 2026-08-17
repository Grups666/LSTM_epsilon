from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parents[1] / "src" / "epsilon_model"
sys.path.insert(0, str(MODULE_DIR))

from analyze_prepost_shifts import bh_fdr, fit_shift  # noqa: E402


class PrePostShiftTests(unittest.TestCase):
    def test_fold_fixed_effect_recovers_known_shift(self) -> None:
        rows = []
        periods = {
            0: (range(1971, 1981), range(1991, 2001)),
            1: (range(1981, 1991), range(2001, 2011)),
        }
        for fold, (pre_years, post_years) in periods.items():
            fold_scale = 1.0 if fold == 0 else 2.5
            for index, year in enumerate(pre_years):
                rows.append((year, fold, fold_scale * (1.0 + index / 100.0), 3))
            for index, year in enumerate(post_years):
                rows.append((year, fold, 1.2 * fold_scale * (1.0 + index / 100.0), 3))
        series = pd.DataFrame(rows, columns=["year", "fold", "value", "n_days"])
        result = fit_shift(series, 1990, 10, 1, 5, 1)
        self.assertAlmostEqual(float(result["shift_pct"]), 20.0, places=8)
        self.assertEqual(result["identifying_pre_years"], 20)
        self.assertEqual(result["identifying_post_years"], 20)

    def test_weakly_paired_series_is_insufficient(self) -> None:
        rows = []
        for year in range(1971, 1981):
            rows.append((year, 0, 1.0, 3))
        for year in range(1991, 2001):
            rows.append((year, 1, 1.1, 3))
        rows.extend([(1988, 2, 1.0, 3), (1989, 2, 1.0, 3), (1991, 2, 1.1, 3)])
        series = pd.DataFrame(rows, columns=["year", "fold", "value", "n_days"])
        result = fit_shift(series, 1990, 10, 1, 5, 1)
        self.assertTrue(np.isnan(result["shift_pct"]))
        self.assertLess(result["identifying_pre_years"], 5)
        self.assertLess(result["identifying_post_years"], 5)

    def test_bh_fdr_ignores_missing_values(self) -> None:
        adjusted = bh_fdr(pd.Series([0.01, 0.04, np.nan, 0.03]))
        np.testing.assert_allclose(adjusted[[0, 1, 3]], [0.03, 0.04, 0.04])
        self.assertTrue(np.isnan(adjusted.iloc[2]))


if __name__ == "__main__":
    unittest.main()
