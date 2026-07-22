from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalBlock:
    fold: int
    period: str
    start_year: int
    end_year: int

    @property
    def years(self) -> set[int]:
        return set(range(self.start_year, self.end_year + 1))


def paired_temporal_blocks(cfg: dict) -> list[TemporalBlock]:
    n_folds = int(cfg["splits"]["n_folds"])
    pre_start, pre_end = pd.to_datetime(cfg["data"]["pre_window"])
    post_start, post_end = pd.to_datetime(cfg["data"]["post_window"])
    periods = {
        "pre": np.arange(pre_start.year, pre_end.year + 1, dtype=int),
        "post": np.arange(post_start.year, post_end.year + 1, dtype=int),
    }
    blocks: list[TemporalBlock] = []
    for period, years in periods.items():
        if len(years) < n_folds:
            raise ValueError(f"{period} period has fewer years than folds")
        for fold, part in enumerate(np.array_split(years, n_folds)):
            blocks.append(
                TemporalBlock(
                    fold=fold,
                    period=period,
                    start_year=int(part[0]),
                    end_year=int(part[-1]),
                )
            )
    return blocks


def blocks_frame(cfg: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold": block.fold,
                "period": block.period,
                "start_year": block.start_year,
                "end_year": block.end_year,
                "n_years": block.end_year - block.start_year + 1,
            }
            for block in paired_temporal_blocks(cfg)
        ]
    ).sort_values(["fold", "period"]).reset_index(drop=True)


def test_years(cfg: dict, fold: int) -> set[int]:
    years: set[int] = set()
    for block in paired_temporal_blocks(cfg):
        if block.fold == int(fold):
            years.update(block.years)
    if not years:
        raise ValueError(f"No temporal test blocks exist for fold {fold}")
    return years


def train_years(cfg: dict, fold: int) -> set[int]:
    all_years = set(range(int(cfg["data"]["start_year"]), int(cfg["data"]["end_year"]) + 1))
    return all_years - test_years(cfg, fold)


def expected_fold_for_year(cfg: dict) -> dict[int, int]:
    out: dict[int, int] = {}
    for block in paired_temporal_blocks(cfg):
        for year in block.years:
            if year in out:
                raise RuntimeError(f"Year {year} appears in more than one temporal test fold")
            out[year] = block.fold
    expected = set(range(int(cfg["data"]["start_year"]), int(cfg["data"]["end_year"]) + 1))
    if set(out) != expected:
        raise RuntimeError(f"Temporal folds do not cover the configured years: missing={sorted(expected - set(out))}")
    return out
