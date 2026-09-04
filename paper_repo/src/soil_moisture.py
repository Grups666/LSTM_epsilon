from __future__ import annotations

import numpy as np
import pandas as pd


ROOT_ZONE_COLUMNS = ("swvl1", "swvl2", "swvl3")
ROOT_ZONE_LAYER_THICKNESS_M = np.array([0.07, 0.21, 0.72], dtype="float64")
ROOT_ZONE_DEPTH_M = float(ROOT_ZONE_LAYER_THICKNESS_M.sum())


def root_zone_soil_moisture(df: pd.DataFrame) -> np.ndarray:
    """Return 0-100 cm ERA5-Land volumetric soil moisture."""

    values = df.loc[:, ROOT_ZONE_COLUMNS].to_numpy("float64")
    soil_moisture = np.average(values, axis=1, weights=ROOT_ZONE_LAYER_THICKNESS_M)
    return np.clip(
        np.nan_to_num(soil_moisture, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    ).astype("float32")
