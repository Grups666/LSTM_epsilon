from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def project_path(path: str | Path) -> Path:
    return Path(path)


def output_dir(cfg: dict[str, Any]) -> Path:
    out = project_path(cfg["paths"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    return out
