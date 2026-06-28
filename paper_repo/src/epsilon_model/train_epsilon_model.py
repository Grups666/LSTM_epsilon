from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from config import load_config, output_dir
from physics_model import EpsilonStateResetModel, PhysicsInformedLoss


EPS = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--year-start", type=int, default=None)
    parser.add_argument("--year-end", type=int, default=None)
    parser.add_argument("--run-label", type=str, default=None)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detect_recession_paper(q: np.ndarray, min_len: int = 4, drop_first: int = 1, decreasing_rate: bool = True) -> np.ndarray:
    q_proc = np.asarray(q, dtype="float64").copy()
    q_proc[~np.isfinite(q_proc)] = np.inf
    mask = np.zeros(len(q_proc), dtype=bool)
    i = 0
    while i < len(q_proc) - 1:
        if q_proc[i + 1] < q_proc[i]:
            seg = [i, i + 1]
            r_prev = q_proc[i] - q_proc[i + 1]
            j = i + 1
            while j < len(q_proc) - 1 and q_proc[j + 1] < q_proc[j]:
                r_cur = q_proc[j] - q_proc[j + 1]
                if (not decreasing_rate) or (r_cur < r_prev):
                    seg.append(j + 1)
                    r_prev = r_cur
                    j += 1
                else:
                    break
            if len(seg) >= min_len:
                for idx in seg[drop_first:]:
                    mask[idx] = True
            i = j
        else:
            i += 1
    return mask


def generate_state_reset_tensors(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask_int = mask.astype("int8")
    start = np.zeros_like(mask_int)
    tau = np.zeros(len(mask_int), dtype="float32")
    if len(mask_int):
        start[0] = mask_int[0]
        start[1:] = (mask_int[1:] == 1) & (mask_int[:-1] == 0)
    current = 0
    for i, flag in enumerate(mask_int):
        if flag == 0:
            current = 0
        else:
            current += 1
            tau[i] = current
    return start.astype("float32"), tau


def load_snow_mask(cfg: dict) -> pd.DataFrame | None:
    snow_path = cfg["recession"].get("snow_mask_csv")
    if not snow_path:
        return None
    path = Path(snow_path)
    if not path.exists():
        raise FileNotFoundError(f"snow_mask_csv does not exist: {path}")
    snow = pd.read_csv(path)
    required = {"GCIN", "month", "mean_q"}
    if "gridcode" in snow.columns and "GCIN" not in snow.columns:
        snow = snow.rename(columns={"gridcode": "GCIN"})
    missing = required.difference(snow.columns)
    if missing:
        raise ValueError(f"snow_mask_csv is missing columns: {sorted(missing)}")
    snow["GCIN"] = snow["GCIN"].astype(int)
    snow["month"] = snow["month"].astype(int)
    return snow


def apply_snow_mask(mask: np.ndarray, dates: pd.Series, gcin: int, snow_df: pd.DataFrame | None, threshold: float) -> np.ndarray:
    if snow_df is None:
        return mask
    basin_snow = snow_df[snow_df["GCIN"] == int(gcin)]
    if basin_snow.empty:
        return mask
    snowy_months = basin_snow.loc[basin_snow["mean_q"] > threshold, "month"].to_numpy()
    if len(snowy_months) == 0:
        return mask
    out = mask.copy()
    months = pd.to_datetime(dates).dt.month.to_numpy()
    out[np.isin(months, snowy_months)] = False
    return out


def apply_cold_temperature_mask(mask: np.ndarray, temperature_c: np.ndarray, threshold_c: float | None) -> np.ndarray:
    if threshold_c is None:
        return mask
    out = mask.copy()
    temp = np.asarray(temperature_c, dtype="float64")
    out[np.isfinite(temp) & (temp <= float(threshold_c))] = False
    return out


def finalize_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0).astype("float32")
    std = np.nanstd(values, axis=0).astype("float32")
    std[~np.isfinite(std) | (std < EPS)] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    return mean, std


@dataclass
class BasinSeries:
    gcin: int
    dates: np.ndarray
    x_raw: np.ndarray
    z_norm: np.ndarray
    c_norm: np.ndarray
    y_raw: np.ndarray
    rec_mask: np.ndarray
    start_mask: np.ndarray
    bounds: np.ndarray
    split_labels: np.ndarray
    train_start: int
    train_end: int


def load_physics_frame(cfg: dict, years: list[int], gcins: set[int]) -> pd.DataFrame:
    daily_dir = Path(cfg["paths"].get("physics_daily_dir", "_private/processed/epsilon_physics_daily_parquet"))
    frames = []
    cols = ["GCIN", "date", *cfg["physics"]["dynamic_columns"], cfg["physics"]["target_column"], "observed_AET_mm"]
    for year in years:
        path = daily_dir / f"epsilon_physics_daily_{year}.parquet"
        log(f"reading {path}")
        df = pd.read_parquet(path, columns=cols)
        df = df[df["GCIN"].astype(int).isin(gcins)]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).sort_values(["GCIN", "date"])
    out["GCIN"] = out["GCIN"].astype("int32")
    return out


def split_gcins(cfg: dict, fold: int, smoke: bool) -> set[int]:
    folds_path = output_dir(cfg) / "inputs" / "fold_assignment.parquet"
    if folds_path.exists():
        folds = pd.read_parquet(folds_path)
        if "fold" in folds.columns:
            gcins = set(folds.loc[folds["fold"].astype(int) == int(fold), "GCIN"].astype(int))
        else:
            gcins = set(folds["GCIN"].astype(int))
    else:
        static = pd.read_parquet(cfg["paths"]["static_attributes"], columns=["GCIN"])
        gcins = set(static["GCIN"].astype(int))
    if smoke:
        gcins = set(sorted(gcins)[: int(cfg["smoke"].get("max_catchments", 128))])
    return gcins


def build_bounds(static: pd.DataFrame, lp_gamma: pd.DataFrame, cfg: dict) -> dict[int, np.ndarray]:
    bounds_cfg = cfg["physics"]["bounds"]
    lp_pad = float(bounds_cfg.get("lp_pad", 0.01))
    gamma_pad = float(bounds_cfg.get("gamma_pad", 0.01))
    lp_global = tuple(bounds_cfg.get("lp", [0.1, 1.0]))
    gamma_global = tuple(bounds_cfg.get("gamma", [0.1, 5.0]))
    alpha_global = tuple(bounds_cfg.get("alpha", [0.0, 1.0]))
    static = static.copy()
    lp_gamma = lp_gamma.copy()
    static["GCIN"] = static["GCIN"].astype(int)
    lp_gamma["GCIN"] = lp_gamma["GCIN"].astype(int)
    merged = static[["GCIN"]].merge(lp_gamma, on="GCIN", how="left")
    out: dict[int, np.ndarray] = {}
    for row in merged.itertuples(index=False):
        lp_low = getattr(row, "Lp_lower_CI", lp_global[0])
        lp_high = getattr(row, "Lp_upper_CI", lp_global[1])
        gamma_low = getattr(row, "gamma_low", gamma_global[0])
        gamma_high = getattr(row, "gamma_high", gamma_global[1])
        vals = np.array(
            [
                alpha_global[0],
                alpha_global[1],
                max(lp_global[0], float(lp_low) - lp_pad),
                min(lp_global[1], float(lp_high) + lp_pad),
                max(gamma_global[0], float(gamma_low) - gamma_pad),
                min(gamma_global[1], float(gamma_high) + gamma_pad),
            ],
            dtype="float32",
        )
        if vals[2] >= vals[3]:
            vals[2:4] = np.array(lp_global, dtype="float32")
        if vals[4] >= vals[5]:
            vals[4:6] = np.array(gamma_global, dtype="float32")
        out[int(getattr(row, "GCIN"))] = vals
    return out


def build_dataset(cfg: dict, frame: pd.DataFrame, gcins: set[int]) -> tuple[dict[int, BasinSeries], dict[str, Any]]:
    dynamic_cols = cfg["physics"]["dynamic_columns"]
    target_col = cfg["physics"]["target_column"]
    static_cols = cfg["data"]["static_columns"]
    warmup_days = int(cfg["physics"]["bufftime"])
    train_frac = float(cfg["physics"]["train_frac"])

    static = pd.read_parquet(cfg["paths"]["static_attributes"])
    static["GCIN"] = static["GCIN"].astype(int)
    static = static[static["GCIN"].astype(int).isin(gcins)].copy()
    lp_gamma = pd.read_parquet(cfg["paths"]["lp_gamma"])
    lp_gamma["GCIN"] = lp_gamma["GCIN"].astype(int)
    bounds = build_bounds(static, lp_gamma, cfg)
    snow_df = load_snow_mask(cfg)
    snow_threshold = float(cfg["recession"].get("snow_free_threshold", 25.0))
    cold_threshold = cfg["recession"].get("cold_temperature_filter_C")
    cold_threshold = None if cold_threshold is None else float(cold_threshold)

    train_dyn = []
    train_static = []
    raw_groups: dict[int, pd.DataFrame] = {}
    split_indices: dict[int, tuple[int, int]] = {}

    for gcin, group in frame.groupby("GCIN", sort=False, observed=True):
        group = group.sort_values("date").reset_index(drop=True)
        nt = len(group)
        if nt <= warmup_days + int(cfg["physics"]["rho"]):
            continue
        train_end = warmup_days + max(1, int(math.floor(train_frac * (nt - warmup_days))))
        train_end = min(train_end, nt - 1)
        raw_groups[int(gcin)] = group
        split_indices[int(gcin)] = (warmup_days, train_end)
        train_dyn.append(group.loc[warmup_days:train_end - 1, dynamic_cols].to_numpy("float32"))
        srow = static.loc[static["GCIN"].astype(int) == int(gcin), static_cols]
        if not srow.empty:
            train_static.append(srow.to_numpy("float32"))

    dyn_mean, dyn_std = finalize_stats(np.vstack(train_dyn))
    static_mean, static_std = finalize_stats(np.vstack(train_static))
    static_map = static.set_index("GCIN")[static_cols]

    basins: dict[int, BasinSeries] = {}
    for gcin, group in raw_groups.items():
        if gcin not in static_map.index:
            continue
        x_raw = group[dynamic_cols].to_numpy("float32")
        z_norm = ((x_raw - dyn_mean.reshape(1, -1)) / dyn_std.reshape(1, -1)).astype("float32")
        c_raw = static_map.loc[gcin].to_numpy("float32")
        c_norm = ((c_raw - static_mean) / static_std).astype("float32")
        y = group[[target_col]].to_numpy("float32")
        rec = detect_recession_paper(
            y[:, 0],
            min_len=int(cfg["recession"]["min_decline_days"]),
            drop_first=1 if bool(cfg["recession"].get("drop_first_decline_day", True)) else 0,
            decreasing_rate=bool(cfg["recession"].get("decreasing_rate", True)),
        ).astype("float32")
        rec = apply_snow_mask(rec.astype(bool), group["date"], gcin, snow_df, snow_threshold).astype("float32")
        rec = apply_cold_temperature_mask(rec.astype(bool), group["temperature_C"].to_numpy(), cold_threshold).astype("float32")
        start, _ = generate_state_reset_tensors(rec.astype(bool))
        train_start, train_end = split_indices[gcin]
        labels = np.array(["val"] * len(group), dtype=object)
        labels[:train_start] = "warmup"
        labels[train_start:train_end] = "train"
        basins[gcin] = BasinSeries(
            gcin=gcin,
            dates=group["date"].to_numpy(),
            x_raw=np.nan_to_num(x_raw, nan=0.0).astype("float32"),
            z_norm=np.nan_to_num(z_norm, nan=0.0).astype("float32"),
            c_norm=np.nan_to_num(c_norm, nan=0.0).astype("float32"),
            y_raw=y,
            rec_mask=rec,
            start_mask=start,
            bounds=bounds[gcin],
            split_labels=labels,
            train_start=train_start,
            train_end=train_end,
        )

    stats = {
        "dynamic_columns": dynamic_cols,
        "dynamic_mean": dyn_mean.tolist(),
        "dynamic_std": dyn_std.tolist(),
        "static_columns": static_cols,
        "static_mean": static_mean.tolist(),
        "static_std": static_std.tolist(),
    }
    return basins, stats


def valid_train_gcins(basins: dict[int, BasinSeries], rho: int, bufftime: int) -> list[int]:
    out = []
    for gcin, basin in basins.items():
        if basin.train_end - max(basin.train_start, bufftime) >= rho:
            out.append(gcin)
    return sorted(out)


def build_dynamic_batch(
    train_gcins: list[int],
    basins: dict[int, BasinSeries],
    batch_size: int,
    rho: int,
    bufftime: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    effective_batch = min(batch_size, len(train_gcins))
    selected = [train_gcins[i] for i in np.random.randint(0, len(train_gcins), size=effective_batch)]
    x_list, z_list, y_list, rec_list, start_list, bounds_list = [], [], [], [], [], []
    for gcin in selected:
        basin = basins[gcin]
        start_min = max(basin.train_start, bufftime)
        i_t = np.random.randint(start_min, basin.train_end - rho + 1)
        x = basin.x_raw[i_t - bufftime : i_t + rho, :]
        z_dyn = basin.z_norm[i_t - bufftime : i_t + rho, :]
        c_rep = np.repeat(basin.c_norm.reshape(1, -1), bufftime + rho, axis=0)
        z = np.concatenate([z_dyn, c_rep], axis=1)
        x_list.append(x)
        z_list.append(z)
        y_list.append(basin.y_raw[i_t : i_t + rho, :])
        rec_list.append(basin.rec_mask[i_t : i_t + rho])
        start_list.append(basin.start_mask[i_t : i_t + rho])
        bounds_list.append(basin.bounds)

    def time_first(items: list[np.ndarray], add_last_dim: bool = False) -> torch.Tensor:
        arr = np.stack(items, axis=0)
        if add_last_dim:
            arr = arr[..., None]
        arr = np.swapaxes(arr, 0, 1)
        return torch.from_numpy(arr).float().to(device)

    return (
        time_first(x_list),
        time_first(z_list),
        time_first(y_list),
        time_first(rec_list, add_last_dim=True),
        time_first(start_list, add_last_dim=True),
        torch.from_numpy(np.stack(bounds_list, axis=0)).float().to(device),
        selected,
    )


def compute_epoch_iterations(train_gcins: list[int], basins: dict[int, BasinSeries], batch_size: int, rho: int, bufftime: int) -> int:
    total = 0
    for gcin in train_gcins:
        basin = basins[gcin]
        total += max(0, basin.train_end - max(basin.train_start, bufftime))
    effective_batch = min(batch_size, len(train_gcins))
    p = min(max((effective_batch * rho) / float(max(total, 1)), 1e-6), 0.99)
    return max(1, int(np.ceil(np.log(0.01) / np.log(1.0 - p))))


def run() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]) + int(args.fold))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_group = args.run_label or ("smoke" if args.smoke else "physics_runs")
    run_dir = output_dir(cfg) / run_group / f"fold_{args.fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"run_dir={run_dir}")
    log(f"device={device}")

    years = list(range(int(cfg["data"]["start_year"]), int(cfg["data"]["end_year"]) + 1))
    if args.smoke:
        years = [int(y) for y in cfg["smoke"]["years"]]
    if args.year_start is not None:
        years = [y for y in years if y >= args.year_start]
    if args.year_end is not None:
        years = [y for y in years if y <= args.year_end]

    gcins = split_gcins(cfg, args.fold, args.smoke)
    frame = load_physics_frame(cfg, years, gcins)
    basins, stats = build_dataset(cfg, frame, gcins)
    rho = int(cfg["physics"]["rho"])
    bufftime = int(cfg["physics"]["bufftime"])
    train_gcins = valid_train_gcins(basins, rho, bufftime)
    if not train_gcins:
        raise RuntimeError("No basins have enough data for physics training")

    training_cfg = cfg["training"].copy()
    if args.smoke:
        training_cfg["epochs"] = int(cfg["smoke"]["epochs"])
        training_cfg["batch_size"] = int(cfg["smoke"]["batch_size"])
        training_cfg["iters_per_epoch"] = int(cfg["smoke"].get("iters_per_epoch", 2))

    batch_size = int(training_cfg["batch_size"])
    iters_per_epoch = int(training_cfg.get("iters_per_epoch") or compute_epoch_iterations(train_gcins, basins, batch_size, rho, bufftime))
    log(f"basins={len(basins)} train_basins={len(train_gcins)} years={years[0]}-{years[-1]} iters_per_epoch={iters_per_epoch}")

    input_dim = len(cfg["physics"]["dynamic_columns"]) + len(cfg["data"]["static_columns"])
    model = EpsilonStateResetModel(
        input_dim=input_dim,
        hidden_size=int(cfg["model"]["hidden_size"]),
        n_mul=int(cfg["model"].get("n_mul", 10)),
        dropout_rate=float(cfg["model"].get("dropout", 0.4)),
    ).to(device)
    criterion = PhysicsInformedLoss(
        lambda_path=float(training_cfg["lambda_path"]),
        lambda_rhs=float(training_cfg["lambda_rhs"]),
        lambda_smooth=float(training_cfg["lambda_smooth"]),
        lambda_q0=float(training_cfg["lambda_q0"]),
        delta=float(training_cfg.get("huber_delta", 0.5)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["learning_rate"]))

    metrics = []
    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        model.train()
        totals = {"total": 0.0, "l_path": 0.0, "l_rhs": 0.0, "l_smooth": 0.0, "l_q0": 0.0}
        epoch_t0 = time.time()
        for _ in range(iters_per_epoch):
            x_batch, z_batch, y_batch, rec_mask, start_mask, bounds, _ = build_dynamic_batch(
                train_gcins, basins, batch_size, rho, bufftime, device
            )
            pet_seq = x_batch[:, :, 2:3]
            sm_seq = x_batch[:, :, 3:4]
            optimizer.zero_grad(set_to_none=True)
            model_out = model(z_batch, pet_seq, sm_seq, rec_mask, start_mask, bounds, bufftime=bufftime)
            loss_dict = criterion(model_out, y_batch, rec_mask, start_mask)
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            for key in totals:
                totals[key] += float(loss_dict[key].detach().cpu())
        row = {key: value / iters_per_epoch for key, value in totals.items()}
        row["epoch"] = epoch
        row["epoch_seconds"] = time.time() - epoch_t0
        metrics.append(row)
        log(str(row))
        if epoch % int(training_cfg.get("save_every", 10)) == 0 or epoch == int(training_cfg["epochs"]):
            torch.save(model.state_dict(), run_dir / f"epsilon_physics_epoch_{epoch}.pt")

    pd.DataFrame(metrics).to_csv(run_dir / "metrics.csv", index=False)
    torch.save(model.state_dict(), run_dir / "best_model.pt")
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "fold": args.fold,
                "smoke": args.smoke,
                "device": str(device),
                "years": years,
                "n_basins": len(basins),
                "n_train_basins": len(train_gcins),
                "rho": rho,
                "bufftime": bufftime,
                "stats": stats,
                "method": "physics-informed epsilon-core state-reset LSTM",
                "reference": "arabayati/LSTM-epsilon",
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    run()
