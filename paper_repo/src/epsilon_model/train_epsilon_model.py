from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--iters-per-epoch", type=int, default=None)
    parser.add_argument("--validation-batches", type=int, default=None)
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


def split_gcins_by_role(cfg: dict, fold: int, smoke: bool) -> dict[str, set[int]]:
    folds_path = output_dir(cfg) / "inputs" / "fold_assignment.parquet"
    if not folds_path.exists():
        raise FileNotFoundError(f"Missing crossfit assignment: {folds_path}")
    folds = pd.read_parquet(folds_path)
    role_col = f"role_fold_{int(fold)}"
    if role_col not in folds.columns:
        raise ValueError(f"Crossfit assignment is missing {role_col}; rerun prepare_experiment_inputs.py")

    roles = {
        role: set(folds.loc[folds[role_col] == role, "GCIN"].astype(int))
        for role in ("train", "validation", "test")
    }
    if any(not values for values in roles.values()):
        raise ValueError(f"Fold {fold} has an empty train, validation, or test role")
    return roles


def split_gcins(cfg: dict, fold: int, smoke: bool, role: str = "test") -> set[int]:
    return split_gcins_by_role(cfg, fold, smoke)[role]


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


def build_dataset(
    cfg: dict,
    frame: pd.DataFrame,
    gcins: set[int],
    stats_gcins: set[int] | None = None,
    normalization_stats: dict[str, Any] | None = None,
) -> tuple[dict[int, BasinSeries], dict[str, Any]]:
    dynamic_cols = cfg["physics"]["dynamic_columns"]
    target_col = cfg["physics"]["target_column"]
    static_cols = cfg["data"]["static_columns"]
    warmup_days = int(cfg["physics"]["bufftime"])
    stats_gcins = gcins if stats_gcins is None else stats_gcins

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

    for gcin, group in frame.groupby("GCIN", sort=False, observed=True):
        group = group.sort_values("date").reset_index(drop=True)
        nt = len(group)
        if nt <= warmup_days + int(cfg["physics"]["rho"]):
            continue
        raw_groups[int(gcin)] = group
        if int(gcin) in stats_gcins and normalization_stats is None:
            train_dyn.append(group.loc[warmup_days:, dynamic_cols].to_numpy("float32"))
            srow = static.loc[static["GCIN"].astype(int) == int(gcin), static_cols]
            if not srow.empty:
                train_static.append(srow.to_numpy("float32"))

    if normalization_stats is None:
        if not train_dyn or not train_static:
            raise RuntimeError("No training basins were available for normalization statistics")
        dyn_mean, dyn_std = finalize_stats(np.vstack(train_dyn))
        static_mean, static_std = finalize_stats(np.vstack(train_static))
    else:
        dyn_mean = np.asarray(normalization_stats["dynamic_mean"], dtype="float32")
        dyn_std = np.asarray(normalization_stats["dynamic_std"], dtype="float32")
        static_mean = np.asarray(normalization_stats["static_mean"], dtype="float32")
        static_std = np.asarray(normalization_stats["static_std"], dtype="float32")
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
        train_start, train_end = warmup_days, len(group)
        labels = np.array(["eval"] * len(group), dtype=object)
        labels[:train_start] = "warmup"
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
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    effective_batch = min(batch_size, len(train_gcins))
    if rng is None:
        selected_idx = np.random.randint(0, len(train_gcins), size=effective_batch)
    else:
        selected_idx = rng.integers(0, len(train_gcins), size=effective_batch)
    selected = [train_gcins[i] for i in selected_idx]
    x_list, z_list, y_list, rec_list, start_list, bounds_list = [], [], [], [], [], []
    for gcin in selected:
        basin = basins[gcin]
        start_min = max(basin.train_start, bufftime)
        if rng is None:
            i_t = np.random.randint(start_min, basin.train_end - rho + 1)
        else:
            i_t = int(rng.integers(start_min, basin.train_end - rho + 1))
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


def evaluate_validation(
    model: EpsilonStateResetModel,
    criterion: PhysicsInformedLoss,
    validation_gcins: list[int],
    basins: dict[int, BasinSeries],
    batch_size: int,
    rho: int,
    bufftime: int,
    device: torch.device,
    n_batches: int,
    seed: int,
) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "l_path": 0.0, "l_rhs": 0.0, "l_smooth": 0.0, "l_q0": 0.0}
    basin_skill: dict[int, dict[str, float]] = {}
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for _ in range(n_batches):
            x_batch, z_batch, y_batch, rec_mask, start_mask, bounds, selected = build_dynamic_batch(
                validation_gcins,
                basins,
                batch_size,
                rho,
                bufftime,
                device,
                rng=rng,
            )
            pet_seq = x_batch[:, :, 2:3]
            sm_seq = x_batch[:, :, 3:4]
            model_out = model(z_batch, pet_seq, sm_seq, rec_mask, start_mask, bounds, bufftime=bufftime)
            loss_dict = criterion(model_out, y_batch, rec_mask, start_mask)
            for key in totals:
                totals[key] += float(loss_dict[key].detach().cpu())

            q_hat = model_out["q_hat"]
            valid = (rec_mask > 0.5) & torch.isfinite(y_batch) & torch.isfinite(q_hat)
            obs = torch.where(valid, y_batch, torch.zeros_like(y_batch))
            squared_error = torch.where(valid, (q_hat - y_batch) ** 2, torch.zeros_like(y_batch))
            counts = valid.sum(dim=0).squeeze(-1).detach().cpu().numpy()
            obs_sum = obs.sum(dim=0).squeeze(-1).detach().cpu().numpy()
            obs_sum_sq = (obs**2).sum(dim=0).squeeze(-1).detach().cpu().numpy()
            error_sum_sq = squared_error.sum(dim=0).squeeze(-1).detach().cpu().numpy()
            for i, gcin in enumerate(selected):
                stats = basin_skill.setdefault(
                    int(gcin),
                    {"count": 0.0, "obs_sum": 0.0, "obs_sum_sq": 0.0, "error_sum_sq": 0.0},
                )
                stats["count"] += float(counts[i])
                stats["obs_sum"] += float(obs_sum[i])
                stats["obs_sum_sq"] += float(obs_sum_sq[i])
                stats["error_sum_sq"] += float(error_sum_sq[i])

    validation = {key: value / n_batches for key, value in totals.items()}
    basin_nse = []
    pooled = {"count": 0.0, "obs_sum": 0.0, "obs_sum_sq": 0.0, "error_sum_sq": 0.0}
    for stats in basin_skill.values():
        for key in pooled:
            pooled[key] += stats[key]
        if stats["count"] < 2:
            continue
        denominator = stats["obs_sum_sq"] - (stats["obs_sum"] ** 2) / stats["count"]
        if denominator > EPS:
            basin_nse.append(1.0 - stats["error_sum_sq"] / denominator)

    pooled_denominator = pooled["obs_sum_sq"] - (pooled["obs_sum"] ** 2) / max(pooled["count"], 1.0)
    validation["median_nse"] = float(np.median(basin_nse)) if basin_nse else float("nan")
    validation["mean_nse"] = float(np.mean(basin_nse)) if basin_nse else float("nan")
    validation["pooled_nse"] = (
        float(1.0 - pooled["error_sum_sq"] / pooled_denominator)
        if pooled_denominator > EPS
        else float("nan")
    )
    validation["nse_basins"] = float(len(basin_nse))
    return validation


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

    roles = split_gcins_by_role(cfg, args.fold, args.smoke)
    fitting_gcins = roles["train"] | roles["validation"]
    frame = load_physics_frame(cfg, years, fitting_gcins)
    basins, stats = build_dataset(cfg, frame, fitting_gcins, stats_gcins=roles["train"])
    rho = int(cfg["physics"]["rho"])
    bufftime = int(cfg["physics"]["bufftime"])
    train_gcins = [gcin for gcin in valid_train_gcins(basins, rho, bufftime) if gcin in roles["train"]]
    validation_gcins = [gcin for gcin in valid_train_gcins(basins, rho, bufftime) if gcin in roles["validation"]]
    if args.smoke:
        max_train = int(cfg["smoke"].get("max_catchments", 128))
        train_gcins = train_gcins[:max_train]
        validation_gcins = validation_gcins[: max(1, max_train // 4)]
    if not train_gcins or not validation_gcins:
        raise RuntimeError("No basins have enough data for physics training or validation")

    training_cfg = cfg["training"].copy()
    if args.smoke:
        training_cfg["epochs"] = int(cfg["smoke"]["epochs"])
        training_cfg["batch_size"] = int(cfg["smoke"]["batch_size"])
        training_cfg["iters_per_epoch"] = int(cfg["smoke"].get("iters_per_epoch", 2))
        training_cfg["validation_batches"] = int(cfg["smoke"].get("validation_batches", 1))
        training_cfg["early_stopping_min_epochs"] = int(training_cfg["epochs"])
        training_cfg["early_stopping_patience"] = int(training_cfg["epochs"])
    for argument, key in (
        (args.epochs, "epochs"),
        (args.batch_size, "batch_size"),
        (args.iters_per_epoch, "iters_per_epoch"),
        (args.validation_batches, "validation_batches"),
    ):
        if argument is not None:
            training_cfg[key] = int(argument)

    batch_size = int(training_cfg["batch_size"])
    iters_per_epoch = int(training_cfg.get("iters_per_epoch") or compute_epoch_iterations(train_gcins, basins, batch_size, rho, bufftime))
    log(
        f"basins={len(basins)} train_basins={len(train_gcins)} "
        f"validation_basins={len(validation_gcins)} test_basins={len(roles['test'])} "
        f"years={years[0]}-{years[-1]} iters_per_epoch={iters_per_epoch}"
    )

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
    run_started_at = time.time()
    best_validation_nse = -float("inf")
    best_validation_total = float("nan")
    best_epoch = 0
    epochs_without_improvement = 0
    validation_batches = int(training_cfg.get("validation_batches", 8))
    min_epochs = int(training_cfg.get("early_stopping_min_epochs", 20))
    patience = int(training_cfg.get("early_stopping_patience", 8))
    min_delta = float(training_cfg.get("early_stopping_min_delta", 0.0))
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
        train_average = {key: value / iters_per_epoch for key, value in totals.items()}
        validation_average = evaluate_validation(
            model,
            criterion,
            validation_gcins,
            basins,
            batch_size,
            rho,
            bufftime,
            device,
            validation_batches,
            seed=int(cfg["seed"]) + 10_000 + int(args.fold),
        )
        row = {f"train_{key}": value for key, value in train_average.items()}
        row.update({f"validation_{key}": value for key, value in validation_average.items()})
        row["epoch"] = epoch
        row["epoch_seconds"] = time.time() - epoch_t0
        metrics.append(row)
        pd.DataFrame(metrics).to_csv(run_dir / "metrics.csv", index=False)
        log(
            f"epoch={epoch} train_loss={train_average['total']:.6f} "
            f"validation_loss={validation_average['total']:.6f} "
            f"validation_median_nse={validation_average['median_nse']:.4f} "
            f"validation_pooled_nse={validation_average['pooled_nse']:.4f}"
        )

        validation_nse = validation_average["median_nse"]
        if best_epoch == 0 or (np.isfinite(validation_nse) and validation_nse > best_validation_nse + min_delta):
            best_validation_nse = validation_nse
            best_validation_total = validation_average["total"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1
        if epoch >= min_epochs and epochs_without_improvement >= patience:
            log(
                f"early stopping at epoch={epoch}; best_epoch={best_epoch} "
                f"best_validation_median_nse={best_validation_nse:.4f}"
            )
            break

    pd.DataFrame(metrics).to_csv(run_dir / "metrics.csv", index=False)
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "fold": args.fold,
                "smoke": args.smoke,
                "device": str(device),
                "years": years,
                "n_basins": len(basins),
                "n_train_basins": len(train_gcins),
                "n_validation_basins": len(validation_gcins),
                "n_test_basins": len(roles["test"]),
                "best_epoch": best_epoch,
                "best_validation_median_nse": best_validation_nse,
                "best_validation_total": best_validation_total,
                "elapsed_seconds": time.time() - run_started_at,
                "rho": rho,
                "bufftime": bufftime,
                "stats": stats,
                "training": training_cfg,
                "seed": int(cfg["seed"]) + int(args.fold),
                "role_column": f"role_fold_{int(args.fold)}",
                "normalization_scope": "train basins only",
                "validation_sampling": "fixed deterministic windows across epochs",
                "checkpoint_selection": "maximum median catchment NSE on validation recession days",
                "method": "physics-informed epsilon-core state-reset LSTM",
                "reference": "arabayati/LSTM-epsilon",
                "split_protocol": "five-fold basin cross-fitting with train-only normalization",
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    run()
