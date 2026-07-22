from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import load_config, output_dir
from physics_model import EpsilonStateResetModel, EPS
from temporal_split import test_years
from train_epsilon_model import build_dataset, load_physics_frame, log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--run-label", type=str, default="temporal_crossfit_1990")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-catchments", type=int, default=None)
    parser.add_argument("--inference-batch-size", type=int, default=256)
    return parser.parse_args()


def summarize(pred: pd.DataFrame, pre_window: list[str], post_window: list[str]) -> dict[str, float]:
    date = pd.to_datetime(pred["date"])
    pre_start, pre_end = pd.to_datetime(pre_window)
    post_start, post_end = pd.to_datetime(post_window)
    pre = pred[(date >= pre_start) & (date <= pre_end)]
    post = pred[(date >= post_start) & (date <= post_end)]
    return {
        "pre_epsilon_mean": float(pre["epsilon_effective"].mean()) if len(pre) else np.nan,
        "post_epsilon_mean": float(post["epsilon_effective"].mean()) if len(post) else np.nan,
        "delta_epsilon_mean": float(post["epsilon_effective"].mean() - pre["epsilon_effective"].mean()) if len(pre) and len(post) else np.nan,
        "pre_epsilon_arithmetic_mean": float(pre["epsilon_mean"].mean()) if len(pre) else np.nan,
        "post_epsilon_arithmetic_mean": float(post["epsilon_mean"].mean()) if len(post) else np.nan,
        "pre_qobs_valid_days": int(pre["observed_Q_mmd"].notna().sum()) if len(pre) else 0,
        "post_qobs_valid_days": int(post["observed_Q_mmd"].notna().sum()) if len(post) else 0,
        "pre_n_recession_days_predicted": int(len(pre)),
        "post_n_recession_days_predicted": int(len(post)),
    }


def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(obs) & np.isfinite(pred)
    if valid.sum() < 2:
        return np.nan
    obs = obs[valid]
    pred = pred[valid]
    denominator = np.sum((obs - obs.mean()) ** 2)
    return float(1.0 - np.sum((pred - obs) ** 2) / denominator) if denominator > 0 else np.nan


def write_fold_skill_report(sim: pd.DataFrame, cfg: dict, run_dir: Path) -> pd.DataFrame:
    work = sim.copy()
    work["date"] = pd.to_datetime(work["date"])
    pre_start, pre_end = pd.to_datetime(cfg["data"]["pre_window"])
    post_start, post_end = pd.to_datetime(cfg["data"]["post_window"])
    masks = {
        "all": np.ones(len(work), dtype=bool),
        "pre": work["date"].between(pre_start, pre_end).to_numpy(),
        "post": work["date"].between(post_start, post_end).to_numpy(),
    }
    rows = []
    for period, mask in masks.items():
        part = work.loc[mask]
        basin_scores = np.asarray(
            [
                nse(group["observed_Q_mmd"].to_numpy(float), group["simulated_Q_mmd"].to_numpy(float))
                for _, group in part.groupby("GCIN", observed=True)
            ],
            dtype=float,
        )
        basin_scores = basin_scores[np.isfinite(basin_scores)]
        pooled = nse(part["observed_Q_mmd"].to_numpy(float), part["simulated_Q_mmd"].to_numpy(float))
        rows.append(
            {
                "period": period,
                "n_catchments": int(len(basin_scores)),
                "n_recession_days": int(len(part)),
                "median_catchment_nse": float(np.median(basin_scores)) if len(basin_scores) else np.nan,
                "mean_catchment_nse": float(np.mean(basin_scores)) if len(basin_scores) else np.nan,
                "p10_catchment_nse": float(np.quantile(basin_scores, 0.10)) if len(basin_scores) else np.nan,
                "p90_catchment_nse": float(np.quantile(basin_scores, 0.90)) if len(basin_scores) else np.nan,
                "pooled_nse": pooled,
            }
        )
    report = pd.DataFrame(rows)
    report.to_csv(run_dir / "heldout_skill_summary.csv", index=False)
    return report


def infer_basin_batch(
    model: EpsilonStateResetModel,
    window_batch: list[tuple[object, int, int]],
    cfg: dict,
    device: torch.device,
) -> list[pd.DataFrame]:
    if not window_batch:
        return []
    bufftime = int(cfg["physics"]["bufftime"])
    target_lengths = {target_end - target_start for _, target_start, target_end in window_batch}
    if len(target_lengths) != 1:
        raise ValueError("Batched inference requires equal target-window lengths")
    target_length = target_lengths.pop()
    sequence_length = bufftime + target_length
    for basin, target_start, target_end in window_batch:
        if target_start < bufftime or target_end > len(basin.dates) or target_end <= target_start:
            raise ValueError(f"Invalid inference window for GCIN {basin.gcin}: {target_start}:{target_end}")

    z = np.stack(
        [
            np.concatenate(
                [
                    basin.z_norm[target_start - bufftime : target_end],
                    np.repeat(basin.c_norm.reshape(1, -1), sequence_length, axis=0),
                ],
                axis=1,
            )
            for basin, target_start, target_end in window_batch
        ],
        axis=1,
    )
    pet = np.stack(
        [basin.x_raw[target_start - bufftime : target_end, 2:3] for basin, target_start, target_end in window_batch],
        axis=1,
    )
    sm = np.stack(
        [basin.x_raw[target_start - bufftime : target_end, 3:4] for basin, target_start, target_end in window_batch],
        axis=1,
    )
    rec = np.stack(
        [basin.rec_mask[target_start:target_end, None] for basin, target_start, target_end in window_batch],
        axis=1,
    )
    start = np.stack(
        [basin.start_mask[target_start:target_end, None] for basin, target_start, target_end in window_batch],
        axis=1,
    )
    bounds = np.stack([basin.bounds for basin, _, _ in window_batch], axis=0)

    with torch.no_grad():
        out = model(
            torch.from_numpy(z).float().to(device),
            torch.from_numpy(pet).float().to(device),
            torch.from_numpy(sm).float().to(device),
            torch.from_numpy(rec).float().to(device),
            torch.from_numpy(start).float().to(device),
            torch.from_numpy(bounds).float().to(device),
            bufftime=bufftime,
        )
    q_hat_all = out["q_hat"].cpu().numpy()
    q_comp_all = out["q_components"].cpu().numpy()
    eps_all = out["eps"].cpu().numpy()
    aet_all = out["aet"].cpu().numpy()
    alpha_all = out["alpha"].cpu().numpy()

    frames = []
    for index, (basin, target_start, target_end) in enumerate(window_batch):
        rec_eval = basin.rec_mask[target_start:target_end] > 0.5
        if not rec_eval.any():
            frames.append(pd.DataFrame())
            continue
        dates = pd.to_datetime(basin.dates[target_start:target_end])[rec_eval]
        qobs = basin.y_raw[target_start:target_end, 0][rec_eval]
        q_hat = q_hat_all[:, index, 0][rec_eval]
        q_comp = q_comp_all[:, index, :][rec_eval]
        eps = eps_all[:, index, :][rec_eval]
        aet = aet_all[:, index, :][rec_eval]
        alpha = alpha_all[index]
        eps_mean = np.mean(eps, axis=1)
        eps_eff = np.mean(eps * (q_comp**2), axis=1) / (q_hat**2 + EPS)
        aet_mean = np.mean(aet, axis=1)
        alpha_eff = np.mean(eps * alpha.reshape(1, -1) * aet * q_comp, axis=1) / (
            eps_eff * aet_mean * q_hat + EPS
        )
        frames.append(
            pd.DataFrame(
                {
                    "GCIN": basin.gcin,
                    "date": dates,
                    "observed_Q_mmd": qobs,
                    "simulated_Q_mmd": q_hat.astype("float32"),
                    "epsilon_mean": eps_mean.astype("float32"),
                    "epsilon_effective": eps_eff.astype("float32"),
                    "simulated_AET_mm": aet_mean.astype("float32"),
                    "alpha_effective": alpha_eff.astype("float32"),
                }
            )
        )
    return frames


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = output_dir(cfg) / args.run_label / f"fold_{args.fold}"
    metadata_path = run_dir / "run_metadata.json"
    model_path = run_dir / "final_model.pt"
    if not metadata_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Missing model outputs in {run_dir}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    years = [int(y) for y in metadata["years"]]
    gcins = set(pd.read_parquet(cfg["paths"]["static_attributes"], columns=["GCIN"])["GCIN"].astype(int))
    if bool(metadata.get("smoke", args.smoke)):
        gcins = set(sorted(gcins)[: int(cfg["smoke"].get("max_catchments", 32))])
    frame = load_physics_frame(cfg, years, gcins)
    basins, _ = build_dataset(
        cfg,
        frame,
        gcins,
        training_years=set(int(y) for y in metadata["train_years"]),
        normalization_stats=metadata["stats"],
    )
    max_catchments = args.max_catchments
    if max_catchments is None and bool(metadata.get("smoke", args.smoke)):
        max_catchments = max(1, int(cfg["smoke"].get("max_catchments", 128)) // 4)
    if max_catchments is not None:
        keep = set(sorted(basins)[:max_catchments])
        basins = {gcin: basin for gcin, basin in basins.items() if gcin in keep}

    input_dim = len(cfg["physics"]["dynamic_columns"]) + len(cfg["data"]["static_columns"])
    model = EpsilonStateResetModel(
        input_dim=input_dim,
        hidden_size=int(cfg["model"]["hidden_size"]),
        n_mul=int(cfg["model"].get("n_mul", 10)),
        dropout_rate=float(cfg["model"].get("dropout", 0.4)),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    summaries = []
    sim_pieces = []
    tasks_by_length: dict[int, list[tuple[object, int, int]]] = {}
    basin_predictions: dict[int, list[pd.DataFrame]] = {basin.gcin: [] for basin in basins.values()}
    rho = int(cfg["physics"]["rho"])
    bufftime = int(cfg["physics"]["bufftime"])
    for basin in basins.values():
        dates = pd.to_datetime(basin.dates)
        eligible = (dates.year.isin(test_years(cfg, args.fold))) & (np.arange(len(dates)) >= bufftime)
        eligible_indices = np.flatnonzero(eligible)
        if not len(eligible_indices):
            continue
        date_days = dates.to_numpy(dtype="datetime64[D]").astype("int64")
        index_breaks = np.diff(eligible_indices) > 1
        date_breaks = np.diff(date_days[eligible_indices]) > 1
        breaks = np.flatnonzero(index_breaks | date_breaks)
        run_starts = np.concatenate([[0], breaks + 1])
        run_ends = np.concatenate([breaks + 1, [len(eligible_indices)]])
        for run_start, run_end in zip(run_starts, run_ends):
            first = int(eligible_indices[run_start])
            stop = int(eligible_indices[run_end - 1]) + 1
            for target_start in range(first, stop, rho):
                target_end = min(target_start + rho, stop)
                target_length = target_end - target_start
                tasks_by_length.setdefault(target_length, []).append((basin, target_start, target_end))

    completed = 0
    total_windows = sum(len(tasks) for tasks in tasks_by_length.values())
    batch_size = max(1, int(args.inference_batch_size))
    for tasks in tasks_by_length.values():
        for offset in range(0, len(tasks), batch_size):
            window_batch = tasks[offset : offset + batch_size]
            predictions = infer_basin_batch(model, window_batch, cfg, device)
            for (basin, _, _), pred in zip(window_batch, predictions):
                if not pred.empty:
                    basin_predictions[basin.gcin].append(pred)
            completed += len(window_batch)
            if completed % 1000 < len(window_batch) or completed == total_windows:
                log(f"inference windows {completed}/{total_windows}")

    for basin in basins.values():
        pieces = basin_predictions[basin.gcin]
        if not pieces:
            continue
        pred = pd.concat(pieces, ignore_index=True).sort_values("date")
        if pred.duplicated("date").any():
            raise RuntimeError(f"Duplicate inference date for GCIN {basin.gcin}")
        pred["fold"] = args.fold
        sim_pieces.append(pred)
        row = summarize(pred, cfg["data"]["pre_window"], cfg["data"]["post_window"])
        row["GCIN"] = basin.gcin
        summaries.append(row)

    summary = pd.DataFrame(summaries)
    summary["fold"] = args.fold
    summary_out = run_dir / "heldout_epsilon_change_summary.parquet"
    summary.to_parquet(summary_out, index=False)
    summary.to_csv(run_dir / "heldout_epsilon_change_summary.csv", index=False)
    if sim_pieces:
        sim = pd.concat(sim_pieces, ignore_index=True)
        sim.to_parquet(run_dir / "recession_day_simulations.parquet", index=False)
        skill = write_fold_skill_report(sim, cfg, run_dir)
        for row in skill.itertuples(index=False):
            log(
                f"heldout_nse period={row.period} n_basins={row.n_catchments} "
                f"median={row.median_catchment_nse:.3f} mean={row.mean_catchment_nse:.3f} "
                f"pooled={row.pooled_nse:.3f}"
            )
    log(f"wrote {summary_out} rows={len(summary)}")
    if len(summary):
        log(f"delta_epsilon_mean_global={summary['delta_epsilon_mean'].mean()}")


if __name__ == "__main__":
    main()
