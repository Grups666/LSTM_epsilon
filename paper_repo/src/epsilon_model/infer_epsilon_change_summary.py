from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import load_config, output_dir
from physics_model import EpsilonStateResetModel, EPS
from train_epsilon_model import build_dataset, load_physics_frame, log, split_gcins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--run-label", type=str, default="physics_runs")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-catchments", type=int, default=None)
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


def infer_basin(
    model: EpsilonStateResetModel,
    basin,
    cfg: dict,
    device: torch.device,
) -> pd.DataFrame:
    bufftime = int(cfg["physics"]["bufftime"])
    if len(basin.dates) <= bufftime:
        return pd.DataFrame()
    z_seq = np.concatenate(
        [basin.z_norm, np.repeat(basin.c_norm.reshape(1, -1), len(basin.dates), axis=0)],
        axis=1,
    )
    z_batch = torch.from_numpy(z_seq).float().unsqueeze(1).to(device)
    pet_batch = torch.from_numpy(basin.x_raw[:, 2:3]).float().unsqueeze(1).to(device)
    sm_batch = torch.from_numpy(basin.x_raw[:, 3:4]).float().unsqueeze(1).to(device)
    rec_mask = torch.from_numpy(basin.rec_mask).float().unsqueeze(1).unsqueeze(-1).to(device)
    start_mask = torch.from_numpy(basin.start_mask).float().unsqueeze(1).unsqueeze(-1).to(device)
    bounds = torch.from_numpy(basin.bounds).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(z_batch, pet_batch, sm_batch, rec_mask, start_mask, bounds, bufftime=bufftime)

    rec_eval = basin.rec_mask[bufftime:] > 0.5
    if not rec_eval.any():
        return pd.DataFrame()
    dates = pd.to_datetime(basin.dates[bufftime:])[rec_eval]
    qobs = basin.y_raw[bufftime:, 0][rec_eval]
    q_hat = out["q_hat"].cpu().numpy().reshape(-1)[rec_eval]
    q_comp = out["q_components"].cpu().numpy().reshape(len(rec_eval), -1)[rec_eval]
    eps = out["eps"].cpu().numpy().reshape(len(rec_eval), -1)[rec_eval]
    aet = out["aet"].cpu().numpy().reshape(len(rec_eval), -1)[rec_eval]
    alpha = out["alpha"].cpu().numpy().reshape(-1)

    eps_mean = np.mean(eps, axis=1)
    eps_eff = np.mean(eps * (q_comp**2), axis=1) / (q_hat**2 + EPS)
    aet_mean = np.mean(aet, axis=1)
    alpha_eff = np.mean(eps * alpha.reshape(1, -1) * aet * q_comp, axis=1) / (eps_eff * aet_mean * q_hat + EPS)
    return pd.DataFrame(
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = output_dir(cfg) / args.run_label / f"fold_{args.fold}"
    metadata_path = run_dir / "run_metadata.json"
    model_path = run_dir / "best_model.pt"
    if not metadata_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Missing model outputs in {run_dir}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    years = [int(y) for y in metadata["years"]]
    gcins = split_gcins(cfg, args.fold, bool(metadata.get("smoke", args.smoke)))
    if args.max_catchments is not None:
        gcins = set(sorted(gcins)[: args.max_catchments])
    frame = load_physics_frame(cfg, years, gcins)
    basins, _ = build_dataset(cfg, frame, gcins)

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
    for idx, basin in enumerate(basins.values(), start=1):
        if idx % 100 == 0:
            log(f"inference basin {idx}/{len(basins)}")
        pred = infer_basin(model, basin, cfg, device)
        if pred.empty:
            continue
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
    log(f"wrote {summary_out} rows={len(summary)}")
    if len(summary):
        log(f"delta_epsilon_mean_global={summary['delta_epsilon_mean'].mean()}")


if __name__ == "__main__":
    main()
