from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config import load_config, output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-label", type=str, required=True)
    parser.add_argument("--figures-dir", type=Path, default=Path("_private/results/paper_figures_pure_gcin"))
    parser.add_argument("--summary-md", type=Path, default=Path("paper_repo/docs/SUMMARY.md"))
    parser.add_argument(
        "--github-pages-out",
        type=Path,
        default=Path("_submission/LSTM_epsilon_publish/public/modules/epsilon-change/data/epsilon-catchment-distributions.json"),
    )
    parser.add_argument("--qobs-coverage", type=Path, default=None)
    parser.add_argument("--skip-pages", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def require_completed_training(cfg: dict, run_label: str) -> Path:
    run_root = output_dir(cfg) / run_label
    missing = []
    for fold in range(int(cfg["splits"]["n_folds"])):
        fold_dir = run_root / f"fold_{fold}"
        for name in ("final_model.pt", "metrics.csv", "run_metadata.json"):
            path = fold_dir / name
            if not path.exists():
                missing.append(path)
    if missing:
        text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Training is not complete; missing:\n{text}")
    return run_root


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_root = require_completed_training(cfg, args.run_label)
    scripts = Path(__file__).resolve().parent
    python = sys.executable

    for fold in range(int(cfg["splits"]["n_folds"])):
        fold_dir = run_root / f"fold_{fold}"
        summary = fold_dir / "heldout_epsilon_change_summary.parquet"
        simulation = fold_dir / "recession_day_simulations.parquet"
        if summary.exists() and simulation.exists():
            print(f"skip fold {fold} inference; outputs already exist", flush=True)
            continue
        run(
            [
                python,
                str(scripts / "infer_epsilon_change_summary.py"),
                "--config",
                str(args.config),
                "--fold",
                str(fold),
                "--run-label",
                args.run_label,
            ]
        )

    run([python, str(scripts / "aggregate_crossfit_results.py"), "--config", str(args.config), "--run-label", args.run_label])
    run(
        [
            python,
            str(scripts / "audit_production_run.py"),
            "--config",
            str(args.config),
            "--run-label",
            args.run_label,
            "--out",
            str(run_root / "production_audit.csv"),
        ]
    )
    run(
        [
            python,
            str(scripts / "make_paper_figures.py"),
            "--config",
            str(args.config),
            "--run-label",
            args.run_label,
            "--out-dir",
            str(args.figures_dir),
        ]
    )

    if not args.skip_summary:
        run(
            [
                python,
                str(scripts / "update_summary_from_results.py"),
                "--config",
                str(args.config),
                "--figures-dir",
                str(args.figures_dir),
                "--summary-md",
                str(args.summary_md),
                "--run-label",
                args.run_label,
            ]
        )

    if not args.skip_pages:
        qobs_coverage = args.qobs_coverage
        if qobs_coverage is None:
            qobs_coverage = Path(cfg["paths"]["output_dir"]) / "inputs" / "qobs_inventory.parquet"
        run(
            [
                python,
                str(scripts / "export_github_pages_data.py"),
                "--config",
                str(args.config),
                "--run-label",
                args.run_label,
                "--qobs-coverage",
                str(qobs_coverage),
                "--out",
                str(args.github_pages_out),
            ]
        )

    print(f"postprocess complete for {run_root}", flush=True)


if __name__ == "__main__":
    main()
