"""Multi-seed training for one ablation configuration.

Usage:
    python train_multiseed.py \
        --cfg ../ablation/yolo11s_M3_R1R2R3.yaml \
        --data ../../dataset.yaml \
        --weights yolo11s.pt \
        --epochs 200 --imgsz 640 --batch 16 --device 0,1,2,3 \
        --seeds 0,1,2,3,4 \
        --project runs/ablation \
        --name M3

Saves results to <project>/<name>_seed<S>/ and a summary CSV at
<project>/<name>_summary.csv with mean ± std for the headline metrics.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def set_seed(seed: int) -> None:
    """Seed PyTorch / NumPy / Python / CUDA RNGs and force deterministic CuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True, help="Path to model YAML (M0/M1/M2/M3).")
    p.add_argument("--data", required=True, help="Dataset YAML.")
    p.add_argument("--weights", default="yolo11s.pt",
                   help="Pretrained weights to transfer-learn from.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0", help="CUDA device(s), e.g. '0,1,2,3'.")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seeds", default="0,1,2,3,4",
                   help="Comma-separated seed list.")
    p.add_argument("--project", default="runs/ablation")
    p.add_argument("--name", required=True, help="Run-name prefix.")
    p.add_argument("--force", action="store_true",
                   help="Retrain even if a previous run with the same name finished.")
    return p.parse_args()


def read_best_metrics_from_results_csv(run_dir: Path) -> dict:
    results_csv = run_dir / "results.csv"

    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found: {results_csv}")

    with results_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = []

        for row in reader:
            clean_row = {k.strip(): v for k, v in row.items()}
            rows.append(clean_row)

    if not rows:
        raise RuntimeError(f"results.csv is empty: {results_csv}")

    def to_float(row: dict, key: str) -> float:
        try:
            return float(row.get(key, 0.0))
        except Exception:
            return 0.0

    best_row = max(rows, key=lambda r: to_float(r, "metrics/mAP50-95(B)"))

    return {
        "precision": to_float(best_row, "metrics/precision(B)"),
        "recall": to_float(best_row, "metrics/recall(B)"),
        "mAP50": to_float(best_row, "metrics/mAP50(B)"),
        "mAP50-95": to_float(best_row, "metrics/mAP50-95(B)"),
        "fitness": to_float(best_row, "fitness"),
    }


def resolve_run_dir(expected_dir: Path, run_name: str, results: object | None) -> Path:
    """Resolve the actual Ultralytics run directory.

    Some Ultralytics versions ignore the requested project/name layout and save
    under a nested runs/detect/... directory. Prefer the reported save_dir when
    available, then fall back to searching for <run_name>/results.csv.
    """
    if results is not None and hasattr(results, "save_dir"):
        try:
            save_dir = Path(results.save_dir)
            if (save_dir / "results.csv").exists():
                return save_dir
        except Exception:
            pass

    if (expected_dir / "results.csv").exists():
        return expected_dir

    for root in {expected_dir.parent, Path.cwd(), Path("runs")}:
        try:
            root = root.resolve()
        except Exception:
            continue
        if not root.exists():
            continue
        for results_csv in root.rglob("results.csv"):
            if results_csv.parent.name == run_name:
                return results_csv.parent

    return expected_dir

def train_one_seed(args: argparse.Namespace, seed: int) -> dict:
    """Train one (cfg, seed) and return a dict with the best-epoch metrics."""
    set_seed(seed)

    run_name = f"{args.name}_seed{seed}"
    run_dir = Path(args.project) / run_name

    # Resume-or-skip: if this seed has already finished training, skip it
    # and just read metrics from the existing results.csv. Set --force to retrain.
    results_csv = run_dir / "results.csv"
    best_pt = run_dir / "weights" / "best.pt"
    if not getattr(args, "force", False) and results_csv.exists() and best_pt.exists():
        print(f"[skip] {run_name}: results.csv and best.pt exist, "
              f"reading metrics without retraining.")
        m = read_best_metrics_from_results_csv(run_dir)
        return {"seed": seed,
                "precision": m["precision"], "recall": m["recall"],
                "mAP50": m["mAP50"], "mAP50-95": m["mAP50-95"],
                "fitness": m["fitness"],
                "best_pt": str(best_pt)}

    model = YOLO(args.cfg).load(args.weights)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        deterministic=True,
        seed=seed,
        project=args.project,
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    # In DDP / multi-GPU mode, Ultralytics may return None.
    # So we read metrics from results.csv instead.
    run_dir = resolve_run_dir(run_dir, run_name, results)
    if results is not None and hasattr(results, "results_dict"):
        rd = results.results_dict

        precision = float(rd.get("metrics/precision(B)", 0.0))
        recall = float(rd.get("metrics/recall(B)", 0.0))
        mAP50 = float(rd.get("metrics/mAP50(B)", 0.0))
        mAP5095 = float(rd.get("metrics/mAP50-95(B)", 0.0))
        fitness = float(rd.get("fitness", 0.0))
    else:
        metrics = read_best_metrics_from_results_csv(run_dir)

        precision = metrics["precision"]
        recall = metrics["recall"]
        mAP50 = metrics["mAP50"]
        mAP5095 = metrics["mAP50-95"]
        fitness = metrics["fitness"]

    return {
        "seed": seed,
        "precision": precision,
        "recall": recall,
        "mAP50": mAP50,
        "mAP50-95": mAP5095,
        "fitness": fitness,
        "best_pt": str(run_dir / "weights" / "best.pt"),
    }


def summarise(rows: list[dict], out_path: Path) -> None:
    """Write per-seed rows + an extra row with mean ± std."""
    keys = ["seed", "precision", "recall", "mAP50", "mAP50-95", "fitness", "best_pt"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

        # Aggregate row
        agg: dict = {"seed": "mean±std", "best_pt": ""}
        for k in ("precision", "recall", "mAP50", "mAP50-95", "fitness"):
            vals = [r[k] for r in rows]
            agg[k] = f"{statistics.mean(vals):.4f}±{statistics.stdev(vals):.4f}" \
                     if len(vals) > 1 else f"{statistics.mean(vals):.4f}"
        w.writerow(agg)


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    rows = [train_one_seed(args, s) for s in seeds]

    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)
    summary_csv = project / f"{args.name}_summary.csv"
    summarise(rows, summary_csv)
    print(f"\nSummary written to {summary_csv}")


if __name__ == "__main__":
    main()
