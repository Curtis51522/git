"""
s1_recognition/experiments.py — YOLO Hyperparameter Grid Search & Ablation Study
===============================================================================

Methodology
-----------
This script conducts a controlled hyperparameter grid search for YOLO11s
fine-tuning on a 28-class bakery detection dataset (16,272 train / 3,048 val
images). The methodology follows standard practices in object detection
thesis work:

- **Grid Search**: Exhaustive combination of learning rate, mosaic augmentation,
  and batch size (3×2×2 = 12 conditions), each trained for 50 epochs.
  Based on: Bergstra & Bengio (2012) "Random Search for Hyper-Parameter
  Optimization"; and modern YOLO tuning guides (Ultralytics, 2024).

- **Ablation A (Pretraining)**: Compares COCO-pretrained vs. from-scratch
  initialization using the best configuration found above. References:
  He et al. (2019) "Rethinking ImageNet Pre-training" — from-scratch
  typically needs 3-5× more epochs for comparable accuracy.

- **Ablation B (Mosaic Augmentation)**: Tests the effect of mosaic=1.0 vs.
  mosaic=0.0 at the best learning rate. References: Bochkovskiy et al. (2020)
  "YOLOv4: Optimal Speed and Accuracy of Object Detection" — mosaic
  augmentation improves small-object detection by mixing 4 images.

- **Evaluation Metrics**: mAP@0.5, mAP@0.5:0.95, per-class AP (top 5 and
  bottom 5 by performance), training time, peak GPU memory.

- **OOM Handling**: Each experiment is wrapped in a try-except for CUDA
  out-of-memory, allowing the grid to continue without manual intervention.

Outputs
-------
- s1_recognition/experiment_results.csv — full results table
- s1_recognition/best_config.json — best hyperparameter configuration
- s1_recognition/runs/<experiment_name>/best.pt — per-experiment checkpoints
"""

import os
import sys
import csv
import json
import time
import logging
import argparse
import itertools
import platform
from datetime import datetime

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import YOLO_MODEL_PATH

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("s1.experiments")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "merged_yolo_30cls", "data.yaml",
)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
RESULTS_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "experiment_results.csv",
)
BEST_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "best_config.json",
)
FINAL_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "yolo",
)

GRID_SEARCH_EPOCHS = 30
FULL_TRAIN_EPOCHS = 200

GRID_LR0 = [0.01, 0.005, 0.001]
GRID_MOSAIC = [1.0, 0.0]
GRID_BATCH = [16, 8]  # batch=32 OOM'd on 12.8 GB VRAM per prior testing
GRID_OPTIMIZER = ["AdamW", "SGD"]

# YOLO11 loss weights (Ultralytics defaults, kept explicit for thesis)
YOLO_BOX_LOSS = 7.5
YOLO_CLS_LOSS = 0.5
YOLO_DFL_LOSS = 1.5
YOLO_DROPOUT = 0.0

# Class names read from data.yaml at runtime (not hardcoded)
CLASS_NAMES = None


# ---------------------------------------------------------------------------
# Helper: GPU device info
# ---------------------------------------------------------------------------
def get_gpu_info() -> dict:
    """Return GPU name and total VRAM in GB."""
    if not torch.cuda.is_available():
        return {"name": "N/A", "vram_gb": 0.0}
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return {"name": name, "vram_gb": round(vram, 1)}


# ---------------------------------------------------------------------------
# Helper: build experiment name
# ---------------------------------------------------------------------------
def experiment_name(lr0: float, mosaic: float, batch: int, optimizer: str = "AdamW", tag: str = "") -> str:
    """Generate a deterministic experiment name from hyperparameters."""
    opt_short = "AdamW" if optimizer == "AdamW" else "SGD"
    base = f"lr{lr0}_mosaic{mosaic}_bs{batch}_{opt_short}"
    return f"{base}_{tag}" if tag else base


# ---------------------------------------------------------------------------
# Core training function (single experiment)
# ---------------------------------------------------------------------------
def run_experiment(
    lr0: float,
    mosaic: float,
    batch: int,
    epochs: int,
    optimizer: str = "AdamW",
    pretrained: bool = True,
    experiment_tag: str = "",
    close_mosaic: int = 15,
) -> dict:
    """
    Train YOLO11s with given hyperparameters and return metrics dict.

    Parameters
    ----------
    lr0 : float
        Initial learning rate.
    mosaic : float
        Mosaic augmentation probability (0.0 to 1.0).
    batch : int
        Batch size per GPU.
    epochs : int
        Number of training epochs.
    pretrained : bool
        Whether to initialise from COCO-pretrained weights.
    experiment_tag : str
        Optional tag appended to experiment folder name.
    close_mosaic : int
        Number of final epochs during which mosaic is disabled (Ultralytics default).

    Returns
    -------
    dict with keys: status, experiment, lr0, mosaic, batch, optimizer, epochs,
    mAP50, mAP50_95, per_class_ap50, training_time_s, peak_gpu_memory_gb,
    pretrained, error_message (if failed).
    """
    from ultralytics import YOLO

    exp_name = experiment_name(lr0, mosaic, batch, optimizer, experiment_tag)
    run_dir = os.path.join(RESULTS_DIR, exp_name)
    os.makedirs(run_dir, exist_ok=True)

    # Seed for reproducibility
    import random as _random
    _random.seed(42)
    import numpy as _np
    _np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Build base training kwargs
    train_kwargs = dict(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=416,
        batch=batch,
        name=exp_name,
        project=RESULTS_DIR,
        exist_ok=True,
        pretrained=pretrained,
        workers=4,
        optimizer=optimizer,
        lr0=lr0,
        box=YOLO_BOX_LOSS,
        cls=YOLO_CLS_LOSS,
        dfl=YOLO_DFL_LOSS,
        dropout=YOLO_DROPOUT,
        cos_lr=True,
        warmup_epochs=5,
        weight_decay=0.0005,
        label_smoothing=0.1,
        multi_scale=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=15,
        translate=0.1,
        scale=0.3,
        shear=0.1,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=mosaic,
        mixup=0.1 if mosaic > 0.0 else 0.0,
        close_mosaic=close_mosaic if mosaic > 0.0 else 0,
        patience=30,
        device=0,
    )

    result = {
        "status": "running",
        "experiment": exp_name,
        "lr0": lr0,
        "mosaic": mosaic,
        "batch": batch,
        "optimizer": optimizer,
        "epochs": epochs,
        "pretrained": pretrained,
        "mAP50": None,
        "mAP50_95": None,
        "per_class_ap50": {},
        "training_time_s": None,
        "peak_gpu_memory_gb": None,
        "error_message": None,
    }

    try:
        torch.cuda.reset_peak_memory_stats()
        t_start = time.time()

        model = YOLO("yolo11s.pt")
        results = model.train(**train_kwargs)

        t_elapsed = time.time() - t_start
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

        # Extract metrics
        result["mAP50"] = round(float(results.results_dict.get("metrics/mAP50(B)", 0)), 4)
        result["mAP50_95"] = round(float(results.results_dict.get("metrics/mAP50-95(B)", 0)), 4)
        result["training_time_s"] = round(t_elapsed, 1)
        result["peak_gpu_memory_gb"] = round(peak_mem, 2)

        # Evaluate on validation set for per-class AP
        val_results = model.val(data=DATA_YAML, split="val", batch=batch, device=0)
        per_class = {}
        if val_results.box.ap_class_index is not None and val_results.names:
            for i, ap in enumerate(val_results.box.ap50):
                cls_name = val_results.names.get(
                    val_results.box.ap_class_index[i], f"class_{i}"
                )
                per_class[cls_name] = round(float(ap), 4)
        result["per_class_ap50"] = per_class

        # Save best.pt to experiment directory
        src_best = os.path.join(RESULTS_DIR, exp_name, "weights", "best.pt")
        if os.path.exists(src_best):
            dst_best = os.path.join(run_dir, "best.pt")
            # shutil.copy to avoid recursion
            import shutil
            shutil.copy2(src_best, dst_best)
            logger.info("Best model saved: %s", dst_best)
        else:
            logger.warning("best.pt not found at %s", src_best)

        result["status"] = "completed"
        logger.info(
            "Experiment %s done — mAP50=%.4f, mAP50-95=%.4f, time=%.1fs, peak=%.2fGB",
            exp_name, result["mAP50"], result["mAP50_95"],
            t_elapsed, peak_mem,
        )

    except torch.cuda.OutOfMemoryError:
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        result["status"] = "OOM"
        result["peak_gpu_memory_gb"] = round(peak_mem, 2)
        result["error_message"] = (
            f"CUDA OOM during {exp_name}. "
            f"Peak memory: {peak_mem:.2f} / {get_gpu_info()['vram_gb']:.1f} GB."
        )
        logger.warning("Experiment %s failed with OOM (peak %.2f GB)", exp_name, peak_mem)
        torch.cuda.empty_cache()

    except Exception as e:
        result["status"] = "failed"
        result["error_message"] = str(e)
        logger.error("Experiment %s failed: %s", exp_name, e)
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Per-class AP ranking helper
# ---------------------------------------------------------------------------
def get_top_bottom_classes(
    per_class: dict, n: int = 5
) -> tuple:
    """
    Return (top_n, bottom_n) class names sorted by AP@0.5.

    Useful for thesis discussion of which categories the model finds easy
    or difficult (typically influenced by class frequency, visual similarity,
    and bounding-box size distribution).
    """
    sorted_items = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
    top = [(name, round(ap, 4)) for name, ap in sorted_items[:n]]
    bottom = [(name, round(ap, 4)) for name, ap in sorted_items[-n:]]
    return top, bottom


# ---------------------------------------------------------------------------
# Save results to CSV
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "experiment", "status", "lr0", "mosaic", "batch", "optimizer",
    "epochs", "pretrained", "mAP50", "mAP50_95",
    "training_time_s", "peak_gpu_memory_gb",
    "top5_classes", "top5_aps",
    "bottom5_classes", "bottom5_aps",
    "error_message",
]


def append_to_csv(results_csv: str, result: dict) -> None:
    """Append a single experiment result row to the CSV file."""
    top5, bot5 = get_top_bottom_classes(result.get("per_class_ap50", {}), n=5)
    row = {
        "experiment": result.get("experiment", ""),
        "status": result.get("status", ""),
        "lr0": result.get("lr0", ""),
        "mosaic": result.get("mosaic", ""),
        "batch": result.get("batch", ""),
        "optimizer": result.get("optimizer", ""),
        "epochs": result.get("epochs", ""),
        "pretrained": result.get("pretrained", ""),
        "mAP50": result.get("mAP50", ""),
        "mAP50_95": result.get("mAP50_95", ""),
        "training_time_s": result.get("training_time_s", ""),
        "peak_gpu_memory_gb": result.get("peak_gpu_memory_gb", ""),
        "top5_classes": json.dumps([c for c, _ in top5]) if top5 else "",
        "top5_aps": json.dumps([a for _, a in top5]) if top5 else "",
        "bottom5_classes": json.dumps([c for c, _ in bot5]) if bot5 else "",
        "bottom5_aps": json.dumps([a for _, a in bot5]) if bot5 else "",
        "error_message": result.get("error_message", ""),
    }
    file_exists = os.path.isfile(results_csv)
    with open(results_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    logger.info("Result appended to %s", results_csv)


# ---------------------------------------------------------------------------
# Find best configuration from results
# ---------------------------------------------------------------------------
def find_best_config(results_csv: str) -> dict:
    """
    Read the experiment CSV and return the config with the highest mAP@0.5.

    Returns dict with keys: lr0, mosaic, batch, optimizer, epochs,
    mAP50, mAP50_95, and the name of the winning experiment.
    """
    import csv as _csv
    best = None
    best_map = -1.0

    with open(results_csv, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            if row["status"] != "completed":
                continue
            try:
                m50 = float(row["mAP50"])
            except (ValueError, TypeError):
                continue
            if m50 > best_map:
                best_map = m50
                best = {
                    "experiment": row["experiment"],
                    "lr0": float(row["lr0"]),
                    "mosaic": float(row["mosaic"]),
                    "batch": int(row["batch"]),
                    "optimizer": row["optimizer"],
                    "pretrained": row.get("pretrained", "True").lower() not in ("false", "0", ""),
                    "epochs": int(row["epochs"]),
                    "mAP50": float(row["mAP50"]),
                    "mAP50_95": float(row["mAP50_95"]),
                    "peak_gpu_memory_gb": float(row["peak_gpu_memory_gb"]) if row["peak_gpu_memory_gb"] else None,
                }

    if best is None:
        raise RuntimeError("No completed experiments found in CSV — cannot select best config.")
    return best


# ---------------------------------------------------------------------------
# Run grid search
# ---------------------------------------------------------------------------
def run_grid_search() -> list:
    """
    Execute the full grid search over lr0, mosaic, and batch.

    Returns list of result dicts for all 12 conditions.
    """
    logger.info("=" * 72)
    logger.info("Starting YOLO hyperparameter grid search")
    logger.info("GPU: %s", get_gpu_info())
    logger.info("Grid: lr0=%s, mosaic=%s, batch=%s, optimizer=%s, epochs=%d",
                GRID_LR0, GRID_MOSAIC, GRID_BATCH, GRID_OPTIMIZER, GRID_SEARCH_EPOCHS)
    logger.info("Total conditions: %d", len(GRID_LR0) * len(GRID_MOSAIC) * len(GRID_BATCH) * len(GRID_OPTIMIZER))
    logger.info("=" * 72)

    # Build set of already-completed experiment names
    completed = set()
    if os.path.exists(RESULTS_CSV):
        import csv as _csv
        with open(RESULTS_CSV, "r", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                if row.get("status") == "completed":
                    completed.add(row["experiment"])
        if completed:
            logger.info("Found %d completed experiments in CSV — will skip", len(completed))

    results_list = []
    conditions = list(itertools.product(GRID_LR0, GRID_MOSAIC, GRID_BATCH, GRID_OPTIMIZER))

    for idx, (lr0, mosaic, batch, optimizer) in enumerate(conditions, 1):
        exp_name = experiment_name(lr0, mosaic, batch, optimizer, "grid")
        if exp_name in completed:
            logger.info("Experiment %d / %d: %s — SKIPPED (already completed)", idx, len(conditions), exp_name)
            continue

        logger.info("-" * 72)
        logger.info("Experiment %d / %d: lr0=%.3f, mosaic=%.1f, batch=%d, optimizer=%s",
                    idx, len(conditions), lr0, mosaic, batch, optimizer)

        result = run_experiment(
            lr0=lr0,
            mosaic=mosaic,
            batch=batch,
            optimizer=optimizer,
            epochs=GRID_SEARCH_EPOCHS,
            pretrained=True,
            experiment_tag="grid",
        )
        append_to_csv(RESULTS_CSV, result)
        results_list.append(result)

    logger.info("Grid search completed.")
    return results_list


# ---------------------------------------------------------------------------
# Run ablation experiments
# ---------------------------------------------------------------------------
def run_ablations(best_config: dict) -> list:
    """
    Run ablation studies using the best configuration found in grid search.

    Ablation A: pretrained=True vs. pretrained=False (best config).
    Ablation B: mosaic=1.0 vs. mosaic=0.0 (best lr0, batch=16).
    """
    logger.info("=" * 72)
    logger.info("Starting ablation experiments")
    logger.info("=" * 72)

    ablations = []
    bc = best_config

    # --- Ablation A: Pretrained vs. From Scratch ---
    for pretrained in [True, False]:
        tag = "pretrained" if pretrained else "scratch"
        exp_name = experiment_name(bc["lr0"], bc["mosaic"], bc["batch"], f"ablationA_{tag}")
        logger.info(
            "Ablation A — %s: lr0=%.3f, mosaic=%.1f, batch=%d, epochs=%d",
            tag, bc["lr0"], bc["mosaic"], bc["batch"], GRID_SEARCH_EPOCHS,
        )
        result = run_experiment(
            lr0=bc["lr0"],
            mosaic=bc["mosaic"],
            batch=bc["batch"],
            epochs=GRID_SEARCH_EPOCHS,
            pretrained=pretrained,
            experiment_tag=f"ablationA_{tag}",
        )
        append_to_csv(RESULTS_CSV, result)
        ablations.append(result)

    # --- Ablation B: Mosaic On vs. Off (best lr0, batch=16) ---
    ablation_b_batch = 16  # use larger batch for better sensitivity
    for mosaic_val in [1.0, 0.0]:
        tag = "mosaic_on" if mosaic_val > 0.0 else "mosaic_off"
        exp_name = experiment_name(bc["lr0"], mosaic_val, ablation_b_batch, f"ablationB_{tag}")
        logger.info(
            "Ablation B — %s: lr0=%.3f, mosaic=%.1f, batch=%d, epochs=%d",
            tag, bc["lr0"], mosaic_val, ablation_b_batch, GRID_SEARCH_EPOCHS,
        )
        result = run_experiment(
            lr0=bc["lr0"],
            mosaic=mosaic_val,
            batch=ablation_b_batch,
            epochs=GRID_SEARCH_EPOCHS,
            pretrained=True,
            experiment_tag=f"ablationB_{tag}",
        )
        append_to_csv(RESULTS_CSV, result)
        ablations.append(result)

    # --- Ablation C: Zero-shot baseline (COCO pretrained, no bakery training) ---
    logger.info("Ablation C — Zero-shot: evaluating yolo11s COCO weights without training")
    from ultralytics import YOLO as _YOLO
    model_zs = _YOLO("yolo11s.pt")
    zs_results = model_zs.val(data=DATA_YAML, split="val", device=0)
    zs_map50 = round(float(zs_results.box.map50), 4)
    zs_map50_95 = round(float(zs_results.box.map), 4)
    zs_row = {
        "experiment": "ablationC_zero_shot", "status": "completed",
        "lr0": 0, "mosaic": 0, "batch": 16, "optimizer": "none",
        "epochs": 0, "pretrained": True,
        "mAP50": zs_map50, "mAP50_95": zs_map50_95,
        "training_time_s": 0, "peak_gpu_memory_gb": 0,
        "top5_classes": "", "top5_aps": "", "bottom5_classes": "", "bottom5_aps": "",
        "error_message": "",
    }
    append_to_csv(RESULTS_CSV, zs_row)
    ablations.append(zs_row)
    logger.info("Zero-shot baseline: mAP@0.5=%.4f, mAP@0.5:0.95=%.4f", zs_map50, zs_map50_95)

    logger.info("Ablation experiments completed.")
    return ablations


# ---------------------------------------------------------------------------
# Print formatted summary
# ---------------------------------------------------------------------------
def print_summary(results_csv: str) -> None:
    """Print a summary table of all experiments."""
    import csv as _csv

    logger.info("=" * 72)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 72)

    rows = []
    with open(results_csv, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Group by experiment type
    grid_rows = [r for r in rows if "grid" in r.get("experiment", "")]
    ablation_rows = [r for r in rows if "ablation" in r.get("experiment", "")]

    logger.info("--- Grid Search (%d conditions) ---", len(grid_rows))
    logger.info("%-30s %-8s %-6s %-6s %-6s %-10s %-10s %-8s",
                "Experiment", "Status", "lr0", "Mosaic", "Batch", "mAP50", "mAP50-95", "Time(s)")
    for r in grid_rows:
        logger.info("%-30s %-8s %-6s %-6s %-6s %-10s %-10s %-8s",
                    r["experiment"][:30], r["status"], r["lr0"], r["mosaic"],
                    r["batch"], r["mAP50"][:6] if r["mAP50"] else "N/A",
                    r["mAP50_95"][:6] if r["mAP50_95"] else "N/A",
                    r["training_time_s"][:6] if r["training_time_s"] else "N/A")

    logger.info("--- Ablation Experiments ---")
    logger.info("%-30s %-8s %-10s %-8s",
                "Experiment", "Status", "mAP50", "mAP50-95")
    for r in ablation_rows:
        logger.info("%-30s %-8s %-10s %-8s",
                    r["experiment"][:30], r["status"],
                    r["mAP50"][:6] if r["mAP50"] else "N/A",
                    r["mAP50_95"][:6] if r["mAP50_95"] else "N/A")

    # Best config
    best = find_best_config(results_csv)
    logger.info("")
    logger.info("BEST CONFIGURATION (by mAP@0.5):")
    for k, v in best.items():
        logger.info("  %s: %s", k, v)
    logger.info("=" * 72)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="YOLO hyperparameter grid search and ablation for bakery detection thesis."
    )
    parser.add_argument(
        "--skip-grid", action="store_true",
        help="Skip grid search and only run ablation experiments.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing results CSV (skip grid if CSV exists).",
    )
    parser.add_argument(
        "--epochs", type=int, default=GRID_SEARCH_EPOCHS,
        help=f"Number of epochs per experiment (default: {GRID_SEARCH_EPOCHS}).",
    )
    args = parser.parse_args()

    logger.info("System: %s | Python %s | PyTorch %s",
                platform.system(), sys.version.split()[0], torch.__version__)
    if torch.cuda.is_available():
        gpu_info = get_gpu_info()
        logger.info("GPU: %s (%.1f GB VRAM)", gpu_info["name"], gpu_info["vram_gb"])
        logger.info("CUDA version: %s", torch.version.cuda)
    else:
        logger.warning("CUDA not available — running on CPU (will be slow)")

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Grid Search ---
    # run_grid_search() now automatically skips completed experiments in CSV
    if not args.skip_grid:
        run_grid_search()
    else:
        logger.info("Skipping grid search (--skip-grid).")

    # --- Find best config ---
    if not os.path.exists(RESULTS_CSV):
        logger.error("No results CSV found. Run grid search first.")
        sys.exit(1)

    best_config = find_best_config(RESULTS_CSV)
    logger.info("Best configuration (so far): %s", json.dumps(best_config, indent=2))

    with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)
    logger.info("Best config saved to %s", BEST_CONFIG_PATH)

    # Count completed grid experiments
    expected_total = len(GRID_LR0) * len(GRID_MOSAIC) * len(GRID_BATCH) * len(GRID_OPTIMIZER)
    csv_completed = sum(1 for r in csv.DictReader(open(RESULTS_CSV)) if r.get("status") == "completed")
    if csv_completed < expected_total:
        logger.info("Grid search incomplete (%d/%d). Re-run without --skip-grid to continue.",
                     csv_completed, expected_total)
        logger.info("Skipping ablations — run again after grid completes.")
        return

    # --- Ablation Experiments ---
    run_ablations(best_config)

    # --- Final Summary ---
    print_summary(RESULTS_CSV)
    logger.info("All experiments complete. Results: %s", RESULTS_CSV)
    logger.info("Best config: %s", BEST_CONFIG_PATH)


if __name__ == "__main__":
    main()
