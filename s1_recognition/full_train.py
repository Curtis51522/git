"""
s1_recognition/full_train.py — Full-length YOLO Training with Optimal Hyperparameters
======================================================================================

Methodology
-----------
Trains YOLO11s for 200 epochs using the optimal hyperparameter configuration
discovered by `experiments.py` (grid search + ablation). The configuration
is read from `s1_recognition/best_config.json`.

Key design decisions for bakery detection:
- **Full training length (200 epochs)**: Based on findings from He et al.
  (2019) "Rethinking ImageNet Pre-training" — longer training benefits
  from-scratch and fine-tuning scenarios. 200 epochs provides sufficient
  convergence for a 16k-image dataset while avoiding overfitting.
- **Transfer learning**: COCO-pretrained YOLO11s backbone unless the best
  config specifies pretrained=False (the ablation study determines this).
- **Regularisation**: Label smoothing (0.1), weight decay (5e-4), cosine LR
  schedule, and warmup (5 epochs) per Szegedy et al. (2016) and
  Ultralytics YOLO recommendations.
- **Augmentation**: Multi-scale training (Ge et al. 2021, YOLOX),
  mosaic + mixup augmentation (Bochkovskiy et al. 2020, YOLOv4),
  colour jitter, rotation, shear, and flip.
- **Output**: The final best.pt is saved to `models/yolo/best.pt` for
  downstream inference (module1_yolo.py, API server).

Usage
-----
    python s1_recognition/full_train.py
    python s1_recognition/full_train.py --config path/to/best_config.json --epochs 200 --batch 16
"""

import os
import sys
import json
import time
import logging
import argparse
import platform
import shutil

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import YOLO_MODEL_PATH

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("s1.full_train")


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

DEFAULT_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "best_config.json")
DEFAULT_DATA_YAML = os.path.join(_PROJECT_DIR, "data", "merged_yolo_30cls", "data.yaml")
FINAL_MODEL_DIR = os.path.join(_PROJECT_DIR, "models", "yolo")
DEFAULT_EPOCHS = 200


# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------
def log_system_info() -> None:
    """Log system, Python, PyTorch, and GPU details for experiment tracking."""
    logger.info("System: %s | Python %s | PyTorch %s",
                platform.system(), sys.version.split()[0], torch.__version__)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info("GPU: %s (%.1f GB VRAM)", gpu_name, vram)
        logger.info("CUDA version: %s", torch.version.cuda)
    else:
        logger.warning("CUDA not available — training will be very slow on CPU.")


# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------
def load_config(config_path: str) -> dict:
    """
    Load hyperparameter configuration from JSON file.

    Expected keys: lr0, mosaic, batch, optimizer, epochs (optional, overridden
    by command-line --epochs), pretrained (optional, defaults to True).
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Run s1_recognition/experiments.py first to generate best_config.json."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    logger.info("Loaded configuration from %s", config_path)
    for k, v in config.items():
        logger.info("  %s: %s", k, v)

    return config


# ---------------------------------------------------------------------------
# Full training
# ---------------------------------------------------------------------------
def full_train(
    config: dict,
    epochs: int = DEFAULT_EPOCHS,
    data_yaml: str = DEFAULT_DATA_YAML,
) -> dict:
    """
    Run full-length YOLO training with the best hyperparameter configuration.

    Parameters
    ----------
    config : dict
        Hyperparameter dictionary with keys: lr0, mosaic, batch, optimizer,
        and optionally pretrained.
    epochs : int
        Number of training epochs (default: 200).
    data_yaml : str
        Path to dataset YAML configuration.

    Returns
    -------
    dict with training metrics: mAP50, mAP50_95, peak_gpu_memory_gb,
    training_time_s, and model_path.
    """
    from ultralytics import YOLO

    lr0 = float(config["lr0"])
    mosaic = float(config["mosaic"])
    batch = int(config["batch"])
    optimizer = config.get("optimizer", "AdamW")
    pretrained = config.get("pretrained", True)

    # Seed everything for reproducibility
    import random as _random
    _random.seed(42)
    import numpy as _np
    _np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

    logger.info("=" * 72)
    logger.info("FULL TRAINING — %d epochs", epochs)
    logger.info("Hyperparameters:")
    logger.info("  lr0=%.4f, mosaic=%.1f, batch=%d, optimizer=%s, pretrained=%s",
                lr0, mosaic, batch, optimizer, pretrained)
    logger.info("  Data: %s", data_yaml)
    logger.info("=" * 72)

    # Validate data YAML
    with open(data_yaml) as f:
        ds = yaml.safe_load(f)
    logger.info("Dataset: %d classes, train=%s, val=%s",
                ds.get("nc", 0), ds.get("train"), ds.get("val"))

    # Reset GPU memory stats
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    model = YOLO("yolo11s.pt")

    training_args = dict(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        name="full_train_bakery",
        project=os.path.join(_PROJECT_DIR, "runs", "detect"),
        exist_ok=True,
        pretrained=pretrained,
        workers=4,
        optimizer=optimizer,
        lr0=lr0,
        cos_lr=True,
        warmup_epochs=5,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        dropout=0.0,
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
        close_mosaic=15 if mosaic > 0.0 else 0,
        patience=50,
        device=0,
    )

    results = model.train(**training_args)

    t_elapsed = time.time() - t_start
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Extract metrics
    metrics = {
        "mAP50": round(float(results.results_dict.get("metrics/mAP50(B)", 0)), 4),
        "mAP50_95": round(float(results.results_dict.get("metrics/mAP50-95(B)", 0)), 4),
        "training_time_s": round(t_elapsed, 1),
        "peak_gpu_memory_gb": round(peak_mem, 2),
        "epochs_trained": epochs,
    }

    logger.info("Training completed in %.1f s (%.2f GB peak VRAM).",
                t_elapsed, peak_mem)
    logger.info("mAP@0.5 = %.4f, mAP@0.5:0.95 = %.4f",
                metrics["mAP50"], metrics["mAP50_95"])

    # Copy best.pt to final model location
    src_best = os.path.join("runs", "detect", "full_train_bakery", "weights", "best.pt")
    if os.path.exists(src_best):
        shutil.copy2(src_best, YOLO_MODEL_PATH)
        logger.info("Final model saved: %s", YOLO_MODEL_PATH)
        metrics["model_path"] = YOLO_MODEL_PATH
    else:
        logger.warning("best.pt not found at %s", src_best)
        metrics["model_path"] = None

    # Also save a copy in s1_recognition/runs
    exp_run_dir = os.path.join(_SCRIPT_DIR, "runs", "full_train_final")
    os.makedirs(exp_run_dir, exist_ok=True)
    if os.path.exists(src_best):
        shutil.copy2(src_best, os.path.join(exp_run_dir, "best.pt"))
        logger.info("Copy also saved: %s", os.path.join(exp_run_dir, "best.pt"))

    return metrics


# ---------------------------------------------------------------------------
# Per-class evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model_path: str, data_yaml: str) -> dict:
    """
    Evaluate the trained model on the validation set with per-class metrics.

    Returns dict with overall mAP and per-class AP for thesis reporting.
    """
    from ultralytics import YOLO

    if not os.path.exists(model_path):
        logger.error("Model not found: %s", model_path)
        return {}

    logger.info("Evaluating model: %s", model_path)
    model = YOLO(model_path)
    val_results = model.val(data=data_yaml, split="val", device=0)

    per_class = {}
    if val_results.box.ap_class_index is not None and val_results.names:
        for i, ap in enumerate(val_results.box.ap50):
            cls_name = val_results.names.get(
                val_results.box.ap_class_index[i], f"class_{i}"
            )
            per_class[cls_name] = round(float(ap), 4)

    metrics = {
        "mAP50": round(float(val_results.box.map50), 4),
        "mAP50_95": round(float(val_results.box.map), 4),
        "per_class_ap50": per_class,
    }

    logger.info("Evaluation:")
    logger.info("  mAP@0.5 = %.4f", metrics["mAP50"])
    logger.info("  mAP@0.5:0.95 = %.4f", metrics["mAP50_95"])
    logger.info("  Per-class AP (top 5):")
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
    for cls_name, ap in sorted_classes[:5]:
        logger.info("    %s: %.4f", cls_name, ap)
    logger.info("  Per-class AP (bottom 5):")
    for cls_name, ap in sorted_classes[-5:]:
        logger.info("    %s: %.4f", cls_name, ap)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Full YOLO training with optimal hyperparameters (200 epochs)."
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG_PATH,
        help=f"Path to best_config.json (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--data", type=str, default=DEFAULT_DATA_YAML,
        help=f"Dataset YAML (default: {DEFAULT_DATA_YAML})",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Override batch size from config.",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; only evaluate an existing model.",
    )
    parser.add_argument(
        "--model", type=str, default=YOLO_MODEL_PATH,
        help="Model path for --eval-only (default: models/yolo/best.pt)",
    )
    args = parser.parse_args()

    log_system_info()

    if args.eval_only:
        evaluate_model(args.model, args.data)
        return

    config = load_config(args.config)

    # Allow command-line batch override
    if args.batch is not None:
        config["batch"] = args.batch
        logger.info("Batch size overridden to %d", args.batch)

    # Run full training
    metrics = full_train(config, epochs=args.epochs, data_yaml=args.data)

    # Evaluate the final model
    evaluate_model(YOLO_MODEL_PATH, args.data)

    logger.info("=" * 72)
    logger.info("FULL TRAINING COMPLETE")
    logger.info("Best model: %s", YOLO_MODEL_PATH)
    logger.info("mAP@0.5 = %.4f, mAP@0.5:0.95 = %.4f",
                metrics["mAP50"], metrics["mAP50_95"])
    logger.info("Training time: %.1f s", metrics["training_time_s"])
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
