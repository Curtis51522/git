"""
s1_recognition/full_train.py -- YOLO11s Ablation Study and Full Training
========================================================================

Ablation experiments to isolate the contribution of each training component:
  1. Baseline:    AdamW, 100ep, basic aug  (default YOLO recipe)
  2. Optimizer:   SGD,   100ep, basic aug  (isolates optimizer effect)
  3. Epochs:      SGD,   200ep, basic aug  (isolates extended training)
  4. Proposed:    SGD,   200ep, full aug   (extended augmentation + mixup)

All experiments use the same effective learning rate (lr0=0.005, nbs=batch)
and the same dataset (merged_yolo_30cls, 30 classes).

Usage
-----
    python s1_recognition/full_train.py --all     # run all 4 experiments
    python s1_recognition/full_train.py --proposed # run only proposed
"""

import os, sys, json, time, logging, argparse, platform, shutil
from pathlib import Path

import torch, yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import YOLO_MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("s1.ablation")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
DEFAULT_DATA_YAML = os.path.join(_PROJECT_DIR, "data", "merged_yolo_30cls", "data.yaml")
FINAL_MODEL_DIR = os.path.join(_PROJECT_DIR, "models", "yolo")


def _portable_model_path(tag: str) -> str:
    return Path("runs", "detect", tag, "weights", "best.pt").as_posix()


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(_PROJECT_DIR) / path

# ---- Experiment definitions ----
# Each dict: name, optimizer, epochs, batch, mixup, multi_scale, label_smoothing, description
EXPERIMENTS = [
    {
        "tag": "ablation_baseline",
        "name": "Baseline",
        "optimizer": "AdamW", "epochs": 100, "batch": 16, "nbs": 16,
        "lr0": 0.005, "mosaic": 1.0, "mixup": 0.0, "multi_scale": False,
        "label_smoothing": 0.0,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 0.0, "shear": 0.0,
        "warmup_epochs": 3, "close_mosaic": 10, "patience": 30,
        "desc": "AdamW 100ep basic aug (lr0=0.005, not default 0.01, for fair comparison)",
    },
    {
        "tag": "ablation_sgd",
        "name": "Ablation: SGD",
        "optimizer": "SGD", "epochs": 100, "batch": 16, "nbs": 16,
        "lr0": 0.005, "mosaic": 1.0, "mixup": 0.0, "multi_scale": False,
        "label_smoothing": 0.0,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 0.0, "shear": 0.0,
        "warmup_epochs": 3, "close_mosaic": 10, "patience": 30,
        "desc": "SGD 100ep basic aug (isolates optimizer change)",
    },
    {
        "tag": "ablation_epochs",
        "name": "Ablation: +Epochs",
        "optimizer": "SGD", "epochs": 200, "batch": 8, "nbs": 8,
        "lr0": 0.005, "mosaic": 1.0, "mixup": 0.0, "multi_scale": False,
        "label_smoothing": 0.0,
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 0.0, "shear": 0.0,
        "warmup_epochs": 3, "close_mosaic": 10, "patience": 50,
        "desc": "SGD 200ep basic aug (isolates extended training)",
    },
    {
        "tag": "ablation_proposed",
        "name": "Proposed (Full)",
        "optimizer": "SGD", "epochs": 200, "batch": 8, "nbs": 8,
        "lr0": 0.005, "mosaic": 1.0, "mixup": 0.1, "multi_scale": True,
        "label_smoothing": 0.1,
        "hsv_h": 0.015, "hsv_s": 0.4, "hsv_v": 0.3,
        "degrees": 15, "shear": 0.1,
        "warmup_epochs": 5, "close_mosaic": 15, "patience": 50,
        "desc": "SGD 200ep full aug (extended augmentation + mixup)",
    },
]

def log_system_info():
    logger.info("System: %s | Python %s | PyTorch %s", platform.system(), sys.version.split()[0], torch.__version__)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info("GPU: %s (%.1f GB VRAM)", gpu_name, vram)

def run_experiment(exp: dict, data_yaml: str) -> dict:
    """Run a single ablation experiment. Returns metrics dict."""
    from ultralytics import YOLO

    logger.info("=" * 72)
    logger.info("EXPERIMENT: %s", exp["name"])
    logger.info("  %s", exp["desc"])
    logger.info("  optimizer=%s epochs=%d batch=%d/%d lr0=%.4f mosaic=%.1f",
                exp["optimizer"], exp["epochs"], exp["batch"], exp["nbs"], exp["lr0"], exp["mosaic"])
    logger.info("  mixup=%.1f multi_scale=%s label_smoothing=%.1f",
                exp["mixup"], exp["multi_scale"], exp["label_smoothing"])
    logger.info("  Data: %s", data_yaml)
    logger.info("=" * 72)

    with open(data_yaml) as f:
        ds = yaml.safe_load(f)
    logger.info("Dataset: %d classes, train=%s, val=%s", ds.get("nc", 0), ds.get("train"), ds.get("val"))

    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    model = YOLO("yolo11s.pt")

    args = dict(
        data=data_yaml,
        epochs=exp["epochs"],
        imgsz=640,
        batch=exp["batch"],
        nbs=exp["nbs"],
        name=exp["tag"],
        project=os.path.join(_PROJECT_DIR, "runs", "detect"),
        exist_ok=True,
        pretrained=True,
        workers=2,
        cache='disk',
        optimizer=exp["optimizer"],
        lr0=exp["lr0"],
        cos_lr=True,
        warmup_epochs=exp["warmup_epochs"],
        box=7.5, cls=0.5, dfl=1.5,
        dropout=0.0,
        weight_decay=0.0005,
        label_smoothing=exp["label_smoothing"],
        multi_scale=exp["multi_scale"],
        augment=True,
        hsv_h=exp["hsv_h"], hsv_s=exp["hsv_s"], hsv_v=exp["hsv_v"],
        degrees=exp["degrees"],
        translate=0.1, scale=0.5,
        shear=exp["shear"],
        perspective=0.0, flipud=0.0, fliplr=0.5,
        mosaic=exp["mosaic"],
        mixup=exp["mixup"],
        close_mosaic=exp["close_mosaic"],
        patience=exp["patience"],
        amp=True,
        device=0,
    )

    results = model.train(**args)

    t_elapsed = time.time() - t_start
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)

    metrics = {
        "experiment": exp["name"],
        "tag": exp["tag"],
        "mAP50": round(float(results.results_dict.get("metrics/mAP50(B)", 0)), 4),
        "mAP50_95": round(float(results.results_dict.get("metrics/mAP50-95(B)", 0)), 4),
        "training_time_s": round(t_elapsed, 1),
        "peak_gpu_memory_gb": round(peak_mem, 2),
        "epochs_trained": exp["epochs"],
        "model_path": _portable_model_path(exp["tag"]),
    }

    logger.info("%s done: mAP50=%.4f mAP50-95=%.4f time=%.0fs VRAM=%.1fGB",
                exp["name"], metrics["mAP50"], metrics["mAP50_95"], t_elapsed, peak_mem)

    # Per-class evaluation
    logger.info("Running per-class validation...")
    val_results = model.val(data=data_yaml, split="val", device=0)
    per_class = {}
    if val_results.box.ap_class_index is not None and val_results.names:
        ap50 = val_results.box.ap50 if hasattr(val_results.box, 'ap50') and val_results.box.ap50 is not None else []
        for idx, ap in enumerate(ap50):
            cls_id = val_results.box.ap_class_index[idx] if idx < len(val_results.box.ap_class_index) else idx
            cls_name = val_results.names.get(int(cls_id), f"class_{cls_id}")
            per_class[cls_name] = round(float(ap), 4)
    metrics["per_class_ap50"] = per_class

    # Save per-experiment JSON
    exp_json = os.path.join(_SCRIPT_DIR, f"{exp['tag']}_metrics.json")
    with open(exp_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Per-class metrics saved: %s (%d classes)", exp_json, len(per_class))

    # Log top/bottom 5
    if per_class:
        sorted_ap = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
        logger.info("  Top 5: %s", ", ".join(f"{n}={v:.3f}" for n, v in sorted_ap[:5]))
        logger.info("  Bottom 5: %s", ", ".join(f"{n}={v:.3f}" for n, v in sorted_ap[-5:]))

    # Free GPU memory before next experiment
    del model
    torch.cuda.empty_cache()

    return metrics

def print_comparison(all_metrics: list):
    """Print ablation comparison table."""
    logger.info("")
    logger.info("=" * 72)
    logger.info("ABLATION STUDY RESULTS")
    logger.info("=" * 72)
    logger.info("%-25s %10s %12s %8s %10s", "Experiment", "mAP@0.5", "mAP@0.5:0.95", "Time(h)", "VRAM(GB)")
    logger.info("-" * 72)
    for m in all_metrics:
        logger.info("%-25s %10.4f %12.4f %7.1f %9.1f",
                    m["experiment"], m["mAP50"], m["mAP50_95"],
                    m["training_time_s"]/3600, m["peak_gpu_memory_gb"])

    # Delta from baseline
    if all_metrics:
        base = all_metrics[0]
        logger.info("")
        logger.info("Improvement over Baseline:")
        for m in all_metrics[1:]:
            delta = m["mAP50"] - base["mAP50"]
            logger.info("  %-25s mAP50: %+.4f  mAP50-95: %+.4f",
                        m["experiment"], delta, m["mAP50_95"] - base["mAP50_95"])
    logger.info("=" * 72)

    # Save JSON
    results_path = os.path.join(_SCRIPT_DIR, "ablation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Results saved: %s", results_path)

def main():
    parser = argparse.ArgumentParser(description="YOLO11s ablation study for bakery detection.")
    parser.add_argument("--all", action="store_true", help="Run all 4 ablation experiments in sequence.")
    parser.add_argument("--proposed", action="store_true", help="Run only the proposed experiment.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_YAML, help="Dataset YAML path.")
    args = parser.parse_args()

    log_system_info()

    if args.proposed:
        to_run = [EXPERIMENTS[-1]]
    elif args.all:
        to_run = EXPERIMENTS
    else:
        to_run = EXPERIMENTS  # default: run all

    all_metrics = []
    for exp in to_run:
        try:
            metrics = run_experiment(exp, args.data)
            all_metrics.append(metrics)
        except Exception as e:
            logger.error("Experiment %s FAILED: %s", exp["name"], e)
            all_metrics.append({"experiment": exp["name"], "error": str(e)})

    if len(all_metrics) >= 2:
        print_comparison(all_metrics)

    # Deploy the successful experiment with the strongest validation mAP50-95.
    successful = [m for m in all_metrics if "error" not in m and m.get("model_path")]
    if successful:
        selected = max(successful, key=lambda item: float(item.get("mAP50_95", -1)))
        src_value = selected.get("model_path")
        src = _resolve_project_path(src_value) if src_value else None
        if src and os.path.exists(src):
            os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
            shutil.copy2(src, YOLO_MODEL_PATH)
            logger.info(
                "Final model: %s selected from %s (mAP50-95=%.4f)",
                YOLO_MODEL_PATH,
                selected.get("experiment", selected.get("tag", "unknown")),
                float(selected.get("mAP50_95", 0)),
            )

    logger.info("ALL DONE")

if __name__ == "__main__":
    main()
