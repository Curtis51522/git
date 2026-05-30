import os, json, hashlib, shutil, argparse, logging
from pathlib import Path
from ultralytics import YOLO
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import YOLO_MODEL_PATH, MODEL_CACHE_DIR, CACHE_MANIFEST, PRODUCT_TYPES

logger = logging.getLogger("s1.training")

def get_dataset_hash(data_dir):
    hasher = hashlib.md5()
    for root, dirs, files in sorted(os.walk(data_dir)):
        for fname in sorted(files):
            if fname.endswith((".jpg",".png",".jpeg",".yaml",".yml",".txt")):
                fpath = os.path.join(root, fname)
                hasher.update(fpath.encode())
                hasher.update(str(os.path.getmtime(fpath)).encode())
    return hasher.hexdigest()

def check_cache(data_dir):
    if not os.path.exists(YOLO_MODEL_PATH): return False
    if not os.path.exists(CACHE_MANIFEST): return False
    try:
        with open(CACHE_MANIFEST) as f: manifest = json.load(f)
        model_arch = "yolo26m"
        entry = manifest.get("yolo",{})
        if (entry.get("dataset_hash") == get_dataset_hash(data_dir)
                and entry.get("model_arch") == model_arch):
            logger.info("YOLO model up to date: %s", YOLO_MODEL_PATH)
            return True
    except: pass
    return False

def update_cache(metrics, data_dir):
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    manifest = {}
    if os.path.exists(CACHE_MANIFEST):
        with open(CACHE_MANIFEST) as f: manifest = json.load(f)
    manifest["yolo"] = {"model_arch": "yolo26m", "dataset_hash": get_dataset_hash(data_dir), "mAP50": metrics.get("mAP50"), "mAP50_95": metrics.get("mAP50-95"), "timestamp": str(metrics.get("timestamp",""))}
    with open(CACHE_MANIFEST, "w") as f: json.dump(manifest, f, indent=2)

def train_yolo(data_yaml, epochs=300, imgsz=640, batch=8, workers=4):
    """Train YOLO26m on bakery dataset.

    Based on: He et al. 2019 "Rethinking ImageNet Pre-training" (from-scratch
    needs 3-5x more epochs); Bochkovskiy et al. 2020 "YOLOv4: Optimal Speed
    and Accuracy" (mosaic/mixup for small datasets); Ultralytics YOLOv8/v11
    technical reports (close_mosaic, multi-scale, label_smoothing).

    Key design decisions for bakery domain:
      - mosaic=0.0, mixup=0.0: bakery products on trays already have natural
        variation; mosaic creates unrealistic composites.
      - close_mosaic=15: per Ultralytics, last N epochs without mosaic-like
        augmentations for real-world convergence.
      - multi_scale=True: randomly rescales input [0.5, 1.5] each batch
        (Ge et al. 2021, YOLOX).
      - label_smoothing=0.1: reduces overfitting on small datasets
        (Szegedy et al. 2016).
    """
    import random as _random
    _random.seed(42)
    import numpy as _np; _np.random.seed(42)
    import torch; torch.manual_seed(42)

    # Validate dataset YAML
    import yaml
    with open(data_yaml) as f:
        ds = yaml.safe_load(f)
    if "train" not in ds or "val" not in ds:
        raise ValueError(f"Dataset YAML missing train/val paths: {data_yaml}")

    os.makedirs(os.path.dirname(YOLO_MODEL_PATH), exist_ok=True)
    data_dir = os.path.dirname(data_yaml)
    if check_cache(data_dir):
        return YOLO(YOLO_MODEL_PATH)

    logger.info("Training YOLO26m on %s, batch=%s, epochs=%s, workers=%s",
                data_yaml, batch, epochs, workers)

    model = YOLO("yolo26m.yaml")  # architecture only, from scratch
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="bakery_yolo",
        exist_ok=True,
        pretrained=False,          # He et al. 2019: from-scratch with 3x epochs
        workers=workers,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        warmup_epochs=5,
        weight_decay=0.0005,       # regularization for from-scratch training
        label_smoothing=0.1,       # Szegedy et al. 2016: reduces overfitting
        multi_scale=True,          # Ge et al. 2021 YOLOX: [0.5, 1.5] random rescale
        augment=True,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        degrees=15, translate=0.1, scale=0.3, shear=0.1,
        perspective=0.0, flipud=0.0, fliplr=0.5,
        mosaic=0.0, mixup=0.0,     # Bochkovskiy et al. 2020: disabled for bakery domain
        close_mosaic=15,           # Ultralytics: last 15 epochs without mosaic-like augs
        patience=20,
    )

    # Save best model
    best_path = os.path.join("runs", "detect", "bakery_yolo", "weights", "best.pt")
    if os.path.exists(best_path):
        shutil.copy(best_path, YOLO_MODEL_PATH)
        logger.info("Best model saved: %s", YOLO_MODEL_PATH)
    else:
        logger.warning("best.pt not found at %s", best_path)

    metrics = {
        "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
        "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
        "epochs_trained": epochs,
    }
    update_cache(metrics, data_dir)
    return model

def evaluate_yolo(model_path, data_yaml):
    """Evaluate trained model with per-class AP metrics.

    Returns dict with overall mAP + per-class AP for thesis analysis.
    """
    model = YOLO(model_path)
    results = model.val(data=data_yaml, split="test" if os.path.exists(
        os.path.join(os.path.dirname(data_yaml), "test", "images")) else "val")
    per_class = {}
    if results.box.ap_class_index is not None and results.names:
        for i, ap in enumerate(results.box.ap50):
            cls_name = results.names.get(results.box.ap_class_index[i], f"cls_{i}")
            per_class[cls_name] = round(float(ap), 4)

    metrics = {
        "mAP50": round(float(results.box.map50), 4),
        "mAP50_95": round(float(results.box.map), 4),
        "per_class_ap50": per_class,
    }
    logger.info("Evaluation: mAP@0.5=%.4f, mAP@0.5:0.95=%.4f",
                metrics["mAP50"], metrics["mAP50_95"])
    for cls_name, ap in per_class.items():
        logger.info("  %s: AP@0.5=%.4f", cls_name, ap)
    return metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-cache", action="store_true")
    args = parser.parse_args()
    model = train_yolo(args.data, args.epochs, batch=args.batch, workers=args.workers)
    metrics = evaluate_yolo(YOLO_MODEL_PATH, args.data)
    logger.info("mAP@0.5: %.4f, mAP@0.5:0.95: %.4f", metrics["mAP50"], metrics["mAP50_95"])
