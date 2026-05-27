import os, json, hashlib
from pathlib import Path
from ultralytics import YOLO
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import YOLO_MODEL_PATH, MODEL_CACHE_DIR, CACHE_MANIFEST, PRODUCT_TYPES

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
        if manifest.get("yolo",{}).get("dataset_hash") == get_dataset_hash(data_dir):
            print(f"[Cache] YOLO model up to date: {YOLO_MODEL_PATH}")
            return True
    except: pass
    return False

def update_cache(metrics, data_dir):
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    manifest = {}
    if os.path.exists(CACHE_MANIFEST):
        with open(CACHE_MANIFEST) as f: manifest = json.load(f)
    manifest["yolo"] = {"dataset_hash": get_dataset_hash(data_dir), "mAP50": metrics.get("mAP50"), "mAP50_95": metrics.get("mAP50-95"), "timestamp": str(metrics.get("timestamp",""))}
    with open(CACHE_MANIFEST, "w") as f: json.dump(manifest, f, indent=2)

def train_yolo(data_yaml, epochs=50, imgsz=640, batch=8):
    os.makedirs(os.path.dirname(YOLO_MODEL_PATH), exist_ok=True)
    data_dir = os.path.dirname(data_yaml)
    if check_cache(data_dir):
        return YOLO(YOLO_MODEL_PATH)
    print(f"[YOLO] Training YOLOv8m on {data_yaml}, batch={batch}, epochs={epochs}")
    model = YOLO("yolov8m.pt")
    results = model.train(
        data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
        name="bakery_yolo", exist_ok=True, pretrained=True,
        workers=1, optimizer="AdamW", lr0=0.001, cos_lr=True,
        warmup_epochs=5, augment=True,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        degrees=15, translate=0.1, scale=0.3, shear=0.1,
        perspective=0.0, flipud=0.0, fliplr=0.5,
        mosaic=0.0, mixup=0.0, patience=20,
    )
    best_path = os.path.join("runs","detect","bakery_yolo","weights","best.pt")
    if os.path.exists(best_path):
        import shutil
        shutil.copy(best_path, YOLO_MODEL_PATH)
        print(f"[YOLO] Best model saved: {YOLO_MODEL_PATH}")
    metrics = {"mAP50": float(results.results_dict.get("metrics/mAP50(B)",0)), "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)",0))}
    update_cache(metrics, data_dir)
    return model

def evaluate_yolo(model_path, data_yaml):
    model = YOLO(model_path)
    results = model.val(data=data_yaml)
    return {"mAP50": float(results.box.map50), "mAP50_95": float(results.box.map)}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--skip-cache", action="store_true")
    args = parser.parse_args()
    model = train_yolo(args.data, args.epochs, batch=args.batch)
    metrics = evaluate_yolo(YOLO_MODEL_PATH, args.data)
    print(f"mAP@0.5: {metrics['mAP50']:.4f}, mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
