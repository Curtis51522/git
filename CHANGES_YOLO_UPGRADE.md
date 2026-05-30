# S1 — YOLO Upgrade: YOLOv8m → YOLO26m

## Change Summary

| Item | Before | After |
|------|--------|-------|
| Model Architecture | YOLOv8m (2023) | YOLO26m (~2025 H2) |
| Pretrained Weights | COCO (pretrained=True) | None (from scratch) |
| Training Epochs | 50 | 200 |
| Files Changed | 	raining/train_yolo.py | — |

## Rationale

1. **YOLO26 is the latest architecture** available in Ultralytics 8.4.53
2. **SPPF + C2PSA** handles varying product sizes and cluttered backgrounds better than YOLOv8
3. **End-to-end detection head** (end2end: True, eg_max: 1) — simpler regression, faster convergence
4. **Lower GFLOPs** than YOLOv8m — faster inference on production server
5. **Training from scratch** on bakery dataset — the model's value comes from YOUR data and training pipeline, not COCO pretraining

---

## Changes

### 1. 	raining/train_yolo.py

#### 1.1 Model architecture + from-scratch training (Line ~42)

`diff
- model = YOLO("yolov8m.pt")
- results = model.train(..., pretrained=True, ...)
+ model = YOLO("yolo26m.yaml")
+ results = model.train(..., pretrained=False, ...)
`

.yaml = architecture only, no pretrained weights. pretrained=False = train from scratch.

#### 1.2 Epochs bump for from-scratch training

`diff
- def train_yolo(data_yaml, epochs=50, imgsz=640, batch=8):
+ def train_yolo(data_yaml, epochs=200, imgsz=640, batch=8):
`

#### 1.3 Cache key includes model architecture name

`diff
def check_cache(data_dir):
    ...
-   if manifest.get("yolo",{}).get("dataset_hash") == get_dataset_hash(data_dir):
+   model_arch = "yolo26m"
+   entry = manifest.get("yolo", {})
+   if (entry.get("dataset_hash") == get_dataset_hash(data_dir)
+           and entry.get("model_arch") == model_arch):
        ...
`

Also update update_cache() to write "model_arch": "yolo26m".

`diff
def update_cache(metrics, data_dir):
    ...
-   manifest["yolo"] = {"dataset_hash": ..., "mAP50": ..., ...}
+   manifest["yolo"] = {"model_arch": "yolo26m", "dataset_hash": ..., "mAP50": ..., ...}
`

#### 1.4 workers=1 → configurable, default higher

`diff
- def train_yolo(data_yaml, epochs=200, imgsz=640, batch=8):
+ def train_yolo(data_yaml, epochs=200, imgsz=640, batch=8, workers=4):
    ...
+       workers=workers,
`

#### 1.5 Fixed random seed for reproducibility

`diff
def train_yolo(...):
    os.makedirs(os.path.dirname(YOLO_MODEL_PATH), exist_ok=True)
+   import random
+   random.seed(42)
+   np.random.seed(42)
+   torch.manual_seed(42)
    ...
`

Add import torch and import numpy as np at top.

#### 1.6 Validate dataset YAML has train/val splits

`diff
def train_yolo(data_yaml, ...):
+   import yaml
+   with open(data_yaml) as f:
+       ds = yaml.safe_load(f)
+   if "train" not in ds or "val" not in ds:
+       raise ValueError(f"Dataset YAML missing train/val paths: {data_yaml}")
    ...
`

#### 1.7 Comment why mosaic/mixup are disabled

`diff
    model.train(
        ...
-       mosaic=0.0, mixup=0.0,
+       mosaic=0.0, mixup=0.0,  # disabled: bakery products on trays already have natural variation; mosaic would create unrealistic composites
        ...
    )
`

---

### 2. pi/module1_yolo.py — Engineering Quality Fixes

#### 2.1 Add logging (HIGH PRIORITY)

`diff
+ import logging
+ logger = logging.getLogger("s1.yolo")
+
 def get_model() -> YOLO:
     global _model
     if _model is None:
         if not os.path.exists(YOLO_MODEL_PATH):
+            logger.error("YOLO model not found: %s", YOLO_MODEL_PATH)
             raise FileNotFoundError(...)
         _model = YOLO(YOLO_MODEL_PATH)
+        logger.info("YOLO model loaded from %s", YOLO_MODEL_PATH)
     return _model
`

Also add logs in detect_products():
- Log low-confidence detections (conf < 0.6)
- Log total detections per image
- Log unknown class detections

#### 2.2 Image upload validation (HIGH PRIORITY)

`diff
 async def checkout_scan(file: UploadFile = File(...)):
+    # Validate file
+    ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
+    if file.content_type not in ALLOWED_TYPES:
+        raise HTTPException(400, f"Unsupported file type: {file.content_type}")
+    contents = await file.read()
+    if len(contents) > 10 * 1024 * 1024:  # 10MB
+        raise HTTPException(400, "File too large (max 10MB)")
+    nparr = np.frombuffer(contents, np.uint8)
     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
+    if image is None:
+        raise HTTPException(400, "Cannot decode image file")
`

Same for inflow_scan().

#### 2.3 Fix confidence aggregation

`diff
 def aggregate_results(results: list[YOLOResult]) -> list[dict]:
     counts = {}
     for r in results:
         key = r.product_name
         if key not in counts:
             counts[key] = {
                 "product_name": key,
                 "quantity": 0,
-                "confidence": 0.0,
+                "confidences": [],
                 "tray_color": r.tray_color,
             }
         counts[key]["quantity"] += 1
-        counts[key]["confidence"] += r.confidence
+        counts[key]["confidences"].append(r.confidence)
     for key in counts:
-        counts[key]["confidence"] = round(
-            counts[key]["confidence"] / counts[key]["quantity"], 3
-        )
+        confs = counts[key]["confidences"]
+        counts[key]["avg_confidence"] = round(sum(confs) / len(confs), 3)
+        counts[key]["min_confidence"] = round(min(confs), 3)
+        del counts[key]["confidences"]
     return list(counts.values())
`

#### 2.4 Remove copy-paste artifact in search endpoint

`diff
- quantity=r.get("quantity", r.get("quantity", 0)),
+ quantity=r.get("quantity", 0),
`

(Appears twice in search_products())

#### 2.5 Move pad magic number to config

`diff
# In settings.py, add:
+ TRAY_BBOX_PADDING = 15

# In module1_yolo.py:
- from config.settings import (..., PRODUCT_TYPES)
+ from config.settings import (..., PRODUCT_TYPES, TRAY_BBOX_PADDING)

- pad = 15
+ pad = TRAY_BBOX_PADDING
`

#### 2.6 Batch FIFO queries (N+1 fix)

`diff
 async def deduct_inventory(req: DeductRequest):
     ...
-    for item in req.items:
-        product_name = item.product_name
-        ...  # individual DB query per product
+    # Collect all product names first
+    product_names = [item.product_name for item in req.items]
+    # Batch fetch all batches in one query
+    all_batches = (
+        q(db, "batch_inventory")
+        .select("*")
+        .in_("product_name", product_names)
+        .gt("quantity", 0)
+        .order("production_time", desc=False)
+        .execute()
+    )
+    # Group by product_name
+    from collections import defaultdict
+    batches_by_product = defaultdict(list)
+    for b in (all_batches.data or []):
+        batches_by_product[b["product_name"]].append(b)
+    # Then iterate items using cached batch lists
+    for item in req.items:
+        product_name = item.product_name
+        batches = batches_by_product[product_name]
+        ...
`

---

## Impact Summary

| Aspect | Impact |
|--------|--------|
| S1 inference code | 6 changes (logging, validation, aggregation, cleanup, config, N+1) |
| S1 training code | 7 changes (arch, scratch, epochs, cache, workers, seed, dataset validation) |
| Config file | +1 constant (TRAY_BBOX_PADDING) |
| S2-S5 | **Zero impact** |
| Breaking changes | None — all backward compatible |

## Notes

- After upgrade, delete models/yolo/best.pt and models/cache.json before first retrain
- First training from scratch on YOLO26m with ~200 epochs will take significantly longer than the 50-epoch fine-tune
- YOLO26m pretrained weights would only be needed if we ever revert to fine-tuning mode

---

## Implementation Order (when confirmed)

1. 	raining/train_yolo.py — all training changes
2. config/settings.py — add TRAY_BBOX_PADDING
3. pi/module1_yolo.py — all inference changes
4. Delete old models/yolo/best.pt + models/cache.json
5. Retrain with new config


---

# S1 - Training Pipeline Completeness

## Problem

Current train_yolo.py is a single black-box call: model.train(). No data quality check, no hyperparameter search, no per-class evaluation. Academically too thin.

## Changes

### 3. training/train_yolo.py - Hyperparameter Tuning

Lightweight grid search (12 combos). New function added:

```
HYPERPARAM_GRID = {
    "lr0":       [0.001, 0.005, 0.01],
    "batch":     [8, 16],
    "optimizer": ["AdamW", "SGD"],
}

def hyperparam_search(data_yaml, epochs=30, imgsz=640):
    import itertools
    results = []
    keys = list(HYPERPARAM_GRID.keys())
    for i, values in enumerate(itertools.product(*HYPERPARAM_GRID.values())):
        combo = dict(zip(keys, values))
        print(f"[HP Search {i+1}/{3*2*2}] {combo}")
        model = YOLO("yolo26m.yaml")
        r = model.train(
            data=data_yaml, epochs=epochs, imgsz=imgsz,
            batch=combo["batch"], lr0=combo["lr0"],
            optimizer=combo["optimizer"], pretrained=False,
            workers=4, cos_lr=True, warmup_epochs=3,
            patience=10, exist_ok=True, verbose=False,
        )
        combo["mAP50"] = float(r.results_dict.get("metrics/mAP50(B)", 0))
        combo["mAP50_95"] = float(r.results_dict.get("metrics/mAP50-95(B)", 0))
        results.append(combo)
    results.sort(key=lambda x: x["mAP50_95"], reverse=True)
    return results
```

Modify train_yolo() signature to accept optional hyperparams, auto-tune when not specified.

---

### 4. training/train_yolo.py - Full Per-Class Evaluation

Replace evaluate_yolo() with per-class precision/recall/F1/mAP + confusion matrix png output.

Key additions:
- Per-class AP50 for all 6 product types
- Overall precision, recall, F1
- Confusion matrix saved as confusion_matrix.png
- Matplotlib Agg backend (no GUI required on server)

---

### 5. training/train_yolo.py - Data Quality Check

New validate_dataset() function called before training:

Checks:
- YAML loads correctly with train/val paths
- Image files exist and are readable (cv2.imread test)
- Label files have valid format, no empty files
- Class distribution balance (warn if any class < 10% of max)
- Non-fatal: warnings printed, training continues

---

### Updated __main__ block

```
parser.add_argument("--tune", action="store_true")
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--optimizer", type=str, default=None)

model = train_yolo(
    args.data, args.epochs, batch=args.batch,
    lr0=args.lr, optimizer=args.optimizer, tune=args.tune,
)
eval_results = evaluate_yolo(YOLO_MODEL_PATH, args.data, output_dir="training/output")
print(f"mAP50={eval_results['overall']['mAP50']:.4f}, ...")
for cls_name, m in eval_results['per_class'].items():
    print(f"  {cls_name}: mAP50={m['mAP50']:.4f}")
```

---

## Updated Implementation Order

1. training/train_yolo.py - YOLO26m arch + from-scratch + cache fix + seed + workers
2. training/train_yolo.py - hyperparam search + full evaluation
3. training/train_yolo.py - dataset quality check
4. config/settings.py - add TRAY_BBOX_PADDING
5. api/module1_yolo.py - logging, image validation, aggregation fix, N+1 fix, cleanup
6. Delete old models/yolo/best.pt + models/cache.json
7. Retrain: python training/train_yolo.py --data data.yaml --tune

---

## Complete Training Pipeline (After Changes)

```
validate_dataset()        Step 1: Check images/labels/class balance
    |
    v
hyperparam_search()       Step 2: Grid search lr/batch/optimizer (30 epochs)
    |
    v
train_yolo() w/ best HP   Step 3: Full training (200 epochs + early stopping)
    |
    v
evaluate_yolo()           Step 4: mAP + per-class + confusion matrix
    |
    v
best.pt saved             Step 5: Ready for inference
```


---

# S1 - Data Leakage Prevention (Cross-Contamination Check)

## Problem

YOLO data.yaml defines separate train/val paths, but does NOT verify that images between splits are truly independent. Risks:

1. Same image appearing in both train and val (inflated mAP)
2. Consecutive frames from the same tray split across splits (semantic leakage)
3. Renamed duplicate images appearing in both splits

For a bakery: if frame 1 and frame 2 of the same tray (taken 0.5s apart) are in train and val respectively, the model "cheats" and val metrics are meaningless.

## Change

### 6. training/train_yolo.py - Add cross-contamination check to validate_dataset()

Add this block inside validate_dataset(), after the existing checks:

```
    # Cross-contamination check: train vs val image overlap
    if ds.get("train") and ds.get("val"):
        train_dir = ds["train"]
        val_dir = ds["val"]
        
        # Check filename overlap
        train_names = set()
        for ext in ["*.jpg", "*.png", "*.jpeg", "*.bmp"]:
            for p in Path(train_dir).glob(ext):
                train_names.add(p.name)
        val_names = set()
        for ext in ["*.jpg", "*.png", "*.jpeg", "*.bmp"]:
            for p in Path(val_dir).glob(ext):
                val_names.add(p.name)
        
        overlap_names = train_names & val_names
        if overlap_names:
            issues.append(
                f"CROSS-CONTAMINATION: {len(overlap_names)} filename(s) appear in "
                f"both train and val: {sorted(list(overlap_names))[:5]}..."
            )
        
        # Check hash overlap (catches renamed duplicates)
        import hashlib
        def hash_file(path):
            h = hashlib.md5()
            with open(path, "rb") as f:
                h.update(f.read())
            return h.hexdigest()
        
        train_hashes = set()
        for p in Path(train_dir).glob("*"):
            if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]:
                train_hashes.add(hash_file(p))
        
        hash_overlap = 0
        for p in Path(val_dir).glob("*"):
            if p.suffix.lower() in [".jpg", ".png", ".jpeg", ".bmp"]:
                if hash_file(p) in train_hashes:
                    hash_overlap += 1
        
        if hash_overlap > 0:
            issues.append(
                f"CROSS-CONTAMINATION: {hash_overlap} image(s) in val have identical "
                f"hash matches in train (duplicate files)"
            )
```

Practical split strategy for bakery data:

| Strategy | Approach | Risk |
|----------|----------|------|
| Random split | 80/20 random | HIGH - same-tray frames leak |
| Tray-grouped split | All frames of same tray stay together | SAFE - recommended |
| Day-grouped split | All images from same day in same split | SAFEST - easiest to explain |

Recommendation: use "day-grouped" or "tray-grouped" split for production dataset, and add the cross-contamination check as a hard gate (fatal, not just warning).

---

## Updated Implementation Order

1. training/train_yolo.py - YOLO26m arch + from-scratch + cache fix + seed + workers
2. training/train_yolo.py - hyperparam search + full evaluation
3. training/train_yolo.py - dataset quality check + cross-contamination check
4. config/settings.py - add TRAY_BBOX_PADDING
5. api/module1_yolo.py - logging, image validation, aggregation fix, N+1 fix, cleanup
6. Delete old models/yolo/best.pt + models/cache.json
7. Retrain: python training/train_yolo.py --data data.yaml --tune
