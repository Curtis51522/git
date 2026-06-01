import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Optional
from datetime import datetime
import time
import sys, os
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    YOLO_MODEL_PATH, YOLO_CONFIDENCE_THRESHOLD,
    PRODUCT_TYPES,
)
from db.mysql_client import get_db, q

from models.schemas import (
    YOLOResult, DeductRequest, DeductResponse, ImageSearchResult,
)

logger = logging.getLogger("s1.yolo")

router = APIRouter(prefix="/s1", tags=["Module 1 - Visual Perception"])

_model: Optional[YOLO] = None


# ======================================================================
# Model loading
# ======================================================================
def get_model() -> YOLO:
    global _model
    if _model is None:
        if not os.path.exists(YOLO_MODEL_PATH):
            logger.error("YOLO model not found: %s", YOLO_MODEL_PATH)
            raise FileNotFoundError(
                f"YOLO model not found: {YOLO_MODEL_PATH}. Run training first."
            )
        _model = YOLO(YOLO_MODEL_PATH)
        logger.info("YOLO model loaded from %s", YOLO_MODEL_PATH)
    return _model


# ======================================================================
# Product detection
# ======================================================================
def detect_products(image: np.ndarray, scenario: str = "checkout", image_id: str = "") -> list[YOLOResult]:
    model = get_model()
    model_version = os.path.basename(YOLO_MODEL_PATH).replace(".pt", "")
    t0 = time.time()
    results = model(image, conf=YOLO_CONFIDENCE_THRESHOLD)
    inference_time = round(time.time() - t0, 4)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            if cls_name in PRODUCT_TYPES:
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                manual_check = conf < 0.6
                if manual_check:
                    logger.warning("Low confidence detection: %s (%.2f)", cls_name, conf)
                detections.append({
                    "product_name": cls_name,
                    "confidence": conf,
                    "bbox": bbox,
                    "manual_check_required": manual_check,
                })
    logger.info("Detected %d products across %d unique types", len(detections), len(set(d["product_name"] for d in detections)))

    # Write detection_log records
    try:
        db = get_db()
        for d in detections:
            q(db, "detection_log").insert({
                "model_version": model_version,
                "image_id": image_id,
                "scenario": scenario,
                "predicted_class": d["product_name"],
                "bbox": str(d["bbox"]),
                "confidence": d["confidence"],
                "inference_time": inference_time,
                "manual_check_required": 1 if d["manual_check_required"] else 0,
                "error_type": "none",
            }).execute()
    except Exception as e:
        logger.warning("Failed to write detection_log: %s", e)

    results_list = []
    for d in detections:
        results_list.append(YOLOResult(
            product_name=d["product_name"],
            quantity=1,
            confidence=d["confidence"],
            bbox=d["bbox"],
            tray_color="green",
        ))
    return results_list


def aggregate_results(results: list[YOLOResult]) -> list[dict]:
    counts = {}
    for r in results:
        key = r.product_name
        if key not in counts:
            counts[key] = {
                "product_name": key,
                "quantity": 0,
                "confidences": [],
                "tray_color": r.tray_color,
            }
        counts[key]["quantity"] += 1
        counts[key]["confidences"].append(r.confidence)
    for key in counts:
        confs = counts[key]["confidences"]
        counts[key]["avg_confidence"] = round(sum(confs) / len(confs), 3)
        counts[key]["min_confidence"] = round(min(confs), 3)
        counts[key]["confidence"] = counts[key]["avg_confidence"]  # backward compat
        del counts[key]["confidences"]
    return list(counts.values())



# ======================================================================
# POST /s1/checkout -- Outbound: recognize tray items + tray colour
# ======================================================================
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

@router.post("/checkout")
async def checkout_scan(file: UploadFile = File(...)):
    """Scan a customer tray at checkout.

    Returns detected products with tray colour so S4 can apply
    pricing (green = fresh price, orange = discount price).
    Inventory deduction happens separately via POST /s1/deduct
    after payment is confirmed by S4.
    """
    # Validate before reading stream
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"File too large ({len(contents)} bytes, max 10MB)")
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Cannot decode image")
    logger.info("Checkout scan: %dx%d", image.shape[1], image.shape[0])
    img_id = f"checkout_{datetime.now().strftime("%Y%m%d_%H%M%S")}"
    results = detect_products(image, scenario="checkout", image_id=img_id)
    aggregated = aggregate_results(results)
    return {"status": "ok", "detections": aggregated}


# ======================================================================
# POST /s1/inflow -- Inbound: batch production intake
# ======================================================================
@router.post("/inflow")
async def inflow_scan(file: UploadFile = File(...)):
    """Scan a tray of freshly-baked goods.  Writes batch_inventory
    and inventory_transactions rows.

    Each unique product in the image gets its own batch record with
    ``quantity`` == ``quantity``.
    """
    # Validate before reading stream
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"File too large ({len(contents)} bytes, max 10MB)")
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Cannot decode image")
    logger.info("Inflow scan: %dx%d", image.shape[1], image.shape[0])
    img_id = f"inbound_{datetime.now().strftime("%Y%m%d_%H%M%S")}"
    results = detect_products(image, scenario="inbound", image_id=img_id)
    aggregated = aggregate_results(results)

    db = get_db()
    batch_prefix = datetime.now().strftime("%Y%m%d%H%M%S")
    created = []

    for item in aggregated:
        batch_id = f"BATCH_{batch_prefix}_{item['product_name'].replace(' ', '_')}"
        qty = item["quantity"]

        # -- batch_inventory row ----------------------------------------
        q(db, "batch_inventory").insert({
            "batch_id": batch_id,
            "product_name": item["product_name"],
            "quantity": qty,
            "production_time": datetime.now().isoformat(),
            "freshness_status": "Fresh",
            "tray_color": item.get("tray_color"),
        }).execute()

        # -- inventory_transactions row ---------------------------------
        q(db, "inventory_transactions").insert({
            "transaction_type": "inflow",
            "batch_id": batch_id,
            "product_name": item["product_name"],
            "quantity": qty,
            "freshness_status": "Fresh",
        }).execute()

        created.append({
            "batch_id": batch_id,
            "product_name": item["product_name"],
            "quantity": qty,
        })

    return {"status": "ok", "batches_created": len(created), "batches": created}


# ======================================================================

# POST /s1/inflow/batch -- Confirm inflow from frontend (no re-scan)
# ======================================================================
@router.post("/inflow/batch")
async def inflow_batch(req: DeductRequest):
    """Called after the user confirms detected items in the inflow UI.
    Creates batch_inventory records without re-scanning the image.
    """
    db = get_db()
    batch_prefix = datetime.now().strftime("%Y%m%d%H%M%S")
    created = []

    for item in req.items:
        product_name = item.get("product_name", "")
        qty = int(item.get("quantity", 0))
        if not product_name or qty <= 0:
            continue

        batch_id = f"BATCH_{batch_prefix}_{product_name.replace(' ', '_')}"
        now_iso = datetime.now().isoformat()

        q(db, "batch_inventory").insert({
            "batch_id": batch_id,
            "product_name": product_name,
            "quantity": qty,
            "production_time": now_iso,
            "freshness_status": "Fresh",
            "tray_color": item.get("tray_color", "green"),
        }).execute()

        q(db, "inventory_transactions").insert({
            "transaction_type": "inflow",
            "batch_id": batch_id,
            "product_name": product_name,
            "quantity": qty,
            "freshness_status": "Fresh",
        }).execute()

        created.append({
            "batch_id": batch_id,
            "product_name": product_name,
            "quantity": qty,
        })

    return {"status": "ok", "batches_created": len(created), "batches": created}


# POST /s1/deduct -- Outbound: deduct inventory after payment
# ======================================================================
@router.post("/deduct", response_model=DeductResponse)
async def deduct_inventory(req: DeductRequest):
    """Called by S4 after a successful payment.

    Deducts ``quantity`` from batch_inventory using FIFO
    (oldest production_time first), and writes outflow transactions.

    Returns per-item deduction details and any errors (e.g. insufficient
    stock for a product).
    """
    db = get_db()
    deducted = []
    errors = []

    # Collect all product names for a single batch query (N+1 fix)
    product_names = list(set(
        item.get("product_name", "")
        for item in req.items
        if item.get("product_name", "") and int(item.get("quantity", 0)) > 0
    ))
    if not product_names:
        return DeductResponse(status="ok", deducted=[], errors=["No valid items"])

    # Single query: fetch all batches for all products at once (raw SQL for IN clause)
    placeholders = ", ".join(["%s"] * len(product_names))
    sql = f"SELECT * FROM batch_inventory WHERE product_name IN ({placeholders}) AND quantity > 0 ORDER BY production_time ASC"
    cur = db.cursor(dictionary=True)
    cur.execute(sql, tuple(product_names))
    all_batches_data = cur.fetchall()
    cur.close()
    # Wrap to match .data API
    class AllBatches:
        data = all_batches_data
    all_batches = AllBatches()

    # Group batches by product_name for O(1) lookup
    batches_by_product = {}
    for b in (all_batches.data or []):
        pn = b["product_name"]
        if pn not in batches_by_product:
            batches_by_product[pn] = []
        batches_by_product[pn].append(b)

    for item in req.items:
        product_name = item.get("product_name", "")
        qty_needed   = int(item.get("quantity", 0))

        if not product_name or qty_needed <= 0:
            errors.append(f"Invalid item: {item}")
            continue

        requested_freshness = item.get("freshness")

        # Look up from pre-fetched map (N+1 eliminated)
        batches_data = batches_by_product.get(product_name, [])
        if requested_freshness:
            batches_data = [b for b in batches_data if b.get("freshness_status") == requested_freshness]
            # Re-sort by production_time to maintain FIFO after filtering
            batches_data.sort(key=lambda b: b.get("production_time", ""))

        if not batches_data:
            errors.append(
                f"No stock available for '{product_name}' (needed {qty_needed})"
            )
            continue
        
        # Wrap in a simple object to match the old .data API
        class BatchResult:
            data = batches_data
        batches = BatchResult()

        remaining_to_deduct = qty_needed

        for batch in batches.data:
            if remaining_to_deduct <= 0:
                break

            available = batch["quantity"]
            take = min(available, remaining_to_deduct)
            new_remaining = available - take

            # Update batch_inventory
            q(db, "batch_inventory").update({
                "quantity": new_remaining,
            }).eq("batch_id", batch["batch_id"]).execute()

            # Write outflow transaction
            q(db, "inventory_transactions").insert({
                "transaction_type": "outflow",
                "batch_id": batch["batch_id"],
                "product_name": product_name,
                "quantity": take,
                "unit_price": req.unit_price,
                "discount_applied": req.discount_applied,
                "freshness_status": batch.get("freshness_status"),
            }).execute()

            deducted.append({
                "product_name": product_name,
                "batch_id": batch["batch_id"],
                "quantity_deducted": take,
                "remaining_after": new_remaining,
            })

            remaining_to_deduct -= take

        if remaining_to_deduct > 0:
            errors.append(
                f"Insufficient stock for '{product_name}': "
                f"short by {remaining_to_deduct}"
            )

    return DeductResponse(
        status="partial" if errors else "ok",
        deducted=deducted,
        errors=errors,
    )


# ======================================================================
# GET /s1/search -- Keyword-based image / product search
# ======================================================================
@router.get("/search")
async def search_products(
    q: str = Query(..., description="Search keyword (product name or batch ID)"),
):
    """Look up product batches by keyword.

    Searches batch_inventory by product_name (partial match) or batch_id.
    In production this would be augmented with pgvector similarity search
    on product embedding vectors.
    """
    db = get_db()

    # Try exact batch_id match first
    batch_result = (
        q(db, "batch_inventory")
        .select("*")
        .eq("batch_id", q)
        .execute()
    )

    if batch_result.data:
        return {
            "status": "ok",
            "match_type": "batch_id",
            "results": [
                ImageSearchResult(
                    product_name=r["product_name"],
                    batch_id=r["batch_id"],
                    quantity=r.get("quantity", 0),
                    freshness_status=r.get("freshness_status", "Fresh"),
                    sales_area=r.get("sales_area", "Fresh Area"),
                    production_time=r.get("production_time", ""),
                ).model_dump()
                for r in batch_result.data
            ],
        }

    # Fallback: partial product_name search
    name_result = (
        q(db, "batch_inventory")
        .select("*")
        .ilike("product_name", f"%{q}%")
        .gt("quantity", 0)
        .order("production_time", desc=False)
        .limit(20)
        .execute()
    )

    return {
        "status": "ok",
        "match_type": "product_name",
        "results": [
            ImageSearchResult(
                product_name=r["product_name"],
                batch_id=r["batch_id"],
                quantity=r.get("quantity", 0),
                freshness_status=r.get("freshness_status", "Fresh"),
                sales_area=r.get("sales_area", "Fresh Area"),
                production_time=r.get("production_time", ""),
            ).model_dump()
            for r in (name_result.data or [])
        ],
    }


# ======================================================================
# GET /s1/batch_inventory -- Full inventory snapshot
# ======================================================================
@router.get("/batch_inventory")
async def get_batch_inventory():
    db = get_db()
    r = q(db, "batch_inventory").select("*").execute()

    AREA_MAP = {
        "Fresh": "Fresh Area",
        "Day-1": "Day-1 Area",
        "Expired": "Expired",
    }

    inventory = []
    for row in (r.data or []):
        item = dict(row)
        fs = item.get("freshness_status", "Fresh")
        item["sales_area"] = AREA_MAP.get(fs, "Fresh Area")
        inventory.append(item)

    return {"status": "ok", "inventory": inventory}
# ======================================================================
# GET /s1/detection_logs -- Query AI detection logs
# ======================================================================
@router.get("/detection_logs")
async def get_detection_logs(
    scenario: str = Query(None, description="Filter: inbound / checkout"),
    limit: int = Query(50, description="Max records to return"),
):
    """Return recent AI detection logs for model evaluation (mAP, Precision, Recall, etc.)."""
    db = get_db()
    query = q(db, "detection_log").select("*").order("created_at", desc=True).limit(limit)
    if scenario:
        query = q(db, "detection_log").select("*").eq("scenario", scenario).order("created_at", desc=True).limit(limit)
    r = query.execute()
    return {"status": "ok", "count": len(r.data), "logs": r.data}
# ======================================================================
# GET /s1/inventory_transactions -- Query all inventory transactions
# ======================================================================
@router.get("/inventory_transactions")
async def get_inventory_transactions(
    product_name: str = Query(None),
    transaction_type: str = Query(None),
    limit: int = Query(100, description="Max records"),
):
    """Return inventory transaction history for S2/S3/S5 consumption."""
    db = get_db()
    qb = q(db, "inventory_transactions").select("*").order("transaction_time", desc=True).limit(limit)
    if product_name:
        qb = q(db, "inventory_transactions").select("*").eq("product_name", product_name).order("transaction_time", desc=True).limit(limit)
    if transaction_type:
        qb = q(db, "inventory_transactions").select("*").eq("transaction_type", transaction_type).order("transaction_time", desc=True).limit(limit)
    r = qb.execute()
    return {"status": "ok", "count": len(r.data), "transactions": r.data}


# ======================================================================
# GET /s1/inventory -- Aggregated inventory summary by product
# ======================================================================
@router.get("/inventory")
async def get_inventory_summary():
    """Return current inventory aggregated by product_name (total qty per product)."""
    db = get_db()
    r = q(db, "batch_inventory").select("*").gt("quantity", 0).execute()
    summary = {}
    for row in (r.data or []):
        pn = row.get("product_name", "unknown")
        qty = row.get("quantity", 0)
        if pn not in summary:
            summary[pn] = {"product_name": pn, "total_quantity": 0, "batches": 0}
        summary[pn]["total_quantity"] += qty
        summary[pn]["batches"] += 1
    return {"status": "ok", "inventory": list(summary.values())}

