import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import APIRouter, UploadFile, File, Query
from typing import Optional
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    YOLO_MODEL_PATH, YOLO_CONFIDENCE_THRESHOLD,
    TRAY_GREEN_THRESHOLD, TRAY_ORANGE_BLUE_MAX,
    TRAY_YELLOW_CHANNEL_MIN, TRAY_YELLOW_BLUE_MAX,
    TRAY_RED_CHANNEL_MIN, TRAY_RED_GREEN_MAX, TRAY_RED_BLUE_MAX,
    PRODUCT_TYPES,
)
from db.mysql_client import get_db, q
from models.schemas import (
    YOLOResult, DeductRequest, DeductResponse, ImageSearchResult,
)

router = APIRouter(prefix="/s1", tags=["Module 1 - Visual Perception"])

_model: Optional[YOLO] = None


# ======================================================================
# Model loading
# ======================================================================
def get_model() -> YOLO:
    global _model
    if _model is None:
        if not os.path.exists(YOLO_MODEL_PATH):
            raise FileNotFoundError(
                f"YOLO model not found: {YOLO_MODEL_PATH}. Run training first."
            )
        _model = YOLO(YOLO_MODEL_PATH)
    return _model


# ======================================================================
# Tray color detection
# ======================================================================
def _classify_color(b: float, g: float, r: float) -> str:
    """Classify BGR mean into tray colour label."""
    if g > TRAY_GREEN_THRESHOLD and g > r and g > b:
        return "green"
    if r > TRAY_YELLOW_CHANNEL_MIN and g > TRAY_YELLOW_CHANNEL_MIN and b < TRAY_YELLOW_BLUE_MAX:
        return "yellow"
    if r > TRAY_RED_CHANNEL_MIN and g < TRAY_RED_GREEN_MAX and b < TRAY_RED_BLUE_MAX:
        return "red"
    if r > 150 and g > 100 and b < TRAY_ORANGE_BLUE_MAX:
        return "orange"
    return "unknown"


def detect_tray_color(image: np.ndarray, bboxes: list) -> str:
    """Extract tray colour from pixels NOT occupied by detected products.

    Builds a mask that excludes all product bounding boxes (with padding),
    then computes mean colour only from the remaining tray-visible area.
    """
    if not bboxes:
        return "unknown"
    h, w = image.shape[:2]
    mask = np.ones((h, w), dtype=np.uint8) * 255
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad = 15
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        mask[y1:y2, x1:x2] = 0
    if cv2.countNonZero(mask) < 100:
        return "unknown"
    mean_color = cv2.mean(image, mask=mask)
    b, g, r = mean_color[0], mean_color[1], mean_color[2]
    return _classify_color(b, g, r)


# ======================================================================
# Product detection
# ======================================================================
def detect_products(image: np.ndarray) -> list[YOLOResult]:
    model = get_model()
    results = model(image, conf=YOLO_CONFIDENCE_THRESHOLD)
    detections = []
    all_bboxes = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            if cls_name in PRODUCT_TYPES:
                conf = float(box.conf[0])
                bbox = box.xyxy[0].tolist()
                all_bboxes.append(bbox)
                detections.append({
                    "product_name": cls_name,
                    "confidence": conf,
                    "bbox": bbox,
                })
    tray_color = detect_tray_color(image, all_bboxes) if all_bboxes else "unknown"
    results_list = []
    for d in detections:
        results_list.append(YOLOResult(
            product_name=d["product_name"],
            quantity=1,
            confidence=d["confidence"],
            bbox=d["bbox"],
            tray_color=tray_color,
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
                "confidence": 0.0,
                "tray_color": r.tray_color,
            }
        counts[key]["quantity"] += 1
        counts[key]["confidence"] += r.confidence
    for key in counts:
        counts[key]["confidence"] = round(
            counts[key]["confidence"] / counts[key]["quantity"], 3
        )
    return list(counts.values())


# ======================================================================
# POST /s1/checkout -- Outbound: recognize tray items + tray colour
# ======================================================================
@router.post("/checkout")
async def checkout_scan(file: UploadFile = File(...)):
    """Scan a customer tray at checkout.

    Returns detected products with tray colour so S4 can apply
    pricing (green = fresh price, orange = discount price).
    Inventory deduction happens separately via POST /s1/deduct
    after payment is confirmed by S4.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = detect_products(image)
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
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    results = detect_products(image)
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

    for item in req.items:
        product_name = item.get("product_name", "")
        qty_needed   = int(item.get("quantity", 0))

        if not product_name or qty_needed <= 0:
            errors.append(f"Invalid item: {item}")
            continue

        requested_freshness = item.get("freshness")

        # Fetch batches with remaining stock, FIFO order
        query = (
            q(db, "batch_inventory")
            .select("*")
            .eq("product_name", product_name)
            .gt("quantity", 0)
        )
        # If customer selected a specific freshness, prioritize matching batches
        if requested_freshness:
            query = query.eq("freshness_status", requested_freshness)
        batches = query.order("production_time", desc=False).execute()


        if not batches.data:
            errors.append(
                f"No stock available for '{product_name}' (needed {qty_needed})"
            )
            continue

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
                    quantity=r.get("quantity", r.get("quantity", 0)),
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
                quantity=r.get("quantity", r.get("quantity", 0)),
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
        "Day-2": "Day-2 Area",
        "Near-Expired": "Discount Area",
        "Expired": "Expired",
    }

    inventory = []
    for row in (r.data or []):
        item = dict(row)
        fs = item.get("freshness_status", "Fresh")
        item["sales_area"] = AREA_MAP.get(fs, "Fresh Area")
        inventory.append(item)

    return {"status": "ok", "inventory": inventory}