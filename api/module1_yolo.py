import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from typing import Optional
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
import time
import sys, os
import logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    YOLO_MODEL_PATH, YOLO_CONFIDENCE_THRESHOLD,
    PRODUCT_TYPES,
)
from db.mysql_client import get_db, q
from api.auth import get_current_user, require_manager
from api.operation_clock import operation_now

from models.schemas import (
    YOLOResult, DeductRequest, DeductResponse, ImageSearchResult,
    is_positive_integer,
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

@router.post("/checkout", dependencies=[Depends(get_current_user)])
async def checkout_scan(file: UploadFile = File(...)):
    """Scan a customer tray at checkout.

    Returns detected products with tray colour so S4 can apply
    pricing (green = fresh price, orange = discount price).
    Inventory deduction is performed inside the S4 payment transaction.
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
    img_id = f"checkout_{operation_now(datetime.now).strftime('%Y%m%d_%H%M%S')}"
    results = detect_products(image, scenario="checkout", image_id=img_id)
    aggregated = aggregate_results(results)
    return {"status": "ok", "detections": aggregated}


# ======================================================================
# POST /s1/inflow -- Inbound: batch production intake
# ======================================================================
def _consume_production_materials(db, items, reference):
    quantities_by_product = {}
    for item in items:
        product_name = str(item.get("product_name", "")).strip()
        quantity = item.get("quantity")
        if not product_name:
            raise HTTPException(400, "Product name is required")
        if not is_positive_integer(quantity):
            raise HTTPException(
                400,
                f"Invalid quantity for '{product_name}': expected a positive integer",
            )
        quantities_by_product[product_name] = (
            quantities_by_product.get(product_name, 0) + quantity
        )

    product_names = sorted(quantities_by_product)
    if not product_names:
        return

    placeholders = ",".join(["%s"] * len(product_names))
    cursor = db.cursor()
    try:
        cursor.execute(
            f"""
            SELECT p.product_name, p.category, p.wastage_pct,
                   pr.material_name, pr.quantity_per_unit,
                   rm.stock_quantity, rm.unit, rm.category,
                   rm.track_inventory
            FROM products p
            LEFT JOIN product_recipes pr
              ON pr.product_name = p.product_name
            LEFT JOIN raw_materials rm
              ON rm.material_name = pr.material_name
            WHERE p.product_name IN ({placeholders})
            ORDER BY p.product_name, pr.material_name
            FOR UPDATE
            """,
            product_names,
        )

        known_products = set()
        recipe_products = set()
        requirements = {}
        material_units = {}
        material_stock = {}
        material_tracking = {}

        for row in cursor.fetchall():
            (
                product_name,
                product_category,
                wastage_pct,
                material_name,
                quantity_per_unit,
                stock_quantity,
                unit,
                material_category,
                track_inventory,
            ) = row
            known_products.add(product_name)
            if not product_category:
                raise HTTPException(
                    409,
                    f"Missing product category: {product_name}",
                )
            if str(product_category).lower() == "beverage":
                raise HTTPException(
                    409,
                    f"Beverage cannot be added as a baked batch: {product_name}",
                )
            if not material_name:
                continue
            if (
                stock_quantity is None
                or not unit
                or not material_category
                or track_inventory is None
            ):
                raise HTTPException(
                    409,
                    f"Missing raw material stock: {material_name}",
                )
            if wastage_pct is None:
                raise HTTPException(
                    409,
                    f"Missing wastage rate: {product_name}",
                )
            try:
                recipe_quantity = Decimal(str(quantity_per_unit))
                wastage_rate = Decimal(str(wastage_pct))
                available_stock = Decimal(str(stock_quantity))
            except (InvalidOperation, TypeError, ValueError) as exc:
                raise HTTPException(
                    409,
                    f"Invalid production material data: {product_name}",
                ) from exc
            if (
                not recipe_quantity.is_finite()
                or recipe_quantity <= 0
                or not wastage_rate.is_finite()
                or wastage_rate < 0
                or not available_stock.is_finite()
                or available_stock < 0
            ):
                raise HTTPException(
                    409,
                    f"Invalid production material data: {product_name}",
                )

            recipe_products.add(product_name)
            tracked = bool(track_inventory)
            required = recipe_quantity * Decimal(
                quantities_by_product[product_name]
            )
            if (
                tracked
                and str(unit).lower() != "pcs"
                and str(material_category).lower() != "packaging"
            ):
                required *= Decimal("1") + wastage_rate
            required = required.quantize(Decimal("0.000001"))
            requirements[material_name] = requirements.get(
                material_name,
                Decimal("0"),
            ) + required
            material_units[material_name] = unit
            material_stock[material_name] = available_stock
            existing_tracking = material_tracking.get(material_name)
            if existing_tracking is not None and existing_tracking != tracked:
                raise HTTPException(
                    409,
                    f"Inconsistent inventory tracking: {material_name}",
                )
            material_tracking[material_name] = tracked

        missing_products = sorted(set(product_names) - known_products)
        if missing_products:
            raise HTTPException(
                409,
                f"Missing product record: {', '.join(missing_products)}",
            )
        missing_recipes = sorted(set(product_names) - recipe_products)
        if missing_recipes:
            raise HTTPException(
                409,
                f"Missing product recipe: {', '.join(missing_recipes)}",
            )

        shortages = []
        for material_name in sorted(requirements):
            if not material_tracking[material_name]:
                continue
            required = requirements[material_name]
            available = material_stock[material_name]
            if available < required:
                shortages.append(
                    f"{material_name} (need {required}, available {available})"
                )
        if shortages:
            raise HTTPException(
                409,
                f"Insufficient production material stock: {'; '.join(shortages)}",
            )

        transaction_reference = f"production:{reference}"
        for material_name in sorted(requirements):
            required = requirements[material_name]
            if material_tracking[material_name]:
                cursor.execute(
                    """
                    UPDATE raw_materials
                    SET stock_quantity = stock_quantity - %s
                    WHERE material_name = %s AND stock_quantity >= %s
                    """,
                    (required, material_name, required),
                )
                if cursor.rowcount != 1:
                    raise HTTPException(
                        409,
                        f"Material stock changed during production: {material_name}",
                    )
            cursor.execute(
                """
                INSERT INTO material_transactions
                    (material_name, transaction_type, quantity, unit, reference)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    material_name,
                    "outflow",
                    required,
                    material_units[material_name],
                    transaction_reference,
                ),
            )
    finally:
        cursor.close()


def _create_production_batches(items):
    for item in items:
        product_name = str(item.get("product_name", "")).strip()
        if not product_name:
            raise HTTPException(400, "Product name is required")
        if not is_positive_integer(item.get("quantity")):
            raise HTTPException(
                400,
                f"Invalid quantity for '{product_name}': expected a positive integer",
            )

    now = operation_now(datetime.now)
    batch_prefix = now.strftime("%Y%m%d%H%M%S%f")
    db = get_db(autocommit=False)
    created = []
    try:
        _consume_production_materials(db, items, batch_prefix)
        for item in items:
            product_name = str(item["product_name"]).strip()
            quantity = item["quantity"]
            batch_id = f"BATCH_{batch_prefix}_{product_name.replace(' ', '_')}"

            q(db, "batch_inventory").insert({
                "batch_id": batch_id,
                "product_name": product_name,
                "quantity": quantity,
                "quantity_initial": quantity,
                "quantity_remaining": quantity,
                "production_time": now.isoformat(),
                "freshness_status": "Fresh",
                "tray_color": item.get("tray_color", "green"),
            }).execute()
            q(db, "inventory_transactions").insert({
                "transaction_type": "inflow",
                "batch_id": batch_id,
                "product_name": product_name,
                "quantity": quantity,
                "freshness_status": "Fresh",
            }).execute()
            created.append({
                "batch_id": batch_id,
                "product_name": product_name,
                "quantity": quantity,
            })
        db.commit()
        return created
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@router.post("/inflow", dependencies=[Depends(require_manager)])
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
    img_id = f"inbound_{operation_now(datetime.now).strftime('%Y%m%d_%H%M%S')}"
    results = detect_products(image, scenario="inbound", image_id=img_id)
    aggregated = aggregate_results(results)
    created = _create_production_batches(aggregated)

    return {"status": "ok", "batches_created": len(created), "batches": created}


# ======================================================================

# POST /s1/inflow/batch -- Confirm inflow from frontend (no re-scan)
# ======================================================================
@router.post("/inflow/batch", dependencies=[Depends(require_manager)])
async def inflow_batch(req: DeductRequest):
    """Called after the user confirms detected items in the inflow UI.
    Creates batch_inventory records without re-scanning the image.
    """
    created = _create_production_batches(req.items)

    return {"status": "ok", "batches_created": len(created), "batches": created}


async def deduct_inventory(req: DeductRequest, db=None):
    """Called by S4 after a successful payment.

    Deducts ``quantity`` from batch_inventory using FIFO
    (oldest production_time first), and writes outflow transactions.

    Returns per-item deduction details and any errors (e.g. insufficient
    stock for a product).
    """
    validation_errors = []
    if not req.receipt_id:
        validation_errors.append("receipt_id required for sold inventory")
    for item in req.items:
        product_name = item.get("product_name", "")
        if not is_positive_integer(item.get("quantity")):
            validation_errors.append(
                f"Invalid quantity for '{product_name}': expected a positive integer"
            )
        requested_freshness = item.get("freshness")
        if requested_freshness not in (None, "Fresh", "Day-1"):
            validation_errors.append(
                f"Invalid freshness for '{product_name}': {requested_freshness}"
            )
        try:
            unit_price = Decimal(str(item.get("unit_price")))
        except (InvalidOperation, TypeError, ValueError):
            unit_price = Decimal("0")
        if not unit_price.is_finite() or unit_price <= 0:
            validation_errors.append(
                f"Invalid canonical unit price for '{product_name}'"
            )
        try:
            discount_applied = Decimal(str(item.get("discount_applied", 0)))
        except (InvalidOperation, TypeError, ValueError):
            discount_applied = Decimal("-1")
        if (
            not discount_applied.is_finite()
            or discount_applied < 0
            or discount_applied > 1
        ):
            validation_errors.append(
                f"Invalid discount rate for '{product_name}'"
            )
    if validation_errors:
        return DeductResponse(status="error", deducted=[], errors=validation_errors)

    owns_connection = db is None
    if owns_connection:
        db = get_db()
        db.autocommit = False
    try:
        result = _deduct_inventory_with_db(req, db)
        if owns_connection and hasattr(db, "commit"):
            db.commit()
        return result
    except Exception:
        if owns_connection and hasattr(db, "rollback"):
            db.rollback()
        raise
    finally:
        if owns_connection and hasattr(db, "close"):
            db.close()


def _deduct_inventory_with_db(req: DeductRequest, db):
    deducted = []
    errors = []

    # Collect all product names for a single batch query (N+1 fix)
    product_names = list(set(
        item.get("product_name", "")
        for item in req.items
        if item.get("product_name", "")
    ))
    if not product_names:
        return DeductResponse(status="ok", deducted=[], errors=["No valid items"])

    # Single query: fetch all batches for all products at once (raw SQL for IN clause)
    placeholders = ", ".join(["%s"] * len(product_names))
    sql = f"SELECT * FROM batch_inventory WHERE product_name IN ({placeholders}) AND quantity_remaining > 0 AND freshness_status IN ('Fresh', 'Day-1') ORDER BY product_name, production_time, batch_id FOR UPDATE"
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
        qty_needed = item["quantity"]

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

            available = batch["quantity_remaining"]
            take = min(available, remaining_to_deduct)
            new_remaining = available - take

            # Update batch_inventory
            q(db, "batch_inventory").update({
                "quantity_remaining": new_remaining,
            }).eq("batch_id", batch["batch_id"]).execute()
            batch["quantity_remaining"] = new_remaining

            # Write outflow transaction
            q(db, "inventory_transactions").insert({
                "transaction_type": "outflow",
                "batch_id": batch["batch_id"],
                "product_name": product_name,
                "quantity": take,
                "unit_price": float(item["unit_price"]),
                "discount_applied": float(item.get("discount_applied", 0)),
                "freshness_status": batch.get("freshness_status"),
                "receipt_id": req.receipt_id,
                "disposition": "sold",
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
@router.get("/search", dependencies=[Depends(get_current_user)])
async def search_products(
    keyword: str = Query(
        ...,
        alias="q",
        description="Search keyword (product name or batch ID)",
    ),
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
        .eq("batch_id", keyword)
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
        .ilike("product_name", f"%{keyword}%")
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
@router.get("/batch_inventory", dependencies=[Depends(get_current_user)])
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
@router.get("/detection_logs", dependencies=[Depends(require_manager)])
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
# GET /s1/inflow/history -- Finished-product inflow history by date
# ======================================================================
@router.get("/inflow/history", dependencies=[Depends(require_manager)])
async def get_inflow_history(
    date: str = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    selected_date = date or operation_now(datetime.now).strftime("%Y-%m-%d")
    try:
        start = datetime.strptime(selected_date, "%Y-%m-%d")
    except ValueError as exc:
        raise HTTPException(400, "Date must use YYYY-MM-DD format") from exc
    if start.strftime("%Y-%m-%d") != selected_date:
        raise HTTPException(400, "Date must use YYYY-MM-DD format")
    end = start + timedelta(days=1)
    now = operation_now(datetime.now)
    is_today = start.date() == now.date()
    boundary = now if is_today else end
    remaining_label = "Left Now" if is_today else "Left at Close"
    snapshot_basis = "current_live" if is_today else "historical_close"

    db = get_db()
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            WITH bounds AS (
                SELECT CAST(%s AS DATETIME) AS day_start,
                       CAST(%s AS DATETIME) AS balance_time
            )
            SELECT MIN(it.id) AS id,
                   bi.batch_id,
                   bi.product_name,
                   bi.production_time AS transaction_time,
                   COALESCE(SUM(
                       CASE WHEN it.transaction_time < bounds.day_start
                            THEN CASE
                                WHEN it.transaction_type = 'inflow' THEN ABS(it.quantity)
                                WHEN it.transaction_type = 'outflow' THEN -ABS(it.quantity)
                                ELSE 0
                            END
                            ELSE 0 END
                   ), 0) AS quantity_opening,
                   COALESCE(SUM(
                       CASE WHEN it.transaction_time >= bounds.day_start
                                      AND it.transaction_time < bounds.balance_time
                                      AND it.transaction_type = 'inflow'
                            THEN ABS(it.quantity) ELSE 0 END
                   ), 0) AS quantity_baked,
                   COALESCE(SUM(
                       CASE WHEN it.transaction_time >= bounds.day_start
                                      AND it.transaction_time < bounds.balance_time
                                      AND it.transaction_type = 'outflow'
                                      AND it.disposition = 'sold'
                            THEN ABS(it.quantity) ELSE 0 END
                   ), 0) AS quantity_sold,
                   COALESCE(SUM(
                       CASE WHEN it.transaction_time >= bounds.day_start
                                      AND it.transaction_time < bounds.balance_time
                                      AND it.transaction_type = 'outflow'
                                      AND COALESCE(it.disposition, '') <> 'sold'
                                   AND (
                                      it.disposition IN ('non_sellable', 'discarded')
                                      OR it.freshness_status = 'Expired'
                                   )
                            THEN ABS(it.quantity) ELSE 0 END
                   ), 0) AS quantity_discarded,
                   COALESCE(SUM(
                       CASE WHEN it.transaction_time >= bounds.day_start
                                      AND it.transaction_time < bounds.balance_time
                                      AND it.transaction_type = 'outflow'
                            THEN ABS(it.quantity) ELSE 0 END
                   ), 0) AS quantity_outflow_total,
                   COALESCE(SUM(
                       CASE WHEN it.transaction_time < bounds.balance_time
                            THEN CASE
                                WHEN it.transaction_type = 'inflow' THEN ABS(it.quantity)
                                WHEN it.transaction_type = 'outflow' THEN -ABS(it.quantity)
                                ELSE 0
                            END
                            ELSE 0 END
                   ), 0) AS quantity_closing
            FROM batch_inventory bi
            JOIN products p ON p.product_name = bi.product_name
            CROSS JOIN bounds
            LEFT JOIN inventory_transactions it
                   ON it.batch_id = bi.batch_id
                  AND it.transaction_time < bounds.balance_time
            WHERE p.category = 'bakery'
              AND bi.production_time < bounds.balance_time
            GROUP BY bi.batch_id, bi.product_name, bi.production_time,
                     bounds.day_start, bounds.balance_time
            HAVING quantity_opening > 0
                OR quantity_baked > 0
                OR quantity_outflow_total > 0
                OR quantity_closing > 0
            ORDER BY bi.production_time, bi.batch_id
            LIMIT %s
            """,
            (
                start.strftime("%Y-%m-%d %H:%M:%S"),
                boundary.strftime("%Y-%m-%d %H:%M:%S"),
                limit,
            ),
        )
        records = cursor.fetchall()
    finally:
        cursor.close()
        db.close()

    for record in records:
        quantity_opening = max(int(record.get("quantity_opening") or 0), 0)
        quantity_baked = max(int(record.get("quantity_baked") or 0), 0)
        quantity_sold = max(int(record.get("quantity_sold") or 0), 0)
        quantity_discarded = max(int(record.get("quantity_discarded") or 0), 0)
        quantity_outflow_total = max(
            int(record.pop("quantity_outflow_total", 0) or 0),
            0,
        )
        quantity_other_outflow = max(
            quantity_outflow_total - quantity_sold - quantity_discarded,
            0,
        )
        calculated_left = (
            quantity_opening
            + quantity_baked
            - quantity_sold
            - quantity_discarded
            - quantity_other_outflow
        )
        raw_closing = record.pop("quantity_closing", calculated_left)
        quantity_closing = int(raw_closing or 0)
        transaction_time = record.get("transaction_time")
        production_date = (
            transaction_time.date()
            if isinstance(transaction_time, datetime)
            else datetime.strptime(str(transaction_time)[:10], "%Y-%m-%d").date()
        )
        fresh_remaining = (
            max(quantity_closing, 0)
            if production_date == start.date()
            else 0
        )
        record.update(
            {
                "quantity_opening": quantity_opening,
                "quantity_baked": quantity_baked,
                "quantity_sold": quantity_sold,
                "quantity_discarded": quantity_discarded,
                "quantity_other_outflow": quantity_other_outflow,
                "quantity_fresh_remaining": fresh_remaining,
                "quantity_carried_to_day1": (
                    fresh_remaining if not is_today else 0
                ),
                "quantity_left": max(quantity_closing, 0),
                "data_quality_issue": (
                    quantity_closing < 0 or calculated_left != quantity_closing
                ),
            }
        )
        if isinstance(transaction_time, datetime):
            record["transaction_time"] = transaction_time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
    return {
        "status": "ok",
        "date": selected_date,
        "count": len(records),
        "remaining_label": remaining_label,
        "snapshot_basis": snapshot_basis,
        "balance_time": boundary.strftime("%Y-%m-%d %H:%M:%S"),
        "records": records,
    }


# ======================================================================
# GET /s1/inventory_transactions -- Query all inventory transactions
# ======================================================================
@router.get("/inventory_transactions", dependencies=[Depends(require_manager)])
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
@router.get("/inventory", dependencies=[Depends(require_manager)])
async def get_inventory_summary():
    """Return current inventory aggregated by product_name (total qty per product)
    including selling_price from the products table."""
    db = get_db()
    r = q(db, "batch_inventory").select("*").gt("quantity_remaining", 0).execute()
    
    # Fetch selling_price per product for pricing info
    try:
        pr = db.cursor()
        pr.execute("SELECT product_name, selling_price FROM products")
        prices = {row[0]: float(row[1]) for row in pr.fetchall()}
        pr.close()
    except Exception:
        prices = {}
    
    summary = {}
    for row in (r.data or []):
        pn = row.get("product_name", "unknown")
        qty = row.get("quantity_remaining", 0) or row.get("quantity", 0)
        if pn not in summary:
            summary[pn] = {"product_name": pn, "total_quantity": 0, "batches": 0,
                          "selling_price": prices.get(pn, 5.90)}
        summary[pn]["total_quantity"] += qty
        summary[pn]["batches"] += 1
    return {"status": "ok", "inventory": list(summary.values())}

