# Inventory Agent - stock levels + freshness + waste risk
# Dashboard data is primary so S5 stays aligned with the selected Inventory BI date.
import logging
from typing import Dict, Any
from urllib.parse import urlencode
from s5_agent.core.dashboard_api import fetch_dashboard_json
from s5_agent.s5_config.settings import (
    S1_INFLOW_HISTORY_URL,
    S1_INVENTORY_URL,
    S4_DASHBOARD_URL,
    THRESHOLDS,
)
from s5_agent.schemas.agent_output import AgentOutput, DataQuality
from s5_agent.schemas.evidence import EvidenceItem
from s5_agent.schemas.recommendation import Recommendation

logger = logging.getLogger("s5.agent.inventory")


def _nonnegative_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _summarize_inflow_history(
    raw: Dict[str, Any],
    target_products: set[str] | None,
) -> Dict[str, Any]:
    payload = raw.get("inflow_history", {})
    if not isinstance(payload, dict):
        payload = {}
    records = payload.get("records", [])
    if not isinstance(records, list):
        records = []

    per_product: Dict[str, Dict[str, int]] = {}
    balance_issue_count = 0
    record_count = 0
    for record in records:
        if not isinstance(record, dict):
            continue
        product_name = str(record.get("product_name") or "").strip()
        if not product_name or (
            target_products is not None and product_name not in target_products
        ):
            continue
        opening = _nonnegative_int(record.get("quantity_opening"))
        baked = _nonnegative_int(record.get("quantity_baked"))
        sold = _nonnegative_int(record.get("quantity_sold"))
        discarded = _nonnegative_int(record.get("quantity_discarded"))
        other_outflow = _nonnegative_int(record.get("quantity_other_outflow"))
        carried_to_day1 = _nonnegative_int(
            record.get("quantity_carried_to_day1")
        )
        left = _nonnegative_int(record.get("quantity_left"))
        balance_issue = bool(record.get("data_quality_issue")) or (
            opening + baked != sold + discarded + other_outflow + left
        )
        totals = per_product.setdefault(
            product_name,
            {
                "opening": 0,
                "baked": 0,
                "sold": 0,
                "discarded": 0,
                "other_outflow": 0,
                "carried_to_day1": 0,
                "left": 0,
            },
        )
        totals["opening"] += opening
        totals["baked"] += baked
        totals["sold"] += sold
        totals["discarded"] += discarded
        totals["other_outflow"] += other_outflow
        totals["carried_to_day1"] += carried_to_day1
        totals["left"] += left
        record_count += 1
        balance_issue_count += int(balance_issue)

    opening_units = sum(item["opening"] for item in per_product.values())
    baked_units = sum(item["baked"] for item in per_product.values())
    available_units = opening_units + baked_units
    sold_units = sum(item["sold"] for item in per_product.values())
    discarded_units = sum(item["discarded"] for item in per_product.values())
    other_outflow_units = sum(item["other_outflow"] for item in per_product.values())
    carried_to_day1_units = sum(
        item["carried_to_day1"] for item in per_product.values()
    )
    left_units = sum(item["left"] for item in per_product.values())
    sell_through_pct = (
        round(sold_units / available_units * 100, 1) if available_units else 0.0
    )
    discard_rate_pct = (
        round(discarded_units / available_units * 100, 1)
        if available_units
        else 0.0
    )

    high_sell_through_products = []
    slow_moving_products = []
    minimum_baked = int(THRESHOLDS["inventory_flow_min_baked_units"])
    for product_name, item in per_product.items():
        product_baked = item["baked"]
        product_available = item["opening"] + product_baked
        product_sell_through = (
            item["sold"] / product_available * 100 if product_available else 0.0
        )
        if (
            product_baked >= minimum_baked
            and product_sell_through >= THRESHOLDS["inventory_high_sell_through_pct"]
            and item["left"] <= 1
        ):
            high_sell_through_products.append(product_name)
        if (
            product_baked >= minimum_baked
            and product_sell_through <= THRESHOLDS["inventory_low_sell_through_pct"]
            and item["left"] >= 2
        ):
            slow_moving_products.append(product_name)

    high_sell_through_products.sort(
        key=lambda product_name: (
            -(
                per_product[product_name]["sold"]
                / (
                    per_product[product_name]["opening"]
                    + per_product[product_name]["baked"]
                )
            ),
            -per_product[product_name]["sold"],
            product_name,
        )
    )

    return {
        "flow_record_count": record_count,
        "flow_opening_units": opening_units,
        "flow_baked_units": baked_units,
        "flow_available_units": available_units,
        "flow_sold_units": sold_units,
        "flow_discarded_units": discarded_units,
        "flow_other_outflow_units": other_outflow_units,
        "flow_carried_to_day1_units": carried_to_day1_units,
        "flow_left_units": left_units,
        "flow_sell_through_pct": sell_through_pct,
        "flow_discard_rate_pct": discard_rate_pct,
        "flow_balance_issue_count": balance_issue_count,
        "flow_date": str(payload.get("date") or ""),
        "flow_remaining_label": str(payload.get("remaining_label") or ""),
        "flow_per_product": per_product,
        "high_sell_through_products": high_sell_through_products,
        "slow_moving_products": sorted(slow_moving_products),
    }


def _format_opinion(total_qty, fresh, day1, waste_risk, per_product, params, product_str):
    """Format inventory opinion. Per-product for comparison, aggregated otherwise."""
    intent = params.get("intent", "")
    if intent == "comparison_analysis" and len(per_product) >= 2:
        parts = []
        for pname, pdata in sorted(per_product.items()):
            parts.append(f"{pname}: stock={pdata['qty']} (fresh={pdata['fresh']}, day-1={pdata['day1']})")
        return " | ".join(parts) + f", waste_risk={waste_risk}"
    if product_str == "all" and per_product:
        details = ", ".join(f"{k}:{v['qty']}" for k, v in sorted(per_product.items()))
        return f"Stock {total_qty} (fresh={fresh}, day-1={day1}) across {len(per_product)} products [{details}]"
    return f"Stock {total_qty} (fresh={fresh}, day-1={day1}), waste_risk={waste_risk}"


class InventoryAgent:
    def __init__(self):
        self.name = "inventory"
        self._fetch_ok = False

    async def fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._fetch_ok = False
        try:
            selected_date = str(params.get("date") or "").strip()
            if selected_date:
                url = f"{S4_DASHBOARD_URL}?{urlencode({'date': selected_date})}"
            else:
                url = S4_DASHBOARD_URL
            payload = fetch_dashboard_json(url, params)
            if not payload:
                if selected_date:
                    payload = {
                        "status": "unavailable",
                        "bread_stock": [],
                        "snapshot_date": selected_date,
                    }
                else:
                    payload = fetch_dashboard_json(S1_INVENTORY_URL, params)
            if payload.get("status") == "ok" and "bread_stock" in payload:
                history_url = S1_INFLOW_HISTORY_URL
                if selected_date:
                    history_url = (
                        f"{history_url}?{urlencode({'date': selected_date})}"
                    )
                try:
                    inflow_history = fetch_dashboard_json(history_url, params)
                except (OSError, TimeoutError, ValueError) as exc:
                    logger.warning("Inventory inflow history fetch failed: %s", exc)
                    inflow_history = {}
                if inflow_history:
                    payload = dict(payload)
                    payload["inflow_history"] = inflow_history
            self._fetch_ok = bool(payload)
            return payload
        except (OSError, TimeoutError, ValueError) as e:
            logger.warning("Inventory fetch failed: %s", e)
            return {}

    def _query_db_freshness(self, product_filter=None):
        """Query batch_inventory directly for authoritative freshness data."""
        try:
            from db.mysql_client import get_db
            db = get_db()
            cur = db.cursor()
            if product_filter:
                placeholders = ",".join(["%s"] * len(product_filter))
                cur.execute(
                    f"SELECT product_name, freshness_status, "
                    f"SUM(COALESCE(quantity_remaining, quantity, 0)) as total "
                    f"FROM batch_inventory WHERE product_name IN ({placeholders}) "
                    f"GROUP BY product_name, freshness_status", list(product_filter))
            else:
                cur.execute(
                    "SELECT product_name, freshness_status, "
                    "SUM(COALESCE(quantity_remaining, quantity, 0)) as total "
                    "FROM batch_inventory GROUP BY product_name, freshness_status")
            rows = cur.fetchall()
            cur.close()
            result = {}
            for pname, status, qty in rows:
                if pname not in result:
                    result[pname] = {"Fresh": 0, "Day-1": 0, "qty": 0}
                status_clean = status.strip()
                result[pname][status_clean] = int(qty)
                result[pname]["qty"] += int(qty)
            # Fetch selling prices
            cur = db.cursor()
            cur.execute("SELECT product_name, selling_price FROM products")
            prices = {r[0]: r[1] for r in cur.fetchall()}
            cur.close()
            for pname in result:
                result[pname]["selling_price"] = prices.get(pname, THRESHOLDS["inventory_default_price"])
            return result
        except Exception as e:
            logger.warning("DB freshness query failed: %s", e)
            return None

    def analyze(self, raw: Dict[str, Any], params: Dict[str, Any],
                history: str = "", key_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        product = params.get("product", "croissant")
        total_qty = 0
        freshness_counts = {"Fresh": 0, "Day-1": 0}
        per_product = {}
        raw_materials = []

        target_products = None
        if "," in product:
            target_products = set(p.strip() for p in product.split(","))
        elif product == "all":
            target_products = None
        else:
            target_products = {product}

        overdue_stock = []
        for item in raw.get("overdue_stock", []) or []:
            if not isinstance(item, dict):
                continue
            product_name = str(item.get("product_name") or "").strip()
            if not product_name or (
                target_products is not None
                and product_name not in target_products
            ):
                continue
            overdue_stock.append(
                {
                    "product_name": product_name,
                    "overdue_qty": _nonnegative_int(item.get("overdue_qty")),
                    "oldest_production_date": str(
                        item.get("oldest_production_date") or ""
                    ),
                }
            )
        overdue_stock_total = sum(
            item["overdue_qty"] for item in overdue_stock
        )
        if not overdue_stock and target_products is None:
            overdue_stock_total = _nonnegative_int(raw.get("overdue_total"))
        overdue_stock_products = sorted(
            item["product_name"]
            for item in overdue_stock
            if item["overdue_qty"] > 0
        )

        flow_metrics = _summarize_inflow_history(raw, target_products)
        has_dashboard_stock = "bread_stock" in raw
        dashboard_stock = raw.get("bread_stock", [])
        db_data = None if has_dashboard_stock else self._query_db_freshness(target_products)
        if has_dashboard_stock:
            for item in dashboard_stock:
                pname = item.get("product_name", "unknown")
                if target_products is not None and pname not in target_products:
                    continue
                p_fresh = int(item.get("fresh_qty", 0) or 0)
                p_day1 = int(item.get("day1_qty", 0) or 0)
                pqty = int(item.get("total_qty", p_fresh + p_day1) or 0)
                total_qty += pqty
                freshness_counts["Fresh"] += p_fresh
                freshness_counts["Day-1"] += p_day1
                per_product[pname] = {
                    "qty": pqty,
                    "fresh": p_fresh,
                    "day1": p_day1,
                    "selling_price": item.get("selling_price", THRESHOLDS["inventory_default_price"]),
                }
            for pname, movement in flow_metrics.get("flow_per_product", {}).items():
                if pname in per_product or _nonnegative_int(movement.get("left")) > 0:
                    continue
                per_product[pname] = {
                    "qty": 0,
                    "fresh": 0,
                    "day1": 0,
                    "selling_price": THRESHOLDS["inventory_default_price"],
                }
        elif db_data:
            for pname, pdata in db_data.items():
                total_qty += pdata["qty"]
                freshness_counts["Fresh"] += pdata.get("Fresh", 0)
                freshness_counts["Day-1"] += pdata.get("Day-1", 0)
                per_product[pname] = {
                    "qty": pdata["qty"],
                    "fresh": pdata.get("Fresh", 0),
                    "day1": pdata.get("Day-1", 0),
                    "selling_price": pdata.get("selling_price", THRESHOLDS["inventory_default_price"]),
                }
        else:
            # Fallback: S1 API plus batch-based freshness estimate.
            inventory_list = raw.get("inventory", [])
            for item in inventory_list:
                pname = item.get("product_name", "unknown")
                if target_products is None or pname in target_products:
                    pqty = item.get("total_quantity", 0)
                    pbatches = item.get("batches", 0)
                    total_qty += pqty
                    p_fresh = pqty if pbatches <= 1 else max(0, pqty // 2)
                    p_day1 = 0 if pbatches <= 1 else pqty - p_fresh
                    pselling = item.get("selling_price", THRESHOLDS["inventory_default_price"])
                    per_product[pname] = {"qty": pqty, "batches": pbatches, "fresh": p_fresh, "day1": p_day1, "selling_price": pselling}
                    if pbatches <= 1:
                        freshness_counts["Fresh"] += pqty
                    else:
                        freshness_counts["Fresh"] += max(0, pqty // 2)
                        freshness_counts["Day-1"] += pqty - max(0, pqty // 2)

        beverage_materials = raw.get("beverage_materials")
        if beverage_materials is None:
            beverage_materials = raw.get("coffee_materials", [])
        material_by_name = {}
        grouped_materials = [
            *raw.get("baking_materials", []),
            *beverage_materials,
            *raw.get("packaging_materials", []),
        ]
        for item in grouped_materials:
            stock = float(item.get("stock", item.get("stock_quantity", 0)) or 0)
            reorder_point = float(item.get("reorder_point", 0) or 0)
            material_name = str(item.get("material_name", "unknown"))
            material_by_name[material_name] = {
                "material_name": material_name,
                "stock": stock,
                "reorder_point": reorder_point,
                "unit": str(item.get("unit", "")),
            }
        raw_materials = [
            material_by_name[name] for name in sorted(material_by_name)
        ]

        fresh = freshness_counts.get("Fresh", 0)
        day1 = freshness_counts.get("Day-1", 0)
        zero_stock_products = sorted(name for name, value in per_product.items() if int(value.get("qty", 0) or 0) <= 0)
        low_stock_products = sorted(
            name
            for name, value in per_product.items()
            if 0 < int(value.get("qty", 0) or 0) <= 1
        )
        day1_products = sorted(name for name, value in per_product.items() if int(value.get("day1", 0) or 0) > 0)
        low_stock_materials = sorted(
            item["material_name"]
            for item in raw_materials
            if item["reorder_point"] > 0 and item["stock"] <= item["reorder_point"]
        )
        critical_materials = sorted(
            item["material_name"] for item in raw_materials if item["stock"] <= 0
        )
        if per_product and total_qty == 0:
            waste_risk = "stockout_risk"
        elif total_qty > THRESHOLDS["inventory_total_high"] and fresh < THRESHOLDS["inventory_fresh_low"]:
            waste_risk = "high"
        elif day1 == 0:
            waste_risk = "low"
        else:
            waste_risk = "medium"

        matched = len(per_product) > 0
        if has_dashboard_stock and matched:
            confidence = 0.90
        elif db_data and matched:
            confidence = 0.95
        elif self._fetch_ok and matched:
            confidence = 0.75
        elif matched:
            confidence = 0.50
        else:
            confidence = 0.10

        opinion = _format_opinion(total_qty, fresh, day1, waste_risk, per_product, params, product) if matched else f"No stock data for {product}"

        constraints = []
        if total_qty == 0 and matched:
            constraints.append("no stock at all - emergency restock needed")

        return {
            "opinion": opinion, "confidence": round(confidence, 2), "constraints": constraints,
            "data": {
                "inventory": total_qty, "fresh": fresh, "day1_available": day1,
                "waste_risk": waste_risk, "freshness_breakdown": freshness_counts,
                "per_product": per_product,
                "product_count": len(per_product),
                "zero_stock_products": zero_stock_products,
                "low_stock_products": low_stock_products,
                "day1_products": day1_products,
                "raw_materials": raw_materials,
                "low_stock_materials": low_stock_materials,
                "critical_materials": critical_materials,
                "overdue_stock_total": overdue_stock_total,
                "overdue_stock_products": overdue_stock_products,
                "overdue_stock": overdue_stock,
                "snapshot_date": str(params.get("date") or "").strip(),
                "snapshot_basis": (
                    "selected_date_dashboard"
                    if has_dashboard_stock and params.get("date")
                    else "current_dashboard"
                    if has_dashboard_stock
                    else "current_batch_inventory"
                ),
                "unit_price": per_product.get(product, {}).get("selling_price", THRESHOLDS["inventory_default_price"]) if target_products and len(target_products) == 1 else 5.90,
                **flow_metrics,
            },
        }

    def analyze_for_graph(self, raw: Dict[str, Any], params: Dict[str, Any]) -> AgentOutput:
        result = self.analyze(raw, params)
        data = result.get("data", {})
        inventory = data.get("inventory", 0)
        fresh = data.get("fresh", 0)
        day1_available = data.get("day1_available", 0)
        waste_risk = data.get("waste_risk")
        product_count = int(data.get("product_count", 0) or 0)
        zero_stock_products = data.get("zero_stock_products", []) or []
        low_stock_products = data.get("low_stock_products", []) or []
        day1_products = data.get("day1_products", []) or []
        raw_materials = data.get("raw_materials", []) or []
        low_stock_materials = data.get("low_stock_materials", []) or []
        critical_materials = data.get("critical_materials", []) or []
        overdue_stock_total = int(data.get("overdue_stock_total", 0) or 0)
        overdue_stock_products = data.get("overdue_stock_products", []) or []
        flow_record_count = int(data.get("flow_record_count", 0) or 0)
        flow_baked_units = int(data.get("flow_baked_units", 0) or 0)
        flow_sold_units = int(data.get("flow_sold_units", 0) or 0)
        flow_discarded_units = int(data.get("flow_discarded_units", 0) or 0)
        flow_other_outflow_units = int(data.get("flow_other_outflow_units", 0) or 0)
        flow_left_units = int(data.get("flow_left_units", 0) or 0)
        flow_sell_through_pct = float(data.get("flow_sell_through_pct", 0.0) or 0.0)
        flow_discard_rate_pct = float(data.get("flow_discard_rate_pct", 0.0) or 0.0)
        flow_balance_issue_count = int(data.get("flow_balance_issue_count", 0) or 0)
        high_sell_through_products = data.get("high_sell_through_products", []) or []
        slow_moving_products = data.get("slow_moving_products", []) or []
        product = params.get("product", "croissant")
        confidence = float(result.get("confidence", 0.0))
        freshness_value = "fresh" if confidence >= 0.75 else "unknown"
        evidence_id = "inventory_total"
        thin_stock_product_count = len(zero_stock_products) + len(low_stock_products)
        thin_stock_product_share_pct = round(
            thin_stock_product_count / product_count * 100,
            1,
        ) if product_count else 0.0
        units_per_product = round(inventory / product_count, 2) if product_count else 0.0

        recommendations = []
        if waste_risk in {"medium", "high"}:
            recommendations.append(
                Recommendation(
                    id="inventory_clearance",
                    action="Run a targeted clearance action for inventory at waste risk.",
                    urgency="high" if waste_risk == "high" else "medium",
                    time_horizon="today",
                    rationale="Inventory waste risk is elevated for the requested product scope.",
                    evidence_ids=[evidence_id],
                )
            )

        risks = []
        if product_count == 0:
            risks.append("inventory_data_gap")
        elif inventory == 0:
            risks.extend(["stockout_risk", "inventory_data_gap"])
        elif zero_stock_products or low_stock_products:
            risks.append("stockout_risk")
        if inventory > 0 and thin_stock_product_share_pct >= 50:
            risks.append("widespread_low_stock_risk")
        if waste_risk in {"medium", "high"}:
            risks.append("inventory_expiry_risk")
        if low_stock_materials:
            risks.append("material_shortage_risk")
        if overdue_stock_total:
            risks.append("expired_stock_pending_disposal_risk")
        if high_sell_through_products:
            risks.append("high_sell_through_stock_risk")
        if slow_moving_products:
            risks.append("overproduction_risk")
        if flow_discarded_units:
            risks.append("finished_product_discard_risk")
        if flow_balance_issue_count:
            risks.append("inventory_flow_data_gap")

        limitations = []
        if inventory == 0 and product_count > 0:
            limitations.append(
                "Zero finished-product stock may reflect a real stockout or a batch inventory synchronization gap."
            )
        elif product_count == 0:
            limitations.append(
                "No finished-product records were available for the selected scope."
            )
        if flow_balance_issue_count:
            limitations.append(
                "One or more baked-product flow records do not reconcile and require transaction review."
            )
        elif params.get("date") and flow_record_count == 0:
            limitations.append(
                "No baked-product inflow records were available to explain the selected-date stock movement."
            )
        if overdue_stock_total:
            limitations.append(
                "Expired positive balances are excluded from sellable stock and require disposal-record verification."
            )

        return AgentOutput(
            agent_name="InventoryAgent",
            claim=result["opinion"],
            confidence=confidence,
            metrics={
                "inventory": inventory,
                "fresh": fresh,
                "day1_available": day1_available,
                "product_count": product_count,
                "zero_stock_product_count": len(zero_stock_products),
                "zero_stock_products": zero_stock_products,
                "low_stock_product_count": len(low_stock_products),
                "low_stock_products": low_stock_products,
                "thin_stock_product_count": thin_stock_product_count,
                "thin_stock_product_share_pct": thin_stock_product_share_pct,
                "units_per_product": units_per_product,
                "day1_product_count": len(day1_products),
                "day1_products": day1_products,
                "raw_material_count": len(raw_materials),
                "low_stock_material_count": len(low_stock_materials),
                "low_stock_materials": low_stock_materials,
                "critical_material_count": len(critical_materials),
                "critical_materials": critical_materials,
                "overdue_stock_total": overdue_stock_total,
                "overdue_stock_products": overdue_stock_products,
                "snapshot_date": data.get("snapshot_date", ""),
                "snapshot_basis": data.get("snapshot_basis", "current_batch_inventory"),
                "flow_record_count": flow_record_count,
                "flow_baked_units": flow_baked_units,
                "flow_sold_units": flow_sold_units,
                "flow_discarded_units": flow_discarded_units,
                "flow_other_outflow_units": flow_other_outflow_units,
                "flow_left_units": flow_left_units,
                "flow_sell_through_pct": flow_sell_through_pct,
                "flow_discard_rate_pct": flow_discard_rate_pct,
                "flow_balance_issue_count": flow_balance_issue_count,
                "flow_date": data.get("flow_date", ""),
                "flow_remaining_label": data.get("flow_remaining_label", ""),
                "flow_per_product": data.get("flow_per_product", {}),
                "high_sell_through_products": high_sell_through_products,
                "slow_moving_products": slow_moving_products,
            },
            evidence_items=[
                EvidenceItem(
                    id=evidence_id,
                    source="inventory",
                    description="Total inventory for requested product scope",
                    value=inventory,
                    metadata={"product": product},
                ),
                EvidenceItem(
                    id="zero_stock_product_count",
                    source="inventory",
                    description="Products with no recorded finished stock",
                    value=len(zero_stock_products),
                ),
                EvidenceItem(
                    id="low_stock_product_count",
                    source="inventory",
                    description="Products with only one finished unit recorded",
                    value=len(low_stock_products),
                ),
                EvidenceItem(
                    id="thin_stock_product_share_pct",
                    source="inventory",
                    description="Share of tracked products with zero or one finished unit",
                    value=thin_stock_product_share_pct,
                ),
                EvidenceItem(
                    id="units_per_product",
                    source="inventory",
                    description="Average finished-stock units per tracked product",
                    value=units_per_product,
                ),
                EvidenceItem(
                    id="low_stock_material_count",
                    source="inventory",
                    description="Raw materials at or below their reorder point",
                    value=len(low_stock_materials),
                ),
                EvidenceItem(
                    id="overdue_stock_total",
                    source="inventory",
                    description="Expired finished-product units pending disposal verification",
                    value=overdue_stock_total,
                    metadata={"products": overdue_stock_products},
                ),
                EvidenceItem(
                    id="flow_baked_units",
                    source="inventory_flow",
                    description="Units baked in the selected-date inflow records",
                    value=flow_baked_units,
                    metadata={"record_count": flow_record_count},
                ),
                EvidenceItem(
                    id="flow_sell_through_pct",
                    source="inventory_flow",
                    description="Sold units as a share of units baked in the selected-date inflow records",
                    value=flow_sell_through_pct,
                    metadata={
                        "sold_units": flow_sold_units,
                        "discarded_units": flow_discarded_units,
                        "left_units": flow_left_units,
                    },
                ),
                EvidenceItem(
                    id="flow_balance_issue_count",
                    source="inventory_flow",
                    description="Baked-product flow records that do not reconcile",
                    value=flow_balance_issue_count,
                ),
            ],
            risks=risks,
            recommendations=recommendations,
            data_quality=DataQuality(
                freshness=freshness_value,
                completeness=(
                    0.5
                    if inventory == 0
                    else 0.8
                    if flow_balance_issue_count
                    else 1.0
                ),
                limitations=limitations,
                source_status={
                    "inventory": freshness_value,
                    "inventory_flow": (
                        "unknown"
                        if flow_balance_issue_count
                        else "fresh"
                        if flow_record_count
                        else "missing"
                    ),
                },
            ),
            limitations=limitations,
        )
