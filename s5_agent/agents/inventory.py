# Inventory Agent - stock levels + freshness + waste risk
# Phase 4: direct DB query for authoritative freshness data (no heuristic guessing).
import logging
from typing import Dict, Any
from .base import BaseAgent
from s5_agent.core.dashboard_api import fetch_dashboard_json
from s5_agent.s5_config.settings import S1_INVENTORY_URL, THRESHOLDS
from s5_agent.schemas.agent_output import AgentOutput, DataQuality
from s5_agent.schemas.evidence import EvidenceItem
from s5_agent.schemas.recommendation import Recommendation

logger = logging.getLogger("s5.agent.inventory")


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


class InventoryAgent(BaseAgent):
    def __init__(self):
        super().__init__("inventory")
        self._fetch_ok = False

    async def fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._fetch_ok = False
        try:
            payload = fetch_dashboard_json(S1_INVENTORY_URL, params)
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

        target_products = None
        if "," in product:
            target_products = set(p.strip() for p in product.split(","))
        elif product == "all":
            target_products = None
        else:
            target_products = {product}

        # Primary: direct DB query for authoritative freshness data
        db_data = self._query_db_freshness(target_products)
        if db_data:
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

        fresh = freshness_counts.get("Fresh", 0)
        day1 = freshness_counts.get("Day-1", 0)
        zero_stock_products = sorted(name for name, value in per_product.items() if int(value.get("qty", 0) or 0) <= 0)
        day1_products = sorted(name for name, value in per_product.items() if int(value.get("day1", 0) or 0) > 0)
        if per_product and total_qty == 0:
            waste_risk = "stockout_risk"
        elif total_qty > THRESHOLDS["inventory_total_high"] and fresh < THRESHOLDS["inventory_fresh_low"]:
            waste_risk = "high"
        elif day1 == 0:
            waste_risk = "low"
        else:
            waste_risk = "medium"

        # Confidence: DB direct query is authoritative (0.95)
        matched = len(per_product) > 0
        if db_data and matched:
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
                "day1_products": day1_products,
                "unit_price": per_product.get(product, {}).get("selling_price", THRESHOLDS["inventory_default_price"]) if target_products and len(target_products) == 1 else 5.90,
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
        day1_products = data.get("day1_products", []) or []
        product = params.get("product", "croissant")
        confidence = float(result.get("confidence", 0.0))
        freshness_value = "fresh" if confidence >= 0.75 else "unknown"
        evidence_id = "inventory_total"

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
                "day1_product_count": len(day1_products),
                "day1_products": day1_products,
            },
            evidence_items=[
                EvidenceItem(
                    id=evidence_id,
                    source="inventory",
                    description="Total inventory for requested product scope",
                    value=inventory,
                    metadata={"product": product},
                )
            ],
            risks=(
                ["inventory_data_gap"]
                if product_count == 0
                else ["stockout_risk", "inventory_data_gap"]
                if inventory == 0 and product_count > 0
                else ([waste_risk] if waste_risk else [])
            ),
            recommendations=recommendations,
            data_quality=DataQuality(
                freshness=freshness_value,
                completeness=1.0 if inventory > 0 else 0.5,
                limitations=[
                    "Zero finished-product stock may reflect a real stockout or a batch inventory synchronization gap."
                ] if inventory == 0 and product_count > 0 else (
                    ["No finished-product records were available for the selected scope."]
                    if product_count == 0 else []
                ),
                source_status={"inventory": freshness_value},
            ),
            limitations=[
                "Zero finished-product stock may reflect a real stockout or a batch inventory synchronization gap."
            ] if inventory == 0 and product_count > 0 else (
                ["No finished-product records were available for the selected scope."]
                if product_count == 0 else []
            ),
        )
