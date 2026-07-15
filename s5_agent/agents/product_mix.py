import os, sys, logging
from datetime import datetime as dt, timedelta
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
from db.mysql_client import get_db
logger = logging.getLogger("s5.agent.product_mix")

class ProductMixAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_product_ranking", description="Get bread and beverage sales ranking",
            parameters={"date": "string"}, primary=True, _handler=self._get_ranking))

    async def _get_ranking(self, date: str = ""):
        return _query_product_ranking(date)

    async def fetch(self, params):
        date_str = str(params.get("date", "")) if isinstance(params, dict) else ""
        today = _query_product_ranking(date_str)
        week_avg = _query_weekly_avg(date_str)
        catalog = _query_catalog_counts()
        data = {**today, "week_avg": week_avg, **catalog}
        return {"success": True, "data": data, "tool": "product_mix_db"}

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {}) if "data" in raw else raw
        bread = data.get("bread_ranking", [])
        bev = data.get("beverage_ranking", [])
        cat = data.get("category", {})
        week_avg = data.get("week_avg", {})

        if not bread and not bev:
            return AgentOpinion(agent=self.name,
                opinion="No product ranking data available.",
                confidence=0.3,
                attribution={"metric": "product_mix", "root_cause": "no_data", "deviation": 0})

        parts = []
        recs = []
        root_cause = "normal_mix"
        deviation = 0

        # Bread: top concentration risk
        if bread:
            total_bread_rev = sum(p.get("revenue", 0) for p in bread)
            top3_rev = sum(p.get("revenue", 0) for p in bread[:3])
            top3_pct = top3_rev / max(total_bread_rev, 1) * 100

            top_name = bread[0].get("name", "?")
            top_qty = bread[0].get("qty", 0)
            sold_sku_count = int(data.get("sold_bread_sku_count", 0) or 0)
            total_sku = data.get("total_bread_sku", 0)
            if sold_sku_count and total_sku > 0:
                sku_context = f"{sold_sku_count} of {total_sku} SKUs sold today"
            elif sold_sku_count:
                sku_context = f"{sold_sku_count} SKUs sold today"
            else:
                sku_context = "ranked bread products"
            parts.append(
                f"Bread: {sku_context}, top is {top_name} "
                f"({top_qty} units, {chr(165)}{bread[0].get('revenue',0):.0f})"
            )

            if top3_pct > 60:
                root_cause = "high_concentration"
                deviation = round(top3_pct - 50)
                parts.append(f"Top 3 breads = {top3_pct:.0f}% of revenue (concentration risk)")
                recs.append({
                    "action": f"Top 3 breads drive {top3_pct:.0f}% revenue. Promote mid-tier products to reduce risk.",
                    "urgency": "low",
                    "rationale": f"If {top_name} sells out or quality drops, {top3_pct:.0f}% of bread revenue is at risk"
                })

        # Beverages
        if bev:
            top_bev = bev[0].get("name", "?")
            top_bev_qty = bev[0].get("qty", 0)
            parts.append(f"Beverages: top is {top_bev} ({top_bev_qty} units)")

        # Category mix
        bread_rev = cat.get("Bread", 0)
        bev_rev = (cat.get("Beverages", 0) or 0) + (cat.get("Coffee", 0) or 0)
        total_cat = bread_rev + bev_rev
        if total_cat > 0:
            bev_pct = bev_rev / total_cat * 100
            parts.append(f"Mix: Bread {chr(165)}{bread_rev:.0f} vs Beverages {chr(165)}{bev_rev:.0f} ({bev_pct:.0f}% beverages)")

        # ATV contribution analysis: compare today vs 7-day avg per product
        atv_parts = self._atv_contribution(bread, week_avg.get("bread_avg", []))
        if atv_parts:
            parts.append("ATV drivers: " + "; ".join(atv_parts))

        opinion = " | ".join(parts)

        return AgentOpinion(agent=self.name, opinion=opinion, confidence=0.75,
            attribution={"metric": "product_mix", "root_cause": root_cause, "deviation": deviation},
            recommendations=recs)

    def _atv_contribution(self, today_products, week_products):
        if not today_products or not week_products:
            return []
        week_map = {p["name"]: p for p in week_products}
        shifts = []
        for p in today_products[:5]:
            name = p.get("name", "")
            today_qty = p.get("qty", 0)
            today_rev = p.get("revenue", 0)
            today_atv = today_rev / max(today_qty, 1)
            w = week_map.get(name, {})
            w_qty = w.get("avg_qty", 0)
            w_rev = w.get("avg_revenue", 0)
            w_atv = w_rev / max(w_qty, 1) if w_qty else today_atv
            if today_qty > 0 and w_qty > 0:
                atv_delta_pct = (today_atv - w_atv) / max(w_atv, 1) * 100
                qty_delta_pct = (today_qty - w_qty) / max(w_qty, 1) * 100
                if abs(atv_delta_pct) > 3 or abs(qty_delta_pct) > 10:
                    shifts.append(
                        f"{name}: {chr(165)}{today_atv:.0f}/unit "
                        f"({'up' if atv_delta_pct>0 else 'down'} {abs(atv_delta_pct):.0f}% vs week), "
                        f"qty {qty_delta_pct:+.0f}%"
                    )
        return shifts[:3]


def _query_product_ranking(date_str=""):
    try:
        db = get_db()
        cur = db.cursor()
        if not date_str:
            cur.execute("SELECT MAX(order_date) FROM orders")
            row = cur.fetchone()
            date_str = str(row[0]) if row and row[0] else ""

        cur.execute("""
            SELECT oi.product_name, SUM(oi.quantity) as qty, SUM(oi.line_total) as revenue,
                   SUM(oi.line_profit) as profit
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.id
            WHERE o.order_date = %s
            GROUP BY oi.product_name
            ORDER BY revenue DESC
        """, (date_str,))
        rows = cur.fetchall()

        BEVERAGE_NAMES = {"latte","americano","cappuccino","mocha","espresso","flat_white",
                          "caramel_macchiato","cold_brew","hot_chocolate","matcha_latte",
                          "milk_tea","chai_latte","earl_grey","english_breakfast","lemonade"}
        bread_ranking = []
        beverage_ranking = []
        bread_rev = 0
        bev_rev = 0
        for r in rows:
            name = str(r[0])
            entry = {"name": name, "qty": int(r[1] or 0),
                     "revenue": round(float(r[2] or 0), 2),
                     "profit": round(float(r[3] or 0), 2)}
            if name.lower() in BEVERAGE_NAMES:
                beverage_ranking.append(entry)
                bev_rev += entry["revenue"]
            else:
                bread_ranking.append(entry)
                bread_rev += entry["revenue"]

        return {
            "bread_ranking": bread_ranking[:5],
            "beverage_ranking": beverage_ranking[:5],
            "sold_bread_sku_count": len(bread_ranking),
            "category": {"Bread": round(bread_rev, 2), "Beverages": round(bev_rev, 2)},
        }
    except Exception as e:
        logger.warning("ProductMix query failed: %s", e)
        return {"bread_ranking": [], "beverage_ranking": [], "category": {}}


def _query_weekly_avg(date_str=""):
    try:
        db = get_db()
        cur = db.cursor()
        if not date_str:
            cur.execute("SELECT MAX(order_date) FROM orders")
            row = cur.fetchone()
            date_str = str(row[0]) if row and row[0] else ""
        d0 = dt.strptime(date_str, "%Y-%m-%d")
        start = (d0 - timedelta(days=6)).strftime("%Y-%m-%d")
        end_excl = (d0 + timedelta(days=1)).strftime("%Y-%m-%d")

        cur.execute("""
            SELECT oi.product_name,
                   SUM(oi.quantity)/7.0 as avg_qty,
                   SUM(oi.line_total)/7.0 as avg_revenue
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.id
            WHERE o.order_date >= %s AND o.order_date < %s
            GROUP BY oi.product_name
            ORDER BY avg_revenue DESC
        """, (start, end_excl))
        rows = cur.fetchall()
        bread_avg = []
        BEVERAGE_NAMES = {"latte","americano","cappuccino","mocha","espresso","flat_white",
                          "caramel_macchiato","cold_brew","hot_chocolate","matcha_latte",
                          "milk_tea","chai_latte","earl_grey","english_breakfast","lemonade"}
        for r in rows:
            name = str(r[0])
            entry = {"name": name, "avg_qty": round(float(r[1] or 0), 2),
                     "avg_revenue": round(float(r[2] or 0), 2)}
            if name.lower() not in BEVERAGE_NAMES:
                bread_avg.append(entry)
        return {"bread_avg": bread_avg[:10]}
    except Exception as e:
        logger.warning("ProductMix weekly avg failed: %s", e)
        return {"bread_avg": []}


def _query_catalog_counts():
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT COUNT(DISTINCT product_name) FROM products WHERE category='bakery'")
        total_bread = cur.fetchone()[0] or 0
        cur.execute("SELECT COUNT(DISTINCT product_name) FROM products WHERE category='beverages'")
        total_bev = cur.fetchone()[0] or 0
        return {"total_bread_sku": total_bread, "total_bev_sku": total_bev}
    except Exception as e:
        logger.warning("Catalog count failed: %s", e)
        return {"total_bread_sku": 0, "total_bev_sku": 0}
