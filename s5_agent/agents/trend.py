import os, sys, logging
from datetime import datetime as dt, timedelta
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
from db.mysql_client import get_db
logger = logging.getLogger("s5.agent.trend")

class TrendAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_weekly_trend", description="Get 7-day revenue/profit trend",
            parameters={"date": "string"}, primary=True, _handler=self._get_trend))

    async def _get_trend(self, date: str = ""):
        try:
            db = get_db()
            cur = db.cursor()
            if not date:
                cur.execute("SELECT MAX(order_date) FROM orders")
                row = cur.fetchone()
                date = str(row[0]) if row and row[0] else ""
            d0 = dt.strptime(date, "%Y-%m-%d")
            trend_dates = []
            trend_bread = []
            trend_orders = []
            trend_avg = []
            for i in range(6, -1, -1):
                dd = (d0 - timedelta(days=i)).strftime("%Y-%m-%d")
                trend_dates.append(dd[5:])
                cur.execute(
                    "SELECT p.category, COALESCE(SUM(oi.line_total), 0) "
                    "FROM orders o JOIN order_items oi ON oi.order_id = o.id "
                    "JOIN products p ON oi.product_name = p.product_name "
                    "WHERE o.order_date = %s GROUP BY p.category",
                    (dd,))
                day_bread = 0
                for crow in cur.fetchall():
                    if crow[0] == "bakery":
                        day_bread = round(float(crow[1]), 2)
                trend_bread.append(day_bread)
                cur.execute(
                    "SELECT COUNT(*), COALESCE(SUM(total_amount),0) FROM orders WHERE order_date = %s",
                    (dd,))
                orow = cur.fetchone()
                day_orders = int(orow[0] or 0)
                day_rev = float(orow[1] or 0)
                trend_orders.append(day_orders)
                trend_avg.append(round(day_rev / day_orders, 2) if day_orders else 0)
            cur.execute(
                "SELECT COALESCE(SUM(total_amount),0) FROM orders WHERE order_date = %s",
                (date,))
            today_rev = round(float(cur.fetchone()[0] or 0), 2)
            yesterday = (d0 - timedelta(days=1)).strftime("%Y-%m-%d")
            cur.execute(
                "SELECT COALESCE(SUM(total_amount),0) FROM orders WHERE order_date = %s",
                (yesterday,))
            y_rev = round(float(cur.fetchone()[0] or 0), 2)
            rev_change = round((today_rev - y_rev) / y_rev * 100, 1) if y_rev else 0
            return {
                "dates": trend_dates,
                "bread": trend_bread,
                "orders": trend_orders,
                "avg_order": trend_avg,
                "today_revenue": today_rev,
                "revenue_change": rev_change,
            }
        except Exception as e:
            logger.warning("Trend DB fetch failed: %s", e)
        return {"dates": [], "bread": [], "orders": [], "avg_order": [], "today_revenue": 0, "revenue_change": 0}

    async def fetch(self, params):
        date_str = str(params.get("date", "")) if isinstance(params, dict) else ""
        data = await self._get_trend(date=date_str)
        return {"success": True, "data": data, "tool": "get_weekly_trend"}

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {}) if "data" in raw else raw
        bread = data.get("bread", [])
        orders = data.get("orders", [])
        avg_order = data.get("avg_order", [])
        dates = data.get("dates", [])
        today_rev = data.get("today_revenue", 0)
        rev_change = data.get("revenue_change", 0)

        if not bread or len(bread) < 3:
            return AgentOpinion(agent=self.name,
                opinion="Insufficient trend data (need >=3 days).",
                confidence=0.3,
                attribution={"metric": "trend", "root_cause": "no_data", "deviation": 0})

        historical = bread[:-1]
        today_val = bread[-1]
        avg7 = sum(historical) / len(historical) if historical else today_val
        pct_vs_avg = (today_val - avg7) / max(avg7, 1) * 100

        mid = len(historical) // 2
        first_half = sum(historical[:mid]) / max(mid, 1)
        second_half = sum(historical[mid:]) / max(len(historical) - mid, 1)
        if second_half > first_half * 1.05:
            direction = "rising"
        elif second_half < first_half * 0.95:
            direction = "falling"
        else:
            direction = "stable"

        hist_orders = orders[:-1] if len(orders) > 1 else []
        today_orders = orders[-1] if orders else 0
        avg_orders = sum(hist_orders) / len(hist_orders) if hist_orders else today_orders
        order_change_pct = (today_orders - avg_orders) / max(avg_orders, 1) * 100

        atv_note = ""
        if avg_order and len(avg_order) > 1:
            hist_atv = avg_order[:-1]
            today_atv = avg_order[-1]
            avg_atv = sum(hist_atv) / len(hist_atv) if hist_atv else today_atv
            atv_change_pct = (today_atv - avg_atv) / max(avg_atv, 1) * 100
            atv_note = f" ATV: today {chr(165)}{today_atv:.2f} vs avg {chr(165)}{avg_atv:.2f} ({atv_change_pct:+.1f}%)."
            if abs(atv_change_pct) > 10:
                direction_word = "higher" if atv_change_pct > 0 else "lower"
                atv_note += f" Customers spending {direction_word} per visit than usual."

        opinion = (
            f"7-day bread revenue: today {chr(165)}{today_val:.0f} vs avg {chr(165)}{avg7:.0f} "
            f"({pct_vs_avg:+.1f}%). "
            f"Trend: {direction} (first half avg {chr(165)}{first_half:.0f}, second half {chr(165)}{second_half:.0f}). "
            f"Orders: today {today_orders:.0f} vs avg {avg_orders:.1f} ({order_change_pct:+.1f}%)." + atv_note
        )

        recs = []
        root_cause = "stable_trend"
        deviation = round(pct_vs_avg, 1)

        if direction == "falling" and pct_vs_avg < -10:
            root_cause = "revenue_decline"
            recs.append({
                "action": "Investigate declining trend: check if competitors, weather, or product quality issues",
                "urgency": "high",
                "rationale": f"Revenue trending down: second half avg {chr(165)}{second_half:.0f} vs first half {chr(165)}{first_half:.0f}"
            })
        elif direction == "rising" and pct_vs_avg > 15:
            root_cause = "revenue_surge"
            recs.append({
                "action": "Sustain growth: ensure production capacity matches rising demand",
                "urgency": "medium",
                "rationale": f"Revenue trending up: today {chr(165)}{today_val:.0f}, +{pct_vs_avg:.1f}% vs 7-day avg"
            })
        elif abs(pct_vs_avg) > 20:
            root_cause = "revenue_volatile"
            recs.append({
                "action": "Check for one-time events (holiday, promotion, weather spike)",
                "urgency": "medium",
                "rationale": f"Today deviates {pct_vs_avg:+.1f}% from 7-day average"
            })

        return AgentOpinion(agent=self.name, opinion=opinion, confidence=0.80,
            attribution={"metric": "trend", "root_cause": root_cause, "deviation": deviation,
                        "revenue_change_pct": round(pct_vs_avg, 1),
                        "order_change_pct": round(order_change_pct, 1)},
            recommendations=recs)
