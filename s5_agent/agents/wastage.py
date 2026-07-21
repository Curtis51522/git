import os, sys, logging
from datetime import datetime as dt, timedelta
from collections import defaultdict
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
from s5_agent.schemas.agent_output import AgentOutput, DataQuality
from s5_agent.schemas.evidence import EvidenceItem
from s5_agent.schemas.recommendation import Recommendation
logger = logging.getLogger("s5.agent.wastage")

WASTAGE_SQL = """
SELECT
    rm.material_name,
    rm.stock_quantity,
    rm.unit,
    rm.unit_price,
    COALESCE(wc.theoretical_consumed, 0) AS theoretical_consumed,
    COALESCE(wc.actual_consumed, 0) AS actual_consumed,
    GREATEST(COALESCE(wc.wastage_qty, 0), 0) AS wastage_qty,
    GREATEST(COALESCE(wc.wastage_rate, 0), 0) AS wastage_rate,
    wc.check_date
FROM raw_materials rm
LEFT JOIN (
    SELECT wc1.*
    FROM material_wastage_log wc1
    INNER JOIN (
        SELECT mw.material_name, MAX(mw.id) AS max_id
        FROM material_wastage_log mw
        WHERE mw.check_date = (
            SELECT MAX(mw2.check_date)
            FROM material_wastage_log mw2
            WHERE mw2.material_name = mw.material_name
              AND mw2.check_date <= %s
        )
        GROUP BY mw.material_name
    ) wc2 ON wc1.id = wc2.max_id
) wc ON rm.material_name = wc.material_name
WHERE rm.track_inventory = 1
ORDER BY GREATEST(COALESCE(wc.wastage_qty, 0), 0) DESC
"""

TREND_SQL = """
SELECT
    mw.id,
    mw.material_name,
    mw.check_date,
    GREATEST(COALESCE(mw.wastage_qty, 0), 0) AS wastage_qty,
    GREATEST(COALESCE(mw.wastage_rate, 0), 0) AS wastage_rate,
    COALESCE(mw.theoretical_consumed, 0) AS theoretical_consumed,
    COALESCE(mw.actual_consumed, 0) AS actual_consumed
FROM material_wastage_log mw
JOIN raw_materials rm ON rm.material_name = mw.material_name
WHERE rm.track_inventory = 1
  AND mw.check_date >= %s AND mw.check_date <= %s
ORDER BY mw.material_name, mw.check_date
"""

ROOT_CAUSE_MAP = {
    "Coffee Beans": "grinder calibration or brewing spillage",
    "Tea": "over-preparation or spillage during steeping",
    "Butter": "measurement errors or refrigeration issues",
    "Eggs": "cracking waste or over-preparation",
    "Milk": "over-pouring or refrigeration spoilage",
    "Bread Flour": "dough waste, scaling errors, or spillage",
    "Cake Flour": "scaling errors or prep spillage",
    "Sugar": "spillage during measuring or dough waste",
    "Yeast": "over-portioning or expired starter",
    "Chocolate": "portioning errors or temperature spoilage",
    "Cream": "over-whipping or refrigeration spoilage",
    "Cheese": "portioning errors or temperature spoilage",
    "Lids": "handling damage or over-dispensing",
    "Cups": "handling damage or over-dispensing",
    "Box": "handling damage or assembly waste",
    "Packaging Bag": "over-dispensing or tearing",
    "Bags": "over-dispensing or tearing",
    "Cup Regular": "handling damage or over-dispensing",
    "Cup Large": "handling damage or over-dispensing",
}


class WastageAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_wastage_trend",
            description="Get 14-day wastage trend per material with root cause analysis",
            parameters={"date": "string"}, primary=True,
            _handler=self._get_wastage))

    async def fetch(self, params):
        date = str(params.get("date", "")) if isinstance(params, dict) else ""
        if not date:
            date = dt.now().strftime("%Y-%m-%d")
        trend_data = _query_trend(date)
        latest_data = _query_wastage(date)
        return {"success": True, "data": {"trend": trend_data, "latest": latest_data}}

    async def _get_wastage(self, date: str = ""):
        return await self.fetch({"date": date})

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {}) if "data" in raw else raw
        trend_data = data.get("trend", {})
        latest_data = data.get("latest", {})
        trend_materials = trend_data.get("materials", [])
        latest_materials = latest_data.get("materials", [])

        if not trend_materials and not latest_materials:
            return AgentOpinion(agent=self.name,
                opinion="No wastage data available.",
                confidence=0.3,
                attribution={"metric": "wastage", "root_cause": "no_data"})

        query_date = str(params.get("date", "")) if isinstance(params, dict) else ""
        if not query_date:
            query_date = dt.now().strftime("%Y-%m-%d")

        # Group trend data by material
        material_days = defaultdict(list)
        for m in trend_materials:
            material_days[m["material_name"]].append(m)

        # Compute per-material trends
        material_analysis = []
        for name, days in sorted(material_days.items()):
            days.sort(key=lambda d: d["check_date"])
            today = [d for d in days if d["check_date"] == query_date]
            today.sort(key=lambda d: int(d.get("id", 0) or 0))
            past_7 = [d for d in days if d["check_date"] < query_date][-7:]

            today_rate = today[-1]["wastage_rate"] if today else 0
            today_qty = today[-1]["wastage_qty"] if today else 0.0
            avg_7_rate = sum(d["wastage_rate"] for d in past_7) / len(past_7) if past_7 else today_rate
            avg_7_qty = sum(d["wastage_qty"] for d in past_7) / len(past_7) if past_7 else today_qty

            # Find matching latest material for cost
            latest_match = next((m for m in latest_materials if m["material_name"] == name), {})
            cost_per_unit = float(latest_match.get("cost_per_unit") or 0)
            unit = latest_match.get("unit", "kg")
            waste_cost_today = today_qty * cost_per_unit

            # Trend direction
            if len(past_7) >= 2 and today_rate > 0:
                delta = today_rate - avg_7_rate
                if delta > 0.01:
                    direction = "rising"
                elif delta < -0.01:
                    direction = "falling"
                else:
                    direction = "stable"
            else:
                direction = "stable"

            # Root cause match
            root_cause = ROOT_CAUSE_MAP.get(name, "general handling or measurement")

            material_analysis.append({
                "name": name,
                "unit": unit,
                "today_rate": today_rate,
                "today_qty": today_qty,
                "today_cost": round(waste_cost_today, 2),
                "avg_7_rate": round(avg_7_rate, 4),
                "avg_7_qty": round(avg_7_qty, 4),
                "direction": direction,
                "root_cause": root_cause,
            })

        # Compute totals
        materials_checked = len(latest_materials) if latest_materials else len(material_analysis)
        total_today_cost = sum(m["today_cost"] for m in material_analysis)
        rising = [m for m in material_analysis if m["direction"] == "rising" and m["today_rate"] >= 0.01]
        stable = [m for m in material_analysis if m["direction"] in ("stable", "falling")]
        has_any_wastage = any(m["today_rate"] > 0 for m in material_analysis)

        if not has_any_wastage:
            top_consumed = sorted(material_analysis, 
                key=lambda m: (sum(d["theoretical_consumed"] for d in material_days.get(m["name"], []) 
                               if d["check_date"] == query_date) or 0), reverse=True)[:3]
            top_names = ", ".join(f"{m['name']}" for m in top_consumed)
            opinion = (f"Wastage check completed: all {materials_checked} materials at 0% wastage. "
                      f"Top consumed today: {top_names}. Material waste records show no logged loss; sales conversion cannot be inferred from wastage data alone.")
            confidence = 0.85
        elif rising:
            # Rising wastage items are the story
            rising_report = []
            for r in sorted(rising, key=lambda x: -x["today_rate"])[:5]:
                trend_note = f"up from {r['avg_7_rate']*100:.1f}% 7-day avg" if r["avg_7_rate"] > 0 else "new this period"
                rising_report.append(
                    f"{r['name']}: {r['today_rate']*100:.1f}% today ({trend_note}), "
                    f"{r['today_qty']:.3f}{r['unit']} lost = CNY{r['today_cost']:.2f}. "
                    f"Likely cause: {r['root_cause']}."
                )
            opinion = (
                f"{materials_checked} materials checked. {len(rising)} with rising wastage. "
                f"Total waste cost today: CNY{total_today_cost:.2f}. "
                + " ".join(rising_report)
            )
            confidence = 0.85
        else:
            # Wastage exists but stable/falling
            highest = max(material_analysis, key=lambda m: m["today_rate"]) if material_analysis else None
            opinion = (
                f"{materials_checked} materials checked. All within normal thresholds. "
                f"Highest: {highest['name']} at {highest['today_rate']*100:.1f}% (stable vs 7-day avg). "
                f"Total waste cost today: CNY{total_today_cost:.2f}. No concerning trends."
            )
            confidence = 0.85

        # Build recommendations
        recs = []
        for r in sorted(rising, key=lambda x: -x["today_rate"])[:3]:
            recs.append({
                "action": f"Investigate {r['name']} ({r['today_rate']*100:.0f}% rate, rising from {r['avg_7_rate']*100:.1f}% 7-day avg, CNY{r['today_cost']:.2f} today)",
                "urgency": "high" if r["today_rate"] > 0.03 else "medium",
                "rationale": f"Wastage rising above 7-day average. Likely cause: {r['root_cause']}.",
                "expected_impact": f"Returning {r['name']} to {r['avg_7_rate']*100:.1f}% average would save CNY{r['today_cost']*0.5:.2f}/day"
            })

        return AgentOpinion(agent=self.name,
            opinion=opinion,
            confidence=confidence,
            attribution={
                "metric": "wastage_trend",
                "materials_checked": materials_checked,
                "rising_count": len(rising),
                "total_waste_cost": round(total_today_cost, 2),
            },
            recommendations=recs)

    def analyze_for_graph(self, raw: dict, params: dict) -> AgentOutput:
        opinion = self.analyze(raw, params)
        data = raw.get("data", {}) if isinstance(raw, dict) and "data" in raw else raw
        if not isinstance(data, dict):
            data = {}
        trend_materials = data.get("trend", {}).get("materials", []) or []
        latest_materials = data.get("latest", {}).get("materials", []) or []
        query_date = str(params.get("date", "") or dt.now().strftime("%Y-%m-%d"))

        material_days = defaultdict(list)
        for material in trend_materials:
            material_days[str(material.get("material_name", ""))].append(material)

        checked_names = {
            str(material.get("material_name", ""))
            for material in latest_materials
            if material.get("material_name")
        }
        if not checked_names:
            checked_names = set(material_days.keys())

        material_rows = []
        check_dates = []
        for name in sorted(checked_names):
            latest = next((item for item in latest_materials if item.get("material_name") == name), {})
            today = [
                item
                for item in material_days.get(name, [])
                if str(item.get("check_date", "")) == query_date
            ]
            today.sort(key=lambda item: int(item.get("id", 0) or 0))
            source = today[-1] if today else latest
            qty = float(source.get("wastage_qty", 0.0) or 0.0)
            rate = float(source.get("wastage_rate", 0.0) or 0.0)
            cost_per_unit = float(latest.get("cost_per_unit", 0.0) or 0.0)
            consumed = float(source.get("theoretical_consumed", 0.0) or latest.get("theoretical_consumed", 0.0) or 0.0)
            source_check_date = str(source.get("check_date") or "")
            if source_check_date:
                check_dates.append(source_check_date)
            material_rows.append(
                {
                    "name": name,
                    "unit": latest.get("unit", "kg"),
                    "wastage_qty": qty,
                    "wastage_rate": rate,
                    "waste_cost": round(qty * cost_per_unit, 2),
                    "consumed": consumed,
                    "check_date": source_check_date,
                }
            )

        material_count = len(checked_names)
        wasted_materials = [row for row in material_rows if row["wastage_qty"] > 0 or row["wastage_rate"] > 0]
        total_waste_cost = round(sum(row["waste_cost"] for row in material_rows), 2)
        latest_record_date = max(check_dates) if check_dates else ""
        has_selected_date_check = query_date in set(check_dates)
        top_consumed = [
            row["name"]
            for row in sorted(material_rows, key=lambda item: item["consumed"], reverse=True)
            if row["name"]
        ][:3]

        if not material_count:
            claim = "No material wastage records are available for this date."
            confidence = 0.3
            freshness = "missing"
            warnings = ["No material wastage records were available for the selected date."]
        elif not wasted_materials:
            claim = (
                f"Material wastage records are clean: {material_count} materials were checked, "
                f"0 materials logged waste, and total recorded waste cost is {chr(165)}{total_waste_cost:.2f}."
            )
            confidence = min(float(opinion.confidence), 0.82)
            freshness = "fresh"
            warnings = []
        else:
            claim = (
                f"Material wastage needs attention: {len(wasted_materials)} of {material_count} materials logged waste, "
                f"with total recorded waste cost at {chr(165)}{total_waste_cost:.2f}."
            )
            confidence = min(float(opinion.confidence), 0.86)
            freshness = "fresh"
            warnings = []

        evidence_items = [
            EvidenceItem(
                id="material_count_checked",
                source="material_wastage",
                description="Number of material wastage records checked for the selected date",
                value=material_count,
                metadata={"date": query_date},
            ),
            EvidenceItem(
                id="wasted_material_count",
                source="material_wastage",
                description="Number of materials with recorded wastage",
                value=len(wasted_materials),
                metadata={"date": query_date},
            ),
            EvidenceItem(
                id="total_waste_cost",
                source="material_wastage",
                description="Total recorded material waste cost for the selected date",
                value=total_waste_cost,
                metadata={"date": query_date},
            ),
            EvidenceItem(
                id="top_consumed_materials",
                source="material_wastage",
                description="Materials with the highest theoretical consumption in the wastage check",
                value=top_consumed,
                metadata={"date": query_date},
            ),
            EvidenceItem(
                id="latest_wastage_record_date",
                source="material_wastage",
                description="Latest material wastage record date available up to the selected date",
                value=latest_record_date,
                metadata={
                    "requested_date": query_date,
                    "has_selected_date_check": has_selected_date_check,
                },
            ),
        ]

        sorted_wasted_materials = sorted(wasted_materials, key=lambda item: item["waste_cost"], reverse=True)
        top_wasted_materials = [
            {
                "name": material["name"],
                "wastage_qty": material["wastage_qty"],
                "unit": material["unit"],
                "waste_cost": material["waste_cost"],
                "wastage_rate": material["wastage_rate"],
                "rate_available": material["wastage_rate"] > 0,
            }
            for material in sorted_wasted_materials[:3]
        ]
        additional_wasted_materials = [
            {
                "name": material["name"],
                "wastage_qty": material["wastage_qty"],
                "unit": material["unit"],
                "waste_cost": material["waste_cost"],
                "wastage_rate": material["wastage_rate"],
                "rate_available": material["wastage_rate"] > 0,
            }
            for material in sorted_wasted_materials[3:]
        ]

        recommendations = []
        if material_count and not wasted_materials:
            recommendations.append(
                Recommendation(
                    id="wastage_zero_record_audit",
                    action="Verify zero-waste material entries against actual count logs before treating the day as operationally loss-free.",
                    urgency="low",
                    time_horizon="ongoing",
                    rationale="Zero recorded waste is positive, but it confirms only the material wastage log, not production yield or sales conversion.",
                    expected_impact="Maintains confidence in zero-waste reporting without overclaiming that all material cost became sales.",
                    evidence_ids=["material_count_checked", "wasted_material_count", "total_waste_cost"],
                )
            )
        for material in sorted_wasted_materials[:3]:
            if material["wastage_rate"] > 0:
                waste_detail = (
                    f"material waste at {material['wastage_rate'] * 100:.1f}% "
                    f"with {chr(165)}{material['waste_cost']:.2f} recorded cost"
                )
            else:
                waste_detail = (
                    f"{material['wastage_qty']:.3f}{material['unit']} material waste "
                    f"with {chr(165)}{material['waste_cost']:.2f} recorded cost; "
                    "wastage rate is unavailable because theoretical consumption is zero"
                )
            recommendations.append(
                Recommendation(
                    id=f"wastage_investigate_{material['name'].lower().replace(' ', '_')}",
                    action=f"Investigate recorded waste for {material['name']} before the next production cycle.",
                    urgency="high" if material["wastage_rate"] >= 0.03 else "medium",
                    time_horizon="today",
                    rationale=f"{material['name']} logged {waste_detail}.",
                    expected_impact="Targets the material with direct recorded loss instead of applying broad process changes.",
                    evidence_ids=["wasted_material_count", "total_waste_cost"],
                )
            )

        return AgentOutput(
            agent_name="WastageAgent",
            claim=claim,
            confidence=confidence,
            metrics={
                "material_count_checked": material_count,
                "wasted_material_count": len(wasted_materials),
                "total_waste_cost": total_waste_cost,
                "top_consumed_materials": top_consumed,
                "top_wasted_materials": top_wasted_materials,
                "additional_wasted_materials": additional_wasted_materials,
                "requested_date": query_date,
                "latest_wastage_record_date": latest_record_date,
                "has_selected_date_wastage_check": has_selected_date_check,
            },
            evidence_items=evidence_items,
            risks=["material_wastage_risk"] if wasted_materials else [],
            recommendations=recommendations,
            data_quality=DataQuality(
                freshness=freshness,
                completeness=1.0 if material_count else 0.0,
                warnings=warnings,
                source_status={"material_wastage": freshness},
            ),
            metadata={"top_consumed_materials": top_consumed},
        )


def _query_wastage(date=""):
    if not date:
        date = dt.now().strftime("%Y-%m-%d")
    try:
        from db.mysql_client import get_db
        db = get_db()
        cursor = db.cursor()
        cursor.execute(WASTAGE_SQL, (date,))
        materials = []
        for row in cursor.fetchall():
            materials.append({
                "material_name": row[0],
                "current_stock": float(row[1]) if row[1] else 0,
                "unit": row[2] or "kg",
                "cost_per_unit": float(row[3]) if row[3] else None,
                "theoretical_consumed": float(row[4]) if row[4] else 0,
                "actual_consumed": float(row[5]) if row[5] else 0,
                "wastage_qty": float(row[6]) if row[6] else 0,
                "wastage_rate": float(row[7]) if row[7] else 0,
                "check_date": str(row[8]) if row[8] else None,
            })
        cursor.close()
        db.close()
        return {"materials": materials}
    except Exception as e:
        logger.warning("WASTAGE_DB: error=%s", e)
        return {"materials": []}


def _query_trend(date=""):
    if not date:
        date = dt.now().strftime("%Y-%m-%d")
    start_date_dt = dt.strptime(date, "%Y-%m-%d") - timedelta(days=14)
    start_date = start_date_dt.strftime("%Y-%m-%d")
    try:
        from db.mysql_client import get_db
        db = get_db()
        cursor = db.cursor()
        cursor.execute(TREND_SQL, (start_date, date))
        materials = []
        for row in cursor.fetchall():
            materials.append({
                "id": int(row[0]) if row[0] else 0,
                "material_name": row[1],
                "check_date": str(row[2]) if row[2] else None,
                "wastage_qty": float(row[3]) if row[3] else 0,
                "wastage_rate": float(row[4]) if row[4] else 0,
                "theoretical_consumed": float(row[5]) if row[5] else 0,
                "actual_consumed": float(row[6]) if row[6] else 0,
            })
        cursor.close()
        db.close()
        return {"materials": materials}
    except Exception as e:
        logger.warning("WASTAGE_TREND_DB: error=%s", e)
        return {"materials": []}
