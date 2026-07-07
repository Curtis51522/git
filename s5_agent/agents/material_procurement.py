import os, sys, logging
from datetime import datetime as dt, timedelta
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
from s5_agent.schemas.agent_output import AgentOutput, DataQuality
from s5_agent.schemas.evidence import EvidenceItem
from s5_agent.schemas.recommendation import Recommendation
from db.mysql_client import get_db
logger = logging.getLogger("s5.agent.material_procurement")

class MaterialProcurementAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_material_procurement",
            description="Get material procurement status",
            parameters={"date": "string"}, primary=True,
            _handler=self._get_materials))

    async def _get_materials(self, date: str = ""):
        return _query_materials(date)

    async def fetch(self, params):
        date_str = str(params.get("date", "")) if isinstance(params, dict) else ""
        data = _query_materials(date_str)
        return {"success": True, "data": data, "tool": "material_procurement"}

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {}) if "data" in raw else raw
        items = data.get("items", {})
        if not items:
            return AgentOpinion(agent=self.name,
                opinion="No material procurement data available.",
                confidence=0.3,
                attribution={"metric": "material_procurement", "root_cause": "no_data", "deviation": 0})

        critical = []
        low = []
        ok_count = 0
        total_order = 0
        biggest_gap = {"name": "", "gap": 0, "unit": "kg"}

        for name, info in items.items():
            alert = str(info.get("alert", info.get("status", ""))).lower()
            need_raw = info.get("weekly_need")
            need = float(need_raw) if need_raw is not None else None
            stock = float(info.get("current_stock", 0))
            order = float(info.get("to_order", 0))
            total_order += order
            if need is not None:
                gap = need - stock
                if gap > biggest_gap["gap"]:
                    biggest_gap = {"name": name, "gap": gap, "unit": info.get("unit", "kg")}
            note = info.get("note", "")
            if note:
                ok_count += 1  # no estimate materials are informational only
            elif alert in ("critical", "urgent", "low") or (need is not None and stock < need * 0.5):
                if stock == 0 or (need is not None and stock < need * 0.3):
                    critical.append({"name": name, "need": need or 0, "stock": stock, "unit": info.get("unit","kg")})
                else:
                    low.append({"name": name, "need": need or 0, "stock": stock, "unit": info.get("unit","kg")})
            else:
                ok_count += 1

        ingr_items = []
        pkg_items = []
        for name, info in items.items():
            need_raw = info.get("weekly_need")
            if need_raw is None or info.get("note", ""):
                continue
            need = float(need_raw)
            stock = float(info.get("current_stock", 0))
            unit = info.get("unit", "kg")
            entry = dict(name=name, need=need, stock=stock, unit=unit)
            if unit == "pcs":
                pkg_items.append(entry)
            else:
                ingr_items.append(entry)
        _fmt = lambda v, u: f"{v:.1f}" if u != "pcs" else f"{v:.0f}"
        def fmt_group(items_list, top_n):
            items_list.sort(key=lambda x: -x["need"])
            parts_list = []
            for ti in items_list[:top_n]:
                gap = ti["need"] - ti["stock"]
                gap_sign = "+" if gap >= 0 else ""
                parts_list.append(f"{ti['name']} (need {_fmt(ti['need'],ti['unit'])}{ti['unit']}, stock {_fmt(ti['stock'],ti['unit'])}{ti['unit']}, gap {gap_sign}{_fmt(gap,ti['unit'])}{ti['unit']})")
            return "; ".join(parts_list) if parts_list else ""
        ingr_line = "Top ingredients: " + fmt_group(ingr_items, 5) if ingr_items else ""
        pkg_line = "Top packaging: " + fmt_group(pkg_items, 3) if pkg_items else ""

        parts = [f"{len(items)} materials monitored."]
        if ingr_line:
            parts.append(ingr_line)
        if pkg_line:
            parts.append(pkg_line)
        if critical:
            parts.append(f"Critical: " + ", ".join(
                f"{c['name']} (need {c['need']:.1f}, have {c['stock']:.1f})" for c in critical
            ))
        if low:
            parts.append(f"Low stock: " + ", ".join(
                f"{l['name']} (need {l['need']:.1f}, have {l['stock']:.1f})" for l in low
            ))
        parts.append(f"{ok_count} materials adequate.")
        parts.append(f"Total to order: {total_order:.0f} units.")
        if biggest_gap["name"]:
            parts.append(f"Largest gap: {biggest_gap['name']} (shortfall {_fmt(biggest_gap['gap'],biggest_gap.get('unit','kg'))}).")

        opinion = " ".join(parts)

        return AgentOpinion(agent=self.name, opinion=opinion, confidence=0.80,
            attribution={"metric": "material_procurement", "root_cause": "materials_analysis",
                         "critical_count": len(critical), "low_count": len(low),
                         "total_order": total_order})

    def analyze_for_graph(self, raw: dict, params: dict) -> AgentOutput:
        opinion = self.analyze(raw, params)
        data = raw.get("data", {}) if isinstance(raw, dict) and "data" in raw else raw
        if not isinstance(data, dict):
            data = {}

        items = data.get("items", {}) or {}
        critical_materials = []
        low_materials = []
        total_order = 0.0
        largest_gap = 0.0
        largest_gap_material = ""

        for name, info in items.items():
            alert = str(info.get("alert", info.get("status", ""))).lower()
            need_raw = info.get("weekly_need")
            need = float(need_raw) if need_raw is not None else None
            stock = float(info.get("current_stock", 0.0) or 0.0)
            order = float(info.get("to_order", 0.0) or 0.0)
            total_order += order

            if need is not None:
                gap = need - stock
                if gap > largest_gap:
                    largest_gap = gap
                    largest_gap_material = str(name)

            if alert in ("critical", "urgent"):
                critical_materials.append(str(name))
            elif alert == "low":
                low_materials.append(str(name))
            elif need is not None and stock < need:
                low_materials.append(str(name))

        evidence_items = [
            EvidenceItem(
                id="material_count",
                source="material_procurement",
                description="Number of materials checked for the production plan",
                value=len(items),
                metadata={"date": params.get("date", "")},
            ),
            EvidenceItem(
                id="material_low_count",
                source="material_procurement",
                description="Materials below required stock for the planning horizon",
                value=len(set(low_materials)),
                metadata={"date": params.get("date", ""), "materials": sorted(set(low_materials))},
            ),
            EvidenceItem(
                id="material_critical_count",
                source="material_procurement",
                description="Materials in critical procurement status",
                value=len(set(critical_materials)),
                metadata={"date": params.get("date", ""), "materials": sorted(set(critical_materials))},
            ),
            EvidenceItem(
                id="material_total_order",
                source="material_procurement",
                description="Total material order quantity required",
                value=round(total_order, 2),
                metadata={"date": params.get("date", "")},
            ),
        ]

        recommendations = []
        material_watchlist = sorted(set(critical_materials + low_materials))
        if material_watchlist:
            recommendations.append(
                Recommendation(
                    id="material_procurement_action_1",
                    action=(
                        "Review procurement for low-stock materials before locking the weekly bake: "
                        + ", ".join(material_watchlist[:5])
                    ),
                    urgency="high" if critical_materials else "medium",
                    time_horizon="this_week",
                    rationale="The production plan depends on materials that are below the required weekly stock level.",
                    expected_impact="Reduces the chance of production shortfall caused by material constraints.",
                    evidence_ids=["material_low_count", "material_critical_count", "material_total_order"],
                )
            )

        return AgentOutput(
            agent_name="MaterialProcurementAgent",
            claim=opinion.opinion,
            confidence=float(opinion.confidence),
            metrics={
                "material_count": len(items),
                "critical_material_count": len(set(critical_materials)),
                "low_material_count": len(set(low_materials)),
                "material_total_order": round(total_order, 2),
                "largest_material_gap": round(largest_gap, 2),
                "largest_gap_material": largest_gap_material,
            },
            evidence_items=evidence_items,
            risks=["material_shortage_risk"] if material_watchlist else [],
            recommendations=recommendations,
            data_quality=DataQuality(
                freshness="fresh" if items else "missing",
                completeness=1.0 if items else 0.0,
                source_status={"material_procurement": "fresh" if items else "missing"},
            ),
            metadata={
                "critical_materials": sorted(set(critical_materials)),
                "low_materials": sorted(set(low_materials)),
            },
        )


def _query_materials(date_str=""):
    try:
        from s3_scheduling.scheduler import Scheduler, generate_7day_s2_forecast
        if not date_str:
            date_str = dt.now().strftime("%Y-%m-%d")
        start_dt = dt.strptime(date_str, "%Y-%m-%d")
        if start_dt.weekday() != 0:
            start_dt -= timedelta(days=start_dt.weekday())
        start_date = start_dt.strftime("%Y-%m-%d")
        s = Scheduler()
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT product_name, stock_day1 FROM products WHERE category='bakery'")
        day1_stock = {str(r[0]): int(r[1] or 0) for r in cur.fetchall()}
        for p in s.breads:
            if p not in day1_stock:
                day1_stock[p] = 0
        forecast = generate_7day_s2_forecast(start_date)
        result = s.generate_7day_plan(start_date, day1_stock, forecast)
        dm = result["dashboard_materials"]
        return {"items": dm.get("items", {})}
    except Exception as e:
        logger.warning("MaterialProcurement fetch failed: %s", e)
        return {"items": {}}
