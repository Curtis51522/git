import os, sys, logging
from datetime import datetime as dt
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
from s5_agent.schemas.agent_output import AgentOutput, DataQuality
from s5_agent.schemas.evidence import EvidenceItem
from s5_agent.schemas.recommendation import Recommendation
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
        stock_data_available = bool(data.get("stock_data_available", True))
        if not stock_data_available:
            return AgentOpinion(
                agent=self.name,
                opinion=(
                    "Current material stock could not be verified because the raw-material "
                    "inventory source was unavailable. Procurement quantities and shortage "
                    "alerts have been withheld to avoid treating missing data as zero stock."
                ),
                confidence=0.3,
                attribution={
                    "metric": "material_procurement",
                    "root_cause": "stock_data_unavailable",
                    "deviation": 0,
                },
            )
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
            elif alert in ("critical", "urgent", "low") or (need is not None and stock < need):
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
        below_count = len(critical) + len(low)
        if below_count:
            material_label = "material" if below_count == 1 else "materials"
            parts.append(f"{below_count} {material_label} below required stock.")
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
        stock_data_available = bool(data.get("stock_data_available", True))
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
                id="material_stock_data_available",
                source="material_procurement",
                description="Whether current raw-material stock data was available for procurement analysis",
                value=stock_data_available,
                metadata={"date": params.get("date", "")},
            ),
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
        if not stock_data_available:
            recommendations.append(
                Recommendation(
                    id="material_stock_data_check",
                    action="Verify the raw-material inventory feed before locking the weekly bake.",
                    urgency="high",
                    time_horizon="today",
                    rationale=(
                        "Current material stock was unavailable, so procurement quantities "
                        "and shortage alerts could not be verified."
                    ),
                    expected_impact="Prevents missing stock data from creating false procurement decisions.",
                    evidence_ids=["material_stock_data_available"],
                )
            )
        elif material_watchlist:
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
                "material_stock_data_available": stock_data_available,
                "material_count": len(items),
                "critical_material_count": len(set(critical_materials)),
                "low_material_count": len(set(low_materials)),
                "material_total_order": round(total_order, 2),
                "largest_material_gap": round(largest_gap, 2),
                "largest_gap_material": largest_gap_material,
            },
            evidence_items=evidence_items,
            risks=(
                ["material_data_gap"]
                if not stock_data_available
                else (["material_shortage_risk"] if material_watchlist else [])
            ),
            recommendations=recommendations,
            data_quality=DataQuality(
                freshness="fresh" if items else "missing",
                completeness=1.0 if items else 0.0,
                source_status={"material_procurement": "fresh" if items else "missing"},
                limitations=(
                    ["Current raw-material stock data was unavailable."]
                    if not stock_data_available
                    else []
                ),
            ),
            metadata={
                "critical_materials": sorted(set(critical_materials)),
                "low_materials": sorted(set(low_materials)),
            },
        )


def _query_materials(date_str=""):
    try:
        from s3_scheduling.scheduler import Scheduler, generate_7day_s2_forecast
        from api.module3_scheduling import _load_day1_stock
        if not date_str:
            date_str = dt.now().strftime("%Y-%m-%d")
        start_date = dt.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
        s = Scheduler()
        day1_stock = _load_day1_stock(s.breads, start_date)
        forecast = generate_7day_s2_forecast(start_date)
        result = s.generate_7day_plan(start_date, day1_stock, forecast)
        dm = result["dashboard_materials"]
        return {
            "items": dm.get("items", {}),
            "stock_data_available": bool(dm.get("stock_data_available", True)),
            "error": dm.get("error", ""),
        }
    except Exception as e:
        logger.warning("MaterialProcurement fetch failed: %s", e)
        return {
            "items": {},
            "stock_data_available": False,
            "error": "material_plan_unavailable",
        }
