import os, sys, re, logging
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
logger = logging.getLogger("s5.agent.plan_feasibility")

class PlanFeasibilityAgent(BaseAgent):
    """L2: Cross-checks ProductionPlanAgent + MaterialProcurementAgent for plan feasibility."""
    def analyze(self, raw, params, context="", history="", key_metrics=None):
        if not context:
            return AgentOpinion(agent=self.name, opinion="No upstream context for feasibility check.", confidence=0.3)

        total_bake = self._parse_num(context, r'(\d+)\s+total\s+bake\s+units')
        capacity = self._parse_num(context, r'capacity\s+(\d+)')
        if not capacity:
            capacity = 960
        critical = self._parse_list(context, r'Critical:\s*(.+?)(?:\.|$)')
        low = self._parse_list(context, r'Low stock:\s*(.+?)(?:\.|$)')
        bottleneck = re.search(r'Largest gap:\s*(\w[\w\s]+?)\s*\(shortfall\s+([\d.]+)', context)

        issues = []
        if total_bake and capacity:
            peak_util = total_bake / max(capacity, 1) * 100 / 7
            if peak_util > 80:
                issues.append(f"peak day utilization may exceed {peak_util:.0f}%")

        if critical:
            issues.append(f"critical materials: {critical}")
        elif low:
            issues.append(f"low-stock materials: {low}")

        if bottleneck:
            issues.append(f"bottleneck: {bottleneck.group(1)} (gap {bottleneck.group(2)})")

        recommendations = []
        if total_bake and capacity:
            peak_util = total_bake / max(capacity, 1) * 100 / 7
            if peak_util > 80:
                peak_day = max(daily_bakes, key=daily_bakes.get) if daily_bakes else "peak"
                recommendations.append({
                    "action": f"Redistribute production: shift 15% of {peak_day} bake volume to adjacent days to keep utilization below 80%",
                    "urgency": "medium", "time_horizon": "this_week",
                    "rationale": f"Peak day utilization at {peak_util:.0f}% risks overtime and quality issues",
                    "expected_impact": "Reduces peak-day strain, evens out staff and oven load"
                })
        if critical:
            recommendations.append({
                "action": f"Expedite procurement for critical materials: {critical}",
                "urgency": "high", "time_horizon": "tomorrow",
                "rationale": f"Critical stock levels on {critical} could halt production",
                "expected_impact": "Prevents production stoppage from material shortages"
            })
        if bottleneck:
            recommendations.append({
                "action": f"Address bottleneck: {bottleneck.group(1)} is short by {bottleneck.group(2)} units",
                "urgency": "high", "time_horizon": "this_week",
                "rationale": f"Material gap of {bottleneck.group(2)} units will constrain bake quantities",
                "expected_impact": "Unblocks full production capacity"
            })

        if not issues:
            opinion = "Plan is feasible: materials adequate, capacity sufficient for projected bake quantities."
        else:
            opinion = "Plan feasibility concerns: " + "; ".join(issues) + "."

        return AgentOpinion(agent=self.name, opinion=opinion, confidence=0.75,
            attribution={"metric": "plan_feasibility", "feasible": len(issues) == 0,
                         "issues": issues},
            recommendations=recommendations)

    def _parse_num(self, text, pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    def _parse_list(self, text, pattern):
        m = re.search(pattern, text)
        return m.group(1).strip() if m else ""


class DemandRiskAgent(BaseAgent):
    """L2: Identifies risk hotspots from forecast overview + uncertainty."""
    def analyze(self, raw, params, context="", history="", key_metrics=None):
        if not context:
            return AgentOpinion(agent=self.name, opinion="No upstream context for demand risk analysis.", confidence=0.3)

        top_uncertain = self._parse_list(context, r'Most uncertain products:\s*(.+?)(?:\.|$)')
        top_products = self._parse_list(context, r'Top products:\s*(.+?)(?:\.|$)')
        bread_pct = self._parse_num(context, r'Bread\s+(\d+)%')

        risks = []
        if bread_pct and bread_pct > 80:
            risks.append(f"High bread concentration ({bread_pct}%) means beverage demand swings have low impact on total revenue")

        if top_uncertain:
            risks.append(f"Uncertainty concentrated in: {top_uncertain}")

        # Check overlap between top products and most uncertain
        if top_products and top_uncertain:
            top_names = set(re.findall(r'(\w[\w\s]+?)\s*\(', top_products))
            uncertain_names = set(re.findall(r'(\w[\w\s]+?)\s*\(', top_uncertain))
            overlap = top_names & uncertain_names
            if overlap:
                risks.append(f"Risk overlap (high demand + high uncertainty): {', '.join(list(overlap)[:3])}")

        if not risks:
            opinion = "No significant demand-concentration risks identified. Product mix is well distributed."
        else:
            opinion = "Demand risks: " + " | ".join(risks) + "."

        recommendations = []
        # Per-product buffer recommendations for risk-overlap products
        if top_products and top_uncertain:
            top_names = set(re.findall(r'(\w[\w\s]+?)\s*\(', top_products))
            uncertain_names = set(re.findall(r'(\w[\w\s]+?)\s*\(', top_uncertain))
            overlap = top_names & uncertain_names
            for product in list(overlap)[:3]:
                recommendations.append({
                    "action": f"Apply 1.3x buffer to {product} and adjust daily by 10am sell-through",
                    "urgency": "high", "time_horizon": "this_week",
                    "rationale": f"{product} has both high demand and high forecast uncertainty — baking at exact forecast risks stockout on peak days",
                    "expected_impact": f"Prevents lost sales on {product} while containing waste risk through daily adjustment"
                })
            # Low-uncertainty products can use tighter buffer
            low_uncertain = top_names - uncertain_names
            if low_uncertain:
                recommendations.append({
                    "action": f"Apply 1.0x buffer to low-uncertainty products: {', '.join(list(low_uncertain)[:3])}",
                    "urgency": "low", "time_horizon": "this_week",
                    "rationale": "These products have narrower prediction intervals — baking exactly to forecast is safe",
                    "expected_impact": "Reduces waste without meaningful stockout risk"
                })

        if bread_pct and bread_pct > 80:
            recommendations.append({
                "action": "Diversify production: introduce 1-2 new bread or beverage SKUs to reduce concentration risk below 80%",
                "urgency": "low", "time_horizon": "next_30_days",
                "rationale": f"Bread at {bread_pct}% of revenue means any bread-demand shock hits total revenue directly",
                "expected_impact": "Reduces revenue volatility from single-category demand swings"
            })

        return AgentOpinion(agent=self.name, opinion=opinion, confidence=0.75,
            attribution={"metric": "demand_risk", "risk_count": len(risks)},
            recommendations=recommendations)

    def _parse_num(self, text, pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    def _parse_list(self, text, pattern):
        m = re.search(pattern, text)
        return m.group(1).strip() if m else ""


class EfficiencyAgent(BaseAgent):
    """L2: Identifies over/under-baking vs accuracy."""
    def analyze(self, raw, params, context="", history="", key_metrics=None):
        if not context:
            return AgentOpinion(agent=self.name, opinion="No upstream context for efficiency analysis.", confidence=0.3)

        wape = self._parse_num(context, r'WAPE\s+([\d.]+)%')
        top_baked = self._parse_list(context, r'Top baked:\s*(.+?)(?:\.|$)')
        buffer = self._parse_num(context, r'Buffer:\s*([\d.]+)%?')

        findings = []
        if wape and wape > 35:
            findings.append(f"Overall WAPE {wape:.0f}% suggests production plan should use conservative buffer")
        elif wape and wape < 25:
            findings.append(f"Low WAPE ({wape:.0f}%) supports confident production planning")

        if buffer and buffer > 1.3:
            findings.append(f"High buffer ({buffer:.0%}) may cause over-production. Consider reducing to 1.2x for low-uncertainty products")

        if top_baked:
            findings.append(f"Top-baked products: {top_baked}")

        recommendations = []
        if buffer and buffer > 1.3:
            recommendations.append({
                "action": f"Reduce global buffer from {buffer:.0%} to 120%, then apply per-product tiered buffers based on uncertainty",
                "urgency": "medium", "time_horizon": "this_week",
                "rationale": f"Buffer at {buffer:.0%} inflates all products equally, ignoring that low-uncertainty products need less buffer",
                "expected_impact": "Cuts waste on stable products while maintaining protection on volatile ones"
            })
        if wape and wape > 35:
            recommendations.append({
                "action": "Switch to phased daily baking instead of full-week pre-bake until forecast accuracy improves",
                "urgency": "high", "time_horizon": "this_week",
                "rationale": f"WAPE at {wape:.0f}% means forecasts are unreliable beyond 1-2 days; committing a full week creates waste risk",
                "expected_impact": "Reduces waste from inaccurate multi-day forecasts by 30-50%"
            })
        if not top_baked or top_baked == "(0u)":
            recommendations.append({
                "action": "Fix production plan data: top baked products show 0 units — the plan grid may have a data mapping error",
                "urgency": "high", "time_horizon": "tomorrow",
                "rationale": "Zero-unit top products make buffer and allocation analysis unreliable",
                "expected_impact": "Restores accurate production visibility and enables data-driven buffer tuning"
            })

        if not findings:
            opinion = "Production allocation appears aligned with forecast accuracy. No obvious efficiency gaps."
        else:
            opinion = "Efficiency assessment: " + " | ".join(findings) + "."

        return AgentOpinion(agent=self.name, opinion=opinion, confidence=0.70,
            attribution={"metric": "forecast_efficiency", "findings": findings},
            recommendations=recommendations)

    def _parse_num(self, text, pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    def _parse_list(self, text, pattern):
        m = re.search(pattern, text)
        return m.group(1).strip() if m else ""


class WastageRiskAgent(BaseAgent):
    """L2: Projects waste if actual demand hits Q10 (worst case)."""
    def analyze(self, raw, params, context="", history="", key_metrics=None):
        if not context:
            return AgentOpinion(agent=self.name, opinion="No upstream context for wastage risk analysis.", confidence=0.3)

        q50_profit = self._parse_num(context, r'Q50 scenario.*?profit,\s*([\d.-]+)')
        q10_profit = None
        m = re.search(r'q10.*?profit.*?([\d.-]+)', context)
        if m: q10_profit = float(m.group(1))

        total_rev = self._parse_num(context, r'revenue,\s*([\d.]+)')

        findings = []
        if q10_profit is not None and q10_profit < 0:
            findings.append(f"Q10 worst-case: loss of {chr(165)}{abs(q10_profit):.0f}")
        if q50_profit is not None and q10_profit is not None:
            gap = q50_profit - q10_profit
            if gap > 1000:
                findings.append(f"wide profit swing ({chr(165)}{gap:.0f}) between Q10 and Q50 scenarios")

        recommendations = []
        if q10_profit is not None and q10_profit < 0:
            recommendations.append({
                "action": f"Bake in two daily tranches: 70% at opening, 30% at noon after checking morning sell-through",
                "urgency": "high", "time_horizon": "this_week",
                "rationale": f"Q10 worst-case projects {chr(165)}{abs(q10_profit):.0f} loss — committing all bake upfront risks catastrophic waste",
                "expected_impact": f"Limits worst-case waste to 70% of Q10 projection, saving approximately {chr(165)}{abs(q10_profit) * 0.3:.0f}"
            })
        if q50_profit is not None and q10_profit is not None:
            gap = q50_profit - q10_profit
            if gap > 1000:
                recommendations.append({
                    "action": f"Set a daily go/no-go threshold: if 11am sales fall below 60% of daily forecast, cancel afternoon tranche",
                    "urgency": "high", "time_horizon": "this_week",
                    "rationale": f"Wide profit swing of {chr(165)}{gap:.0f} between Q10 and Q50 means daily conditions can flip from profit to loss",
                    "expected_impact": "Prevents baking into a confirmed low-demand day, saving up to full afternoon tranche cost"
                })

        if not findings:
            opinion = "Wastage risk is moderate. Production plan has acceptable downside protection."
        else:
            opinion = "Wastage risk: " + " | ".join(findings) + "."

        return AgentOpinion(agent=self.name, opinion=opinion, confidence=0.70,
            attribution={"metric": "wastage_risk", "findings": findings},
            recommendations=recommendations)

    def _parse_num(self, text, pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None
