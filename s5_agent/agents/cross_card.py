import os, sys, re, logging
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
logger = logging.getLogger("s5.agent.cross_card")

class MetricConflictAgent(BaseAgent):
    """L2: Cross-checks metrics from TrendAgent, DemandAgent, ProductMixAgent for contradictions."""
    def analyze(self, raw, params, context="", history="", key_metrics=None):
        if not context:
            return AgentOpinion(agent=self.name, opinion="No upstream agent context available.", confidence=0.3)

        conflicts = []
        aligned = []

        # Extract numbers from upstream agent opinions
        trend_data = self._parse_trend(context)
        demand_data = self._parse_demand(context)
        mix_data = self._parse_mix(context)

        # If regex parsing fails, fall back to reporting what we received
        if not trend_data and not demand_data:
            return AgentOpinion(agent=self.name,
                opinion=f"Received upstream context but could not parse numeric metrics. Context length: {len(context)} chars.",
                confidence=0.4)

        # Check 1: Revenue vs Orders divergence
        rev_change = trend_data.get("revenue_change_pct", 0)
        order_change = demand_data.get("order_change_pct", 0)
        if rev_change is not None and order_change is not None:
            atv_shift = abs(rev_change - order_change)
            if atv_shift > 2:  # >2% gap = meaningful ATV shift
                direction = "higher" if order_change > rev_change else "lower"
                conflicts.append(
                    f"Revenue ({rev_change:+.1f}%) and orders ({order_change:+.1f}%) diverge by {atv_shift:.1f}% "
                    f"-> ATV is {direction} than usual. ProductMixAgent reports top 3 breads at {mix_data.get('concentration_pct', '?')}% "
                    f"concentration -> mix shift within top products may explain the ATV gap."
                )
            else:
                aligned.append(f"Revenue ({rev_change:+.1f}%) and orders ({order_change:+.1f}%) move together (gap {atv_shift:.1f}%) -> consistent demand pattern.")

        # Check 2: Margin vs Revenue trend
        margin = trend_data.get("margin", 0)
        if margin > 70 and rev_change is not None and rev_change < -5:
            conflicts.append(
                f"Profit margin is very high ({margin:.0f}%) while bread revenue is down ({rev_change:+.1f}%). "
                f"High margins may be masking revenue erosion. Check if cost structure or mix is hiding a demand problem."
            )
        elif margin > 50 and rev_change is not None and rev_change > -5:
            aligned.append(f"Margin ({margin:.0f}%) and revenue trend ({rev_change:+.1f}%) are both healthy.")

        # Build opinion
        opinions = []
        if conflicts:
            opinions.append("CONFLICTS DETECTED: " + " | ".join(conflicts))
        if aligned:
            opinions.append("ALIGNED: " + " | ".join(aligned))

        final = " ".join(opinions) if opinions else "No metric conflicts detected between cards."

        return AgentOpinion(agent=self.name, opinion=final, confidence=0.80,
            attribution={"metric": "cross_card_conflict", "conflicts": len(conflicts), "aligned": len(aligned)})

    def _parse_trend(self, context):
        result = {}
        m = re.search(r'bread revenue.*?\(([+-]?\d+\.?\d*)%\)', context)
        if not m:
            m = re.search(r'today.*?\(([+-]?\d+\.?\d*)%\)', context)
        if not m:
            m = re.search(r'\(([+-]?\d+\.?\d*)%\)', context)
        if m: result["revenue_change_pct"] = float(m.group(1))
        m_atv = re.search(r'ATV.*?\(([+-]?\d+\.?\d*)%\)', context)
        if m_atv: result["atv_change_pct"] = float(m_atv.group(1))
        m = re.search(r'[Mm]argin\s+(\d+\.?\d*)%', context)
        if m: result["margin"] = float(m.group(1))
        return result

    def _parse_demand(self, context):
        result = {}
        try:
            m = re.search(r'[Oo]rders?.*?today\s+(\d+)\s+vs\s+avg\s+(\d+(?:\.\d+)?)', context)
            if m:
                today = float(m.group(1))
                avg = float(m.group(2))
                if avg > 0:
                    result["order_change_pct"] = round((today - avg) / avg * 100, 1)
        except (ValueError, TypeError):
            pass
        return result

    def _parse_mix(self, context):
        result = {}
        m = re.search(r'[Tt]op\s+\d+\s+\w+\s*=\s*(\d+)%', context)
        if m: result["concentration_pct"] = int(m.group(1))
        return result

class CausalChainAgent(BaseAgent):
    """L2: Traces root cause chain using FeatureSensitivityAgent + ExternalFactorsAgent + TrendAgent."""
    def analyze(self, raw, params, context="", history="", key_metrics=None):
        if not context:
            return AgentOpinion(agent=self.name, opinion="No upstream context for causal chain.", confidence=0.3)

        # Parse feature importance
        top_features = []
        for m in re.finditer(r'([\w_]+)\s*\((\d+\.?\d*)%\)', context):
            name = m.group(1)
            pct = float(m.group(2))
            if name not in ("today", "avg", "vs", "orders", "revenue", "margin", "profit", "agent", "confidence"):
                top_features.append((name, pct))

        # Determine if external factors are active (check negation first to avoid false positives)
        no_external = bool(re.search(r'No (?:significant )?external factors', context, re.IGNORECASE))
        no_active = bool(re.search(r'No high-impact external features are active', context, re.IGNORECASE))
        # Only check for positive external signals if no negation found
        has_external = False
        if not no_external and not no_active:
            has_external = bool(re.search(r'(?:is_rainy|is_holiday|is_competitor|is_member_day|is_day1).*?(?:1|active|yes|true)', context, re.IGNORECASE))

        # Build causal chain
        if top_features:
            dominant = top_features[0]
            secondary = top_features[1] if len(top_features) > 1 else None

            if (no_external or no_active) and not has_external:
                chain = (
                    f"Root cause chain: {dominant[0]} ({dominant[1]}%) is the dominant predictor. "
                    f"No external factors are active (FeatureSensitivityAgent + ExternalFactorsAgent confirm). "
                    f"-> Today's outcome is purely momentum-driven by recent sales history. "
                )
                if secondary:
                    chain += f"The secondary driver {secondary[0]} ({secondary[1]}%) reinforces the temporal pattern."
                chain += " This means: (a) today's deviation is NOT a shock, it's a continuation of recent trends; "
                chain += "(b) improvements will take days to show in the model; (c) external levers (promos, new products) "
                chain += "can break the momentum cycle if the trend is negative."
            elif has_external:
                chain = (
                    f"Root cause chain: {dominant[0]} ({dominant[1]}%) is dominant, BUT external factors are active. "
                    f"-> Today's outcome is a mix of momentum + external influence. "
                    f"If the external factor is temporary (e.g., one rainy day), expect reversion when it clears."
                )
            else:
                chain = f"Primary driver: {dominant[0]} ({dominant[1]}%). External factors status unclear from agent data."
        else:
            chain = "Insufficient feature importance data for causal chain analysis."

        return AgentOpinion(agent=self.name, opinion=chain, confidence=0.75,
            attribution={"metric": "causal_chain", "dominant_feature": top_features[0][0] if top_features else ""})


class CrossRiskAgent(BaseAgent):
    """L2: Synthesizes cross-card risks from ProductMixAgent + FeatureSensitivityAgent + ProfitAgent."""
    def analyze(self, raw, params, context="", history="", key_metrics=None):
        if not context:
            return AgentOpinion(agent=self.name, opinion="No upstream context for risk synthesis.", confidence=0.3)

        risks = []

        # Risk 1: Concentration + Lag dominance = self-reinforcing cycle
        concentration = None
        m = re.search(r'[Tt]op\s+\d+\s+\w+\s*=\s*(\d+)%', context)
        if m: concentration = int(m.group(1))

        lag_pct = None
        m = re.search(r'(?:lag_30|30.day|rolling.*?average).*?(\d+\.?\d*)%', context)
        if m: lag_pct = float(m.group(1))

        margin = None
        m = re.search(r'[Mm]argin\s+(\d+\.?\d*)%', context)
        if m: margin = float(m.group(1))

        if concentration and concentration > 75 and lag_pct and lag_pct > 30:
            risks.append(
                f"CONCENTRATION-LAG LOOP: Top products = {concentration}% of revenue, "
                f"and 30-day rolling average drives {lag_pct}% of prediction. "
                f"If a top product dips, the 30-day average drops -> model predicts lower -> actual sales follow lower -> "
                f"self-reinforcing decline. Break the loop by diversifying the product mix or running targeted promos on weakening items."
            )

        # Risk 2: High margin masks structural risk
        if margin and margin > 75 and concentration and concentration > 75:
            risks.append(
                f"MARGIN MASKING: Profit margin at {margin:.0f}% is excellent, but with {concentration}% concentration, "
                f"a single supply or quality issue on a top product could erase that margin quickly. "
                f"High margins should not delay diversification."
            )

        # Risk 3: Single peak dependency
        peak_pct = None
        m = re.search(r'09:00.*?(\d+)%', context)
        if m: peak_pct = int(m.group(1))
        if peak_pct and peak_pct > 18:
            risks.append(
                f"SINGLE-PEAK DEPENDENCY: 09:00 hour alone contributes {peak_pct}% of daily revenue. "
                f"If that peak is missed (late bake, staffing gap, weather), a significant portion of revenue is lost and "
                f"cannot be recovered later in the day. Mitigate by building a second morning peak or extending the 09:00 window."
            )

        if not risks:
            return AgentOpinion(agent=self.name, opinion="No significant cross-card risks identified. The business structure appears balanced.", confidence=0.70)

        opinion = "CROSS-CARD RISKS:\n" + "\n".join(f"{i+1}. {r}" for i, r in enumerate(risks))
        return AgentOpinion(agent=self.name, opinion=opinion, confidence=0.80,
            attribution={"metric": "cross_risk", "risk_count": len(risks)})
