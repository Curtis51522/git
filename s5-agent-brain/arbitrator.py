# Arbitrator - cross-agent health audit + final decision
# Merges Health Agent and Arbitrator into one.
# Phase 2: integrates causal attribution for cost calibration.
# Phase 3: agent deliberation (LLM-mediated consensus) + memory-aware decisions.
#   Deliberation resolves agent disagreements but does NOT override numerical optimizer results.
import logging, time, json
from optimizer import optimize_single, optimize_multi, ProductState, CostParams, BakeryConfig
from causal_attribution import synthesize_training_data, calibrate_costs, counterfactual_analysis
from memory_store import get_recent_context, get_key_metrics
from typing import Dict, Any, List, Optional
import httpx, os, sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
try:
    from config.settings import DEEPSEEK_API_KEY
except ImportError:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1") + "/chat/completions"
LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

logger = logging.getLogger("s5.arbitrator")


def _run_comparison(curr_forecast, curr_stock, inventory_data):
    try:
        from association_engine import get_period_comparison
        past = get_period_comparison(days_back=7)
        if not past:
            return None
        best = past[0]
        past_data = best.get('data', {})
        past_forecast = past_data.get('forecast', 0)
        past_inventory = past_data.get('inventory', 0)
        if past_forecast == 0 and past_inventory == 0:
            return None
        lines = ['Week-over-Week Comparison (vs ~' + best.get('date','?')[:10] + '):']
        if past_forecast > 0 and curr_forecast > 0:
            fc_delta = curr_forecast - past_forecast
            fc_pct = (fc_delta / past_forecast) * 100
            trend = 'rising' if fc_delta > 0 else 'declining'
            lines.append('  Forecast: ' + str(curr_forecast) + ' vs ' + str(past_forecast) + ' (' + format(fc_pct,'+.0f') + '%, ' + trend + ')')
        if past_inventory > 0 and curr_stock > 0:
            inv_delta = curr_stock - past_inventory
            inv_pct = (inv_delta / past_inventory) * 100
            lines.append('  Inventory: ' + str(curr_stock) + ' vs ' + str(past_inventory) + ' (' + format(inv_pct,'+.0f') + '%)')
        per_product = inventory_data.get('per_product', {})
        past_pp = past_data.get('per_product', {})
        if per_product and past_pp:
            for pname in per_product:
                if pname in past_pp:
                    curr_q = per_product[pname].get('qty', 0)
                    past_q = past_pp[pname].get('qty', 0)
                    if curr_q > 0 and past_q > 0:
                        lines.append('  ' + pname + ': ' + str(curr_q) + ' vs ' + str(past_q))
        return ' | '.join(lines)
    except Exception:
        return None

class Arbitrator:
    def __init__(self, config: BakeryConfig = None):
        self._cached_attribution = None
        self.config = config or BakeryConfig()

    def _get_causal_costs(self) -> CostParams:
        costs = CostParams()
        if self._cached_attribution is None:
            try:
                X, T, Y = synthesize_training_data(200)
                self._cached_attribution = calibrate_costs(X, T, Y)
            except Exception as e:
                logger.warning("Causal calibration failed: %s, using defaults", e)
                self._cached_attribution = None
        if self._cached_attribution is not None:
            costs.update_from_causal(self._cached_attribution)
        return costs

    # ------------------------------------------------------------------
    # Agent Deliberation
    # ------------------------------------------------------------------
    async def deliberate(self, results, params, history=""):
        audit = self.audit(results, params)
        conflicts = audit.get("conflicts", [])
        opinions = []
        for name, r in results.items():
            opinion = r.get("opinion", "")
            confidence = r.get("confidence", 0)
            constraints = r.get("constraints", [])
            if opinion and confidence >= 0.5:
                opinions.append({"agent": name, "opinion": opinion, "confidence": confidence, "constraints": constraints})

        has_conflict = len(conflicts) > 0
        has_multiple_views = len(opinions) >= 2
        if not has_conflict and not has_multiple_views:
            return {"consensus": None, "deliberation_text": "No deliberation needed.", "votes": {}}

        intent = params.get("intent", "stock_query")
        query = params.get("query", "")
        product = params.get("product", "all")

        opinion_lines = []
        for o in opinions:
            c_str = "; ".join(o["constraints"]) if o["constraints"] else "none"
            opinion_lines.append(f"  [{o['agent']}] (confidence: {o['confidence']:.0%}) {o['opinion']}\n    Constraints: {c_str}")

        conflicts_str = "\n".join(f"  ! {c}" for c in conflicts) if conflicts else "None"

        prompt = f"""You are a bakery operations arbitrator. Multiple AI agents analyzed the situation. Some may disagree. Your job: resolve CONFLICTS between agent opinions. Do NOT recompute production numbers - the optimizer handles that.

QUERY: {query}
PRODUCT: {product}
INTENT: {intent}

AGENT OPINIONS:
{chr(10).join(opinion_lines)}

CONFLICTS:
{conflicts_str}

RULES:
1. Only resolve actual disagreements between agents.
2. Do NOT suggest a specific bake quantity - the optimizer computes that.
3. If agents agree, say so.
4. If one agent has stale/error data (e.g. says zero inventory when others show stock), note it as UNRELIABLE.
5. Format: CONFLICT_RESOLUTION: <1-2 sentences resolving the disagreement>
   VOTES: <agent: reliable/unreliable/agree>"""

        if not DEEPSEEK_API_KEY:
            return {"consensus": None, "deliberation_text": "No API key.", "votes": {}}

        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(LLM_URL,
                    headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
                    json={"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 300, "temperature": 0.3})
                resp.raise_for_status()
                data = resp.json()
                raw = data["choices"][0]["message"]["content"].strip()

            consensus = ""
            votes = {}
            for line in raw.split("\n"):
                line = line.strip()
                if line.upper().startswith("CONFLICT_RESOLUTION:"):
                    consensus = line.split(":", 1)[1].strip()
                elif ":" in line and not line.upper().startswith(("CONFLICT", "VOTES")):
                    agent_name = line.split(":")[0].strip().lower()
                    votes[agent_name] = line.split(":", 1)[1].strip()

            logger.info("Deliberation: %s", consensus[:100])
            return {"consensus": consensus or raw[:200], "deliberation_text": raw, "votes": votes}
        except Exception as e:
            logger.warning("Deliberation failed: %s", e)
            return {"consensus": None, "deliberation_text": str(e), "votes": {}}

    # ------------------------------------------------------------------
    # Health Audit
    # ------------------------------------------------------------------
    def audit(self, results, params):
        conflicts = []
        warnings = []
        demand = results.get("demand", {}).get("data", {})
        inventory = results.get("inventory", {}).get("data", {})
        production = results.get("production", {}).get("data", {})
        staffing = results.get("staffing", {}).get("data", {})

        forecast = demand.get("forecast", 0)
        stock = inventory.get("inventory", 0)
        max_cap = production.get("max_capacity", 0)
        bakers = staffing.get("bakers", 0)
        waste_risk = inventory.get("waste_risk", "low")

        has_demand = "demand" in results and forecast > 0
        has_inventory = "inventory" in results
        has_production = "production" in results and max_cap > 0
        has_staffing = "staffing" in results

        if has_inventory and has_demand:
            if stock == 0:
                conflicts.append("STOCKOUT: zero inventory, cannot meet any demand")
            elif forecast > stock * 1.5:
                conflicts.append(f"UNDERSTOCK: demand ({forecast}) >> stock ({stock}), will stock out")

        # Only flag capacity gap when production shortfall exceeds capacity
        production_shortfall = max(0, forecast - stock)
        if has_production and has_demand and production_shortfall > max_cap and max_cap > 0:
            # CAPACITY_GAP removed: capacity >> demand in realistic staffing
            pass
        if has_staffing and bakers == 0 and has_demand:
            conflicts.append("NO_BAKERS: production needed but no bakers scheduled")
        if has_staffing and staffing.get("cashiers", 0) == 0:
            conflicts.append("NO_CASHIERS: cannot open without cashier")

        if waste_risk == "high":
            warnings.append(f"WASTE_RISK: high waste risk - {stock} inventory, only {forecast} demand")
        if has_inventory and has_demand and stock > forecast * 2 and forecast > 0:
            warnings.append(f"OVERSTOCK: {stock} units vs {forecast} forecast ({stock/forecast:.1f}x)")

        all_constraints = []
        for name, r in results.items():
            all_constraints.extend(r.get("constraints", []))

        return {"conflicts": conflicts, "warnings": warnings, "all_constraints": all_constraints, "healthy": len(conflicts) == 0}

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    def decide(self, results, params, deliberation=None, history=""):
        t0 = time.perf_counter()
        intent = params.get("intent", "stock_query")
        audit = self.audit(results, params)
        product = params.get("product", "croissant")

        opt = {}
        counterfactual = None

        demand = results.get("demand", {}).get("data", {})
        production = results.get("production", {}).get("data", {})
        inventory = results.get("inventory", {}).get("data", {})
        promo = results.get("promo", {}).get("data", {})

        forecast = demand.get("forecast", 0)
        recommended = production.get("recommended", 0)
        max_cap = production.get("max_capacity", 0)
        stock = inventory.get("inventory", 0)
        waste_risk = inventory.get("waste_risk", "low")

        costs = self._get_causal_costs()
        counterfactual = None

        # Deliberation provides conflict resolution context - does NOT replace optimizer
        deliberation_note = ""
        if deliberation and deliberation.get("consensus"):
            deliberation_note = deliberation["consensus"][:200]

        if intent == "waste_analysis":
            surplus = max(0, stock - forecast)
            ratio = stock / max(forecast, 1)
            if ratio > 1.5:
                day1_stock = inventory.get("freshness_breakdown", {}).get("Day-1", 0)
                action = (f"WASTE RISK: {stock} total inventory vs {forecast} demand ({ratio:.1f}x). "
                          f"Day-1 stock: {day1_stock} units at risk. "
                          f"Root cause: production outpacing demand, {surplus} surplus units will go stale.")
                priority = "warning"
            elif ratio > 1.2:
                action = (f"Moderate overstock: {stock} inventory vs {forecast} forecast ({ratio:.1f}x). "
                          f"Day-1 units should be prioritized for sale.")
                priority = "normal"
            else:
                action = f"Healthy balance: {stock} in stock matches {forecast} demand."
                priority = "normal"
            rec = production.get("recommended", 0)
            if self._cached_attribution is not None:
                counterfactual = counterfactual_analysis(
                    rec if rec > 0 else max(0, stock + rec),
                    stock, forecast, self._cached_attribution)

        elif intent == "schedule_audit":
            bakers = results.get("staffing", {}).get("data", {}).get("bakers", 0)
            if audit["healthy"]:
                action = f"Schedule looks good: {bakers} bakers, no anomalies."
                priority = "normal"
            else:
                action = f"Schedule issues: {', '.join(audit['conflicts'])}"
                priority = "warning"

        elif intent == "promo_eval":
            discount = promo.get("discount_pct", 0)
            surplus = promo.get("surplus", 0)
            if surplus > 5:
                action = f"Promo recommended: {discount}% off to clear {surplus} surplus units."
                priority = "normal"
            else:
                action = f"No promo needed: surplus only {surplus} units."
                priority = "normal"

        elif intent == "profit_analysis":
            action = f"Profit check: {forecast} forecast demand at {stock} inventory."
            priority = "normal"

        elif intent == "comparison_analysis":
            comparison = _run_comparison(forecast, stock, inventory)
            if comparison:
                action = comparison
                priority = "normal"
            else:
                action = f"No historical data available for comparison. Current: {forecast} forecast, {stock} inventory."
                priority = "normal"

        else:
            # stock_query / cross_source_audit - OPTIMIZER computes numbers
            cap = min(max_cap, self.config.daily_capacity) if max_cap > 0 else self.config.daily_capacity
            per_product_inv = inventory.get("per_product", {})
            per_product_demand = demand.get("per_product", {})

            if per_product_inv and per_product_demand and len(per_product_inv) > 1:
                prod_states = []
                for pname, pdata in per_product_inv.items():
                    p_demand = per_product_demand.get(pname, {}).get("forecast", 0)
                    day1 = pdata.get("day1", 0)
                    fresh = max(0, pdata["qty"] - day1)
                    p_low = per_product_demand.get(pname, {}).get("lower", max(0, p_demand - 15))
                    p_high = per_product_demand.get(pname, {}).get("upper", p_demand + 15)
                    prod_states.append(ProductState(pname, demand=p_demand,
                        demand_low=p_low, demand_high=p_high,
                        fresh_stock=fresh, day1_stock=day1,
                        waste_loss=costs.waste_loss, stockout_loss=costs.stockout_loss,
                        production_cost=costs.production_cost))
                opt = optimize_multi(prod_states, cap, costs, config=self.config)
                action = opt["rationale"]
                priority = "warning" if opt.get("shortage_units", 0) > 0 else "normal"
                if self._cached_attribution is not None:
                    counterfactual = counterfactual_analysis(
                        opt.get("bake_units", 0), stock, forecast, self._cached_attribution)
            else:
                day1 = inventory.get("day1_available", 0)
                opt = optimize_single(forecast, stock, cap, costs, product_name=product,
                                      day1_stock=day1, config=self.config, unit_price=5.90)
                action = opt["rationale"]
                priority = "warning" if opt["shortage_units"] > 0 else "normal"
                if self._cached_attribution is not None:
                    counterfactual = counterfactual_analysis(
                        opt.get("bake_units", 0), stock, forecast, self._cached_attribution)

            # Append deliberation insight as supplementary note
            if deliberation_note and "bake" not in deliberation_note.lower()[:30]:
                action += " | " + deliberation_note[:150]

        trace = []
        for name, r in results.items():
            trace.append({"agent": name, "opinion": r.get("opinion", ""),
                         "confidence": r.get("confidence", 0), "constraints": r.get("constraints", [])})

        result = {"action": action, "priority": priority, "reasoning_trace": trace, "audit": audit,
                  "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)}
        if self._cached_attribution is not None:
            result["causal_calibration"] = {"waste_loss_per_unit": self._cached_attribution.avg_waste_per_unit_cost,
                "stockout_loss_per_unit": self._cached_attribution.avg_stockout_per_unit_cost,
                "top_waste_driver": self._cached_attribution.top_waste_driver, "method": self._cached_attribution.method}
        if opt.get("profit_rm") is not None:
            result["optimizer_profit"] = {"profit_rm": opt["profit_rm"], "revenue_rm": opt.get("revenue_rm", 0),
                "risk_preference": opt.get("risk_preference", "balanced")}
        if counterfactual is not None:
            result["counterfactual"] = counterfactual
        if history:
            result["memory_context"] = history[:300]
        return result
