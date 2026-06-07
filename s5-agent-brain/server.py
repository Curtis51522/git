# s5-agent-brain - Independent Multi-Agent Service
# Runs on port 8001, decoupled from the main bakery system (:8002).
# Orchestrates 6 agents in parallel -> Arbitrator -> Optimizer -> LLM Synthesis.
# System Alerts - background monitor + persistent alert history.
import asyncio, logging, time, sys, os

# Windows: use SelectorEventLoop to avoid Proactor connection-reset noise
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Make parent project importable for shared modules
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.demand import DemandAgent
from agents.inventory import InventoryAgent
from agents.production import ProductionAgent
from agents.staffing import StaffingAgent
from agents.promo import PromoAgent
from agents.profit import ProfitAgent
from arbitrator import Arbitrator
from s5_config.settings import PORT, HOST, PRODUCT_NAMES
from alert_store import get_alerts, acknowledge, get_unacked_count, clear_expired
from monitor import start_monitor, run_full_check
from llm_synthesis import synthesize, SYNTHESIS_ENABLED
from intent_classifier import classify_intent
from optimizer import project_multi_period

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("s5.server")

app = FastAPI(title="AI Brain - Multi-Agent Bakery Intelligence")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

demand_agent = DemandAgent()
inventory_agent = InventoryAgent()
production_agent = ProductionAgent()
staffing_agent = StaffingAgent()
promo_agent = PromoAgent()
profit_agent = ProfitAgent()
arbitrator = Arbitrator()
AGENTS = [demand_agent, inventory_agent, production_agent, staffing_agent, promo_agent, profit_agent]


class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"
    params: dict = {}


class AckRequest(BaseModel):
    alert_id: int = None
    ack_all: bool = False



# ---------------------------------------------------------------------------
# LLM Query Planner - replaces hardcoded intent->agent routing
# Inspired by LLMCompiler (Kim et al., ICML 2024): DAG-based agent planning
# Uses DeepSeek function calling to select agents + extract parameters
# ---------------------------------------------------------------------------
LLM_PLANNER_ENABLED = True  # set False to fall back to keyword rules only

AGENT_TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "route_query",
        "description": "Route a bakery query to the correct AI agents. Select ALL agents needed to answer the query.",
        "parameters": {
            "type": "object",
            "properties": {
                "agents": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["demand", "inventory", "production", "staffing", "promo", "profit"]},
                    "description": "Agents to activate for this query. Choose from: demand (forecast/trends), inventory (stock/freshness), production (baker capacity), staffing (schedule/shifts), promo (discounts/bundles), profit (revenue/margin)."
                },
                "product": {
                    "type": "string",
                    "description": "Target product name(s). Use comma-separated for comparison: 'croissant,donut'. Use 'all' for store-wide queries. Use '-' if no product relevant."
                },
                "intent": {
                    "type": "string",
                    "enum": ["stock_query", "waste_analysis", "promo_eval", "schedule_audit", "cross_source_audit", "profit_analysis", "out_of_scope"],
                    "description": "Primary intent of the query."
                },
                "date": {
                    "type": "string",
                    "description": "Target date in YYYY-MM-DD format. Default to tomorrow if not specified."
                }
            },
            "required": ["agents", "product", "intent"]
        }
    }}
]

AGENT_PLANNER_PROMPT = """You are a bakery operations query router. Given a user query, decide:
1. Which agents to activate: demand(forecast/trends), inventory(stock/freshness), production(capacity), staffing(schedule), promo(discounts/bundles), profit(revenue/margin)
2. The target product(s) - use product names exactly as defined
3. The primary intent

Rules:
- "How many X tomorrow?" -> agents: [demand, inventory, production, staffing, promo, profit], intent: stock_query
- "Why is waste high?" -> agents: [demand, inventory], intent: waste_analysis, product: all
- "Run promo on X" -> agents: [demand, inventory, promo], intent: promo_eval
- "Check schedule" -> agents: [staffing], intent: schedule_audit
- "Store health check" or "audit" -> agents: [demand, inventory, production, staffing, promo, profit], intent: cross_source_audit
- "Profit margin" -> agents: [demand, inventory, profit], intent: profit_analysis
- "Compare X and Y" -> agents: [demand, inventory, production, staffing, promo, profit], intent: stock_query, product: "X,Y"
- Jokes, weather, unrelated -> agents: [], intent: out_of_scope, product: "-"
- For vague queries asking about a product: activate all agents to give comprehensive answer
- If query mentions a specific product, set product to that exact name
- Product names: croissant, donut, chiffon, bread_roll, bread_coconut, croissant_chocolate
- Date: default to tomorrow unless query specifies otherwise
- For "how is X doing" or "tell me about X": activate all agents for comprehensive overview"""

async def llm_plan_query(query: str) -> dict:
    """Use DeepSeek function calling to plan which agents to run.
    Returns dict with agents, product, intent, date, llm_confidence.
    Falls back gracefully on any error."""
    try:
        import httpx, os, sys
        _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _parent not in sys.path:
            sys.path.insert(0, _parent)
        try:
            from config.settings import DEEPSEEK_API_KEY as key
        except ImportError:
            key = os.getenv("DEEPSEEK_API_KEY", "")
        if not key:
            return None

        url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1") + "/chat/completions"
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": AGENT_PLANNER_PROMPT},
                        {"role": "user", "content": query}
                    ],
                    "tools": AGENT_TOOLS_SCHEMA,
                    "tool_choice": {"type": "function", "function": {"name": "route_query"}},
                    "temperature": 0.1,
                    "max_tokens": 300,
                })
            resp.raise_for_status()
            data = resp.json()

        # Extract function call arguments
        choice = data["choices"][0]["message"]
        if "tool_calls" not in choice:
            return None

        tool_call = choice["tool_calls"][0]
        if tool_call["function"]["name"] != "route_query":
            return None

        import json
        args = json.loads(tool_call["function"]["arguments"])
        return {
            "agents": args.get("agents", ["demand", "inventory", "production", "staffing", "promo", "profit"]),
            "product": args.get("product", "croissant"),
            "intent": args.get("intent", "stock_query"),
            "date": args.get("date", ""),
            "llm_confidence": 0.95,
        }
    except Exception as e:
        logging.getLogger("s5.server").warning("LLM query planner failed: %s, falling back to DistilBERT", e)
        return None

def parse_query(query: str) -> dict:
    """Minimal sync parser: date extraction + empty check only. LLM Planner is primary."""
    if not query or not query.strip():
        return {"intent": "out_of_scope", "intent_confidence": 1.0, "product": "-",
                "days": 7, "date": "", "planned_agents": []}

    from datetime import datetime, timedelta
    today = datetime.now()
    ql = query.lower()
    target = today + timedelta(days=1)
    is_comparison = False

    # Relative date words (English + Malay)
    if any(w in ql for w in ["tomorrow", "next day", "esok", "keesokan"]):
        target = today + timedelta(days=1)
    elif any(w in ql for w in ["day after tomorrow", "lusa"]):
        target = today + timedelta(days=2)
    elif "yesterday" in ql or "semalam" in ql:
        target = today - timedelta(days=1)
    elif "today" in ql or "hari ini" in ql:
        target = today
    else:
        # Map day-of-week names (English + Malay)
        day_names = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6,
            "isnin": 0, "selasa": 1, "rabu": 2, "khamis": 3,
            "jumaat": 4, "sabtu": 5, "ahad": 6,
        }
        for day_name, day_num in day_names.items():
            if day_name in ql:
                days_ahead = (day_num - today.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7  # "Monday" on Monday -> next Monday
                target = today + timedelta(days=days_ahead)
                break
    # Skip Monday (rest day): if target lands on Monday, push to Tuesday
    ql_date_shifted = False
    if target.weekday() == 0:
        target += timedelta(days=1)
        ql_date_shifted = True

    rest_note = ""
    if ql_date_shifted:
        is_malay = any(w in ql for w in ["esok", "hari ini", "lusa", "semalam", "isnin", "selasa", "rabu", "khamis", "jumaat", "sabtu", "ahad"])
        if is_malay:
            rest_note = "PENTING: Isnin adalah hari rehat (kedai tutup). Semua data di bawah adalah untuk hari Selasa. Anda MESTI nyatakan ini pada permulaan ringkasan. "
        else:
            rest_note = "IMPORTANT: Monday is a rest day (bakery closed). All data below is for Tuesday (the next open day). You MUST state this clearly at the start of your summary. "
    return {"intent": "pending", "intent_confidence": 0.0, "product": "pending",
            "days": 7, "date": target.strftime("%Y-%m-%d"), "planned_agents": [], "rest_note": rest_note}


async def llm_decide_plan(pareto_plans: list, pareto_context: dict,
                          intent: str, query: str, demand_trend: str = "stable",
                          audit_conflicts: list = None) -> dict:
    """LLM selects from Pareto-optimal plans with contextual reasoning.
    
    Returns {selected_plan: str, reason: str} or None on failure.
    LLM can only SELECT a plan, never create one - this is the safety constraint.
    """
    if not pareto_plans or not LLM_PLANNER_ENABLED:
        return None

    plan_lines = []
    for p in pareto_plans:
        sc = p.get("scenarios", {})
        sc_str = ""
        if sc:
            parts = []
            for s in ["low_demand", "predicted", "high_demand"]:
                if s in sc:
                    parts.append(f"{s}:P=RM{sc[s]['profit_rm']:.0f},w={sc[s]['waste']},s={sc[s]['shortage']}")
            sc_str = " [" + " | ".join(parts) + "]"
        plan_lines.append(
            f"  {p['label']}: bake={p['bake']}, profit=RM{p.get('profit_rm',0):.0f}, "
            f"waste={p.get('waste',0)}, shortage={p.get('shortage',0)}, "
            f"worst_profit=RM{p.get('worst_case_profit',0):.0f}"
            f"{sc_str}"
        )

    audit_conflicts = audit_conflicts or []
    conflicts_str = "; ".join(audit_conflicts) if audit_conflicts else "None"
    audit_healthy = "Yes" if not audit_conflicts else "No"

    prompt = f"""You are a bakery operations decision-maker. Choose ONE plan from the options below. You MUST select exactly one plan label (A_aggressive, B_balanced, C_conservative, or D_baseline).

QUERY: {query}
INTENT: {intent}

CONTEXT:
  Stock: {pareto_context.get('stock', '?')} units
  Forecast: {pareto_context.get('forecast', '?')} (range: {pareto_context.get('forecast_range', '?')})
  Capacity: {pareto_context.get('max_capacity', '?')} units
  Day-1 stock at risk: {pareto_context.get('day1_stock', 0)} units
  Demand trend: {demand_trend}

AUDIT:
  Conflicts: {conflicts_str}
  Healthy: {audit_healthy}

PLANS:
{chr(10).join(plan_lines)}

RULES:
1. You MUST pick exactly one plan. Output format: PLAN=A_aggressive|CONFLICT_ADDRESSED=Y/N|REASON=your reasoning (max 50 words)
2. If AUDIT shows conflicts, your chosen plan MUST address them. Explain HOW in REASON.
3. Prioritize the plan with the HIGHEST profit, unless another plan has significantly less waste/shortage with similar profit.
4. Declining trend -> prefer conservative; rising trend -> aggressive; day-1 stock -> minimize waste.
5. Do NOT modify the numbers or create a new plan.
6. REASON must explain WHY this plan fits the context AND how it resolves audit issues."""

    try:
        import httpx, os, sys
        _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _parent not in sys.path:
            sys.path.insert(0, _parent)
        try:
            from config.settings import DEEPSEEK_API_KEY as key
        except ImportError:
            key = os.getenv("DEEPSEEK_API_KEY", "")
        if not key:
            return None

        url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1") + "/chat/completions"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 150, "temperature": 0.3})
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"].strip()

        # Parse PLAN=X|REASON=Y
        plan_label = "B_balanced"
        reason = ""
        for part in raw.split("|"):
            part = part.strip()
            if part.upper().startswith("PLAN="):
                plan_label = part.split("=", 1)[1].strip()
            elif part.upper().startswith("REASON="):
                reason = part.split("=", 1)[1].strip()

        # Validate: plan must exist in pareto_plans
        valid_labels = {p["label"] for p in pareto_plans}
        if plan_label not in valid_labels:
            plan_label = "B_balanced"  # safety: LLM hallucinated/failed, fallback to recommended plan

        return {"selected_plan": plan_label, "reason": reason or raw[:150]}
    except Exception as e:
        logging.getLogger("s5.server").warning("LLM decision failed: %s", e)
        return None

@app.post("/query")
async def handle_query(req: QueryRequest):
    """Orchestrate: agents -> arbitrator -> optimizer -> LLM synthesis."""
    t_start = time.perf_counter()
    params = parse_query(req.query)
    session_id = req.session_id or "default"

    # Override date if frontend sent one explicitly
    if req.params.get("date"):
        params["date"] = req.params["date"]

    intent = params.get("intent", "stock_query")
    logger.info("Query: %s -> intent=%s product=%s session=%s",
                req.query[:60], intent, params["product"], session_id)

    # Load session memory
    history_text = ""
    key_metrics = ""

    # Build agent name->instance map for dynamic LLM-planned routing
    _agent_map = {"demand": demand_agent, "inventory": inventory_agent,
                  "production": production_agent, "staffing": staffing_agent,
                  "promo": promo_agent, "profit": profit_agent}

    # Primary: DistilBERT intent classifier (local, fast, free)
    # Backup: LLM Planner (DeepSeek function calling)
    # Last resort: keyword rules
    ql = req.query.lower()
    db_intent, db_conf = classify_intent(req.query)
    intent = db_intent
    params["intent"] = db_intent
    params["intent_confidence"] = round(db_conf, 4)

    # If DistilBERT confidence is very low, try LLM Planner as backup
    if db_conf < 0.3 and LLM_PLANNER_ENABLED:
        plan = await llm_plan_query(req.query)
        if plan and plan.get("agents"):
            params["planned_agents"] = plan["agents"]
            params["intent"] = plan.get("intent", db_intent)
            params["intent_confidence"] = plan.get("llm_confidence", 0.95)
            params["product"] = plan.get("product", "croissant")
            intent = params["intent"]

    # Product extraction
    # DistilBERT handles all intent classification including comparison_analysis

    if params.get("product") in ("pending", None):
        product_found = None
        for p in sorted(PRODUCT_NAMES, key=len, reverse=True):
            if p.replace("_", " ") in ql or p in ql:
                product_found = p; break

        mentioned = []
        for p in sorted(PRODUCT_NAMES, key=len, reverse=True):
            if p.replace("_", " ") in ql or p in ql:
                if any(p in m for m in mentioned): continue
                mentioned.append(p)

        if len(mentioned) >= 2 or "compare" in ql or "versus" in ql or " vs " in ql:
            params["product"] = ",".join(mentioned) if len(mentioned) >= 2 else "all"
        elif intent in ("schedule_audit", "cross_source_audit", "profit_analysis", "waste_analysis"):
            params["product"] = product_found or "all"
        else:
            params["product"] = product_found or "croissant"

    # Out-of-scope check
    if intent == "out_of_scope":
        return {
            "status": "out_of_scope", "intent": "out_of_scope", "product": "-",
            "decision": "I can help with stock, waste, promo, schedule, and profit questions.",
            "priority": "normal", "agents": {}, "audit": {"conflicts": [], "warnings": []},
            "reasoning_trace": [], "elapsed_ms": round((time.perf_counter() - t_start) * 1000, 1), "errors": [],
        }

    planned = params.get("planned_agents", [])
    if planned:
        active_agents = [_agent_map[a] for a in planned if a in _agent_map] or AGENTS
    elif intent == "schedule_audit":
        active_agents = [staffing_agent]
    elif intent == "waste_analysis":
        active_agents = [demand_agent, inventory_agent]
    elif intent == "promo_eval":
        active_agents = [demand_agent, inventory_agent, promo_agent, profit_agent]
    elif intent == "profit_analysis":
        active_agents = [demand_agent, inventory_agent, profit_agent]
    elif intent == "comparison_analysis":
        active_agents = [demand_agent, inventory_agent, profit_agent]
    else:
        active_agents = AGENTS

    # Run agents with memory context
    agent_coros = [agent.run(params, history_text, key_metrics) for agent in active_agents]
    results_list = await asyncio.gather(*agent_coros, return_exceptions=True)

    results = {}
    errors = []
    for agent, result in zip(active_agents, results_list):
        if isinstance(result, Exception):
            logger.warning("Agent %s failed: %s", agent.name, result)
            errors.append(str(result))
            results[agent.name] = {"opinion": f"Error: {result}", "confidence": 0, "constraints": [], "data": {}}
        else:
            results[agent.name] = result

    # Cross-agent data passes
    if "demand" in results and "staffing" in results and intent in ("stock_query", "cross_source_audit"):
        merged = {"_demand": results["demand"].get("data", {}), "_staffing": results["staffing"].get("data", {}), "_inventory": results["inventory"].get("data", {})}
        try:
            prod_result = await production_agent.run({**params, **merged}, history_text, key_metrics)
            if not isinstance(prod_result, Exception):
                results["production"] = prod_result
        except Exception as e:
            logger.warning("Production re-run failed: %s", e)

    if "demand" in results and "inventory" in results and intent in ("promo_eval", "stock_query", "cross_source_audit"):
        merged2 = {"_demand": results["demand"].get("data", {}), "_inventory": results["inventory"].get("data", {})}
        if "profit" in results:
            merged2["_profit"] = results["profit"].get("data", {})
        try:
            promo_result = await promo_agent.run({**params, **merged2}, history_text, key_metrics)
            if not isinstance(promo_result, Exception):
                results["promo"] = promo_result
        except Exception as e:
            logger.warning("Promo re-run failed: %s", e)


    # Final decision with Pareto plan selection
    decision = arbitrator.decide(results, params)

    # LLM Decision Layer: select from Pareto plans with contextual reasoning
    pareto_plans = decision.get("pareto_plans") if isinstance(decision, dict) else None
    if not pareto_plans:
        pareto_plans = decision.get("pareto_plans") if hasattr(decision, "get") else None
    if isinstance(decision, dict) and "pareto_context" in decision:
        demand_trend = results.get("demand", {}).get("data", {}).get("trend", "stable")
        audit_data = decision.get("audit", {})
        llm_choice = await llm_decide_plan(
            decision.get("pareto_plans", []),
            decision.get("pareto_context", {}),
            intent, req.query, demand_trend,
            audit_conflicts=audit_data.get("conflicts", []))
        if llm_choice:
            decision["llm_choice"] = llm_choice
            # Update decision text to reflect LLM choice
            chosen = next((p for p in decision.get("pareto_plans", [])
                          if p["label"] == llm_choice["selected_plan"]), None)
            if chosen:
                decision["action"] = (f"LLM chose {llm_choice['selected_plan']}: "
                                      f"{chosen['rationale']} | {llm_choice['reason']}")
                # Counterfactual: why this plan over alternatives?
                all_plans = decision.get("pareto_plans", [])
                alternatives = [p for p in all_plans if p["label"] != llm_choice["selected_plan"]]
                # Sync production agent with LLM-chosen plan
                chosen_bake = chosen.get("bake", 0)
                if "production" in results and chosen_bake > 0:
                    prod = results["production"]
                    prod["data"]["recommended"] = chosen_bake
                    # Rebuild opinion with correct bake
                    cap = prod["data"].get("max_capacity", 720)
                    bakers = prod["data"].get("bakers", 1)
                    ovens = prod["data"].get("ovens_used", 2)
                    rate = prod["data"].get("oven_rate", 80)
                    eff_hrs = prod["data"].get("effective_hours", 4.5)
                    window = prod["data"].get("effective_hours", 4.5)  # same as baking_window
                    prod["opinion"] = f"Capacity {cap} ({bakers} bakers x {ovens} ovens, {eff_hrs:.1f}h eff, {rate:.0f}/hr/oven, window={window:.1f}h), bake {chosen_bake}"

                if alternatives:
                    cf_parts = []
                    for alt in alternatives:
                        pdiff = round(chosen["profit_rm"] - alt["profit_rm"], 2)
                        wdiff = chosen.get("waste", 0) - alt.get("waste", 0)
                        sdiff = chosen.get("shortage", 0) - alt.get("shortage", 0)
                        cf_parts.append(
                            f"vs {alt['label']}: dP=RM{pdiff:+.0f} dW={wdiff:+d} dS={sdiff:+d}"
                        )
                    decision["counterfactual"] = " | ".join(cf_parts)

                # Multi-period projection: 7-day impact of this decision
                demand_data = results.get("demand", {}).get("data", {})
                raw_forecasts = demand_data.get("_raw_forecasts", [])
                if raw_forecasts and chosen:
                    product_name = params.get("product", "croissant")
                    if "," not in product_name and product_name != "all":
                        fc_list = [f.get("predicted_demand", 0) for f in raw_forecasts[:7]]
                        inv_data = results.get("inventory", {}).get("data", {})
                        pp_inv = inv_data.get("per_product", {}).get(product_name, {})
                        init_fresh = pp_inv.get("fresh", demand_data.get("stock", 0))
                        init_day1 = pp_inv.get("day1", 0)
                        if fc_list and fc_list[0] > 0:
                            proj = project_multi_period(fc_list, init_fresh, init_day1, chosen["bake"],
                                                        demand_data.get("unit_price", 5.90))
                            decision["projection"] = proj
                            cum_w = proj["cumulative_waste"]
                            cum_s = proj["cumulative_shortage"]
                            trend = proj["risk_trend"]
                            decision["action"] += (
                                f" [7d projection: {trend}, cum_waste={cum_w}, cum_shortage={cum_s}]"
                            )

    # LLM synthesis
    llm_summary = None
    if SYNTHESIS_ENABLED:
        # Augment decision text with LLM choice reasoning
        rest_note = params.get("rest_note", "")
        synth_decision = (rest_note + decision["action"]) if rest_note else decision["action"]
        if decision.get("llm_choice") and decision["llm_choice"].get("reason"):
            synth_decision += " [Decision rationale: " + decision["llm_choice"]["reason"] + "]"

        # Fetch memory context for richer synthesis
        memory_ctx = ""
        try:
            from memory_store import get_context
            memory_ctx = get_context(
                product=params.get("product", "croissant"),
                intent=intent,
                days=14)
        except Exception:
            pass

        # Enrich context with snapshot comparison for comparison_analysis
        if intent == "comparison_analysis" and "," in params.get("product", ""):
            try:
                from toolbox import compare_products
                prods = params["product"].split(",")
                cmp = compare_products(prods[0].strip(), prods[1].strip())
                if cmp:
                    parts = []
                    for pn, pd in cmp.items():
                        parts.append(f"{pn}: inv={pd['inventory']}, fc={pd['forecast']}")
                    cmp_text = " | Snapshot: " + "; ".join(parts)
                    memory_ctx = (memory_ctx or "") + cmp_text
            except Exception:
                pass

        # Enrich context with SHAP explainability
        shap_text = ""
        try:
            from toolbox import explain_forecast
            product_list = params.get("product", "croissant").split(",")
            for p in product_list[:2]:  # max 2 products
                shap = explain_forecast(p.strip())
                if shap and shap.get("top_features"):
                    feats = ", ".join(f"{f['feature']}({f['contribution']:.0%})" for f in shap["top_features"][:3])
                    shap_text += f"{p}: {feats}; "
            if shap_text and memory_ctx:
                memory_ctx += " | SHAP drivers: " + shap_text
            elif shap_text:
                memory_ctx = "SHAP drivers: " + shap_text
        except Exception:
            pass

        llm_summary = await synthesize(
            query=req.query, intent=intent,
            decision=synth_decision, priority=decision["priority"],
            agent_data=results,
            conflicts=decision["audit"].get("conflicts", []),
            counterfactual=decision.get("counterfactual"),
            causal_calibration=decision.get("causal_calibration"),
            memory_context=memory_ctx,
        )

    agent_summaries = {
        name: {"opinion": r.get("opinion", ""), "confidence": r.get("confidence", 0), "data": r.get("data", {})}
        for name, r in results.items()
    }

    response = {
        "status": "ok",
        "elapsed_ms": round((time.perf_counter() - t_start) * 1000, 1),
        "intent": intent,
        "intent_confidence": params.get("intent_confidence", 0),
        "product": params.get("product", "croissant"),
        "target_date": params.get("date", ""),
        "rest_note": params.get("rest_note", ""),
        "agents": agent_summaries,
        "decision": decision["action"],
        "priority": decision["priority"],
        "reasoning_trace": decision["reasoning_trace"],
        "audit": decision["audit"],
        "errors": errors,
    }
    if "causal_calibration" in decision:
        response["causal_calibration"] = decision["causal_calibration"]
    if "llm_choice" in decision:
        response["llm_choice"] = decision["llm_choice"]

    if "counterfactual" in decision:
        response["counterfactual"] = decision["counterfactual"]
    if "projection" in decision:
        response["projection"] = decision["projection"]
    if "pareto_plans" in decision:
        response["pareto_plans"] = decision["pareto_plans"]

    # Attach SHAP explainability
    try:
        from toolbox import explain_forecast
        product_list = params.get("product", "croissant").split(",")
        shap_data = {}
        for p in product_list[:3]:
            p = p.strip()
            s = explain_forecast(p)
            if s:
                shap_data[p] = s
        if shap_data:
            response["shap"] = shap_data
    except Exception:
        pass

    if llm_summary:
        response["llm_summary"] = llm_summary

    # Log query to memory (fire-and-forget)
    try:
        from memory_store import save_query
        save_query(
            query=req.query,
            intent=intent,
            product=params.get("product", "croissant"),
            agent_results=agent_summaries,
            decision=decision.get("action", ""),
            summary=llm_summary or "",
            target_date=params.get("date", ""))
    except Exception:
        pass

    return response


# ---------------------------------------------------------------------------
# Lightweight discount lookup for S4 frontend
# ---------------------------------------------------------------------------
@app.post("/discounts")
async def get_discounts(req: dict):
    """Return dynamic discounts for given products. Called by S4 frontend."""
    product_names = req.get("products", [])
    if not product_names:
        return {"discounts": {}}
    # Minimal agent run: inventory + demand (with date for forecast)
    import asyncio as _asyncio
    from datetime import date as _date, timedelta as _td
    _target = (_date.today() + _td(days=1)).isoformat()
    inv_result = await inventory_agent.run({"product": ",".join(product_names)}, "", {})
    dem_result = await demand_agent.run({"product": ",".join(product_names), "date": _target}, "", {})
    merged = {"_demand": dem_result.get("data", {}), "_inventory": inv_result.get("data", {})}
    # Include profit data for accurate dynamic margins (matches full AI Brain pipeline)
    try:
        prof_result = await profit_agent.run({"product": ",".join(product_names)}, "", {})
        merged["_profit"] = prof_result.get("data", {})
    except Exception:
        pass
    promo_result = await promo_agent.run(merged, "", {})
    details = promo_result.get("data", {}).get("discount_details", {})
    discounts = {}
    for pname, dd in details.items():
        discounts[pname] = {
            "discount_pct": dd.get("discount_pct", 0),
            "urgency": dd.get("urgency", 0),
            "breakdown": dd.get("breakdown", {}),
        }
    return {"discounts": discounts}

# ---------------------------------------------------------------------------
# Alert API endpoints
# ---------------------------------------------------------------------------
@app.get("/alerts/count")
async def alerts_count():
    return {"unacked_count": get_unacked_count()}


@app.get("/alerts/list")
async def alerts_list(limit: int = 100, unacked_only: bool = False):
    return {"alerts": get_alerts(limit=limit, unacked_only=unacked_only)}


@app.post("/alerts/ack")
async def alerts_ack(req: AckRequest):
    count = acknowledge(alert_id=req.alert_id, ack_all=req.ack_all)
    return {"acknowledged": count}


@app.post("/alerts/check")
async def alerts_check():
    results = await run_full_check()
    return {"status": "ok", "new_alerts": results}


@app.get("/health")
async def health():
    return {"status": "ok", "agents": [a.name for a in AGENTS]}


@app.on_event("startup")
async def startup_events():
    asyncio.create_task(start_monitor())


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting AI Brain on %s:%s with %d agents", HOST, PORT, len(AGENTS))
    uvicorn.run(app, host=HOST, port=PORT)
