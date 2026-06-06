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
    comparison_keywords = ["vs last", "versus last", "compared to last", "week over week", "period over period", "change from last", "last week"]
    is_comparison = any(kw in ql for kw in comparison_keywords)

    if "tomorrow" in ql or "next day" in ql: target = today + timedelta(days=1)
    elif "today" in ql: target = today
    if target.weekday() == 0: target += timedelta(days=1)  # skip Monday

    return {"intent": "comparison_analysis" if is_comparison else "pending", "intent_confidence": 0.6 if is_comparison else 0.0, "product": "pending",
            "days": 7, "date": target.strftime("%Y-%m-%d"), "planned_agents": []}

@app.post("/query")
async def handle_query(req: QueryRequest):
    """Orchestrate: agents -> deliberation -> arbitrator -> synthesis -> memory."""
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

    # Primary: LLM Planner (DeepSeek function calling)
    # Fallback: DistilBERT (keyword classifier) - only if LLM fails
    plan = None
    if LLM_PLANNER_ENABLED:
        plan = await llm_plan_query(req.query)

    if plan and plan.get("agents"):
        # LLM Planner succeeded - use its routing
        params["planned_agents"] = plan["agents"]
        params["intent"] = plan.get("intent", "stock_query")
        params["intent_confidence"] = plan.get("llm_confidence", 0.95)
        params["product"] = plan.get("product", "croissant")
        # Post-process: detect period comparison queries (this week vs last week)
        ql_lower = req.query.lower()
        period_keywords = ["vs last", "versus last", "compared to last", "week over week",
                           "period over period", "change from last", "last week",
                           "compare this week", "compare last week"]
        if any(kw in ql_lower for kw in period_keywords):
            params["intent"] = "comparison_analysis"
            intent = "comparison_analysis"
        else:
            intent = params["intent"]
    else:
        # LLM Planner failed - fall back to keyword intent
        ql = req.query.lower()
        _kw_rules = {
            "stock_query": ["stock","restock","inventory","replenish","bake","prepare","how many","how is the","what is my stock","how is","tell me about","status","check","show","forecast","demand","selling","look","report","summary","what about","should i","do i","need to","doing","issues","update","berapa","banyak","esok","stok","bakar","sediakan"],
            "waste_analysis": ["waste","loss","expired","expiring","expiry","spoilage","throw","why"],
            "promo_eval": ["promo","discount","off","sale","markdown","deal","bundle","should i","do i","run a","apply"],
            "schedule_audit": ["schedule","shift","staff","anomal","who","swap","sick","leave","roster","staff issues","schedule issues","shift issues","coverage gap","understaff"],
            "cross_source_audit": ["health check","audit","full","cross","all product","store health","overview"],
            "profit_analysis": ["profit","margin","revenue","cost","earn","income"],
        }
        _scores = {k: 0 for k in _kw_rules}
        for k, kws in _kw_rules.items():
            _scores[k] = sum(1 for w in kws if w in ql)
        _best = max(_scores, key=_scores.get)
        intent = _best if _scores[_best] > 0 else "stock_query"
        params["intent"] = intent
        params["intent_confidence"] = 0.6

        # Product extraction fallback
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
        merged = {"_demand": results["demand"].get("data", {}), "_staffing": results["staffing"].get("data", {})}
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

    # Agent Deliberation (LLM-mediated consensus when conflicts exist)
    deliberation = await arbitrator.deliberate(results, params, history_text)

    # Final decision with deliberation context and memory
    decision = arbitrator.decide(results, params, deliberation, history_text)

    # LLM synthesis
    llm_summary = None
    if SYNTHESIS_ENABLED:
        llm_summary = await synthesize(
            query=req.query, intent=intent,
            decision=decision["action"], priority=decision["priority"],
            agent_data=results,
            conflicts=decision["audit"].get("conflicts", []),
            counterfactual=decision.get("counterfactual"),
            causal_calibration=decision.get("causal_calibration"),
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
        "agents": agent_summaries,
        "decision": decision["action"],
        "priority": decision["priority"],
        "reasoning_trace": decision["reasoning_trace"],
        "audit": decision["audit"],
        "errors": errors,
    }
    if "causal_calibration" in decision:
        response["causal_calibration"] = decision["causal_calibration"]
    if "counterfactual" in decision:
        response["counterfactual"] = decision["counterfactual"]
    if llm_summary:
        response["llm_summary"] = llm_summary
    if deliberation.get("consensus"):
        response["deliberation"] = {
            "consensus": deliberation["consensus"],
            "rationale": deliberation.get("rationale", ""),
            "votes": deliberation.get("votes", {}),
        }

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
