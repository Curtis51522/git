# s5-agent-brain - Independent Multi-Agent Service
# Runs on port 8001, decoupled from the main bakery system (:8000).
# Orchestrates 5 agents in parallel -> Arbitrator health audit -> final decision.
import asyncio, logging, time, sys, os

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


def parse_query(query: str) -> dict:
    """Keyword-based product, date, and intent extraction from natural language."""
    ql = query.lower()
    params = {"product": "croissant", "days": 7}

    # Date resolution
    from datetime import datetime, timedelta
    today = datetime.now()
    if "tomorrow" in ql or "next day" in ql:
        target = today + timedelta(days=1)
    elif "today" in ql:
        target = today
    else:
        target = today + timedelta(days=1)  # default: tomorrow

    # Skip Monday (closed)
    while target.weekday() == 0:
        target += timedelta(days=1)
    params["date"] = target.strftime("%Y-%m-%d")

    # Product detection
    product_found = None
    for p in PRODUCT_NAMES:
        if p.replace("_", " ") in ql or p in ql:
            product_found = p
            break

    # Intent detection
    if any(kw in ql for kw in ["schedule", "staff", "shift", "anomalies", "roster"]):
        params["intent"] = "schedule_audit"
    elif any(kw in ql for kw in ["health", "audit", "full", "check"]):
        params["intent"] = "cross_source_audit"
    elif any(kw in ql for kw in ["waste", "spoilage", "throw"]):
        params["intent"] = "waste_analysis"
    elif any(kw in ql for kw in ["promo", "discount", "deal", "offer"]):
        params["intent"] = "promo_eval"
    elif any(kw in ql for kw in ["profit", "margin", "revenue", "earn"]):
        params["intent"] = "profit_analysis"
    else:
        bakery_kw = ["bake", "croissant", "donut", "bread", "chiffon", "stock", "inventory", "forecast", "schedule", "staff", "waste", "profit", "margin", "promo", "discount"]
        if any(kw in ql for kw in bakery_kw):
            params["intent"] = "stock_query"
        else:
            params["intent"] = "out_of_scope"

    # Product resolution: stock/promo use detected product, cross queries use "all"
    if params.get("intent") in ("schedule_audit", "cross_source_audit", "profit_analysis", "waste_analysis"):
        params["product"] = product_found or "all"
    else:
        params["product"] = product_found or "croissant"

    return params


@app.post("/query")
async def handle_query(req: QueryRequest):
    """Orchestrate agents based on intent -> Arbitrator -> response."""
    t_start = time.perf_counter()
    params = parse_query(req.query)
    intent = params.get("intent", "stock_query")
    logger.info("Query: %s -> intent=%s product=%s date=%s", req.query[:60], intent, params["product"], params["date"])
    if intent == "out_of_scope":
        return {"status":"out_of_scope","product":"-","decision":"I can help with stock, waste, promo, schedule, and profit questions.","priority":"normal","agents":{},"audit":{"conflicts":[],"warnings":[]},"reasoning_trace":[],"elapsed_ms":0,"errors":[]}

    if intent == "schedule_audit":
        active_agents = [staffing_agent]
    elif intent == "waste_analysis":
        active_agents = [demand_agent, inventory_agent]
    elif intent == "promo_eval":
        active_agents = [demand_agent, inventory_agent, promo_agent]
    elif intent == "profit_analysis":
        active_agents = [demand_agent, inventory_agent, profit_agent]
    else:
        active_agents = AGENTS

    agent_coros = [agent.run(params) for agent in active_agents]
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

    # Cross-agent data pass: Production needs Demand + Staffing
    if "demand" in results and "staffing" in results and intent in ("stock_query", "cross_source_audit"):
        merged = {
            "_demand": results["demand"].get("data", {}),
            "_staffing": results["staffing"].get("data", {}),
        }
        try:
            prod_result = await production_agent.run({**params, **merged})
            if not isinstance(prod_result, Exception):
                results["production"] = prod_result
        except Exception as e:
            logger.warning("Production re-run failed: %s", e)

    # Cross-agent data pass: Promo needs Demand + Inventory
        merged2 = {
    if "demand" in results and "inventory" in results and intent in ("promo_eval", "stock_query"):
            "_demand": results["demand"].get("data", {}),
            "_inventory": results["inventory"].get("data", {}),
        }
        try:
            promo_result = await promo_agent.run({**params, **merged2})
            if not isinstance(promo_result, Exception):
                results["promo"] = promo_result
        except Exception as e:
            logger.warning("Promo re-run failed: %s", e)

    decision = arbitrator.decide(results, params)

    agent_summaries = {
        name: {
            "opinion": r.get("opinion", ""),
            "confidence": r.get("confidence", 0),
            "data": r.get("data", {}),
        }
        for name, r in results.items()
    }

    return {
        "status": "ok",
        "elapsed_ms": round((time.perf_counter() - t_start) * 1000, 1),
        "product": params.get("product", "croissant"),
        "agents": agent_summaries,
        "decision": decision["action"],
        "priority": decision["priority"],
        "reasoning_trace": decision["reasoning_trace"],
        "audit": decision["audit"],
        "errors": errors,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "agents": [a.name for a in AGENTS]}


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting AI Brain on %s:%s with %d agents", HOST, PORT, len(AGENTS))
    uvicorn.run(app, host=HOST, port=PORT)
