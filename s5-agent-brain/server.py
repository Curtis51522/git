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
arbitrator = Arbitrator()
AGENTS = [demand_agent, inventory_agent, production_agent, staffing_agent, promo_agent]


class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"
    params: dict = {}


def parse_query(query: str) -> dict:
    """Keyword-based product and intent extraction from natural language."""
    ql = query.lower()
    params = {"product": "croissant", "days": 7}
    for p in PRODUCT_NAMES:
        if p.replace("_", " ") in ql or p in ql:
            params["product"] = p
            break
    if "tomorrow" in ql:
        params["date"] = "tomorrow"
    return params


@app.post("/query")
async def handle_query(req: QueryRequest):
    """Orchestrate all agents in parallel -> Arbitrator -> response."""
    t_start = time.perf_counter()
    params = parse_query(req.query)

    agent_coros = [agent.run(params) for agent in AGENTS]
    results_list = await asyncio.gather(*agent_coros, return_exceptions=True)

    results = {}
    errors = []
    for agent, result in zip(AGENTS, results_list):
        if isinstance(result, Exception):
            logger.warning("Agent %s failed: %s", agent.name, result)
            errors.append(str(result))
            results[agent.name] = {"opinion": f"Error: {result}", "confidence": 0, "constraints": [], "data": {}}
        else:
            results[agent.name] = result

    # Cross-agent data pass: Production needs Demand + Staffing
    if "demand" in results and "staffing" in results:
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
    if "demand" in results and "inventory" in results:
        merged2 = {
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
