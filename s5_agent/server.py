# s5_agent/server.py — S5 AI Brain server (port 8001)
import asyncio, logging, time, sys, os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from s5_agent.core.dag import DAGExecutor
from s5_agent.core.deliberator import Deliberator
from s5_agent.core.synthesizer import Synthesizer
from s5_agent.core.memory import StructuredMemory
from s5_agent.router.templates import get_template
from s5_agent.router.intent_router import route_intent
from s5_agent.agents import (
    ExternalFactorsAgent, DemandAgent, MaterialStockAgent, ProductStockAgent,
    WastageAgent, ProductionAgent, YieldAgent, StaffingAgent,
    PricingAgent, ProfitAgent, PromoAgent, AttendanceAgent,
    TrendAgent, HourlyPatternAgent, ProductMixAgent,
    FeatureSensitivityAgent, MetricConflictAgent, CausalChainAgent, CrossRiskAgent,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("s5.server")

@asynccontextmanager
async def lifespan(app):
    logger.info("S5 AI Brain starting with 19 agents, 7 templates...")
    yield

app = FastAPI(title="S5 AI Brain - Multi-Agent Bakery Intelligence", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

memory = StructuredMemory()
_response_cache: dict = {}
_response_cache_max = 100
deliberator = Deliberator(memory=memory)
synthesizer = Synthesizer()

AGENTS = {
    "ExternalFactorsAgent": ExternalFactorsAgent("ExternalFactorsAgent"),
    "DemandAgent": DemandAgent("DemandAgent"),
    "MaterialStockAgent": MaterialStockAgent("MaterialStockAgent"),
    "ProductStockAgent": ProductStockAgent("ProductStockAgent"),
    "WastageAgent": WastageAgent("WastageAgent"),
    "ProductionAgent": ProductionAgent("ProductionAgent"),
    "YieldAgent": YieldAgent("YieldAgent"),
    "StaffingAgent": StaffingAgent("StaffingAgent"),
    "PricingAgent": PricingAgent("PricingAgent"),
    "ProfitAgent": ProfitAgent("ProfitAgent"),
    "PromoAgent": PromoAgent("PromoAgent"),
    "AttendanceAgent": AttendanceAgent("AttendanceAgent"),
    "TrendAgent": TrendAgent("TrendAgent"),
    "HourlyPatternAgent": HourlyPatternAgent("HourlyPatternAgent"),
    "ProductMixAgent": ProductMixAgent("ProductMixAgent"),
    "FeatureSensitivityAgent": FeatureSensitivityAgent("FeatureSensitivityAgent"),
    "MetricConflictAgent": MetricConflictAgent("MetricConflictAgent"),
    "CausalChainAgent": CausalChainAgent("CausalChainAgent"),
    "CrossRiskAgent": CrossRiskAgent("CrossRiskAgent"),
}
dag_executor = DAGExecutor(AGENTS, memory=memory)

class AnalyzeRequest(BaseModel):
    query: str = ""
    intent: str = ""
    params: dict = {}
    session_id: str = "default"
    lang: str = "en"

class ModuleAnalyzeRequest(BaseModel):
    module: str
    date: str = ""
    params: dict = {}
    lang: str = "en"

@app.get("/health")
async def health():
    return {"status": "ok", "agents": len(AGENTS), "templates": 7}

def _response_cache_key(intent: str, params: dict) -> str:
    date = str(params.get("date", "")) if params else ""
    module = str(params.get("module", "")) if params else ""
    return f"{intent}:{date}:{module}"

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    t0 = time.perf_counter()
    intent = req.intent or route_intent(req.query)[0]
    template = get_template(intent)
    if not template:
        raise HTTPException(400, f"Unknown intent: {intent}")
    
    # Check response cache
    cache_key = _response_cache_key(intent, req.params)
    if cache_key in _response_cache:
        logger.info("Response cache HIT for %s", cache_key)
        cached = dict(_response_cache[cache_key])
        cached["cache_hit"] = True
        cached["total_elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return cached

    logger.info("Analyze: intent=%s query=%s", intent, req.query)
    dag_result = await dag_executor.execute(template, req.params, req.query, intent)
    deliberation = None
    if len(dag_result["results"]) >= 2:
        conflicts = deliberator.detect_conflict(dag_result["results"])
        if conflicts:
            a, b, op_a, op_b = conflicts[0]
            classification = await deliberator.classify_agreement(a, b, op_a, op_b)
            if classification == "conflict":
                conflict_type = f"{op_a.get('attribution',{}).get('metric','')}"
                deliberation = await deliberator.deliberate(a, b, op_a, op_b, conflict_type)
    output = await synthesizer.synthesize(dag_result, deliberation, memory, lang=req.lang)
    memory.add_query(req.query or intent, intent, output.get("summary", ""),
                     {"significant": output.get("significance", {}).get("significant", False)})
    result = {
        **output, "intent": intent,
        "total_elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        "agents_executed": len(dag_result["results"]),
        "cache_hit": False,
    }
    # Write to response cache
    _response_cache[cache_key] = result
    if len(_response_cache) > _response_cache_max:
        oldest = next(iter(_response_cache))
        del _response_cache[oldest]
    return result

@app.post("/analyze/module")
async def analyze_module(req: ModuleAnalyzeRequest):
    module_intent_map = {
        "revenue": "profit_root_cause",
        "wastage": "wastage_root_cause",
        "forecast": "production_advice",
        "inventory": "inventory_diagnosis",
        "schedule": "staffing_diagnosis",
        "kpi": "full_diagnosis",
    }
    intent = module_intent_map.get(req.module, "full_diagnosis")
    params = {"date": req.date, **(req.params or {})}
    return await analyze(AnalyzeRequest(intent=intent, params=params, lang=req.lang))

@app.get("/templates")
async def list_templates():
    from s5_agent.router.templates import TEMPLATES
    return {k: {"intent": v.intent, "description": v.description, "nodes": len(v.nodes)}
            for k, v in TEMPLATES.items()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
