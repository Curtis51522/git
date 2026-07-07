# s5_agent/server.py - S5 dashboard analysis server (port 8001)
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
from s5_agent.graph.registry import module_to_template
from s5_agent.graph.runner import run_s5_graph
from s5_agent.graph.state import S5Request
from s5_agent.router.templates import TEMPLATES, get_template
from s5_agent.router.intent_router import route_intent
from s5_agent.agents import (
    ExternalFactorsAgent, DemandAgent, MaterialStockAgent, ProductStockAgent,
    WastageAgent, ProductionAgent, YieldAgent, StaffingAgent,
    PricingAgent, ProfitAgent, PromoAgent, AttendanceAgent,
    TrendAgent, HourlyPatternAgent, ProductMixAgent,
    FeatureSensitivityAgent, MetricConflictAgent, CausalChainAgent, CrossRiskAgent,
    ForecastOverviewAgent, ForecastUncertaintyAgent, ProductionPlanAgent,
    MaterialProcurementAgent, ForecastAccuracyAgent,
    PlanFeasibilityAgent, DemandRiskAgent, EfficiencyAgent, WastageRiskAgent,
    RecommendationAgent,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("s5.server")

@asynccontextmanager
async def lifespan(app):
    logger.info("S5 dashboard analysis starting with %d agents, 7 templates...", len(AGENTS))
    yield

app = FastAPI(title="S5 Dashboard Analysis - Multi-Agent Bakery Intelligence", lifespan=lifespan)
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
    "ForecastOverviewAgent": ForecastOverviewAgent("ForecastOverviewAgent"),
    "ForecastUncertaintyAgent": ForecastUncertaintyAgent("ForecastUncertaintyAgent"),
    "ProductionPlanAgent": ProductionPlanAgent("ProductionPlanAgent"),
    "MaterialProcurementAgent": MaterialProcurementAgent("MaterialProcurementAgent"),
    "ForecastAccuracyAgent": ForecastAccuracyAgent("ForecastAccuracyAgent"),
    "PlanFeasibilityAgent": PlanFeasibilityAgent("PlanFeasibilityAgent"),
    "DemandRiskAgent": DemandRiskAgent("DemandRiskAgent"),
    "EfficiencyAgent": EfficiencyAgent("EfficiencyAgent"),
    "WastageRiskAgent": WastageRiskAgent("WastageRiskAgent"),
    "RecommendationAgent": RecommendationAgent("RecommendationAgent"),
}
dag_executor = DAGExecutor(AGENTS, memory=memory)
LANGGRAPH_MODULES = {"inventory", "revenue", "forecast", "wastage"}

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
    force_refresh: bool = False

@app.get("/health")
async def health():
    return {"status": "ok", "agents": len(AGENTS), "templates": len(TEMPLATES)}

@app.get("/templates")
async def list_templates():
    return {
        key: {"intent": value.intent, "description": value.description, "nodes": len(value.nodes)}
        for key, value in TEMPLATES.items()
    }

def _normalize_lang(lang: str) -> str:
    value = (lang or "en").strip().lower()
    if value in ("zh", "zh-cn", "cn", "chinese", "simplified_chinese"):
        return "zh"
    return "en"

def _response_cache_key(intent: str, params: dict, lang: str = "en") -> str:
    date = str(params.get("date", "")) if params else ""
    module = str(params.get("module", "")) if params else ""
    return f"{intent}:{date}:{module}:{_normalize_lang(lang)}"

def _latest_cached_synthesis(intent: str) -> dict:
    for value in reversed(list(_response_cache.values())):
        if value.get("intent") == intent:
            return value
    for entry in reversed(memory.data.get("query_history", [])):
        if entry.get("intent") == intent:
            return {"summary": entry.get("summary", ""), "recommendations": []}
    return {}

def _priority_context(cached: dict) -> str:
    parts = [cached.get("summary", "")]
    for rec in cached.get("recommendations", []):
        parts.append(rec.get("action", ""))
        parts.append(rec.get("rationale", ""))
    for evidence in cached.get("evidence", {}).values():
        if isinstance(evidence, dict):
            parts.append(evidence.get("opinion", ""))
    return " ".join(part for part in parts if part)

def _build_priority_recommendations(context: str) -> list:
    if not context:
        return []
    agent = AGENTS["RecommendationAgent"]
    try:
        result = agent.analyze(None, {}, context=context)
        metadata = getattr(result, "metadata", {}) or {}
        priorities = metadata.get("priority_recommendations", [])
        if priorities:
            return priorities[:3]
    except Exception as exc:
        logger.debug("RecommendationAgent direct analysis unavailable: %s", exc)

    signals = agent._parse_signals(context)
    priorities = []

    def add_priority(product: str, coffee: str, reason: str, boost: float, strategy: str) -> None:
        if product and product not in [item["product"] for item in priorities]:
            priorities.append({
                "product": product,
                "coffee": coffee,
                "reason": reason,
                "boost": boost,
                "strategy": strategy,
            })

    for product in signals.get("day1_products", [])[:3]:
        add_priority(
            product,
            agent._best_coffee_for(product, signals),
            f"Day-1 clearance: {product} needs to move before expiry",
            2.5,
            "clearance",
        )
    for product in signals.get("rising_products", [])[:2]:
        add_priority(
            product,
            agent._best_coffee_for(product, signals),
            f"Momentum: {product} volume rising, amplify with bundle",
            2.0,
            "amplify",
        )
    for product in signals.get("high_margin_products", [])[:2]:
        add_priority(
            product,
            signals.get("high_margin_coffee", "cold_brew"),
            f"Margin play: {product} has strong profit margin",
            1.8,
            "margin",
        )
    for product in signals.get("concentration_risk_products", [])[:2]:
        add_priority(
            product,
            agent._best_coffee_for(product, signals),
            f"Diversification: reduce reliance on {signals.get('top_seller', 'hero item')}",
            1.5,
            "diversify",
        )
    return priorities[:3]

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    t0 = time.perf_counter()
    req.lang = _normalize_lang(req.lang)
    intent = req.intent or route_intent(req.query)[0]
    template = get_template(intent)
    if not template:
        raise HTTPException(400, f"Unknown intent: {intent}")
    
    # Check response cache (skip if force_refresh)
    force_refresh = req.params.get("force_refresh", False) if req.params else False
    cache_key = _response_cache_key(intent, req.params, req.lang)
    if not force_refresh and cache_key in _response_cache:
        logger.info("Response cache HIT for %s", cache_key)
        cached = dict(_response_cache[cache_key])
        cached["cache_hit"] = True
        cached["total_elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return cached
    if force_refresh:
        logger.info("Force refresh requested, bypassing cache for %s", cache_key)

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
    lang = _normalize_lang(req.lang)
    if req.module in LANGGRAPH_MODULES:
        template_id = module_to_template(req.module)
        graph_params = {
            "date": req.date,
            "module": req.module,
            "product": "all",
            **(req.params or {}),
        }
        graph_request = S5Request(
            query=req.module,
            module=req.module,
            params=graph_params,
            lang=lang,
            force_refresh=req.force_refresh,
        )
        graph_response = await run_s5_graph(template_id, graph_request)
        return graph_response.model_dump()

    module_intent_map = {
        "revenue": "profit_root_cause",
        "wastage": "wastage_root_cause",
        "forecast": "production_advice",
        "inventory": "inventory_diagnosis",
        "schedule": "staffing_diagnosis",
        "kpi": "full_diagnosis",
    }
    intent = module_intent_map.get(req.module, "full_diagnosis")
    params = {"date": req.date, "module": req.module, "force_refresh": req.force_refresh, **(req.params or {})}
    return await analyze(AnalyzeRequest(intent=intent, params=params, lang=lang))





@app.get("/priorities")
async def get_priorities():
    """Return cached bundle priority recommendations from RecommendationAgent."""
    try:
        cached = _latest_cached_synthesis("profit_root_cause")
        priorities = _build_priority_recommendations(_priority_context(cached))
        if priorities:
            return {"status": "ok", "priorities": priorities, "cached": True}
    except Exception as e:
        logger.warning("Priority lookup failed: %s", e)
    return {"status": "ok", "priorities": [], "cached": False}

class DiscountRequest(BaseModel):
    products: list[str] = []

@app.post("/discounts")
async def get_discounts(req: DiscountRequest):
    """Return dynamic discount rates combining freshness + RecommendationAgent signals.
    
    Priority strategy -> discount mapping:
    - clearance: 40% (aggressive, must move stock)
    - amplify: 15% (ride momentum with mild promo)
    - margin: 25% (high margin can absorb deeper discount)
    - diversify: 12% (small nudge to spread demand)
    - no signal: freshness-based (20% Day-1, 0% Fresh)
    """
    from api.freshness_service import get_discount_rate, get_sellable_batches
    
    STRATEGY_DISCOUNT = {
        "clearance": 40,
        "amplify": 15,
        "margin": 25,
        "diversify": 12,
    }
    
    try:
        batches = get_sellable_batches()
        freshness_map = {}
        for b in (batches.data or []):
            pn = b.get("product_name", "")
            f = b.get("freshness_status", "Fresh")
            if pn not in freshness_map or f == "Day-1":
                freshness_map[pn] = f
        
        # Try to get RecommendationAgent priorities for dynamic discounts
        priority_map = {}
        try:
            cached = _latest_cached_synthesis("profit_root_cause")
            for p in _build_priority_recommendations(_priority_context(cached)):
                prod = p.get("product", "").lower().replace(" ", "_")
                strategy = p.get("strategy", "")
                priority_map[prod] = {
                    "strategy": strategy,
                    "discount_pct": STRATEGY_DISCOUNT.get(strategy, 20),
                    "reason": p.get("reason", ""),
                }
        except Exception:
            pass  # Fall through to freshness-based
        
        # Auto-clearance: Day-1 items get clearance strategy even without S5 cache
        discounts = {}
        for pn in req.products:
            freshness = freshness_map.get(pn, "Fresh")
            base_discount = int(get_discount_rate(freshness) * 100)
            
            if pn in priority_map:
                priority = priority_map[pn]
                final_pct = min(max(base_discount, priority["discount_pct"]), 50)
                discounts[pn] = {
                    "discount_pct": final_pct,
                    "freshness": freshness,
                    "strategy": priority["strategy"],
                    "reason": priority["reason"],
                    "dynamic": True,
                }
            elif freshness == "Day-1":
                discounts[pn] = {
                    "discount_pct": min(max(base_discount, 40), 50),
                    "freshness": freshness,
                    "strategy": "clearance",
                    "reason": "Day-1 stock: automatic clearance discount",
                    "dynamic": True,
                }
            else:
                discounts[pn] = {
                    "discount_pct": base_discount,
                    "freshness": freshness,
                    "dynamic": False,
                }
        
        return {"discounts": discounts}
    except Exception as e:
        logger.warning("Discount lookup failed: %s", e)
        return {"discounts": {pn: {"discount_pct": 0, "freshness": "Fresh", "dynamic": False} for pn in req.products}}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
