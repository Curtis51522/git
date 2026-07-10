# s5_agent/server.py - S5 dashboard analysis server (port 8001)
import asyncio, logging, sys, os
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

from s5_agent.graph.registry import module_to_template
from s5_agent.graph.runner import run_s5_graph
from s5_agent.graph.state import S5Request
from s5_agent.agents.recommendation import RecommendationAgent
from s5_agent.discount_policy import STRATEGY_DISCOUNT_PCT, get_live_discounts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("s5.server")

@asynccontextmanager
async def lifespan(app):
    logger.info("S5 LangGraph analysis starting with modules: %s", ", ".join(sorted(LANGGRAPH_MODULES)))
    yield

app = FastAPI(title="S5 Dashboard Analysis - Multi-Agent Bakery Intelligence", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_response_cache: dict = {}
_response_cache_max = 100
recommendation_agent = RecommendationAgent("RecommendationAgent")
LANGGRAPH_MODULES = frozenset({"inventory", "revenue", "forecast", "wastage", "promotion_mix"})

class ModuleAnalyzeRequest(BaseModel):
    module: str
    date: str = ""
    params: dict = {}
    lang: str = "en"
    force_refresh: bool = False

@app.get("/health")
async def health():
    return {"status": "ok", "architecture": "langgraph", "modules": sorted(LANGGRAPH_MODULES)}

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
    agent = recommendation_agent
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

@app.post("/analyze/module")
async def analyze_module(req: ModuleAnalyzeRequest):
    lang = _normalize_lang(req.lang)
    module = (req.module or "").strip().lower()
    if module not in LANGGRAPH_MODULES:
        supported = ", ".join(sorted(LANGGRAPH_MODULES))
        raise HTTPException(400, f"Unsupported S5 module: {req.module}. Supported modules: {supported}")

    template_id = module_to_template(module)
    graph_params = {
        "date": req.date,
        "module": module,
        "product": "all",
        **(req.params or {}),
    }
    graph_request = S5Request(
        query=module,
        module=module,
        params=graph_params,
        lang=lang,
        force_refresh=req.force_refresh,
    )
    graph_response = await run_s5_graph(template_id, graph_request)
    response_payload = graph_response.model_dump()
    cache_key = _response_cache_key(template_id, graph_params, lang)
    _response_cache[cache_key] = {"intent": template_id, **response_payload}
    if len(_response_cache) > _response_cache_max:
        oldest = next(iter(_response_cache))
        del _response_cache[oldest]
    return response_payload





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
    """Return validated discounts from live operations and cached revenue evidence.
    
    Priority strategy -> discount mapping:
    - clearance: 40% (aggressive, must move stock)
    - amplify: 15% (ride momentum with mild promo)
    - margin: 25% (high margin can absorb deeper discount)
    - diversify: 12% (small nudge to spread demand)
    - no signal: 0% for fresh stock
    """
    try:
        priority_map = {}
        try:
            cached = _latest_cached_synthesis("profit_root_cause")
            for p in _build_priority_recommendations(_priority_context(cached)):
                prod = p.get("product", "").lower().replace(" ", "_")
                strategy = p.get("strategy", "")
                priority_map[prod] = {
                    "strategy": strategy,
                    "discount_pct": STRATEGY_DISCOUNT_PCT.get(strategy, 0),
                    "reason": p.get("reason", ""),
                }
        except Exception:
            priority_map = {}

        return {"discounts": get_live_discounts(req.products, priority_map=priority_map)}
    except Exception as e:
        logger.warning("Discount lookup failed: %s", e)
        return {"discounts": {pn: {"discount_pct": 0, "freshness": "Fresh", "dynamic": False} for pn in req.products}}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
