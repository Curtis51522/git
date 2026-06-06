# LLM Synthesis - converts structured agent outputs into natural language summary
# Calls DeepSeek API to generate conversational, actionable responses.
# Phase 3: comparison template for multi-product queries.
# Phase 3.1: causal narrative integration.
import os, sys, logging, httpx
from typing import Dict, Any, Optional

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

logger = logging.getLogger("s5.llm_synth")

try:
    from config.settings import DEEPSEEK_API_KEY as _PARENT_KEY
except ImportError:
    _PARENT_KEY = None

LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", _PARENT_KEY or "")
LLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

SYNTHESIS_ENABLED = bool(LLM_API_KEY)

PROMPT_TEMPLATES = {
    "stock_query": """You are a bakery operations advisor. Given the following structured analysis, write a concise, factual summary (2-3 sentences) for a bakery manager.

Query: {query}
Decision: {decision}
Priority: {priority}
Agents:
{agent_summaries}
Conflicts: {conflicts}
Counterfactual: {counterfactual}

Rules:
- The DECISION field is authoritative. Do NOT contradict or override it.
- Be direct and factual. Report the numbers as they are - do not sugarcoat.
- If priority is critical or warning, use direct language about the risk.
- If there is a shortage or stockout risk, say so explicitly with the gap size.
- If conflicts exist, mention the contradiction honestly rather than hiding it.
- Do NOT say "healthy", "strong", or "balanced" unless the data genuinely supports it.
- Keep it under 80 words.""",

    "comparison": """You are a bakery operations advisor. The user wants to COMPARE products. Write a factual side-by-side comparison (3-4 sentences).

Query: {query}
Decision: {decision}
Agents:
{agent_summaries}
Conflicts: {conflicts}
Associations: {associations}

Rules:
- Compare products directly with numbers: "X has Y stock vs Z demand".
- Rank by urgency: which product needs attention first and why.
- The DECISION field is authoritative - never contradict it.
- Report shortages and surpluses honestly. Do not minimize gaps.
- Do NOT say "balanced" or "healthy" unless every product's stock is within 10% of demand.
- Keep it under 100 words.""",

    "waste_analysis": """You are a bakery waste reduction specialist. Write a factual summary (2-3 sentences) explaining the waste situation.

Query: {query}
Decision: {decision}
Agents:
{agent_summaries}
Counterfactual: {counterfactual}
Causal narrative: {causal_narrative}

Rules:
- The DECISION field is authoritative. Report its finding directly.
- If the Decision says UNDERSTOCK, say waste is NOT the real problem - understock is.
- If the Decision says WASTE RISK, report the overstock ratio and at-risk units.
- Do NOT say "healthy balance" when stock is far below or above demand.
- Keep it under 80 words.""",

    "promo_eval": """You are a bakery pricing strategist. Write a factual recommendation (2-3 sentences).

Query: {query}
Decision: {decision}
Agents:
{agent_summaries}
Conflicts: {conflicts}

Rules:
- The DECISION field is authoritative.
- State the recommended discount and the surplus/shortfall driving it.
- Reference urgency level (LOW/MEDIUM/HIGH) and explain which dimensions (F/S/M/T/P) drive it.
- If the user suggested a different discount, explain why the recommendation differs.
- Keep it under 80 words.""",

    "schedule_audit": """You are a bakery staffing coordinator. Write a factual summary (1-2 sentences).

Query: {query}
Decision: {decision}
Agents:
{agent_summaries}
Conflicts: {conflicts}

Rules:
- State exact staffing levels with numbers.
- If there is a gap (no baker/no cashier), say so explicitly with the consequence.
- Do NOT say "looks good" or "no issues" unless every required role is filled.
- Keep it under 50 words.""",

    "profit_analysis": """You are a bakery financial analyst. Write a factual summary (2-3 sentences).

Query: {query}
Decision: {decision}
Agents:
{agent_summaries}

Rules:
- State the profit margin and revenue/cost breakdown precisely.
- If the main cost driver is inventory overstock, say so with specific numbers.
- Do NOT inflate or downplay the margin.
- Keep it under 80 words.""",

    "cross_source_audit": """You are a bakery operations auditor. Write a factual executive summary (3-4 sentences). NEVER sugarcoat.

Query: {query}
Decision: {decision}
Agents:
{agent_summaries}
Conflicts: {conflicts}
Causal calibration: {causal_calibration}
Causal narrative: {causal_narrative}

Rules:
- The DECISION field is authoritative. Report it verbatim if needed.
- OPEN with the most critical finding first (shortage, overstock, conflict).
- If conflicts exist, acknowledge them: "The audit found X conflict: [detail]".
- Report capacity vs demand exactly: "Capacity: X, Demand: Y, Gap: Z".
- Do NOT say "healthy", "strong", "balanced", "no conflicts" when conflicts exist.
- Do NOT say "no corrective actions required" when the Decision recommends baking/production changes.
- The manager needs to know what is WRONG, not what is fine.
- Keep it under 100 words.""",
}

DEFAULT_TEMPLATE = """You are a bakery AI assistant. Write a concise, FACTUAL summary (2-3 sentences). Do not sugarcoat.

Query: {query}
Decision: {decision}
Agents:
{agent_summaries}
Conflicts: {conflicts}

Rules:
- The DECISION field is authoritative - never contradict it.
- Report the actual numbers. Do not use vague terms like "healthy" or "balanced" unless the data supports it.
- If there are conflicts, mention them directly.
- Be honest about what is wrong, not just what is right.
- Keep it under 80 words."""


def _build_prompt(intent: str, query: str, decision: str, priority: str,
                  agent_summaries: str, conflicts: str, counterfactual: str = "",
                  causal_calibration: str = "", causal_narrative: str = "",
                  product: str = "", associations: str = "") -> str:
    if "," in product and intent == "stock_query":
        template = PROMPT_TEMPLATES.get("comparison", DEFAULT_TEMPLATE)
    else:
        template = PROMPT_TEMPLATES.get(intent, DEFAULT_TEMPLATE)
    return template.format(
        query=query, decision=decision, priority=priority,
        agent_summaries=agent_summaries, conflicts=conflicts or "None",
        counterfactual=counterfactual or "N/A",
        causal_calibration=causal_calibration or "N/A",
        causal_narrative=causal_narrative or "N/A",
        associations=associations or "N/A",
    )


async def synthesize(query: str, intent: str, decision: str, priority: str,
                     agent_data: Dict[str, Any], conflicts: list,
                     counterfactual: Optional[dict] = None,
                     causal_calibration: Optional[dict] = None,
                     causal_narrative: str = "",
                     product: str = "",
                     associations: str = "") -> Optional[str]:
    if not SYNTHESIS_ENABLED:
        return None

    lines = []
    for name, data in agent_data.items():
        opinion = data.get("opinion", "")
        if opinion:
            lines.append(f"  {name}: {opinion}")
    agent_text = "\n".join(lines) if lines else "No agent data"

    conflict_text = ", ".join(conflicts) if conflicts else "None"

    cf_text = ""
    if counterfactual and "scenarios" in counterfactual:
        cf_parts = []
        for label, sc in counterfactual["scenarios"].items():
            cf_parts.append(f"{label}: bake={sc['bake']}, waste={sc['waste']}, shortage={sc['shortage']}, cost=RM{sc['cost_rm']}")
        cf_text = "; ".join(cf_parts)

    causal_text = ""
    if causal_calibration:
        causal_text = f"waste_loss={causal_calibration.get('waste_loss_per_unit', '?')}, top_driver={causal_calibration.get('top_waste_driver', '?')}"

    prompt = _build_prompt(intent, query, decision, priority, agent_text, conflict_text, cf_text, causal_text, causal_narrative, product, associations)

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{LLM_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
                json={"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 200, "temperature": 0.4},
            )
            resp.raise_for_status()
            data = resp.json()
            summary = data["choices"][0]["message"]["content"].strip()
            logger.info("LLM synthesis: %s", summary[:100])
            return summary
    except Exception as e:
        logger.warning("LLM synthesis failed: %s", e)
        return None
