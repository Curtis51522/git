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
- Write in natural, flowing English. The DECISION field is authoritative - use it as your source of truth.
- LANGUAGE: Detect the query language. If the query is in Malay or mixed Malay-English, respond in natural Malay (Bahasa Malaysia). If English, respond in English. Match the user's language.
- Be direct and honest. If there is a shortage, state the gap explicitly: "You are short by X units."
- If there are conflicts or risks, address them plainly without hiding behind vague terms.
- Do not use words like "healthy" or "balanced" unless the data genuinely supports it.
- Keep it under 80 words.""",

    "comparison": """You are a bakery operations advisor. The user wants to COMPARE products. Write a factual side-by-side comparison (3-4 sentences).

Query: {query}
Decision: {decision}
Agents:
{agent_summaries}
Conflicts: {conflicts}
Associations: {associations}

Rules:
- Write in natural, flowing English. Compare products side-by-side in a conversational tone.
- LANGUAGE: Detect the query language. If the query is in Malay or mixed Malay-English, respond in natural Malay (Bahasa Malaysia). If English, respond in English. Match the user's language.
- Use ONLY per-product numbers from the DECISION text. Never aggregate totals - keep each product separate.
- End by identifying which product needs attention first and why.
- The DECISION field is authoritative - never contradict it.
- Report shortages and surpluses honestly without minimizing them.
- Do not say "balanced" or "healthy" unless every product's stock is within 10% of demand.
- Keep it under 100 words.""",

    "waste_analysis": """You are a bakery waste reduction specialist. Write a factual summary (2-3 sentences) explaining the waste situation.

Query: {query}
Decision: {decision}
Agents:
{agent_summaries}
Counterfactual: {counterfactual}
Causal narrative: {causal_narrative}

Rules:
- Write in natural, flowing English. The DECISION field is authoritative.
- LANGUAGE: Detect the query language. If the query is in Malay or mixed Malay-English, respond in natural Malay (Bahasa Malaysia). If English, respond in English. Match the user's language.
- If the Decision says UNDERSTOCK, explain that waste is not the real concern - the priority is restocking.
- If the Decision says WASTE RISK, describe which products are overstocked and how many units are at risk.
- Do not use "healthy balance" when the numbers show a clear imbalance.
- Keep it under 80 words.""",

    "promo_eval": """You are a bakery pricing strategist. Write a factual recommendation (2-3 sentences).

Query: {query}
Decision: {decision}
Agents:
{agent_summaries}
Conflicts: {conflicts}

Rules:
- Write in natural, flowing English. The DECISION field is authoritative.
- LANGUAGE: Detect the query language. If the query is in Malay or mixed Malay-English, respond in natural Malay (Bahasa Malaysia). If English, respond in English. Match the user's language.
- State the recommended discount and explain what is driving it (surplus, stock levels, urgency).
- Reference urgency level and briefly note which factors (freshness, stock, market, trend, promo sensitivity) are most influential.
- If the user suggested a different discount, explain why the recommendation differs.
- Keep it under 80 words.""",

    "schedule_audit": """You are a bakery staffing coordinator. Write a factual summary (1-2 sentences).

Query: {query}
Decision: {decision}
Agents:
{agent_summaries}
Conflicts: {conflicts}

Rules:
- Write in natural, flowing English. State the exact staffing levels with numbers.
- LANGUAGE: Detect the query language. If the query is in Malay or mixed Malay-English, respond in natural Malay (Bahasa Malaysia). If English, respond in English. Match the user's language.
- If any role is unfilled, explain the consequence clearly.
- Only say things are fine when every required role is actually filled.
- Keep it under 50 words.""",

    "profit_analysis": """You are a bakery financial analyst. Write a factual summary (2-3 sentences).

Query: {query}
Decision: {decision}
Agents:
{agent_summaries}

Rules:
- Write in natural, flowing English. State the profit margin and revenue/cost breakdown clearly.
- LANGUAGE: Detect the query language. If the query is in Malay or mixed Malay-English, respond in natural Malay (Bahasa Malaysia). If English, respond in English. Match the user's language.
- Compare inventory against demand using specific numbers.
- Do not inflate or downplay the margin - report it as it is.
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
- LANGUAGE: Detect the query language. If Malay or mixed, respond in natural Malay. If English, respond in English.
- Write naturally and connect sentences smoothly. Avoid robotic bullet-point style.
- The DECISION field is authoritative. Translate its format into plain language: "bake 54 (stock 10+54=64/64, 100%)" means "bake 54 croissants, bringing total stock to 64 to fully meet the forecast of 64."
- OPEN with the most critical finding in a natural lead sentence, then explain the context.
- If conflicts exist, mention them honestly, then immediately explain how the Decision resolves them.
- Report capacity vs demand in context. If total bake is within capacity and covers demand, say so naturally.
- Do NOT use words like "healthy", "strong", "balanced" unless genuinely supported by all data.
- Do NOT claim the Decision is "ignoring" anything when it clearly addresses the issue.
- Focus on what the manager needs to act on, without unnecessary drama.
- Keep it under 100 words.""",
}

DEFAULT_TEMPLATE = """You are a bakery AI assistant. Write a concise, FACTUAL summary (2-3 sentences). Do not sugarcoat.

Query: {query}
Decision: {decision}
Agents:
{agent_summaries}
Conflicts: {conflicts}

Rules:
- Write in natural, flowing English. The DECISION field is authoritative - never contradict it.
- LANGUAGE: Detect the query language. If the query is in Malay or mixed Malay-English, respond in natural Malay (Bahasa Malaysia). If English, respond in English. Match the user's language.
- Report the actual numbers clearly. Avoid vague terms unless the data genuinely supports them.
- If there are conflicts, address them directly and honestly.
- Keep it under 80 words."""


def _build_prompt(intent: str, query: str, decision: str, priority: str,
                  agent_summaries: str, conflicts: str, counterfactual: str = "",
                  causal_calibration: str = "", causal_narrative: str = "",
                  product: str = "", associations: str = "", memory_context: str = "") -> str:
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
        memory_context=memory_context or "No historical baseline available.",
    )


async def synthesize(query: str, intent: str, decision: str, priority: str,
                     agent_data: Dict[str, Any], conflicts: list,
                     counterfactual: Optional[dict] = None,
                     causal_calibration: Optional[dict] = None,
                     causal_narrative: str = "",
                     product: str = "",
                     associations: str = "", memory_context: str = "") -> Optional[str]:
    if not SYNTHESIS_ENABLED:
        return None

    lines = []
    for name, data in agent_data.items():
        opinion = data.get("opinion", "")
        if opinion:
            lines.append(f"  {name}: {opinion}")
    agent_text = "\n".join(lines) if lines else "No agent data"
    # For comparison, replace aggregated agent opinions with per-product decision
    if intent == "comparison_analysis" and decision:
        agent_text = "Per-product data: " + decision

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

    prompt = _build_prompt(intent, query, decision, priority, agent_text, conflict_text, cf_text, causal_text, causal_narrative, product, associations, memory_context)

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
