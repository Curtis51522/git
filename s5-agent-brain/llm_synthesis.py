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
    "stock_query": """You are a bakery operations advisor. Given the following structured analysis, write a concise, actionable summary (2-3 sentences) for a bakery manager.

Query: {query}
Decision: {decision}
Priority: {priority}
Agents:
{agent_summaries}
Conflicts: {conflicts}
Counterfactual: {counterfactual}

Rules:
- The DECISION field is authoritative. Do NOT contradict or override it.
- Use counterfactual data only to illustrate the decision, not to change it.
- Be direct and actionable. Use specific numbers.
- If there is a capacity gap or stockout risk, flag it prominently.
- Waste numbers in counterfactual may include pre-existing surplus from overstocked products - do not attribute to the baking decision.
- Keep it under 80 words.""",

    "comparison": """You are a bakery operations advisor. The user wants to COMPARE products. Write a side-by-side comparison (3-4 sentences).

Query: {query}
Decision: {decision}
Agents:
{agent_summaries}
Conflicts: {conflicts}
Associations: {associations}

Rules:
- Compare products directly: "X needs Y units, Z is overstocked by W".
- Rank by urgency: which product needs attention first.
- The DECISION field is authoritative - never contradict it.
- Use specific numbers for each product.
- Waste from overstocked products is PRE-EXISTING and unavoidable - do NOT blame it on baking for understocked products.
- If the Decision says bake X for product A, trust it. Counterfactual waste numbers may include surplus from OTHER products.
- Keep it under 100 words.""",

    "waste_analysis": """You are a bakery waste reduction specialist. Write a concise summary (2-3 sentences) explaining the waste situation.

Query: {query}
Decision: {decision}
Agents:
{agent_summaries}
Counterfactual: {counterfactual}
Causal narrative: {causal_narrative}

Rules:
- The DECISION field is authoritative. Do NOT override it.
- Use causal analysis to identify the root cause of waste.
- Suggest specific actions (promo, reduce production, bundle).
- Keep it under 80 words.""",

    "promo_eval": """You are a bakery pricing strategist. Write a concise recommendation (2-3 sentences).

Query: {query}
Decision: {decision}
Agents:
{agent_summaries}
Conflicts: {conflicts}

Rules:
- The DECISION field is authoritative.
- Recommend a specific discount percentage.
- If the user suggested a different discount, explain why the recommended one is more appropriate.
- Reference the urgency level (LOW/MEDIUM/HIGH) and breakdown (F/S/M/T/P) from agent data to explain what dimensions drove the discount.
- Explain why this product needs promotion.
- Mention bundle suggestions if available.
- Keep it under 80 words.""",

    "schedule_audit": """You are a bakery staffing coordinator. Write a concise summary (1-2 sentences).

Query: {query}
Decision: {decision}
Agents:
{agent_summaries}
Conflicts: {conflicts}

Rules:
- State whether staffing is adequate.
- Flag any gaps (no bakers, no cashiers).
- Keep it under 50 words.""",

    "profit_analysis": """You are a bakery financial analyst. Write a concise summary (2-3 sentences).

Query: {query}
Decision: {decision}
Agents:
{agent_summaries}

Rules:
- State the profit margin clearly.
- Identify the main cost driver.
- Keep it under 80 words.""",

    "cross_source_audit": """You are a bakery operations auditor. Write a concise executive summary (3-4 sentences).

Query: {query}
Decision: {decision}
Agents:
{agent_summaries}
Conflicts: {conflicts}
Causal calibration: {causal_calibration}
Causal narrative: {causal_narrative}

Rules:
- The DECISION field is authoritative.
- Summarize overall store health in one line.
- Call out the most critical issue if any.
- Use causal analysis to explain WHY issues exist (not just what they are).
- Production capacity vs demand: NEVER say they match if the numbers are different. Say "capacity X vs demand Y" factually. Capacity may be lower than demand because existing inventory fills the gap.
- Waste from overstocked products is PRE-EXISTING and unavoidable - do NOT blame it on baking for understocked products.
- If the Decision says bake X for product A, trust it. Counterfactual waste numbers may include surplus from OTHER products.
- Keep it under 100 words.""",
}

DEFAULT_TEMPLATE = """You are a bakery AI assistant. Write a concise, helpful summary (2-3 sentences).

Query: {query}
Decision: {decision}
Agents:
{agent_summaries}
Conflicts: {conflicts}

The DECISION field is authoritative - never contradict it. Keep it under 80 words."""


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
