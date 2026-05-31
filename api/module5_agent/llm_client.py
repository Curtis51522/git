"""
DeepSeek LLM client -- OpenAI-compatible API call.

DeepSeek's API is fully compatible with the OpenAI SDK.
Just point base_url to https://api.deepseek.com/v1.

Models:
- deepseek-chat      (DeepSeek-V3, cheap, ~$0.27/M input tokens)
- deepseek-reasoner  (DeepSeek-R1, reasoning model)
"""

import os
import logging
from openai import OpenAI

logger = logging.getLogger("s5.llm")

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"


def get_deepseek_client() -> OpenAI | None:
    """Return a configured DeepSeek client, or None if no API key."""
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        logger.warning("DEEPSEEK_API_KEY not set -- falling back to mock")
        return None
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def call_deepseek(prompt: str, system: str = "", max_tokens: int = 256) -> str:
    """Call DeepSeek API and return the response text.
    
    Falls back to empty string on failure.
    """
    client = get_deepseek_client()
    if client is None:
        return ""

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("DeepSeek API call failed: %s", e)
        return ""


def compose_summary_real(result: dict, query: str = "", intent: str = "stock_query") -> str:
    """Real DeepSeek call for store-manager decision summary."""
    prompt = _build_summary_prompt(result, query, intent)
    system = (
        "You are a bakery operations assistant for a Malaysian bakery-cafe. "
        "Answer the manager's question DIRECTLY based on the data provided. "
        "Be specific - mention product names, numbers, and actionable advice. "
        "If the question is about stock, talk about stock. If about waste, talk about waste. "
        "If about schedule, talk about staffing. If about promos, talk about ROI. If a full audit, list issues by severity then confirm what is healthy. "
        "Keep it under 3 sentences. Use RM currency. "
        "CRITICAL: Only mention public holidays, Ramadan, or special events if the External context explicitly says one is active on the target date. "
        "Do NOT invent or assume holidays. If the SHAP attribution mentions weather or day-of-week, use those instead. "
        "NEVER mention internal technical details like 'R6', 'SHAP', or 'rule results'."
    )
    return call_deepseek(prompt, system, max_tokens=200)


def compose_script_real(combo_data: list):
    """Real DeepSeek call for staff upselling script. Returns list of script dicts."""
    if not combo_data:
        return []
    prompt = _build_script_prompt(combo_data)
    system = (
        "You are a warm, friendly cashier at a Malaysian bakery-cafe. "
        "Your job is to naturally recommend combo deals. "
        "Voice: Casual, warm. Use natural English. "
        "For each combo, write 1-2 short sentences (max 40 words). "
        "Mention bread name, coffee name, price, and savings if any. "
        "Separate each script with '---SCRIPT---' on its own line."
    )
    full_text = call_deepseek(prompt, system, max_tokens=300)
    scripts = [s.strip() for s in full_text.split("---SCRIPT---") if s.strip()]
    results = []
    for i, item in enumerate(combo_data[:len(scripts)]):
        bread = item.get("product_name", item.get("bread", "pastry"))
        coffee = item.get("coffee_name", item.get("coffee", "drink"))
        results.append({
            "products": f"{bread} + {coffee}",
            "script": scripts[i] if i < len(scripts) else scripts[0],
        })
    return results

def _build_summary_prompt(result: dict, query: str = "", intent: str = "stock_query") -> str:
    """Build a prompt that includes the user's question and intent for context."""
    parts = [f"Manager's question: {query}", f"Intent: {intent}"]

    # Handle multi-product comparison
    if result.get("multi_product") and result.get("products"):
        parts.append("")
        parts.append("=== Multi-Product Comparison ===")
        prods = result["products"]
        for i, p in enumerate(prods):
            cover = f"{(p['inventory']/max(p['forecast'],1)*100):.0f}%"
            risk = "STOCKOUT RISK" if p['inventory'] < p['forecast'] * 0.3 else ("OVERSTOCK" if p['inventory'] > p['forecast'] * 2 else "healthy")
            parts.append(f"Product {i+1}: {p['name']} | Forecast: {p['forecast']}u | Inventory: {p['inventory']}u | Coverage: {cover} | Status: {risk}")
        most_urgent = min(prods, key=lambda x: x['inventory']/max(x['forecast'],1))
        parts.append(f"Most urgent: {most_urgent['name']} (coverage {(most_urgent['inventory']/max(most_urgent['forecast'],1)*100):.0f}%)")
        parts.append("")
        parts.append("Compare these products. Which needs more attention? Mention specific numbers. Keep it under 3 sentences.")
        return chr(10).join(parts)

    if intent == "stock_query":
        target_date = result.get("target_date", "")
        if target_date:
            parts.append(f"Target date: {target_date}")
        low = result.get("forecast_low", result.get("forecast", 0))
        high = result.get("forecast_high", result.get("forecast", 0))
        carry = result.get("carryover", 0)
        parts.append(f"Forecast range: {low}-{high} units (midpoint: {result.get('forecast', 'N/A')})")
        parts.append(f"Current inventory: {result.get('inventory', 'N/A')} units")
        if carry > 0 and result.get('inventory', 0) == carry:
            parts.append("(This inventory is carryover from the previous day's production)")
        elif carry > 0:
            parts.append(f"Carryover from previous day: {carry} units")
        parts.append(f"Production capacity: {result.get('capacity', 'N/A')} units")
        parts.append("STOCKING RULE: Recommend stocking to the UPPER bound of the forecast range. restock = upper_bound - inventory - carryover. Explain that the upper bound is used to avoid stockouts.")
        
        user_target = result.get('user_target')
        if user_target is not None:
            parts.append(f"Manager's target: {user_target} units (system recommendation: {result.get('system_recommendation', 0)} units)")
            feasible = result.get('user_target_feasible', True)
            blocker = result.get('user_target_blocker', '')
            warning = result.get('user_target_warning', '')
            risk = result.get('user_target_risk', '')
            
            if not feasible:
                gap = result.get('user_target_gap', 0)
                parts.append(f"FEASIBILITY: Target of {user_target} is NOT feasible. Capacity exceeded by {gap} units.")
            elif risk == 'high':
                parts.append(f"RISK: Target {user_target} significantly exceeds forecast. High overproduction risk: {warning}")
            elif risk == 'medium':
                parts.append(f"CAUTION: Target {user_target} exceeds forecast moderately. {warning}")
            elif warning:
                parts.append(f"NOTE: {warning}")
            else:
                parts.append(f"Target {user_target} is aligned with forecast. No issues.")
            
            parts.append("\nIMPORTANT: The manager asked about a specific target. Address whether their requested quantity is feasible, and mention any risks of overproduction or underproduction compared to the forecast.")
        else:
            parts.append(f"Recommended restock: {result.get('recommended_restock', 0)} units")
    elif intent == "waste_analysis":
        parts.append(f"Forecast vs actual deviations: {result.get('deviations', [])}")
        parts.append(f"Average deviation: {result.get('avg_deviation', 0)}%")
        parts.append(f"Waste flag: {result.get('waste_flag', False)}")
    elif intent == "promo_eval":
        parts.append(f"Net profit: RM{result.get('net_profit', 0)}")
        parts.append(f"ROI: {result.get('roi_percent', 0)}%")
        parts.append(f"Recommendation: {result.get('recommendation', 'N/A')}")
    elif intent == "schedule_audit":
        sched_summary = result.get('schedule_summary', {})
        target_date = result.get('target_date', '')
        if sched_summary:
            if target_date:
                parts.append(f"Schedule from {target_date} shows {sched_summary.get('total_shifts', 0)} shifts over the next days")
            else:
                parts.append(f"Schedule covers {sched_summary.get('days',[])}  days")
            parts.append(f"Total shifts: {sched_summary.get('total_shifts', 0)}")
            parts.append(f"Employees on duty: {sched_summary.get('employees', 0)}")
            parts.append(f"Roles breakdown: {sched_summary.get('roles', {})}")
        anomalies = result.get('anomalies', [])
        if anomalies:
            parts.append(f"Anomalies found: {len(anomalies)}")
        if target_date:
            parts.append(f"IMPORTANT: The manager asked about {target_date} specifically. Focus the answer on that date.")
        parts.append("GUIDELINES: List the actual numbers from the breakdown above (bakers, baristas, cashiers, cleaners). Do NOT invent shortages or 'short by X' claims unless explicit gaps exist. If all roles are covered, say so clearly. Keep it to 2-3 sentences.")
        # If query asks about a specific product, the answer should be driven by product data, not schedule gaps
        target_product = result.get('target_product', '')
        if target_product:
            parts.append(f"IMPORTANT: The manager asked about {target_product} specifically. Whether bakers are needed depends on {target_product}'s inventory vs forecast, NOT on general schedule gaps. If inventory exceeds forecast, say the product needs NO additional bakers. If inventory is below forecast, say more bakers are needed. Answer the question DIRECTLY about {target_product}, then mention schedule gaps only as secondary context.")
    elif intent == "cross_source_audit":
        parts.append(f"Store health status: {result.get('status', 'unknown')}")
        parts.append(f"Issue count: {result.get('issue_count', 0)}")
        issues = result.get('issues', [])
        if issues:
            parts.append("Issues found:")
            for i in issues[:10]:
                parts.append(f"  [{i.get('rule', '?')}] {i.get('severity', '?')}: {i.get('message', '')}")
        all_clear = result.get('all_clear', [])
        if all_clear:
            parts.append("Passed checks:")
            for c in all_clear[:5]:
                parts.append(f"  {c}")
        parts.append("\nSummarize the store health concisely - list issues first, then all-clear items.")

    # Inject R12 causal reasoning if available
    r12_causal = result.get("L4_R12_causal_report", "")
    r12_shap = result.get("L4_R12_shap", [])
    r12_ctx = result.get("L4_R12_external_context", {})
    r12_top = result.get("L4_R12_top_driver", "")
    if r12_causal or r12_shap:
        if r12_causal:
            parts.append(f"\nCAUSAL EXPLANATION (from SHAP): {r12_causal}")
        elif r12_shap:
            shap_str = "; ".join(c['effective_sign'] + c['label'] + '(' + str(round(c['abs_impact'], 1)) + ')' for c in r12_shap[:3])
            parts.append(f"\nSHAP ATTRIBUTION: {shap_str}. Top driver: {r12_top}.")
        parts.append("Incorporate this causal insight when explaining WHY the forecast changed.")
        if r12_ctx:
            ctx_bits = []
            if r12_ctx.get("is_public_holiday"):
                ctx_bits.append(f"Public holiday: {r12_ctx.get("holiday_name", "yes")}")
            if r12_ctx.get("is_ramadan"):
                ctx_bits.append("Ramadan period")
            if r12_ctx.get("day_of_week"):
                ctx_bits.append(f"Day: {r12_ctx["day_of_week"]}")
            if ctx_bits:
                parts.append(f"External context: {"; ".join(ctx_bits)}. Only mention these if actually active.")

    warnings = result.get("audit_warnings", [])
    if warnings:
        parts.append(f"Data quality notes: {'; '.join(warnings)}")

    parts.append("\nAnswer the manager's question directly and concisely.")
    return "\n".join(parts)


def _build_script_prompt(combo_data: list) -> str:
    if not combo_data:
        return "No combos. Suggest the customer grab any pastry with a latte."

    lines = []
    for i, item in enumerate(combo_data[:3], 1):
        bread = item.get("product_name", item.get("bread", "pastry"))
        coffee = item.get("coffee_name", item.get("coffee", "drink"))
        score = item.get("score", item.get("total_score", 0))
        price = item.get("combo_price", item.get("total_price", 0))
        savings = item.get("savings", 0)
        freshness = item.get("freshness_status", "Fresh")
        stock = item.get("stock_qty", "?")

        line = f"#{i}: {bread} + {coffee} | combo RM{price:.2f}"
        if savings > 0:
            line += f" | save RM{savings:.2f}"
        line += f" | {freshness}"
        if freshness != "Fresh":
            pct = {"Day-1": "10%"}.get(freshness, "")
            line += f" ({pct} off)"
        line += f" | stock: {stock}"
        lines.append(line)

    prompt = "Best combos for this customer:\n" + "\n".join(lines)
    prompt += (
        "\n\nWrite a natural recommendation for EACH combo. "
        "Separate with ---SCRIPT---. "
        "Emphasize savings, hint limited-time deals, sound like a real cashier."
    )
    return prompt



def compose_alert_description(feature_values: dict, issues: list, audit_result: dict, product: str = "") -> str:
    """Generate a human-readable alert description from anomaly detection results.

    Called by B1 Monitor when an anomaly is detected.
    Falls back to a rule-based message if DeepSeek is unavailable.
    """
    deviation = feature_values.get("deviation_pct", 0)
    stock_cov = feature_values.get("stock_coverage", 1.0)
    hc_gap = feature_values.get("headcount_gap", 0)
    waste = feature_values.get("waste_rate", 0)
    cap_pres = feature_values.get("capacity_pressure", 0)
    anomaly_score = feature_values.get("anomaly_score", 0)
    severity = feature_values.get("severity", "warning")

    forecast = audit_result.get("forecast", "N/A")
    inventory = audit_result.get("inventory", "N/A")
    capacity = audit_result.get("capacity", "N/A")

    issue_msgs = [i.get("message", "") for i in (issues or [])[:5]]
    issue_text = "\n".join(f"  - {m}" for m in issue_msgs) if issue_msgs else "  - No specific issues flagged"

    prompt = f"""A bakery monitoring system detected an anomaly. Generate a concise, actionable alert for the store manager.

Severity: {severity}
Anomaly Score: {anomaly_score:.3f} (lower = more anomalous)

Product: {product or 'all products'}

Store State:
- Forecasted demand: {forecast} units
- Current inventory: {inventory} units
- Production capacity: {capacity} units
- Stock coverage: {stock_cov:.0%} of daily demand
- Forecast deviation: {deviation:.0f}%
- Headcount gap: {hc_gap:.0f} missing staff
- Waste rate: {waste:.0f}%
- Capacity pressure: {cap_pres:.0%}

Issues detected:
{issue_text}

Write ONE sentence (max 30 words) that:
- States what's wrong in plain language
- Mentions the affected product or area if clear
- Suggests ONE concrete action the manager should take
- Uses RM currency if mentioning money
- Does NOT mention technical terms like "anomaly score" or "feature vector"

Alert:"""

    system = (
        "You are a bakery operations monitor. "
        "Write concise alerts that a busy store manager can read in 5 seconds and act on immediately. "
        "Be specific and actionable."
    )

    result = call_deepseek(prompt, system, max_tokens=80)
    if result:
        return result

    # Fallback: rule-based description
    reasons = []
    if stock_cov < 0.2:
        reasons.append(f"Stock critically low ({(stock_cov*100):.0f}% of daily demand) — restock immediately")
    elif stock_cov < 0.5:
        reasons.append(f"Stock running low ({(stock_cov*100):.0f}% coverage)")
    if hc_gap >= 1:
        reasons.append(f"Understaffed by {hc_gap:.0f} people �� adjust schedule")
    if deviation > 30:
        reasons.append(f"Forecast off by {deviation:.0f}% �� verify production plan")
    if cap_pres > 1.0:
        reasons.append(f"Production near capacity ({(cap_pres*100):.0f}%) �� check oven load")
    if waste > 20:
        reasons.append(f"Waste rate {waste:.0f}% — review batch sizes")
    if not reasons:
        reasons.append(f"Unusual pattern detected — review store operations")
    return "; ".join(reasons)


def compose_reflection(session_id: str, episodes: list) -> str:
    """Synthesize high-level insights from a set of query episodes.

    Called by MemoryStore.generate_reflection() when 5+ episodes accumulate.
    Returns empty string if DeepSeek is unavailable.
    """
    if not episodes:
        return ""

    summary_lines = []
    products_seen = set()
    for ep in episodes[-20:]:
        p = ep.get("product", "")
        products_seen.add(p)
        summary_lines.append(
            f"Q: {ep['query'][:100]} | Intent: {ep['intent']} | A: {ep['response'][:100]}"
        )

    products_str = ", ".join(p for p in products_seen if p) or "various products"

    prompt = f"""You are analyzing a bakery manager's query history to extract insights.

Products discussed: {products_str}

Recent queries:
{chr(10).join(summary_lines)}

Based on these queries, write 2-3 concise insights (max 100 words total) about:
- What the manager seems most concerned about (e.g., stockouts, waste, staffing)
- Any recurring patterns (same product asked about repeatedly, same day of week)
- One proactive suggestion based on the patterns

Insights:"""

    system = (
        "You are a bakery operations analyst. "
        "Extract actionable patterns from query history. Be specific."
    )

    return call_deepseek(prompt, system, max_tokens=150)
