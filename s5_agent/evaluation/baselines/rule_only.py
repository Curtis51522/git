def _total_units(value: object) -> int:
    if isinstance(value, dict):
        return sum(int(amount or 0) for amount in value.values())
    return int(value or 0)


def inventory_rule_only(scenario: dict) -> dict:
    inventory = scenario.get("inventory", {})
    total_inventory = _total_units(inventory)
    day1_available = int(scenario.get("day1_available", 0) or 0)
    product = scenario.get("product") or "unknown product"

    recommendations = []
    if day1_available > 0:
        recommendations.append(
            {
                "id": "rule_clear_day1",
                "action": "Apply Day-1 clearance promotion.",
                "evidence_ids": ["scenario_day1_available"],
            }
        )

    return {
        "summary": (
            f"{product} has {total_inventory} inventory units, including "
            f"{day1_available} Day-1 units."
        ),
        "recommendations": recommendations,
    }


def _raw_inputs(scenario: dict, key: str) -> dict:
    value = scenario.get("raw_inputs", {}).get(key, {})
    return value.get("data", {}) if isinstance(value, dict) and "data" in value else value


def _baseline_response(summary: str, recommendations: list[dict]) -> dict:
    return {
        "summary": summary,
        "recommendations": recommendations,
        "agent_outputs": [
            {
                "agent_name": "RuleOnlyBaseline",
                "confidence": 0.4,
            }
        ],
        "verification_report": {
            "passed": False,
            "conflicting_claims": [],
            "data_quality_warnings": ["Baseline output is not evidence-verified."],
        },
    }


def rule_only_response(scenario: dict) -> dict:
    module = scenario.get("module", "")
    if module == "inventory":
        result = inventory_rule_only(scenario)
        recommendations = [
            {key: value for key, value in recommendation.items() if key != "evidence_ids"}
            for recommendation in result["recommendations"]
        ]
        return _baseline_response(result["summary"], recommendations)

    if module == "revenue":
        data = _raw_inputs(scenario, "profit")
        revenue = float(data.get("today_revenue", 0.0) or 0.0)
        profit = float(data.get("today_profit", 0.0) or 0.0)
        margin_pct = round(profit / max(revenue, 1.0) * 100, 2)
        recommendations = []
        if margin_pct < 20:
            recommendations.append(
                {
                    "id": "rule_margin_review",
                    "action": "Review low-margin revenue performance.",
                }
            )
        return _baseline_response(
            f"Revenue is {revenue} and profit margin is {margin_pct} percent.",
            recommendations,
        )

    if module == "forecast":
        data = _raw_inputs(scenario, "production")
        buffer = float(data.get("buffer", 1.0) or 1.0)
        recommendations = []
        if buffer > 1.3:
            recommendations.append(
                {
                    "id": "rule_buffer_review",
                    "action": "Reduce production buffer when it exceeds the rule threshold.",
                }
            )
        return _baseline_response(
            f"Production buffer is {buffer}.",
            recommendations,
        )

    return _baseline_response("No rule-only baseline is available.", [])
