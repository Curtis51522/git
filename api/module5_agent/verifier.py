"""
Verifier Agent -- Four-tier audit for S5 decision pipeline.

Architecture (redesigned):
- L1 Data Integrity: merged R1-R5, zero-token, <1ms. Fails = immediate reject.
- L2 Capacity Constraint: R8 restock vs capacity. Fails = warn, continue.
- L3 Cross-Module Contradiction: R9/R10/R11. Fails = warn, continue.
- L4 Explainability Audit: R12. Anomaly with explanation = pass; unexplained = flag.
"""

import json, logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger("s5.verifier")

# ---------------------------------------------------------------------------
# Borderline thresholds for LLM judgment
# ---------------------------------------------------------------------------
R8_CAPACITY_RATIO = 1.0
R8_BORDER_LOW = 0.9
R8_BORDER_HIGH = 1.1


def _call_deepseek_judge(rule_id: str, desc: str, data: dict, context: str) -> Tuple[bool, str]:
    """Call DeepSeek LLM to judge a borderline case."""
    try:
        from api.module5_agent.llm_client import call_deepseek

        prompt = f"""You are an auditor for a Malaysian bakery-cafe system.

Rule: {rule_id} - {desc}

Context:
{context}

Numerical data:
{json.dumps(data, default=str, indent=2)}

This rule is in the BORDERLINE zone -- the numbers are close to the threshold.
Use your judgment: is this a genuine concern worth flagging, or can it pass?

Return ONLY valid JSON:
{{"passed": true/false, "reason": "one short sentence explaining why"}}

Consider:
- Malaysian bakery context (small shop, RM pricing)
- Practical business tolerance
- Whether flagging this would create unnecessary noise"""

        system = "You are a pragmatic bakery operations auditor. Be decisive. Return only JSON."
        response = call_deepseek(prompt, system, max_tokens=150)

        if not response:
            raise ValueError("Empty LLM response")

        response = response.strip()
        if response.startswith("```"):
            parts = response.split("```")
            response = parts[1] if len(parts) > 1 else response
            if response.startswith("json"):
                response = response[4:]
        result = json.loads(response)
        return result.get("passed", True), result.get("reason", "LLM judgment unavailable")

    except Exception as exc:
        logger.warning("DeepSeek LLM judgment failed for %s: %s", rule_id, exc)
        return None, f"LLM unavailable ({exc})"


class VerifierAgent:
    """Four-tier verifier: Integrity -> Capacity -> Cross-Module -> Explainability."""

    # ==================================================================
    # L1: Data Integrity (merged R1-R5)
    # ==================================================================
    def verify_integrity(self, result: dict) -> Tuple[bool, list, dict]:
        """L1: Single-pass data integrity check. Fails = reject immediately."""
        import math
        checks = {}
        warnings = []

        # Numeric sanity
        forecast = result.get("forecast", 0)
        if not isinstance(forecast, (int, float)) or forecast < 0 or (isinstance(forecast, float) and math.isnan(forecast)):
            checks["L1"] = False
            warnings.append("L1 FAILED: forecast value is invalid")
            return False, warnings, checks

        inventory = result.get("inventory", 0)
        if not isinstance(inventory, (int, float)) or inventory < 0 or (isinstance(inventory, float) and math.isnan(inventory)):
            checks["L1"] = False
            warnings.append("L1 FAILED: inventory value is invalid")
            return False, warnings, checks

        # Status field check
        status = result.get("status", "ok")
        if status not in ("ok", "capacity_limited", "degraded", "healthy", "attention_needed", "monitor", ""):
            checks["L1"] = False
            warnings.append(f"L1 FAILED: unexpected status '{status}'")
            return False, warnings, checks

        checks["L1"] = True
        return True, [], checks

    # ==================================================================
    # L2: Capacity Constraint (R8)
    # ==================================================================
    def verify_capacity(self, result: dict, use_llm: bool = True) -> Tuple[bool, list, dict]:
        """L2: Restock must not exceed capacity."""
        restock = result.get("recommended_restock", 0)
        capacity = result.get("capacity", 999999)

        if capacity == 0:
            return restock == 0, ["L2 FAILED: capacity is zero"], {"L2": restock == 0}

        ratio = restock / capacity if capacity > 0 else 0
        passed = restock <= capacity
        is_borderline = R8_BORDER_LOW <= ratio <= R8_BORDER_HIGH
        ctx = f"Restock: {restock}, Capacity: {capacity}, Ratio: {ratio:.2f}"

        if passed and not is_borderline:
            return True, [], {"L2": True}

        if is_borderline and use_llm:
            llm_passed, llm_reason = _call_deepseek_judge("L2", "Capacity check", {"ratio": ratio}, ctx)
            if llm_passed is None:
                return passed, [f"L2: {ctx} (LLM unavailable)"], {"L2": passed}
            if llm_passed:
                return True, [f"L2 PASSED (LLM): {llm_reason}"], {"L2": True}
            return False, [f"L2 FAILED (LLM): {llm_reason}"], {"L2": False}

        return passed, [f"L2 FAILED: {ctx}"] if not passed else [], {"L2": passed}

    # ==================================================================
    # L3: Cross-Module Contradiction (R9/R10/R11)
    # ==================================================================
    def verify_cross_module(
        self, fusion_result, executor_data=None, use_llm=True
    ) -> Tuple[bool, list, dict]:
        """L3: Cross-module contradiction detection."""
        data = executor_data or {}
        results = {}
        warnings = []
        all_passed = True

        checks = [
            ("L3_R9", lambda: self._check_r9_forecast_vs_staffing(fusion_result, data)),
            ("L3_R10", lambda: self._check_r10_forecast_vs_inventory(fusion_result, data)),
            ("L3_R11", lambda: self._check_r11_schedule_coverage(fusion_result, data)),
        ]

        for rid, check_fn in checks:
            try:
                passed, is_borderline, ctx = check_fn()
            except Exception as e:
                results[rid] = True
                continue

            if passed and not is_borderline:
                results[rid] = True
            elif is_borderline and use_llm:
                llm_passed, llm_reason = _call_deepseek_judge(rid, "Cross-module", {"context": ctx}, ctx)
                if llm_passed is None:
                    results[rid] = passed
                    warnings.append(f"{rid} PASSED (LLM unavailable)")
                elif llm_passed:
                    results[rid] = True
                    warnings.append(f"{rid} PASSED (LLM): {llm_reason}")
                else:
                    results[rid] = False
                    all_passed = False
                    warnings.append(f"{rid} FAILED (LLM): {llm_reason}")
            else:
                results[rid] = passed
                if not passed:
                    all_passed = False
                    warnings.append(f"{rid} FAILED: {ctx}")

        return all_passed, warnings, results

    # ==================================================================
    # L4: Explainability Audit (R12)
    # ==================================================================
    def verify_explainability(
        self, fusion_result, executor_data=None, use_llm=True
    ) -> Tuple[bool, list, dict]:
        """L4 (R12): SHAP-based causal attribution for forecast anomalies.

        Uses per-instance SHAP values from XGBoost models to attribute
        forecast changes to specific drivers (weather, holidays, promotions).
        Generates LLM causal report when anomaly is significant.
        """
        import json, os

        forecast = fusion_result.get("forecast", 0)
        product = (executor_data or {}).get("product", "croissant")

        # Get raw forecasts from executor data
        forecasts_list = (executor_data or {}).get("forecasts", [])
        if not forecasts_list:
            for key in (executor_data or {}):
                if isinstance((executor_data or {}).get(key), dict) and "forecasts" in (executor_data or {}).get(key, {}):
                    forecasts_list = (executor_data or {})[key]["forecasts"]
                    break

        if not forecasts_list or forecast <= 0:
            return True, [], {"L4_R12": "skipped: no forecast data"}

        # Filter to this product and group by date
        from collections import OrderedDict
        product_forecasts = [
            f for f in forecasts_list
            if f.get("product_name", "") == product
        ]
        by_date = OrderedDict()
        for f in product_forecasts:
            d = f.get("forecast_date", "")
            if d not in by_date:
                by_date[d] = f

        dates = list(by_date.keys())
        if len(dates) < 2:
            return True, [], {"L4_R12": "skipped: need 2+ forecast dates"}

        # dates[0] is first forecast date (today if call includes today)
        # dates[1] is tomorrow -- use that for SHAP to get correct external context
        tomorrow_fc = by_date[dates[1]] if len(dates) >= 2 else by_date[dates[0]]
        today_fc = by_date[dates[0]]
        today_val = today_fc.get("predicted_demand", forecast)
        tomorrow_date = tomorrow_fc.get("forecast_date", "")

        if today_val > 0:
            change_pct = (forecast - today_val) / today_val * 100
        else:
            change_pct = 0

        date_info = f" for {tomorrow_date}" if tomorrow_date else ""

        # ---- SHAP-based causal attribution ----
        try:
            from api.module5_agent.causal_reasoning import (
                compute_shap_attribution, generate_causal_chain,
                build_llm_causal_prompt,
            )
            # Use tomorrow (dates[1]) for SHAP to avoid today's holiday context leaking
            shap_date = tomorrow_date or (dates[1] if len(dates) > 1 else dates[0])
            attr = compute_shap_attribution(product, shap_date)
            if attr.get("error"):
                raise ValueError(attr["error"])

            shap_top = attr.get("shap_contributions", [])
            top_driver = attr.get("top_driver", "unknown")
            pred = attr.get("predicted_demand", forecast)
            base_val = attr.get("base_value", 0)
            delta = pred - base_val
            delta_dir = "higher" if delta > 0 else "lower"

            # Build structured context
            shap_summary = "; ".join(
                f"{c['effective_sign']}{c['label']}({c['abs_impact']:.1f})"
                for c in shap_top[:3]
            )
            ctx = (
                f"Forecast {forecast} vs today {today_val}{date_info} "
                f"({change_pct:+.0f}%). SHAP: {shap_summary}. "
                f"Top driver: {top_driver}."
            )

            # For significant changes (>30%), generate LLM causal report
            causal_report = ""
            if abs(change_pct) >= 15 and use_llm and shap_top:
                try:
                    prompt = build_llm_causal_prompt(attr, product, tomorrow_date or dates[0])
                    llm_passed, llm_reason = _call_deepseek_judge(
                        "L4_R12", "Forecast causal attribution",
                        {
                            "product": product,
                            "forecast": forecast,
                            "today": today_val,
                            "change_pct": f"{change_pct:+.0f}%",
                            "top_driver": top_driver,
                            "shap_summary": shap_summary,
                        },
                        prompt,
                    )
                    if llm_reason:
                        causal_report = llm_reason
                except Exception as e:
                                    logger.warning("LLM causal report generation failed: %s", e)

            # SHAP explained = PASS (always: SHAP always provides attribution)
            result_data = {
                "L4_R12": ctx,
                "L4_R12_shap": shap_top,
                "L4_R12_top_driver": top_driver,
                "L4_R12_delta": round(delta, 1),
                "L4_R12_external_context": attr.get("external_context", {}),
            }
            if causal_report:
                result_data["L4_R12_causal_report"] = causal_report
                return True, [], result_data

            return True, [], result_data

        except Exception as e:
            # SHAP failed, fall back to basic change check
            if abs(change_pct) < 15:
                return True, [], {"L4_R12": f"change {change_pct:.0f}% within normal range"}
            ctx = (
                f"Forecast {forecast} vs today {today_val}{date_info} "
                f"({change_pct:+.0f}%). SHAP unavailable, flagging for review."
            )
            if use_llm:
                llm_passed, llm_reason = _call_deepseek_judge(
                    "L4_R12", "Unexplained forecast anomaly",
                    {"change_pct": change_pct, "product": product}, ctx
                )
                if llm_passed is not None and not llm_passed:
                    return False, [f"L4_R12 FAILED (LLM): {llm_reason}"], {"L4_R12": ctx}
                return True, [f"L4_R12 FLAGGED: {ctx}"], {"L4_R12": ctx}
            return False, [f"L4_R12 FLAGGED: {ctx}"], {"L4_R12": ctx}

    # ------------------------------------------------------------------
    # R9: Forecast vs Staffing capacity
    # ------------------------------------------------------------------
    @staticmethod
    def _check_r9_forecast_vs_staffing(fusion_result, data):
        forecast = fusion_result.get("forecast", 0)
        schedule = data.get("schedule", [])
        if not schedule or forecast <= 0:
            return True, False, "no schedule or forecast"

        from collections import defaultdict
        bakers_per_day = defaultdict(int)
        for s in schedule:
            role = str(s.get("role", s.get("employee_role", ""))).lower()
            if "baker" in role:
                bakers_per_day[s.get("schedule_date", s.get("date", ""))] += 1

        max_bakers = max(bakers_per_day.values()) if bakers_per_day else 0
        daily_capacity = max_bakers * 8 * 15
        if daily_capacity <= 0:
            return True, False, "no bakers on schedule"

        ratio = forecast / daily_capacity
        passed = forecast <= daily_capacity
        is_borderline = 0.85 < ratio <= 1.15
        ctx = (
            f"Forecast: {forecast}u, Bakers: {max_bakers}, "
            f"Capacity: {daily_capacity}u/day, Ratio: {ratio:.0%}"
        )
        return passed, is_borderline, ctx

    @staticmethod
    def _check_r10_forecast_vs_inventory(fusion_result, data):
        forecast = fusion_result.get("forecast", 0)
        inventory = fusion_result.get("inventory", 0)
        if forecast <= 0:
            return True, False, "no forecast"

        ratio = inventory / forecast
        stockout = ratio < 0.2
        overstock = ratio > 2.0

        if stockout:
            return False, 0.1 <= ratio < 0.25, (
                f"Forecast: {forecast}, Inventory: {inventory}, "
                f"Coverage: {ratio:.0%} (STOCKOUT RISK)"
            )
        if overstock:
            return False, 1.8 < ratio <= 2.2, (
                f"Forecast: {forecast}, Inventory: {inventory}, "
                f"Coverage: {ratio:.0%} (OVERSTOCK)"
            )
        return True, False, (
            f"Forecast: {forecast}, Inventory: {inventory}, "
            f"Coverage: {ratio:.0%} (healthy)"
        )

    @staticmethod
    def _check_r11_schedule_coverage(fusion_result, data):
        schedule = data.get("schedule", [])
        if not schedule:
            return True, False, "no schedule data"

        from collections import defaultdict
        coverage = defaultdict(lambda: defaultdict(lambda: {"roles": set(), "demand": "normal"}))
        for s in schedule:
            date = s.get("schedule_date", s.get("date", ""))
            slot = s.get("time_slot", "")
            role = str(s.get("role", s.get("employee_role", "")))
            demand = str(s.get("demand_level", "normal")).lower()
            if date and slot:
                coverage[date][slot]["roles"].add(role.lower())
                # Track the highest demand level seen for this slot
                if demand == "high" or coverage[date][slot]["demand"] != "high":
                    coverage[date][slot]["demand"] = demand

        gaps = []
        for date, slots in coverage.items():
            for slot, info in slots.items():
                demand = info.get("demand", "normal")
                roles = info.get("roles", set())
                has_baker = any("baker" in r for r in roles)
                has_front = any(r in ("cashier", "barista") for r in roles)
                # Only flag gaps when demand is HIGH (normal/low means intentional)
                if demand == "high":
                    if not has_baker:
                        gaps.append(f"{date} {slot}: no baker")
                    if not has_front:
                        gaps.append(f"{date} {slot}: no cashier/barista")

        if gaps:
            return False, len(gaps) <= 2, "; ".join(gaps[:5])
        return True, False, "all slots covered (normal/low demand slots may have intentional gaps)"
