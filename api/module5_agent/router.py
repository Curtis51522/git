"""
S5 Router -- FastAPI endpoints for the Store Manager Intelligence Engine.

Query flow (POST /s5/query):
1. Intent classification (DistilBERT, < 50 ms, zero tokens)
2. DAG generation + deterministic validation (Planner)
   - On validation failure: retry (max 2) or fallback to canned DAG
3. Tool-call execution against S1/S2/S3 endpoints
4. Fusion (deterministic business logic, zero tokens)
5. Verifier: R1-R5 fast path first; then R6-R8 cross-table
   - R1-R5 failure -> immediate reject
   - R6-R8 failure -> self-reflection retry (max 2), then degrade
6. Composer: natural-language summary from structured data

Script flow (POST /s5/script):
- Lightweight path: Composer only, P50 < 2.0 s
"""

import json
import time
import logging
from fastapi import APIRouter, HTTPException

from api.module5_agent.intent import IntentClassifier
from api.module5_agent.planner import PlannerAgent, DAGValidationError
from api.module5_agent.fusion import FusionModule
from api.module5_agent.verifier import VerifierAgent
from api.module5_agent.composer import ComposerAgent
from api.module5_agent.memory import get_memory
from config.settings import PRODUCTION_CAPACITY

logger = logging.getLogger("s5.router")

router = APIRouter(prefix="/s5", tags=["Module 5 - Agent Engine"])

# -- Singleton agents ----------------------------------------------------
intent_clf = IntentClassifier()
planner    = PlannerAgent(use_mock=False)
fusion     = FusionModule()
verifier   = VerifierAgent()
composer   = ComposerAgent(use_mock=False)

# -- Constants -----------------------------------------------------------
MAX_DAG_RETRIES  = 2
MAX_AUDIT_RETRIES = 2


# ======================================================================
# POST /s5/query -- Full decision pipeline
# ======================================================================
@router.post("/query")
async def handle_query(payload: dict):
    t_start = time.perf_counter()
    query  = payload.get("query", "")
    params = payload.get("params", {})

    if not query:
        raise HTTPException(400, "Query required")

    # ---- Step 0: Memory context retrieval (deferred to after intent) ----
    session_id = payload.get("session_id", "default")
    memory = get_memory()

    # Product name list (used by multiple blocks below)
    PRODUCT_NAMES = ["croissant", "croissant_chocolate", "donut", "chiffon",
                     "bread_roll", "bread_coconut", "bread roll", "bread coconut",
                     "chocolate croissant"]

    # ---- Step 1: Intent classification (on RAW query, not memory-enriched) -
    intent, confidence = intent_clf.classify(query)
    if intent == "out_of_scope" and confidence >= 0.75:
        # Only fallback to memory if query looks like a follow-up
        ql = query.lower()
        is_followup = len(query.split()) <= 4 or any(
            pn.replace("_", " ") in ql or pn in ql for pn in PRODUCT_NAMES
        ) or any(kw in ql for kw in ["and", "next", "also", "too", "what about"])
        
        if is_followup:
            product_hint = params.get("product", "")
            if not product_hint:
                product_hint = memory.get_recent_context(session_id, n=1)
                for pn in PRODUCT_NAMES:
                    pn_display = pn.replace("_", " ")
                    if pn_display in product_hint.lower() or pn in product_hint.lower():
                        product_hint = pn.replace(" ", "_")
                        break
            if product_hint:
                params["product"] = product_hint
                intent = "stock_query"
                confidence = 0.6
                # Skip the return, continue to DAG
            else:
                return {
                    "_elapsed_ms": round((time.perf_counter() - t_start) * 1000, 1),
                    "status": "out_of_scope",
                    "intent": intent,
                    "confidence": round(confidence, 2),
                    "summary": (
                        "I can help with stock preparation, waste analysis, "
                        "promotion evaluation, and schedule audits. "
                        "Try: 'How many Croissants for tomorrow?'"
                    ),
                }
        else:
            return {
                "_elapsed_ms": round((time.perf_counter() - t_start) * 1000, 1),
                "status": "out_of_scope",
                "intent": intent,
                "confidence": round(confidence, 2),
                "summary": (
                    "I can help with stock preparation, waste analysis, "
                    "promotion evaluation, and schedule audits. "
                    "Try: 'How many Croissants for tomorrow?'"
                ),
            }
    if confidence < 0.5:
        # Fallback: check memory context for product hint (only for follow-up-like queries)
        fallback_product = ""
        if intent == "out_of_scope":
            ql = query.lower()
            # Only fallback if query looks like a follow-up (short, or has product keywords)
            is_followup = len(query.split()) <= 4 or any(
                pn.replace("_", " ") in ql or pn in ql for pn in PRODUCT_NAMES
            ) or any(kw in ql for kw in ["and", "next", "also", "too", "what about"])
            if is_followup:
                mem_ctx = memory.get_recent_context(session_id, n=1)
                for pn in PRODUCT_NAMES:
                    pn_display = pn.replace("_", " ")
                    if pn_display in mem_ctx.lower() or pn in mem_ctx.lower():
                        fallback_product = pn.replace(" ", "_")
                        break
        if fallback_product:
            params["product"] = fallback_product
            intent = "stock_query"
            confidence = 0.6
        else:
            return {
                "_elapsed_ms": round((time.perf_counter() - t_start) * 1000, 1),
                "status": "out_of_scope",
                "intent": intent,
                "confidence": round(confidence, 2),
                "summary": "I could not understand that query. Try asking about stock, waste, promotions, or schedules.",
            }

    # Auto-extract product name from query if not already in params

    query_lower = query.lower()
    
    # Detect multi-product comparison queries
    comparison_kw = [" or ", " vs ", " compare ", " versus ", " between "]
    query_padded = " " + query_lower + " "
    is_multi_product = any(kw in query_padded for kw in comparison_kw)
    multi_products = []
    if is_multi_product:
        # Detect "all/every" keyword -> expand to all canonical products
        all_kw = [" all ", " every ", " each "]
        has_all = any(kw in query_padded for kw in all_kw)
        if has_all:
            multi_products = ["croissant", "croissant_chocolate", "donut", "chiffon", "bread_roll", "bread_coconut"]
        else:
            for pn in PRODUCT_NAMES:
                pn_display = pn.replace("_", " ")
                if pn_display in query_lower or pn in query_lower:
                    canonical = pn.replace(" ", "_")
                    if canonical not in multi_products:
                        multi_products.append(canonical)
    
    if not params.get("product"):
        for pn in PRODUCT_NAMES:
            if pn.replace("_", " ") in query_lower or pn in query_lower:
                params["product"] = pn.replace(" ", "_")
                break

    # Auto-extract date from query (e.g. "16th June", "June 16", "2026-06-16")
    # Also resolve relative dates: today, tomorrow, yesterday
    if not params.get("date"):
        from datetime import datetime as _dt, timedelta as _td
        ql = query.lower()
        today = _dt.now()
        if "lusa" in ql or "next day" in ql or "day after" in ql:
            params["date"] = (today + _td(days=2)).strftime("%Y-%m-%d")
        elif "esok" in ql or "tomorrow" in ql:
            params["date"] = (today + _td(days=1)).strftime("%Y-%m-%d")
        elif "hari ini" in ql or "today" in ql:
            params["date"] = today.strftime("%Y-%m-%d")
        elif "semalam" in ql or "yesterday" in ql:
            params["date"] = (today - _td(days=1)).strftime("%Y-%m-%d")
    
    # Auto-extract date from query (e.g. "16th June", "June 16", "2026-06-16")
    if not params.get("date"):
        import re as _re
        from datetime import datetime as _dt
        # Match patterns like "16th June", "June 16", "16 June", "2026-06-16"
        patterns = [
            _re.search(r'(\d{1,2})(?:st|nd|rd|th)?\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s*(\d{4})?', query, _re.IGNORECASE),
            _re.search(r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?\s*(\d{4})?', query, _re.IGNORECASE),
            _re.search(r'(\d{4})-(\d{2})-(\d{2})', query),
        ]
        MONTH_MAP = {m.lower()[:3]: i for i, m in enumerate(["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], 1)}
        for m in patterns:
            if m:
                grp = m.groups()
                try:
                    if len(grp) >= 3 and grp[2] and grp[2].isdigit() and len(grp[2]) == 4:
                        # ISO format: 2026-06-16
                        params["date"] = _dt(int(grp[0]), int(grp[1]), int(grp[2])).strftime("%Y-%m-%d")
                    elif grp[0].isdigit():
                        # "16th June" or "16 June 2026"
                        day = int(grp[0])
                        month_str = grp[1][:3].lower()
                        month = MONTH_MAP.get(month_str, 6)
                        year = int(grp[2]) if len(grp) >= 3 and grp[2] and grp[2].isdigit() else _dt.now().year
                        params["date"] = _dt(year, month, day).strftime("%Y-%m-%d")
                    else:
                        # "June 16th" or "June 16 2026"
                        month_str = grp[0][:3].lower()
                        month = MONTH_MAP.get(month_str, 6)
                        day = int(grp[1])
                        year = int(grp[2]) if len(grp) >= 3 and grp[2] and grp[2].isdigit() else _dt.now().year
                        params["date"] = _dt(year, month, day).strftime("%Y-%m-%d")
                except (ValueError, IndexError):
                                    pass  # date parsing fallthrough, try next format
                if params.get("date"):
                    break

    # ---- Step 2: Planner DAG generation + validation -------------------
    dag = None
    dag_errors = []
    for attempt in range(1, MAX_DAG_RETRIES + 2):  # 1 initial + N retries
        try:
            dag = planner.plan(intent, params, confidence)
            break
        except DAGValidationError as exc:
            dag_errors.append(str(exc))
            logger.warning("DAG validation attempt %d failed: %s", attempt, exc)
            if attempt > MAX_DAG_RETRIES:
                # All retries exhausted -- degrade to canned DAG
                logger.error("DAG validation exhausted, falling back to canned DAG")
                dag = _canned_dag(intent)
                dag_errors.append("Falling back to canned DAG.")

    # ---- Step 3: Tool-call execution (real S1/S2/S3 endpoint calls) --------
    data = await _execute_dag(dag, params)

    # ---- Step 4: Fusion (dispatch by intent) --------------------------
    if intent == "stock_query" and len(multi_products) >= 2:
        # Multi-product comparison: directly fetch all forecasts (bypass executor)
        import httpx
        all_forecasts = []
        try:
            async with httpx.AsyncClient() as fc_client:
                fc_resp = await fc_client.get(
                    "http://localhost:8000/s2/forecast",
                    params={"days": 7},
                    timeout=httpx.Timeout(30.0)
                )
                if fc_resp.status_code == 200:
                    fc_data = fc_resp.json()
                    all_forecasts = fc_data.get("forecasts", fc_data.get("forecast", []))
                    logger.info("Multi-product refetch: got %d forecast entries", len(all_forecasts))
                else:
                    logger.warning("Multi-product refetch HTTP %d, falling back to initial data", fc_resp.status_code)
        except Exception as e:
            logger.warning("Multi-product refetch failed: %s, falling back to initial data", e)
        if not all_forecasts:
            all_forecasts = data.get("_all_forecasts", [])
            logger.info("Multi-product: using initial _all_forecasts (%d entries)", len(all_forecasts))
        all_inventory = data.get("_all_inventory", [])
        capacity = data.get("capacity", PRODUCTION_CAPACITY)
        
        # Determine target date: tomorrow, always skip Monday
        from datetime import datetime as dt, timedelta as td
        target_date = params.get("date", "")
        if target_date:
            try:
                td_date = dt.strptime(target_date[:10], "%Y-%m-%d")
                if td_date.weekday() == 0:
                    td_date += td(days=1)
                    target_date = td_date.strftime("%Y-%m-%d")
            except ValueError:
                pass
        else:
            tm = dt.now() + td(days=1)
            if tm.weekday() == 0:
                tm += td(days=1)
            target_date = tm.strftime("%Y-%m-%d")
        products_data = []
        for p in multi_products:
            p_forecast = 0
            for f in all_forecasts:
                fd = f.get("forecast_date", "")
                if isinstance(fd, str) and len(fd) > 10:
                    fd = fd[:10]
                if f.get("product_name") == p and fd == target_date:
                    p_forecast += f.get("predicted_demand", 0)
            if p_forecast == 0:
                # Fallback: any date for this product
                for f in all_forecasts:
                    if f.get("product_name") == p:
                        p_forecast += f.get("predicted_demand", 0)
            
            p_inventory = sum(
                b.get("quantity", 0) for b in all_inventory
                if b.get("product_name", "") == p
            )
            p_restock = max(0, p_forecast - p_inventory)
            products_data.append({
                "name": p,
                "forecast": p_forecast,
                "inventory": p_inventory,
                "restock": p_restock,
                "capacity": capacity,
            })
        
        result = {
            "multi_product": True,
            "products": products_data,
            "forecast": sum(pd["forecast"] for pd in products_data),
            "inventory": sum(pd["inventory"] for pd in products_data),
            "capacity": capacity,
        }
    elif intent == "stock_query":
        # For follow-up queries (e.g. "next day"), chain inventory from memory
        carryover = 0
        effective_inventory = data.get("inventory", 0)
        if confidence < 0.7 and params.get("date"):
            prev_eps = memory.retrieve_episodes(session_id, product=params.get("product", ""), limit=1)
            if prev_eps:
                prev_snap = prev_eps[0].get("data_snapshot", {})
                if isinstance(prev_snap, str):
                    try: import json; prev_snap = json.loads(prev_snap)
                    except: prev_snap = {}
                prev_high = prev_snap.get("forecast_high", prev_snap.get("forecast", 0))
                prev_low = prev_snap.get("forecast_low", 0)
                # Q1 max leftover = upper - lower (best case carryover to Q2)
                if prev_high and prev_low:
                    carryover = max(0, int(prev_high - prev_low))
                    # Carryover REPLACES inventory for Q2 (Q1 already consumed it)
                    effective_inventory = carryover
        
        result = fusion.compute_restock(
            data.get("forecast", 0),
            effective_inventory,
            data.get("capacity", 0),
            user_target=params.get("user_target"),
            query_type=params.get("query_type", "forecast_query"),
            forecast_low=data.get("forecast_low"),
            forecast_high=data.get("forecast_high"),
            carryover=carryover,
        )
    elif intent == "waste_analysis":
        result = fusion.compute_waste(
            data.get("predictions", []),
            data.get("actuals", []),
        )
    elif intent == "promo_eval":
        result = fusion.compute_promo_roi(
            data.get("incremental_revenue", 0),
            data.get("discount_cost", 0),
        )
    elif intent == "schedule_audit":
        result = fusion.compute_schedule_audit(
            data.get("schedule", []),
            data.get("transactions", []),
        )
    elif intent == "profit_analysis":
        # Directly query DB for transactions and product costs
        from db.mysql_client import get_db
        db = get_db()
        # Fetch all outflow transactions (sales)
        txn_cursor = db.cursor(dictionary=True)
        txn_cursor.execute(
            "SELECT product_name, quantity, unit_price, transaction_time "
            "FROM inventory_transactions WHERE transaction_type = %s "
            "ORDER BY transaction_time DESC",
            ("outflow",)
        )
        transactions = txn_cursor.fetchall()
        # Fetch product cost prices
        prod_cursor = db.cursor(dictionary=True)
        prod_cursor.execute("SELECT product_name, selling_price, cost_price FROM products")
        products_rows = prod_cursor.fetchall()
        products_map = {r["product_name"]: r for r in products_rows}
        result = fusion.compute_profit(transactions, products_map)
    elif intent == "cross_source_audit":
        result = fusion.compute_cross_audit(
            data.get("forecast", 0),
            data.get("inventory", 0),
            data.get("capacity", 0),
            data.get("schedule", []),
            data.get("transactions", []),
        )
    else:
        # Fallback: treat as stock query
        result = fusion.compute_restock(
            data.get("forecast", 0),
            data.get("inventory", 0),
            data.get("capacity", 0),
        )

    # ---- Step 5: Verifier (4-tier: L1 -> L2 -> L3 -> L4) ------------
    # L1: Data Integrity (fail = immediate reject)
    l1_ok, l1_warnings, l1_results = verifier.verify_integrity(result)
    if not l1_ok:
        return {
            "_elapsed_ms": round((time.perf_counter() - t_start) * 1000, 1),
            "status": "rejected",
            "reason": "L1 data-integrity check failed",
            "audit": {"rule_results": l1_results, "audit_warnings": l1_warnings},
        }

    # L2: Capacity Constraint (fail = warn, continue)
    l2_ok, l2_warnings, l2_results = verifier.verify_capacity(result, use_llm=True)
    audit_results = {**l1_results, **l2_results}
    audit_warnings = l1_warnings + l2_warnings
    audit_passed = l2_ok

    # L3: Cross-Module Contradiction (R9/R10/R11)
    l3_ok, l3_warnings, l3_results = verifier.verify_cross_module(
        result, executor_data=data, use_llm=True
    )
    audit_results = {**audit_results, **l3_results}
    audit_warnings = audit_warnings + l3_warnings
    if not l3_ok:
        audit_passed = False
        logger.warning("L3 cross-module audit found contradictions: %s", l3_warnings)

    # L4: Explainability Audit (R12)
    l4_ok, l4_warnings, l4_results = verifier.verify_explainability(
        result, executor_data=data, use_llm=True
    )
    audit_results = {**audit_results, **l4_results}
    audit_warnings = audit_warnings + l4_warnings
    if not l4_ok:
        audit_passed = False
        logger.warning("L4 explainability audit flagged: %s", l4_warnings)

    # ---- Step 6: Composer -----------------------------------------------
    # R9-R12 may add warnings even when R6-R8 passes
    cross_module_ok = r912_ok if 'r912_ok' in dir() else True
    final_passed = audit_passed and cross_module_ok
    all_warnings = audit_warnings if audit_warnings else []

    # Build memory-enriched query for LLM context
    product_hint = params.get("product", "")
    memory_context = memory.get_recent_context(session_id, n=3, product=product_hint)
    enriched_query = query
    if memory_context:
        enriched_query = memory_context + "Current query: " + query

    # Inject R12 causal report into result for composer
    r12_causal = audit_results.get("L4_R12_causal_report", "")
    r12_shap = audit_results.get("L4_R12_shap", [])
    r12_top = audit_results.get("L4_R12_top_driver", "")

    if intent == "profit_analysis":
        by_prod = result.get("by_product", [])
        # Check if user is asking about a specific product
        target = params.get("product", "")
        if not target:
            # Try to extract product name from query
            ql = query.lower()
            for p in ["croissant_chocolate", "bread_coconut", "bread_roll", "croissant", "chiffon", "donut"]:
                if p.replace("_", " ") in ql or p in ql:
                    target = p
                    break
        if target and by_prod:
            # Show specific product profit
            match = None
            for p in by_prod:
                if p.get("product_name") == target:
                    match = p
                    break
            if match:
                summary = (
                    f"{target.replace('_',' ').title()}: revenue RM {match['revenue']:.2f}, "
                    f"cost RM {match['cost']:.2f}, profit RM {match['profit']:.2f}, "
                    f"margin {match['margin_pct']}%. "
                    f"({match['quantity_sold']} units sold across {match['transactions']} transactions)"
                )
            else:
                summary = f"No sales data found for {target}."
        else:
            # Full profit overview
            rev = result.get("total_revenue", 0)
            cost = result.get("total_cost", 0)
            profit = result.get("gross_profit", 0)
            margin = result.get("margin_pct", 0)
            top = by_prod[0]["product_name"] if by_prod else "N/A"
            top_profit = by_prod[0]["profit"] if by_prod else 0
            summary = (
                f"Total revenue: RM {rev:.2f}, cost: RM {cost:.2f}. "
                f"Gross profit: RM {profit:.2f} (margin {margin}%). "
                f"Top product: {top} (profit RM {top_profit:.2f}). "
                f"Based on {result.get('transaction_count', 0)} sales transactions."
            )
    else:
        summary = composer.compose_summary({
            **result,
            "audit_warnings": all_warnings,
        "multi_products": multi_products if len(multi_products) >= 2 else [],
        "target_product": product_hint,
        "target_date": params.get("date", ""),
        "L4_R12_causal_report": r12_causal,
        "L4_R12_shap": r12_shap,
        "L4_R12_top_driver": r12_top,
        "L4_R12_external_context": audit_results.get("L4_R12_external_context", {}),
            }, query=enriched_query, intent=intent)


    # ---- Step 7: Store episode to memory ---------------------------------
    try:
        memory.store_episode(
            session_id=session_id,
            query=query,
            intent=intent,
            product=params.get("product", ""),
            target_date=params.get("date", ""),
            response=summary,
            data_snapshot=result,
        )
        # Trigger reflection every ~20 episodes (async, fire-and-forget)
        import random
        if random.random() < 0.05:  # ~5% chance per query
            try:
                memory.generate_reflection(session_id)
            except Exception as e:
                logger.warning("Reflection generation failed: %s", e)
    except Exception as e:
        logger.warning("Memory store failed: %s", e)

    # Determine whether to show the schedule table in frontend
    show_table = False
    if intent == "schedule_audit":
        show_keywords = ["show", "view", "display", "table", "roster", "who", "list"]
        if any(kw in query.lower() for kw in show_keywords):
            show_table = True

    return {
        "_elapsed_ms": round((time.perf_counter() - t_start) * 1000, 1),
        "status": "ok" if final_passed else "degraded",
        "intent": intent,
        "confidence": confidence,
        "result": result,
        "audit": {
            "passed": final_passed,
            "rule_results": audit_results,
            "audit_warnings": all_warnings,
        },
        "summary": summary,
        "show_table": show_table,
        "dag_validation_errors": dag_errors if dag_errors else None,
    }


# ======================================================================
# POST /s5/script -- Staff upselling script (lightweight path)
# ======================================================================
@router.post("/script")
async def handle_script(payload: dict):
    scores = payload.get("combo_scores", [])
    scripts = composer.compose_script(scores)
    return {"status": "ok", "scripts": scripts}


# ======================================================================
# Internal helpers
# ======================================================================

def _canned_dag(intent: str) -> dict:
    """Fallback DAG when Planner LLM fails validation after max retries."""
    canned = {
        "stock_query": {
            "nodes": [
                {"id": "s2_forecast",  "endpoint": "/s2/forecast"},
                {"id": "s1_inventory", "endpoint": "/s1/batch_inventory"},
                {"id": "s3_capacity",  "endpoint": "/s3/capacity", "depends_on": ["s2_forecast"]},
            ],
        },
        "waste_analysis": {
            "nodes": [
                {"id": "s1_txn",       "endpoint": "/s1/batch_inventory"},
                {"id": "s2_forecast",  "endpoint": "/s2/forecast"},
            ],
        },
        "promo_eval": {
            "nodes": [
                {"id": "s1_txn",       "endpoint": "/s1/batch_inventory"},
                {"id": "db_promo",     "endpoint": "/db/sql_query"},
            ],
        },
        "schedule_audit": {
            "nodes": [
                {"id": "s3_schedule",  "endpoint": "/s3/schedule"},
                {"id": "s1_txn",       "endpoint": "/s1/batch_inventory"},
            ],
        },
        "cross_source_audit": {
            "nodes": [
                {"id": "s2_forecast",  "endpoint": "/s2/forecast"},
                {"id": "s1_inventory", "endpoint": "/s1/batch_inventory"},
                {"id": "s3_schedule",  "endpoint": "/s3/schedule"},
                {"id": "s1_txn",       "endpoint": "/s1/batch_inventory"},
            ],
        },
    }
    return canned.get(intent, {"nodes": []})


def _self_reflect(intent: str, params: dict, result: dict, warnings: list, attempt: int) -> dict:
    """Ask Planner LLM to revise parameters based on Verifier audit warnings.

    Returns revised params dict, or empty dict if no revision possible.
    """
    try:
        from api.module5_agent.llm_client import call_deepseek

        prompt = f"""You are a Planner Agent for a bakery-cafe system. Your previous plan produced a result that FAILED audit verification.

Intent: {intent}
Original params: {json.dumps(params, default=str)}
Current result: {json.dumps(result, default=str)}
Audit warnings:
{chr(10).join(f"  - {w}" for w in warnings)}

Self-reflection attempt: {attempt}/{MAX_AUDIT_RETRIES}

What went wrong? How can you fix it?
Suggest ONE concrete parameter change (e.g., add a product filter, narrow date range, add a data source).

Return ONLY valid JSON:
{{"revised_params": {{"key": "value"}}, "reasoning": "one sentence"}}"""

        import json as _json
        system = "You are a Planner doing self-reflection. Be concise. Return only JSON."
        response = call_deepseek(prompt, system, max_tokens=200)

        if not response:
            return {}

        response = response.strip()
        if response.startswith("```"):
            parts = response.split("```")
            response = parts[1] if len(parts) > 1 else response
            if response.startswith("json"):
                response = response[4:]
        revised = _json.loads(response)
        return revised.get("revised_params", {})

    except Exception as exc:
        logger.warning("Self-reflection failed: %s", exc)
        return {}


def _recompute_fusion(intent: str, data: dict, prev_result: dict) -> dict:
    """Re-run Fusion with new data, preserving fields not in the new data."""
    try:
        if intent == "stock_query":
            return fusion.compute_restock(
                data.get("forecast", prev_result.get("forecast", 0)),
                data.get("inventory", prev_result.get("inventory", 0)),
                data.get("capacity", prev_result.get("capacity", 0)),
            )
        elif intent == "waste_analysis":
            return fusion.compute_waste(
                data.get("predictions", prev_result.get("predictions", [])),
                data.get("actuals", prev_result.get("actuals", [])),
            )
        elif intent == "promo_eval":
            return fusion.compute_promo_roi(
                data.get("incremental_revenue", prev_result.get("incremental_revenue", 0)),
                data.get("discount_cost", prev_result.get("discount_cost", 0)),
            )
        elif intent == "schedule_audit":
            return fusion.compute_schedule_audit(
                data.get("schedule", prev_result.get("schedule", [])),
                data.get("transactions", prev_result.get("transactions", [])),
            )
    except Exception:
        pass
    return prev_result



# ======================================================================
# C1 Proactive Reflection endpoints
# ======================================================================
@router.get("/reflections")
async def handle_reflections(session_id: str = "", limit: int = 10):
    """GET /s5/reflections ? retrieve proactive insights from memory reflections."""
    memory = get_memory()
    if session_id:
        reflections = memory.get_reflections(session_id, limit=limit)
    else:
        # All recent reflections
        from db.mysql_client import get_db
        db = get_db()
        cur = db.cursor(dictionary=True)
        cur.execute(
            "SELECT * FROM s5_memory_reflections ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
        reflections = cur.fetchall()
        for r in reflections:
            if r.get("created_at") and hasattr(r["created_at"], "isoformat"):
                r["created_at"] = r["created_at"].isoformat()
    return {"ok": True, "reflections": reflections}


@router.post("/reflections/run")
async def handle_run_reflections():
    """POST /s5/reflections/run ? manually trigger auto-reflection."""
    memory = get_memory()
    results = memory.auto_reflect_all()
    return {"ok": True, "generated": len(results), "results": results}


# ======================================================================
# B1 Alert endpoints
# ======================================================================
from api.module5_agent.alert_store import (
    list_alerts, acknowledge_alert, acknowledge_all, get_unacked_count,
)

@router.get("/alerts/list")
async def handle_alerts_list(since: str = "", severity: str = "", acked: str = "", limit: int = 50):
    acked_bool = None
    if acked.lower() in ("true", "1"):
        acked_bool = True
    elif acked.lower() in ("false", "0"):
        acked_bool = False

    alerts = list_alerts(
        since=since or None,
        severity=severity or None,
        acknowledged=acked_bool,
        limit=limit,
    )
    return {"ok": True, "alerts": alerts, "unacked_count": get_unacked_count()}


@router.post("/alerts/ack")
async def handle_alerts_ack(payload: dict):
    if payload.get("ack_all"):
        count = acknowledge_all()
        return {"ok": True, "acked_count": count}
    
    alert_id = payload.get("alert_id")
    if not alert_id:
        return {"ok": False, "error": "Missing alert_id"}
    
    success = acknowledge_alert(int(alert_id))
    return {"ok": success, "alert_id": alert_id}


@router.get("/alerts/count")
async def handle_alerts_count():
    return {"ok": True, "unacked_count": get_unacked_count()}



# ======================================================================
# B2 What-if Simulator endpoints
# ======================================================================
from api.module5_agent.scenario_engine import ScenarioEngine

_scenario_engine = ScenarioEngine()

@router.post("/whatif/compare")
async def handle_whatif_compare(payload: dict):
    """POST /s5/whatif/compare
    
    Body: {
      product: "croissant",
      forecast: 22, inventory: 12, capacity: 50,
      price: 3.50,
      scenario_type: "discount" | "staffing" | "production",
      adjustments: {discount_pct: 30}
    }
    Returns: Plan A + Plan B + Plan C comparison + attribution
    """
    product = payload.get("product", "croissant")
    forecast = float(payload.get("forecast", 0))
    inventory = int(payload.get("inventory", 0))
    capacity = int(payload.get("capacity", 0))
    price = float(payload.get("price", 0) or 0)
    cost_price = float(payload.get("cost_price", 0) or 0)
    scenario_type = payload.get("scenario_type", "discount")
    adjustments = payload.get("adjustments", {})
    forecast_low = payload.get("forecast_low")
    forecast_high = payload.get("forecast_high")

    if forecast <= 0:
        return {"ok": False, "error": "Forecast required"}

    report = _scenario_engine.compare(
        product=product,
        forecast=forecast,
        inventory=inventory,
        capacity=capacity,
        base_price=price if price > 0 else None,
        scenario_type=scenario_type,
        adjustments=adjustments,
        forecast_low=float(forecast_low) if forecast_low is not None else None,
        forecast_high=float(forecast_high) if forecast_high is not None else None,
    )

    def plan_to_dict(p):
        if p is None: return None
        return {
            "label": p.label,
            "production": p.production,
            "projected_sales": p.projected_sales,
            "waste": p.waste,
            "revenue": p.revenue,
            "discount_cost": p.discount_cost,
            "profit": p.profit,
            "capacity_util_pct": p.capacity_util_pct,
            "params": p.params,
        }

    return {
        "ok": True,
        "plan_a": plan_to_dict(report.plan_a),
        "plan_b": plan_to_dict(report.plan_b),
        "plan_c": plan_to_dict(report.plan_c),
        "attribution": report.attribution,
        "recommendation": report.recommendation,
    }


from api.module5_agent.executor import execute_dag_real as _execute_dag

