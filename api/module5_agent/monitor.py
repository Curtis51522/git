"""Background Monitor -- periodic proactive anomaly detection for B1.

Runs every N minutes (default: 30), calling cross_source_audit,
feeding results into AnomalyDetector, and creating alerts for
anything that looks wrong.

Designed as an asyncio background task started in main.py.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from api.module5_agent.fusion import FusionModule
from api.module5_agent.anomaly_detector import (
    AnomalyDetector, build_feature_vector, get_detector,
)
from api.module5_agent.llm_client import compose_alert_description
from api.module5_agent.memory import get_memory
from api.module5_agent.alert_store import (
    create_alert, get_unacked_count,
)
from api.module5_agent.router import _execute_dag, _self_reflect
# Will be set after WebSocket manager is created
_ws_manager: Optional[object] = None

logger = logging.getLogger("s5.monitor")

DEFAULT_INTERVAL_SEC = 30 * 60  # 30 minutes

_fusion = FusionModule()


async def _get_cross_audit_data() -> dict:
    """Run cross_source_audit DAG and return Fusion result."""
    dag = {
        "nodes": [
            {"id": "s2_forecast",  "endpoint": "/s2/forecast"},
            {"id": "s1_inventory", "endpoint": "/s1/batch_inventory"},
            {"id": "s3_schedule",  "endpoint": "/s3/schedule"},
            {"id": "s1_txn",       "endpoint": "/s1/batch_inventory"},
        ],
    }

    try:
        data = await _execute_dag(dag, {"days": 7})
    except Exception as e:
        logger.warning("Monitor DAG execution failed: %s", e)
        return {"status": "error", "issues": [], "all_clear": []}

    result = _fusion.compute_cross_audit(
        forecast=data.get("forecast", 0),
        inventory=data.get("inventory", 0),
        capacity=data.get("capacity", 0),
        schedule=data.get("schedule", []),
        transactions=data.get("transactions", []),
    )
    # Merge raw data into result for feature extraction
    result["forecast"] = data.get("forecast", 0)
    result["inventory"] = data.get("inventory", 0)
    result["capacity"] = data.get("capacity", 0)
    result["actual"] = data.get("actual", result.get("forecast", 0))
    result["waste_rate"] = data.get("waste_rate", 0)
    result["recommended_restock"] = data.get("recommended_restock", 0)
    # Pass raw per-product data for per-product anomaly detection
    result["_all_forecasts"] = data.get("_all_forecasts", [])
    result["_all_inventory"] = data.get("_all_inventory", [])
    return result


async def _investigate_root_cause(audit_result: dict) -> Optional[str]:
    """Auto-investigate by running waste_analysis or schedule_audit pipeline."""
    issues = audit_result.get("issues", [])
    if not issues:
        return None

    # Determine which pipeline to run based on the first issue
    first_rule = issues[0].get("rule", "")
    try:
        if first_rule in ("R6", "R8"):
            # Run waste_analysis
            dag = {
                "nodes": [
                    {"id": "s1_txn",      "endpoint": "/s1/batch_inventory"},
                    {"id": "s2_forecast", "endpoint": "/s2/forecast"},
                ],
            }
            data = await _execute_dag(dag, {"days": 7})
            result = _fusion.compute_waste(
                data.get("predictions", []),
                data.get("actuals", []),
            )
            if result.get("waste_flag"):
                return f"Root cause: Forecast deviation avg {result['avg_deviation']}% -- overproduction likely."
            return f"Root cause: Waste analysis complete (avg deviation {result['avg_deviation']}%)."

        elif first_rule == "R7":
            # Run schedule_audit
            dag = {
                "nodes": [
                    {"id": "s3_schedule", "endpoint": "/s3/schedule"},
                    {"id": "s1_txn",      "endpoint": "/s1/batch_inventory"},
                ],
            }
            data = await _execute_dag(dag, {"days": 7})
            result = _fusion.compute_schedule_audit(
                data.get("schedule", []),
                data.get("transactions", []),
            )
            anom = result.get("anomalies", [])
            if anom:
                return f"Root cause: {len(anom)} scheduling gaps detected -- peak hours understaffed."
            return "Root cause: Schedule audit complete -- check individual shift gaps."
    except Exception as e:
        logger.warning("Root cause investigation failed: %s", e)

    return None



def _classify_alert_rule(stock_cov: float, deviation: float, hc_gap: float, waste: float, cap_pres: float) -> str:
    """Classify the alert rule based on which feature triggered the anomaly."""
    if stock_cov < 0.2:
        return "stockout"
    if stock_cov > 2.0:
        return "overstock"
    if hc_gap >= 2:
        return "understaffed"
    if deviation > 50:
        return "forecast_deviation"
    if cap_pres > 1.5:
        return "capacity_pressure"
    if waste > 20:
        return "waste"
    if stock_cov < 0.5:
        return "low_stock"
    if stock_cov > 1.5:
        return "excess_stock"
    if deviation > 30:
        return "forecast_deviation"
    if hc_gap >= 1:
        return "understaffed"
    if cap_pres > 1.0:
        return "capacity_pressure"
    return "general"

async def _run_monitor_cycle(detector: AnomalyDetector):
    """Single monitoring cycle: audit -> detect -> alert."""
    logger.info("Monitor cycle starting...")
    
    # 1. Run cross_source_audit
    audit_result = await _get_cross_audit_data()
    if audit_result.get("status") == "error":
        logger.warning("Monitor cycle skipped -- audit returned error")
        return

    # 2. Get raw per-product data from executor
    all_forecasts = audit_result.get("_all_forecasts", [])
    all_inventory = audit_result.get("_all_inventory", [])
    
    # Extract unique products from forecasts
    products = list(set(
        f.get("product_name", "") for f in all_forecasts
        if f.get("product_name", "")
    ))
    
    if not products:
        logger.warning("Monitor: no products found in forecast data")
        return
    
    logger.info("Monitor: checking %d products...", len(products))
    
    for product in products:
        # Per-product forecast (tomorrow''s value)
        tomorrow = datetime.now() + timedelta(days=1)
        if tomorrow.weekday() == 0:
            tomorrow += timedelta(days=1)
        tomorrow_str = tomorrow.strftime("%Y-%m-%d")
        p_forecasts = [f for f in all_forecasts if f.get("product_name", "") == product and f.get("forecast_date", "") == tomorrow_str]
        p_forecast = sum(f.get("predicted_demand", 0) for f in p_forecasts)
        
        # Per-product inventory
        p_inventory = sum(
            b.get("quantity", 0) for b in all_inventory
            if b.get("product_name", "") == product
        )
        
        if p_forecast <= 0:
            logger.info("Monitor: %s tomorrow forecast=0, skipped", product)
            continue
        
        # Build per-product audit result for feature vector
        p_audit = {
            **audit_result,
            "forecast": p_forecast,
            "inventory": p_inventory,
            "actual": audit_result.get("actual", p_forecast),
            "recommended_restock": max(0, p_forecast - p_inventory),
        }
        
        # Build feature vector
        vec = build_feature_vector(p_audit)
        if vec is None:
            continue
        
        # Detect
        is_anomaly, score = detector.predict(vec)
        severity = detector.classify_severity(vec, is_anomaly, score)
        
        if not is_anomaly and severity == "info":
            continue
        
        # Build alert message via LLM
        deviation, stock_cov, hc_gap, waste, cap_pres, issue_cnt = vec
        issues = audit_result.get("issues", [])
        top_issue = compose_alert_description(
            feature_values={
                "deviation_pct": float(deviation),
                "stock_coverage": float(stock_cov),
                "headcount_gap": float(hc_gap),
                "waste_rate": float(waste),
                "capacity_pressure": float(cap_pres),
                "anomaly_score": float(score),
                "severity": severity,
            },
            issues=issues,
            audit_result=p_audit,
            product=product,
        )
        
        # Root cause for critical
        root_cause = None
        if severity == "critical":
            root_cause = await _investigate_root_cause(p_audit)
        
        # Create alert
        alert_id = create_alert(
            severity=severity,
            rule=_classify_alert_rule(stock_cov, deviation, hc_gap, waste, cap_pres),
            message=top_issue,
            params={
                "product": product,
                "anomaly_score": round(score, 3),
                "forecast": p_forecast,
                "inventory": p_inventory,
                "feature_vector": vec.tolist(),
                "issue_count": len(issues),
            },
            root_cause=root_cause,
        )
        
        logger.info("Monitor: ALERT #%d [%s] %s: %s", alert_id, severity, product, top_issue[:60])
        
        # Push via WebSocket
        if _ws_manager is not None:
            try:
                await _ws_manager.broadcast({
                    "type": "alert",
                    "alert_id": alert_id,
                    "severity": severity,
                    "product": product,
                    "message": top_issue,
                    "root_cause": root_cause,
                    "unacked_count": get_unacked_count(),
                })
            except Exception as e:
                logger.warning("WebSocket broadcast failed: %s", e)

async def start_monitor(interval_sec: int = DEFAULT_INTERVAL_SEC,
                        ws_manager: Optional[object] = None):
    """Start the background monitor loop. Called from main.py on startup."""
    global _ws_manager
    _ws_manager = ws_manager

    detector = get_detector()

    # Bootstrap: train with a few cycles of data first, or load saved model
    detector.load_or_train([])

    logger.info("B1 Monitor started (interval=%ds)", interval_sec)

    while True:
        try:
            await _run_monitor_cycle(detector)
        except Exception as e:
            logger.error("Monitor cycle crashed: %s", e, exc_info=True)

        # Check if retrain needed
        if detector.should_retrain():
            pass

        # Daily: run proactive memory reflection
        now = datetime.now()
        if now.hour == 3 and now.minute < 30:  # ~3:00-3:30 AM
            try:
                mem = get_memory()
                result = mem.auto_reflect_all()
                if result:
                    logger.info("Daily reflection: generated %d insights", len(result))
            except Exception as e:
                logger.warning("Daily reflection failed: %s", e)

        await asyncio.sleep(interval_sec)
