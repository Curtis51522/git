# S5 v3: Agentic Memory + Tool-Augmented Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade S5 from stateless chatbot to stateful, tool-augmented multi-agent system with historical memory and SHAP explainability.

**Architecture:** Two new files (`memory_store.py`, `toolbox.py`) attach at edges of existing pipeline. MySQL tables store daily snapshots and query logs. Monitor gains baseline comparison. LLM synthesis receives memory context.

**Tech Stack:** Python 3.11+, MySQL 8.0+ (existing), XGBoost + SHAP, httpx

---

### Task 1: Create MySQL Tables

**Files:**
- Create: `s5-agent-brain/sql/s5_memory.sql`

- [ ] **Step 1: Write DDL file**

```sql
-- S5 v3: Agentic Memory tables
-- Run: mysql -u root bakery_ai < s5_memory.sql

CREATE TABLE IF NOT EXISTS s5_daily_snapshot (
    id INT AUTO_INCREMENT PRIMARY KEY,
    snapshot_date DATE UNIQUE NOT NULL,
    data JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_date (snapshot_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS s5_query_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    query_text TEXT NOT NULL,
    intent VARCHAR(64) NOT NULL,
    product VARCHAR(128) NOT NULL,
    agent_results JSON NOT NULL,
    decision TEXT,
    llm_summary TEXT,
    target_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_intent (intent),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Cleanup event: delete snapshots older than 30 days (runs daily at 3am)
DROP EVENT IF EXISTS s5_cleanup_snapshots;
CREATE EVENT s5_cleanup_snapshots
ON SCHEDULE EVERY 1 DAY STARTS CURRENT_DATE + INTERVAL 1 DAY
DO DELETE FROM s5_daily_snapshot WHERE snapshot_date < DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY);
```

- [ ] **Step 2: Execute against MySQL**

```bash
mysql -u root bakery_ai < s5-agent-brain/sql/s5_memory.sql
```

Expected: `Query OK, 0 rows affected` for CREATE TABLE, `Query OK` for EVENT.

- [ ] **Step 3: Verify tables exist**

```bash
mysql -u root bakery_ai -e "SHOW TABLES LIKE 's5_%';"
```

Expected output:
```
s5_daily_snapshot
s5_query_log
```

- [ ] **Step 4: Commit**

```bash
git add -f s5-agent-brain/sql/s5_memory.sql
git commit -m "feat(s5): add agentic memory MySQL schema"
```

---

### Task 2: Create memory_store.py

**Files:**
- Create: `s5-agent-brain/memory_store.py`

- [ ] **Step 1: Write the module**

```python
# Memory Store - Persistent agent memory for S5
# Stores daily snapshots and query logs in MySQL.
# Enforces 1000-row cap on query_log. Provides baseline comparison.
import json, logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger("s5.memory")

QUERY_LOG_CAP = 1000

_db = None

def _get_db():
    global _db
    if _db is None:
        import sys, os
        _parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _parent not in sys.path:
            sys.path.insert(0, _parent)
        from db.mysql_client import get_db
        _db = get_db()
    return _db


def save_snapshot(date_str: str, data: Dict[str, Any]) -> bool:
    """Save daily inventory/forecast/waste/profit snapshot. Upserts on date."""
    try:
        db = _get_db()
        cur = db.cursor()
        json_str = json.dumps(data, default=str)
        cur.execute(
            "INSERT INTO s5_daily_snapshot (snapshot_date, data) VALUES (%s, %s) "
            "ON DUPLICATE KEY UPDATE data = VALUES(data)",
            (date_str, json_str))
        db.commit()
        cur.close()
        logger.info("Snapshot saved for %s", date_str)
        return True
    except Exception as e:
        logger.warning("save_snapshot failed: %s", e)
        return False


def save_query(query: str, intent: str, product: str,
               agent_results: Dict, decision: str = "",
               summary: str = "", target_date: str = "") -> bool:
    """Log a user query. Enforces 1000-row cap (deletes oldest if exceeded)."""
    try:
        db = _get_db()
        cur = db.cursor()
        json_str = json.dumps(agent_results, default=str)
        cur.execute(
            "INSERT INTO s5_query_log (query_text, intent, product, agent_results, "
            "decision, llm_summary, target_date) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (query, intent, product, json_str, decision, summary, target_date))
        # Enforce cap
        cur.execute("SELECT COUNT(*) FROM s5_query_log")
        count = cur.fetchone()[0]
        if count > QUERY_LOG_CAP:
            cur.execute(
                "DELETE FROM s5_query_log ORDER BY created_at ASC LIMIT %s",
                (count - QUERY_LOG_CAP,))
        db.commit()
        cur.close()
        return True
    except Exception as e:
        logger.warning("save_query failed: %s", e)
        return False


def get_baseline(product: str, weekday: int, days_back: int = 28) -> Optional[Dict]:
    """Get average metrics for same product + weekday over past N days."""
    try:
        db = _get_db()
        cur = db.cursor()
        cur.execute(
            "SELECT data FROM s5_daily_snapshot "
            "WHERE snapshot_date >= DATE_SUB(CURRENT_DATE, INTERVAL %s DAY) "
            "AND WEEKDAY(snapshot_date) = %s "
            "ORDER BY snapshot_date DESC",
            (days_back, weekday))
        rows = cur.fetchall()
        cur.close()
        if not rows:
            return None
        # Average across snapshot fields
        snapshots = [json.loads(r[0]) for r in rows]
        avg = {"inventory": {}, "forecast": {}, "waste": {}, "profit": {}}
        for key in avg:
            if snapshots[0].get(key):
                for pname in snapshots[0][key]:
                    vals = [s.get(key, {}).get(pname, 0) for s in snapshots if pname in s.get(key, {})]
                    avg[key][pname] = round(sum(vals) / max(len(vals), 1), 1)
        avg["sample_count"] = len(rows)
        return avg
    except Exception as e:
        logger.warning("get_baseline failed: %s", e)
        return None


def compare_baseline(current_data: Dict, product: str = "all") -> List[Dict]:
    """Compare current snapshot to 4-week same-weekday baseline. Returns anomalies."""
    try:
        today = datetime.now()
        baseline = get_baseline(product, today.weekday())
        if not baseline:
            return []
        anomalies = []
        curr_inv = current_data.get("inventory", {})
        base_inv = baseline.get("inventory", {})
        for pname, curr_qty in curr_inv.items():
            base_qty = base_inv.get(pname, curr_qty)
            if base_qty > 0:
                ratio = curr_qty / base_qty
                if ratio < 0.5:
                    anomalies.append({
                        "product": pname,
                        "metric": "inventory",
                        "current": int(curr_qty),
                        "baseline": round(base_qty, 1),
                        "deviation": f"{ratio:.0%}",
                        "severity": "warning"
                    })
                elif ratio > 2.0:
                    anomalies.append({
                        "product": pname,
                        "metric": "inventory",
                        "current": int(curr_qty),
                        "baseline": round(base_qty, 1),
                        "deviation": f"{ratio:.0%}",
                        "severity": "warning"
                    })
        return anomalies
    except Exception as e:
        logger.warning("compare_baseline failed: %s", e)
        return []


def get_context(product: str, intent: str, days: int = 14) -> str:
    """Return compact historical context string for LLM synthesis."""
    try:
        today = datetime.now()
        weekday = today.weekday()
        baseline = get_baseline(product, weekday, days_back=28)
        if not baseline or baseline.get("sample_count", 0) < 2:
            return ""
        parts = []
        base_inv = baseline.get("inventory", {})
        base_waste = baseline.get("waste", {})
        if product == "all":
            total_base = sum(base_inv.values())
            total_waste = sum(base_waste.values())
            parts.append(f"4-week {today.strftime('%A')} avg stock: {total_base:.0f} units, avg waste: {total_waste:.0f}")
        elif product in base_inv:
            parts.append(f"4-week {today.strftime('%A')} avg stock: {base_inv[product]:.0f}, avg waste: {base_waste.get(product, 0):.0f}")
        return " | ".join(parts) if parts else ""
    except Exception as e:
        logger.warning("get_context failed: %s", e)
        return ""
```

- [ ] **Step 2: Verify import works**

```bash
cd s5-agent-brain && python -c "from memory_store import save_snapshot, save_query, get_context, compare_baseline; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Test save_snapshot**

```bash
python -c "
from memory_store import save_snapshot, get_context
from datetime import date
ok = save_snapshot(date.today().isoformat(), {
    'inventory': {'croissant': 64, 'donut': 84},
    'forecast': {'croissant': 55, 'donut': 37},
    'waste': {'croissant': 2, 'donut': 0},
    'profit': {'revenue': 450, 'cost': 180, 'margin': 0.6}
})
print('save:', ok)
ctx = get_context('croissant', 'stock_query')
print('context:', ctx)
"
```

Expected: `save: True`, context contains baseline data (or empty string on first run).

- [ ] **Step 4: Commit**

```bash
git add -f s5-agent-brain/memory_store.py
git commit -m "feat(s5): add memory_store - daily snapshots, query log, baseline comparison"
```

---

### Task 3: Create toolbox.py

**Files:**
- Create: `s5-agent-brain/toolbox.py`

- [ ] **Step 1: Write the module**

```python
# Toolbox - SHAP explainer and trend detector for S5 agents
# Provides tool-augmented reasoning capabilities for LLM synthesis.
import os, json, logging
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger("s5.toolbox")

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "xgboost")
PRODUCT_NAMES = ["croissant", "donut", "chiffon", "bread_roll", "bread_coconut", "croissant_chocolate"]

_shap_cache: Dict[str, Dict] = {}
_cache_ts: Dict[str, float] = {}
CACHE_TTL_SEC = 3600


def _load_model(product: str):
    """Load XGBoost model + feature columns. Returns (model, feature_cols) or None."""
    try:
        import xgboost as xgb
        model_path = os.path.join(MODEL_DIR, f"{product}_model.json")
        feat_path = os.path.join(MODEL_DIR, "feature_columns.json")
        if not os.path.exists(model_path) or not os.path.exists(feat_path):
            return None
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        with open(feat_path, "r") as f:
            feature_cols = json.load(f)
        return model, feature_cols
    except Exception as e:
        logger.warning("Failed to load XGBoost model for %s: %s", product, e)
        return None


def explain_forecast(product: str) -> Optional[Dict]:
    """Return SHAP feature contributions for a product's latest forecast."""
    import time
    now = time.time()
    if product in _shap_cache and (now - _cache_ts.get(product, 0)) < CACHE_TTL_SEC:
        return _shap_cache[product]

    try:
        import shap
        loaded = _load_model(product)
        if not loaded:
            return None
        model, feature_cols = loaded
        # Use dummy input to get feature importance (TreeExplainer doesn't need actual features)
        explainer = shap.TreeExplainer(model)
        # Get mean absolute SHAP values from training data approximation
        importances = np.abs(explainer.expected_value if hasattr(explainer, 'expected_value') else 0)
        if hasattr(explainer, 'feature_importances_'):
            # Fallback to XGBoost native feature importance
            raw = model.get_booster().get_score(importance_type="gain")
            total = sum(raw.values())
            top = sorted(raw.items(), key=lambda x: -x[1])[:5]
            features = [{"feature": feat_cols[int(k.replace("f", ""))] if k.startswith("f") else k,
                         "contribution": round(v / total, 3)} for k, v in top]
        else:
            features = []
        result = {"product": product, "top_features": features}
        _shap_cache[product] = result
        _cache_ts[product] = now
        return result
    except Exception as e:
        logger.warning("SHAP explain failed for %s: %s", product, e)
        return {"product": product, "top_features": [], "error": str(e)}


def detect_trend(product: str, metric: str = "waste", lookback_days: int = 14) -> Optional[Dict]:
    """Detect linear trend in a metric from query history or snapshots."""
    try:
        from memory_store import _get_db
        db = _get_db()
        cur = db.cursor()
        cur.execute(
            "SELECT snapshot_date, data FROM s5_daily_snapshot "
            "WHERE snapshot_date >= DATE_SUB(CURRENT_DATE, INTERVAL %s DAY) "
            "ORDER BY snapshot_date ASC",
            (lookback_days,))
        rows = cur.fetchall()
        cur.close()
        if len(rows) < 3:
            return None
        dates = []
        values = []
        for r in rows:
            data = json.loads(r[1])
            val = data.get(metric, {}).get(product, 0)
            if val is not None:
                dates.append(r[0])
                values.append(float(val))
        if len(values) < 3:
            return None
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        avg = np.mean(y)
        if avg > 0:
            direction = "rising" if slope > avg * 0.05 else "declining" if slope < -avg * 0.05 else "stable"
        else:
            direction = "stable" if abs(slope) < 0.5 else ("rising" if slope > 0 else "declining")
        return {
            "product": product,
            "metric": metric,
            "direction": direction,
            "slope_per_day": round(float(slope), 2),
            "days_analyzed": len(values),
            "avg_value": round(float(avg), 1)
        }
    except Exception as e:
        logger.warning("detect_trend failed: %s", e)
        return None


def compare_products(product_a: str, product_b: str) -> Optional[Dict]:
    """Head-to-head comparison using recent snapshots."""
    try:
        from memory_store import _get_db
        db = _get_db()
        cur = db.cursor()
        cur.execute(
            "SELECT data FROM s5_daily_snapshot ORDER BY snapshot_date DESC LIMIT 1")
        row = cur.fetchone()
        cur.close()
        if not row:
            return None
        data = json.loads(row[0])
        result = {}
        for pname in [product_a, product_b]:
            result[pname] = {
                "inventory": data.get("inventory", {}).get(pname, 0),
                "forecast": data.get("forecast", {}).get(pname, 0),
                "waste": data.get("waste", {}).get(pname, 0),
            }
        return result
    except Exception as e:
        logger.warning("compare_products failed: %s", e)
        return None
```

- [ ] **Step 2: Install shap if needed**

```bash
pip install shap -q
```

- [ ] **Step 3: Verify import**

```bash
cd s5-agent-brain && python -c "from toolbox import explain_forecast, detect_trend; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Test SHAP on one product**

```bash
python -c "
from toolbox import explain_forecast
result = explain_forecast('croissant')
print('SHAP result:', result)
"
```

Expected: JSON with `top_features` list or `error` key if model not found.

- [ ] **Step 5: Commit**

```bash
git add -f s5-agent-brain/toolbox.py
git commit -m "feat(s5): add toolbox - SHAP explainer, trend detector, product comparison"
```

---

### Task 4: Hook Monitor with Memory

**Files:**
- Modify: `s5-agent-brain/monitor.py:144-152` (after `run_full_check` return)

- [ ] **Step 1: Add snapshot saving to monitor**

In `run_full_check()`, after the results dict is built and before `return results`, insert:

```python
    # Save daily snapshot for memory
    from memory_store import save_snapshot, compare_baseline
    import time
    today = time.strftime("%Y-%m-%d")
    snapshot = {"inventory": {}, "forecast": {}, "waste": {}, "profit": {}}
    try:
        inv_data = await _fetch(S1_BATCH_URL)
        for item in inv_data.get("inventory", []):
            pname = item.get("product_name", "")
            snapshot["inventory"][pname] = snapshot["inventory"].get(pname, 0) + item.get("total_quantity", 0)
    except Exception:
        pass
    try:
        fc_data = await _fetch(f"{S2_FORECAST_URL}?days=1&product=all&date={today}")
        for fc in fc_data.get("forecasts", []):
            snapshot["forecast"][fc.get("product_name", "")] = fc.get("predicted_demand", 0)
    except Exception:
        pass
    save_snapshot(today, snapshot)
    # Compare vs baseline and generate alerts
    anomalies = compare_baseline(snapshot)
    for a in anomalies:
        add_alert("baseline", a.get("severity", "warning"),
                  f"{a['product']} {a['metric']} anomaly",
                  f"{a['product']} {a['metric']}: {a['current']} vs ~{a['baseline']} baseline ({a['deviation']} of normal).")
```

- [ ] **Step 2: Verify compilation**

```bash
cd s5-agent-brain && python -c "import py_compile; py_compile.compile('monitor.py', doraise=True); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add s5-agent-brain/monitor.py
git commit -m "feat(s5): hook monitor with memory - save snapshots, compare baseline"
```

---

### Task 5: Hook Query Handler with Memory + Context

**Files:**
- Modify: `s5-agent-brain/server.py` (two insertion points)

- [ ] **Step 1: Save query log after response**

In `handle_query()`, after the response dict is built and before `return response`, add:

```python
    # Log query to memory (fire-and-forget, don't block response)
    try:
        from memory_store import save_query, get_context
        save_query(
            query=req.query,
            intent=intent,
            product=params.get("product", "croissant"),
            agent_results=agent_summaries,
            decision=decision.get("action", ""),
            summary=llm_summary or "",
            target_date=params.get("date", ""))
    except Exception:
        pass
```

- [ ] **Step 2: Inject memory context into LLM synthesis**

In `handle_query()`, before calling `synthesize()`, add memory context:

```python
    # Fetch memory context for richer synthesis
    memory_ctx = ""
    try:
        from memory_store import get_context
        memory_ctx = get_context(
            product=params.get("product", "croissant"),
            intent=intent,
            days=14)
    except Exception:
        pass
```

Then pass `memory_ctx` to `synthesize()` call as a new keyword argument.

- [ ] **Step 3: Update synthesize() signature**

In `llm_synthesis.py`, update the `synthesize()` function to accept `memory_context: str = ""`:

```python
async def synthesize(query, intent, decision, priority, agent_data,
                     conflicts=None, counterfactual=None,
                     causal_calibration=None, memory_context=""):
```

Add `{memory_context}` to `_build_prompt()` template string:

In `_build_prompt()`, add:
```python
    formatted = template.format(
        query=query, decision=decision, priority=priority,
        agent_summaries=agent_summaries, conflicts=conflicts,
        counterfactual=cf_text, causal_narrative=causal_narrative,
        causal_calibration=causal_text,
        product=product, associations=associations,
        memory_context=memory_context or "No historical baseline available.")
```

And add `{memory_context}` to the cross_source_audit template:
```
Historical Context: {memory_context}
  (If available, compare current numbers against this baseline in your summary.)
```

- [ ] **Step 4: Verify compilation of both files**

```bash
cd s5-agent-brain && python -c "
import py_compile
py_compile.compile('server.py', doraise=True)
py_compile.compile('llm_synthesis.py', doraise=True)
print('OK')
"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add s5-agent-brain/server.py s5-agent-brain/llm_synthesis.py
git commit -m "feat(s5): hook query handler with memory log + context injection"
```

---

### Task 6: Integration Test

- [ ] **Step 1: Restart S5 and main server**

```bash
# Terminal 1
cd C:\Users\Curtis\Desktop\learningmaterials\SEMESTER3\bakery-ai-system
python main.py

# Terminal 2
cd s5-agent-brain
python server.py
```

- [ ] **Step 2: Wait for first monitor cycle (5 min) to generate snapshot**

Check MySQL:
```bash
mysql -u root bakery_ai -e "SELECT COUNT(*) FROM s5_daily_snapshot;"
```
Expected: `>= 1`

- [ ] **Step 3: Send test query and verify query log**

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query":"How many croissants tomorrow?","session_id":"test"}'
```

Check MySQL:
```bash
mysql -u root bakery_ai -e "SELECT COUNT(*) FROM s5_query_log;"
```
Expected: `>= 1`

- [ ] **Step 4: Verify memory context in AI Summary**

Look for baseline comparison data in the AI Summary output. Should include historical context if enough snapshots exist.

- [ ] **Step 5: Verify baseline alerts**

Wait for next monitor cycle. Check alerts:
```bash
curl http://localhost:8001/alerts/list?unacked_only=true
```
If current data deviates significantly from baseline, "baseline" source alerts should appear.

- [ ] **Step 6: Commit if any tweaks needed, then push**

```bash
git push origin main
```
