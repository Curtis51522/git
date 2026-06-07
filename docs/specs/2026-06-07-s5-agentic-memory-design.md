# S5 v3: Agentic Memory + Tool-Augmented Multi-Agent System

**Status**: Draft | **Date**: 2026-06-07 | **Author**: Curtis

## 1. Motivation

S5 currently operates as a stateless request-response system. Each query is independent, with no cross-query awareness, no historical comparison, and no proactive insight generation beyond static threshold alerts. This design upgrades S5 from a reactive chatbot to a stateful, tool-augmented multi-agent system.

## 2. Design Goals

| Goal | How |
|---|---|
| **Stateful memory** | Store daily snapshots + query logs in MySQL |
| **Proactive intelligence** | Monitor compares current state vs historical baseline, generating anomaly alerts |
| **Tool-augmented reasoning** | SHAP explainer + trend detector available to LLM synthesis |
| **Zero pipeline disruption** | New components attach at edges only; 6 agents, arbitrator, optimizer unchanged |

## 3. Architecture

```
User Query ¡ú DistilBERT ¡ú 6 Agents ¡ú Arbitrator+MIP ¡ú LLM Synthesis ¡ú Response
                ¡ü              ¡ü                         ¡ü
                |              |              [3] memory context injected
                |     [2] toolbox.py ¡û©¤©¤ (SHAP, trend, baseline compare)
                |
        [1] memory_store.py ¡û¡ú MySQL (s5_query_log, s5_daily_snapshot)

Monitor (5min) ¡ú run_full_check() ¡ú save_snapshot() ¡ú compare_baseline() ¡ú alerts
```

### Attachment Points (Zero Intrusion)

| # | File | Change | Lines |
|---|------|--------|-------|
| 1 | `server.py` `/query` | After response: `memory_store.save_query(...)`. Before synthesis: inject memory context | +5 |
| 2 | `monitor.py` `run_full_check()` | After checks: `memory_store.save_snapshot(...)` + `memory_store.compare_baseline(...)` | +3 |
| 3 | `llm_synthesis.py` templates | Add `{memory_context}` placeholder to cross_source_audit, waste_analysis, profit_analysis | +3 lines per template |

## 4. New Files

### 4.1 `memory_store.py`

**Responsibilities:**
- `save_snapshot(date, data)` - Insert daily inventory/forecast/waste/profit snapshot
- `save_query(query, intent, product, agent_results, decision, summary, target_date)` - Log query, enforce 1000-row cap
- `get_context(product, intent, days=14)` - Return relevant historical context for LLM synthesis
- `get_baseline(product, weekday)` - Return 4-week average for same weekday
- `compare_baseline(current_data)` - Compare vs baseline, return anomalies

**Data Schema (MySQL):**

```sql
CREATE TABLE s5_daily_snapshot (
    id INT AUTO_INCREMENT PRIMARY KEY,
    snapshot_date DATE UNIQUE,
    data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_date (snapshot_date)
);

CREATE TABLE s5_query_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    query_text TEXT,
    intent VARCHAR(64),
    product VARCHAR(128),
    agent_results JSON,
    decision TEXT,
    llm_summary TEXT,
    target_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_intent (intent),
    INDEX idx_created (created_at)
);
```

**Retention Policy:**
- `s5_query_log`: Hard cap 1000 rows. `INSERT` triggers `DELETE FROM s5_query_log ORDER BY created_at ASC LIMIT 1` when count > 1000.
- `s5_daily_snapshot`: 30-day TTL via MySQL `CREATE EVENT`.

### 4.2 `toolbox.py`

**Components:**

| Tool | Input | Output | Implementation |
|---|---|---|---|
| `explain_forecast(product)` | Product name | SHAP feature contributions (top 5) | Load XGBoost model, compute SHAP on recent window |
| `detect_trend(metric, days=14)` | Metric name + product | Trend direction + confidence | Linear regression on daily values |
| `compare_baseline(snapshot, product)` | Current snapshot + product | Deviation from 4-week same-weekday avg | Query memory_store |

**SHAP Integration:**
- Reuses existing XGBoost models at `models/xgboost/{product}_model.json`
- Cached per product per 1 hour to avoid recomputation
- Returns: `{"top_features": [{"feature": "day_of_week", "contribution": 0.35}, ...]}`

### 4.3 `sql/s5_memory.sql`

DDL file for table creation + cleanup event.

## 5. Prompt Integration

### Current LLM Synthesis Template (cross_source_audit)

```
Query: {query}
Decision: {decision}
Agents: {agent_summaries}
Conflicts: {conflicts}
```

### Upgraded Template

```
Query: {query}
Decision: {decision}
Agents: {agent_summaries}
Conflicts: {conflicts}
Historical Context: {memory_context}
  (If provided, reference this data. Format: "Last 4 Tuesdays averaged X waste vs today's Y.")
```

## 6. Example Output Upgrade

**Before (current):**
> "The bakery is critically understocked, with only 60 units on hand against a forecast of 411."

**After (v3):**
> "The bakery is critically understocked at 60 units vs forecast 411. This is 40% below your 4-week Tuesday average of 100 units, marking the third consecutive week of declining stock. SHAP analysis shows the top demand driver this week is weather (temperature +2C above seasonal)."

## 7. Non-Goals

- No change to 6 agents, arbitrator, optimizer
- No new intent (DistilBERT unchanged)
- No change to alert_store.py (existing alert infrastructure reused)
- No real-time streaming (batch is sufficient for bakery use case)

## 8. Implementation Order

| Phase | Task | Est. Time |
|---|---|---|
| 1 | `sql/s5_memory.sql` - Create tables | 30 min |
| 2 | `memory_store.py` - Read/write + cap | 1 hr |
| 3 | `toolbox.py` - SHAP + trend + baseline | 2 hr |
| 4 | `monitor.py` - Hook snapshot + comparison | 30 min |
| 5 | `server.py` + `llm_synthesis.py` - Hook query log + context | 30 min |
| 6 | Integration test | 1 hr |

**Total: ~5.5 hours**

## 9. Risks

| Risk | Mitigation |
|---|---|
| SHAP computation slow on CPU | Cache 1hr, use TreeExplainer (fast for XGBoost) |
| MySQL JSON query slow at scale | 1000 row cap ensures <5MB total |
| Memory context makes LLM prompt too long | Truncate context to 300 chars |

## 10. Paper Hook

> *"Memory-Augmented Multi-Agent LLM System with Tool-Augmented Reasoning for Autonomous Retail Operations"*

Key claims:
1. Stateful agent memory enabling cross-query awareness
2. Proactive monitoring with historical baseline comparison
3. SHAP-driven explainability integrated into LLM synthesis
4. Zero-disruption upgrade to an existing deployed system
