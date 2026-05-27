# S5 Multi-Agent Decision Pipeline ? Architecture & Reproducibility


---

## 1. Environment

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.13.13 | MSC v.1944 64 bit (AMD64) |
| OS | Windows 11 | Also works on Linux/macOS |
| MySQL | 8.0+ | Database: `bakery_ai` |
| CUDA | 12.8 (optional) | CPU-only also works |

### 1.1 Python Dependencies

```
fastapi==0.115.0
hypercorn==0.18.0
uvicorn==0.30.6
httpx==0.27.2
mysql-connector-python==9.7.0
numpy==2.4.4
pandas==3.0.2
scikit-learn==1.8.0
scikit-fuzzy==0.5.0
xgboost==3.2.0
shap==0.51.0
torch==2.12.0
torchvision==0.27.0
transformers==5.7.0
datasets==4.8.5
openai==2.38.0
```

### 1.2 Environment Variables (.env)

```bash
DEEPSEEK_API_KEY=sk-xxx          # DeepSeek V3 for composer/summaries
WEATHER_API_KEY=xxx              # VisualCrossing (15-day forecast)
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=
MYSQL_DATABASE=bakery_ai
```

### 1.3 MySQL Setup

```sql
CREATE DATABASE IF NOT EXISTS bakery_ai;
-- Tables auto-created by db/mysql_client.py:init_db()
-- Key tables: batch_inventory, shift_schedule, employees, users
```

---

## 2. Architecture Overview

```
                          ???????????????????????????????????????
                          ?         S5 AGENTIC PIPELINE          ?
                          ?                                      ?
  User Query ??? Step 1: Intent Classification                  ?
                ?    (DistilBERT + Keyword Hybrid)               ?
                ?    Labels: stock_query | waste_analysis |      ?
                ?    promo_eval | schedule_audit |               ?
                ?    cross_source_audit | out_of_scope           ?
                ?    Supports: EN / MS / ZH code-switching       ?
                ?                                                ?
                ??? Step 2: Planner (DAG Generation)             ?
                ?    Dynamic tool selection per intent           ?
                ?    Retry up to 3x on validation failure        ?
                ?                                                ?
                ??? Step 3: Executor (Tool Calls)                ?
                ?    ???????????? ???????????? ????????????     ?
                ?    ? S1: Inv  ? ? S2: Fcst ? ? S3: Sched?     ?
                ?    ? batch_inv? ? forecast ? ? schedule ?     ?
                ?    ???????????? ???????????? ????????????     ?
                ?    Product-filtered. Timeout: 30s.             ?
                ?    Fallback: mock data on all-fail.            ?
                ?                                                ?
                ??? Step 4: Fusion (Intent-Specific)             ?
                ?    stock_query:   compute_restock()            ?
                ?    waste:         compute_waste()              ?
                ?    promo:         compute_promo_roi()          ?
                ?    schedule:      compute_schedule_audit()     ?
                ?    cross_source:  compute_cross_audit()        ?
                ?    Multi-product: per-product comparison       ?
                ?    Carryover:     Q1_max - Q1_min ? Q2_inv     ?
                ?                                                ?
                ??? Step 5: Verifier (4-Tier, R1-R12)            ?
                ?    L1: Data Integrity (null/negative/missing)  ?
                ?    L2: Capacity (R8: restock ? capacity)       ?
                ?    L3: Cross-Module (R9/R10/R11)               ?
                ?         R9:  forecast vs staffing              ?
                ?         R10: forecast vs inventory ? STOCKOUT  ?
                ?         R11: schedule role coverage            ?
                ?    L4: SHAP Causal Attribution (R12)           ?
                ?         TreeExplainer per XGBoost model        ?
                ?         Baseline-feature filtering             ?
                ?         External context (holiday/Ramadan)     ?
                ?         LLM causal report (?15% change)        ?
                ?                                                ?
                ??? Step 6: Composer (LLM Summary)               ?
                ?    DeepSeek V3 with structured prompt          ?
                ?    Includes SHAP attribution + external ctx    ?
                ?    Fallback: mock composer                     ?
                ?                                                ?
                ??? Step 7: Memory Stream (MySQL-Backed)         ?
                     store_episode(query, intent, response,      ?
                                   data_snapshot, product)        ?
                     get_recent_context(session_id, n=3)          ?
                     Auto-reflection (~5% probability)            ?
                     /s5/reflections endpoint                     ?
                          ???????????????????????????????????????

  Cross-Cutting:
  ???????????????  ????????????????  ?????????????????????
  ? B1 Anomaly  ?  ? Code-Switch  ?  ? Proactive         ?
  ? Detector    ?  ? (MS/EN/ZH)   ?  ? Reflection        ?
  ? Isolation   ?  ? DistilBERT   ?  ? Pattern detection ?
  ? Forest +    ?  ? multilingual ?  ? per session       ?
  ? LLM desc    ?  ? date parsing ?  ?                   ?
  ???????????????  ????????????????  ?????????????????????
```

---

## 3. File Map (api/module5_agent/)

| File | Lines | Role |
|------|-------|------|
| `router.py` | 720 | **Main pipeline** ? 7-step orchestration, `/s5/query` endpoint |
| `verifier.py` | 415 | **4-tier verifier** ? R1-R12 rules, LLM judge, SHAP integration |
| `causal_reasoning.py` | 307 | **SHAP attribution** ? TreeExplainer, causal chains, LLM prompts |
| `memory.py` | 317 | **Memory Stream** ? Episodic storage, retrieval, reflection |
| `llm_client.py` | 385 | **LLM interface** ? DeepSeek API, summary/script composers |
| `executor.py` | 181 | **DAG executor** ? Async HTTP calls to S1/S2/S3 |
| `intent.py` | 134 | **Intent classifier** ? DistilBERT + keyword hybrid |
| `fusion.py` | 272 | **Decision fusion** ? Per-intent computation logic |
| `anomaly_detector.py` | 261 | **B1 alerts** ? Isolation Forest + LLM descriptions |
| `planner.py` | 267 | **DAG planner** ? LLM-generated execution plans |
| `monitor.py` | 299 | **System monitor** ? Background health checks |
| `alert_store.py` | 144 | **Alert persistence** ? MySQL-backed alert storage |
| `composer.py` | 84 | **Composer facade** ? LLM vs mock dispatch |
| `sql_templates.py` | 150 | **SQL templates** ? Parameterized query templates |
| `elasticity.py` | 201 | **Price elasticity** ? Demand sensitivity analysis |
| `scenario_engine.py` | 365 | **What-if engine** ? Scenario comparison (deprecated) |

---

## 4. Training Procedures

### 4.1 DistilBERT Intent Classifier

**Model**: `distilbert-base-multilingual-cased`
**Training data**: `training/intent_data.json` (2,506 labeled query-intent pairs)
**Output**: `models/distilbert/` (541 MB safetensors)

```bash
python training/train_distilbert.py
```

**Results** (8 epochs, 20% validation split):
- Accuracy: 96.8%
- stock_query F1: 0.97
- waste_analysis F1: 0.98
- promo_eval F1: 0.99
- schedule_audit F1: 0.97
- cross_source_audit F1: 0.87
- out_of_scope F1: 0.95

**Intent labels**: `stock_query`, `waste_analysis`, `promo_eval`, `schedule_audit`, `cross_source_audit`, `out_of_scope`

**Code-switching support**: ~210 Malay/English mixed samples + Malay keyword fallback rules.

### 4.2 XGBoost Demand Forecast Models

**Features** (15): day_of_week, is_weekend, day_of_month, month, discount_rate, is_public_holiday, is_ramadan, temperature, rainfall, humidity, is_rainy, weather_sunny, weather_cloudy, weather_rainy, weather_storm

**Products** (6): croissant, croissant_chocolate, donut, chiffon, bread_roll, bread_coconut

**Output**: `models/xgboost/{product}_model.json` (total 5.6 MB)

```bash
python training/train_xgboost_full.py
```

---

## 5. Running the System

### 5.1 Start Server

```bash
cd bakery-ai-system
python main.py
# ? MySQL ready: bakery_ai
# ? Running on http://0.0.0.0:8000
```

### 5.2 Frontend

Open `api/module4_frontend/static/index.html` in browser or navigate to `http://localhost:8000`.

### 5.3 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/s5/query` | Main agent query endpoint |
| POST | `/s5/script` | Upselling script generation |
| GET | `/s5/reflections` | Retrieve proactive reflections |
| POST | `/s5/reflections/run` | Trigger reflection generation |
| GET | `/s5/alerts/list` | List B1 anomaly alerts |
| POST | `/s5/alerts/ack` | Acknowledge alerts |
| GET | `/s5/alerts/count` | Unacknowledged alert count |
| POST | `/s5/whatif/compare` | Scenario comparison (deprecated) |
| GET | `/s1/batch_inventory` | Inventory snapshot |
| GET | `/s2/forecast` | Demand forecast (7 days) |
| GET | `/s3/schedule` | Staffing schedule |
| GET | `/s3/capacity` | Production capacity |

---

## 6. Testing the S5 Pipeline

### 6.1 Intent Classification (Code-Switched)

Test queries for the 6 intent labels, including Malay/English mixed:

| Query | Expected Intent |
|-------|----------------|
| `How many croissants tomorrow?` | stock_query |
| `Esok nak bake berapa croissant?` | stock_query |
| `Kenapa banyak sangat waste minggu ni?` | waste_analysis |
| `Promo donut semalam berkesan tak?` | promo_eval |
| `Siapa baker shift petang esok?` | schedule_audit |
| `Check semua system ada problem tak` | cross_source_audit |
| `Cuaca hari ini macam mana?` | out_of_scope |

### 6.2 R12 Causal Reasoning

```bash
curl -X POST http://localhost:8000/s5/query   -H "Content-Type: application/json"   -d '{"query":"How many croissants tomorrow and why?","params":{}}'
```

Expected: Summary includes SHAP attribution (e.g., "Rainy weather is the main factor...")

### 6.3 Multi-Turn Conversation Memory

```bash
# Turn 1
curl -X POST http://localhost:8000/s5/query   -H "Content-Type: application/json"   -d '{"query":"How many croissants tomorrow?","session_id":"test1","params":{}}'

# Turn 2 (follow-up, references previous context)
curl -X POST http://localhost:8000/s5/query   -H "Content-Type: application/json"   -d '{"query":"And the next day?","session_id":"test1","params":{}}'
```

### 6.4 Full Store Health Check

```
Query: "Run a full store health check"
Expected intent: cross_source_audit
Expected: R1-R12 audit results, pass/fail per rule, SHAP causal report
```

---

## 7. Key Design Decisions

### 7.1 Why Hybrid Intent Classification?

DistilBERT provides 96.8% accuracy but keyword rules serve as fallback when:
- Model not trained / PyTorch unavailable
- DL confidence < 0.75 threshold
- DL says `out_of_scope` but keywords match (override)

### 7.2 Why 4-Tier Verification?

| Tier | Fails = | Rationale |
|------|---------|-----------|
| L1 (Data) | Reject | Garbage in = garbage out |
| L2 (Capacity) | Warn | Bottleneck, but can attempt |
| L3 (Cross-Module) | Warn | Contradictions need attention |
| L4 (SHAP) | Flag | Unexplained anomalies ? human review |

### 7.3 Why SHAP Over LIME/Integrated Gradients?

- **TreeExplainer**: Exact Shapley values for tree ensembles (no sampling error)
- **Per-instance**: Each prediction gets its own attribution
- **Axiomatic**: Efficiency, symmetry, dummy, additivity properties

### 7.4 Why Memory Stream (Not LangChain)?

- Zero external dependency
- MySQL-backed (same DB as everything else)
- Product-aware retrieval
- Auto-reflection at ~5% query rate

---

