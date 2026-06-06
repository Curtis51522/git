# Bakery AI System

Multi-agent AI operations system for a medium-sized Malaysian bakery-cafe (6 products, 10 employees, 2 ovens).

## Architecture

| Module | Function | Tech | Port |
|--------|----------|------|------|
| S1 | Visual perception -- YOLO-based product detection + tray color classification + FIFO deduction | YOLOv8, OpenCV | 8002 |
| S2 | 7-day demand forecasting with weather integration + multi-model comparison | XGBoost, TimeSeriesSplit, VisualCrossing/Open-Meteo | 8002 |
| S3 | CP-SAT shift scheduling with demand-aware role coverage + dual-role support | OR-Tools CP-SAT | 8002 |
| S4 | BFF layer -- JWT auth, 5-dim combo scoring, checkout + receipt, web POS | FastAPI, JWT, HTML/CSS/JS | 8002 |
| S5 | Multi-agent AI Brain -- 6 agents, DistilBERT intent classifier, arbitrated MIP, LLM synthesis | DeepSeek, DistilBERT, CP-SAT MIP | 8001 |

## Quick Start

### Prerequisites

- Python 3.11+
- MySQL 8.0+
- API keys: DeepSeek (S4 pairing + S5 synthesis) and VisualCrossing (weather)

### Setup

```bash
# 1. Clone
git clone https://github.com/Curtis51522/git.git
cd git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your DeepSeek API key, VisualCrossing API key, and MySQL credentials

# 4. Start MySQL and create database
# mysql -u root -e "CREATE DATABASE IF NOT EXISTS bakery_ai"

# 5. Train DistilBERT intent classifier (~3 min, one-time)
cd s5-agent-brain
python -c "from intent_classifier import train_intent_classifier; train_intent_classifier()"
cd ..

# 6. Start S5 AI Brain (port 8001)
cd s5-agent-brain
python server.py

# 7. Start main server (port 8002) in another terminal
cd ..
python main.py
```

Open http://localhost:8002 for the web UI.

### Default Accounts

| Role | Username | Password |
|------|----------|----------|
| Manager | manager | hash123 |
| Staff | staff1 | hash123 |

## API Endpoints

| Method | Path | Module | Description |
|--------|------|--------|-------------|
| GET | /s1/batch_inventory | S1 | Current inventory with freshness |
| POST | /s1/checkout | S1 | Visual scan (checkout) |
| POST | /s1/inflow | S1 | Visual scan (batch inflow) |
| POST | /s1/query | S1 | Text-based inventory query |
| GET | /s2/forecast | S2 | 7-day forecast (low/median/high) |
| POST | /s3/solve | S3 | Generate shift schedule |
| GET | /s3/schedule | S3 | Current shift schedule |
| POST | /s3/sick | S3 | Mark employee sick + auto-replace |
| POST | /s3/unsick | S3 | Clear sick status |
| POST | /s4/login | S4 | JWT login |
| POST | /s4/combo | S4 | Combo recommendations (6 breads x 8 coffees) |
| POST | /s4/checkout/complete | S4 | Payment + FIFO deduction + receipt |
| GET | /s4/products | S4 | Product prices + freshness |
| GET | /s4/receipts | S4 | Receipt history |
| POST | /s5/query | S5 | Agent query (8 intents, DistilBERT-classified) |
| POST | /s5/discounts | S5 | 5-dim dynamic discount engine |
| GET | /s5/alerts/list | S5 | Anomaly alerts |
| GET | /s5/alerts/count | S5 | Unacknowledged alert count |
| POST | /s5/alerts/ack | S5 | Acknowledge alert |

## S5 Multi-Agent Brain

### Pipeline

```
User Query -> DistilBERT Intent Classifier (8 labels) -> Agent Orchestration (6 agents) -> Arbitrator + MIP -> LLM Decision + Synthesis
```

### Agents

| Agent | Confidence | Role |
|-------|-----------|------|
| demand | 90% | XGBoost forecast with per-product trend analysis |
| inventory | 95% | Stock levels, freshness, waste risk, per-product breakdown |
| production | 85% | Net capacity (bakers x ovens x hours), inventory-aware recommendation |
| staffing | 95% | Shift schedule from S3 CP-SAT solver |
| promo | 85% | 5-dim dynamic discount (Freshness, Surplus, Margin, Trend, Pairing) |
| profit | 90% | 7-day averaged revenue/cost/margin from transaction data |

### Intents (DistilBERT, 7764 training samples, 8 labels)

| Intent | Description | Active Agents |
|--------|-------------|---------------|
| stock_query | Single/multi-product stock inquiries | all 6 |
| comparison_analysis | Side-by-side product comparison | demand, inventory, profit |
| waste_analysis | Spoilage risk, per-product ranking | demand, inventory |
| promo_eval | Discount recommendation | demand, inventory, promo, profit |
| schedule_audit | Staffing check | staffing |
| profit_analysis | Revenue/cost/margin | demand, inventory, profit |
| cross_source_audit | Full store health check | all 6 |
| out_of_scope | Non-bakery query rejection | none |

### 5-Dimension Dynamic Discount Engine

| Dimension | Weight | Source |
|-----------|--------|--------|
| Freshness (F) | 30% | Day-1 stock ratio from S1 inventory |
| Surplus (S) | 30% | Stock vs forecast gap |
| Margin (M) | 20% | Product profit margin |
| Trend (T) | 15% | Demand trend direction from S2 |
| Pairing (P) | 5% | Combo score from S4 flavor matrix |

Urgency = F*0.30 + S*0.30 + M*0.20 + T*0.15 + P*0.05 -> mapped to discount tiers (0%-50%).

### LLM Synthesis
- DeepSeek-powered natural language summaries
- Language detection: English queries -> English response, Malay -> Malay
- Anti-hallucination: LLM selects from pre-computed Pareto plans, cannot fabricate
- Counterfactual explanation + 7-day projection for stock queries

## S4 Combo Recommendation

- 6 breads x 8 coffees flavor pairing matrix (LLM-generated by DeepSeek)
- 5-dim scoring: flavor pairing, discount value, freshness urgency, inventory pressure, order context
- Top-3 bundles with savings breakdown
- Discount always on bread (never coffee)
- Real-time POS checkout with freshness-aware pricing (Fresh/Day-1/Near-Expired)

## S3 Employee Constraints

10 employees with dual-role support:

| ID | Name | Role | Secondary | Max Hrs | Note |
|----|------|------|-----------|---------|------|
| E001 | Ali | baker | - | 56h | |
| E002 | Mei | cashier | - | 56h | |
| E003 | Raj | barista | cashier | 56h | Dual-role |
| E004 | Siti | baker | - | 56h | |
| E005 | Ahmad | baker | - | 56h | Deputy manager |
| E006 | Priya | cashier | - | 56h | |
| E007 | Kumar | baker | - | 56h | |
| E008 | David | baker | - | 56h | |
| E009 | Chen | barista | - | 56h | |
| E010 | Fatima | manager | - | 42h | |

Demand-driven staffing levels:
- **High** (>250 units): baker>=4, cashier=2, barista=2, manager=1, no dual-role
- **Normal** (150-250): baker=3, cashier=2, barista=1, manager=1 (dual-role allowed)
- **Low** (<150): baker=2, cashier=2, barista=1, manager=1 (dual-role allowed)

Shifts: 06:00-13:00 (morning), 12:00-19:00 (afternoon). Store open 9:00-19:00. Prep 6:00-9:00.

## Features

### POS and Receipt
- Real-time POS checkout with freshness-aware pricing
- Thermal-style receipt generation with print support
- Receipts stored in receipts table for audit trail

### Inventory
- FIFO batch deduction with freshness tracking
- 4-stage aging: Fresh -> Day-1 -> Day-2 -> Near-Expired -> Expired
- Auto-alert when >50% Day-1 stock
- Freshness dots hidden when stock=0

### System Alerts (S5)
- 5-minute monitoring interval
- Six-product coverage (all 6 products checked)
- Inventory waste warnings (Day-1 stock threshold)
- Duplicate alerts update instead of duplicate
- Acknowledge/dismiss from frontend

### Language Support
- English and Bahasa Malaysia (mixed input supported)
- LLM summaries auto-detect query language
- Malay date words: esok, hari ini, lusa, semalam, plus day names (Isnin-Ahad)
- Monday rest-day detection with bilingual note

## Project Structure

```
bakery-ai-system/
|-- main.py                        # FastAPI entry (port 8002)
|-- requirements.txt
|-- config/settings.py             # Configuration + secrets
|-- db/mysql_client.py             # MySQL database client
|-- models/
|   |-- xgboost/                   # 6 product forecast models (JSON)
|   |-- anomaly_isolation_forest.pkl
|   +-- schedule_baseline.json     # S3 schedule persistence
|-- api/
|   |-- module1_yolo.py            # Visual perception + inventory
|   |-- module2_forecast.py        # Sales forecasting (XGBoost)
|   |-- module3_scheduling.py      # Shift scheduling (CP-SAT)
|   |-- module4_frontend/
|   |   |-- bff.py                 # Backend-for-frontend + JWT
|   |   |-- pairing_llm.py         # LLM flavor matrix generator
|   |   +-- static/
|   |       |-- index.html         # Full web POS (inline CSS + JS)
|   +-- weather.py                 # VisualCrossing + Open-Meteo fallback
|-- s5-agent-brain/                # S5 AI Brain (port 8001)
|   |-- server.py                  # FastAPI entry + LLM planner
|   |-- arbitrator.py              # Cross-agent health audit + MIP decision
|   |-- optimizer.py               # Pareto-optimal bake plans + multi-period projection
|   |-- llm_synthesis.py           # DeepSeek natural language summary (8 templates)
|   |-- intent_classifier.py       # DistilBERT intent classifier (8 labels)
|   |-- monitor.py                 # Periodic alert monitor (5-min interval)
|   |-- alert_store.py             # Alert JSON file store
|   |-- agents/
|   |   |-- base.py                # Agent base class
|   |   |-- demand.py              # Forecast agent (per-product)
|   |   |-- inventory.py           # Stock + freshness agent
|   |   |-- production.py          # Capacity + net-demand agent
|   |   |-- staffing.py            # Schedule agent
|   |   |-- promo.py               # 5-dim dynamic discount agent
|   |   +-- profit.py              # 7-day revenue/cost/margin agent
|   |-- models/distilbert_intent/  # Trained classifier (after training)
|   |   |-- config.json
|   |   |-- tokenizer.json
|   |   +-- model.safetensors      # Generate via: train_intent_classifier()
|   +-- s5_config/
|       |-- settings.py            # S5 constants + product definitions
|       +-- training_data.json     # 7764 labeled intent samples
|-- data/
|   |-- kaggle/Bakery sales.csv
|   |-- sales_history.csv
|   +-- synthetic_sales_1year.csv
+-- test_tray_images/              # S1 test images (gitignored)
```

## Training Models

```bash
# DistilBERT intent classifier (required for S5, run once)
cd s5-agent-brain
python -c "from intent_classifier import train_intent_classifier; train_intent_classifier()"

# XGBoost demand forecasting (6 products)
python training/train_xgboost_full.py
```
