# Bakery AI System

Multi-agent AI operations system for a medium-sized Malaysian bakery-cafe (6 products, 10 employees, 2 ovens).  
**Location:** Kuala Lumpur, Malaysia | **Rest Day:** Monday

## Architecture

| Module | Function | Tech | Port |
|--------|----------|------|------|
| S1 | YOLO recognition ¡ª product detection + tray color classification + FIFO deduction | YOLOv8n, OpenCV | 8002 |
| S2 | 7-day demand forecasting with weather + holiday integration | XGBoost, VisualCrossing | 8002 |
| S3 | CP-SAT shift scheduling with demand-aware role coverage + dual-role support | OR-Tools CP-SAT | 8002 |
| S4 | BFF + Responsive Web POS ¡ª JWT auth, 5-dim combo scoring, checkout + receipt | FastAPI, JWT, HTML/CSS/JS | 8002 |
| S5 | Multi-agent AI Brain ¡ª 6 agents, DistilBERT intent classifier, MIP Pareto plans, LLM synthesis | DeepSeek V4, DistilBERT, HiGHS MIP | 8001 |

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
cd s5_agent
python -c "from intent_classifier import train_intent_classifier; train_intent_classifier()"
cd ..

# 6. Start S5 AI Brain (port 8001)
cd s5_agent
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
| POST | /s4/combo | S4 | Combo recommendations (6 breads x 6 coffees) |
| POST | /s4/checkout/complete | S4 | Payment + FIFO deduction + receipt |
| GET | /s4/products | S4 | Product prices + freshness |
| GET | /s4/receipts | S4 | Receipt history |
| POST | /s5/query | S5 | Agent query (8 intents, DistilBERT-classified) |
| POST | /s5/discounts | S5 | 5-dim dynamic discount engine |
| POST | /s5/script | S5 | Sales script generation (Composer) |
| GET | /s5/alerts/list | S5 | System alerts |
| GET | /s5/alerts/count | S5 | Unacknowledged alert count |
| POST | /s5/alerts/ack | S5 | Acknowledge alert |
| GET | /s5/health | S5 | Health check + active agents |

## S5 Multi-Agent Brain

### Pipeline

```
User Query -> DistilBERT Intent Classifier (8 intents) -> 6-Agent Orchestration -> Arbitrator (MIP Pareto Plans) -> LLM Plan Selection -> LLM Synthesis (AI Summary)
```

### 6 Agents (programmatic, no LLM)

| Agent | Confidence | Data Source |
|-------|-----------|-------------|
| demand | 90% | S2 XGBoost forecast + per-product trend |
| inventory | 95% | MySQL batch_inventory + freshness tracking |
| production | 85% | Baker count x oven capacity x effective hours |
| staffing | 95% | S3 CP-SAT schedule |
| promo | 85% | 5-dim scoring: F(25%) S(25%) M(20%) T(15%) P(15%) |
| profit | 50-90% | MySQL transaction history (7-day average) |

### 8 Intents (DistilBERT, 7764 samples)

| Intent | Active Agents | Example |
|--------|---------------|---------|
| stock_query | all 6 | "How many croissants tomorrow?" |
| comparison_analysis | demand, inventory, profit | "Compare croissant and donut" |
| waste_analysis | demand, inventory | "Why is waste high this week?" |
| promo_eval | demand, inventory, promo, profit | "How to promo all products?" |
| schedule_audit | staffing | "Check tomorrow's schedule" |
| profit_analysis | demand, inventory, profit | "What's our profit margin?" |
| cross_source_audit | all 6 | "Run a full store health check" |
| out_of_scope | none | Jokes, weather, unrelated |

### LLM Decision Layer

- **DeepSeek V4 Pro** (`deepseek-v4-pro`) ¡ª Pareto plan selection (Arbitrator)
- **DeepSeek V4 Flash** (`deepseek-v4-flash`) ¡ª AI Summary synthesis + Sales script (Composer)
- Arbritrator generates 4 plans (A_aggressive / B_balanced / C_conservative / D_baseline) ¡ú dominance filter ¡ú LLM selects best

### 5-Dimension Dynamic Discount Engine

| Dimension | Weight | Source |
|-----------|--------|--------|
| Freshness (F) | 25% | Day-1 stock ratio from S1 batch_inventory |
| Surplus (S) | 25% | Stock vs forecast gap |
| Margin (M) | 20% | Product profit margin |
| Trend (T) | 15% | Demand trend direction from S2 |
| Pairing (P) | 15% | Combo score from S4 (inverse: good pairing ¡ú lower discount need) |

## S4 POS Features

### Combo Scoring (5-dim)
- Flavor Pairing (25%): Bread-coffee affinity matrix (LLM-generated, cached)
- Discount Value (20%): Higher discount = better deal
- Freshness (20%): Day-1 items need promotion
- Inventory Pressure (20%): High stock = push harder
- Order Context (15%): Complement cart items
- Top-3 bundles with savings breakdown

### POS Checkout
- YOLO scan ¡ú product detection + tray color ¡ú freshness classification
- Tray color: Green (Fresh) / Orange (Day-1/Day-2 discount)
- Coffee items excluded from stock check (made-to-order)
- Real-time FIFO deduction + receipt generation

## S3 Employee Constraints

10 employees with dual-role support:

| ID | Name | Role | Secondary | Max Hrs |
|----|------|------|-----------|---------|
| E001 | Ali | baker | ¡ª | 56h |
| E002 | Mei | cashier | ¡ª | 56h |
| E003 | Raj | barista | cashier | 56h |
| E004 | Siti | baker | ¡ª | 56h |
| E005 | Ahmad | baker | ¡ª | 56h |
| E006 | Priya | cashier | ¡ª | 56h |
| E007 | Kumar | baker | ¡ª | 56h |
| E008 | David | baker | ¡ª | 56h |
| E009 | Chen | barista | ¡ª | 56h |
| E010 | Fatima | manager | ¡ª | 42h |

Demand-driven staffing:
- **High** (>250 units/day): baker>=4, cashier=2, barista=2, manager=1
- **Normal** (150-250): baker>=3, cashier=2, barista=1, manager=1
- **Low** (<150): baker>=2, cashier=2, barista=1, manager=1

Shifts: 06:00-13:00 (morning), 12:00-19:00 (afternoon). Store open 9:00-19:00, prep 6:00-9:00.

## System Alerts (S5)

- 5-minute monitoring interval
- 4 tiers: inventory (waste risk), forecast (anomaly), schedule (coverage gap), trend (rising/falling)
- Monday = rest day (no alerts for Monday)
- Acknowledge/dismiss from frontend

## Key Design Decisions

- **Coffee = no inventory**: Made-to-order, appears only in S3 barista ratio + S4 pairing
- **2 roles**: staff (limited) / manager (full access) with JWT auth
- **BFF architecture**: S4 has lightweight Python API layer (FastAPI) between frontend and S5
- **Heterogeneous LLM**: DeepSeek for synthesis/decision, DistilBERT (local) for intent classification
- **MILP ¡ú LP fallback**: HiGHS MIP for exact integer plans, LP + batch rounding as graceful degradation
- **All code in English**, system language: English

## Project Structure

```
bakery-ai-system/
©À©¤©¤ main.py                        # FastAPI entry (port 8002)
©À©¤©¤ requirements.txt
©À©¤©¤ .env.example
©À©¤©¤ config/settings.py             # Configuration + secrets
©À©¤©¤ db/mysql_client.py             # MySQL database client
©À©¤©¤ models/
©¦   ©À©¤©¤ xgboost/                   # 6 product forecast models (JSON)
©¦   ©À©¤©¤ anomaly_isolation_forest.pkl
©¦   ©¸©¤©¤ schedule_baseline.json     # S3 schedule persistence
©À©¤©¤ api/
©¦   ©À©¤©¤ module1_yolo.py            # Visual perception + inventory
©¦   ©À©¤©¤ module2_forecast.py        # Sales forecasting (XGBoost)
©¦   ©À©¤©¤ module3_scheduling.py      # Shift scheduling (CP-SAT)
©¦   ©À©¤©¤ freshness_service.py       # Freshness discount engine
©¦   ©À©¤©¤ weather.py                 # VisualCrossing API
©¦   ©À©¤©¤ mock_llm.py                # LLM mock for testing
©¦   ©À©¤©¤ train_quantile.py          # Quantile regression training
©¦   ©¸©¤©¤ module4_frontend/
©¦       ©À©¤©¤ bff.py                 # Backend-for-frontend + JWT + combo
©¦       ©À©¤©¤ pairing_llm.py         # LLM flavor matrix generator
©¦       ©¸©¤©¤ static/
©¦           ©¸©¤©¤ index.html         # Full web POS (inline CSS + JS)
©À©¤©¤ s1_recognition/                # YOLO training + detection scripts
©À©¤©¤ s5_agent/                      # S5 AI Brain (port 8001)
©¦   ©À©¤©¤ server.py                  # FastAPI entry + agent orchestration
©¦   ©À©¤©¤ arbitrator.py              # Cross-agent audit + MIP plan generation
©¦   ©À©¤©¤ optimizer.py               # Pareto-optimal bake plans + 7-day projection
©¦   ©À©¤©¤ llm_synthesis.py           # DeepSeek natural language summary (8 templates)
©¦   ©À©¤©¤ intent_classifier.py       # DistilBERT intent classifier (8 intents)
©¦   ©À©¤©¤ monitor.py                 # Periodic alert monitor (5-min interval)
©¦   ©À©¤©¤ alert_store.py             # Alert JSON file store
©¦   ©À©¤©¤ memory_store.py            # Query memory + trend tracking
©¦   ©À©¤©¤ agents/
©¦   ©¦   ©À©¤©¤ base.py                # Agent base class
©¦   ©¦   ©À©¤©¤ demand.py              # Forecast agent (per-product)
©¦   ©¦   ©À©¤©¤ inventory.py           # Stock + freshness agent
©¦   ©¦   ©À©¤©¤ production.py          # Capacity + net-demand agent
©¦   ©¦   ©À©¤©¤ staffing.py            # Schedule agent
©¦   ©¦   ©À©¤©¤ promo.py               # 5-dim dynamic discount agent
©¦   ©¦   ©¸©¤©¤ profit.py              # 7-day revenue/cost/margin agent
©¦   ©À©¤©¤ models/distilbert_intent/  # Trained classifier
©¦   ©¸©¤©¤ s5_config/
©¦       ©À©¤©¤ settings.py            # S5 constants
©¦       ©¸©¤©¤ training_data.json     # 7764 labeled intent samples
©À©¤©¤ scripts/                       # Utility scripts
©À©¤©¤ tests/                         # Test suite
©¸©¤©¤ data/                          # Training + historical data
```

## Training Models

```bash
# DistilBERT intent classifier (required for S5)
cd s5_agent
python -c "from intent_classifier import train_intent_classifier; train_intent_classifier()"

# XGBoost demand forecasting (6 products)
python scripts/train_xgboost_full.py
```