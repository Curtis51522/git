# Bakery AI System

Multi-agent AI operations system for a Malaysian bakery-cafe.

## Architecture

| Module | Function | Tech |
|--------|----------|------|
| S1 | Visual perception -- YOLO-based product detection + tray color classification + FIFO deduction | YOLOv8/YOLO26m, OpenCV |
| S2 | 7-day sales forecasting with confidence bounds + multi-model comparison | XGBoost, TimeSeriesSplit, WAPE |
| S3 | CP-SAT shift scheduling with demand-aware role coverage + swap support | OR-Tools CP-SAT |
| S4 | BFF layer -- JWT auth, 5-dim combo scoring, checkout + receipt, web POS | FastAPI, JWT, HTML/CSS/JS |
| S5 | Agentic pipeline -- 6 intents (stock/waste/promo/schedule/audit/profit), 12-rule audit, SHAP causal | DistilBERT, DeepSeek, SHAP |

## Quick Start

### Prerequisites

- Python 3.11+
- MySQL 8.0+
- API keys: DeepSeek and VisualCrossing (weather)

### Setup

```bash
# 1. Clone
git clone https://github.com/Curtis51522/git.git
cd git

# 2. Download pre-trained models (~570 MB)
python download_models.py

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your DeepSeek API key, VisualCrossing API key, and MySQL credentials

# 5. Start MySQL and create database
# mysql -u root -e "CREATE DATABASE IF NOT EXISTS bakery_ai"

# 6. (Optional) Train models from scratch
python training/train_xgboost_full.py
python training/train_distilbert.py

# 7. Run
python main.py
```

Open http://localhost:8000 for the web UI.

### Default Accounts

| Role | Username | Password |
|------|----------|----------|
| Manager | manager | hash123 |
| Staff | staff1 | hash123 |

## API Endpoints

| Method | Path | Module |
|--------|------|--------|
| GET | /s1/batch_inventory | Current inventory with freshness |
| POST | /s1/checkout | Visual scan (checkout) |
| POST | /s1/inflow | Visual scan (batch inflow) |
| GET | /s2/forecast | 7-day forecast (low/median/high) |
| GET | /s3/schedule | Shift schedule |
| GET | /s3/capacity | Production capacity |
| POST | /s4/login | JWT login |
| POST | /s4/combo | Combo recommendations |
| POST | /s4/checkout/complete | Payment + FIFO deduction + receipt |
| GET | /s4/products | Product prices (bakery only) |
| POST | /s5/query | Agent query (6 intents incl. profit_analysis) |
| POST | /s5/script | Sales script generation |
| GET | /s5/alerts/list | Anomaly alerts |
| GET | /s5/alerts/count | Unacknowledged alert count |
| GET | /s5/reflections | Reflective memory insights |



## Features

### POS and Receipt
- Real-time POS checkout with freshness-aware pricing (Fresh/Day-1/Day-2/Near-Expired)
- Thermal-style receipt generation with print support
- Receipts stored in receipts table for audit trail

### AI Bundle Recommendations
- 5-dimension scoring: flavor pairing, discount value, freshness urgency, inventory pressure, order context
- Top-3 bundles with savings breakdown

### Agent Intelligence (S5)
- 6 intents: stock_query, waste_analysis, promo_eval, schedule_audit, cross_source_audit, profit_analysis
- 4-tier audit (L1-L4) with SHAP causal attribution
- Profit analysis: revenue/cost/margin by product from real transaction data
- Expired batch auto-cleanup (30-day retention)

### Inventory
- FIFO batch deduction with freshness tracking
- 4-stage aging: Fresh to Day-1 to Day-2 to Near-Expired to Expired
- Product prices unified in database (single source of truth)

## Training Models

```bash
python training/train_yolo.py          # YOLOv8 product detection
python training/train_xgboost_full.py  # XGBoost demand forecasting
python training/train_distilbert.py    # DistilBERT intent classifier
```

## Project Structure

```
bakery-ai-system/
|-- main.py                     # FastAPI entry point
|-- download_models.py          # Model downloader (from GitHub Releases)
|-- config/settings.py          # Configuration
|-- db/mysql_client.py          # MySQL database client
|-- models/
|   |-- distilbert/             # Intent classifier (516 MB)
|   |-- xgboost/                # 6 product forecast models
|   |-- yolo/                   # Freshness detection (~50 MB)
|   +-- anomaly_isolation_forest.pkl
|-- api/
|   |-- module1_yolo.py         # Visual perception
|   |-- module2_forecast.py     # Sales forecasting
|   |-- module3_scheduling.py   # Shift scheduling
|   |-- module4_frontend/       # BFF + web UI
|   |   |-- bff.py
|   |   +-- static/
|   |-- module5_agent/          # S5 decision pipeline
|   |   |-- router.py           # Main pipeline orchestration
|   |   |-- intent.py           # DistilBERT + keyword classifier
|   |   |-- planner.py          # DAG planner + validation
|   |   |-- executor.py         # API call executor
|   |   |-- fusion.py           # Deterministic business logic
|   |   |-- verifier.py         # 12-rule audit (L1-L4)
|   |   |-- composer.py         # Natural language summary
|   |   |-- memory.py           # Episodic + reflective memory
|   |   |-- causal_reasoning.py # SHAP causal attribution
|   |   |-- anomaly_detector.py # Isolation Forest alerts
|   |   +-- llm_client.py       # DeepSeek API client
|   +-- weather.py              # VisualCrossing + fallbacks
|-- training/                   # Model training scripts
+-- data/sales_history.csv      # Historical sales data
```
