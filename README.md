# Bakery AI System

Multi-agent AI operations system for a Malaysian bakery-cafe.

## Architecture

| Module | Function | Tech |
|--------|----------|------|
| M1 (S1) | Visual perception -- YOLO-based product detection + tray color classification | YOLOv8, OpenCV |
| M2 (S2) | 7-day sales forecasting with confidence bounds | XGBoost, TimeSeriesSplit |
| M3 (S3) | CP-SAT shift scheduling, P_staff/P_oven capacity model | OR-Tools CP-SAT |
| M4 (S4) | BFF layer -- JWT auth, combo scoring, web UI | FastAPI, JWT, HTML/CSS/JS |
| M5 (S5) | 5-layer heterogeneous multi-agent engine | DeepSeek, Qwen-2.5, GPT-4o-mini |

## Quick Start

### Prerequisites
- Python 3.11+
- Supabase account (configured in .env)
- API keys: DeepSeek, OpenAI, Qwen (or mock mode)

### Setup

pip install -r requirements.txt
cp .env.example .env   # then edit with your keys
python main.py

Open http://localhost:8000 for the web UI.

### Docker

docker-compose up -d

## Default Accounts
- Manager: manager / hash123
- Staff: staff01 / hash123

## API Endpoints

| Method | Path | Module |
|--------|------|--------|
| POST | /s1/checkout | Visual scan (checkout) |
| POST | /s1/inflow | Visual scan (batch inflow) |
| GET | /s1/batch_inventory | Current inventory |
| GET | /s2/forecast | 7-day forecast |
| GET | /s2/sales_history | Transaction history |
| GET | /s3/schedule | Shift schedule |
| GET | /s3/capacity | Production capacity |
| POST | /s4/login | JWT login |
| POST | /s4/combo | Combo recommendations |
| GET | /s4/me | Current user |
| POST | /s5/query | Agent query (manager) |
| POST | /s5/script | Sales script generation |

## Training Models

python training/train_yolo.py      # YOLOv8 product detection
python training/train_xgboost.py   # XGBoost demand forecasting
python training/train_distilbert.py # DistilBERT intent classifier

## Project Structure

bakery-ai-system/
|-- main.py                 # FastAPI entry point
|-- config/settings.py      # Configuration
|-- db/supabase_client.py   # Database client
|-- models/schemas.py       # Pydantic models
|-- api/
|   |-- module1_yolo.py     # Visual perception
|   |-- module2_forecast.py # Sales forecasting
|   |-- module3_scheduling.py # Shift scheduling
|   |-- module4_frontend/   # BFF + web UI
|   |   |-- bff.py
|   |   +-- static/index.html
|   |-- module5_agent/      # Multi-agent engine
|   |   |-- router.py
|   |   |-- intent.py
|   |   |-- planner.py
|   |   |-- fusion.py
|   |   |-- verifier.py
|   |   +-- composer.py
|   +-- mock_llm.py         # Mock LLM for dev
|-- training/               # Model training scripts
+-- models/                 # Saved models + cache
