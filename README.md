# Bakery AI System

Multi-agent operations system for a medium-sized bakery-cafe.

Scope: 30 bread and pastry products, 15 beverage menu items, 10 employees, and 2 ovens.

## Architecture

| Module | Function | Main Tech | Runtime Status |
| --- | --- | --- | --- |
| S1 | Product recognition, batch inflow, checkout outflow, FIFO inventory deduction | YOLOv11s, OpenCV, FastAPI | Integrated |
| S2 | Seven-day demand forecasting with weather and calendar features | XGBoost, pandas, VisualCrossing | Integrated |
| S3 | Demand-aware staff scheduling and production capacity estimation | OR-Tools CP-SAT, FastAPI | Integrated |
| S4 | Unified web POS and manager dashboard | FastAPI, HTML, CSS, JavaScript, JWT | Integrated |
| S5 | Dashboard-embedded multi-agent analysis and decision support | FastAPI, LangGraph, evidence-grounded rule synthesis | Integrated |

## Quick Start

### Prerequisites

- Python 3.11+
- MySQL 8.0+
- API keys for DeepSeek and VisualCrossing, if live LLM/weather paths are used

### Setup

```bash
git clone https://github.com/Curtis51522/git.git
cd git
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with database credentials and API keys.

Start the main server:

```bash
python main.py
```

Open:

```text
http://localhost:8002
```

The main server proxies S5 requests from `/s5/*` to the S5 analysis service on port `8001`.

### S5 Service

Start S5 separately when running the full dashboard analysis flow:

```bash
python -m s5_agent.server
```

S5 endpoints:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Returns service status and supported LangGraph modules |
| `/analyze/module` | POST | Runs dashboard module analysis for revenue, wastage, forecast, inventory, or promotion mix |
| `/priorities` | GET | Returns cached bundle priority recommendations |
| `/discounts` | POST | Returns freshness-aware and S5-priority-aware discount rates |

## S5 Current Design

S5 is a module-based analysis service used by the manager dashboard and POS recommendation flow.

### Runtime Flow

```text
Dashboard or POS request
-> FastAPI S5 endpoint
-> module-to-template mapping
-> LangGraph workflow
-> deterministic specialist agents
-> evidence graph and verification checks
-> constrained summary and recommendations
-> structured JSON response
```

### Module Template Map

| Dashboard Module | LangGraph Template |
| --- | --- |
| revenue | `profit_root_cause` |
| wastage | `wastage_root_cause` |
| forecast | `production_advice` |
| inventory | `inventory_diagnosis` |
| promotion_mix | `promotion_mix_analysis` |

Schedule, attendance, and KPI ranking remain S3 dashboard functions. They do not currently expose S5 AI analysis endpoints.

### LangGraph Templates

S5 currently ships these module workflows:

| Template | Purpose |
| --- | --- |
| `profit_root_cause` | Daily profit and revenue root-cause analysis |
| `wastage_root_cause` | Wastage and spoilage analysis |
| `production_advice` | Seven-day forecast, production, and material planning |
| `inventory_diagnosis` | Product and material stock assessment |
| `promotion_mix_analysis` | Promotion signal, product mix, and bundle decision analysis |

### Registered Agent Areas

S5 uses registered Python agents rather than a single monolithic planner. The active analysis areas include:

- Revenue, profit, pricing, and promotion analysis
- Forecast overview, uncertainty, accuracy, and production planning
- Material procurement and finished-product inventory diagnosis
- Wastage, yield, production, and operational risk analysis
- Product mix and bundle recommendation synthesis

### Safety Checks

The LangGraph runtime keeps module execution constrained:

- Only explicitly supported dashboard modules can call `/analyze/module`.
- Each workflow produces structured agent outputs.
- Recommendations must link back to evidence IDs.
- Verification reports expose missing evidence, unsupported recommendations, and data-quality warnings.
- Unsupported modules fail fast instead of falling back to legacy routes.

This prevents silent execution of stale analysis paths.

## Frontend Integration

The frontend uses:

- `S5_API + "/analyze/module"` for module analysis buttons.
- `S5_API + "/priorities"` and `S5_API + "/discounts"` for bundle and discount support.

S5 analysis text is escaped before being inserted into HTML in the dashboard analysis component.

## Default Accounts

| Role | Username | Password |
| --- | --- | --- |
| Manager | manager | hash123 |
| Staff | staff1 | hash123 |

## Key Design Decisions

- The browser UI is the single frontend surface for POS and manager workflows.
- S4 owns the visible dashboard and POS experience.
- S5 owns backend analysis and decision support, exposed through structured JSON APIs.
- Common dashboard tasks use predefined LangGraph workflows instead of open-ended chat planning.
- Current S5 summaries are constrained to verified metrics, evidence, and module-specific rules.
- Deterministic agent outputs keep the dashboard usable without a live LLM dependency.
