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
| S5 | Module-based multi-agent analysis and decision support | FastAPI, DAGExecutor, DeepSeek synthesis, DistilBERT/keyword routing | Integrated |

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
| `/health` | GET | Returns service status, registered agent count, and template count |
| `/templates` | GET | Lists configured DAG templates |
| `/analyze` | POST | Runs free-text or intent-based S5 analysis |
| `/analyze/module` | POST | Runs dashboard module analysis for revenue, wastage, forecast, inventory, schedule, or KPI |
| `/priorities` | GET | Returns cached bundle priority recommendations |
| `/discounts` | POST | Returns freshness-aware and S5-priority-aware discount rates |

## S5 Current Design

S5 is a module-based analysis service used by the manager dashboard and POS recommendation flow.

### Runtime Flow

```text
Dashboard or POS request
-> FastAPI S5 endpoint
-> intent routing or module-to-intent mapping
-> predefined DAG template
-> DAGExecutor phase validation and execution
-> deterministic agent analysis
-> optional cross-agent deliberation
-> Synthesizer summary and recommendations
-> structured JSON response
```

### Module Intent Map

| Dashboard Module | S5 Intent |
| --- | --- |
| revenue | `profit_root_cause` |
| wastage | `wastage_root_cause` |
| forecast | `production_advice` |
| inventory | `inventory_diagnosis` |
| schedule | `staffing_diagnosis` |
| kpi | `full_diagnosis` |

### DAG Templates

S5 currently ships these predefined templates:

| Template | Purpose |
| --- | --- |
| `profit_root_cause` | Daily profit and revenue root-cause analysis |
| `wastage_root_cause` | Wastage and spoilage analysis |
| `production_advice` | Seven-day forecast, production, and material planning |
| `inventory_diagnosis` | Product and material stock assessment |
| `staffing_diagnosis` | Schedule and staffing issue analysis |
| `full_diagnosis` | Comprehensive cross-module store health check |
| `promo_evaluation` | Promotion and pricing impact analysis |

### Registered Agent Areas

S5 uses registered Python agents rather than a single monolithic planner. The active analysis areas include:

- Revenue, profit, pricing, and promotion analysis
- Forecast overview, uncertainty, accuracy, and production planning
- Material stock, material procurement, and product stock diagnosis
- Wastage, yield, production, and operational risk analysis
- Attendance, staffing, and schedule diagnosis
- Product mix, hourly pattern, trend, feature sensitivity, and external factor analysis
- Cross-card risk, causal chain, metric conflict, and recommendation synthesis

### Safety Checks

The DAG executor validates every template before execution:

- Template must contain nodes.
- Agent names must be registered.
- Agent names cannot be duplicated within one template.
- Phase numbers must be within the configured range.
- Dependencies must exist in the same template.
- Dependencies must run in an earlier phase than the dependent agent.

This prevents silent execution of malformed templates.

## Frontend Integration

The frontend uses:

- `S5_API + "/analyze/module"` for module analysis buttons.
- `S5_API + "/analyze"` for the free-text manager query panel.
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
- Common dashboard tasks use predefined DAG templates instead of open-ended planning.
- LLM usage is limited to synthesis and language generation when an API key is available.
- Deterministic agent outputs remain available when the LLM path is unavailable.
