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

## Training Data

The complete 36,969-image, 30-class S1 training dataset is retained under
`data/merged_yolo_30cls/` with its labels and `data.yaml` configuration.

## Quick Start

### Prerequisites

- Python 3.13
- MySQL Community Server 8.4 LTS
- API keys for DeepSeek and VisualCrossing, if live LLM/weather paths are used

### Setup

```powershell
git clone https://github.com/Curtis51522/git.git
cd git
$venv = Join-Path $env:LOCALAPPDATA "BakeryAI\venv313"
py -3.13 -m venv $venv
& "$venv\Scripts\python.exe" -m pip install --upgrade pip
& "$venv\Scripts\python.exe" -m pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` with database credentials and API keys.

Create the application database from the canonical structure-only schema before
the first run. Import `schema.sql` through MySQL Workbench, or run this from
Command Prompt when the MySQL client is available on `PATH`:

```bat
mysql --host=127.0.0.1 --port=3307 --user=root -p < schema.sql
```

The schema creates the `bakery_ai` database and its 19 runtime tables. It does
not contain operational records or seed data.

Verify the isolated runtime before starting the system:

```powershell
& "$venv\Scripts\python.exe" -m pip check
```

Do not run the project with the system-wide Python interpreter. The launchers use
the isolated Python 3.13 runtime and stop early when installed dependencies are
inconsistent with the project manifest.

Start the full local system:

```powershell
.\start_all.bat
```

Open:

```text
http://127.0.0.1:8002
```

The main server proxies S5 requests from `/s5/*` to the S5 analysis service on port `8001`.

## Offline Desktop Deployment

The final deployment path uses one Docker Compose definition with MySQL 8.4,
the main application, and the S5 service. Windows x64 packages contain native
`linux/amd64` images. Apple Silicon packages contain native `linux/arm64`
images and do not rely on x64 emulation.

The packaged launcher selects an available loopback port, starts the three
containers, waits for `/health` and `/s5-health`, and opens Microsoft Edge in
App Mode. Only the main application port is published. MySQL and S5 remain on
the private Compose network.

Operational data is stored in a named MySQL Docker volume. Closing the desktop
window stops the containers without deleting the volume. Standard uninstall
keeps the database, configuration, and backups. Complete removal is a separate
two-confirmation action that lists every exact file and Docker volume first.

Platform instructions:

- [Final installer entry](docs/deployment/FINAL_INSTALLATION.md)
- [最终安装说明（中文）](docs/deployment/FINAL_INSTALLATION.zh-CN.md)
- [Windows offline installation](docs/deployment/OFFLINE_INSTALL_WINDOWS.md)
- [Apple Silicon offline installation](docs/deployment/OFFLINE_INSTALL_MACOS.md)
- [Upgrade and recovery](docs/deployment/UPGRADE_AND_RECOVERY.md)
- [Deployment acceptance checklist](docs/deployment/ACCEPTANCE_CHECKLIST.md)

The source-tree batch files remain developer fallbacks. They are not the
end-user installation path and are not included in the final offline ZIPs.

### S5 Service

The full dashboard analysis flow requires both servers:

| Script | Purpose |
| --- | --- |
| `start_all.bat` | Starts the S5 service on `127.0.0.1:8001` and the main server on `127.0.0.1:8002` |
| `start.bat` | Starts only the main server with the Python 3.13 project runtime |
| `start_s5.bat` | Starts only the S5 LangGraph service with the Python 3.13 project runtime |

For manual startup, use the same project virtual environment:

```powershell
$venv = Join-Path $env:LOCALAPPDATA "BakeryAI\venv313"
& "$venv\Scripts\python.exe" -m s5_agent.server
& "$venv\Scripts\python.exe" main.py
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
-> module-to-workflow registry
-> LangGraph workflow
-> deterministic specialist agents
-> evidence graph and verification checks
-> constrained summary and recommendations
-> structured JSON response
```

### Module Workflow Map

| Dashboard Module | LangGraph Workflow ID |
| --- | --- |
| revenue | `profit_root_cause` |
| wastage | `wastage_root_cause` |
| forecast | `production_advice` |
| inventory | `inventory_diagnosis` |
| promotion_mix | `promotion_mix_analysis` |

Schedule, attendance, and KPI ranking remain S3 dashboard functions. They do not currently expose S5 AI analysis endpoints.

### LangGraph Workflows

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
- Recommendation records expose their supporting evidence IDs.
- Verification reports surface missing evidence links and data-quality warnings.
- Unsupported modules fail fast instead of falling back to legacy routes.

This prevents silent execution of stale analysis paths.

## Frontend Integration

The frontend uses:

- `S5_API + "/analyze/module"` for module analysis buttons.
- `S5_API + "/priorities"` and `S5_API + "/discounts"` for bundle and discount support.

S5 analysis text is escaped before being inserted into HTML in the dashboard analysis component.

## Local Access Accounts

The runtime never creates tables or seed records during import. `schema.sql` is
the only authoritative database structure, and operational data is restored
separately. The final local database stores BCrypt password hashes for these
local access accounts:

| Username | Password | Role |
| --- | --- | --- |
| `manager` | `BakeryAI@2026` | Manager |
| `staff1` | `Staff@2026` | Staff |

Production mode requires a non-empty `JWT_SECRET` containing at least 32 bytes.

## Key Design Decisions

- The browser UI is the single frontend surface for POS and manager workflows.
- S4 owns the visible dashboard and POS experience.
- S5 owns backend analysis and decision support, exposed through structured JSON APIs.
- Common dashboard tasks use predefined LangGraph workflows instead of open-ended chat planning.
- Current S5 summaries are constrained to verified metrics, evidence, and module-specific rules.
- Deterministic agent outputs keep the dashboard usable without a live LLM dependency.
