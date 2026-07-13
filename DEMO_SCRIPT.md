# Bakery AI System Demo Script

Audience: bakery owner, store manager, or project evaluator.

Goal: show the complete operational workflow from login, POS checkout, forecasting, scheduling, inventory tracking, and S5 module-based analysis.

## 1. Opening

Welcome to the Bakery AI System. This system supports a medium-sized bakery-cafe by connecting product recognition, POS checkout, inventory tracking, sales forecasting, staff scheduling, and manager-level analysis in one browser-based workflow.

The key point is that the system does not treat AI as a black box. S1 to S4 handle operational data and user interaction, while S5 provides backend analysis through structured multi-agent workflows.

## 2. Login And Roles

Open the web application at:

```text
http://127.0.0.1:8002
```

Log in as the manager account to show the full dashboard. Explain that staff accounts focus on POS workflows, while manager accounts can access revenue, forecast, schedule, inventory, and S5 analysis features.

Main message:

- Staff users get a fast checkout-oriented interface.
- Manager users get additional decision-support panels.
- Role separation keeps daily operations simple while protecting manager-only analysis.

## 3. POS Checkout

Open the POS panel.

Show the product recognition area, cart, bread buttons, beverage buttons, and checkout workflow.

Explain the operational flow:

1. A tray image can be uploaded for product recognition.
2. Recognized products can be confirmed, edited, or removed.
3. Manual bread and beverage buttons remain available for fast checkout.
4. Cart items show quantity, price, and freshness-based discounts.
5. Checkout deducts inventory and records the transaction.

Main message:

The POS workflow keeps a human in control. AI recognition speeds up entry, but the cashier can correct every item before checkout.

## 4. Bundle Recommendation

Add one or more items to the cart, then generate the Top-3 bundle recommendations.

Explain that S4 calculates checkout recommendations using product prices, freshness status, cart context, and S5 priority signals when available.

Main message:

The recommendation card is designed for daily retail use. It helps staff suggest pairings without requiring them to manually inspect inventory pressure or margin context.

## 5. Fresh Batch Inflow

Show the fresh batch inflow area.

Explain:

- Newly baked products can be added into inventory.
- Freshness status is tracked at batch level.
- Day-1 products can receive automatic discounts.
- The system records logical inventory status, while real product movement is still handled by staff.

Main message:

Inventory is not just a number. The system tracks freshness and batch status so that POS, discounting, and wastage analysis all use the same operational data.

## 6. Forecast Panel

Open the forecast panel.

Show the seven-day product forecast and the S5 analysis button for the forecast module.

Explain:

- S2 provides demand forecast data.
- S5 reads the forecast context through the `production_advice` template.
- The analysis summarizes demand outlook, uncertainty, material readiness, production feasibility, and forecast reliability.

Main message:

Forecasting is connected to action. The manager does not only see predicted demand; S5 turns the forecast into production and procurement guidance.

## 7. Schedule Panel

Open the schedule panel.

Show staff shifts, demand levels, and schedule controls.

Explain:

- S3 uses demand-aware scheduling logic.
- The schedule panel shows staffing coverage and role allocation.
- Schedule and attendance remain S3 operational functions and do not expose S5 analysis buttons.

Main message:

Scheduling is tied to demand. The manager can inspect whether staffing matches expected store activity without adding an unnecessary AI layer to routine attendance data.

## 8. Inventory And Wastage

Open the inventory panel.

Show batch inventory, freshness status, and wastage history if available.

Use the S5 analysis button for inventory or wastage.

Explain:

- Inventory diagnosis uses `inventory_diagnosis`.
- Wastage analysis uses `wastage_root_cause`.
- S5 combines product stock, material stock, wastage, production, and yield signals depending on the selected module.

Main message:

The manager can move from raw stock records to a direct explanation of inventory pressure and wastage risk.

## 9. Revenue And KPI Analysis

Open the revenue panel.

Show daily revenue cards, charts, and the S5 analysis button.

Explain:

- Revenue analysis uses the `profit_root_cause` template.
- Monthly KPI ranking remains an S3 dashboard function and does not expose an S5 analysis endpoint.
- S5 runs deterministic specialist agents through a LangGraph workflow and returns a structured summary, evidence, risks, and recommendations.

Main message:

The revenue dashboard does more than report numbers. Its S5 workflow explains why results changed and which evidence-backed actions matter next, while KPI ranking remains a direct operational comparison.

## 10. S5 Architecture Explanation

Use this short explanation when asked how S5 works:

S5 is a backend analysis service. It exposes a constrained `/analyze/module` endpoint for the supported dashboard modules. Each module maps to a predefined LangGraph workflow. Deterministic specialist agents produce findings from dashboard-aligned data and business rules, the verifier checks evidence coverage, and the synthesizer returns a concise manager-facing summary and recommendation list.

This design avoids relying on a free-form planner for every request. The system uses structured workflows for common operations and remains usable without a live LLM dependency.

## 11. Current S5 Endpoints

Show these endpoints only if the audience is technical:

| Endpoint | Purpose |
| --- | --- |
| `/s5/health` | Service status |
| `/s5/analyze/module` | Dashboard module analysis |
| `/s5/priorities` | Bundle priority signals |
| `/s5/discounts` | Dynamic discount signals |

## 12. Closing

End with this message:

The system connects daily bakery operations into one loop. S1 captures product activity, S2 forecasts demand, S3 plans staffing, S4 provides the usable web interface, and S5 turns cross-module data into manager-ready analysis. The result is a practical system for reducing waste, improving checkout efficiency, and making store decisions from traceable data.
