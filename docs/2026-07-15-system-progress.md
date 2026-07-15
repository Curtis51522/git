# System Progress - 2026-07-15

## Scope

This note records the current final-submission state of the bakery AI system. The implementation remains a graduation-project system rather than an industrial deployment. S5 is integrated into the manager dashboard while remaining a separately runnable LangGraph analysis service.

## S1 - Product Recognition and Stock Flow

- YOLO11s remains the deployed product-recognition model used by the POS image-scanning workflow.
- Finished-product inflow now records production batches and exposes stock consistently to POS checkout and Inventory BI.
- Bakery checkout deducts finished-product stock only; made-to-order beverage checkout deducts beverage materials.
- Production inflow deducts recipe materials through the shared recipe contract.
- The 30 bakery-product recipe catalog is version controlled with source references, normalized yields, and per-unit ingredients.
- Reconciliation tools and tests cover production quantities, recipe material consumption, and inventory movement.

## S2 - Demand Forecasting

- The deployed forecast remains the selected XGBoost quantile family with Q10, Q50, and Q90 outputs.
- Training and evaluation use chronological data boundaries and held-out historical evaluation rather than random cross-validation.
- Conformal interval calibration and post-processing keep interval output usable for low-volume products.
- New-product and competitor fields remain reserved scenario features and are not silently injected into the deployed 27-feature model.
- Forecast BI reports seven-day total demand, bakery demand, made-to-order beverage demand, projected revenue, model accuracy, average deviation, and interval hit rate.
- Supply coverage and demand gap use bakery demand only because beverages are made to order.
- Current live scope for 2026-07-15 reports 2,471 total forecast units: 2,005 bakery units and 466 beverage units.

## S3 - Scheduling and Production Planning

- Staff scheduling remains demand aware and uses the current production-capacity and attendance contracts.
- Production planning uses the shared product recipes without approximate fallback recipes.
- Material requirements respect tracked and untracked materials, including the explicit handling of recipe water.
- Weekly bake totals, available bakery supply, shortage exposure, waste exposure, revenue, and profit remain aligned with the Forecast BI scope.
- Scheduling, attendance, and Monthly KPI Ranking remain S3 dashboard functions and do not expose unused S5 analysis endpoints.

## S4 - POS, Inventory, Wastage, and Revenue BI

- POS stock validation uses current batch inventory and refreshes after inflow and checkout operations.
- Beverage customization preserves the selected bundle discount while applying temperature, size, ice, and sugar choices.
- Inventory BI includes finished-product snapshots, batch movement, inflow history, raw-material restock history, freshness ratio, and material units.
- Revenue BI includes category mix, dine-in versus takeaway, payment method, seven-day revenue and order trends, hourly revenue and profit, and discarded-product loss.
- Closing unsold bakery loss is deducted once from reported profit; separately recorded material-wastage variance is not deducted again.
- Historical date selection is used consistently by the dashboard and its S5 analysis buttons.

## S5 - LangGraph Multi-Agent Decision Support

- The active S5 runtime uses LangGraph module workflows only; the legacy DAG path is no longer used by dashboard analysis.
- The manager dashboard exposes Revenue, Forecast, Inventory, Wastage, and Promotion and Product Mix analysis without a separate manual chat input.
- Analysis combines module-specific agents, evidence graphs, verification reports, readable risk labels, and evidence-grounded recommendations.
- Forecast analysis separates bakery production demand from made-to-order beverage demand and uses staged-production release rules.
- Revenue and Promotion and Product Mix analysis distinguish traffic, basket size, margin, discount exposure, product mix, and closing loss.
- Inventory analysis combines the selected-date stock snapshot with same-day batch movement without treating a point-in-time stock level as proof of lost sales.
- Wastage analysis separates material-waste records, production yield, and finished-product stock context.
- Live verification on 2026-07-15 returned HTTP 200 and passed the S5 verification report for Revenue, Forecast, Inventory, and Wastage, with no unsupported claims, unsupported recommendations, conflicting claims, missing evidence, or data-quality warnings.

## Verification

- Full automated suite: 459 passed.
- Forecast frontend suite: 28 passed.
- S5 health endpoint reports the LangGraph architecture and the active module registry.
- Live Forecast BI values reconcile: 1,575 planned bakery units cover 78.6% of 2,005 bakery forecast units, leaving a 430-unit gap.
- Generated runtime files, local databases, logs, caches, and local GitNexus metadata are excluded from the submission scope.

## Next Operational Step

Continue the realistic store-operation walkthrough one trading day at a time. Use the visible S2, S3, and S5 outputs as decision support, then record actual POS, production, attendance, inventory, wastage, and closing outcomes through the normal system workflows.
