# System Progress - 2026-07-24

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
- The active historical layer now covers 2025-06-24 through 2026-06-23, with 48,987 orders and 83,898 normalized item rows across 365 trading dates.
- The held-out probabilistic test reports WAPE 13.4%, MAE 0.7 units, 82.1% conformal coverage for the 80% target, and 2.3 units average interval width.

## S3 - Scheduling and Production Planning

- Staff scheduling remains demand aware and uses the current production-capacity and attendance contracts.
- Production planning uses the shared product recipes without approximate fallback recipes.
- Material requirements respect tracked and untracked materials, including the explicit handling of recipe water.
- Weekly bake totals, available bakery supply, shortage exposure, waste exposure, revenue, and profit remain aligned with the Forecast BI scope.
- Scheduling, attendance, and Monthly KPI Ranking remain S3 dashboard functions and do not expose unused S5 analysis endpoints.
- `shift_schedule` is now the attendance source of truth: unscheduled employees cannot punch and are excluded from current and historical attendance views.
- Daily schedule slots are resolved into one employee working window, including merged double shifts, and all displayed punch times use zero-padded `HH:MM` values.
- Attendance status now evaluates both boundaries and supports `Scheduled`, `On Time`, `Late`, `Early Leave`, and `Late & Early Leave` without a fallback shift time.
- Monthly attendance reporting now shows attended versus completed scheduled days, Attendance Rate, On-Time Rate, Shift Completion Rate, late count, early-leave count, and schedule-overlap work hours.
- Monthly KPI Ranking uses the same schedule-based calculation and includes Shift Completion as an objective Internal Process metric.
- The completed 2026-06-24 through 2026-07-24 operation period contains 270 schedule rows and 200 schedule-backed attendance records: 173 on-time, 13 late, 3 early-leave, and 11 late-and-early-leave outcomes. Missing scheduled records remain absences.

## S4 - POS, Inventory, Wastage, and Revenue BI

- POS stock validation uses current batch inventory and refreshes after inflow and checkout operations.
- Beverage customization preserves the selected bundle discount while applying temperature, size, ice, and sugar choices.
- Inventory BI now distinguishes the current live finished-product balance from a historical end-of-day closing balance.
- Historical bread stock is reconstructed from inventory transactions at the selected date boundary instead of reading the present batch balance.
- Baked Product Stock Records reconcile opening stock plus baked units against sold, discarded, other outflow, Day-1 carryover, and closing stock while retaining batch IDs.
- A genuine zero closing balance is shown explicitly as no bread stock remaining at close instead of appearing as a blank or missing chart.
- Inventory BI also includes raw-material restock history, freshness ratio, and normalized material units.
- Revenue BI includes category mix, dine-in versus takeaway, payment method, seven-day revenue and order trends, hourly revenue and profit, and discarded-product loss.
- Closing unsold bakery loss is deducted once from reported profit; separately recorded material-wastage variance is not deducted again.
- Historical date selection is used consistently by the dashboard and its S5 analysis buttons.

### Final Operation Closure - 2026-07-24

- The visible frontend replay completed all 4,547 planned actions across 31 trading days, including 4,075 checkouts and five Wednesday material checks.
- The period records 7,131 purchased units, CNY 89,458.40 revenue, CNY 76,873.80 profit, CNY 21.95 average order value, and 1.750 units per order.
- Service mix is 79.90% takeaway and 20.10% dine-in; payment mix is 88.10% QR, 8.12% card, and 3.78% cash.
- All finished-product and raw-material balances remain nonnegative, weekly material checks contain realistic small variance after the baseline week, and revenue, inventory, wastage, attendance, payment, and business-event evidence reconcile.

### Inventory History Repair - 2026-07-16

- The generated operation history from 2026-07-06 through 2026-07-13 previously assigned 89 next-day Day-1 discard transactions to the preceding closing timestamp, which forced otherwise valid closing stock to zero.
- The same transaction IDs and 106 discarded units were retained, but their timestamps were moved to the actual next trading day and their reason was normalized from `closing_unsold` to `day1_unsold`.
- Revenue totals and period loss totals were not changed; only the operational date attribution and stock carryover sequence were corrected.
- The repaired history reconciles opening, baked, sold, discarded, and closing quantities for every affected date. A zero closing balance on 2026-07-10 remains a valid operational result.
- A pre-repair SQL dump is stored outside the repository at `C:\Users\Curtis\AppData\Local\BakeryAI\backups\bakery_ai_before_inventory_history_repair_20260716_222219.sql` with SHA-256 `D47120B5F59328C4243BEF6159BF55D30987BEAEB24F4E4A8420B2A443BA0A58`.

### Expired Inventory Audit Closure - 2026-07-17

- Finished-product age is now exact: production-day stock is Fresh, previous-day stock is Day-1, and stock two or more calendar days old is Expired.
- Expiration records a discarded `day1_unsold` outflow, clears both active quantity fields, and retains the batch as a zero-balance Expired audit row instead of deleting it.
- Inventory BI excludes overdue positive balances from sellable stock, exposes them separately as a conditional data-quality warning, and passes the same evidence to S5 Inventory analysis.
- The previous destructive expiration path had removed 30 batch rows: 4 from 2026-07-14 and 26 from 2026-07-15. All 30 were reconstructed from complete inflow and outflow transactions as zero-balance Expired audit rows; no sellable stock was restored.
- The repaired lifecycle reconciles to 45 units at the 2026-07-14 close, 29 Fresh units at the 2026-07-15 close, 29 Day-1 units on 2026-07-16, and 0 sellable units after the 29-unit expiration outflow on 2026-07-17.
- The pre-repair application-level SQL dump is stored outside the repository at `C:\Users\Curtis\AppData\Local\BakeryAI\backups\bakery_ai_before_expired_batch_audit_repair_20260717_004952.sql` with SHA-256 `3785AE48AC08987E666A87B31274BDF68574A2E6D8A4CC61FB740526222D9700`.

## S5 - LangGraph Multi-Agent Decision Support

- The active S5 runtime uses five explicit Python LangGraph workflows only; the legacy DAG and unused YAML-template paths have been removed.
- The manager dashboard exposes Revenue, Forecast, Inventory, Wastage, and Promotion and Product Mix analysis without a separate manual chat input.
- Analysis combines module-specific agents, evidence graphs, verification reports, readable risk labels, and evidence-grounded recommendations.
- Forecast analysis separates bakery production demand from made-to-order beverage demand and uses staged-production release rules.
- Revenue and Promotion and Product Mix analysis distinguish traffic, basket size, margin, discount exposure, product mix, and closing loss.
- Inventory analysis combines the selected-date stock snapshot with same-day batch movement without treating a point-in-time stock level as proof of lost sales.
- Inventory analysis uses opening-aware movement reconciliation so historical carryover stock is included in sell-through and data-quality checks.
- Wastage analysis separates material-waste records, production yield, and finished-product stock context.
- Historical Inventory analysis no longer falls back to current stock when a selected-date dashboard request is unavailable.
- Revenue analysis cache entries retain their effective analysis date, so historical analysis cannot drive current POS priorities or discounts.
- The POS recommendation path now consumes structured RecommendationAgent metadata instead of duplicating private recommendation rules in the API server.
- The duplicate legacy agent base and the unused standalone production graph builder have been removed.
- Final runtime verification returned HTTP 200 and passed the data contracts supporting Revenue, Forecast, Inventory, Wastage, and Promotion and Product Mix analysis.

## Verification

- Full automated suite under the configured Python 3.13 runtime: 659 passed with 9 third-party `python-jose` deprecation warnings.
- External replay utility suite: 77 passed.
- Operational and combined historical/operational data-contract reports both passed every required check.
- Attendance integrity audit for 2026-06-24 through 2026-07-24: 270 schedule rows, 200 schedule-linked attendance rows, zero unscheduled punch records, zero invalid punch windows, and zero status mismatches.
- S5 health endpoint reports the LangGraph architecture and the active module registry.
- Live Forecast BI values reconcile: 1,575 planned bakery units cover 78.6% of 2,005 bakery forecast units, leaving a 430-unit gap.
- Live Inventory verification returned HTTP 200 for the 2026-07-09 historical close with 11 units remaining and for the valid 2026-07-10 zero close; all repaired movement rows passed balance checks.
- Browser verification confirmed dynamic current-versus-closing labels, opening-aware movement columns, and an explicit zero-closing-stock state.
- Generated runtime files, local databases, logs, caches, and local GitNexus metadata are excluded from the submission scope.

## Next Step

Freeze the validated data and code scope, complete the final thesis and presentation narrative, and avoid further functional expansion unless final acceptance testing identifies a reproducible defect.
