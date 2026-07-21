# Bakery AI System Operational Validation

## Status

The final-submission worktree and its MySQL 8.4 database passed the historical
and operational data contracts. The system remains scoped as an AI master's
graduation project rather than an industrial deployment. S5 is integrated into
the manager dashboard and remains separately runnable as a LangGraph analysis
service.

## Validated Scope

- Historical sales: 2025-06-24 through 2026-06-23, representing the first year
  after the store renovation and reopening.
- Store operation: 2026-06-24 through 2026-07-24, covering 31 complete trading
  days with Monday trading retained.
- All operational mutations were submitted through visible frontend controls.
  Direct database access was limited to baseline recovery and read-only
  validation.
- The visible replay completed all 4,880 planned actions with zero failed
  actions. The plan included 4,400 completed POS orders, two sick-leave
  adjustments, three shift swaps, three manager attendance corrections, five
  weekly material checks, and two business events.

## S1 - Product Recognition and Stock Flow

- YOLO11s remains the deployed recognition model for POS tray scanning.
- Bakery production creates batch inflow records, deducts recipe materials, and
  exposes finished stock to POS and Inventory BI.
- Bakery checkout deducts finished-product stock only. Made-to-order beverages
  deduct beverage and packaging materials during checkout and never create
  Day-1 stock.
- The final operational period contains 1,952 batch rows and 9,942 finished-stock
  movements. A total of 5,671 bakery units were produced, 5,606 were sold, 30
  were discarded after Day-1, and 35 remained. These quantities reconcile and
  no finished-product balance is negative.

## S2 - Demand Forecasting

- The active historical layer contains 48,987 orders, 83,898 normalized item
  rows, and 86,646 sold units across 365 trading dates.
- Historical sales are limited to 06:00-18:59, use no delivery channel, and
  retain weather, weekday, lag, product, and reserved scenario features.
- Model selection and calibration use chronological boundaries: train through
  2026-01-31, validation from 2026-02-01 to 2026-03-31, and untouched test data
  from 2026-04-01 to 2026-06-23.
- The proposed probabilistic model reports test WAPE 13.4%, MAE 0.7 units,
  82.1% conformal coverage for the 80% target, and 2.3 units average interval
  width. The deterministic XGBoost test MAE is 0.7815 versus 1.1120 for the
  fixed lag baseline.
- The runtime models are refitted on the complete active year only after the
  held-out evaluation artifacts are fixed. The operational replay does not
  alter the frozen historical training period, so another S2 retraining pass is
  not required for this validation.

## S3 - Scheduling, Attendance, and Production Planning

- The operation period contains 292 schedule rows, 203 distinct scheduled
  employee-days, and 200 schedule-backed attendance records.
- Attendance records comprise 173 on-time, 11 late, four early-leave, and 12
  late-and-early-leave outcomes. Three missing scheduled records remain visible
  as absences instead of being converted into attendance.
- Every punch record is linked to a scheduled employee window, and every stored
  status agrees with the punch-in, punch-out, and merged shift boundaries.
- Aggregate attendance is 98.52%. No off-duty employee has an attendance
  record.
- Three manager corrections record the original values, corrected values,
  reason, manager identity, and timestamp. Two sick-leave records have persisted
  replacements, and three shift swaps were completed through the frontend.
- Production plans, staged baking, recipe demand, material readiness, Day-1
  handling, and closing decisions remain connected to the same operating date.

## S4 - POS, Inventory, Wastage, and Revenue BI

- The period contains 4,400 completed orders and 8,060 purchased units,
  averaging 1.83 units per order.
- Revenue is CNY 101,066.70, recorded line profit before closing loss is CNY
  86,820.82, average order value is CNY 22.97, and recorded discounts total CNY
  1,678.80.
- Compared with the preceding 31 completed trading days, orders increased 6.80%,
  revenue increased 12.02%, recorded line profit increased 9.88%, average order
  value increased 4.89%, and purchased units increased 10.71%.
- Profit after CNY 29.70 of closing product loss is CNY 86,791.12, producing an
  85.88% margin compared with 87.58% in the preceding period.
  The evidence therefore supports higher revenue and absolute profit, but it
  does not prove causal uplift or support a claim that every cost-efficiency
  metric improved.
- Service mix is 79.91% takeaway and 20.09% dine-in. Payment mix is 88.14% QR,
  8.14% card, and 3.73% cash, with no delivery orders and no payment
  reconciliation mismatch.
- Every one of the 30 bakery products recorded monthly sales. Product totals
  range from 119 to 416 units; the top three account for 21.57% and the top ten
  account for 47.65%. This creates a visible demand hierarchy without zero-sale
  products or an excessive long tail.
- Bakery sell-through is 98.85%. Discarded Day-1 stock equals 30 units, or 0.53%
  of produced bakery units, with CNY 29.70 recorded loss.
- Material use produced 14,124 outflow rows and 70 alert-driven restock records.
  No material transaction or final balance is negative.
- All 44 tracked materials were checked on each of five Wednesdays. The checks
  contain no negative wastage; 15 positive material-variance rows cost CNY
  10.72. Combined recorded material and finished-product loss is CNY 40.42,
  equal to 0.0400% of operational revenue.
- One legacy material check differed from its reconstructed equation by exactly
  0.001 kg, the configured storage precision. New checks now calculate and store
  from the same quantized values, while the validator permits no more than this
  one-unit legacy rounding tolerance.

## S5 - LangGraph Decision Support

- S5 exposes only Forecast, Revenue, Promotion and Product Mix, Inventory, and
  Wastage analysis through five explicit LangGraph workflows.
- The analysis consumes the same date-scoped BI evidence shown to managers and
  produces verified claims, readable risk labels, evidence links, and actionable
  recommendations.
- Top-3 recommendations produced 5,922 ranked exposures. A total of 280
  selections were linked to 280 completed purchases, representing 4.73% of
  exposures and 6.36% of all orders.
- The two business events persisted through the Forecast frontend. Their event
  windows contain 38 matching discounted receipts, so event monitoring remains
  auditable.

## Verification Evidence

- The operational validator passed every required check.
- The combined historical and operational validator passed every required check.
- The replay utility suite passed 90 tests and six subtests.
- The repository suite passed 666 tests with nine third-party `python-jose`
  deprecation warnings and no test failures.
- The main frontend returned HTTP 200. S5 `/health` returned HTTP
  200 with the LangGraph architecture and its five active modules.
- Validation reports:
  `C:\Users\Curtis\AppData\Local\BakeryAI\reports\2026-07-24-operational-contract.json`
  and
  `C:\Users\Curtis\AppData\Local\BakeryAI\reports\2026-07-24-all-data-contract.json`.
- Replay evidence:
  `C:\Users\Curtis\AppData\Local\BakeryAI\tools\sales-mix-prior-blend-plan.json`,
  `C:\Users\Curtis\AppData\Local\BakeryAI\replay\sales-mix-prior-blend-v2-actions.jsonl`,
  and the segmented `sales-mix-prior-blend-v2-trace` archives.

## Submission Notes

- The replay utility, action journal, and browser trace remain outside the
  repository and are not part of the submission source code.
- Runtime databases, credentials, logs, caches, and local evidence are excluded
  from the repository.
- No commit or push was performed during this validation pass.
