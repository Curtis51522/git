# S5 Promotion and Product Mix Analysis Design

## Purpose

Promotion and Product Mix Analysis should be a lightweight decision extension of the Revenue dashboard, not a standalone business intelligence module. The goal is to help the store manager decide how to sell next after the revenue result is known.

The feature should answer three practical questions:

- Did discount activity damage profit quality?
- Is revenue too dependent on a small set of products?
- Should the next action be no intervention, targeted bundles, clearance promotion, or margin-protecting product pairing?

This keeps the module commercially useful while preserving the current S5 design principle: S5 is embedded inside dashboard workflows and does not become a separate manual chat surface.

## Scope

In scope:

- Add a new S5 module identifier named `promotion_mix`.
- Reuse existing revenue, promotion, product mix, discount, and Top-3 bundle evidence where possible.
- Produce evidence-grounded recommendations for discount quality and product-mix decisions.
- Add a Revenue dashboard entry point beside the existing Revenue AI Analysis control.
- Keep all generated text in English.

Out of scope:

- A full campaign management system.
- Manual S5 prompt input for promotion planning.
- Re-training the deployed S2 forecast model.
- Claiming causal uplift from business events without enough historical event data.
- Replacing the existing Revenue AI Analysis.

## Interface Design

The backend should continue using the existing unified S5 endpoint:

```http
POST /analyze/module
```

Request body:

```json
{
  "module": "promotion_mix",
  "date": "2026-06-30",
  "lang": "en",
  "force_refresh": true
}
```

The endpoint should route `promotion_mix` to a dedicated LangGraph workflow. This keeps the external API consistent with the existing Revenue, Forecast, Inventory, and Wastage AI analysis flows.

## Frontend Placement

The frontend entry point should live on the Revenue dashboard.

Recommended layout:

```text
Revenue AI Analysis
Promotion & Product Mix AI
```

The new button should call the existing frontend helper:

```javascript
runModuleS5Analysis('promotion_mix', 'rev-date', 'rev-promotion-mix-s5-result')
```

The result container should be separate from the main Revenue AI result container so the user can compare the two analyses without overwriting either one.

## LangGraph Workflow

Recommended graph:

```text
promotion_signal
  -> product_mix
  -> discount_impact
  -> bundle_opportunity
  -> evidence
  -> verify
  -> synthesize
```

Node responsibilities:

- `promotion_signal`: Read active discount exposure, active business events, and discount source context.
- `product_mix`: Analyze product ranking, top-product concentration, top-three concentration, and bakery-versus-beverage mix.
- `discount_impact`: Evaluate whether discounts are low, controlled, margin-eroding, or unsupported by revenue movement.
- `bundle_opportunity`: Identify whether targeted bundles are more suitable than broad discounts.
- `evidence`: Build traceable evidence links for all claims and recommendations.
- `verify`: Reject unsupported claims, especially causal claims about promotion uplift.
- `synthesize`: Produce a concise business-facing summary and recommendations.

## Reused Agents and Data Sources

Existing agents should be reused before adding new ones:

- `PromoAgent`
- `ProductMixAgent`
- Revenue discount evidence from the existing Revenue analysis workflow
- Revenue product-mix evidence from the existing Revenue analysis workflow
- Top-3 bundle context from the S4 recommendation flow where available
- Business Events context from the S2 scenario layer where available

The first implementation should avoid introducing a new database table. It should read from existing dashboard-compatible data sources so that AI numbers match Revenue BI charts.

## Evidence Fields

Minimum evidence fields:

- `revenue`
- `profit`
- `profit_margin_pct`
- `order_count`
- `average_order_value`
- `discount_total`
- `discount_rate_pct`
- `top_product_name`
- `top_product_revenue_share_pct`
- `top3_product_revenue_share_pct`
- `bread_revenue_share_pct`
- `beverage_revenue_share_pct`
- `active_business_event_count`
- `bundle_candidate_count`

Optional evidence fields:

- `revenue_change_pct`
- `order_change_pct`
- `average_order_value_change_pct`
- `inventory_pressure_products`
- `day1_stock_products`
- `low_stock_products`

## Recommendation Rules

The module should prefer no intervention when the evidence does not justify action.

Recommended rule set:

- If discount exposure is low and revenue quality is stable, recommend no broad promotion.
- If order count falls but average order value rises, recommend monitoring traffic before broad discounting.
- If top-three concentration is high, recommend targeted support for mid-tier products.
- If discount exposure is high and margin is weak, recommend reviewing promotion rules before repeating the same discount pattern.
- If Day-1 or high-inventory products are available, recommend targeted bundles instead of store-wide discounts.
- If business events are active, recommend tracking event products separately during the next trading window.
- If product mix data is missing, recommend data verification instead of promotion action.

The module should not recommend broad discounts by default. Broad discounts should require strong evidence of traffic weakness, controlled margin impact, and product-level fit.

## Output Structure

The summary should follow a business-readable structure:

1. Overall promotion and product-mix judgment.
2. Discount quality and margin impact.
3. Product concentration and category mix.
4. Bundle or promotion opportunity.
5. Data limitations if any.

Example style:

```text
Revenue quality is stable, so a broad promotion is not justified yet. Discount exposure is low at 2.4% of revenue, while margin remains above the operating threshold. Product mix is moderately concentrated: the top product contributes 19.8% of tracked revenue and the top three products contribute 41.7%. A targeted bundle is safer than a store-wide discount because it can support weaker traffic without sacrificing margin across all products.
```

Recommendations should be concrete:

- `Do not launch a broad discount today.`
- `Test a targeted bundle around the leading bakery item and one beverage.`
- `Monitor traffic for 2-3 trading days before changing price rules.`
- `Use clearance-style promotion only for products with inventory pressure.`

## Boundary With Other S5 Modules

Revenue AI Analysis:

- Explains the financial result of the selected day.
- Promotion Mix Analysis explains what sales action should follow from that result.

Forecast AI Analysis:

- Explains future demand, production coverage, and staged bake decisions.
- Promotion Mix Analysis may reference forecast or business event context, but it should not recalculate demand forecasts.

Inventory AI Analysis:

- Explains stock risk and inventory data reliability.
- Promotion Mix Analysis may use inventory pressure as a recommendation signal, but it should not diagnose stock records.

Wastage AI Analysis:

- Explains material waste and production-yield evidence.
- Promotion Mix Analysis may reference clearance opportunity only when finished-product stock pressure is available.

## Testing Plan

Backend tests:

- `promotion_mix` routes through `/analyze/module`.
- The LangGraph workflow returns a verified S5 response.
- The response includes evidence for discount and product-mix claims.
- Missing product-mix data produces a data-quality recommendation, not a fake promotion recommendation.
- High discount exposure can trigger a margin-protection recommendation.
- Stable revenue and low discount exposure can trigger a no-intervention recommendation.

Frontend tests:

- Revenue dashboard contains the `Promotion & Product Mix AI` button.
- The button calls `runModuleS5Analysis('promotion_mix', 'rev-date', 'rev-promotion-mix-s5-result')`.
- The result container does not overwrite the existing Revenue AI result.

Compatibility tests:

- Existing Revenue AI Analysis still works.
- Existing Forecast, Inventory, and Wastage module analysis still work.
- Existing Business Events and Top-3 recommendation flows are not broken.

## Academic Value

This feature strengthens S5 by moving from descriptive analytics to decision support. Revenue analysis already explains what happened; Promotion and Product Mix Analysis explains what action is justified by the evidence.

The academic contribution is not a new forecasting model. The contribution is an evidence-grounded, multi-agent decision layer that separates:

- observed financial performance,
- product-mix structure,
- discount exposure,
- scenario context,
- and recommended commercial action.

This makes the S5 module more defensible as a decision intelligence component rather than a simple dashboard explanation tool.

## Implementation Order

1. Add `promotion_mix` routing to the S5 module endpoint.
2. Add a LangGraph workflow for promotion and product mix analysis.
3. Reuse existing `PromoAgent` and `ProductMixAgent` where practical.
4. Add synthesis and recommendation rules.
5. Add backend tests.
6. Add the Revenue dashboard button and result container.
7. Add frontend static tests.
8. Run full regression tests.

