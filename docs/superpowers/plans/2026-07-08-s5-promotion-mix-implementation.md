# S5 Promotion Mix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight Revenue-dashboard S5 analysis module that evaluates promotion quality, product-mix concentration, and targeted bundle opportunities without creating a separate BI campaign system.

**Architecture:** The implementation extends the existing `/analyze/module` contract with `module: "promotion_mix"`. A dedicated LangGraph workflow reuses existing promotion, revenue product-mix, discount-impact, and optional bundle-context evidence, then synthesizes a verified business-facing recommendation. The frontend adds a Revenue dashboard button with a separate result container so the existing Revenue AI Analysis remains unchanged.

**Tech Stack:** Python, FastAPI, Pydantic, LangGraph, pytest, JavaScript, existing S5 evidence and verification schemas.

---

## Implementation Constraints

- Do not auto commit changes unless the user explicitly asks.
- Keep generated code and documents in English.
- Prefer the Python standard library when it is enough.
- Use `apply_patch` for manual file edits.
- Do not use PowerShell to write source code directly.
- After implementation, scan touched files for stale comments, duplicate blocks, and unused leftovers.

## File Structure

- Modify `.gitignore`
  - Keep the new implementation plan trackable while preserving the general `docs/*` ignore rule.
- Create `docs/superpowers/plans/2026-07-08-s5-promotion-mix-implementation.md`
  - This implementation plan.
- Modify `s5_agent/server.py`
  - Add `promotion_mix` to `LANGGRAPH_MODULES`.
- Modify `s5_agent/graph/registry.py`
  - Map `promotion_mix` to a new graph template id, recommended as `promotion_mix_analysis`.
- Modify `s5_agent/graph/builder.py`
  - Add promotion-mix graph nodes, synthesis, and recommendations.
- Modify `s5_agent/agents/promo.py`
  - Add structured metrics/evidence output that fits LangGraph agent outputs.
- Modify `s5_agent/agents/product_mix.py`
  - Add structured metrics/evidence output for product concentration and category mix.
- Modify `api/module4_frontend/static/index.html`
  - Add Revenue dashboard button and result container.
- Modify `api/module4_frontend/static/s5_analysis.js`
  - Add display labels for new evidence and risks if needed.
- Create `tests/test_s5_langgraph_promotion_mix.py`
  - Covers graph output, recommendations, no-action behavior, and data-gap behavior.
- Modify `tests/test_s5_api_compatibility.py`
  - Covers `/analyze/module` routing for `promotion_mix`.
- Modify `tests/test_frontend_business_events_static.py`
  - Covers the Revenue dashboard button and result container.

## Task 1: Route `promotion_mix` Through the Existing S5 Module Endpoint

**Files:**
- Modify: `s5_agent/server.py`
- Modify: `s5_agent/graph/registry.py`
- Test: `tests/test_s5_api_compatibility.py`

- [ ] **Step 1: Write the failing API compatibility test**

Add this test near the existing module routing tests:

```python
def test_promotion_mix_module_uses_langgraph_route():
    from s5_agent.server import LANGGRAPH_MODULES
    from s5_agent.graph.registry import module_to_template

    assert "promotion_mix" in LANGGRAPH_MODULES
    assert module_to_template("promotion_mix") == "promotion_mix_analysis"
```

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
python -m pytest tests/test_s5_api_compatibility.py::test_promotion_mix_module_uses_langgraph_route -q
```

Expected: FAIL because `promotion_mix` is not routed yet.

- [ ] **Step 3: Add the module to `LANGGRAPH_MODULES`**

In `s5_agent/server.py`, change:

```python
LANGGRAPH_MODULES = {"inventory", "revenue", "forecast", "wastage"}
```

to:

```python
LANGGRAPH_MODULES = {"inventory", "revenue", "forecast", "wastage", "promotion_mix"}
```

- [ ] **Step 4: Add the registry mapping**

In `s5_agent/graph/registry.py`, add:

```python
"promotion_mix": "promotion_mix_analysis",
```

inside the module-to-template mapping used by `module_to_template`.

- [ ] **Step 5: Run the focused test and verify it passes**

Run:

```bash
python -m pytest tests/test_s5_api_compatibility.py::test_promotion_mix_module_uses_langgraph_route -q
```

Expected: PASS.

## Task 2: Add Structured Promotion Signal Output

**Files:**
- Modify: `s5_agent/agents/promo.py`
- Test: `tests/test_s5_langgraph_promotion_mix.py`

- [ ] **Step 1: Write a focused unit test for promotion metrics**

Create `tests/test_s5_langgraph_promotion_mix.py` with this first test:

```python
import pytest

from s5_agent.agents.promo import PromoAgent


def test_promo_agent_structures_discount_metrics():
    agent = PromoAgent("PromoAgent")
    raw = {
        "today_revenue": 1000,
        "today_discount": 80,
        "discount_rate": 0.08,
    }

    output = agent.analyze(raw, {"date": "2026-06-30"})

    assert output.agent == "PromoAgent"
    assert output.confidence >= 0.6
    assert output.attribution["metric"] == "discount_rate"
    assert output.attribution["deviation"] == pytest.approx(8.0)
```

- [ ] **Step 2: Run the focused test**

Run:

```bash
python -m pytest tests/test_s5_langgraph_promotion_mix.py::test_promo_agent_structures_discount_metrics -q
```

Expected: PASS or FAIL depending on the current output shape. If it passes, keep the test and proceed to structured LangGraph output in Task 4.

- [ ] **Step 3: Keep implementation minimal**

If the test fails because the raw payload shape differs, normalize revenue fields in `PromoAgent.analyze()`:

```python
api_data = raw.get("data", {}) if isinstance(raw, dict) and "data" in raw else raw
data = api_data.get("data", api_data) if isinstance(api_data, dict) else {}
discount_total = float(data.get("today_discount", data.get("discount_total", 0)) or 0)
revenue = float(data.get("today_revenue", data.get("revenue", 0)) or 0)
discount_rate = float(data.get("discount_rate", 0) or 0)
if not discount_rate and revenue > 0:
    discount_rate = discount_total / revenue
```

- [ ] **Step 4: Run the focused test again**

Run:

```bash
python -m pytest tests/test_s5_langgraph_promotion_mix.py::test_promo_agent_structures_discount_metrics -q
```

Expected: PASS.

## Task 3: Add Product Mix Concentration Coverage

**Files:**
- Modify: `s5_agent/agents/product_mix.py`
- Test: `tests/test_s5_langgraph_promotion_mix.py`

- [ ] **Step 1: Add a product-mix concentration test**

Add this test to `tests/test_s5_langgraph_promotion_mix.py`:

```python
from s5_agent.agents.product_mix import ProductMixAgent


def test_product_mix_agent_detects_top3_concentration():
    agent = ProductMixAgent("ProductMixAgent")
    raw = {
        "bread_ranking": [
            {"name": "macaron", "qty": 30, "revenue": 300, "profit": 240},
            {"name": "baguette", "qty": 20, "revenue": 200, "profit": 140},
            {"name": "croissant", "qty": 10, "revenue": 100, "profit": 70},
            {"name": "donut", "qty": 10, "revenue": 80, "profit": 48},
        ],
        "beverage_ranking": [
            {"name": "latte", "qty": 15, "revenue": 270, "profit": 180},
        ],
        "category": {"Bread": 680, "Beverages": 270},
        "total_bread_sku": 20,
        "week_avg": {"bread_avg": []},
    }

    output = agent.analyze(raw, {"date": "2026-06-30"})

    assert output.agent == "ProductMixAgent"
    assert output.attribution["metric"] == "product_mix"
    assert output.confidence >= 0.7
    assert "Top 3" in output.opinion
```

- [ ] **Step 2: Run the focused test**

Run:

```bash
python -m pytest tests/test_s5_langgraph_promotion_mix.py::test_product_mix_agent_detects_top3_concentration -q
```

Expected: PASS if the existing agent already reports top-three concentration.

- [ ] **Step 3: Add missing concentration text only if needed**

If the test fails because the message does not include top-three concentration, add a concise concentration sentence in `ProductMixAgent.analyze()` after `top3_pct` is computed:

```python
parts.append(f"Top 3 breads = {top3_pct:.0f}% of bread revenue")
```

- [ ] **Step 4: Run the focused test again**

Run:

```bash
python -m pytest tests/test_s5_langgraph_promotion_mix.py::test_product_mix_agent_detects_top3_concentration -q
```

Expected: PASS.

## Task 4: Build the Promotion Mix LangGraph Workflow

**Files:**
- Modify: `s5_agent/graph/builder.py`
- Test: `tests/test_s5_langgraph_promotion_mix.py`

- [ ] **Step 1: Add a graph-level response test**

Add this test:

```python
import pytest

from s5_agent.graph.runner import run_s5_graph
from s5_agent.graph.state import S5Request


@pytest.mark.asyncio
async def test_promotion_mix_graph_returns_verified_response():
    response = await run_s5_graph(
        "promotion_mix_analysis",
        S5Request(
            query="promotion_mix",
            module="promotion_mix",
            params={"date": "2026-06-30", "module": "promotion_mix"},
            lang="en",
            force_refresh=True,
        ),
        raw_inputs={
            "promotion_signal": {"today_revenue": 1000, "today_discount": 0, "discount_rate": 0},
            "product_mix": {
                "bread_ranking": [
                    {"name": "macaron", "qty": 30, "revenue": 300, "profit": 240},
                    {"name": "baguette", "qty": 20, "revenue": 200, "profit": 140},
                    {"name": "croissant", "qty": 10, "revenue": 100, "profit": 70},
                ],
                "beverage_ranking": [{"name": "latte", "qty": 15, "revenue": 270, "profit": 180}],
                "category": {"Bread": 600, "Beverages": 270},
                "total_bread_sku": 20,
                "week_avg": {"bread_avg": []},
            },
        },
    )

    assert response.verification_report.passed is True
    assert response.summary
    assert any(output.agent_name in {"PromotionSignalAgent", "PromotionProductMixAgent"} for output in response.agent_outputs)
```

- [ ] **Step 2: Run the graph test and verify failure**

Run:

```bash
python -m pytest tests/test_s5_langgraph_promotion_mix.py::test_promotion_mix_graph_returns_verified_response -q
```

Expected: FAIL because `promotion_mix_analysis` is not supported yet.

- [ ] **Step 3: Add `promotion_mix_analysis` to supported templates**

In `s5_agent/graph/builder.py`, add `promotion_mix_analysis` to `SUPPORTED_GRAPH_TEMPLATES`.

Expected shape:

```python
SUPPORTED_GRAPH_TEMPLATES = {
    "inventory_diagnosis",
    "profit_root_cause",
    "production_advice",
    "wastage_root_cause",
    "promotion_mix_analysis",
}
```

- [ ] **Step 4: Add graph nodes**

Add minimal node functions that return `AgentOutput` objects. Follow the existing `_revenue_product_mix_node()` and `_discount_impact_node()` patterns.

Recommended node names:

```python
async def _promotion_signal_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    raw = graph_state.raw_inputs.get("promotion_signal", {})
    output = _agent_output_from_opinion(
        "PromotionSignalAgent",
        PromoAgent("PromotionSignalAgent").analyze(raw, graph_state.request.params),
    )
    graph_state.agent_outputs["promotion_signal"] = output
    return {"agent_outputs": graph_state.agent_outputs}


async def _promotion_product_mix_node(state: S5GraphState | dict[str, Any]) -> dict[str, Any]:
    graph_state = _normalize_state(state)
    raw = graph_state.raw_inputs.get("product_mix", {})
    output = _agent_output_from_opinion(
        "PromotionProductMixAgent",
        ProductMixAgent("PromotionProductMixAgent").analyze(raw, graph_state.request.params),
    )
    graph_state.agent_outputs["promotion_product_mix"] = output
    return {"agent_outputs": graph_state.agent_outputs}
```

- [ ] **Step 5: Add the graph builder**

Add:

```python
def build_promotion_mix_graph():
    graph = StateGraph(S5GraphState)
    graph.add_node("promotion_signal", _promotion_signal_node)
    graph.add_node("promotion_product_mix", _promotion_product_mix_node)
    graph.add_node("evidence", _evidence_node)
    graph.add_node("verify", _verify_node)
    graph.add_node("synthesize", _synthesize_node)

    graph.set_entry_point("promotion_signal")
    graph.add_edge("promotion_signal", "promotion_product_mix")
    graph.add_edge("promotion_product_mix", "evidence")
    graph.add_edge("evidence", "verify")
    graph.add_edge("verify", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()
```

- [ ] **Step 6: Route the template**

In `build_s5_graph()`, add:

```python
if template_id == "promotion_mix_analysis":
    return build_promotion_mix_graph()
```

- [ ] **Step 7: Run the graph test**

Run:

```bash
python -m pytest tests/test_s5_langgraph_promotion_mix.py::test_promotion_mix_graph_returns_verified_response -q
```

Expected: PASS.

## Task 5: Add Promotion Mix Synthesis and Recommendation Rules

**Files:**
- Modify: `s5_agent/graph/builder.py`
- Test: `tests/test_s5_langgraph_promotion_mix.py`

- [ ] **Step 1: Add a no-intervention test**

Add:

```python
@pytest.mark.asyncio
async def test_promotion_mix_recommends_no_broad_discount_when_stable():
    response = await run_s5_graph(
        "promotion_mix_analysis",
        S5Request(
            query="promotion_mix",
            module="promotion_mix",
            params={"date": "2026-06-30"},
            lang="en",
            force_refresh=True,
        ),
        raw_inputs={
            "promotion_signal": {"today_revenue": 2984, "today_discount": 0, "discount_rate": 0},
            "product_mix": {
                "bread_ranking": [
                    {"name": "macaron", "qty": 59, "revenue": 590, "profit": 470},
                    {"name": "baguette", "qty": 20, "revenue": 350, "profit": 250},
                    {"name": "croissant", "qty": 18, "revenue": 304, "profit": 210},
                ],
                "beverage_ranking": [{"name": "latte", "qty": 15, "revenue": 270, "profit": 180}],
                "category": {"Bread": 2324, "Beverages": 660},
                "total_bread_sku": 20,
                "week_avg": {"bread_avg": []},
            },
        },
    )

    assert "broad" in response.summary.lower()
    assert any("broad discount" in rec.action.lower() for rec in response.recommendations)
```

- [ ] **Step 2: Add synthesis helper**

Add a helper such as:

```python
def _synthesize_promotion_mix_summary(outputs: dict[str, Any]) -> str:
    promo = outputs.get("promotion_signal")
    product_mix = outputs.get("promotion_product_mix")
    sentences = []
    if promo:
        sentences.append(promo.claim)
    if product_mix:
        sentences.append(product_mix.claim)
    if not sentences:
        return "Promotion and product-mix analysis could not be completed because supporting dashboard data is missing."
    sentences.append("Promotion action should be evidence-led: broad discounts require clear traffic weakness and acceptable margin impact.")
    return " ".join(sentences)
```

- [ ] **Step 3: Add recommendation helper**

Add:

```python
def _promotion_mix_recommendations(outputs: dict[str, Any]) -> list[Any]:
    recommendations = []
    promo = outputs.get("promotion_signal")
    product_mix = outputs.get("promotion_product_mix")
    discount_rate = _float_value(promo.metrics.get("discount_rate_pct")) if promo else 0.0
    top3_share = _float_value(product_mix.metrics.get("top3_product_revenue_share_pct")) if product_mix else 0.0

    if discount_rate <= 5:
        recommendations.append(
            Recommendation(
                id="promotion_no_broad_discount",
                action="Do not launch a broad discount unless traffic weakness persists.",
                urgency="low",
                rationale="Discount exposure is controlled, so a broad price cut is not justified by the current evidence.",
                expected_impact="Protects margin while keeping promotion decisions evidence-led.",
                evidence_ids=["discount_rate_pct"],
            )
        )
    if top3_share >= 50:
        recommendations.append(
            Recommendation(
                id="promotion_mid_tier_bundle",
                action="Use targeted bundles to support mid-tier products instead of discounting the full menu.",
                urgency="medium",
                rationale="Revenue is concentrated in the leading products, so targeted support can reduce dependency on a few items.",
                expected_impact="Improves product-mix balance without broad margin erosion.",
                evidence_ids=["top3_product_revenue_share_pct"],
            )
        )
    return recommendations
```

Import or reuse the existing `Recommendation` schema exactly as `builder.py` currently does for other helpers.

- [ ] **Step 4: Wire synthesis for the template**

Inside `_synthesize_node()`, add:

```python
if graph_state.template_id == "promotion_mix_analysis":
    graph_state.synthesis = S5Synthesis(
        summary=_synthesize_promotion_mix_summary(graph_state.agent_outputs),
        recommendations=_promotion_mix_recommendations(graph_state.agent_outputs),
    )
    return {"synthesis": graph_state.synthesis}
```

Keep the exact field names consistent with the current `S5Synthesis` usage in `builder.py`.

- [ ] **Step 5: Run promotion mix tests**

Run:

```bash
python -m pytest tests/test_s5_langgraph_promotion_mix.py -q
```

Expected: PASS.

## Task 6: Add Frontend Entry Point on the Revenue Dashboard

**Files:**
- Modify: `api/module4_frontend/static/index.html`
- Modify: `api/module4_frontend/static/s5_analysis.js`
- Test: `tests/test_frontend_business_events_static.py`

- [ ] **Step 1: Add static frontend test**

Add:

```python
def test_revenue_promotion_mix_ai_analysis_entry_exists():
    html = _html()

    assert "Promotion & Product Mix AI" in html
    assert 'id="rev-promotion-mix-s5-result"' in html
    assert "runModuleS5Analysis(\\'promotion_mix\\',\\'rev-date\\',\\'rev-promotion-mix-s5-result\\')" in html
```

- [ ] **Step 2: Run the focused frontend test and verify failure**

Run:

```bash
python -m pytest tests/test_frontend_business_events_static.py::test_revenue_promotion_mix_ai_analysis_entry_exists -q
```

Expected: FAIL because the button does not exist yet.

- [ ] **Step 3: Add the Revenue dashboard button**

In the Revenue dashboard render function, add a second S5 button near the existing Revenue AI control:

```javascript
<button class="btn btn-sm" id="rev-promotion-mix-s5-btn" onclick="runModuleS5Analysis(\'promotion_mix\',\'rev-date\',\'rev-promotion-mix-s5-result\')">Promotion & Product Mix AI</button>
```

Add a separate result container:

```javascript
<div id="rev-promotion-mix-s5-result" style="display:none;background:#fdfaf5;border:1px solid #d4c5a9;box-shadow:0 2px 8px rgba(139,105,20,0.08);border-radius:10px;padding:14px;margin-top:8px;max-height:50vh;overflow-y:auto"></div>
```

- [ ] **Step 4: Add S5 display labels only when missing**

In `api/module4_frontend/static/s5_analysis.js`, add labels for new evidence ids only if they are displayed as raw ids:

```javascript
promotion_no_broad_discount: 'No broad discount decision',
promotion_mid_tier_bundle: 'Targeted bundle opportunity',
top3_product_revenue_share_pct: 'Top product concentration',
discount_rate_pct: 'Discount rate',
```

- [ ] **Step 5: Run frontend static tests**

Run:

```bash
python -m pytest tests/test_frontend_business_events_static.py -q
```

Expected: PASS.

## Task 7: Validate API Behavior With Real Module Request

**Files:**
- Modify: `tests/test_s5_api_compatibility.py`
- Test: `tests/test_s5_api_compatibility.py`

- [ ] **Step 1: Add route-level async test**

Add a test that patches graph execution and verifies request routing:

```python
import pytest


@pytest.mark.asyncio
async def test_analyze_module_routes_promotion_mix_to_langgraph(monkeypatch):
    from s5_agent import server

    captured = {}

    async def fake_run_s5_graph(template_id, graph_request):
        captured["template_id"] = template_id
        captured["module"] = graph_request.module

        class FakeResponse:
            def model_dump(self):
                return {
                    "summary": "Promotion mix response",
                    "agent_outputs": [],
                    "evidence_graph": {"nodes": [], "edges": []},
                    "verification_report": {"passed": True},
                    "recommendations": [],
                    "warnings": [],
                    "metadata": {},
                }

        return FakeResponse()

    monkeypatch.setattr(server, "run_s5_graph", fake_run_s5_graph)

    response = await server.analyze_module(
        server.ModuleAnalyzeRequest(
            module="promotion_mix",
            date="2026-06-30",
            lang="en",
            force_refresh=True,
            params={},
        )
    )

    assert captured == {"template_id": "promotion_mix_analysis", "module": "promotion_mix"}
    assert response["summary"] == "Promotion mix response"
```

- [ ] **Step 2: Run the route test**

Run:

```bash
python -m pytest tests/test_s5_api_compatibility.py::test_analyze_module_routes_promotion_mix_to_langgraph -q
```

Expected: PASS.

## Task 8: Regression and Residue Checks

**Files:**
- Check all touched files.

- [ ] **Step 1: Run focused S5 and frontend tests**

Run:

```bash
python -m pytest tests/test_s5_langgraph_promotion_mix.py tests/test_s5_api_compatibility.py tests/test_frontend_business_events_static.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run:

```bash
python -m pytest tests -q
```

Expected: PASS.

- [ ] **Step 3: Check formatting and whitespace**

Run:

```bash
git diff --check -- .gitignore docs/superpowers/plans/2026-07-08-s5-promotion-mix-implementation.md s5_agent/server.py s5_agent/graph/registry.py s5_agent/graph/builder.py s5_agent/agents/promo.py s5_agent/agents/product_mix.py api/module4_frontend/static/index.html api/module4_frontend/static/s5_analysis.js tests/test_s5_langgraph_promotion_mix.py tests/test_s5_api_compatibility.py tests/test_frontend_business_events_static.py
```

Expected: no diff-check errors.

- [ ] **Step 4: Scan added lines for Chinese characters**

Run:

```powershell
$files = @(
  ".gitignore",
  "docs/superpowers/plans/2026-07-08-s5-promotion-mix-implementation.md",
  "s5_agent/server.py",
  "s5_agent/graph/registry.py",
  "s5_agent/graph/builder.py",
  "s5_agent/agents/promo.py",
  "s5_agent/agents/product_mix.py",
  "api/module4_frontend/static/index.html",
  "api/module4_frontend/static/s5_analysis.js",
  "tests/test_s5_langgraph_promotion_mix.py",
  "tests/test_s5_api_compatibility.py",
  "tests/test_frontend_business_events_static.py"
)
$diff = git diff --unified=0 -- $files
$added = $diff | Where-Object { $_ -match '^\+' -and $_ -notmatch '^\+\+\+' }
$matches = $added | Select-String -Pattern '[\u4e00-\u9fff]'
if ($matches) { $matches | ForEach-Object { $_.Line } } else { 'no CJK in added lines' }
```

Expected: `no CJK in added lines`.

- [ ] **Step 5: Scan for stale labels and duplicate blocks**

Run:

```bash
rg -n "promotion_mix|promotion_mix_analysis|Promotion & Product Mix AI|rev-promotion-mix-s5-result|PromotionSignalAgent|PromotionProductMixAgent" s5_agent api tests docs
```

Expected: only the intentional route, graph, frontend, tests, and plan references appear.

- [ ] **Step 6: Review Git status**

Run:

```bash
git status --short
```

Expected: touched files are limited to the implementation set and previously approved working-tree changes. Do not revert unrelated user or prior-session changes.

## Execution Notes

- Do not create a new standalone Promotion page.
- Do not add a manual S5 prompt input.
- Do not claim promotion uplift or causal impact unless the evidence supports it.
- Keep the first implementation lightweight and evidence-grounded.
- If existing `PromoAgent` or `ProductMixAgent` already passes a planned test, keep the implementation unchanged and move to the next task.
- If the frontend static file already contains related Revenue AI controls, extend the existing render pattern rather than creating a new layout style.

