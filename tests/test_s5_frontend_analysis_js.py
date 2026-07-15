import json
import subprocess
from pathlib import Path


def test_collect_s5_details_keeps_agent_context_with_recommendation_evidence():
    js_path = Path(__file__).resolve().parents[1] / "api/module4_frontend/static/s5_analysis.js"
    payload = {
        "agent_outputs": [
            {
                "confidence": 0.88,
                "risks": ["material_wastage_risk"],
                "evidence_items": [
                    {"id": "material_count_checked"},
                    {"id": "wasted_material_count"},
                    {"id": "total_waste_cost"},
                    {"id": "yield_data_available"},
                ],
            }
        ],
        "recommendations": [
            {
                "evidence_ids": ["wasted_material_count", "total_waste_cost"],
            }
        ],
    }
    script = f"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync({json.dumps(str(js_path))}, 'utf8');
vm.runInThisContext(source);
const details = collectS5Details({json.dumps(payload)});
console.log(JSON.stringify(details.evidenceIds));
"""

    completed = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    evidence_ids = json.loads(completed.stdout)
    assert evidence_ids == [
        "wasted_material_count",
        "total_waste_cost",
        "material_count_checked",
        "yield_data_available",
    ]


def test_promotion_basket_metrics_have_manager_friendly_labels():
    js_path = Path(__file__).resolve().parents[1] / "api/module4_frontend/static/s5_analysis.js"
    script = f"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync({json.dumps(str(js_path))}, 'utf8');
vm.runInThisContext(source);
console.log(JSON.stringify([
  labelS5Token('basket_size_weakness'),
  labelS5Token('items_per_order'),
  labelS5Token('items_per_order_change_pct'),
  labelS5Token('revenue_per_item'),
  labelS5Token('revenue_per_item_change_pct')
]));
"""

    completed = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == [
        "Basket size weakness",
        "Items per order",
        "Items per order change",
        "Revenue per item",
        "Revenue per item change",
    ]


def test_forecast_evidence_uses_manager_friendly_historical_labels():
    js_path = Path(__file__).resolve().parents[1] / "api/module4_frontend/static/s5_analysis.js"
    script = f"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync({json.dumps(str(js_path))}, 'utf8');
vm.runInThisContext(source);
console.log(JSON.stringify([
  labelS5Token('forecast_wape'),
  labelS5Token('q90_shortage_units'),
  labelS5Token('material_stock_data_available'),
  labelS5Token('material_data_gap')
]));
"""

    completed = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == [
        "Held-out historical forecast error",
        "High-demand capacity gap",
        "Raw-material stock data status",
        "Raw-material stock data unavailable",
    ]


def test_s5_recommendations_render_before_verification_details():
    js_path = Path(__file__).resolve().parents[1] / "api/module4_frontend/static/s5_analysis.js"
    source = js_path.read_text(encoding="utf-8")
    render_source = source[
        source.index("async function runModuleS5Analysis") : source.index(
            "resDiv.innerHTML = html;",
            source.index("async function runModuleS5Analysis"),
        )
    ]

    recommendation_position = render_source.index(
        "if (d.recommendations && d.recommendations.length > 0)"
    )
    details_position = render_source.index("html += renderS5Details(d);")

    assert recommendation_position < details_position
