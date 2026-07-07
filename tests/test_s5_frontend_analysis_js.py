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
        "material_count_checked",
        "wasted_material_count",
        "total_waste_cost",
        "yield_data_available",
    ]
