import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_SOURCE = (REPO_ROOT / "api" / "module4_frontend" / "static" / "index.html").read_text(encoding="utf-8")


def _function_body(name):
    match = re.search(rf"async function {name}\(\)\{{", INDEX_SOURCE)
    assert match, f"{name} not found"
    start = match.end()
    depth = 1
    pos = start
    while pos < len(INDEX_SOURCE) and depth:
        char = INDEX_SOURCE[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        pos += 1
    assert depth == 0, f"{name} body is not balanced"
    return INDEX_SOURCE[start:pos - 1]


def test_wastage_rates_use_material_units_instead_of_fixed_grams():
    body = _function_body("loadInvWastageRates")

    assert "fmtMaterialQty" in body
    assert "wr.unit||it.unit" in body
    assert "*1000" not in body
    assert "+'g'" not in body


def test_wastage_material_input_uses_material_units_for_consumed_quantity():
    body = _function_body("loadInvWastageMaterials")

    assert "fmtMaterialQty(it.consumed_since,it.unit)" in body
    assert "consumed_since*1000" not in body
    assert "+'g'" not in body


def test_wastage_history_renders_stock_and_waste_with_units():
    body = _function_body("loadInvWastageHistory")

    assert "fmtMaterialQty(it.theoretical_stock,it.unit)+' '+it.unit" in body
    assert "fmtMaterialQty(it.actual_stock,it.unit)+' '+it.unit" in body
    assert "fmtMaterialQty(it.wastage_qty,it.unit)+' '+it.unit" in body
