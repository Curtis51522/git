from pathlib import Path


FRONTEND = Path("api/module4_frontend/static/index.html")


def test_manager_can_build_a_manual_bakery_inflow_before_confirmation():
    source = FRONTEND.read_text(encoding="utf-8")

    assert 'onclick="addManualInflowItem()"' in source
    assert "function addManualInflowItem()" in source
    assert "EDITING_INFLOW&&EDITING_IDX===-3" in source
    assert "inflowDets.push({product_name:productName" in source
    assert "function confirmInflow()" in source
