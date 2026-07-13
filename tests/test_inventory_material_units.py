from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BFF_SOURCE = (REPO_ROOT / "api" / "module4_frontend" / "bff.py").read_text(encoding="utf-8")


def test_checkout_uses_raw_material_units_for_recipe_transactions():
    assert "JOIN raw_materials rm ON rm.material_name = pr.material_name" in BFF_SOURCE
    assert "rm.unit" in BFF_SOURCE
    assert "(mat_name, 'outflow', actual_used_qty, 'kg', receipt_id)" not in BFF_SOURCE
    assert "SELECT material_name, stock_quantity, unit" in BFF_SOURCE
    assert "material_units[material_name]" in BFF_SOURCE
    assert "INSERT INTO material_transactions" in BFF_SOURCE


def test_checkout_does_not_apply_wastage_to_piece_based_materials():
    assert 'if unit != "pcs" and (category or "") != "packaging":' in BFF_SOURCE
    assert 'required *= Decimal("1") + products[product_name]["wastage_pct"]' in BFF_SOURCE
