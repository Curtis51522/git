from api.module4_frontend import bff


def _row(name, category, unit, bakery=False, beverage=False):
    return {
        "material_name": name,
        "category": category,
        "unit": unit,
        "stock": 10.0,
        "reorder_point": 2.0,
        "baseline": 12.0,
        "used_by_bakery": bakery,
        "used_by_beverage": beverage,
    }


def test_inventory_material_groups_follow_recipe_usage_and_service_supplies():
    rows = [
        _row("Bread Flour", "flour", "kg", bakery=True),
        _row("Milk", "dairy", "L", bakery=True, beverage=True),
        _row("Coffee Beans", "coffee", "kg", beverage=True),
        _row("Cup Regular", "packaging", "pcs"),
        _row("Packaging Bag", "packaging", "pcs"),
    ]

    groups = bff._group_inventory_material_rows(rows)

    assert [item["material_name"] for item in groups["baking"]] == [
        "Bread Flour",
        "Milk",
    ]
    assert [item["material_name"] for item in groups["beverage"]] == [
        "Coffee Beans",
        "Cup Regular",
        "Milk",
    ]
    assert [item["material_name"] for item in groups["packaging"]] == [
        "Packaging Bag"
    ]
    assert next(
        item for item in groups["baking"] if item["material_name"] == "Milk"
    )["usage_scope"] == "Both"
    assert next(
        item for item in groups["beverage"] if item["material_name"] == "Cup Regular"
    )["usage_scope"] == "Beverage"


def test_inventory_material_groups_do_not_drop_tracked_packaging():
    groups = bff._group_inventory_material_rows(
        [_row("Box", "packaging", "pcs")]
    )

    assert groups["baking"] == []
    assert groups["beverage"] == []
    assert groups["packaging"][0]["usage_scope"] == "Packaging"
