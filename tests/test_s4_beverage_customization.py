import asyncio
from decimal import Decimal
from pathlib import Path

import pytest

from api.module4_frontend import bff
from api.module4_frontend.beverage_options import (
    beverage_capabilities,
    beverage_unit_price,
    bundle_price_values,
    discounted_unit_values,
    normalize_beverage_item,
    round_pos_money,
)


BFF_SOURCE = Path("api/module4_frontend/bff.py")


def test_beverage_options_endpoint_returns_backend_contract():
    result = asyncio.run(bff.get_beverage_options())

    assert result["status"] == "ok"
    by_name = {row["product_name"]: row for row in result["beverages"]}
    assert by_name["cold_brew"]["allowed_temperatures"] == ["cold"]
    assert by_name["latte"]["default_size"] == "regular"


def test_cold_brew_and_lemonade_are_cold_only():
    cold_brew = beverage_capabilities("cold_brew")
    lemonade = beverage_capabilities("lemonade")

    assert cold_brew["allowed_temperatures"] == ["cold"]
    assert lemonade["allowed_temperatures"] == ["cold"]
    assert cold_brew["default_temperature"] == "cold"
    assert lemonade["default_temperature"] == "cold"


def test_latte_defaults_are_complete_and_hot_has_no_ice():
    item = normalize_beverage_item("latte", {})

    assert item == {
        "size": "regular",
        "temperature": "hot",
        "sugar": "normal",
        "ice_level": "none",
    }


def test_cold_beverage_preserves_valid_ice_choice():
    item = normalize_beverage_item(
        "latte",
        {"size": "large", "temperature": "cold", "sugar": "less", "ice_level": "less"},
    )

    assert item == {
        "size": "large",
        "temperature": "cold",
        "sugar": "less",
        "ice_level": "less",
    }


def test_invalid_temperature_is_rejected():
    with pytest.raises(ValueError, match="temperature"):
        normalize_beverage_item("cold_brew", {"temperature": "hot"})


@pytest.mark.parametrize("value", [0, False, [], {}])
@pytest.mark.parametrize("field", ["size", "temperature", "sugar", "ice_level"])
def test_present_non_string_beverage_options_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        normalize_beverage_item("latte", {field: value})


def test_valid_beverage_option_strings_are_trimmed_and_lowercased():
    assert normalize_beverage_item(
        "latte",
        {
            "size": " Large ",
            "temperature": " COLD ",
            "sugar": " LESS ",
            "ice_level": " LESS ",
        },
    ) == {
        "size": "large",
        "temperature": "cold",
        "sugar": "less",
        "ice_level": "less",
    }


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1.92", "1.9"), ("2.25", "2.3"), ("2.24", "2.2")],
)
def test_pos_money_rounds_to_one_decimal_half_up(raw, expected):
    assert round_pos_money(raw) == Decimal(expected)


def test_large_beverage_adds_three_yuan_with_lowercase_size():
    assert beverage_unit_price("18.00", "large") == Decimal("21.0")
    assert beverage_unit_price("18.00", "Large") == Decimal("21.0")


def test_discounted_unit_values_round_discount_before_final_price():
    values = discounted_unit_values("9.00", "0.25")

    assert values == {
        "unit_price": Decimal("9.0"),
        "unit_discount": Decimal("2.3"),
        "discounted_unit_price": Decimal("6.7"),
    }


def test_discounted_unit_values_rounds_raw_boundary_price_once():
    values = discounted_unit_values("1.45", "0.10")

    assert values == {
        "unit_price": Decimal("1.5"),
        "unit_discount": Decimal("0.1"),
        "discounted_unit_price": Decimal("1.4"),
    }


def test_bundle_price_uses_rounded_bakery_discount_only():
    regular = bundle_price_values("9.00", "0.25", "18.00", "regular")
    large = bundle_price_values("9.00", "0.25", "18.00", "large")

    assert regular["total"] == Decimal("24.7")
    assert large["total"] == Decimal("27.7")
    assert regular["savings"] == Decimal("2.3")
    assert large["savings"] == Decimal("2.3")


def test_checkout_persists_all_beverage_options():
    source = BFF_SOURCE.read_text(encoding="utf-8")

    assert "coffee_size, coffee_temp, coffee_ice, coffee_sugar" in source
    assert 'item.get("size")' in source
    assert 'item.get("temperature")' in source
    assert 'item.get("ice_level")' in source
    assert 'item.get("sugar")' in source


def test_refund_never_restores_cups_for_non_sellable_returns():
    source = BFF_SOURCE.read_text(encoding="utf-8")
    refund_source = source[
        source.index("async def refund_order") : source.index(
            '@router.get("/revenue/daily")'
        )
    ]

    assert "stock_quantity = stock_quantity +" not in refund_source
    assert '"non_sellable"' in refund_source


def test_products_endpoint_returns_bakery_and_beverage_prices(monkeypatch):
    rows = [
        {"product_name": "croissant", "category": "bakery", "unit_price": 10, "material_cost": 2},
        {"product_name": "latte", "category": "beverage", "unit_price": 19, "material_cost": 4},
    ]
    filters = []

    class Query:
        def select(self, _columns):
            return self

        def eq(self, column, value):
            filters.append((column, value))
            return self

        def execute(self):
            return type("Response", (), {"data": rows})()

    class DB:
        closed = False

        def close(self):
            self.closed = True

    db = DB()
    monkeypatch.setattr(bff, "get_db", lambda: db)
    monkeypatch.setattr(bff, "q", lambda _db, _table: Query())
    monkeypatch.setattr(bff, "_product_prices_cache", {"latte": 14.0})

    result = asyncio.run(bff.list_products())

    assert filters == []
    assert {row["product_name"]: row["unit_price"] for row in result["products"]} == {
        "croissant": 10.0,
        "latte": 19.0,
    }
    assert bff._product_prices_cache == {"croissant": 10.0, "latte": 19.0}
    assert {row["product_name"]: row["cost_price"] for row in result["products"]} == {
        "croissant": 2.0,
        "latte": 4.0,
    }
    assert db.closed
