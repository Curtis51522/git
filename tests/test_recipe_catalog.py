import json
from decimal import Decimal
from pathlib import Path

import pytest

from scripts.recipe_catalog import (
    aggregate_requirements,
    calculate_material_costs,
    load_catalog,
)


CATALOG_PATH = Path("data/reference/product_recipe_catalog.json")

EXPECTED_BAKERY_PRODUCTS = {
    "apple_pie",
    "bagel",
    "baguette",
    "bread_coconut",
    "bread_roll",
    "brioche",
    "brownie",
    "chiffon",
    "chocolate_cake",
    "chocopie",
    "cookie",
    "cornbread",
    "cream_horn",
    "croissant",
    "croissant_chocolate",
    "donut",
    "eggtart",
    "flatbread",
    "macaron",
    "mantequilla",
    "melon_bread",
    "muffin",
    "pancake",
    "pandesal",
    "pizza_bread",
    "pullman",
    "soboru_bread",
    "sourdough",
    "stickbread",
    "tostada",
}


def _load_raw_catalog():
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def _formula_map(catalog):
    return {
        product["product_name"]: {
            ingredient["material_name"]: ingredient
            for ingredient in product["ingredients"]
        }
        for product in catalog["products"]
    }


def test_catalog_has_exactly_the_database_bakery_products():
    catalog = _load_raw_catalog()
    names = [item["product_name"] for item in catalog["products"]]

    assert len(names) == 30
    assert len(set(names)) == 30
    assert set(names) == EXPECTED_BAKERY_PRODUCTS


def test_every_product_has_source_and_positive_unique_ingredients():
    catalog = _load_raw_catalog()

    for product in catalog["products"]:
        assert product["retail_identity"].strip()
        assert Decimal(product["portion_weight_g"]) > 0
        assert product["source_title"].strip()
        assert product["source_url"].startswith("https://")
        assert product["source_yield"].strip()
        assert product["adaptation_note"].strip()

        ingredients = product["ingredients"]
        names = [ingredient["material_name"] for ingredient in ingredients]
        assert ingredients
        assert len(names) == len(set(names))
        assert all(
            Decimal(ingredient["quantity_per_unit"]) > 0
            for ingredient in ingredients
        )


def test_recipe_materials_have_matching_canonical_metadata():
    catalog = _load_raw_catalog()
    materials = {
        material["material_name"]: material for material in catalog["materials"]
    }

    assert catalog["canonical_units"] == ["kg", "L", "pcs"]
    assert len(materials) == len(catalog["materials"])

    for product in catalog["products"]:
        for ingredient in product["ingredients"]:
            material = materials[ingredient["material_name"]]
            assert ingredient["unit"] == material["unit"]
            assert ingredient["unit"] in catalog["canonical_units"]


def test_realistic_signature_ingredients_are_present():
    formulas = _formula_map(_load_raw_catalog())

    assert "Almond Flour" in formulas["macaron"]
    assert "Cornmeal" in formulas["cornbread"]
    assert "Desiccated Coconut" in formulas["bread_coconut"]
    assert "Cake Flour" in formulas["chiffon"]
    assert "Marshmallow" in formulas["chocopie"]
    assert "Yeast" in formulas["baguette"]
    assert "Salt" in formulas["sourdough"]
    assert "Water" in formulas["flatbread"]


def test_water_is_a_non_stocked_utility_material():
    catalog = _load_raw_catalog()
    water = next(
        material
        for material in catalog["materials"]
        if material["material_name"] == "Water"
    )

    assert water["unit"] == "L"
    assert water["category"] == "utility"
    assert water["track_inventory"] is False
    assert Decimal(water["reorder_point"]) == 0


def _minimal_catalog():
    return {
        "schema_version": 1,
        "canonical_units": ["kg", "L", "pcs"],
        "materials": [
            {
                "material_name": "Bread Flour",
                "unit": "kg",
                "unit_price": "8.00",
                "category": "flour",
                "reorder_point": "1.000000",
                "opening_stock": "10.000000",
                "track_inventory": True,
            }
        ],
        "products": [
            {
                "product_name": "bread_roll",
                "retail_identity": "Test bread roll",
                "portion_weight_g": "50",
                "source_title": "Test formula",
                "source_url": "https://example.com/formula",
                "source_yield": "Ten rolls",
                "adaptation_note": "Used only for calculation tests.",
                "ingredients": [
                    {
                        "material_name": "Bread Flour",
                        "quantity_per_unit": "0.100000",
                        "unit": "kg",
                    }
                ],
            }
        ],
    }


def test_load_catalog_rejects_duplicate_product_material(tmp_path):
    catalog = _minimal_catalog()
    catalog["products"][0]["ingredients"].append(
        dict(catalog["products"][0]["ingredients"][0])
    )
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate recipe material"):
        load_catalog(path)


def test_aggregate_requirements_uses_decimal_and_optional_wastage():
    catalog = _minimal_catalog()

    result = aggregate_requirements(
        catalog,
        {"bread_roll": 10},
        apply_wastage=True,
        wastage_by_product={"bread_roll": Decimal("0.05")},
    )

    assert result["Bread Flour"] == Decimal("1.050000")


def test_calculate_material_cost_excludes_wastage():
    catalog = _minimal_catalog()

    assert calculate_material_costs(catalog)["bread_roll"] == Decimal("0.800000")


def test_load_catalog_accepts_the_production_catalog():
    catalog = load_catalog(CATALOG_PATH)

    assert len(catalog["products"]) == 30
