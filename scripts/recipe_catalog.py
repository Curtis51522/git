import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import urlparse


QUANTITY_PRECISION = Decimal("0.000001")


def _decimal(value, label, *, allow_zero=False):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid decimal for {label}") from exc
    if not number.is_finite():
        raise ValueError(f"Non-finite decimal for {label}")
    if number < 0 or (number == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be {qualifier}")
    return number


def _required_text(record, key, label):
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing {key} for {label}")
    return value.strip()


def _validate_catalog(catalog):
    if not isinstance(catalog, dict):
        raise ValueError("Catalog root must be an object")

    units = catalog.get("canonical_units")
    if not isinstance(units, list) or not units or len(units) != len(set(units)):
        raise ValueError("canonical_units must be a non-empty unique list")
    if any(not isinstance(unit, str) or not unit for unit in units):
        raise ValueError("canonical_units contains an invalid unit")

    material_rows = catalog.get("materials")
    if not isinstance(material_rows, list) or not material_rows:
        raise ValueError("materials must be a non-empty list")

    materials = {}
    for index, material in enumerate(material_rows):
        if not isinstance(material, dict):
            raise ValueError(f"Material at index {index} must be an object")
        name = _required_text(material, "material_name", f"material {index}")
        if name in materials:
            raise ValueError(f"Duplicate material: {name}")
        unit = _required_text(material, "unit", name)
        if unit not in units:
            raise ValueError(f"Unknown unit for material {name}: {unit}")
        _required_text(material, "category", name)
        _decimal(material.get("unit_price"), f"{name} unit_price", allow_zero=True)
        _decimal(
            material.get("reorder_point"),
            f"{name} reorder_point",
            allow_zero=True,
        )
        _decimal(
            material.get("opening_stock"),
            f"{name} opening_stock",
            allow_zero=True,
        )
        if not isinstance(material.get("track_inventory"), bool):
            raise ValueError(f"track_inventory must be boolean for {name}")
        materials[name] = material

    product_rows = catalog.get("products")
    if not isinstance(product_rows, list) or not product_rows:
        raise ValueError("products must be a non-empty list")

    product_names = set()
    for index, product in enumerate(product_rows):
        if not isinstance(product, dict):
            raise ValueError(f"Product at index {index} must be an object")
        name = _required_text(product, "product_name", f"product {index}")
        if name in product_names:
            raise ValueError(f"Duplicate product: {name}")
        product_names.add(name)

        _required_text(product, "retail_identity", name)
        _decimal(product.get("portion_weight_g"), f"{name} portion_weight_g")
        _required_text(product, "source_title", name)
        source_url = _required_text(product, "source_url", name)
        parsed_url = urlparse(source_url)
        if parsed_url.scheme != "https" or not parsed_url.netloc:
            raise ValueError(f"Invalid source URL for {name}")
        _required_text(product, "source_yield", name)
        _required_text(product, "adaptation_note", name)

        ingredients = product.get("ingredients")
        if not isinstance(ingredients, list) or not ingredients:
            raise ValueError(f"Missing ingredients for {name}")
        recipe_materials = set()
        for ingredient_index, ingredient in enumerate(ingredients):
            if not isinstance(ingredient, dict):
                raise ValueError(
                    f"Ingredient at index {ingredient_index} for {name} must be an object"
                )
            material_name = _required_text(
                ingredient,
                "material_name",
                f"ingredient {ingredient_index} for {name}",
            )
            if material_name in recipe_materials:
                raise ValueError(
                    f"Duplicate recipe material for {name}: {material_name}"
                )
            recipe_materials.add(material_name)
            if material_name not in materials:
                raise ValueError(f"Unknown recipe material for {name}: {material_name}")

            unit = _required_text(ingredient, "unit", f"{name}/{material_name}")
            expected_unit = materials[material_name]["unit"]
            if unit != expected_unit:
                raise ValueError(
                    f"Unit mismatch for {name}/{material_name}: {unit} != {expected_unit}"
                )
            _decimal(
                ingredient.get("quantity_per_unit"),
                f"{name}/{material_name} quantity_per_unit",
            )

    return catalog


def load_catalog(path):
    catalog_path = Path(path)
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load recipe catalog: {catalog_path}") from exc
    return _validate_catalog(catalog)


def aggregate_requirements(
    catalog,
    quantities,
    *,
    apply_wastage=False,
    wastage_by_product=None,
):
    _validate_catalog(catalog)
    if not isinstance(quantities, dict):
        raise ValueError("quantities must be a mapping")
    if apply_wastage and not isinstance(wastage_by_product, dict):
        raise ValueError("wastage_by_product is required when apply_wastage is true")

    formulas = {
        product["product_name"]: product["ingredients"]
        for product in catalog["products"]
    }
    totals = {}
    for product_name, raw_quantity in quantities.items():
        if product_name not in formulas:
            raise ValueError(f"Unknown product: {product_name}")
        quantity = _decimal(raw_quantity, f"{product_name} quantity")
        factor = Decimal("1")
        if apply_wastage:
            if product_name not in wastage_by_product:
                raise ValueError(f"Missing wastage for product: {product_name}")
            wastage = _decimal(
                wastage_by_product[product_name],
                f"{product_name} wastage",
                allow_zero=True,
            )
            factor += wastage

        for ingredient in formulas[product_name]:
            material_name = ingredient["material_name"]
            per_unit = _decimal(
                ingredient["quantity_per_unit"],
                f"{product_name}/{material_name} quantity_per_unit",
            )
            totals[material_name] = totals.get(material_name, Decimal("0")) + (
                per_unit * quantity * factor
            )

    return {
        material_name: total.quantize(QUANTITY_PRECISION)
        for material_name, total in totals.items()
    }


def calculate_material_costs(catalog):
    _validate_catalog(catalog)
    prices = {
        material["material_name"]: _decimal(
            material["unit_price"],
            f"{material['material_name']} unit_price",
            allow_zero=True,
        )
        for material in catalog["materials"]
    }

    costs = {}
    for product in catalog["products"]:
        cost = Decimal("0")
        for ingredient in product["ingredients"]:
            quantity = _decimal(
                ingredient["quantity_per_unit"],
                f"{product['product_name']}/{ingredient['material_name']} quantity_per_unit",
            )
            cost += quantity * prices[ingredient["material_name"]]
        costs[product["product_name"]] = cost.quantize(QUANTITY_PRECISION)
    return costs
