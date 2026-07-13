from decimal import Decimal, ROUND_HALF_UP


_DEFAULTS = {
    "latte": ("hot", "normal"),
    "americano": ("hot", "none"),
    "cappuccino": ("hot", "less"),
    "mocha": ("hot", "normal"),
    "espresso": ("hot", "none"),
    "flat_white": ("hot", "less"),
    "caramel_macchiato": ("hot", "normal"),
    "cold_brew": ("cold", "slight"),
    "hot_chocolate": ("hot", "normal"),
    "matcha_latte": ("hot", "less"),
    "milk_tea": ("hot", "normal"),
    "chai_latte": ("hot", "less"),
    "earl_grey": ("hot", "none"),
    "english_breakfast": ("hot", "none"),
    "lemonade": ("cold", "normal"),
}

_COLD_ONLY = {"cold_brew", "lemonade"}
_SIZES = ["regular", "large"]
_TEMPERATURES = ["hot", "cold"]
_SUGAR = ["normal", "less", "slight", "none"]
_ICE = ["normal", "less", "none"]
POS_QUANTUM = Decimal("0.1")
LARGE_SURCHARGE = Decimal("3.0")


def is_beverage(product_name):
    return product_name in _DEFAULTS


def beverage_capabilities(product_name):
    if product_name not in _DEFAULTS:
        raise ValueError(f"Unknown beverage: {product_name}")
    default_temperature, default_sugar = _DEFAULTS[product_name]
    temperatures = ["cold"] if product_name in _COLD_ONLY else list(_TEMPERATURES)
    return {
        "product_name": product_name,
        "default_size": "regular",
        "default_temperature": default_temperature,
        "default_sugar": default_sugar,
        "default_ice": "normal" if default_temperature == "cold" else "none",
        "allowed_sizes": list(_SIZES),
        "allowed_temperatures": temperatures,
        "allowed_sugar": list(_SUGAR),
        "allowed_ice": list(_ICE),
    }


def list_beverage_capabilities():
    return [beverage_capabilities(name) for name in _DEFAULTS]


def _normalize_option(item, field, default):
    value = item.get(field)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"Invalid {field}: expected a string")
    normalized = value.strip().lower()
    return normalized or default


def normalize_beverage_item(product_name, item):
    capability = beverage_capabilities(product_name)
    size = _normalize_option(item, "size", capability["default_size"])
    temperature = _normalize_option(item, "temperature", capability["default_temperature"])
    sugar = _normalize_option(item, "sugar", capability["default_sugar"])
    ice_level = _normalize_option(item, "ice_level", capability["default_ice"])

    checks = (
        ("size", size, capability["allowed_sizes"]),
        ("temperature", temperature, capability["allowed_temperatures"]),
        ("sugar", sugar, capability["allowed_sugar"]),
        ("ice_level", ice_level, capability["allowed_ice"]),
    )
    for field, value, allowed in checks:
        if value not in allowed:
            raise ValueError(f"Invalid {field} for {product_name}: {value}")

    if temperature == "hot":
        ice_level = "none"

    return {
        "size": size,
        "temperature": temperature,
        "sugar": sugar,
        "ice_level": ice_level,
    }


def _decimal(value):
    return Decimal(str(value))


def round_pos_money(value):
    return _decimal(value).quantize(POS_QUANTUM, rounding=ROUND_HALF_UP)


def beverage_unit_price(base_price, size):
    normalized_size = str(size or "regular").lower()
    if normalized_size not in _SIZES:
        raise ValueError(f"Invalid beverage size: {size}")
    surcharge = LARGE_SURCHARGE if normalized_size == "large" else Decimal("0.0")
    return round_pos_money(_decimal(base_price) + surcharge)


def discounted_unit_values(base_price, discount_rate):
    unit_price = round_pos_money(base_price)
    unit_discount = round_pos_money(_decimal(base_price) * _decimal(discount_rate))
    return {
        "unit_price": unit_price,
        "unit_discount": unit_discount,
        "discounted_unit_price": round_pos_money(unit_price - unit_discount),
    }


def bundle_price_values(bakery_price, bakery_discount_rate, beverage_price, beverage_size):
    bakery = discounted_unit_values(bakery_price, bakery_discount_rate)
    drink_price = beverage_unit_price(beverage_price, beverage_size)
    return {
        "bakery_price": bakery["discounted_unit_price"],
        "beverage_price": drink_price,
        "total": round_pos_money(bakery["discounted_unit_price"] + drink_price),
        "savings": bakery["unit_discount"],
    }
