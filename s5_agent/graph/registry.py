from __future__ import annotations

_MODULE_TO_TEMPLATE = {
    "inventory": "inventory_diagnosis",
    "revenue": "profit_root_cause",
    "forecast": "production_advice",
    "wastage": "wastage_root_cause",
    "promotion_mix": "promotion_mix_analysis",
}


def module_to_template(module: str) -> str:
    try:
        return _MODULE_TO_TEMPLATE[module]
    except KeyError as exc:
        supported = ", ".join(sorted(_MODULE_TO_TEMPLATE))
        raise ValueError(f"Unsupported S5 module: {module}. Supported modules: {supported}") from exc


def supported_templates() -> list[str]:
    return sorted(set(_MODULE_TO_TEMPLATE.values()))
