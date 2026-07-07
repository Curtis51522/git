from __future__ import annotations

_MODULE_TO_TEMPLATE = {
    "inventory": "inventory_diagnosis",
    "revenue": "profit_root_cause",
    "forecast": "production_advice",
    "schedule": "staffing_diagnosis",
    "wastage": "wastage_root_cause",
    "kpi": "full_diagnosis",
}


def module_to_template(module: str) -> str:
    return _MODULE_TO_TEMPLATE.get(module, "full_diagnosis")


def supported_templates() -> list[str]:
    return sorted(set(_MODULE_TO_TEMPLATE.values()) | {"full_diagnosis"})
