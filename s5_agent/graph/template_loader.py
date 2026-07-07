from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "configs" / "templates"


def load_template(template_id: str) -> dict[str, Any]:
    template_path = _TEMPLATE_DIR / f"{template_id}.yaml"
    if not template_path.exists():
        raise FileNotFoundError(template_path)

    with template_path.open("r", encoding="utf-8") as template_file:
        template = yaml.safe_load(template_file) or {}

    if template.get("id") != template_id:
        raise ValueError(
            f"Template id mismatch: expected {template_id!r}, got {template.get('id')!r}"
        )

    return template
