"""Authenticated JSON reads from the manager dashboard API."""

from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen


def fetch_dashboard_json(
    url: str,
    params: dict[str, Any] | None = None,
    timeout: int | float = 10,
) -> dict[str, Any]:
    """Read dashboard JSON while preserving the caller's authorization header."""
    authorization = str((params or {}).get("_authorization", "")).strip()
    operation_at = str((params or {}).get("_operation_at", "")).strip()
    headers = {}
    if authorization:
        headers["Authorization"] = authorization
    if operation_at:
        headers["X-Operation-At"] = operation_at
    request = Request(url, headers=headers, method="GET")
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            return {}
        payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else {}
