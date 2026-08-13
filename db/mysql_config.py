"""Shared MySQL connection configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def get_mysql_connection_config() -> dict[str, object]:
    """Return validated MySQL connector arguments from the environment."""
    raw_port = os.getenv("MYSQL_PORT", "3306")
    try:
        port = int(raw_port)
    except (TypeError, ValueError) as exc:
        raise ValueError("MYSQL_PORT must be an integer between 1 and 65535") from exc

    if not 1 <= port <= 65535:
        raise ValueError("MYSQL_PORT must be an integer between 1 and 65535")

    return {
        "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
        "port": port,
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "bakery_ai"),
        "time_zone": os.getenv("MYSQL_SESSION_TIME_ZONE", "+08:00"),
    }
