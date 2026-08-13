from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import os
from pathlib import Path
import subprocess
from typing import Any

from deployment.launcher.database import create_snapshot


def _required(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def export_command(environment: Mapping[str, str]) -> list[str]:
    host = environment.get("MYSQL_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port_text = environment.get("MYSQL_PORT", "3306").strip() or "3306"
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError("MYSQL_PORT must be an integer") from exc
    if not 1 <= port <= 65_535:
        raise ValueError("MYSQL_PORT must be between 1 and 65535")
    database = _required(environment, "MYSQL_DATABASE")
    user = _required(environment, "MYSQL_USER")
    return [
        "mysqldump",
        f"--host={host}",
        f"--port={port}",
        f"--user={user}",
        "--single-transaction",
        "--routines",
        "--triggers",
        "--events",
        "--set-gtid-purged=OFF",
        "--no-tablespaces",
        database,
    ]


def export_snapshot(
    destination: str | Path,
    *,
    application_version: str,
    environment: Mapping[str, str] | None = None,
    replace: bool = False,
    popen: Callable[..., Any] = subprocess.Popen,
) -> Path:
    source_environment = os.environ if environment is None else environment
    password = _required(source_environment, "MYSQL_PASSWORD")
    database = _required(source_environment, "MYSQL_DATABASE")
    command = export_command(source_environment)
    child_environment = os.environ.copy()
    child_environment.update(source_environment)
    child_environment.pop("MYSQL_PASSWORD", None)
    child_environment["MYSQL_PWD"] = password
    return create_snapshot(
        command,
        destination,
        database=database,
        application_version=application_version,
        environment=child_environment,
        replace=replace,
        popen=popen,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export an approved MySQL 8.4 Bakery AI release snapshot."
    )
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--application-version",
        default=os.environ.get("BAKERY_APP_VERSION", ""),
    )
    parser.add_argument("--replace", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parser = _parser()
    options = parser.parse_args(arguments)
    if not options.application_version.strip():
        parser.error(
            "--application-version or BAKERY_APP_VERSION is required"
        )
    export_snapshot(
        options.destination,
        application_version=options.application_version,
        replace=options.replace,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
