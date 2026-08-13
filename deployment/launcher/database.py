from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any


SQL_HEADER_MARKER = b"-- MySQL dump"
SQL_BODY_MARKERS = (b"CREATE TABLE", b"INSERT INTO")
SQL_TRAILER_MARKER = b"-- Dump completed"


class DatabaseOperationError(RuntimeError):
    pass


class SnapshotValidationError(ValueError):
    pass


def _compose_prefix(
    compose_file: str | Path,
    env_file: str | Path,
    project_name: str,
) -> list[str]:
    return [
        "docker",
        "compose",
        "--project-name",
        project_name,
        "--file",
        str(Path(compose_file).resolve()),
        "--env-file",
        str(Path(env_file).resolve()),
    ]


def _container_mysql_client_command(
    executable: str,
    *arguments: str,
) -> list[str]:
    return [
        "sh",
        "-c",
        'MYSQL_PWD="$MYSQL_ROOT_PASSWORD" exec "$@"',
        "bakery-ai-mysql-client",
        executable,
        *arguments,
    ]


def backup_command(
    compose_file: str | Path,
    env_file: str | Path,
    database: str,
    *,
    project_name: str = "bakery-ai",
    user: str = "root",
) -> list[str]:
    return [
        *_compose_prefix(compose_file, env_file, project_name),
        "exec",
        "-T",
        "mysql",
        *_container_mysql_client_command(
            "mysqldump",
            "--single-transaction",
            "--routines",
            "--triggers",
            "--events",
            "--set-gtid-purged=OFF",
            "--no-tablespaces",
            "-u",
            user,
            database,
        ),
    ]


def restore_command(
    compose_file: str | Path,
    env_file: str | Path,
    database: str,
    *,
    project_name: str = "bakery-ai",
    user: str = "root",
) -> list[str]:
    return [
        *_compose_prefix(compose_file, env_file, project_name),
        "exec",
        "-T",
        "mysql",
        *_container_mysql_client_command(
            "mysql",
            "-u",
            user,
            database,
        ),
    ]


def metadata_path(snapshot: str | Path) -> Path:
    path = Path(snapshot)
    return path.with_name(f"{path.name}.sha256.json")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_is_usable(snapshot: str | Path) -> bool:
    path = Path(snapshot)
    if not path.is_file() or path.stat().st_size == 0:
        return False

    header_found = False
    body_found = False
    trailer_found = False
    try:
        with path.open("rb") as handle:
            for line in handle:
                header_found = header_found or SQL_HEADER_MARKER in line
                body_found = body_found or any(
                    marker in line for marker in SQL_BODY_MARKERS
                )
                trailer_found = trailer_found or SQL_TRAILER_MARKER in line
    except OSError:
        return False
    return header_found and body_found and trailer_found


def _utc_timestamp(now: datetime) -> str:
    value = now.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    return value.replace("+00:00", "Z")


def _metadata(
    snapshot: Path,
    *,
    database: str,
    application_version: str,
    now: datetime,
) -> dict[str, str]:
    if not database.strip():
        raise ValueError("Database name is required")
    if not application_version.strip():
        raise ValueError("Application version is required")
    return {
        "algorithm": "sha256",
        "application_version": application_version,
        "created_at": _utc_timestamp(now),
        "database": database,
        "sha256": file_sha256(snapshot),
        "snapshot": snapshot.name,
    }


def _write_json_exclusive(path: Path, content: Mapping[str, str]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(content), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_snapshot_metadata(
    snapshot: str | Path,
    *,
    database: str,
    application_version: str,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> Path:
    snapshot_path = Path(snapshot)
    sidecar = metadata_path(snapshot_path)
    content = _metadata(
        snapshot_path,
        database=database,
        application_version=application_version,
        now=now(),
    )
    _write_json_exclusive(sidecar, content)
    return sidecar


def _read_snapshot_metadata(
    snapshot: Path,
    *,
    required: bool,
) -> Mapping[str, Any] | None:
    sidecar = metadata_path(snapshot)
    if not sidecar.is_file():
        if required:
            raise SnapshotValidationError(
                "Snapshot metadata sidecar is required"
            )
        return None
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotValidationError("Snapshot metadata is invalid") from exc
    if not isinstance(value, dict):
        raise SnapshotValidationError("Snapshot metadata is invalid")
    return value


def validate_snapshot(
    snapshot: str | Path,
    *,
    require_metadata: bool = False,
    expected_database: str | None = None,
) -> None:
    snapshot_path = Path(snapshot)
    if not snapshot_is_usable(snapshot_path):
        raise SnapshotValidationError(
            "Snapshot is empty or missing required MySQL dump markers"
        )

    metadata = _read_snapshot_metadata(
        snapshot_path,
        required=require_metadata,
    )
    if metadata is None:
        return
    if metadata.get("algorithm") != "sha256":
        raise SnapshotValidationError(
            "Snapshot metadata algorithm must be sha256"
        )
    if metadata.get("snapshot") != snapshot_path.name:
        raise SnapshotValidationError(
            "Snapshot metadata filename does not match the SQL file"
        )
    database = metadata.get("database")
    if not isinstance(database, str) or not database.strip():
        raise SnapshotValidationError(
            "Snapshot metadata database is required"
        )
    if expected_database is not None and database != expected_database:
        raise SnapshotValidationError(
            "Snapshot metadata database does not match the restore target"
        )
    application_version = metadata.get("application_version")
    if (
        not isinstance(application_version, str)
        or not application_version.strip()
    ):
        raise SnapshotValidationError(
            "Snapshot metadata application version is required"
        )
    expected = metadata.get("sha256")
    if not isinstance(expected, str) or expected != file_sha256(snapshot_path):
        raise SnapshotValidationError(
            "Snapshot checksum does not match metadata"
        )


def _error_text(value: bytes | str | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return (value or "").strip()


def _remove_generated_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _copy_to_temporary(path: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=f".{path.name}.backup.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        backup = Path(handle.name)
        with path.open("rb") as source:
            shutil.copyfileobj(source, handle)
        handle.flush()
        os.fsync(handle.fileno())
    return backup


def _install_snapshot_pair(
    temporary_snapshot: Path,
    temporary_metadata: Path,
    destination: Path,
    sidecar: Path,
) -> None:
    backup_snapshot: Path | None = None
    backup_metadata: Path | None = None
    snapshot_installed = False
    metadata_installed = False
    try:
        if destination.exists():
            backup_snapshot = _copy_to_temporary(destination)
        if sidecar.exists():
            backup_metadata = _copy_to_temporary(sidecar)
        os.replace(temporary_snapshot, destination)
        snapshot_installed = True
        os.replace(temporary_metadata, sidecar)
        metadata_installed = True
    except Exception:
        if snapshot_installed:
            if backup_snapshot is None:
                _remove_generated_file(destination)
            else:
                os.replace(backup_snapshot, destination)
                backup_snapshot = None
        if metadata_installed:
            if backup_metadata is None:
                _remove_generated_file(sidecar)
            else:
                os.replace(backup_metadata, sidecar)
                backup_metadata = None
        raise
    finally:
        for generated in (
            temporary_snapshot,
            temporary_metadata,
            backup_snapshot,
            backup_metadata,
        ):
            if generated is not None:
                _remove_generated_file(generated)


def create_snapshot(
    command: Sequence[str],
    destination: str | Path,
    *,
    database: str,
    application_version: str,
    environment: Mapping[str, str] | None = None,
    replace: bool = False,
    popen: Callable[..., Any] = subprocess.Popen,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> Path:
    destination_path = Path(destination)
    sidecar = metadata_path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if not replace and (destination_path.exists() or sidecar.exists()):
        raise FileExistsError(f"Snapshot already exists: {destination_path}")

    temporary_snapshot: Path | None = None
    temporary_metadata: Path | None = None
    output_handle = None
    try:
        if replace:
            output_handle = tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{destination_path.name}.",
                suffix=".tmp",
                dir=destination_path.parent,
                delete=False,
            )
            work_path = Path(output_handle.name)
            temporary_snapshot = work_path
        else:
            output_handle = destination_path.open("xb")
            work_path = destination_path

        with output_handle:
            process = popen(
                list(command),
                stdout=output_handle,
                stderr=subprocess.PIPE,
                shell=False,
                env=None if environment is None else dict(environment),
            )
            _, stderr = process.communicate()
        if process.returncode != 0:
            detail = _error_text(stderr) or "unknown database client error"
            raise DatabaseOperationError(f"Database backup failed: {detail}")
        validate_snapshot(work_path)

        metadata = _metadata(
            work_path,
            database=database,
            application_version=application_version,
            now=now(),
        )
        metadata["snapshot"] = destination_path.name

        if replace:
            with tempfile.NamedTemporaryFile(
                mode="x",
                encoding="utf-8",
                newline="\n",
                prefix=f".{sidecar.name}.",
                suffix=".tmp",
                dir=sidecar.parent,
                delete=False,
            ) as metadata_handle:
                json.dump(metadata, metadata_handle, indent=2, sort_keys=True)
                metadata_handle.write("\n")
                temporary_metadata = Path(metadata_handle.name)
            _install_snapshot_pair(
                work_path,
                temporary_metadata,
                destination_path,
                sidecar,
            )
            temporary_snapshot = None
            temporary_metadata = None
        else:
            _write_json_exclusive(sidecar, metadata)
        return destination_path
    except Exception:
        if replace and temporary_snapshot is not None:
            _remove_generated_file(temporary_snapshot)
            if temporary_metadata is not None:
                _remove_generated_file(temporary_metadata)
        elif not replace:
            _remove_generated_file(destination_path)
            _remove_generated_file(sidecar)
        raise


def backup_database(
    compose_file: str | Path,
    env_file: str | Path,
    destination: str | Path,
    *,
    database: str,
    application_version: str,
    project_name: str = "bakery-ai",
    user: str = "root",
    popen: Callable[..., Any] = subprocess.Popen,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> Path:
    command = backup_command(
        compose_file,
        env_file,
        database,
        project_name=project_name,
        user=user,
    )
    return create_snapshot(
        command,
        destination,
        database=database,
        application_version=application_version,
        popen=popen,
        now=now,
    )


def restore_snapshot(
    compose_file: str | Path,
    env_file: str | Path,
    snapshot: str | Path,
    *,
    database: str,
    project_name: str = "bakery-ai",
    user: str = "root",
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> None:
    snapshot_path = Path(snapshot)
    validate_snapshot(
        snapshot_path,
        require_metadata=True,
        expected_database=database,
    )
    command = restore_command(
        compose_file,
        env_file,
        database,
        project_name=project_name,
        user=user,
    )
    with snapshot_path.open("rb") as input_handle:
        result = runner(
            command,
            stdin=input_handle,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
        )
    if result.returncode != 0:
        detail = _error_text(result.stderr) or "unknown database client error"
        raise DatabaseOperationError(f"Database restore failed: {detail}")
