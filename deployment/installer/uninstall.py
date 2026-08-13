from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import re
import subprocess
from typing import Callable


class RemovalMode(str, Enum):
    STANDARD = "standard"
    COMPLETE = "complete"


class ConfirmationRequired(RuntimeError):
    pass


class RemovalError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemovalPaths:
    application_files: tuple[Path, ...]
    launcher_files: tuple[Path, ...]
    configuration_files: tuple[Path, ...]
    backup_files: tuple[Path, ...]
    database_volume: str


@dataclass(frozen=True)
class RemovalPlan:
    mode: RemovalMode
    targets: frozenset[str]
    exact_files: tuple[Path, ...]
    database_volume: str | None
    preview_text: str
    confirmation_phrase: str | None


@dataclass(frozen=True)
class RemovalResult:
    removed_files: tuple[Path, ...]
    skipped_paths: tuple[Path, ...]
    removed_volume: bool


def removal_targets(mode: RemovalMode) -> set[str]:
    targets = {"application"}
    if mode is RemovalMode.COMPLETE:
        targets.update({"database", "backups", "configuration"})
    return targets


_CONTAINER_ID = re.compile(r"[0-9a-f]{12,64}", re.IGNORECASE)
_MANAGED_VOLUME = re.compile(
    r"bakery-ai(?:[-_][a-z0-9][a-z0-9_.-]*)+",
    re.IGNORECASE,
)


def _absolute_without_resolving(path: Path) -> Path:
    expanded = os.path.expanduser(os.fspath(path))
    return Path(os.path.abspath(expanded))


def _unique_absolute(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    ordered: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        absolute = _absolute_without_resolving(path)
        if absolute not in seen:
            seen.add(absolute)
            ordered.append(absolute)
    return tuple(ordered)


def build_removal_plan(mode: RemovalMode, paths: RemovalPaths) -> RemovalPlan:
    application_files = _unique_absolute(
        paths.application_files + paths.launcher_files
    )
    protected_files = set(
        _unique_absolute(paths.configuration_files + paths.backup_files)
    )
    files = tuple(
        path for path in application_files if path not in protected_files
    )
    volume: str | None = None
    phrase: str | None = None
    if mode is RemovalMode.COMPLETE:
        if not paths.database_volume.strip():
            raise ValueError("Database volume is required for complete removal")
        files = _unique_absolute(
            application_files
            + paths.configuration_files
            + paths.backup_files
        )
        volume = paths.database_volume.strip()
        if not _MANAGED_VOLUME.fullmatch(volume):
            raise ValueError(
                "Complete removal is limited to a Bakery AI managed volume"
            )
        phrase = f"REMOVE {volume}"
    exact_files = _unique_absolute(files)
    lines = [f"Removal mode: {mode.value}", "Exact files:"]
    lines.extend(f"- {path}" for path in exact_files)
    lines.append(f"Docker volume: {volume or 'retained'}")
    return RemovalPlan(
        mode=mode,
        targets=frozenset(removal_targets(mode)),
        exact_files=exact_files,
        database_volume=volume,
        preview_text="\n".join(lines),
        confirmation_phrase=phrase,
    )


def execute_removal(
    plan: RemovalPlan,
    *,
    first_confirmation: bool,
    second_confirmation: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    docker_executable: str = "docker",
) -> RemovalResult:
    if not first_confirmation:
        raise ConfirmationRequired("The first confirmation is required")
    if (
        plan.mode is RemovalMode.COMPLETE
        and second_confirmation != plan.confirmation_phrase
    ):
        raise ConfirmationRequired(
            "The second confirmation must match the displayed removal phrase"
        )

    skipped = tuple(
        path
        for path in plan.exact_files
        if path.is_dir() and not path.is_symlink()
    )

    removed_volume = False
    if plan.database_volume is not None:
        _remove_managed_volume(
            plan.database_volume,
            runner=runner,
            docker_executable=docker_executable,
        )
        removed_volume = True

    removed: list[Path] = []
    for path in plan.exact_files:
        if path.is_dir() and not path.is_symlink():
            continue
        if not path.exists() and not path.is_symlink():
            continue
        try:
            path.unlink()
        except OSError as exc:
            raise RemovalError(f"Could not remove file: {path}") from exc
        removed.append(path)

    return RemovalResult(
        removed_files=tuple(removed),
        skipped_paths=skipped,
        removed_volume=removed_volume,
    )


def _run_checked(
    command: list[str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    failure_message: str,
) -> subprocess.CompletedProcess[str]:
    result = runner(
        command,
        capture_output=True,
        text=True,
        shell=False,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        raise RemovalError(f"{failure_message}: {detail}")
    return result


def _remove_managed_volume(
    volume: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
    docker_executable: str,
) -> None:
    if not _MANAGED_VOLUME.fullmatch(volume):
        raise RemovalError("Refusing to remove an unmanaged Docker volume")
    attached = _run_checked(
        [
            docker_executable,
            "ps",
            "-aq",
            "--filter",
            f"volume={volume}",
        ],
        runner=runner,
        failure_message=f"Could not inspect containers attached to {volume}",
    )
    container_ids = tuple(
        value.strip() for value in attached.stdout.splitlines() if value.strip()
    )
    if any(not _CONTAINER_ID.fullmatch(value) for value in container_ids):
        raise RemovalError("Docker returned an invalid attached-container identifier")
    for container_id in container_ids:
        _run_checked(
            [docker_executable, "stop", container_id],
            runner=runner,
            failure_message=f"Could not stop container {container_id}",
        )
        _run_checked(
            [docker_executable, "rm", container_id],
            runner=runner,
            failure_message=f"Could not remove container {container_id}",
        )
    _run_checked(
        [docker_executable, "volume", "rm", volume],
        runner=runner,
        failure_message=f"Could not remove Docker volume {volume}",
    )
