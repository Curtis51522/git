from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import secrets
import shutil
import sys
from typing import Callable, Mapping


def _resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _user_data_home(system: str, environment: Mapping[str, str]) -> Path | None:
    if system == "Windows":
        root = environment.get("LOCALAPPDATA")
        return _resolved_path(Path(root) / "BakeryAI") if root else None
    if system == "Darwin":
        root = environment.get("HOME")
        return (
            _resolved_path(Path(root) / "Library" / "Application Support" / "BakeryAI")
            if root
            else None
        )
    root = environment.get("XDG_DATA_HOME")
    if root:
        return _resolved_path(Path(root) / "BakeryAI")
    home = environment.get("HOME")
    return _resolved_path(Path(home) / ".local" / "share" / "BakeryAI") if home else None


def _read_install_manifest(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid Bakery AI install manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid Bakery AI install manifest: {path}")
    return {
        str(key): str(value)
        for key, value in payload.items()
        if isinstance(value, (str, Path)) and str(value).strip()
    }


def find_docker_executable(
    *,
    system: str | None = None,
    environment: Mapping[str, str] | None = None,
    manifest_value: str | None = None,
    exists: Callable[[Path], bool] = Path.is_file,
    which: Callable[[str], str | None] = shutil.which,
) -> str | Path:
    operating_system = system or platform.system()
    values = os.environ if environment is None else environment
    configured = values.get("BAKERY_DOCKER_CLI") or manifest_value
    if configured:
        return _resolved_path(configured)

    candidates: list[Path] = []
    if operating_system == "Windows":
        for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)"):
            root = values.get(variable)
            if root:
                candidates.append(
                    Path(root)
                    / "Docker"
                    / "Docker"
                    / "resources"
                    / "bin"
                    / "docker.exe"
                )
    elif operating_system == "Darwin":
        candidates.extend(
            (
                Path("/Applications/Docker.app/Contents/Resources/bin/docker"),
                Path("/usr/local/bin/docker"),
                Path("/opt/homebrew/bin/docker"),
            )
        )
    for candidate in candidates:
        if exists(candidate):
            return _resolved_path(candidate)

    path_match = which("docker")
    return _resolved_path(path_match) if path_match else "docker"


@dataclass(frozen=True)
class LauncherConfig:
    app_home: Path
    compose_file: Path
    runtime_env_file: Path
    edge_profile_dir: Path
    log_file: Path
    instance_lock_file: Path
    docker_executable: str | Path = "docker"
    project_name: str = "bakery-ai"
    preferred_ports: tuple[int, ...] = tuple(range(8002, 8011))
    dynamic_port_range: str = "49152-65535"
    startup_timeout: float = 180.0
    health_interval: float = 1.0
    http_timeout: float = 3.0

    @classmethod
    def for_app_home(
        cls,
        app_home: str | Path,
        *,
        startup_timeout: float = 180.0,
    ) -> "LauncherConfig":
        home = _resolved_path(app_home)
        return cls(
            app_home=home,
            compose_file=home / "compose.yaml",
            runtime_env_file=home / ".env",
            edge_profile_dir=home / "runtime" / "edge-profile",
            log_file=home / "runtime" / "launcher.log",
            instance_lock_file=home / "runtime" / "launcher.lock",
            docker_executable=find_docker_executable(),
            startup_timeout=startup_timeout,
        )

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        system: str | None = None,
        frozen_executable: str | Path | None = None,
    ) -> "LauncherConfig":
        values = os.environ if environment is None else environment
        operating_system = system or platform.system()
        default_data_home = _user_data_home(operating_system, values)
        explicit_home = values.get("BAKERY_APP_HOME")
        manifest = (
            {}
            if explicit_home
            else _read_install_manifest(
                default_data_home / "install.json" if default_data_home else None
            )
        )
        if explicit_home:
            home = _resolved_path(explicit_home)
        elif manifest.get("app_home"):
            home = _resolved_path(manifest["app_home"])
        elif frozen_executable is not None:
            home = _resolved_path(frozen_executable).parent
        elif getattr(sys, "frozen", False):
            home = _resolved_path(sys.executable).parent
        else:
            raise RuntimeError(
                "Bakery AI installation directory is not configured"
            )

        data_home = _resolved_path(
            values.get("BAKERY_DATA_HOME")
            or manifest.get("data_home")
            or default_data_home
            or home
        )
        compose_file = _resolved_path(
            values.get("BAKERY_COMPOSE_FILE", home / "compose.yaml")
        )
        runtime_env_file = _resolved_path(
            values.get("BAKERY_RUNTIME_ENV", data_home / "runtime.env")
        )
        edge_profile_dir = _resolved_path(
            values.get(
                "BAKERY_EDGE_PROFILE",
                data_home / "runtime" / "edge-profile",
            )
        )
        log_file = _resolved_path(
            values.get(
                "BAKERY_LAUNCHER_LOG",
                data_home / "runtime" / "launcher.log",
            )
        )
        instance_lock_file = _resolved_path(
            values.get(
                "BAKERY_INSTANCE_LOCK",
                data_home / "runtime" / "launcher.lock",
            )
        )
        return cls(
            app_home=home,
            compose_file=compose_file,
            runtime_env_file=runtime_env_file,
            edge_profile_dir=edge_profile_dir,
            log_file=log_file,
            instance_lock_file=instance_lock_file,
            docker_executable=find_docker_executable(
                system=operating_system,
                environment=values,
                manifest_value=manifest.get("docker_cli"),
            ),
        )


def _generated_secret() -> str:
    return secrets.token_urlsafe(48)


def _validate_env_value(value: str) -> str:
    if not value or "\n" in value or "\r" in value:
        raise ValueError("Secret contains invalid environment-file characters")
    return value


def ensure_runtime_env(
    config: LauncherConfig,
    *,
    secret_factory=_generated_secret,
) -> Path:
    path = config.runtime_env_file
    if path.exists():
        try:
            path.chmod(0o600)
        except OSError:
            pass
        return path

    values = {
        "BAKERY_ENV": "production",
        "JWT_SECRET": _validate_env_value(secret_factory()),
        "MYSQL_ROOT_PASSWORD": _validate_env_value(secret_factory()),
        "MYSQL_DATABASE": "bakery_ai",
        "MYSQL_USER": "bakery_app",
        "MYSQL_PASSWORD": _validate_env_value(secret_factory()),
        "BAKERY_FIXED_BUSINESS_DATE": "2026-07-24",
        "LLM_API_KEY": "",
        "LLM_BASE_URL": "",
        "LLM_MODEL": "",
    }
    content = "".join(f"{key}={value}\n" for key, value in values.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        return path
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path
