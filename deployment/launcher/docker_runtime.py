from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import ctypes
import logging
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
from typing import Callable, Iterable, Iterator, Mapping
from urllib.error import URLError
from urllib.request import urlopen

from deployment.launcher.config import LauncherConfig, ensure_runtime_env


LOGGER = logging.getLogger(__name__)
MAX_DIAGNOSTIC_CHARACTERS = 65_536
WINDOWS_ERROR_MODE = 0x0001 | 0x0002 | 0x8000


class StartupError(RuntimeError):
    def __init__(self, message: str, diagnostics: str = "") -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class RuntimeState:
    port: int
    url: str


def port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def preferred_port(
    candidates: Iterable[int],
    dynamic_range: str = "49152-65535",
) -> str:
    for port in candidates:
        if port_available(port):
            return str(port)
    return dynamic_range


def published_port(output: str) -> int:
    try:
        value = output.strip().rsplit(":", 1)[-1]
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Docker returned an invalid published port") from exc
    if not 1 <= port <= 65_535:
        raise ValueError("Docker returned an invalid published port")
    return port


def compose_command(config: LauncherConfig, *arguments: str) -> list[str]:
    return [
        str(config.docker_executable),
        "compose",
        "--project-name",
        config.project_name,
        "--file",
        str(config.compose_file),
        "--env-file",
        str(config.runtime_env_file),
        *arguments,
    ]


def find_docker_desktop_executable(
    docker_executable: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> Path | None:
    values = os.environ if environment is None else environment
    candidates: list[Path] = []
    docker_path = Path(docker_executable).expanduser()
    if docker_path.is_absolute():
        install_root = docker_path.resolve().parent.parent.parent
        candidates.extend(
            (
                install_root / "frontend" / "Docker Desktop.exe",
                install_root / "Docker Desktop.exe",
            )
        )
    local_app_data = values.get("LOCALAPPDATA")
    if local_app_data:
        local_root = Path(local_app_data)
        candidates.extend(
            (
                local_root
                / "Programs"
                / "DockerDesktop"
                / "frontend"
                / "Docker Desktop.exe",
                local_root
                / "Programs"
                / "Docker"
                / "Docker"
                / "Docker Desktop.exe",
            )
        )
    program_files = values.get("PROGRAMFILES")
    if program_files:
        candidates.append(
            Path(program_files) / "Docker" / "Docker" / "Docker Desktop.exe"
        )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _command_failure_detail(
    result: subprocess.CompletedProcess[str],
) -> str:
    detail = (result.stderr or result.stdout or "unknown error").strip()
    code = int(result.returncode)
    return f"exit code {code} (0x{code & 0xFFFFFFFF:08X}): {detail}"


@contextmanager
def _suppress_windows_error_dialogs(system: str) -> Iterator[None]:
    if system != "Windows":
        yield
        return
    try:
        kernel32 = ctypes.windll.kernel32
        previous_mode = kernel32.SetErrorMode(WINDOWS_ERROR_MODE)
    except (AttributeError, OSError):
        yield
        return
    try:
        yield
    finally:
        kernel32.SetErrorMode(previous_mode)


def _http_ready(url: str, timeout: float) -> bool:
    try:
        with urlopen(url, timeout=timeout) as response:
            return getattr(response, "status", 200) == 200
    except (OSError, URLError):
        return False


class DockerRuntime:
    def __init__(
        self,
        config: LauncherConfig,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        health_checker: Callable[[str], bool] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        logger: logging.Logger = LOGGER,
        system: str | None = None,
        desktop_launcher: Callable[..., object] = subprocess.Popen,
    ) -> None:
        self.config = config
        self.runner = runner
        self.health_checker = health_checker or (
            lambda url: _http_ready(url, self.config.http_timeout)
        )
        self.sleeper = sleeper
        self.clock = clock
        self.logger = logger
        self.system = system or platform.system()
        self.desktop_launcher = desktop_launcher
        self._docker_desktop_process: object | None = None

    def _windows_subprocess_options(self) -> dict[str, object]:
        if self.system != "Windows":
            return {}
        return {
            "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
        }

    def _run(
        self,
        arguments: list[str],
        *,
        environment: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        options: dict[str, object] = {
            "capture_output": True,
            "text": True,
            "shell": False,
            "check": False,
            "env": environment,
            **self._windows_subprocess_options(),
        }
        if timeout is not None:
            options["timeout"] = timeout
        with _suppress_windows_error_dialogs(self.system):
            return self.runner(arguments, **options)

    def _require_success(
        self,
        arguments: list[str],
        *,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        result = self._run(arguments, environment=environment)
        if result.returncode != 0:
            detail = _command_failure_detail(result)
            raise StartupError(f"Command failed: {self._redact(detail)}")
        return result

    def _start_docker_desktop(self) -> bool:
        desktop = find_docker_desktop_executable(self.config.docker_executable)
        if desktop is None:
            return False
        options: dict[str, object] = {
            "shell": False,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            **self._windows_subprocess_options(),
        }
        try:
            with _suppress_windows_error_dialogs(self.system):
                self._docker_desktop_process = self.desktop_launcher(
                    [str(desktop)], **options
                )
        except OSError as exc:
            self.logger.warning("Docker Desktop could not be started: %s", exc)
            return False
        return True

    def _wait_for_docker_engine(self) -> None:
        command = [
            str(self.config.docker_executable),
            "info",
            "--format",
            "{{.ServerVersion}}",
        ]
        probe_timeout = min(10.0, self.config.startup_timeout)
        try:
            result = self._run(command, timeout=probe_timeout)
            last_detail = _command_failure_detail(result)
        except (OSError, subprocess.TimeoutExpired) as exc:
            result = None
            last_detail = str(exc)
        if result is not None and result.returncode == 0 and result.stdout.strip():
            return

        deadline = self.clock() + self.config.startup_timeout
        desktop_started = self.system == "Windows" and self._start_docker_desktop()
        while True:
            if self.clock() >= deadline:
                action = (
                    " Docker Desktop could not be started automatically."
                    if self.system == "Windows" and not desktop_started
                    else ""
                )
                raise StartupError(
                    "Docker Engine did not become ready before the timeout: "
                    f"{self._redact(last_detail)}.{action}"
                )
            self.sleeper(self.config.health_interval)
            try:
                result = self._run(command, timeout=probe_timeout)
                last_detail = _command_failure_detail(result)
            except (OSError, subprocess.TimeoutExpired) as exc:
                result = None
                last_detail = str(exc)
            if result is not None and result.returncode == 0 and result.stdout.strip():
                return

    def start(self) -> RuntimeState:
        ensure_runtime_env(self.config)
        environment = os.environ.copy()
        environment["BAKERY_HOST_PORT_RANGE"] = preferred_port(
            self.config.preferred_ports,
            self.config.dynamic_port_range,
        )
        try:
            self._wait_for_docker_engine()
            self._require_success(
                compose_command(self.config, "up", "-d"),
                environment=environment,
            )
            port_result = self._require_success(
                compose_command(self.config, "port", "app", "8002"),
                environment=environment,
            )
            port = published_port(port_result.stdout)
            state = RuntimeState(port=port, url=f"http://127.0.0.1:{port}")
            self._wait_until_ready(state.url)
            return state
        except Exception as exc:
            diagnostics = self.collect_diagnostics(environment=environment)
            self.logger.error("Bakery AI startup failed.\n%s", diagnostics)
            if isinstance(exc, StartupError):
                raise StartupError(str(exc), diagnostics) from exc
            raise StartupError(str(exc), diagnostics) from exc

    def _wait_until_ready(self, base_url: str) -> None:
        endpoints = ("/health", "/s5-health")
        deadline = self.clock() + self.config.startup_timeout
        while True:
            statuses = [
                self.health_checker(f"{base_url}{endpoint}")
                for endpoint in endpoints
            ]
            if all(statuses):
                return
            if self.clock() >= deadline:
                raise StartupError("Bakery AI services did not become ready in time")
            self.sleeper(self.config.health_interval)

    def collect_diagnostics(
        self,
        *,
        environment: dict[str, str] | None = None,
    ) -> str:
        sections = []
        commands = (
            ("Compose status", compose_command(self.config, "ps")),
            (
                "Service logs",
                compose_command(
                    self.config,
                    "logs",
                    "--no-color",
                    "--tail",
                    "200",
                    "mysql",
                    "app",
                    "s5",
                ),
            ),
        )
        limits = (8_192, MAX_DIAGNOSTIC_CHARACTERS - 8_192)
        for (title, command), limit in zip(commands, limits):
            try:
                result = self._run(command, environment=environment)
                content = "\n".join(
                    part for part in (result.stdout, result.stderr) if part
                ).strip()
            except OSError as exc:
                content = str(exc)
            sections.append(f"[{title}]\n{(content or 'No output')[-limit:]}")
        return self._redact("\n\n".join(sections))[:MAX_DIAGNOSTIC_CHARACTERS]

    def _redact(self, content: str) -> str:
        try:
            lines = self.config.runtime_env_file.read_text(
                encoding="utf-8"
            ).splitlines()
        except OSError:
            return content
        secrets_to_redact = []
        for line in lines:
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, value = line.split("=", 1)
            if value and key.endswith(("SECRET", "PASSWORD", "API_KEY")):
                secrets_to_redact.append(value)
        for value in sorted(secrets_to_redact, key=len, reverse=True):
            content = content.replace(value, "[REDACTED]")
        return content

    def stop(self) -> None:
        try:
            result = self._run(compose_command(self.config, "stop"))
        except OSError as exc:
            self.logger.warning(
                "Could not stop Bakery AI containers: %s",
                self._redact(str(exc)),
            )
            return
        if result.returncode != 0:
            detail = _command_failure_detail(result)
            self.logger.warning(
                "Could not stop Bakery AI containers: %s",
                self._redact(detail),
            )
