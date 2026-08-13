from __future__ import annotations

import logging
import os
from pathlib import Path
import platform
from typing import Callable

from deployment.launcher.browser import (
    launch_app,
    show_error,
    show_information,
    wait_for_browser_session,
)
from deployment.launcher.config import LauncherConfig
from deployment.launcher.docker_runtime import DockerRuntime, StartupError


LOGGER = logging.getLogger(__name__)


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if platform.system() == "Windows":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        )
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        process_query_limited_information = 0x1000
        still_active = 259
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class SingleInstanceLock:
    def __init__(
        self,
        path: Path,
        *,
        pid: int | None = None,
        process_checker: Callable[[int], bool] = _process_exists,
    ) -> None:
        self.path = path
        self.pid = os.getpid() if pid is None else pid
        self.process_checker = process_checker
        self.acquired = False

    def _owner_pid(self) -> int | None:
        try:
            return int(self.path.read_text(encoding="ascii").strip())
        except (OSError, UnicodeError, ValueError):
            return None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _attempt in range(3):
            try:
                descriptor = os.open(
                    self.path,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
            except FileExistsError:
                owner_pid = self._owner_pid()
                if owner_pid is not None and self.process_checker(owner_pid):
                    return False
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    continue
                except OSError:
                    return False
                continue
            with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
                handle.write(str(self.pid))
            self.acquired = True
            return True
        return False

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            if self._owner_pid() == self.pid:
                self.path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self.acquired = False


def _configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8")],
        force=True,
    )


def run_launcher(
    config: LauncherConfig,
    *,
    runtime: DockerRuntime | None = None,
    browser_launcher: Callable[..., object | None] = launch_app,
    fallback_waiter: Callable[[], None] = wait_for_browser_session,
    instance_lock: SingleInstanceLock | None = None,
    info_notifier: Callable[[str, str], None] = show_information,
) -> int:
    lock = instance_lock or SingleInstanceLock(config.instance_lock_file)
    if not lock.acquire():
        info_notifier(
            "Bakery AI is already running",
            "Close the existing Bakery AI window before starting another one.",
        )
        return 0

    docker_runtime = runtime or DockerRuntime(config)
    try:
        state = docker_runtime.start()
        process = browser_launcher(state.url, config.edge_profile_dir)
        if process is None:
            fallback_waiter()
        else:
            process.wait()
        return 0
    finally:
        try:
            docker_runtime.stop()
        finally:
            lock.release()


def main(
    *,
    config_factory: Callable[[], LauncherConfig] = LauncherConfig.from_environment,
    runtime_factory: Callable[[LauncherConfig], DockerRuntime] = DockerRuntime,
    error_notifier: Callable[[str, str], None] = show_error,
) -> int:
    try:
        config = config_factory()
        _configure_logging(config.log_file)
        return run_launcher(config, runtime=runtime_factory(config))
    except StartupError as exc:
        LOGGER.error("Bakery AI could not start: %s", exc)
        error_notifier("Bakery AI could not start", str(exc))
        return 1
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.error("Bakery AI configuration failed: %s", exc)
        error_notifier("Bakery AI could not start", str(exc))
        return 1
    except KeyboardInterrupt:
        LOGGER.info("Bakery AI launcher interrupted")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
