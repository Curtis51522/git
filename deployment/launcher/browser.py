from __future__ import annotations

import logging
import os
from pathlib import Path
import platform
import subprocess
from typing import Callable, Mapping
import webbrowser


LOGGER = logging.getLogger(__name__)


def _show_dialog(title: str, message: str, *, error: bool) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            if error:
                messagebox.showerror(title, message, parent=root)
            else:
                messagebox.showinfo(title, message, parent=root)
        finally:
            root.destroy()
        return
    except Exception as exc:
        LOGGER.warning("The graphical notification could not be displayed: %s", exc)

    if platform.system() == "Windows":
        try:
            import ctypes

            icon = 0x10 if error else 0x40
            ctypes.windll.user32.MessageBoxW(None, message, title, icon)
            return
        except (AttributeError, OSError) as exc:
            LOGGER.warning("The Windows notification could not be displayed: %s", exc)
    elif platform.system() == "Darwin":
        script = (
            'display alert "' + title.replace('"', '\\"') + '" message "'
            + message.replace('"', '\\"') + '"'
        )
        try:
            subprocess.run(
                ["/usr/bin/osascript", "-e", script],
                check=False,
                capture_output=True,
                shell=False,
            )
            return
        except OSError as exc:
            LOGGER.warning("The macOS notification could not be displayed: %s", exc)


def show_information(title: str, message: str) -> None:
    _show_dialog(title, message, error=False)


def show_error(title: str, message: str) -> None:
    _show_dialog(title, message, error=True)


def wait_for_browser_session(
    *,
    dialog: Callable[[str, str], None] = show_information,
) -> None:
    dialog(
        "Bakery AI",
        "Bakery AI is open in your browser. Click OK to stop it.",
    )


def edge_arguments(edge_path: Path, url: str, profile_dir: Path) -> list[str]:
    return [
        str(edge_path),
        f"--app={url}",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--edge-skip-compat-layer-relaunch",
    ]


def find_edge(
    *,
    system: str | None = None,
    environment: Mapping[str, str] | None = None,
    exists: Callable[[Path], bool] = Path.exists,
) -> Path | None:
    operating_system = system or platform.system()
    values = os.environ if environment is None else environment
    candidates: list[Path] = []
    if operating_system == "Windows":
        for variable in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
            root = values.get(variable)
            if root:
                candidates.append(
                    Path(root) / "Microsoft" / "Edge" / "Application" / "msedge.exe"
                )
    elif operating_system == "Darwin":
        candidates.append(
            Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge")
        )
        home = values.get("HOME")
        if home:
            candidates.append(
                Path(home)
                / "Applications"
                / "Microsoft Edge.app"
                / "Contents"
                / "MacOS"
                / "Microsoft Edge"
            )
    return next((candidate for candidate in candidates if exists(candidate)), None)


def launch_app(
    url: str,
    profile_dir: Path,
    *,
    edge_path: Path | None = None,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    browser_open: Callable[[str], bool] = webbrowser.open,
    logger: logging.Logger = LOGGER,
) -> subprocess.Popen | None:
    selected_edge = edge_path or find_edge()
    if selected_edge is not None:
        profile_dir.mkdir(parents=True, exist_ok=True)
        try:
            return popen(edge_arguments(selected_edge, url, profile_dir), shell=False)
        except OSError as exc:
            logger.warning("Edge App Mode failed; using browser fallback: %s", exc)
    else:
        logger.warning("Microsoft Edge was not found; using browser fallback")
    browser_open(url)
    return None
