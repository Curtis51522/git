from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from deployment.release.build_images import build_release, load_dependency_spec
from deployment.release.build_package import assemble_package


MACOS_TARGET = "macos-apple-silicon"
CONTAINER_PLATFORM = "linux/arm64"


def require_native_apple_silicon(*, system: str, machine: str) -> None:
    if system.strip().casefold() != "darwin":
        raise RuntimeError("The native macOS release must be built on macOS")
    if machine.strip().casefold() not in {"arm64", "aarch64"}:
        raise RuntimeError("The native macOS release requires Apple Silicon")


def pyinstaller_commands(
    *,
    project_root: str | Path,
    output_root: str | Path,
    python_executable: str = sys.executable,
) -> tuple[list[str], list[str]]:
    project = Path(project_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    common = [python_executable, "-m", "PyInstaller", "--noconfirm", "--clean"]
    application_output = output / "native-apps"
    return (
        [
            *common,
            "--distpath",
            str(application_output),
            "--workpath",
            str(output / "work-launcher"),
            str(project / "deployment" / "pyinstaller-launcher.spec"),
        ],
        [
            *common,
            "--distpath",
            str(application_output),
            "--workpath",
            str(output / "work-installer"),
            str(project / "deployment" / "pyinstaller-installer.spec"),
        ],
    )


def _require_application_bundle(path: Path) -> Path:
    executable_root = path / "Contents" / "MacOS"
    if not path.is_dir() or not executable_root.is_dir():
        raise FileNotFoundError(f"Native application bundle was not created: {path}")
    if not any(candidate.is_file() for candidate in executable_root.iterdir()):
        raise FileNotFoundError(f"Native application bundle has no executable: {path}")
    return path


def build_native_macos_release(
    *,
    release_version: str,
    project_root: str | Path,
    output_root: str | Path,
    compose: str | Path,
    model: str | Path,
    snapshot: str | Path,
    dependency_spec: str | Path,
    python_executable: str = sys.executable,
    system: str | None = None,
    machine: str | None = None,
    run: Callable[..., Any] = subprocess.run,
    image_builder: Callable[..., Path] = build_release,
    package_builder: Callable[..., Path] = assemble_package,
    dependency_loader: Callable[[str | Path], Mapping[str, object]] = (
        load_dependency_spec
    ),
) -> Path:
    require_native_apple_silicon(
        system=system or platform.system(),
        machine=machine or platform.machine(),
    )
    project = Path(project_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for command in pyinstaller_commands(
        project_root=project,
        output_root=output,
        python_executable=python_executable,
    ):
        run(command, check=True, cwd=project, shell=False)

    native_apps = output / "native-apps"
    launcher = _require_application_bundle(native_apps / "Bakery AI.app")
    installer = _require_application_bundle(native_apps / "BakeryAI Installer.app")
    release_root = output / "macos-arm64-release"
    installers = dependency_loader(dependency_spec)
    image_builder(
        release_version=release_version,
        platform=CONTAINER_PLATFORM,
        output_directory=release_root,
        compose=Path(compose),
        model=Path(model),
        snapshot=Path(snapshot),
        launcher=launcher,
        installers=installers,
        context=project,
    )
    destination = output / f"BakeryAI-{release_version}-macOS-arm64-Offline.zip"
    return package_builder(
        target=MACOS_TARGET,
        release_root=release_root,
        installer=installer,
        destination=destination,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the native Bakery AI Apple Silicon offline release."
    )
    parser.add_argument("--release-version", required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--compose", type=Path, default=Path("compose.yaml"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/yolo/best.pt"),
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--dependency-spec", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    build_native_macos_release(
        release_version=options.release_version,
        project_root=options.project_root,
        output_root=options.output_root,
        compose=options.compose,
        model=options.model,
        snapshot=options.snapshot,
        dependency_spec=options.dependency_spec,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
