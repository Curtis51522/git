from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import getpass
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any, Callable, Mapping
from urllib.request import urlopen

from deployment.installer.main import InstallLocations, LifecycleRequest, Manifest
from deployment.launcher.config import LauncherConfig, ensure_runtime_env, find_docker_executable
from deployment.launcher.database import backup_database, restore_snapshot
from deployment.launcher.docker_runtime import DockerRuntime, compose_command
from deployment.release.build_images import load_command
from deployment.release.build_package import verify_checksum_manifest
from deployment.release.verify_payload import verify_payload


class InstallerOperationError(RuntimeError):
    pass


class RebootRequired(InstallerOperationError):
    pass


@dataclass(frozen=True)
class ReleaseBackup:
    root: Path
    files: tuple[tuple[Path, Path], ...]


def _safe_files(root: Path) -> tuple[Path, ...]:
    if not root.is_dir() or root.is_symlink():
        return ()
    files: list[Path] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories[:] = [
            name
            for name in directories
            if not (current_path / name).is_symlink()
        ]
        for name in names:
            path = current_path / name
            if path.is_file() and not path.is_symlink():
                files.append(path)
    return tuple(sorted(files, key=lambda path: path.as_posix().casefold()))


def _atomic_copy(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise InstallerOperationError(f"Required file is unavailable: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.installing")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_path(payload_root: Path, manifest: Manifest, name: str) -> Path:
    artifacts = manifest.get("artifacts")
    entry = artifacts.get(name) if isinstance(artifacts, Mapping) else None
    if not isinstance(entry, Mapping):
        raise InstallerOperationError(f"Release artifact is missing: {name}")
    relative = Path(str(entry.get("path", "")))
    candidate = (payload_root / relative).resolve()
    try:
        candidate.relative_to(payload_root.resolve())
    except ValueError as exc:
        raise InstallerOperationError(f"Unsafe release artifact path: {name}") from exc
    if not candidate.is_file() or candidate.is_symlink():
        raise InstallerOperationError(f"Release artifact is unavailable: {name}")
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PlatformInstallerOperations:
    def __init__(
        self,
        request: LifecycleRequest,
        *,
        system: str | None = None,
        runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
        popen: Callable[..., Any] = subprocess.Popen,
        health_opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.request = request
        self.system = system or platform.system()
        self.runner = runner
        self.popen = popen
        self.health_opener = health_opener
        self.payload_root = request.payload_root
        self.manifest: Manifest = {}
        self._runtime_state = None

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[Any]:
        result = self.runner(
            command,
            capture_output=True,
            text=True,
            shell=False,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown error").strip()
            raise InstallerOperationError(f"Command failed: {detail}")
        return result

    def _installer_entry(self, name: str) -> tuple[Path, Mapping[str, Any]]:
        artifacts = self.manifest.get("artifacts")
        installers = (
            artifacts.get("installers") if isinstance(artifacts, Mapping) else None
        )
        entry = installers.get(name) if isinstance(installers, Mapping) else None
        if not isinstance(entry, Mapping):
            raise InstallerOperationError(
                f"Required offline installer is unavailable: {name}"
            )
        relative = Path(str(entry.get("path", "")))
        path = (self.payload_root / relative).resolve()
        try:
            path.relative_to(self.payload_root.resolve())
        except ValueError as exc:
            raise InstallerOperationError(
                f"Unsafe offline installer path: {name}"
            ) from exc
        if not path.is_file() or path.is_symlink():
            raise InstallerOperationError(
                f"Required offline installer is unavailable: {name}"
            )
        return path, entry

    def _verify_windows_installer(self, name: str, *, msi: bool) -> Path:
        from deployment.installer.windows import (
            read_authenticode_publisher,
            validate_signed_msi_package,
        )

        path, entry = self._installer_entry(name)
        expected_hash = str(entry.get("sha256", "")).lower()
        publisher = str(entry.get("publisher", "")).strip()
        if len(expected_hash) != 64 or not publisher:
            raise InstallerOperationError(
                f"Offline installer verification evidence is incomplete: {name}"
            )
        def signature_reader(source: Path) -> tuple[bool, str]:
            return read_authenticode_publisher(source, runner=self.runner)
        if msi:
            return validate_signed_msi_package(
                path,
                expected_sha256=expected_hash,
                allowed_publishers={publisher},
                signature_reader=signature_reader,
            )
        if _sha256(path) != expected_hash:
            raise InstallerOperationError(
                f"Offline installer SHA-256 verification failed: {name}"
            )
        signature_valid, actual_publisher = signature_reader(path)
        expected_normalized = publisher.casefold().rstrip(". ")
        actual_normalized = actual_publisher.casefold().rstrip(". ")
        if not signature_valid or actual_normalized != expected_normalized:
            raise InstallerOperationError(
                f"Offline installer signature verification failed: {name}"
            )
        return path

    def _wait_for_docker(self, docker_cli: Path) -> None:
        from deployment.installer.windows import wait_for_docker_engine

        wait_for_docker_engine(
            docker_cli,
            timeout=180.0,
            interval=2.0,
            runner=self.runner,
        )

    @staticmethod
    def _existing_parent(path: Path) -> Path:
        candidate = path.expanduser()
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        return candidate

    def _command_succeeds(self, command: list[str]) -> bool:
        try:
            result = self.runner(
                command,
                capture_output=True,
                text=True,
                shell=False,
                check=False,
            )
        except OSError:
            return False
        return result.returncode == 0

    def _windows_prerequisites(self) -> None:
        from deployment.installer.windows import (
            ContinuationState,
            docker_desktop_start_command,
            docker_install_command,
            edge_msi_install_command,
            find_windows_docker_cli,
            resume_once_command,
            wsl_feature_commands,
            wsl_msi_install_command,
            write_continuation_state,
        )

        if not self._command_succeeds(["wsl.exe", "--version"]):
            wsl_package = self._verify_windows_installer("wsl", msi=True)
            self._run(
                wsl_msi_install_command(
                    wsl_package,
                    authorization_granted=True,
                )
            )
            for command in wsl_feature_commands():
                self._run(command)
            manifest = self.payload_root / "release.json"
            continuation = self.request.locations.data_root / "installer" / "continuation.json"
            write_continuation_state(
                continuation,
                ContinuationState(
                    stage="wsl_reboot",
                    target=self.request.target,
                    install_root=str(self.request.locations.application_root),
                    payload_root=str(self.payload_root),
                    data_root=str(self.request.locations.data_root),
                    backup_root=str(self.request.locations.backup_root),
                    launcher_path=str(self.request.locations.launcher_path),
                    runtime_env_path=str(self.request.locations.runtime_env_path),
                    database_volume=self.request.locations.database_volume,
                    manifest_sha256=_sha256(manifest),
                ),
            )
            self._run(resume_once_command(sys.executable, continuation))
            raise RebootRequired(
                "Windows must restart once to complete the WSL 2 installation"
            )

        docker_cli = find_windows_docker_cli()
        if docker_cli is None:
            docker_package = self._verify_windows_installer(
                "docker_desktop",
                msi=False,
            )
            self._run(docker_install_command(docker_package))
            docker_cli = find_windows_docker_cli()
        if docker_cli is None:
            raise InstallerOperationError(
                "Docker Desktop installation finished but docker.exe was not found. "
                "Start Docker Desktop once, then run Bakery AI Setup again."
            )
        if not self._command_succeeds([str(docker_cli), "info"]):
            desktop_candidates: list[Path] = []
            local_app_data = os.environ.get("LOCALAPPDATA")
            if local_app_data:
                desktop_candidates.append(
                    Path(local_app_data)
                    / "Programs"
                    / "DockerDesktop"
                    / "Docker Desktop.exe"
                )
            for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)"):
                root = os.environ.get(variable)
                if root:
                    desktop_candidates.append(
                        Path(root) / "Docker" / "Docker" / "Docker Desktop.exe"
                    )
            desktop = next(
                (path for path in desktop_candidates if path.is_file()),
                None,
            )
            if desktop is None:
                raise InstallerOperationError(
                    "Docker Desktop is installed but cannot be started automatically. "
                    "Start Docker Desktop, wait until it is ready, then run Bakery AI "
                    "Setup again."
                )
            self.popen(
                docker_desktop_start_command(desktop),
                shell=False,
            )
            self._wait_for_docker(docker_cli)

        edge_candidates = []
        for variable in ("PROGRAMFILES", "PROGRAMFILES(X86)"):
            root = os.environ.get(variable)
            if root:
                edge_candidates.append(
                    Path(root)
                    / "Microsoft"
                    / "Edge"
                    / "Application"
                    / "msedge.exe"
                )
        if not any(path.is_file() for path in edge_candidates):
            edge_package = self._verify_windows_installer(
                "microsoft_edge",
                msi=True,
            )
            self._run(
                edge_msi_install_command(
                    edge_package,
                    authorization_granted=True,
                )
            )

    def _macos_prerequisites(self) -> None:
        from deployment.installer.macos import (
            check_macos_system,
            install_docker_desktop,
            install_edge_package,
            validate_macos_payload,
        )

        memory = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        disk_root = self._existing_parent(
            self.request.locations.application_root.parent
        )
        free_disk = shutil.disk_usage(disk_root).free
        report = check_macos_system(
            machine=platform.machine(),
            macos_version=platform.mac_ver()[0],
            total_memory_bytes=memory,
            free_disk_bytes=free_disk,
        )
        if not report.supported:
            raise InstallerOperationError("This macOS release is not supported")
        payload = validate_macos_payload(self.payload_root)
        docker_cli = find_docker_executable(system="Darwin")
        if docker_cli == "docker" or not Path(docker_cli).is_file():
            install_docker_desktop(
                payload.docker_dmg,
                getpass.getuser(),
                authorization_granted=True,
                run=self.runner,
            )
            docker_cli = find_docker_executable(system="Darwin")
        if docker_cli == "docker" or not Path(docker_cli).is_file():
            raise InstallerOperationError(
                "Docker Desktop installation finished but the Docker CLI was not "
                "found. Start Docker Desktop once, then run Bakery AI Setup again."
            )
        if not self._command_succeeds([str(docker_cli), "info"]):
            self._run(["/usr/bin/open", "-a", "Docker"])
            self._wait_for_docker(Path(docker_cli))
        if payload.edge_binary is None and payload.edge_installer is not None:
            install_edge_package(
                payload.edge_installer,
                authorization_granted=True,
                run=self.runner,
            )

    def _config(self, locations: InstallLocations) -> LauncherConfig:
        return LauncherConfig(
            app_home=locations.application_root,
            compose_file=locations.application_root / "compose.yaml",
            runtime_env_file=locations.runtime_env_path,
            edge_profile_dir=locations.data_root / "runtime" / "edge-profile",
            log_file=locations.data_root / "runtime" / "launcher.log",
            instance_lock_file=locations.data_root / "runtime" / "launcher.lock",
            docker_executable=find_docker_executable(system=self.system),
        )

    def _health_ready(self, url: str) -> bool:
        try:
            with self.health_opener(url, timeout=5) as response:
                return getattr(response, "status", 200) == 200
        except OSError:
            return False

    def _runtime(self, locations: InstallLocations) -> DockerRuntime:
        return DockerRuntime(
            self._config(locations),
            runner=self.runner,
            health_checker=self._health_ready,
        )

    def verify_payload(self, payload_root: Path, target: str) -> Manifest:
        package_root = payload_root.parent
        if (package_root / "SHA256SUMS.txt").is_file():
            verify_checksum_manifest(package_root)
        manifest = verify_payload(
            payload_root,
            target=target,
            manifest_name="release.json",
        )
        self.payload_root = payload_root
        self.manifest = manifest
        return manifest

    def check_prerequisites(self, target: str, manifest: Manifest) -> None:
        machine = platform.machine().casefold()
        if target == "windows-x64":
            if self.system != "Windows" or machine not in {"amd64", "x86_64"}:
                raise InstallerOperationError("This release requires Windows x64")
            self._windows_prerequisites()
        elif target == "macos-apple-silicon":
            if self.system != "Darwin" or machine not in {"arm64", "aarch64"}:
                raise InstallerOperationError(
                    "This release requires Apple Silicon macOS"
                )
            self._macos_prerequisites()
        else:
            raise InstallerOperationError(f"Unsupported install target: {target}")
        compose = _artifact_path(self.payload_root, manifest, "compose")
        required_mount = "./deployment/database/init:/docker-entrypoint-initdb.d:ro"
        if required_mount not in compose.read_text(encoding="utf-8"):
            raise InstallerOperationError(
                "Compose does not use the packaged database initialization path"
            )

    def copy_payload(
        self,
        payload_root: Path,
        locations: InstallLocations,
        manifest: Manifest,
    ) -> None:
        del manifest
        for source in _safe_files(payload_root):
            relative = source.relative_to(payload_root)
            if relative.parts and relative.parts[0] == "installers":
                continue
            _atomic_copy(source, locations.application_root / relative)

    def load_images(self, payload_root: Path, manifest: Manifest) -> None:
        archive = _artifact_path(payload_root, manifest, "image_archive")
        platform_value = str(manifest.get("platform", ""))
        command = load_command(platform_value, archive)
        docker_cli = find_docker_executable(system=self.system)
        if docker_cli == "docker" or not Path(docker_cli).is_file():
            raise InstallerOperationError(
                "The verified Docker CLI is unavailable. Start Docker Desktop and "
                "run Bakery AI Setup again."
            )
        command[0] = str(docker_cli)
        self._run(command)

    def initialize_runtime(self, locations: InstallLocations) -> None:
        config = self._config(locations)
        ensure_runtime_env(config)
        locations.data_root.mkdir(parents=True, exist_ok=True)
        install_manifest = {
            "app_home": str(locations.application_root),
            "data_home": str(locations.data_root),
            "docker_cli": str(config.docker_executable),
            "release_version": str(self.manifest.get("release_version", "")),
            "schema_version": 1,
            "target": self.request.target,
        }
        destination = locations.data_root / "install.json"
        destination.write_text(
            json.dumps(install_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    def initialize_database(self, locations: InstallLocations) -> None:
        config = self._config(locations)
        self._run(compose_command(config, "up", "-d", "mysql"))

    def stop_runtime(self, locations: InstallLocations) -> None:
        config = self._config(locations)
        self._run(compose_command(config, "down", "--remove-orphans"))

    def install_launcher(
        self,
        payload_root: Path,
        locations: InstallLocations,
        manifest: Manifest,
    ) -> None:
        del manifest
        launcher_root = payload_root / "launcher"
        if self.request.target == "windows-x64":
            from deployment.installer.windows import (
                create_windows_shortcut,
                terminate_running_bakery_launcher,
                windows_desktop_directory,
            )

            source = launcher_root / "Bakery AI.exe"
            icon_source = launcher_root / "bakery-ai.ico"
            icon_destination = locations.launcher_path.with_name("bakery-ai.ico")
            terminate_running_bakery_launcher(runner=self.runner)
            _atomic_copy(source, locations.launcher_path)
            _atomic_copy(icon_source, icon_destination)
            desktop = windows_desktop_directory()
            desktop_shortcut = desktop / "Bakery AI.lnk"
            create_windows_shortcut(
                desktop_shortcut,
                target=locations.launcher_path,
                icon=icon_destination,
                working_directory=locations.launcher_path.parent,
                runner=self.runner,
            )
            legacy_desktop_launcher = desktop / "Bakery AI.exe"
            if (
                legacy_desktop_launcher.resolve()
                != locations.launcher_path.resolve()
                and (
                    legacy_desktop_launcher.is_file()
                    or legacy_desktop_launcher.is_symlink()
                )
            ):
                legacy_desktop_launcher.unlink()
            return
        source_bundle = launcher_root / "Bakery AI.app"
        for source in _safe_files(source_bundle):
            relative = source.relative_to(source_bundle)
            _atomic_copy(source, locations.launcher_path / relative)
            try:
                os.chmod(locations.launcher_path / relative, source.stat().st_mode)
            except OSError:
                pass

    def verify_health(self, locations: InstallLocations) -> None:
        if self._runtime_state is None:
            self._runtime_state = self._runtime(locations).start()
        for endpoint in ("/health", "/s5-health"):
            with self.health_opener(
                f"{self._runtime_state.url}{endpoint}", timeout=5
            ) as response:
                if getattr(response, "status", 200) != 200:
                    raise InstallerOperationError(
                        f"Health check failed for {endpoint}"
                    )

    def backup_database(
        self,
        locations: InstallLocations,
        destination: Path,
    ) -> Path:
        return backup_database(
            locations.application_root / "compose.yaml",
            locations.runtime_env_path,
            destination,
            database="bakery_ai",
            application_version=str(self.manifest.get("release_version", "unknown")),
            popen=self.popen,
        )

    def capture_release(self, locations: InstallLocations) -> object:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = locations.backup_root / f"release-{stamp}"
        entries: list[tuple[Path, Path]] = []
        for source in _safe_files(locations.application_root):
            relative = source.relative_to(locations.application_root)
            destination = root / relative
            _atomic_copy(source, destination)
            entries.append((relative, destination))
        return ReleaseBackup(root=root, files=tuple(entries))

    def restore_release(
        self,
        locations: InstallLocations,
        previous_release: object,
    ) -> None:
        if not isinstance(previous_release, ReleaseBackup):
            raise InstallerOperationError("Previous release backup is invalid")
        for relative, source in previous_release.files:
            _atomic_copy(source, locations.application_root / relative)

    def restore_database(
        self,
        locations: InstallLocations,
        backup: Path,
    ) -> None:
        restore_snapshot(
            locations.application_root / "compose.yaml",
            locations.runtime_env_path,
            backup,
            database="bakery_ai",
            runner=self.runner,
        )

    def reload_runtime(self, locations: InstallLocations) -> None:
        self._runtime_state = self._runtime(locations).start()


def create_platform_operations(
    request: LifecycleRequest,
    **kwargs: Any,
) -> PlatformInstallerOperations:
    return PlatformInstallerOperations(request, **kwargs)


__all__ = (
    "InstallerOperationError",
    "PlatformInstallerOperations",
    "RebootRequired",
    "ReleaseBackup",
    "create_platform_operations",
)
