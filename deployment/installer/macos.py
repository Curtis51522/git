from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import plistlib
import re
import shutil
import subprocess
import tempfile
from typing import Any

from deployment.release.verify_payload import verify_payload


MIN_SUPPORTED_MACOS_MAJOR = 14
RECOMMENDED_MEMORY_BYTES = 16 * 1024**3
RECOMMENDED_FREE_DISK_BYTES = 30 * 1024**3
DOCKER_VOLUME = "/Volumes/Docker"
DOCKER_APPLICATION = f"{DOCKER_VOLUME}/Docker.app"
DOCKER_INSTALLER = f"{DOCKER_APPLICATION}/Contents/MacOS/install"
DOCKER_TEAM_IDS = frozenset({"9BNSXJN65R"})
MICROSOFT_TEAM_IDS = frozenset({"UBF8T346G9"})
SYSTEM_EDGE_BINARY = Path(
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
)


class MacOSPrerequisiteError(RuntimeError):
    pass


class MacOSPayloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class MacOSSystemReport:
    supported: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class MacOSPayload:
    manifest: Mapping[str, Any]
    docker_dmg: Path
    edge_binary: Path | None
    edge_installer: Path | None


@dataclass(frozen=True)
class MacOSInstallPaths:
    application_bundle: Path
    support_root: Path
    backup_root: Path
    config_path: Path


def _macos_major(version: str) -> int:
    value = version.strip()
    if not value:
        raise MacOSPrerequisiteError("The macOS version could not be determined")
    try:
        return int(value.split(".", 1)[0])
    except ValueError as exc:
        raise MacOSPrerequisiteError(
            f"The macOS version is invalid: {version!r}"
        ) from exc


def check_macos_system(
    *,
    machine: str,
    macos_version: str,
    total_memory_bytes: int,
    free_disk_bytes: int,
) -> MacOSSystemReport:
    if machine.strip().lower() != "arm64":
        raise MacOSPrerequisiteError(
            "Bakery AI for macOS requires Apple Silicon with arm64 architecture"
        )
    if _macos_major(macos_version) < MIN_SUPPORTED_MACOS_MAJOR:
        raise MacOSPrerequisiteError(
            f"macOS {MIN_SUPPORTED_MACOS_MAJOR} or later is required by the bundled "
            "Docker Desktop release"
        )
    if total_memory_bytes < 0 or free_disk_bytes < 0:
        raise MacOSPrerequisiteError("System resource values cannot be negative")

    warnings: list[str] = []
    if total_memory_bytes < RECOMMENDED_MEMORY_BYTES:
        warnings.append(
            "16 GB RAM is recommended; "
            f"{total_memory_bytes / 1024**3:.1f} GB is available."
        )
    if free_disk_bytes < RECOMMENDED_FREE_DISK_BYTES:
        warnings.append(
            "30 GB free disk space is recommended; "
            f"{free_disk_bytes / 1024**3:.1f} GB is available."
        )
    return MacOSSystemReport(supported=True, warnings=tuple(warnings))


def _artifact_path(
    payload_root: Path,
    entry: object,
    *,
    name: str,
    suffix: str,
) -> Path:
    if not isinstance(entry, Mapping):
        raise MacOSPayloadError(f"The {name} manifest entry is missing")
    relative_path = str(entry.get("path", ""))
    path = (payload_root / relative_path).resolve()
    try:
        path.relative_to(payload_root.resolve())
    except ValueError as exc:
        raise MacOSPayloadError(f"The {name} path is outside the payload") from exc
    if path.suffix.lower() != suffix or not path.is_file():
        raise MacOSPayloadError(
            f"The {name} must be a verified {suffix} file in the payload"
        )
    return path


def _default_edge_binary_paths() -> tuple[Path, ...]:
    return (
        SYSTEM_EDGE_BINARY,
        Path.home()
        / "Applications"
        / "Microsoft Edge.app"
        / "Contents"
        / "MacOS"
        / "Microsoft Edge",
    )


def validate_macos_payload(
    payload_root: str | Path,
    *,
    edge_binary_paths: Sequence[Path] | None = None,
    edge_signature_validator: Callable[[Path, frozenset[str]], str] | None = None,
) -> MacOSPayload:
    root = Path(payload_root).expanduser().resolve()
    manifest_name = "release.json" if (root / "release.json").is_file() else "manifest.json"
    manifest = verify_payload(
        root,
        target="macos-apple-silicon",
        manifest_name=manifest_name,
    )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise MacOSPayloadError("The payload artifact manifest is missing")
    installers = artifacts.get("installers")
    if not isinstance(installers, Mapping):
        raise MacOSPayloadError("The official installer manifest is missing")

    docker_dmg = _artifact_path(
        root,
        installers.get("docker_desktop"),
        name="official Docker Desktop installer",
        suffix=".dmg",
    )
    candidates = (
        tuple(edge_binary_paths)
        if edge_binary_paths is not None
        else _default_edge_binary_paths()
    )
    edge_binary = next((Path(path) for path in candidates if Path(path).is_file()), None)
    if edge_binary is not None:
        validator = edge_signature_validator or verify_application_publisher
        validator(edge_binary, MICROSOFT_TEAM_IDS)
    edge_installer: Path | None = None
    if edge_binary is None and "microsoft_edge" in installers:
        edge_installer = _artifact_path(
            root,
            installers["microsoft_edge"],
            name="official Microsoft Edge installer",
            suffix=".pkg",
        )
    if edge_binary is None and edge_installer is None:
        raise MacOSPayloadError(
            "Microsoft Edge must already be installed or supplied as a verified "
            "offline installer"
        )
    return MacOSPayload(
        manifest=manifest,
        docker_dmg=docker_dmg,
        edge_binary=edge_binary,
        edge_installer=edge_installer,
    )


def _authorized_command(
    executable: str,
    arguments: Sequence[str],
    *,
    authorization_granted: bool,
    privilege_escalation: str,
) -> list[str]:
    if not authorization_granted:
        raise ValueError("Explicit administrator authorization is required")
    return [privilege_escalation, executable, *arguments]


def docker_install_command(
    username: str,
    *,
    mount_point: str | Path = DOCKER_VOLUME,
    authorization_granted: bool = False,
    privilege_escalation: str = "/usr/bin/sudo",
) -> list[str]:
    normalized = username.strip()
    if (
        not normalized
        or normalized in {".", ".."}
        or re.fullmatch(r"[A-Za-z0-9._-]+", normalized) is None
    ):
        raise ValueError("A valid macOS username is required")
    if not authorization_granted:
        raise ValueError("Explicit administrator authorization is required")
    mount_text = str(mount_point).replace("\\", "/")
    mount = PurePosixPath(mount_text)
    if not mount.is_absolute() or mount.parent != PurePosixPath("/Volumes"):
        raise ValueError("A valid hdiutil mount point is required")
    installer = str(mount / "Docker.app" / "Contents" / "MacOS" / "install")
    return _authorized_command(
        installer,
        [f"--user={normalized}"],
        authorization_granted=True,
        privilege_escalation=privilege_escalation,
    )


def edge_pkg_install_command(
    package: str | Path,
    *,
    authorization_granted: bool = False,
    privilege_escalation: str = "/usr/bin/sudo",
) -> list[str]:
    path = Path(package).expanduser().resolve()
    if path.suffix.casefold() != ".pkg" or not path.is_file() or path.is_symlink():
        raise MacOSPayloadError(f"Verified Microsoft Edge PKG was not found: {path}")
    return _authorized_command(
        "/usr/sbin/installer",
        ["-pkg", str(path), "-target", "/"],
        authorization_granted=authorization_granted,
        privilege_escalation=privilege_escalation,
    )


def _run_checked(
    command: list[str],
    run: Callable[..., Any],
) -> Any:
    return run(
        command,
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )


def _team_identifier(output: str) -> str | None:
    match = re.search(r"(?:TeamIdentifier=|\()([A-Z0-9]{10})(?:\)|\s|$)", output)
    return match.group(1) if match else None


def _macos_path_text(path: str | Path) -> str:
    value = str(path)
    if value.startswith("\\Volumes\\"):
        return value.replace("\\", "/")
    return value


def verify_application_publisher(
    application: str | Path,
    expected_team_ids: set[str] | frozenset[str],
    *,
    run: Callable[..., Any] = subprocess.run,
) -> str:
    path = _macos_path_text(application)
    _run_checked(
        ["/usr/bin/codesign", "--verify", "--deep", "--strict", path],
        run,
    )
    details = _run_checked(
        ["/usr/bin/codesign", "--display", "--verbose=4", path],
        run,
    )
    team_id = _team_identifier(f"{details.stdout or ''}\n{details.stderr or ''}")
    if team_id not in expected_team_ids:
        raise MacOSPayloadError("Application publisher is not allowed")
    return team_id


def verify_pkg_publisher(
    package: str | Path,
    expected_team_ids: set[str] | frozenset[str],
    *,
    run: Callable[..., Any] = subprocess.run,
) -> str:
    path = Path(package).expanduser().resolve()
    if path.suffix.casefold() != ".pkg" or not path.is_file() or path.is_symlink():
        raise MacOSPayloadError(f"Signed PKG was not found: {path}")
    result = _run_checked(
        ["/usr/sbin/pkgutil", "--check-signature", str(path)],
        run,
    )
    team_id = _team_identifier(f"{result.stdout or ''}\n{result.stderr or ''}")
    if team_id not in expected_team_ids:
        raise MacOSPayloadError("PKG publisher is not allowed")
    return team_id


def install_edge_package(
    package: str | Path,
    *,
    authorization_granted: bool,
    expected_team_ids: set[str] | frozenset[str] = MICROSOFT_TEAM_IDS,
    run: Callable[..., Any] = subprocess.run,
) -> None:
    path = Path(package).expanduser().resolve()
    verify_pkg_publisher(path, expected_team_ids, run=run)
    _run_checked(
        edge_pkg_install_command(
            path,
            authorization_granted=authorization_granted,
        ),
        run,
    )


def parse_hdiutil_mount_point(payload: str | bytes) -> Path:
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    try:
        document = plistlib.loads(raw)
    except (ValueError, TypeError, plistlib.InvalidFileException) as exc:
        raise MacOSPayloadError("hdiutil returned an invalid property list") from exc
    entities = document.get("system-entities") if isinstance(document, dict) else None
    if not isinstance(entities, list):
        raise MacOSPayloadError("hdiutil did not report a mounted volume")
    for entity in entities:
        if not isinstance(entity, dict) or not entity.get("mount-point"):
            continue
        mount_text = str(entity["mount-point"])
        posix_mount = PurePosixPath(mount_text)
        if posix_mount.is_absolute() and posix_mount.parent == PurePosixPath("/Volumes"):
            return Path(mount_text)
    raise MacOSPayloadError("hdiutil did not report a safe mounted volume")


def install_docker_desktop(
    docker_dmg: str | Path,
    username: str,
    *,
    authorization_granted: bool = False,
    expected_team_ids: set[str] | frozenset[str] = DOCKER_TEAM_IDS,
    run: Callable[..., Any] = subprocess.run,
) -> None:
    dmg = Path(docker_dmg).expanduser().resolve()
    if dmg.suffix.lower() != ".dmg" or not dmg.is_file():
        raise MacOSPayloadError("A verified official Docker Desktop DMG is required")

    _run_checked(["/usr/bin/hdiutil", "verify", str(dmg)], run)
    if not authorization_granted:
        raise ValueError("Explicit administrator authorization is required")
    mount_point: Path | None = None
    primary_error: BaseException | None = None
    try:
        attached = _run_checked(
            [
                "/usr/bin/hdiutil",
                "attach",
                str(dmg),
                "-nobrowse",
                "-readonly",
                "-plist",
            ],
            run,
        )
        mount_point = parse_hdiutil_mount_point(attached.stdout)
        mount_text = _macos_path_text(mount_point)
        docker_application = str(PurePosixPath(mount_text) / "Docker.app")
        verify_application_publisher(
            docker_application,
            expected_team_ids,
            run=run,
        )
        _run_checked(
            [
                "/usr/sbin/spctl",
                "--assess",
                "--type",
                "execute",
                "--verbose=2",
                docker_application,
            ],
            run,
        )
        _run_checked(
            docker_install_command(
                username,
                mount_point=mount_text,
                authorization_granted=authorization_granted,
            ),
            run,
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if mount_point is not None:
            try:
                _run_checked(
                    ["/usr/bin/hdiutil", "detach", _macos_path_text(mount_point)],
                    run,
                )
            except BaseException:
                if primary_error is None:
                    raise


def macos_install_paths(
    home: str | Path,
    *,
    system_wide: bool,
) -> MacOSInstallPaths:
    user_home = Path(home).expanduser()
    application_root = Path("/Applications") if system_wide else user_home / "Applications"
    support_root = user_home / "Library" / "Application Support" / "BakeryAI"
    return MacOSInstallPaths(
        application_bundle=application_root / "Bakery AI.app",
        support_root=support_root,
        backup_root=support_root / "backups",
        config_path=support_root / "runtime.env",
    )


def write_sensitive_config(path: str | Path, content: str) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    (destination.parent / "backups").mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
        temporary = None
        os.chmod(destination, 0o600)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def install_application_bundle(
    source_bundle: str | Path,
    destination_bundle: str | Path,
) -> Path:
    source = Path(source_bundle).expanduser()
    destination = Path(destination_bundle).expanduser()
    if not source.is_dir() or source.suffix.lower() != ".app":
        raise FileNotFoundError(f"Application bundle not found: {source}")
    if destination.exists():
        raise FileExistsError(f"Application bundle already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    return destination


def edge_app_mode_command(url: str, edge_binary: str | Path) -> list[str]:
    normalized_url = url.strip()
    if not normalized_url:
        raise ValueError("The application URL is required")
    return [str(Path(edge_binary)), f"--app={normalized_url}"]


def browser_launch_command(
    url: str,
    *,
    edge_binary: str | Path | None,
) -> list[str]:
    if edge_binary is not None:
        return edge_app_mode_command(url, edge_binary)
    normalized_url = url.strip()
    if not normalized_url:
        raise ValueError("The application URL is required")
    return ["/usr/bin/open", normalized_url]
