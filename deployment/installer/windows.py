from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
import ctypes
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any
import uuid

from deployment.release.verify_payload import verify_payload


GIBIBYTE = 1024**3
MINIMUM_MEMORY_BYTES = 8 * GIBIBYTE
RECOMMENDED_MEMORY_BYTES = 16 * GIBIBYTE
RECOMMENDED_DISK_BYTES = 30 * GIBIBYTE
MINIMUM_WSL_VERSION = (2, 1, 5)
MICROSOFT_PUBLISHERS = frozenset({"Microsoft Corporation"})
CONTINUATION_FIELDS = {
    "schema_version",
    "stage",
    "target",
    "install_root",
    "payload_root",
    "data_root",
    "backup_root",
    "launcher_path",
    "runtime_env_path",
    "database_volume",
    "manifest_sha256",
}


class WindowsInstallerError(RuntimeError):
    pass


class CheckStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    message: str

    @property
    def passed(self) -> bool:
        return self.status is not CheckStatus.FAIL


@dataclass(frozen=True)
class WindowsPrerequisiteReport:
    checks: tuple[CheckResult, ...]

    @property
    def blockers(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if check.status is CheckStatus.FAIL)

    @property
    def install_ready(self) -> bool:
        return not self.blockers

    def by_name(self, name: str) -> CheckResult:
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(name)


@dataclass(frozen=True)
class WindowsSystemSnapshot:
    architecture: str
    windows_release: int
    windows_build: int
    virtualization_enabled: bool
    wsl_version: tuple[int, ...] | None
    total_memory_bytes: int
    free_disk_bytes: int
    docker_cli: Path | None
    docker_engine_running: bool
    edge_path: Path | None
    payload_hashes_verified: bool


@dataclass(frozen=True)
class WindowsInstallLayout:
    app_home: Path
    data_home: Path
    manifest_path: Path
    root_launcher: Path
    launcher_icon: Path
    desktop_shortcut: Path
    continuation_path: Path


@dataclass(frozen=True)
class ContinuationState:
    stage: str
    target: str
    install_root: str
    payload_root: str
    data_root: str
    backup_root: str
    launcher_path: str
    runtime_env_path: str
    database_volume: str
    manifest_sha256: str
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("Unsupported continuation schema version")
        if self.stage != "wsl_reboot":
            raise ValueError("Unsupported continuation stage")
        if self.target != "windows-x64":
            raise ValueError("Unsupported continuation target")
        if len(self.manifest_sha256) != 64:
            raise ValueError("Continuation manifest SHA-256 must contain 64 characters")
        int(self.manifest_sha256, 16)
        required_paths = (
            self.install_root,
            self.payload_root,
            self.data_root,
            self.backup_root,
            self.launcher_path,
            self.runtime_env_path,
        )
        if any(not value.strip() for value in required_paths):
            raise ValueError("Continuation paths are required")
        if not self.database_volume.strip():
            raise ValueError("Continuation database volume is required")


def _check(name: str, condition: bool, success: str, failure: str) -> CheckResult:
    return CheckResult(
        name=name,
        status=CheckStatus.PASS if condition else CheckStatus.FAIL,
        message=success if condition else failure,
    )


def _remediable_check(
    name: str,
    condition: bool,
    success: str,
    remediation: str,
) -> CheckResult:
    return CheckResult(
        name=name,
        status=CheckStatus.PASS if condition else CheckStatus.WARNING,
        message=success if condition else remediation,
    )


def _windows_version_supported(release: int, build: int) -> bool:
    if release == 10:
        return build >= 19045
    return release >= 11 and build >= 22631


def _normalized_version(version: tuple[int, ...] | None) -> tuple[int, int, int]:
    values = tuple(version or ())[:3]
    return (values + (0, 0, 0))[:3]


def evaluate_prerequisites(
    snapshot: WindowsSystemSnapshot,
) -> WindowsPrerequisiteReport:
    architecture = snapshot.architecture.strip().lower()
    checks: list[CheckResult] = [
        _check(
            "architecture",
            architecture in {"amd64", "x86_64"},
            "Windows x64 architecture is available.",
            "This package requires Windows x64.",
        ),
        _check(
            "windows_version",
            _windows_version_supported(
                snapshot.windows_release,
                snapshot.windows_build,
            ),
            "The Windows release is supported.",
            "Windows 10 build 19045 or Windows 11 build 22631 or newer is required.",
        ),
        _check(
            "virtualization",
            snapshot.virtualization_enabled,
            "Hardware virtualization is enabled.",
            "Enable hardware virtualization in BIOS or UEFI before continuing.",
        ),
        _remediable_check(
            "wsl2",
            _normalized_version(snapshot.wsl_version) >= MINIMUM_WSL_VERSION,
            "A supported WSL 2 runtime is available.",
            "Install the verified offline WSL MSI and reboot if requested.",
        ),
    ]

    if snapshot.total_memory_bytes < MINIMUM_MEMORY_BYTES:
        memory = CheckResult(
            "memory",
            CheckStatus.FAIL,
            "At least 8 GB of system memory is required.",
        )
    elif snapshot.total_memory_bytes < RECOMMENDED_MEMORY_BYTES:
        memory = CheckResult(
            "memory",
            CheckStatus.WARNING,
            "16 GB of system memory is recommended for smoother operation.",
        )
    else:
        memory = CheckResult(
            "memory",
            CheckStatus.PASS,
            "The recommended 16 GB of system memory is available.",
        )
    checks.append(memory)

    if snapshot.free_disk_bytes <= 0:
        disk = CheckResult(
            "disk",
            CheckStatus.FAIL,
            "Available installation disk space could not be verified.",
        )
    elif snapshot.free_disk_bytes < RECOMMENDED_DISK_BYTES:
        disk = CheckResult(
            "disk",
            CheckStatus.WARNING,
            "30 GB of free disk space is recommended for images, data, and backups.",
        )
    else:
        disk = CheckResult(
            "disk",
            CheckStatus.PASS,
            "The recommended 30 GB of free disk space is available.",
        )
    checks.append(disk)

    checks.extend(
        (
            _remediable_check(
                "docker_cli",
                snapshot.docker_cli is not None and Path(snapshot.docker_cli).is_absolute(),
                "The Docker CLI was found at an absolute path.",
                "Install Docker Desktop or locate its Docker CLI before continuing.",
            ),
            _remediable_check(
                "docker_engine",
                snapshot.docker_engine_running,
                "The Docker Engine is ready.",
                "Start Docker Desktop, accept its terms, and wait for the engine.",
            ),
            _remediable_check(
                "edge",
                snapshot.edge_path is not None and Path(snapshot.edge_path).is_absolute(),
                "Microsoft Edge is available for App Mode.",
                "Install the verified offline Microsoft Edge package before continuing.",
            ),
            _check(
                "payload_hashes",
                snapshot.payload_hashes_verified,
                "Every required payload SHA-256 hash was verified.",
                "The offline payload is incomplete or failed its SHA-256 checks.",
            ),
        )
    )
    return WindowsPrerequisiteReport(tuple(checks))


def docker_install_command(
    installer: str | Path,
    *,
    accept_license: bool = False,
    license_authorized: bool = False,
) -> list[str]:
    if accept_license and not license_authorized:
        raise ValueError(
            "Docker Desktop license acceptance requires explicit authorization"
        )
    command = [str(installer), "install", "--user", "--backend=wsl-2"]
    if accept_license:
        command.append("--accept-license")
    return command


def docker_desktop_start_command(executable: str | Path) -> list[str]:
    path = Path(executable).expanduser().resolve()
    if not path.is_file():
        raise WindowsInstallerError(f"Docker Desktop executable was not found: {path}")
    return [str(path)]


def find_windows_docker_cli(
    environment: Mapping[str, str] | None = None,
    *,
    which: Callable[[str], str | None] = shutil.which,
) -> Path | None:
    values = os.environ if environment is None else environment
    candidates: list[Path] = []
    explicit = values.get("BAKERY_DOCKER_CLI")
    if explicit:
        candidates.append(Path(explicit))
    local_app_data = values.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(
            Path(local_app_data)
            / "Programs"
            / "DockerDesktop"
            / "resources"
            / "bin"
            / "docker.exe"
        )
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
    path_match = which("docker.exe") or which("docker")
    if path_match:
        candidates.append(Path(path_match))
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_absolute() and resolved.is_file():
            return resolved
    return None


def verify_windows_payload(
    payload_root: str | Path,
    *,
    verifier: Callable[..., dict[str, Any]] = verify_payload,
) -> dict[str, Any]:
    root = Path(payload_root).expanduser().resolve()
    manifest_name = "release.json" if (root / "release.json").is_file() else "manifest.json"
    return verifier(
        root,
        target="windows-x64",
        manifest_name=manifest_name,
    )


class _GUID(ctypes.Structure):
    _fields_ = (
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    )


def _windows_known_folder_path(folder_name: str) -> Path:
    if os.name != "nt":
        raise WindowsInstallerError("Windows Known Folder API is unavailable")
    folder_ids = {
        "Desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    }
    try:
        identifier = folder_ids[folder_name]
    except KeyError as exc:
        raise WindowsInstallerError(f"Unsupported Windows Known Folder: {folder_name}") from exc

    guid = _GUID.from_buffer_copy(uuid.UUID(identifier).bytes_le)
    output = ctypes.c_wchar_p()
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    ole32 = ctypes.WinDLL("ole32", use_last_error=True)
    function = shell32.SHGetKnownFolderPath
    function.argtypes = (
        ctypes.POINTER(_GUID),
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_wchar_p),
    )
    function.restype = ctypes.c_long
    result = function(ctypes.byref(guid), 0, None, ctypes.byref(output))
    if result != 0 or not output.value:
        raise WindowsInstallerError(
            f"Windows Known Folder lookup failed with HRESULT 0x{result & 0xFFFFFFFF:08X}"
        )
    try:
        return Path(output.value)
    finally:
        ole32.CoTaskMemFree(output)


def windows_desktop_directory(
    *,
    environment: Mapping[str, str] | None = None,
    desktop: str | Path | None = None,
    known_folder_getter: Callable[[str], Path] | None = None,
) -> Path:
    values = os.environ if environment is None else environment
    if desktop is not None:
        return Path(desktop).expanduser().resolve()
    resolver = known_folder_getter or _windows_known_folder_path
    try:
        return Path(resolver("Desktop")).expanduser().resolve()
    except (OSError, WindowsInstallerError):
        user_profile = values.get("USERPROFILE")
        if not user_profile:
            raise WindowsInstallerError(
                "The current user's Desktop path is unavailable"
            )
        return (Path(user_profile) / "Desktop").resolve()


def windows_install_layout(
    install_root: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
    desktop: str | Path | None = None,
    known_folder_getter: Callable[[str], Path] | None = None,
) -> WindowsInstallLayout:
    values = os.environ if environment is None else environment
    local_app_data = values.get("LOCALAPPDATA")
    if not local_app_data:
        raise WindowsInstallerError("LOCALAPPDATA is required for per-user configuration")
    app_home = Path(install_root).expanduser().resolve()
    data_home = (Path(local_app_data) / "BakeryAI").resolve()
    desktop_root = windows_desktop_directory(
        environment=values,
        desktop=desktop,
        known_folder_getter=known_folder_getter,
    )
    return WindowsInstallLayout(
        app_home=app_home,
        data_home=data_home,
        manifest_path=data_home / "install.json",
        root_launcher=app_home / "Bakery AI.exe",
        launcher_icon=app_home / "bakery-ai.ico",
        desktop_shortcut=desktop_root / "Bakery AI.lnk",
        continuation_path=data_home / "installer" / "continuation.json",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary = Path(name)
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


WINDOWS_SHORTCUT_SCRIPT = (
    "$ErrorActionPreference = 'Stop'; "
    "$shortcutPath = $env:BAKERY_AI_SHORTCUT; "
    "$targetPath = $env:BAKERY_AI_TARGET; "
    "$iconPath = $env:BAKERY_AI_ICON; "
    "$workingDirectory = $env:BAKERY_AI_WORKING_DIRECTORY; "
    "$shell = New-Object -ComObject WScript.Shell; "
    "$shortcut = $shell.CreateShortcut($shortcutPath); "
    "$shortcut.TargetPath = $targetPath; "
    "$shortcut.WorkingDirectory = $workingDirectory; "
    "$shortcut.IconLocation = \"$iconPath,0\"; "
    "$shortcut.Description = 'Bakery AI'; "
    "$shortcut.Save(); "
    "if (-not (Test-Path -LiteralPath $shortcutPath -PathType Leaf)) { "
    "throw 'Desktop shortcut was not created.' }"
)


def terminate_running_bakery_launcher(
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> bool:
    command = ["taskkill.exe", "/F", "/IM", "Bakery AI.exe"]
    try:
        result = runner(
            command,
            capture_output=True,
            text=True,
            shell=False,
            check=False,
        )
    except OSError as exc:
        raise WindowsInstallerError(
            f"Could not close the running Bakery AI launcher: {exc}"
        ) from exc
    if result.returncode == 0:
        return True
    if result.returncode == 128:
        return False
    detail = (result.stderr or result.stdout or "unknown error").strip()
    raise WindowsInstallerError(
        f"Could not close the running Bakery AI launcher: {detail}"
    )


def create_windows_shortcut(
    shortcut: str | Path,
    *,
    target: str | Path,
    icon: str | Path,
    working_directory: str | Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    destination = Path(shortcut).expanduser().resolve()
    target_path = Path(target).expanduser().resolve()
    icon_path = Path(icon).expanduser().resolve()
    working_path = Path(working_directory or target_path.parent).expanduser().resolve()
    if destination.suffix.casefold() != ".lnk":
        raise WindowsInstallerError("Desktop shortcut must use the .lnk extension")
    if not target_path.is_file() or target_path.is_symlink():
        raise WindowsInstallerError(f"Installed launcher was not found: {target_path}")
    if not icon_path.is_file() or icon_path.is_symlink():
        raise WindowsInstallerError(f"Launcher icon was not found: {icon_path}")
    if not working_path.is_dir() or working_path.is_symlink():
        raise WindowsInstallerError(
            f"Launcher working directory was not found: {working_path}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        destination.unlink()
    environment = os.environ.copy()
    environment.update(
        {
            "BAKERY_AI_SHORTCUT": str(destination),
            "BAKERY_AI_TARGET": str(target_path),
            "BAKERY_AI_ICON": str(icon_path),
            "BAKERY_AI_WORKING_DIRECTORY": str(working_path),
        }
    )
    encoded_script = base64.b64encode(
        WINDOWS_SHORTCUT_SCRIPT.encode("utf-16-le")
    ).decode("ascii")
    command = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
        encoded_script,
    ]
    result = runner(
        command,
        capture_output=True,
        text=True,
        shell=False,
        check=False,
        timeout=30,
        env=environment,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        raise WindowsInstallerError(f"Desktop shortcut creation failed: {detail}")
    if not destination.is_file() or destination.is_symlink():
        raise WindowsInstallerError("Desktop shortcut creation returned no file")
    return destination


def install_verified_launchers(
    source_launcher: str | Path,
    *,
    source_icon: str | Path,
    expected_sha256: str,
    expected_icon_sha256: str,
    layout: WindowsInstallLayout,
    docker_cli: str | Path,
    release_version: str,
    shortcut_creator: Callable[..., Path] = create_windows_shortcut,
) -> None:
    source = Path(source_launcher).expanduser().resolve()
    icon_source = Path(source_icon).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise WindowsInstallerError(f"Verified launcher artifact was not found: {source}")
    if _sha256_file(source) != expected_sha256.lower():
        raise WindowsInstallerError("Launcher SHA-256 verification failed")
    if not icon_source.is_file() or icon_source.is_symlink():
        raise WindowsInstallerError(
            f"Verified launcher icon was not found: {icon_source}"
        )
    if _sha256_file(icon_source) != expected_icon_sha256.lower():
        raise WindowsInstallerError("Launcher icon SHA-256 verification failed")
    docker_path = Path(docker_cli).expanduser().resolve()
    if not docker_path.is_absolute() or not docker_path.is_file():
        raise WindowsInstallerError("The Docker CLI must be an absolute existing file")
    if not release_version.strip():
        raise WindowsInstallerError("Release version is required")

    _atomic_copy(source, layout.root_launcher)
    _atomic_copy(icon_source, layout.launcher_icon)
    shortcut_creator(
        layout.desktop_shortcut,
        target=layout.root_launcher,
        icon=layout.launcher_icon,
        working_directory=layout.app_home,
    )
    manifest = {
        "app_home": str(layout.app_home),
        "data_home": str(layout.data_home),
        "docker_cli": str(docker_path),
        "release_version": release_version.strip(),
        "schema_version": 1,
        "target": "windows-x64",
    }
    _atomic_write(
        layout.manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


AUTHENTICODE_SCRIPT = (
    "$ErrorActionPreference = 'Stop'; "
    "$securityModule = Join-Path $PSHOME "
    "'Modules\\Microsoft.PowerShell.Security\\Microsoft.PowerShell.Security.psd1'; "
    "Import-Module -Name $securityModule -ErrorAction Stop; "
    "$signature = Get-AuthenticodeSignature -LiteralPath $args[0]; "
    "$publisher = if ($signature.SignerCertificate) { "
    "$signature.SignerCertificate.GetNameInfo("
    "[System.Security.Cryptography.X509Certificates.X509NameType]::SimpleName, $false) "
    "} else { '' }; "
    "[pscustomobject]@{ Status = [string]$signature.Status; "
    "Publisher = [string]$publisher; "
    "Thumbprint = [string]$signature.SignerCertificate.Thumbprint } | "
    "ConvertTo-Json -Compress"
)


def read_authenticode_publisher(
    package: str | Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[bool, str]:
    path = Path(package).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise WindowsInstallerError(f"Signed package was not found: {path}")
    escaped_path = str(path).replace("'", "''")
    script = AUTHENTICODE_SCRIPT.replace("$args[0]", f"'{escaped_path}'")
    encoded_script = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    command = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
        encoded_script,
    ]
    result = runner(
        command,
        capture_output=True,
        text=True,
        shell=False,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        raise WindowsInstallerError(f"Authenticode validation failed: {detail}")
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise WindowsInstallerError("Authenticode validation returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise WindowsInstallerError("Authenticode validation returned invalid data")
    publisher = str(payload.get("Publisher", "")).strip()
    return str(payload.get("Status", "")).casefold() == "valid", publisher


def validate_signed_msi_package(
    package: str | Path,
    *,
    expected_sha256: str,
    allowed_publishers: set[str] | frozenset[str] = MICROSOFT_PUBLISHERS,
    signature_reader: Callable[[Path], tuple[bool, str]] = read_authenticode_publisher,
) -> Path:
    path = Path(package).expanduser().resolve()
    if path.suffix.casefold() != ".msi" or not path.is_file() or path.is_symlink():
        raise WindowsInstallerError(f"Verified MSI package was not found: {path}")
    if _sha256_file(path) != expected_sha256.lower():
        raise WindowsInstallerError("MSI package SHA-256 verification failed")
    signature_valid, publisher = signature_reader(path)
    if not signature_valid:
        raise WindowsInstallerError("MSI package signature is invalid")
    normalized_publishers = {value.casefold() for value in allowed_publishers}
    if publisher.casefold() not in normalized_publishers:
        raise WindowsInstallerError(
            f"MSI package publisher is not allowed: {publisher}"
        )
    return path


def wsl_feature_commands() -> tuple[list[str], list[str]]:
    base = ["dism.exe", "/online", "/enable-feature", "/all", "/norestart"]
    return (
        [*base, "/featurename:Microsoft-Windows-Subsystem-Linux"],
        [*base, "/featurename:VirtualMachinePlatform"],
    )


def _msi_install_command(
    package: str | Path,
    *,
    authorization_granted: bool,
) -> list[str]:
    if not authorization_granted:
        raise ValueError("Explicit administrator authorization is required")
    path = Path(package).expanduser().resolve()
    if path.suffix.casefold() != ".msi" or not path.is_file():
        raise WindowsInstallerError(f"MSI package was not found: {path}")
    return [
        "msiexec.exe",
        "/i",
        str(path),
        "/passive",
        "/norestart",
    ]


def wsl_msi_install_command(
    package: str | Path,
    *,
    authorization_granted: bool = False,
) -> list[str]:
    return _msi_install_command(
        package,
        authorization_granted=authorization_granted,
    )


def edge_msi_install_command(
    package: str | Path,
    *,
    authorization_granted: bool = False,
) -> list[str]:
    return _msi_install_command(
        package,
        authorization_granted=authorization_granted,
    )


def write_continuation_state(path: str | Path, state: ContinuationState) -> Path:
    destination = Path(path).expanduser().resolve()
    payload = asdict(state)
    if set(payload) != CONTINUATION_FIELDS:
        raise WindowsInstallerError("Invalid installer continuation state")
    _atomic_write(
        destination,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return destination


def load_continuation_state(path: str | Path) -> ContinuationState:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WindowsInstallerError("Invalid installer continuation state") from exc
    if not isinstance(payload, dict) or set(payload) != CONTINUATION_FIELDS:
        raise WindowsInstallerError("Invalid installer continuation state")
    try:
        return ContinuationState(**payload)
    except (TypeError, ValueError) as exc:
        raise WindowsInstallerError("Invalid installer continuation state") from exc


def consume_continuation_state(path: str | Path) -> ContinuationState:
    source = Path(path).expanduser().resolve()
    state = load_continuation_state(source)
    payload_root = Path(state.payload_root).expanduser().resolve()
    candidates = tuple(
        candidate
        for candidate in (payload_root / "release.json", payload_root / "manifest.json")
        if candidate.is_file() and not candidate.is_symlink()
    )
    if len(candidates) != 1:
        raise WindowsInstallerError(
            "Exactly one continuation release manifest must be available"
        )
    if _sha256_file(candidates[0]) != state.manifest_sha256.lower():
        raise WindowsInstallerError("Continuation manifest SHA-256 verification failed")
    try:
        source.unlink()
    except OSError as exc:
        raise WindowsInstallerError(
            "Installer continuation state could not be removed after resume"
        ) from exc
    return state


def resume_once_command(
    installer_executable: str | Path,
    continuation_path: str | Path,
) -> list[str]:
    executable = Path(installer_executable).expanduser().resolve()
    state_path = Path(continuation_path).expanduser().resolve()
    resume_value = subprocess.list2cmdline(
        [str(executable), "--resume", str(state_path)]
    )
    return [
        "reg.exe",
        "add",
        r"HKCU\Software\Microsoft\Windows\CurrentVersion\RunOnce",
        "/v",
        "BakeryAIInstallerResume",
        "/t",
        "REG_SZ",
        "/d",
        resume_value,
        "/f",
    ]


def wait_for_docker_engine(
    docker_cli: str | Path,
    *,
    timeout: float = 180.0,
    interval: float = 2.0,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    path = Path(docker_cli).expanduser().resolve()
    if not path.is_absolute() or not path.is_file():
        raise WindowsInstallerError("The Docker CLI must be an absolute existing file")
    if timeout <= 0 or interval < 0:
        raise ValueError("Docker readiness timing values are invalid")
    command = [str(path), "info", "--format", "{{json .ServerVersion}}"]
    deadline = monotonic() + timeout
    last_detail = "Docker Engine did not return readiness information."
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise WindowsInstallerError(
                "Docker Engine did not become ready before the timeout: "
                f"{last_detail}"
            )
        try:
            result = runner(
                command,
                capture_output=True,
                text=True,
                shell=False,
                check=False,
                timeout=max(0.1, min(10.0, remaining)),
            )
        except subprocess.TimeoutExpired as exc:
            last_detail = str(exc)
            sleeper(interval)
            continue
        if result.returncode == 0:
            return
        last_detail = (result.stderr or result.stdout or last_detail).strip()
        sleeper(interval)


__all__: Sequence[str] = (
    "CheckResult",
    "CheckStatus",
    "ContinuationState",
    "WindowsInstallLayout",
    "WindowsInstallerError",
    "WindowsPrerequisiteReport",
    "WindowsSystemSnapshot",
    "consume_continuation_state",
    "create_windows_shortcut",
    "docker_desktop_start_command",
    "docker_install_command",
    "edge_msi_install_command",
    "evaluate_prerequisites",
    "find_windows_docker_cli",
    "install_verified_launchers",
    "load_continuation_state",
    "read_authenticode_publisher",
    "resume_once_command",
    "terminate_running_bakery_launcher",
    "validate_signed_msi_package",
    "verify_windows_payload",
    "wait_for_docker_engine",
    "windows_desktop_directory",
    "windows_install_layout",
    "write_continuation_state",
    "wsl_feature_commands",
    "wsl_msi_install_command",
)
