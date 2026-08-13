from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import tempfile
from urllib.parse import urlparse
import zipfile


TARGETS = {
    "windows-x64": {
        "architecture": "amd64",
        "platform": "linux/amd64",
        "installer": "Install Bakery AI.exe",
        "launcher": "Bakery AI.exe",
        "archive": "BakeryAI-Offline-Windows-x64.zip",
        "dependencies": ("docker_desktop", "wsl", "microsoft_edge"),
    },
    "macos-apple-silicon": {
        "architecture": "arm64",
        "platform": "linux/arm64",
        "installer": "BakeryAI Installer.app",
        "launcher": "Bakery AI.app",
        "archive": "BakeryAI-Offline-macOS-AppleSilicon.zip",
        "dependencies": ("docker_desktop", "microsoft_edge"),
    },
}

FORBIDDEN_PARTS = {
    ".git",
    ".github",
    ".pytest_cache",
    "__pycache__",
    "tests",
}
FORBIDDEN_NAMES = {".env", ".env.local", ".env.production"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".env",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sql",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
}
API_KEY_PATTERN = re.compile(rb"\b(?:sk|ds)-[A-Za-z0-9_-]{16,}\b")
DEVELOPER_PATH_PATTERNS = (
    re.compile(rb"[A-Za-z]:\\Users\\[^\\\r\n]+", re.IGNORECASE),
    re.compile(rb"/Users/[^/\r\n]+"),
)
ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)
CHECKSUM_PATTERN = re.compile(r"^([0-9a-fA-F]{64})  (.+)$")
DEPENDENCY_EVIDENCE_FIELDS = (
    "vendor",
    "version",
    "source",
    "signing",
    "publisher",
    "license_evidence",
)
OFFICIAL_DEPENDENCY_CONTRACTS = {
    "docker_desktop": {
        "vendors": frozenset({"docker inc.", "docker inc"}),
        "publishers": frozenset({"docker inc.", "docker inc"}),
        "source_hosts": frozenset({"docker.com"}),
    },
    "wsl": {
        "vendors": frozenset({"microsoft corporation"}),
        "publishers": frozenset({"microsoft corporation"}),
        "source_hosts": frozenset({"github.com", "microsoft.com"}),
    },
    "microsoft_edge": {
        "vendors": frozenset({"microsoft corporation"}),
        "publishers": frozenset({"microsoft corporation"}),
        "source_hosts": frozenset({"microsoft.com"}),
    },
}
WINDOWS_ICON_SOURCE = Path(__file__).resolve().parents[1] / "assets" / "bakery-ai.ico"


class PackageBuildError(ValueError):
    pass


def required_package_paths(target: str) -> set[str]:
    settings = _target_settings(target)
    architecture = settings["architecture"]
    installer = settings["installer"]
    launcher = settings["launcher"]
    if target == "macos-apple-silicon":
        installer = f"{installer}/"
        launcher = f"payload/launcher/{launcher}/"
    else:
        launcher = f"payload/launcher/{launcher}"
    required = {
        installer,
        launcher,
        f"payload/bakery-ai-{architecture}-images.tar.gz",
        "payload/compose.yaml",
        "payload/deployment/database/init/001-final-snapshot.sql",
        "payload/deployment/database/init/001-final-snapshot.sql.sha256.json",
        "payload/deployment/database/init/999-deployment-ready.sql",
        "payload/release.json",
        "SHA256SUMS.txt",
    }
    if target == "windows-x64":
        required.add("payload/launcher/bakery-ai.ico")
    return required


def _target_settings(target: str) -> Mapping[str, object]:
    try:
        return TARGETS[target]
    except KeyError as exc:
        supported = ", ".join(sorted(TARGETS))
        raise PackageBuildError(
            f"Unsupported package target {target!r}; expected one of: {supported}"
        ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise PackageBuildError(f"Unsafe {label} path: {value!r}")
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    parts = value.split("/")
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise PackageBuildError(f"Unsafe {label} path: {value!r}")
    return posix_path


def _resolved_child(root: Path, relative: PurePosixPath, *, label: str) -> Path:
    base = root.resolve()
    child = base.joinpath(*relative.parts).resolve()
    try:
        child.relative_to(base)
    except ValueError as exc:
        raise PackageBuildError(f"Unsafe {label} path: {relative}") from exc
    return child


def _source_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise PackageBuildError(f"Symlinks are forbidden in release input: {relative}")
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda value: value.relative_to(root).as_posix())


def _contains_sensitive_bytes(path: Path, secrets: tuple[bytes, ...]) -> bool:
    overlap = max((len(secret) for secret in secrets), default=0)
    overlap = max(overlap, 256)
    tail = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            content = tail + chunk
            if API_KEY_PATTERN.search(content) or any(
                secret in content for secret in secrets
            ):
                return True
            tail = content[-overlap:]
    return False


def _validate_release_input(
    root: Path,
    *,
    secret_values: Iterable[str],
) -> None:
    encoded_secrets = tuple(value.encode("utf-8") for value in secret_values if value.strip())
    for path in _source_files(root):
        relative = path.relative_to(root)
        lowered_parts = {part.lower() for part in relative.parts}
        if lowered_parts & FORBIDDEN_PARTS:
            raise PackageBuildError(f"A forbidden path is present: {relative}")
        if path.name.lower() in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise PackageBuildError(f"A forbidden release file is present: {relative}")
        if _contains_sensitive_bytes(path, encoded_secrets):
            raise PackageBuildError(f"Potential sensitive API key in release input: {relative}")
        if (
            path.suffix.lower() in TEXT_SUFFIXES
            and path.stat().st_size <= 32 * 1024 * 1024
            and any(
                pattern.search(path.read_bytes())
                for pattern in DEVELOPER_PATH_PATTERNS
            )
        ):
            raise PackageBuildError(f"Potential sensitive developer path in release input: {relative}")


def _load_manifest(root: Path, expected_platform: str) -> dict[str, object]:
    path = root / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageBuildError("The native release manifest is missing or invalid") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise PackageBuildError("The native release manifest schema is invalid")
    if value.get("platform") != expected_platform:
        raise PackageBuildError(
            f"The native release targets {value.get('platform')!r}, not {expected_platform}"
        )
    return value


def _dependency_entries(
    manifest: Mapping[str, object],
    release_root: Path,
    settings: Mapping[str, object],
) -> dict[str, tuple[Path, dict[str, str]]]:
    artifacts = manifest.get("artifacts")
    installers = artifacts.get("installers") if isinstance(artifacts, Mapping) else None
    if not isinstance(installers, Mapping):
        raise PackageBuildError("The official dependency manifest is missing")
    required = settings.get("dependencies")
    if not isinstance(required, tuple):
        raise PackageBuildError("The platform dependency contract is invalid")
    missing = [name for name in required if name not in installers]
    if missing:
        raise PackageBuildError(
            "The platform package is missing required offline dependencies: "
            + ", ".join(missing)
        )

    validated: dict[str, tuple[Path, dict[str, str]]] = {}
    basenames: set[str] = set()
    for name, raw_entry in sorted(installers.items()):
        if not isinstance(name, str) or not isinstance(raw_entry, Mapping):
            raise PackageBuildError(f"Invalid official dependency entry: {name!r}")
        raw_path = raw_entry.get("path")
        if not isinstance(raw_path, str):
            raise PackageBuildError(f"Invalid official dependency path: {name}")
        relative = _safe_relative_path(raw_path, label="official dependency")
        source = _resolved_child(release_root, relative, label="official dependency")
        if not source.is_file() or source.is_symlink():
            raise PackageBuildError(f"Required official dependency is missing: {name}")
        expected_hash = raw_entry.get("sha256")
        if not isinstance(expected_hash, str) or not re.fullmatch(
            r"[0-9a-fA-F]{64}", expected_hash
        ):
            raise PackageBuildError(f"Invalid sha256 evidence for dependency {name}")
        if _sha256(source) != expected_hash.lower():
            raise PackageBuildError(f"Checksum mismatch for dependency {name}")

        evidence: dict[str, str] = {}
        for field in DEPENDENCY_EVIDENCE_FIELDS:
            value = raw_entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise PackageBuildError(
                    f"Dependency {name} is missing required {field} evidence"
                )
            evidence[field] = value.strip()
        _validate_official_dependency_evidence(name, evidence)
        basename = source.name.casefold()
        if basename in basenames:
            raise PackageBuildError("Duplicate official dependency filename")
        basenames.add(basename)
        validated[name] = (source, evidence)
    return validated


def _host_matches(host: str, allowed: frozenset[str]) -> bool:
    return any(host == value or host.endswith(f".{value}") for value in allowed)


def _validate_official_dependency_evidence(
    name: str,
    evidence: Mapping[str, str],
) -> None:
    contract = OFFICIAL_DEPENDENCY_CONTRACTS.get(name)
    if contract is None:
        raise PackageBuildError(f"Unsupported official dependency: {name}")
    vendor = evidence["vendor"].casefold()
    publisher = evidence["publisher"].casefold()
    if vendor not in contract["vendors"]:
        raise PackageBuildError(f"Dependency {name} has an untrusted vendor")
    if publisher not in contract["publishers"]:
        raise PackageBuildError(f"Dependency {name} has an untrusted publisher")
    parsed = urlparse(evidence["source"])
    host = (parsed.hostname or "").casefold()
    if parsed.scheme.casefold() != "https" or not _host_matches(
        host, contract["source_hosts"]
    ):
        raise PackageBuildError(
            f"Dependency {name} does not use an approved official HTTPS source"
        )


def _copy_file(source: Path, destination: Path) -> Path:
    if not source.is_file() or source.is_symlink():
        raise PackageBuildError(f"Required release file is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _artifact(path: str, source: Path) -> dict[str, str]:
    return {"path": path, "sha256": _sha256(source)}


def _stage_payload(
    release_root: Path,
    stage: Path,
    settings: Mapping[str, object],
) -> None:
    architecture = str(settings["architecture"])
    original = _load_manifest(release_root, str(settings["platform"]))
    payload = stage / "payload"
    image_target = _copy_file(
        release_root / "images" / f"bakery-ai-images-{architecture}.tar.gz",
        payload / f"bakery-ai-{architecture}-images.tar.gz",
    )
    compose_target = _copy_file(
        release_root / "compose.yaml",
        payload / "compose.yaml",
    )
    model_target = _copy_file(
        release_root / "models" / "yolo" / "best.pt",
        payload / "models" / "yolo" / "best.pt",
    )
    snapshot_name = "001-final-snapshot.sql"
    snapshot_source = (
        release_root / "deployment" / "database" / "init" / snapshot_name
    )
    init_target = payload / "deployment" / "database" / "init"
    snapshot_target = _copy_file(snapshot_source, init_target / snapshot_name)
    metadata_target = _copy_file(
        snapshot_source.with_name(f"{snapshot_name}.sha256.json"),
        init_target / f"{snapshot_name}.sha256.json",
    )
    ready_marker_source = snapshot_source.with_name("999-deployment-ready.sql")
    ready_marker_target = _copy_file(
        ready_marker_source,
        init_target / ready_marker_source.name,
    )
    launcher_name = str(settings["launcher"])
    launcher_source = release_root / "launcher" / launcher_name
    _stage_installer(
        launcher_source,
        payload / "launcher",
        launcher_name,
    )
    if architecture == "amd64":
        _copy_file(
            WINDOWS_ICON_SOURCE,
            payload / "launcher" / "bakery-ai.ico",
        )

    installer_entries: dict[str, dict[str, str]] = {}
    for name, (source, evidence) in _dependency_entries(
        original, release_root, settings
    ).items():
        target = _copy_file(source, payload / "installers" / source.name)
        installer_entries[name] = {
            **_artifact(f"installers/{source.name}", target),
            **evidence,
        }

    release_manifest = dict(original)
    release_manifest["artifacts"] = {
        "image_archive": _artifact(image_target.name, image_target),
        "compose": _artifact("compose.yaml", compose_target),
        "model": _artifact("models/yolo/best.pt", model_target),
        "snapshot": _artifact(
            f"deployment/database/init/{snapshot_name}", snapshot_target
        ),
        "snapshot_metadata": _artifact(
            f"deployment/database/init/{metadata_target.name}", metadata_target
        ),
        "ready_marker": _artifact(
            f"deployment/database/init/{ready_marker_target.name}",
            ready_marker_target,
        ),
        "installers": installer_entries,
    }
    (payload / "release.json").write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _stage_installer(installer: Path, stage: Path, expected_name: str) -> None:
    if installer.is_symlink() or not installer.exists():
        raise PackageBuildError(f"Installer is missing: {installer}")
    destination = stage / expected_name
    if installer.is_file():
        _copy_file(installer, destination)
        return
    destination.mkdir(parents=True, exist_ok=False)
    for source in _source_files(installer):
        _copy_file(source, destination / source.relative_to(installer))


def _write_checksums(stage: Path) -> Path:
    entries = []
    for path in _source_files(stage):
        if path.name == "SHA256SUMS.txt":
            continue
        relative = path.relative_to(stage).as_posix()
        entries.append(f"{_sha256(path)}  {relative}\n")
    destination = stage / "SHA256SUMS.txt"
    destination.write_text("".join(entries), encoding="ascii", newline="\n")
    return destination


def _parse_checksum_manifest(content: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(content.splitlines(), start=1):
        match = CHECKSUM_PATTERN.fullmatch(line)
        if match is None:
            raise PackageBuildError(
                f"Invalid checksum manifest entry on line {line_number}"
            )
        digest, raw_path = match.groups()
        relative = _safe_relative_path(raw_path, label="checksum")
        normalized = relative.as_posix()
        if normalized in entries:
            raise PackageBuildError(f"Duplicate checksum entry: {normalized}")
        entries[normalized] = digest.lower()
    if not entries:
        raise PackageBuildError("The checksum manifest is empty")
    return entries


def verify_checksum_manifest(
    root: str | Path,
    manifest_name: str = "SHA256SUMS.txt",
) -> tuple[str, ...]:
    package_root = Path(root).expanduser().resolve()
    manifest_relative = _safe_relative_path(manifest_name, label="checksum manifest")
    manifest_path = _resolved_child(
        package_root, manifest_relative, label="checksum manifest"
    )
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise PackageBuildError("The checksum manifest is missing")
    try:
        entries = _parse_checksum_manifest(manifest_path.read_text(encoding="ascii"))
    except UnicodeDecodeError as exc:
        raise PackageBuildError("The checksum manifest must be ASCII") from exc

    actual = {
        path.relative_to(package_root).as_posix(): path
        for path in _source_files(package_root)
        if path.resolve() != manifest_path
    }
    listed = set(entries)
    missing = sorted(listed - set(actual))
    extra = sorted(set(actual) - listed)
    if missing:
        raise PackageBuildError("Checksum manifest references missing files: " + ", ".join(missing))
    if extra:
        raise PackageBuildError("Checksum manifest has extra unlisted files: " + ", ".join(extra))
    for relative, expected in entries.items():
        path = _resolved_child(package_root, PurePosixPath(relative), label="checksum")
        if _sha256(path) != expected:
            raise PackageBuildError(f"Checksum mismatch for {relative}")
    return tuple(sorted(entries))


def _zip_entries(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    entries: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        raw_name = info.filename[:-1] if info.is_dir() else info.filename
        relative = _safe_relative_path(raw_name, label="ZIP entry")
        normalized = relative.as_posix() + ("/" if info.is_dir() else "")
        if normalized in entries:
            raise PackageBuildError(f"Duplicate ZIP entry: {normalized}")
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise PackageBuildError(f"Symlink ZIP entries are forbidden: {normalized}")
        entries[normalized] = info
    return entries


def _verify_zip_checksums(
    archive: zipfile.ZipFile,
    entries: Mapping[str, zipfile.ZipInfo],
) -> tuple[str, ...]:
    checksum_info = entries.get("SHA256SUMS.txt")
    if checksum_info is None or checksum_info.is_dir():
        raise PackageBuildError("The checksum manifest is missing from the ZIP")
    try:
        expected = _parse_checksum_manifest(
            archive.read(checksum_info).decode("ascii")
        )
    except UnicodeDecodeError as exc:
        raise PackageBuildError("The checksum manifest must be ASCII") from exc
    actual_files = {
        name
        for name, info in entries.items()
        if not info.is_dir() and name != "SHA256SUMS.txt"
    }
    missing = sorted(set(expected) - actual_files)
    extra = sorted(actual_files - set(expected))
    if missing:
        raise PackageBuildError("ZIP checksum manifest references missing files: " + ", ".join(missing))
    if extra:
        raise PackageBuildError("ZIP contains extra unlisted files: " + ", ".join(extra))
    for name, expected_hash in expected.items():
        digest = hashlib.sha256()
        with archive.open(entries[name]) as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise PackageBuildError(f"ZIP checksum mismatch for {name}")
    return tuple(sorted(expected))


def verify_package_archive(
    archive_path: str | Path,
    *,
    target: str,
) -> tuple[str, ...]:
    required = required_package_paths(target)
    path = Path(archive_path).expanduser().resolve()
    try:
        with zipfile.ZipFile(path) as archive:
            entries = _zip_entries(archive)
            missing = sorted(required - set(entries))
            if missing:
                raise PackageBuildError(
                    "The final package is missing required entries: " + ", ".join(missing)
                )
            return _verify_zip_checksums(archive, entries)
    except (OSError, zipfile.BadZipFile) as exc:
        raise PackageBuildError(f"The final package ZIP is invalid: {path}") from exc


def extract_package_archive(
    archive_path: str | Path,
    destination: str | Path,
    *,
    permission_setter: Callable[[Path, int], None] = os.chmod,
) -> tuple[Path, ...]:
    output_root = Path(destination).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    try:
        with zipfile.ZipFile(Path(archive_path).expanduser().resolve()) as archive:
            entries = _zip_entries(archive)
            _verify_zip_checksums(archive, entries)
            directories = [
                (name, info) for name, info in entries.items() if info.is_dir()
            ]
            files = [
                (name, info) for name, info in entries.items() if not info.is_dir()
            ]
            for name, _info in sorted(directories):
                relative = PurePosixPath(name.rstrip("/"))
                target = _resolved_child(output_root, relative, label="ZIP entry")
                target.mkdir(parents=True, exist_ok=True)
                permission_setter(target, 0o755)
                extracted.append(target)
            for name, info in sorted(files):
                target = _resolved_child(
                    output_root, PurePosixPath(name), label="ZIP entry"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                mode = stat.S_IMODE(info.external_attr >> 16) or 0o644
                permission_setter(target, mode)
                extracted.append(target)
    except (OSError, zipfile.BadZipFile) as exc:
        raise PackageBuildError("The package ZIP could not be extracted safely") from exc
    return tuple(extracted)


def _file_mode(path: Path, relative: PurePosixPath) -> int:
    source_mode = stat.S_IMODE(path.stat().st_mode)
    macos_executable = "Contents" in relative.parts and "MacOS" in relative.parts
    return 0o755 if macos_executable or source_mode & 0o111 else 0o644


def _write_zip(stage: Path, destination: Path, *, target: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            files = _source_files(stage)
            directory_names = sorted(
                {
                    f"{parent.as_posix()}/"
                    for path in files
                    for parent in path.relative_to(stage).parents
                    if parent != Path(".")
                }
            )
            for directory_name in directory_names:
                info = zipfile.ZipInfo(directory_name, ZIP_TIMESTAMP)
                info.create_system = 3
                info.external_attr = ((stat.S_IFDIR | 0o755) << 16) | 0x10
                archive.writestr(info, b"")
            for path in files:
                relative = path.relative_to(stage).as_posix()
                info = zipfile.ZipInfo(relative, ZIP_TIMESTAMP)
                info.create_system = 3
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (
                    stat.S_IFREG | _file_mode(path, PurePosixPath(relative))
                ) << 16
                with path.open("rb") as source, archive.open(info, "w") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        verify_package_archive(temporary, target=target)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def assemble_package(
    *,
    target: str,
    release_root: str | Path,
    installer: str | Path,
    destination: str | Path,
    secret_values: Iterable[str] = (),
) -> Path:
    settings = _target_settings(target)
    source = Path(release_root).expanduser().resolve()
    installer_path = Path(installer).expanduser().resolve()
    _validate_release_input(source, secret_values=secret_values)
    with tempfile.TemporaryDirectory(prefix="bakery-ai-package-") as directory:
        stage = Path(directory)
        _stage_payload(source, stage, settings)
        _stage_installer(installer_path, stage, str(settings["installer"]))
        _validate_release_input(stage, secret_values=secret_values)
        _write_checksums(stage)
        return _write_zip(
            stage,
            Path(destination).expanduser().resolve(),
            target=target,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble a Bakery AI offline ZIP package.")
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--installer", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    assemble_package(
        target=options.target,
        release_root=options.release_root,
        installer=options.installer,
        destination=options.destination,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
