from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import gzip
import hashlib
import json
from pathlib import Path
import os
import shutil
import subprocess
from typing import Any


SUPPORTED_PLATFORMS = {
    "linux/amd64": "amd64",
    "linux/arm64": "arm64",
}
MYSQL_IMAGE = "mysql:8.4"
DEPENDENCY_EVIDENCE_FIELDS = (
    "vendor",
    "version",
    "source",
    "signing",
    "publisher",
    "license_evidence",
)


def validate_platform(platform: str) -> str:
    normalized = platform.strip().lower()
    if normalized not in SUPPORTED_PLATFORMS:
        supported = ", ".join(sorted(SUPPORTED_PLATFORMS))
        raise ValueError(f"Unsupported platform {platform!r}; expected one of: {supported}")
    return normalized


def build_command(platform: str, tag: str, context: str = ".") -> list[str]:
    native_platform = validate_platform(platform)
    if not tag.strip():
        raise ValueError("Image tag is required")
    return [
        "docker",
        "buildx",
        "build",
        "--platform",
        native_platform,
        "--tag",
        tag,
        "--load",
        context,
    ]


def mysql_pull_command(
    platform: str,
    image: str = MYSQL_IMAGE,
) -> list[str]:
    native_platform = validate_platform(platform)
    return [
        "docker",
        "image",
        "pull",
        f"--platform={native_platform}",
        image,
    ]


def save_command(
    platform: str,
    destination: str | Path,
    images: Sequence[str],
) -> list[str]:
    native_platform = validate_platform(platform)
    if not images:
        raise ValueError("At least one image is required")
    return [
        "docker",
        "image",
        "save",
        f"--platform={native_platform}",
        f"--output={destination}",
        *images,
    ]


def load_command(platform: str, archive: str | Path) -> list[str]:
    native_platform = validate_platform(platform)
    return [
        "docker",
        "image",
        "load",
        f"--platform={native_platform}",
        f"--input={archive}",
    ]


def image_inspect_command(image: str) -> list[str]:
    return ["docker", "image", "inspect", image]


def tag_command(source: str, destination: str) -> list[str]:
    return ["docker", "image", "tag", source, destination]


def verify_image_platform(
    inspection: Mapping[str, object],
    expected_platform: str,
    image: str,
) -> None:
    native_platform = validate_platform(expected_platform)
    expected_os, expected_architecture = native_platform.split("/", 1)
    actual_os = str(inspection.get("Os", "")).lower()
    actual_architecture = str(inspection.get("Architecture", "")).lower()
    actual_platform = f"{actual_os}/{actual_architecture}"
    if actual_platform != native_platform:
        raise ValueError(
            f"Image {image!r} is {actual_platform}, expected {native_platform}"
        )


def inspect_image(
    image: str,
    expected_platform: str,
    *,
    run: Callable[..., Any] = subprocess.run,
) -> str:
    result = run(
        image_inspect_command(image),
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"Unexpected Docker inspection result for {image!r}")
    inspection = payload[0]
    if not isinstance(inspection, dict):
        raise ValueError(f"Unexpected Docker inspection result for {image!r}")
    verify_image_platform(inspection, expected_platform, image)
    image_id = str(inspection.get("Id", "")).strip()
    if not image_id:
        raise ValueError(f"Docker inspection did not return an image ID for {image!r}")
    return image_id


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gzip_archive(
    source: str | Path,
    destination: str | Path | None = None,
) -> Path:
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination_path = (
        Path(destination)
        if destination is not None
        else source_path.with_name(f"{source_path.name}.gz")
    )
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("The gzip source and destination must be different files")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("rb") as input_stream:
        with destination_path.open("wb") as raw_output:
            with gzip.GzipFile(
                filename=source_path.name,
                mode="wb",
                fileobj=raw_output,
                mtime=0,
            ) as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
    return destination_path


def _package_path(path: Path, payload_root: Path) -> str:
    try:
        return path.resolve().relative_to(payload_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Payload artifact is outside the payload root: {path}") from exc


def _artifact(path: str | Path, payload_root: Path) -> dict[str, str]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": _package_path(source, payload_root),
        "sha256": sha256_file(source),
    }


def create_manifest(
    destination: str | Path,
    *,
    release_version: str,
    platform: str,
    images: Mapping[str, str],
    archive: str | Path,
    compose: str | Path,
    model: str | Path,
    snapshot: str | Path,
    installers: Mapping[str, str | Path | Mapping[str, str | Path]],
    payload_root: str | Path,
) -> dict[str, object]:
    native_platform = validate_platform(platform)
    if not release_version.strip():
        raise ValueError("Release version is required")
    if not images:
        raise ValueError("Image metadata is required")
    if not installers:
        raise ValueError("At least one official installer is required")
    root = Path(payload_root)
    model_path = Path(model)
    if _package_path(model_path, root) != "models/yolo/best.pt":
        raise ValueError("The YOLO model must be packaged as models/yolo/best.pt")
    image_entries = [
        {"tag": tag, "id": image_id}
        for tag, image_id in sorted(images.items())
    ]
    if any(not item["id"].strip() for item in image_entries):
        raise ValueError("Every image must include its Docker image ID")
    snapshot_path = Path(snapshot)
    snapshot_metadata = snapshot_path.with_name(
        f"{snapshot_path.name}.sha256.json"
    )
    ready_marker = snapshot_path.with_name("999-deployment-ready.sql")
    expected_database_paths = {
        snapshot_path: "deployment/database/init/001-final-snapshot.sql",
        snapshot_metadata: (
            "deployment/database/init/001-final-snapshot.sql.sha256.json"
        ),
        ready_marker: "deployment/database/init/999-deployment-ready.sql",
    }
    for path, expected in expected_database_paths.items():
        if _package_path(path, root) != expected:
            raise ValueError(f"Database artifact must be packaged as {expected}")

    installer_entries: dict[str, dict[str, str]] = {}
    for name, raw_entry in sorted(installers.items()):
        if isinstance(raw_entry, Mapping):
            raw_path = raw_entry.get("path")
            if raw_path is None:
                raise ValueError(f"Installer {name} is missing its path")
            entry = _artifact(raw_path, root)
            for field in DEPENDENCY_EVIDENCE_FIELDS:
                value = raw_entry.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"Installer {name} is missing required {field} evidence"
                    )
                entry[field] = value.strip()
            installer_entries[name] = entry
        else:
            installer_entries[name] = _artifact(raw_entry, root)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "release_version": release_version.strip(),
        "platform": native_platform,
        "architecture": SUPPORTED_PLATFORMS[native_platform],
        "images": image_entries,
        "artifacts": {
            "image_archive": _artifact(archive, root),
            "compose": _artifact(compose, root),
            "model": _artifact(model_path, root),
            "snapshot": _artifact(snapshot_path, root),
            "snapshot_metadata": _artifact(snapshot_metadata, root),
            "ready_marker": _artifact(ready_marker, root),
            "installers": installer_entries,
        },
    }
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _stage_file(source: str | Path, destination: Path) -> Path:
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != destination.resolve():
        shutil.copy2(source_path, destination)
    return destination


def _stage_launcher(source: str | Path, destination: Path) -> Path:
    source_path = Path(source)
    if source_path.is_symlink() or not source_path.exists():
        raise FileNotFoundError(source_path)
    if source_path.is_file():
        return _stage_file(source_path, destination)
    destination.mkdir(parents=True, exist_ok=False)
    for current, directories, names in os.walk(source_path, followlinks=False):
        current_path = Path(current)
        for name in directories:
            if (current_path / name).is_symlink():
                raise ValueError("Launcher bundles must not contain symlinks")
        for name in names:
            file_path = current_path / name
            if file_path.is_symlink():
                raise ValueError("Launcher bundles must not contain symlinks")
            _stage_file(
                file_path,
                destination / file_path.relative_to(source_path),
            )
    return destination


def build_release(
    *,
    release_version: str,
    platform: str,
    output_directory: str | Path,
    compose: str | Path,
    model: str | Path,
    snapshot: str | Path,
    launcher: str | Path,
    installers: Mapping[str, str | Path | Mapping[str, str | Path]],
    context: str | Path = ".",
    run: Callable[..., Any] = subprocess.run,
) -> Path:
    native_platform = validate_platform(platform)
    architecture = SUPPORTED_PLATFORMS[native_platform]
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    expected_launcher = (
        "Bakery AI.exe" if architecture == "amd64" else "Bakery AI.app"
    )
    launcher_path = Path(launcher)
    if launcher_path.name != expected_launcher:
        raise ValueError(f"Launcher must be named {expected_launcher}")
    _stage_launcher(
        launcher_path,
        output / "launcher" / expected_launcher,
    )
    def installer_source(value: str | Path | Mapping[str, str | Path]) -> Path:
        if isinstance(value, Mapping):
            path = value.get("path")
            if path is None:
                raise ValueError("Installer metadata is missing its path")
            return Path(path)
        return Path(value)

    installer_names = [
        installer_source(value).name.casefold() for value in installers.values()
    ]
    if len(installer_names) != len(set(installer_names)):
        raise ValueError("Duplicate installer filename in release payload")
    staged_compose = _stage_file(compose, output / "compose.yaml")
    staged_model = _stage_file(model, output / "models" / "yolo" / "best.pt")
    staged_snapshot = _stage_file(
        snapshot,
        output
        / "deployment"
        / "database"
        / "init"
        / "001-final-snapshot.sql",
    )
    source_snapshot = Path(snapshot)
    staged_metadata = _stage_file(
        source_snapshot.with_name(f"{source_snapshot.name}.sha256.json"),
        staged_snapshot.with_name(f"{staged_snapshot.name}.sha256.json"),
    )
    try:
        metadata = json.loads(staged_metadata.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Snapshot metadata sidecar is invalid") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Snapshot metadata sidecar must contain a JSON object")
    metadata["snapshot"] = staged_snapshot.name
    staged_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _stage_file(
        source_snapshot.with_name("999-deployment-ready.sql"),
        staged_snapshot.with_name("999-deployment-ready.sql"),
    )
    staged_installers: dict[str, str | Path | Mapping[str, str | Path]] = {}
    for name, raw_entry in installers.items():
        source_path = installer_source(raw_entry)
        staged_path = _stage_file(
            source_path,
            output / "installers" / source_path.name,
        )
        if isinstance(raw_entry, Mapping):
            staged_installers[name] = {**raw_entry, "path": staged_path}
        else:
            staged_installers[name] = staged_path
    application_image = f"bakery-ai:{release_version}-{architecture}"
    compose_image = "bakery-ai:local"
    inspected_images = [application_image, MYSQL_IMAGE]
    for command in (
        build_command(native_platform, application_image, str(context)),
        mysql_pull_command(native_platform),
    ):
        run(command, check=True, shell=False)
    for image in inspected_images:
        inspect_image(image, native_platform, run=run)
    run(tag_command(application_image, compose_image), check=True, shell=False)
    archived_images = [application_image, compose_image, MYSQL_IMAGE]
    image_directory = output / "images"
    image_directory.mkdir(parents=True, exist_ok=True)
    tar_path = image_directory / f"bakery-ai-images-{architecture}.tar"
    run(
        save_command(native_platform, tar_path, archived_images),
        check=True,
        shell=False,
    )
    archive_path = gzip_archive(tar_path)
    tar_path.unlink()
    from deployment.release.verify_payload import _archive_images, verify_payload

    image_ids = _archive_images(archive_path, native_platform)
    if set(image_ids) != set(archived_images):
        raise ValueError("Image archive does not contain every requested image tag")
    manifest_path = output / "manifest.json"
    create_manifest(
        manifest_path,
        release_version=release_version,
        platform=native_platform,
        images=image_ids,
        archive=archive_path,
        compose=staged_compose,
        model=staged_model,
        snapshot=staged_snapshot,
        installers=staged_installers,
        payload_root=output,
    )
    verify_payload(output, run=run)
    return manifest_path


def load_dependency_spec(path: str | Path) -> dict[str, dict[str, str | Path]]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Dependency evidence JSON is invalid") from exc
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Dependency evidence JSON must contain dependencies")
    dependencies: dict[str, dict[str, str | Path]] = {}
    for name, entry in payload.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError("Dependency evidence entry is invalid")
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"Dependency {name} is missing its path")
        dependency_path = Path(raw_path)
        if not dependency_path.is_absolute():
            dependency_path = source.parent / dependency_path
        dependencies[name] = {**entry, "path": dependency_path.resolve()}
    return dependencies


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build one native Bakery AI offline Docker image archive."
    )
    parser.add_argument("--release-version", required=True)
    parser.add_argument("--platform", choices=sorted(SUPPORTED_PLATFORMS), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--compose", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--dependency-spec", type=Path, required=True)
    parser.add_argument("--context", type=Path, default=Path("."))
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    build_release(
        release_version=options.release_version,
        platform=options.platform,
        output_directory=options.output_directory,
        compose=options.compose,
        model=options.model,
        snapshot=options.snapshot,
        launcher=options.launcher,
        installers=load_dependency_spec(options.dependency_spec),
        context=options.context,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
