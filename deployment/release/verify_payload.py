from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import subprocess
import tarfile
from typing import Any

from deployment.release.build_images import (
    inspect_image,
    load_command,
    sha256_file,
    validate_platform,
)


TARGET_PLATFORMS = {
    "windows-x64": "linux/amd64",
    "macos-apple-silicon": "linux/arm64",
}
REQUIRED_ARTIFACTS = {
    "image_archive",
    "compose",
    "model",
    "snapshot",
    "snapshot_metadata",
    "ready_marker",
    "installers",
}
MAX_ARCHIVE_METADATA_BYTES = 16 * 1024 * 1024


class PayloadVerificationError(ValueError):
    pass


def _safe_artifact_path(payload_root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PayloadVerificationError(f"Unsafe payload path: {relative_path}")
    resolved_root = payload_root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise PayloadVerificationError(f"Unsafe payload path: {relative_path}") from exc
    return resolved


def _verify_artifact(
    payload_root: Path,
    name: str,
    entry: object,
) -> None:
    if not isinstance(entry, dict):
        raise PayloadVerificationError(f"Invalid manifest entry for {name}")
    relative_path = str(entry.get("path", ""))
    expected_hash = str(entry.get("sha256", "")).lower()
    if len(expected_hash) != 64:
        raise PayloadVerificationError(f"Invalid SHA-256 for {name}")
    path = _safe_artifact_path(payload_root, relative_path)
    if not path.is_file():
        raise PayloadVerificationError(f"Missing payload artifact {name}: {relative_path}")
    if sha256_file(path) != expected_hash:
        raise PayloadVerificationError(f"SHA-256 mismatch for {name}: {relative_path}")


def _verify_images(images: object, platform: str) -> dict[str, str]:
    if not isinstance(images, list) or not images:
        raise PayloadVerificationError("Manifest image metadata is missing")
    image_map: dict[str, str] = {}
    for entry in images:
        if not isinstance(entry, dict):
            raise PayloadVerificationError("Invalid image metadata")
        tag = str(entry.get("tag", "")).strip()
        image_id = str(entry.get("id", "")).strip()
        if not tag or not image_id.startswith("sha256:"):
            raise PayloadVerificationError(f"Invalid image metadata for {tag or 'unknown'}")
        if tag in image_map:
            raise PayloadVerificationError(f"Duplicate image tag in manifest: {tag}")
        image_map[tag] = image_id
    architecture = platform.split("/", 1)[1]
    if "mysql:8.4" not in image_map:
        raise PayloadVerificationError("The native mysql:8.4 image is missing")
    if not any(
        tag.startswith("bakery-ai:") and architecture in tag for tag in image_map
    ):
        raise PayloadVerificationError(
            f"The Bakery AI image does not declare the required {platform} architecture"
        )
    return image_map


def _safe_tar_member(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise PayloadVerificationError(f"Unsafe image archive member: {name}")
    return path.as_posix()


def _read_archive_member(
    archive: tarfile.TarFile,
    name: str,
) -> bytes:
    safe_name = _safe_tar_member(name)
    matches = [member for member in archive.getmembers() if member.name == safe_name]
    if len(matches) != 1:
        raise PayloadVerificationError(
            f"Image archive must contain exactly one {safe_name} member"
        )
    member = matches[0]
    if not member.isfile() or member.size > MAX_ARCHIVE_METADATA_BYTES:
        raise PayloadVerificationError(f"Invalid image archive member: {safe_name}")
    stream = archive.extractfile(member)
    if stream is None:
        raise PayloadVerificationError(f"Cannot read image archive member: {safe_name}")
    return stream.read(MAX_ARCHIVE_METADATA_BYTES + 1)


def _archive_images(
    archive_path: Path,
    expected_platform: str,
) -> dict[str, str]:
    expected_os, expected_architecture = validate_platform(expected_platform).split("/", 1)
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            manifest_bytes = _read_archive_member(archive, "manifest.json")
            manifest = json.loads(manifest_bytes)
            if not isinstance(manifest, list) or not manifest:
                raise PayloadVerificationError("Image archive manifest is empty or invalid")
            images: dict[str, str] = {}
            config_digests: dict[str, str] = {}
            for entry in manifest:
                if not isinstance(entry, dict):
                    raise PayloadVerificationError("Invalid image archive manifest entry")
                config_name = str(entry.get("Config", ""))
                config_bytes = _read_archive_member(archive, config_name)
                config = json.loads(config_bytes)
                if not isinstance(config, dict):
                    raise PayloadVerificationError("Invalid image configuration metadata")
                actual_os = str(config.get("os", "")).lower()
                actual_architecture = str(config.get("architecture", "")).lower()
                actual_platform = f"{actual_os}/{actual_architecture}"
                if (actual_os, actual_architecture) != (
                    expected_os,
                    expected_architecture,
                ):
                    raise PayloadVerificationError(
                        "Image archive contains "
                        f"{actual_platform}, expected {expected_platform}"
                    )
                image_id = f"sha256:{hashlib.sha256(config_bytes).hexdigest()}"
                tags = entry.get("RepoTags")
                if not isinstance(tags, list) or not tags:
                    raise PayloadVerificationError("Image archive entry has no repository tags")
                for raw_tag in tags:
                    tag = str(raw_tag).strip()
                    if not tag or tag in images:
                        raise PayloadVerificationError(
                            f"Invalid or duplicate image archive tag: {tag or 'unknown'}"
                        )
                    images[tag] = image_id
                    config_digests[tag] = image_id
            if not any(member.name == "index.json" for member in archive.getmembers()):
                return images
            index_bytes = _read_archive_member(archive, "index.json")
            index = json.loads(index_bytes)
            descriptors = index.get("manifests") if isinstance(index, dict) else None
            if not isinstance(descriptors, list) or not descriptors:
                raise PayloadVerificationError("OCI image index is empty or invalid")
            indexed_images: dict[str, str] = {}
            for descriptor in descriptors:
                if not isinstance(descriptor, dict):
                    raise PayloadVerificationError("Invalid OCI image descriptor")
                annotations = descriptor.get("annotations")
                image_name = (
                    str(annotations.get("io.containerd.image.name", "")).strip()
                    if isinstance(annotations, dict)
                    else ""
                )
                matching_tags = [
                    tag
                    for tag in config_digests
                    if image_name == tag or image_name.endswith(f"/{tag}")
                ]
                if not matching_tags:
                    continue
                platform = descriptor.get("platform")
                if not isinstance(platform, dict):
                    raise PayloadVerificationError(
                        "Tagged OCI image descriptor has no platform"
                    )
                descriptor_os = str(platform.get("os", "")).lower()
                descriptor_architecture = str(
                    platform.get("architecture", "")
                ).lower()
                if (descriptor_os, descriptor_architecture) != (
                    expected_os,
                    expected_architecture,
                ):
                    raise PayloadVerificationError(
                        "OCI image index contains "
                        f"{descriptor_os}/{descriptor_architecture}, "
                        f"expected {expected_platform}"
                    )
                digest = str(descriptor.get("digest", "")).lower()
                if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                    raise PayloadVerificationError("Invalid OCI image manifest digest")
                image_manifest_bytes = _read_archive_member(
                    archive,
                    f"blobs/sha256/{digest.removeprefix('sha256:')}",
                )
                if hashlib.sha256(image_manifest_bytes).hexdigest() != digest.removeprefix(
                    "sha256:"
                ):
                    raise PayloadVerificationError(
                        "OCI image manifest digest does not match its content"
                    )
                image_manifest = json.loads(image_manifest_bytes)
                config_descriptor = (
                    image_manifest.get("config")
                    if isinstance(image_manifest, dict)
                    else None
                )
                config_digest = (
                    str(config_descriptor.get("digest", "")).lower()
                    if isinstance(config_descriptor, dict)
                    else ""
                )
                for tag in matching_tags:
                    if config_digest != config_digests[tag]:
                        raise PayloadVerificationError(
                            f"OCI image configuration does not match tag {tag}"
                        )
                    if tag in indexed_images and indexed_images[tag] != digest:
                        raise PayloadVerificationError(
                            f"Duplicate OCI image descriptor for tag {tag}"
                        )
                    indexed_images[tag] = digest
            if set(indexed_images) != set(config_digests):
                raise PayloadVerificationError(
                    "OCI image index does not describe every archived image tag"
                )
            return indexed_images
    except PayloadVerificationError:
        raise
    except (OSError, tarfile.TarError, json.JSONDecodeError) as exc:
        raise PayloadVerificationError("Image archive is not a valid Docker archive") from exc


def _verify_image_archive(
    archive_path: Path,
    platform: str,
    expected_images: Mapping[str, str],
    *,
    run: Callable[..., Any],
) -> None:
    if "bakery-ai:local" not in expected_images:
        raise PayloadVerificationError("The runtime bakery-ai:local image tag is missing")
    archived_images = _archive_images(archive_path, platform)
    if archived_images != dict(expected_images):
        raise PayloadVerificationError(
            "Image archive tags or image IDs do not match the release manifest"
        )
    try:
        run(
            load_command(platform, archive_path),
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
        for tag, expected_id in expected_images.items():
            actual_id = inspect_image(tag, platform, run=run)
            if actual_id != expected_id:
                raise PayloadVerificationError(
                    f"Loaded image ID mismatch for {tag}: expected {expected_id}"
                )
    except PayloadVerificationError:
        raise
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
        raise PayloadVerificationError(
            f"Docker could not load or inspect the {platform} image archive"
        ) from exc


def verify_payload(
    payload_root: str | Path,
    *,
    target: str | None = None,
    manifest_name: str = "manifest.json",
    run: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    root = Path(payload_root)
    manifest_path = root / manifest_name
    if not manifest_path.is_file():
        raise PayloadVerificationError(f"Missing payload manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PayloadVerificationError("Payload manifest is not valid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise PayloadVerificationError("Unsupported payload manifest schema")
    try:
        platform = validate_platform(str(manifest.get("platform", "")))
    except ValueError as exc:
        raise PayloadVerificationError(str(exc)) from exc
    if target is not None:
        expected_platform = TARGET_PLATFORMS.get(target)
        if expected_platform is None:
            supported = ", ".join(sorted(TARGET_PLATFORMS))
            raise PayloadVerificationError(
                f"Unsupported release target {target!r}; expected one of: {supported}"
            )
        if platform != expected_platform:
            raise PayloadVerificationError(
                f"Target {target!r} requires {expected_platform}; package contains {platform}"
            )
    expected_architecture = platform.split("/", 1)[1]
    if manifest.get("architecture") != expected_architecture:
        raise PayloadVerificationError("Manifest architecture does not match its platform")
    images = _verify_images(manifest.get("images"), platform)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not REQUIRED_ARTIFACTS.issubset(artifacts):
        raise PayloadVerificationError("Manifest does not contain every required artifact")
    model_entry = artifacts["model"]
    if not isinstance(model_entry, dict) or model_entry.get("path") != "models/yolo/best.pt":
        raise PayloadVerificationError("The package must contain models/yolo/best.pt")
    expected_database_paths = {
        "snapshot": "deployment/database/init/001-final-snapshot.sql",
        "snapshot_metadata": (
            "deployment/database/init/001-final-snapshot.sql.sha256.json"
        ),
        "ready_marker": "deployment/database/init/999-deployment-ready.sql",
    }
    for name, expected_path in expected_database_paths.items():
        entry = artifacts[name]
        if not isinstance(entry, dict) or entry.get("path") != expected_path:
            raise PayloadVerificationError(
                f"The {name} artifact must use {expected_path}"
            )
    for name in (
        "image_archive",
        "compose",
        "model",
        "snapshot",
        "snapshot_metadata",
        "ready_marker",
    ):
        _verify_artifact(root, name, artifacts[name])
    installers = artifacts["installers"]
    if not isinstance(installers, dict) or not installers:
        raise PayloadVerificationError("At least one official installer is required")
    for name, entry in installers.items():
        _verify_artifact(root, str(name), entry)
    archive_entry = artifacts["image_archive"]
    archive_path = _safe_artifact_path(root, str(archive_entry["path"]))
    if run is not None:
        _verify_image_archive(archive_path, platform, images, run=run)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a Bakery AI offline release payload and its SHA-256 manifest."
    )
    parser.add_argument("payload_root", type=Path)
    parser.add_argument("--target", choices=sorted(TARGET_PLATFORMS))
    parser.add_argument("--manifest-name", default="manifest.json")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    verify_payload(
        options.payload_root,
        target=options.target,
        manifest_name=options.manifest_name,
        run=subprocess.run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
