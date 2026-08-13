# Bakery AI Offline Installation for Apple Silicon macOS

## Scope

This installation contract targets Apple Silicon Macs running the native arm64
build. It does not support Intel Macs or amd64 container emulation. The release
package is designed for an offline installation after all payload files and
their SHA-256 values have been prepared on a trusted build machine.

The current Windows build machine can test command construction, manifest
validation, and failure cleanup with mocks. It cannot produce or validate the final macOS application bundle, Docker Desktop installation, Gatekeeper flow,
camera permission, or native arm64 container execution. Those checks must be
completed on a clean Apple Silicon Mac before the package is released.

## System requirements

- Apple Silicon with `arm64` architecture. M1, M2, M3, and later M-series
  processors use the same native package contract.
- macOS 14 or later for the Docker Desktop release currently bundled with the
  package. A future package must raise this requirement if its official Docker
  Desktop release requires a newer macOS version.
- 16 GB RAM is recommended.
- 30 GB free disk space is recommended before installation.
- An administrator account for the Docker Desktop installation step.
- Microsoft Edge already installed, or a verified official offline Edge PKG in
  the release payload.

Docker Desktop licensing remains governed by Docker's terms. The installer
does not bypass the Docker Desktop license or first-run acceptance process.

## Required offline payload

The package verifier must accept the payload for target
`macos-apple-silicon`. Its manifest must declare `linux/arm64`, include native
arm64 Bakery AI and MySQL 8.4 images, and provide matching SHA-256 hashes for
the following artifacts:

- The container image archive.
- `compose.yaml`.
- The final database snapshot.
- `models/yolo/best.pt`.
- An official Docker Desktop `Docker.dmg`.
- An official Microsoft Edge PKG when Edge is not already installed.

The installer stops before mounting or copying anything when the architecture,
manifest, file hash, Docker DMG type, or Edge requirement is invalid. Existing
Edge applications must use Microsoft's `UBF8T346G9` TeamIdentifier. A packaged
Edge PKG is checked with `pkgutil --check-signature` against the same expected
publisher identity before installation. Docker.app must use Docker's expected
`9BNSXJN65R` TeamIdentifier.

## Installation flow

1. Verify that the Mac is arm64 and runs a supported macOS release.
2. Report a warning when the Mac has less than 16 GB RAM or less than 30 GB
   free disk space.
3. Verify the ARM64 release manifest and every payload hash.
4. Verify the Docker DMG with `hdiutil verify`.
5. Mount the DMG read-only with `hdiutil attach -plist` and parse the returned
   property list to obtain the actual mount point. The installer never assumes
   that the volume is named `/Volumes/Docker`; an existing volume may cause a
   name such as `/Volumes/Docker 1`.
6. Validate the mounted `Docker.app` with `codesign` and `spctl`, and require
   Docker's expected `TeamIdentifier` before executing it.
7. After explicit administrator authorization, run the official mounted
   installer through `/usr/bin/sudo`:

   ```text
   /usr/bin/sudo <actual-mount-point>/Docker.app/Contents/MacOS/install --user=<macOS-user>
   ```

8. Detach the actual mount point in a guaranteed cleanup step, including when
   signature verification or installation fails. A detach failure does not
   replace the original installation error.
9. Install `Bakery AI.app` in `/Applications` when system-wide installation is
   selected, or in `~/Applications` for a per-user installation. File metadata
   is copied so the launcher's executable permission is preserved.
10. If Edge is absent, validate the official Edge PKG publisher and request
    explicit administrator authorization before running:

    ```text
    /usr/bin/sudo /usr/sbin/installer -pkg MicrosoftEdge.pkg -target /
    ```

11. Create the operational data location and start the verified ARM64 images.

The installer uses only the official mounted Docker installer. It does not copy
or modify files inside `Docker.app`.

## Operational data and permissions

Operational data is independent from the application bundle:

```text
~/Library/Application Support/BakeryAI/
  runtime.env
  backups/
```

`runtime.env` is written atomically and assigned mode 0600. Database backups
remain under the same application-support root. A standard uninstall retains
configuration, backups, and the Docker database volume. Complete removal is a
separate confirmed operation.

## Browser shell

The supported browser shell launches the native Microsoft Edge binary in App
Mode:

```text
/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge --app=http://127.0.0.1:<port>/
```

If Edge cannot be launched after the payload and installation checks, the
launcher may use `/usr/bin/open <url>` as a clearly reported fallback. The
fallback uses the default browser and is not the normal supported application
shell.

## Gatekeeper first open

This graduation-project package does not add a commercial code-signing and
notarization pipeline. On first open, Gatekeeper may ask the user to confirm the
locally installed application. Open `Bakery AI.app` from Finder. If macOS blocks
the first launch, use **System Settings > Privacy & Security > Open Anyway** and
confirm the exact Bakery AI application. Do not disable Gatekeeper globally and
do not remove quarantine attributes from unrelated files.

Docker Desktop and Microsoft Edge must retain their official vendor signatures
and expected TeamIdentifier values. These checks are executed as structured
argument arrays with `shell=False`. The payload verifier and macOS signature
checks do not replace Gatekeeper.

## Native release build on Apple Silicon

The final application and installer bundles must be built on an Apple Silicon
Mac. Windows can prepare and verify the source, dependency evidence, database
snapshot, and release contracts, but it must not supply substitute `.app`
bundles. On the M-series build Mac, install Python 3.13 and the pinned packages
from `requirements-deployment.txt`, start Docker Desktop, and run:

```text
python3 -m deployment.release.build_macos_native \
  --release-version 1.0.0 \
  --project-root . \
  --output-root ./build/macos \
  --compose ./compose.yaml \
  --model ./models/yolo/best.pt \
  --snapshot ./deployment/database/init/001-final-snapshot.sql \
  --dependency-spec ./release-inputs/dependencies-macos.json
```

The command first builds `Bakery AI.app` and `BakeryAI Installer.app` with the
native PyInstaller bootloader. It then builds and verifies the `linux/arm64`
Bakery AI and MySQL images, stages the approved official dependencies, and
creates `BakeryAI-1.0.0-macOS-arm64-Offline.zip`. The ZIP is still a candidate
until the clean-machine acceptance checklist below passes on Apple Silicon.

## Clean-machine acceptance on an M-series Mac

Perform this checklist with network access disabled during installation:

1. Confirm `uname -m` reports `arm64`.
2. Verify every packaged file against the release manifest.
3. Install Docker Desktop from the verified DMG and confirm the volume detaches.
4. Confirm Docker Desktop starts and accepts its applicable license terms.
5. Load the image archive and confirm both Bakery AI and MySQL report arm64,
   with no amd64 emulation.
6. Start Bakery AI and confirm it selects an available loopback port.
7. Confirm Microsoft Edge opens in App Mode; separately test the default-browser
   fallback.
8. Grant camera permission when prompted and verify POS camera capture.
9. Restart the Mac and confirm database, configuration, and backups persist.
10. Verify a successful upgrade retains an operational record.
11. Force a failed upgrade health check and verify rollback restores the previous
    release and the operational record.
12. Verify standard uninstall retains operational data.
13. Verify complete removal displays exact targets and requires its second
    confirmation before deleting operational data.

Record the tested Mac model, macOS version, Docker Desktop version, package
hash, installation duration, warm-start duration, and peak memory in the final
release acceptance report.
