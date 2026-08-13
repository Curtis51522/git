# Bakery AI Deployment Acceptance Checklist

Record observed results only. Do not mark a platform item complete from a unit
test, another CPU architecture, or an estimated value.

## Release identity

- Release version:
- Git revision:
- Package filename:
- Package SHA-256:
- Package compressed size:
- Installed size:
- Test computer and CPU:
- Operating-system version:
- Docker Desktop version:
- Microsoft Edge version:

## Package gate

- [ ] `SHA256SUMS.txt` validates every packaged file.
- [ ] The payload manifest matches the target operating system and CPU.
- [ ] The Docker archive loads without a network connection.
- [ ] Every loaded image has the expected native architecture.
- [ ] Compose does not pull or build images at runtime.
- [ ] No environment file, API key, password, developer path, test, cache, or
      source-control metadata is present.
- [ ] The final MySQL snapshot and checksum sidecar validate together.
- [ ] The YOLO model is present at `payload/models/yolo/best.pt`.

## Clean installation

- [ ] Disable the network before starting the installer.
- [ ] Complete platform prerequisite checks.
- [ ] Complete any Windows WSL2 reboot continuation exactly once.
- [ ] Complete the official Docker Desktop installation and license flow.
- [ ] Install Microsoft Edge from the approved offline source when absent.
- [ ] Install Bakery AI outside the system drive when that option is selected.
- [ ] Confirm only the selected installation and per-user data paths are used.
- [ ] Confirm the launcher selects a free `127.0.0.1` port when preferred ports
      are occupied.
- [ ] Confirm MySQL, the main application, and S5 become healthy.
- [ ] Confirm Edge App Mode opens the selected dynamic URL.

Observed first-start duration:

Observed warm-start duration:

Observed published port:

Observed peak memory:

## Application workflow

- [ ] Sign in and confirm manager/staff access boundaries.
- [ ] Complete one POS checkout and verify finished-product inventory movement.
- [ ] Verify beverage material consumption at checkout.
- [ ] Verify camera permission and product-recognition capture.
- [ ] Verify S1 product recognition and stock inflow.
- [ ] Verify S2 forecast dashboard and seven-day forecast data.
- [ ] Verify S3 schedule, attendance, correction, and KPI views.
- [ ] Verify S4 POS, revenue, inventory, and historical views.
- [ ] Verify S5 revenue, promotion mix, forecast, inventory, and wastage analysis.
- [ ] Remove or invalidate the LLM key and confirm deterministic fallback behavior.
- [ ] Restart the computer and confirm database persistence.

## Upgrade and rollback

- [ ] Add a known operational record through the application.
- [ ] Upgrade from version N to version N+1.
- [ ] Confirm the known record and runtime configuration remain available.
- [ ] Repeat with an intentionally failing N+1 health check.
- [ ] Confirm version N, the known record, and the original runtime secrets are
      restored.
- [ ] Validate the pre-upgrade SQL backup and metadata sidecar.

## Removal and recovery

- [ ] Run standard uninstall.
- [ ] Confirm the MySQL volume, configuration, and backups remain.
- [ ] Reinstall and confirm the known operational record returns.
- [ ] Start complete removal and confirm exact files and the Docker volume are
      displayed before the second confirmation.
- [ ] Confirm cancellation leaves every file and volume unchanged.
- [ ] Complete the separate full-removal test on a disposable acceptance machine.

## Platform-specific evidence

### Windows x64

- [ ] Windows 10/11, virtualization, WSL2, memory, and disk checks are accurate.
- [ ] Offline WSL package signature and SHA-256 validation succeed when required.
- [ ] Reboot continuation contains no password or API key.
- [ ] The desktop shortcut targets the installation-root launcher and uses the
      packaged Bakery AI icon.

### Apple Silicon macOS

- [ ] `platform.machine()` reports `arm64`.
- [ ] Docker DMG verification, code-signature assessment, mount, install, and
      detach complete successfully.
- [ ] The image archive contains native ARM64 images with no x64 emulation.
- [ ] Configuration permissions are `0600` under Application Support.
- [ ] Gatekeeper, Edge App Mode, and camera permission work on the M-series Mac.

## Final result

- Automated test count:
- New warnings:
- Windows result:
- Apple Silicon result:
- Remaining limitations:
- Reviewer and date:
