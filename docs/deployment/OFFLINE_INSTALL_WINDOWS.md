# Bakery AI Offline Installation on Windows x64

This contract describes the fully offline Windows installation flow. The release package targets Windows x64 and runs the application in Linux containers through the WSL 2 backend of Docker Desktop. The installer must not contact the network or execute an unverified payload.

## Supported host

- Windows 10 22H2 build 19045 or newer, or Windows 11 build 22631 or newer.
- An AMD64 or x86-64 processor with hardware virtualization enabled in BIOS or UEFI.
- WSL 2 version 2.1.5 or newer.
- At least 8 GB of memory; 16 GB is recommended.
- 30 GB of free disk space is recommended for application images, operational data, and backups.
- Microsoft Edge for Microsoft Edge App Mode.
- Docker Desktop using its per-user WSL 2 installation mode.

The installer presents each prerequisite as a structured pass, warning, or failure. Memory between 8 GB and 16 GB and free disk space below 30 GB are recommendations rather than silent failures. Unsupported architecture, Windows version, virtualization, or payload integrity blocks installation. Missing WSL 2, Docker Desktop, a running Docker Engine, or Edge is reported as a remediable warning because a clean machine may install those verified prerequisites from the offline payload before application startup.

## Verified offline payload

Before changing the host, the installer validates the release manifest and the SHA-256 hash of every required artifact. This includes the native `linux/amd64` application and MySQL images, Compose file, database snapshot, YOLO model, Bakery AI launcher, official Docker Desktop installer, official signed WSL MSI, and Microsoft Edge Enterprise MSI when supplied in the release.

Each Microsoft MSI must have both a matching SHA-256 value and a valid Authenticode signature from an approved Microsoft publisher. The implementation invokes `Get-AuthenticodeSignature` through a fixed PowerShell program with the package path passed as a separate process argument, requests JSON output, and compares the certificate simple name with an exact publisher allowlist. It uses `shell=False` and never interpolates a package path into PowerShell source. A changed hash, invalid signature, unknown publisher, missing file, wrong architecture, or unsafe relative path stops installation.

## Installation locations

The user may select an application location such as `D:\BakeryAI`. Application payload files and the primary `Bakery AI.exe` launcher are installed there. Per-user configuration remains under `%LOCALAPPDATA%\BakeryAI`, regardless of the selected drive:

```text
%LOCALAPPDATA%\BakeryAI\install.json
%LOCALAPPDATA%\BakeryAI\runtime.env
%LOCALAPPDATA%\BakeryAI\runtime\launcher.log
%LOCALAPPDATA%\BakeryAI\backups\
```

`install.json` is the canonical installation manifest. It records the application home, data home, absolute Docker CLI path, release version, target, and schema version. It contains no API key, database password, JWT secret, or other credential.

The verified launcher is copied to both the selected installation root and the current user's Desktop. The Desktop is resolved through the Windows Known Folder API so OneDrive and enterprise folder redirection are respected; `%USERPROFILE%\Desktop` is used only as an explicit fallback when the API is unavailable. Both copies read the same canonical manifest from `LOCALAPPDATA` and therefore resolve the same application home. Neither launcher assumes that its own directory contains Compose files or application data. Repair and upgrade replace both copies from one verified launcher artifact.

## WSL 2 setup and reboot continuation

If WSL 2 is unavailable, the installer enables `Microsoft-Windows-Subsystem-Linux` and `VirtualMachinePlatform` without downloading packages. It then validates the current official WSL MSI and runs the authorized offline command:

```text
msiexec.exe /i "wsl.<version>.x64.msi" /passive /norestart
```

The legacy `Add-AppxPackage` WSL bundle flow is not used.

When Windows requires a reboot, the installer writes one continuation file below `%LOCALAPPDATA%\BakeryAI\installer`. The file contains only the continuation schema version, stage, selected install root, payload root, and release-manifest SHA-256. A per-user `RunOnce` entry resumes the installer once after sign-in and points to that file. Passwords, tokens, API keys, runtime environment values, and database credentials must never appear in the continuation file or registry command.

Before a resume is consumed, the installer revalidates the SHA-256 of the single `release.json` or `manifest.json` inside the recorded payload root. A missing, duplicated, symlinked, or changed manifest stops continuation and leaves the state file available for diagnosis. After a successful validation, the installer removes the continuation file and does not register a persistent startup task.

## Docker Desktop installation

The official per-user installation command is equivalent to:

```text
"Docker Desktop Installer.exe" install --user --backend=wsl-2
```

The installer does not append `--accept-license` by default. Docker Desktop license acceptance remains an explicit user choice. The flag may be added only when distribution policy explicitly authorizes it and the agreement is displayed before execution.

After installation, Bakery AI starts the verified Docker Desktop executable and detects the Docker CLI by absolute path. It checks the per-user Docker Desktop path, standard program locations, and finally the system path. The selected path is stored in the canonical manifest. The installer polls `docker info` with a bounded timeout and displays the last Docker diagnostic if the engine does not become ready.

## Microsoft Edge installation

When Edge is absent, the installer validates the packaged Microsoft Edge Enterprise MSI hash and Authenticode publisher before requesting administrator authorization. The structured install command is:

```text
msiexec.exe /i "MicrosoftEdgeEnterpriseX64.msi" /passive /norestart
```

The command is executed with `shell=False`. Declining authorization leaves the host unchanged and Edge remains a remediable prerequisite.

## Application startup

Once Docker is ready, the shared installer verifies and copies the payload, loads the offline native image archive, creates the runtime configuration, initializes MySQL from the verified snapshot, and checks application health. `Bakery AI.exe` then selects a free loopback port, starts the containers, waits for both application and S5 readiness, and opens the interface in Microsoft Edge App Mode.

No MySQL or S5 port is exposed to the local network. Operational data remains in the Docker volume when the launcher closes, Windows restarts, the application is repaired, or a standard uninstall is performed.

## Clean-machine acceptance

With networking disabled, verify all of the following on a clean Windows x64 host:

1. Payload SHA-256 and signature failures are rejected before installation.
2. WSL 2 feature enablement, signed MSI installation, reboot, manifest revalidation, and one-time continuation complete without stored secrets.
3. Docker Desktop installs per user, displays its license flow, starts, and reports a ready engine.
4. The application can be installed on a non-system drive while configuration remains in `LOCALAPPDATA`.
5. Both launcher copies resolve the canonical application home and open Microsoft Edge App Mode.
6. Offline image load, MySQL initialization, dynamic loopback port selection, POS camera permission, and S1-S5 workflows succeed.
7. Operational data survives restart, repair, upgrade, standard uninstall, and reinstall.
8. Upgrade rollback restores the prior release after an intentionally failed health check.
9. Complete removal occurs only after the separate destructive confirmation and removes only the displayed Bakery AI files and volume.

The command contracts are unit-tested with mocked process runners. This development machine does not execute the Docker Desktop installer, WSL feature commands, registry continuation command, or clean-machine acceptance workflow.

## Official references

- Docker Desktop for Windows: https://docs.docker.com/desktop/setup/install/windows-install/
- Docker Desktop permission requirements: https://docs.docker.com/desktop/setup/install/windows-permission-requirements/
- Microsoft WSL installation: https://learn.microsoft.com/windows/wsl/install
