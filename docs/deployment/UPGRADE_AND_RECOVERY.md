# Bakery AI Upgrade and Recovery

## Data ownership

Bakery AI keeps operational MySQL data in a named Docker volume. Runtime secrets,
launcher logs, browser profile data, and database backups remain in the per-user
application-data directory. Application binaries and offline payload files live in
the selected installation directory.

Standard uninstall removes only application files and launchers. It keeps the
database volume, runtime configuration, and backups so a later reinstall can recover
the store history. Complete removal is separate, lists every exact file and volume,
and requires two confirmations.

## Before an upgrade

1. Close the Bakery AI application window.
2. Start the installer from the new verified offline package.
3. Review the release version and package integrity result.
4. Allow the installer to create a versioned MySQL snapshot and checksum sidecar.

Do not delete the previous package or the new backup until the upgraded application
has completed its health checks and a business record has been opened successfully.

## Automatic rollback

The installer verifies the package, backs up MySQL, captures the installed release,
and then replaces the application payload. If the main or S5 health check fails, it
restores the previous application release, preserves the original runtime secrets,
and verifies the restored health state. If rollback itself fails, stop using the
application and retain the diagnostic log, package, and database backup.

## Manual recovery

Use only a snapshot whose SQL file and `.sha256.json` sidecar both pass validation.
The sidecar must identify the expected database, application version, filename, and
SHA-256 digest. Restore through the installer recovery action so the database client
is run inside the controlled Docker environment and credentials are not placed in
command-line arguments.

## Offline package integrity

Each platform ZIP contains `SHA256SUMS.txt`. The installer verifies every listed file
before loading Docker images or changing an existing installation. A package is
rejected when it has the wrong CPU architecture, a changed checksum, an unsafe path,
a development cache, a source-control directory, an environment file, or a detected
secret. Windows packages contain native `linux/amd64` images. Apple Silicon packages
contain native `linux/arm64` images and must not use x64 emulation.

The checksum contract is exact: every non-directory ZIP entry except
`SHA256SUMS.txt` must be listed once, every listed file must exist, and no unlisted
file is accepted. Paths containing backslashes, drive letters, absolute roots, UNC
prefixes, empty components, or `..` components are rejected before extraction. The
callable `verify_checksum_manifest()` function validates an extracted package, while
`verify_package_archive()` validates the final ZIP before it replaces an existing
release archive. A failed final validation leaves the previous archive unchanged.

macOS application-bundle directories are stored explicitly in the ZIP. ZIP entries
use Unix metadata, and files below `Contents/MacOS` retain executable permission.
The safe extraction helper validates all paths and checksums before writing files,
then applies the recorded permission mode.

## Offline dependency compliance

The release builder does not grant or imply redistribution rights for Docker
Desktop, Microsoft Edge, WSL, or any other third-party installer. Docker Desktop and
the complete Microsoft Edge offline installer must be supplied in advance by the
person or institution performing the deployment under the applicable vendor terms.
The final release must not be published or handed to another party until that party
has confirmed the required license and redistribution permissions.

Windows x64 packaging requires user-supplied Docker Desktop, WSL, and Microsoft Edge
offline dependencies. Apple Silicon packaging requires user-supplied Docker Desktop
and Microsoft Edge offline dependencies. Every dependency manifest entry must record
`path`, `sha256`, `vendor`, `version`, `source`, `signing`, `publisher`, and
`license_evidence`. The signing and publisher evidence documents what was verified;
`license_evidence` records the deployer's compliance basis and is not a license
issued by Bakery AI. Package assembly validates the recorded hashes, official source,
vendor, publisher, and evidence presence. The platform installer must still validate
the operating-system signature immediately before installation. Missing dependencies,
incomplete evidence, changed hashes, and unsafe paths block package creation.

## Acceptance record

For each final platform release, record the package size, installed size, first-start
time, warm-start time, peak memory, Docker Desktop version, operating-system version,
and the result of install, restart, upgrade, rollback, standard uninstall, reinstall,
and complete-removal tests. These measurements must come from the reference Windows
computer and the M-series Mac; they must not be estimated on another platform.
