# Bakery AI Final Offline Installer

## End-user entry

Extract the complete platform ZIP and open the installer without command-line
arguments:

- Windows x64: `Install Bakery AI.exe`
- Apple Silicon macOS: `BakeryAI Installer.app`

The installer locates the adjacent `payload` folder, detects the supported
platform, and supplies default application, data, and backup locations. The user
can select `install`, `repair`, `upgrade`, or `uninstall` in the setup window.
The packaged platform launcher is read from `payload/launcher/Bakery AI.exe` or
`payload/launcher/Bakery AI.app`.

## Verified offline prerequisites

On a clean Windows computer, the installer verifies and installs the supplied WSL,
Docker Desktop, and Microsoft Edge packages when required. A WSL reboot is resumed
once through `--resume` after the continuation manifest hash is revalidated. On
Apple Silicon macOS, the installer verifies and installs the supplied Docker DMG
and Edge PKG, starts Docker Desktop, and waits for the Docker Engine before loading
images. If Docker cannot be started automatically, the error tells the user to
start Docker Desktop and run the same installer again.

## Database and lifecycle contract

The ZIP contains the database initialization files at the same path mounted by
Compose:

```text
payload/deployment/database/init/001-final-snapshot.sql
payload/deployment/database/init/001-final-snapshot.sql.sha256.json
payload/deployment/database/init/999-deployment-ready.sql
```

Standard uninstall removes only the displayed application and launcher files. It
retains business data, configuration, backups, and the MySQL Docker volume.
Complete removal displays every exact file and the managed volume and requires a
second confirmation phrase. The installer never follows symlinks or recursively
deletes a directory.

