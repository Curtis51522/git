from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Callable, Mapping, Protocol, Sequence


Manifest = Mapping[str, Any]


@dataclass(frozen=True)
class InstallLocations:
    application_root: Path
    data_root: Path
    backup_root: Path
    launcher_path: Path
    runtime_env_path: Path
    database_volume: str

    def __post_init__(self) -> None:
        if not self.database_volume.strip():
            raise ValueError("Database volume is required")
        for field_name in (
            "application_root",
            "data_root",
            "backup_root",
            "launcher_path",
            "runtime_env_path",
        ):
            value = Path(getattr(self, field_name)).expanduser().resolve()
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True)
class LifecycleRequest:
    payload_root: Path
    target: str
    locations: InstallLocations
    backup_destination: Path | None = None

    def __post_init__(self) -> None:
        if not self.target.strip():
            raise ValueError("Release target is required")
        object.__setattr__(
            self,
            "payload_root",
            Path(self.payload_root).expanduser().resolve(),
        )
        if self.backup_destination is not None:
            object.__setattr__(
                self,
                "backup_destination",
                Path(self.backup_destination).expanduser().resolve(),
            )


class InstallerOperations(Protocol):
    def verify_payload(self, payload_root: Path, target: str) -> Manifest: ...

    def check_prerequisites(self, target: str, manifest: Manifest) -> None: ...

    def copy_payload(
        self,
        payload_root: Path,
        locations: InstallLocations,
        manifest: Manifest,
    ) -> None: ...

    def load_images(self, payload_root: Path, manifest: Manifest) -> None: ...

    def initialize_runtime(self, locations: InstallLocations) -> None: ...

    def initialize_database(self, locations: InstallLocations) -> None: ...

    def install_launcher(
        self,
        payload_root: Path,
        locations: InstallLocations,
        manifest: Manifest,
    ) -> None: ...

    def verify_health(self, locations: InstallLocations) -> None: ...

    def backup_database(
        self,
        locations: InstallLocations,
        destination: Path,
    ) -> Path: ...

    def capture_release(self, locations: InstallLocations) -> object: ...

    def restore_release(
        self,
        locations: InstallLocations,
        previous_release: object,
    ) -> None: ...

    def restore_database(
        self,
        locations: InstallLocations,
        backup: Path,
    ) -> None: ...

    def reload_runtime(self, locations: InstallLocations) -> None: ...


OperationsFactory = Callable[[], InstallerOperations]


@dataclass(frozen=True)
class InstallerDefaults:
    payload_root: Path
    target: str
    locations: InstallLocations


@dataclass(frozen=True)
class InstallerChoice:
    action: str
    defaults: InstallerDefaults
    complete_removal: bool = False
    confirmed_complete_phrase: str | None = None


class UpgradeRollbackError(RuntimeError):
    def __init__(self, upgrade_error: Exception, rollback_error: Exception) -> None:
        super().__init__(
            "Upgrade failed and the previous release could not be restored: "
            f"{rollback_error}"
        )
        self.upgrade_error = upgrade_error
        self.rollback_error = rollback_error


class RollbackStepsError(RuntimeError):
    def __init__(self, failures: list[tuple[str, Exception]]) -> None:
        details = "; ".join(f"{name}: {error}" for name, error in failures)
        super().__init__(f"Rollback steps failed: {details}")
        self.failures = tuple(failures)


class InstallerLifecycle:
    def __init__(self, operations: InstallerOperations) -> None:
        self.operations = operations

    def _prepare(self, request: LifecycleRequest) -> Manifest:
        manifest = self.operations.verify_payload(
            request.payload_root,
            request.target,
        )
        self.operations.check_prerequisites(request.target, manifest)
        return manifest

    def _deploy(
        self,
        request: LifecycleRequest,
        manifest: Manifest,
        *,
        initialize_database: bool,
        preserved_runtime: bytes | None = None,
    ) -> None:
        locations = request.locations
        self.operations.copy_payload(
            request.payload_root,
            locations,
            manifest,
        )
        self.operations.load_images(request.payload_root, manifest)
        try:
            self.operations.initialize_runtime(locations)
        finally:
            _restore_file(locations.runtime_env_path, preserved_runtime)
        if initialize_database:
            self.operations.initialize_database(locations)
        self.operations.install_launcher(
            request.payload_root,
            locations,
            manifest,
        )
        self.operations.reload_runtime(locations)
        self.operations.verify_health(locations)

    def install(self, request: LifecycleRequest) -> Manifest:
        manifest = self._prepare(request)
        self._deploy(request, manifest, initialize_database=True)
        return manifest

    def repair(self, request: LifecycleRequest) -> Manifest:
        manifest = self._prepare(request)
        runtime_env = request.locations.runtime_env_path
        preserved_runtime = (
            runtime_env.read_bytes() if runtime_env.is_file() else None
        )
        self._deploy(
            request,
            manifest,
            initialize_database=False,
            preserved_runtime=preserved_runtime,
        )
        return manifest

    def upgrade(self, request: LifecycleRequest) -> Manifest:
        manifest = self._prepare(request)
        destination = request.backup_destination
        if destination is None:
            raise ValueError("Upgrade backup destination is required")
        destination.parent.mkdir(parents=True, exist_ok=True)
        database_backup = self.operations.backup_database(
            request.locations,
            destination,
        )
        previous_release = self.operations.capture_release(request.locations)
        runtime_env = request.locations.runtime_env_path
        preserved_runtime = (
            runtime_env.read_bytes() if runtime_env.is_file() else None
        )
        try:
            self._deploy(
                request,
                manifest,
                initialize_database=True,
                preserved_runtime=preserved_runtime,
            )
        except Exception as upgrade_error:
            rollback_failures = self._rollback_upgrade(
                request.locations,
                previous_release=previous_release,
                database_backup=database_backup,
                runtime_content=preserved_runtime,
            )
            if rollback_failures:
                raise UpgradeRollbackError(
                    upgrade_error,
                    RollbackStepsError(rollback_failures),
                ) from upgrade_error
            raise
        return manifest

    def _rollback_upgrade(
        self,
        locations: InstallLocations,
        *,
        previous_release: object,
        database_backup: Path,
        runtime_content: bytes | None,
    ) -> list[tuple[str, Exception]]:
        failures: list[tuple[str, Exception]] = []
        steps: tuple[tuple[str, Callable[[], None]], ...] = (
            (
                "application release",
                lambda: self.operations.restore_release(
                    locations,
                    previous_release,
                ),
            ),
            (
                "database",
                lambda: self.operations.restore_database(
                    locations,
                    database_backup,
                ),
            ),
            (
                "runtime configuration",
                lambda: _restore_file(
                    locations.runtime_env_path,
                    runtime_content,
                ),
            ),
            ("runtime reload", lambda: self.operations.reload_runtime(locations)),
            ("health verification", lambda: self.operations.verify_health(locations)),
        )
        for name, step in steps:
            try:
                step()
            except Exception as error:
                failures.append((name, error))
        return failures


def _restore_file(path: Path, content: bytes | None) -> None:
    if content is None:
        return
    try:
        if path.is_file() and path.read_bytes() == content:
            return
    except OSError:
        pass
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


def load_operations_factory(reference: str) -> OperationsFactory:
    module_name, separator, attribute_name = reference.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError(
            "Operations factory must use the form 'module.path:callable'"
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise ValueError(f"Operations factory is not callable: {reference}")
    return factory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install, repair, upgrade, or uninstall Bakery AI."
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=("install", "repair", "upgrade", "uninstall"),
    )
    parser.add_argument("--payload-root", type=Path)
    parser.add_argument("--target")
    parser.add_argument("--application-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--launcher-path", type=Path)
    parser.add_argument("--runtime-env-path", type=Path)
    parser.add_argument("--database-volume")
    parser.add_argument("--backup-destination", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--complete-removal", action="store_true")
    parser.add_argument("--confirm-complete")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--operations-factory",
        help=(
            "Import reference for the platform operations factory, in "
            "'module.path:callable' form."
        ),
    )
    return parser


def _target_for_system(system: str) -> str:
    if system == "Windows":
        return "windows-x64"
    if system == "Darwin":
        return "macos-apple-silicon"
    raise RuntimeError("Bakery AI supports Windows x64 and Apple Silicon macOS")


def _find_payload_root(executable: Path) -> Path:
    candidates = [executable.parent]
    candidates.extend(executable.parents[:6])
    for root in candidates:
        payload = root / "payload"
        if (payload / "release.json").is_file():
            return payload.resolve()
    raise FileNotFoundError(
        "The verified payload folder was not found beside the installer"
    )


def default_installer_paths(
    *,
    system: str | None = None,
    environment: Mapping[str, str] | None = None,
    executable: str | Path | None = None,
) -> InstallerDefaults:
    operating_system = system or platform.system()
    values = os.environ if environment is None else environment
    executable_path = Path(executable or sys.executable).expanduser().resolve()
    payload_root = _find_payload_root(executable_path)
    target = _target_for_system(operating_system)
    if operating_system == "Windows":
        local = values.get("LOCALAPPDATA")
        if not local:
            raise RuntimeError("LOCALAPPDATA is required")
        data_root = Path(local) / "BakeryAI"
        application_root = Path(local) / "Programs" / "BakeryAI"
        launcher_path = application_root / "Bakery AI.exe"
    else:
        home = Path(values.get("HOME") or Path.home())
        data_root = home / "Library" / "Application Support" / "BakeryAI"
        application_root = data_root / "application"
        launcher_path = home / "Applications" / "Bakery AI.app"
    locations = InstallLocations(
        application_root=application_root,
        data_root=data_root,
        backup_root=data_root / "backups",
        launcher_path=launcher_path,
        runtime_env_path=data_root / "runtime.env",
        database_volume="bakery-ai_mysql-data",
    )
    return InstallerDefaults(
        payload_root=payload_root,
        target=target,
        locations=locations,
    )


def choose_interactively(defaults: InstallerDefaults) -> InstallerChoice | None:
    import tkinter as tk
    from tkinter import messagebox, simpledialog, ttk

    root = tk.Tk()
    root.title("Bakery AI Setup")
    root.resizable(False, False)
    result: list[InstallerChoice] = []
    action = tk.StringVar(value="install")
    application = tk.StringVar(value=str(defaults.locations.application_root))
    data = tk.StringVar(value=str(defaults.locations.data_root))
    complete_removal = tk.BooleanVar(value=False)

    ttk.Label(root, text="Bakery AI Setup", font=("Segoe UI", 14, "bold")).grid(
        row=0, column=0, columnspan=2, padx=18, pady=(16, 10), sticky="w"
    )
    ttk.Label(root, text="Action").grid(row=1, column=0, padx=18, pady=5, sticky="w")
    action_box = ttk.Combobox(
        root,
        textvariable=action,
        values=("install", "repair", "upgrade", "uninstall"),
        state="readonly",
        width=36,
    )
    action_box.grid(row=1, column=1, padx=18, pady=5)
    ttk.Label(root, text="Application folder").grid(
        row=2, column=0, padx=18, pady=5, sticky="w"
    )
    ttk.Entry(root, textvariable=application, width=48).grid(
        row=2, column=1, padx=18, pady=5
    )
    ttk.Label(root, text="Data and backup folder").grid(
        row=3, column=0, padx=18, pady=5, sticky="w"
    )
    ttk.Entry(root, textvariable=data, width=48).grid(
        row=3, column=1, padx=18, pady=5
    )
    ttk.Checkbutton(
        root,
        text="Delete business data, configuration, backups, and database volume",
        variable=complete_removal,
    ).grid(row=4, column=0, columnspan=2, padx=18, pady=5, sticky="w")

    def submit() -> None:
        app_root = Path(application.get()).expanduser()
        data_root = Path(data.get()).expanduser()
        locations = InstallLocations(
            application_root=app_root,
            data_root=data_root,
            backup_root=data_root / "backups",
            launcher_path=(
                app_root / "Bakery AI.exe"
                if defaults.target == "windows-x64"
                else Path.home() / "Applications" / "Bakery AI.app"
            ),
            runtime_env_path=data_root / "runtime.env",
            database_volume=defaults.locations.database_volume,
        )
        confirmation: str | None = None
        if action.get() == "uninstall" and complete_removal.get():
            phrase = f"REMOVE {defaults.locations.database_volume}"
            preview = _build_removal_plan(
                LifecycleRequest(
                    payload_root=defaults.payload_root,
                    target=defaults.target,
                    locations=locations,
                ),
                complete=True,
            ).preview_text
            if not messagebox.askyesno(
                "Complete removal",
                "The following exact files and Docker volume will be removed:\n\n"
                f"{preview}\n\nContinue?",
                parent=root,
            ):
                return
            confirmation = simpledialog.askstring(
                "Complete removal",
                f"Type {phrase} to confirm:",
                parent=root,
            )
            if confirmation != phrase:
                messagebox.showerror(
                    "Complete removal",
                    "The confirmation phrase did not match.",
                    parent=root,
                )
                return
        elif action.get() == "uninstall":
            preview = _build_removal_plan(
                LifecycleRequest(
                    payload_root=defaults.payload_root,
                    target=defaults.target,
                    locations=locations,
                ),
                complete=False,
            ).preview_text
            if not messagebox.askyesno(
                "Standard uninstall",
                "Business data, configuration, backups, and the database volume "
                "will be retained. The following exact application files will be "
                f"removed:\n\n{preview}\n\nContinue?",
                parent=root,
            ):
                return
        result.append(
            InstallerChoice(
                action=action.get(),
                defaults=InstallerDefaults(
                    payload_root=defaults.payload_root,
                    target=defaults.target,
                    locations=locations,
                ),
                complete_removal=complete_removal.get(),
                confirmed_complete_phrase=confirmation,
            )
        )
        root.destroy()

    ttk.Button(root, text="Continue", command=submit).grid(
        row=5, column=1, padx=18, pady=16, sticky="e"
    )
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    if not result:
        return None
    return result[0]


def _request_from_options(
    options: argparse.Namespace,
    defaults: InstallerDefaults | None = None,
) -> LifecycleRequest:
    needs_defaults = any(
        value is None
        for value in (
            options.payload_root,
            options.target,
            options.application_root,
            options.data_root,
            options.backup_root,
            options.launcher_path,
            options.runtime_env_path,
            options.database_volume,
        )
    )
    if defaults is None and needs_defaults:
        defaults = default_installer_paths()
    if defaults is None:
        defaults = InstallerDefaults(
            payload_root=Path(options.payload_root),
            target=str(options.target),
            locations=InstallLocations(
                application_root=Path(options.application_root),
                data_root=Path(options.data_root),
                backup_root=Path(options.backup_root),
                launcher_path=Path(options.launcher_path),
                runtime_env_path=Path(options.runtime_env_path),
                database_volume=str(options.database_volume),
            ),
        )
    locations = InstallLocations(
        application_root=options.application_root or defaults.locations.application_root,
        data_root=options.data_root or defaults.locations.data_root,
        backup_root=options.backup_root or defaults.locations.backup_root,
        launcher_path=options.launcher_path or defaults.locations.launcher_path,
        runtime_env_path=(
            options.runtime_env_path or defaults.locations.runtime_env_path
        ),
        database_volume=(
            options.database_volume or defaults.locations.database_volume
        ),
    )
    backup_destination = options.backup_destination
    if options.action == "upgrade" and backup_destination is None:
        backup_destination = (
            locations.backup_root / "pre-upgrade-database.sql"
        )
    return LifecycleRequest(
        payload_root=options.payload_root or defaults.payload_root,
        target=options.target or defaults.target,
        locations=locations,
        backup_destination=backup_destination,
    )


def _build_removal_plan(
    request: LifecycleRequest,
    *,
    complete: bool,
) -> Any:
    from deployment.installer.uninstall import (
        RemovalMode,
        RemovalPaths,
        build_removal_plan,
    )

    def listed_files(root: Path) -> tuple[Path, ...]:
        files: list[Path] = []
        if not root.is_dir() or root.is_symlink():
            return ()
        for current, directories, names in os.walk(root, followlinks=False):
            current_path = Path(current)
            directories[:] = [
                name
                for name in directories
                if not (current_path / name).is_symlink()
            ]
            files.extend(
                current_path / name
                for name in names
                if not (current_path / name).is_symlink()
            )
        return tuple(files)

    application_files = listed_files(request.locations.application_root)
    launcher_files = listed_files(request.locations.launcher_path)
    if (
        request.locations.launcher_path.is_file()
        or request.locations.launcher_path.is_symlink()
    ):
        launcher_files += (request.locations.launcher_path,)
    if request.target == "windows-x64":
        from deployment.installer.windows import windows_desktop_directory

        desktop = windows_desktop_directory()
        for desktop_entry in (
            desktop / "Bakery AI.lnk",
            desktop / "Bakery AI.exe",
        ):
            if desktop_entry.is_file() or desktop_entry.is_symlink():
                launcher_files += (desktop_entry,)
    configuration_files = listed_files(request.locations.data_root)
    backup_files = listed_files(request.locations.backup_root)
    mode = RemovalMode.COMPLETE if complete else RemovalMode.STANDARD
    return build_removal_plan(
        mode,
        RemovalPaths(
            application_files=application_files,
            launcher_files=launcher_files,
            configuration_files=configuration_files,
            backup_files=backup_files,
            database_volume=request.locations.database_volume,
        ),
    )


def _remove_installation(
    request: LifecycleRequest,
    *,
    complete: bool,
    confirmation: str | None,
) -> None:
    from deployment.installer.operations import create_platform_operations
    from deployment.installer.uninstall import execute_removal

    plan = _build_removal_plan(request, complete=complete)
    create_platform_operations(request).stop_runtime(request.locations)
    execute_removal(
        plan,
        first_confirmation=True,
        second_confirmation=confirmation,
    )


def default_operations_factory(request: LifecycleRequest) -> OperationsFactory:
    from deployment.installer.operations import create_platform_operations

    return lambda: create_platform_operations(request)


def _notify(message: str) -> None:
    try:
        from tkinter import messagebox

        messagebox.showinfo("Bakery AI Setup", message)
    except Exception:
        print(message)


def _record_installer_event(
    locations: InstallLocations,
    *,
    event: str,
    action: str,
    detail: str | None = None,
) -> Path:
    log_path = locations.data_root / "installer" / "install.log"
    normalized_detail = " ".join(str(detail or "").split())
    fields = [
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        f"event={event}",
        f"action={action}",
    ]
    if normalized_detail:
        fields.append(f"detail={normalized_detail[:4000]}")
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(" ".join(fields) + "\n")
    except OSError:
        pass
    return log_path


def main(
    arguments: Sequence[str] | None = None,
    *,
    operations_factory: OperationsFactory | None = None,
    ui_factory: Callable[[InstallerDefaults], InstallerChoice | None] | None = None,
    defaults_factory: Callable[[], InstallerDefaults] = default_installer_paths,
) -> int:
    parser = _parser()
    options = parser.parse_args(arguments)
    interactive = options.action is None and options.resume is None
    selected_defaults: InstallerDefaults | None = None
    if options.smoke:
        defaults_factory()
        return 0
    if options.resume is not None:
        from deployment.installer.windows import consume_continuation_state

        state = consume_continuation_state(options.resume)
        selected_defaults = defaults_factory()
        options.action = "install"
        options.payload_root = Path(state.payload_root)
        options.target = state.target
        options.application_root = Path(state.install_root)
        options.data_root = Path(state.data_root)
        options.backup_root = Path(state.backup_root)
        options.launcher_path = Path(state.launcher_path)
        options.runtime_env_path = Path(state.runtime_env_path)
        options.database_volume = state.database_volume
    if options.action is None:
        defaults = defaults_factory()
        choice = (ui_factory or choose_interactively)(defaults)
        if choice is None:
            return 0
        options.action = choice.action
        options.payload_root = choice.defaults.payload_root
        options.target = choice.defaults.target
        options.application_root = choice.defaults.locations.application_root
        options.data_root = choice.defaults.locations.data_root
        options.backup_root = choice.defaults.locations.backup_root
        options.launcher_path = choice.defaults.locations.launcher_path
        options.runtime_env_path = choice.defaults.locations.runtime_env_path
        options.database_volume = choice.defaults.locations.database_volume
        options.complete_removal = choice.complete_removal
        options.confirm_complete = choice.confirmed_complete_phrase
        selected_defaults = choice.defaults
    request = _request_from_options(
        options,
        selected_defaults,
    )
    if options.action == "uninstall":
        _remove_installation(
            request,
            complete=options.complete_removal,
            confirmation=options.confirm_complete,
        )
        return 0
    action = str(options.action)
    action_label = {
        "install": "installation",
        "repair": "repair",
        "upgrade": "upgrade",
    }.get(action, action)
    log_path = _record_installer_event(
        request.locations,
        event="start",
        action=action,
    )
    if interactive:
        _notify(
            "Bakery AI installation is starting. Docker Desktop and the offline "
            "images may take several minutes. Keep this window open."
        )
    try:
        factory = operations_factory
        if factory is None:
            reference = options.operations_factory or os.environ.get(
                "BAKERY_INSTALLER_OPERATIONS_FACTORY"
            )
            factory = (
                load_operations_factory(reference)
                if reference
                else default_operations_factory(request)
            )
        lifecycle = InstallerLifecycle(factory())
        operation = getattr(lifecycle, action)
        operation(request)
    except Exception as exc:
        from deployment.installer.operations import RebootRequired

        if isinstance(exc, RebootRequired):
            _record_installer_event(
                request.locations,
                event="reboot_required",
                action=action,
                detail=str(exc),
            )
            _notify(str(exc))
            return 0
        _record_installer_event(
            request.locations,
            event="failure",
            action=action,
            detail=str(exc),
        )
        message = (
            f"Bakery AI {action_label} failed.\n\n"
            f"{exc}\n\n"
            f"Installer log: {log_path}"
        )
        if interactive:
            _notify(message)
        else:
            print(message, file=sys.stderr)
        return 1
    _record_installer_event(
        request.locations,
        event="success",
        action=action,
    )
    if interactive:
        _notify(f"Bakery AI {action_label} completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
