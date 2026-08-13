"""Shared offline installer lifecycle for Bakery AI."""

from deployment.installer.main import (
    InstallLocations,
    InstallerLifecycle,
    LifecycleRequest,
    UpgradeRollbackError,
)
from deployment.installer.uninstall import (
    RemovalMode,
    RemovalPaths,
    build_removal_plan,
    execute_removal,
)

__all__ = [
    "InstallLocations",
    "InstallerLifecycle",
    "LifecycleRequest",
    "RemovalMode",
    "RemovalPaths",
    "UpgradeRollbackError",
    "build_removal_plan",
    "execute_removal",
]
