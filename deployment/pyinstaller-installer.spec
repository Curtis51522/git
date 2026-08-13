# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path


PROJECT_ROOT = Path(SPECPATH).parent


analysis = Analysis(
    [str(PROJECT_ROOT / "deployment" / "installer" / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=[
        "deployment.installer.macos",
        "deployment.installer.operations",
        "deployment.installer.uninstall",
        "deployment.installer.windows",
        "deployment.release.build_images",
        "deployment.release.verify_payload",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["api", "models", "s2_forecasting", "s5_agent", "torch", "ultralytics"],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
executable = EXE(
    pyz,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="Install Bakery AI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    uac_admin=True,
    icon=str(PROJECT_ROOT / "deployment" / "assets" / "bakery-ai.ico"),
)
if sys.platform == "darwin":
    application = BUNDLE(
        executable,
        name="BakeryAI Installer.app",
        bundle_identifier="com.bakeryai.installer",
    )
