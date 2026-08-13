# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path


PROJECT_ROOT = Path(SPECPATH).parent


analysis = Analysis(
    [str(PROJECT_ROOT / "deployment" / "launcher" / "main.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=[
        "deployment.launcher.browser",
        "deployment.launcher.config",
        "deployment.launcher.database",
        "deployment.launcher.docker_runtime",
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
    name="Bakery AI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(PROJECT_ROOT / "deployment" / "assets" / "bakery-ai.ico"),
)
if sys.platform == "darwin":
    application = BUNDLE(
        executable,
        name="Bakery AI.app",
        bundle_identifier="com.bakeryai.launcher",
    )
