#!/usr/bin/env python3
"""
Download pre-trained model files from GitHub Releases.

Usage: python download_models.py

This downloads and extracts the model files needed to run the system:
  - models/distilbert/   (~519 MB, intent classifier)
  - models/yolo/         (~50 MB,  freshness detector)
"""
import os
import sys
import zipfile
import urllib.request

# ---- Config ----------------------------------------------------------
REPO_OWNER = "Curtis51522"
REPO_NAME = "git"
TAG = "v1.0.0"
ASSET_NAME = "models.zip"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DOWNLOAD_URL = (
    f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/"
    f"{TAG}/{ASSET_NAME}"
)


# ---- Helpers ---------------------------------------------------------
def _progress(count: int, block_size: int, total_size: int):
    """Display a simple download progress bar."""
    if total_size > 0:
        percent = min(int(count * block_size * 100 / total_size), 100)
        bar_len = 40
        filled = int(bar_len * percent / 100)
        bar = chr(9608) * filled + chr(9617) * (bar_len - filled)
        mb_done = count * block_size / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r  [{bar}] {percent:3d}%  {mb_done:.0f}/{mb_total:.0f} MB", end="")
        if percent == 100:
            print()


def _check_existing() -> bool:
    """Return True if model files already exist (skip download)."""
    required = [
        os.path.join("models", "distilbert", "config.json"),
        os.path.join("models", "distilbert", "model.safetensors"),
        os.path.join("models", "distilbert", "tokenizer.json"),
        os.path.join("models", "distilbert", "tokenizer_config.json"),
        os.path.join("models", "yolo", "best.pt"),
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(PROJECT_ROOT, f))]
    if not missing:
        print("All model files already exist. Skipping download.")
        return True
    return False


# ---- Main ------------------------------------------------------------
def main():
    os.chdir(PROJECT_ROOT)

    if _check_existing():
        return

    zip_path = os.path.join(PROJECT_ROOT, ASSET_NAME)

    print("Downloading models from GitHub Releases...")
    print(f"  URL: {DOWNLOAD_URL}")
    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, zip_path, _progress)
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nManual download instructions:")
        print(f"  1. Open: {DOWNLOAD_URL}")
        print(f"  2. Save to: {zip_path}")
        print(f"  3. Run: python download_models.py")
        sys.exit(1)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(PROJECT_ROOT)

    os.remove(zip_path)

    print("\nModels installed successfully!")
    print("  - models/distilbert/  (intent classifier)")
    print("  - models/yolo/        (freshness detector)")


if __name__ == "__main__":
    main()
