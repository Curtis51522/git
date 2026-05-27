#!/usr/bin/env python3
"""
Download pre-trained model files from GitHub Releases.

Usage: python download_models.py

Downloads and extracts:
  - models/distilbert/   (~519 MB, intent classifier)
  - models/yolo/         (~50 MB,  freshness detector)

For private repos, set a GitHub Personal Access Token:
  PowerShell: $env:GITHUB_TOKEN = "ghp_xxxx"
  bash:       export GITHUB_TOKEN="ghp_xxxx"

Create a token at: https://github.com/settings/tokens
Required scope: repo (for private repos)
"""
import os
import sys
import zipfile
import urllib.request
import ssl

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


def _download(url: str, dest: str):
    """Download a file with progress bar, using GITHUB_TOKEN if set."""
    token = os.environ.get("GITHUB_TOKEN", "")

    if not token:
        print("Note: GITHUB_TOKEN not set. Private repos will fail.")
        print("Set: $env:GITHUB_TOKEN = 'ghp_xxxx'")
        print("Create token: https://github.com/settings/tokens\n")

    req = urllib.request.Request(url)
    req.add_header("Accept", "application/octet-stream")
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    ctx = ssl.create_default_context()

    with urllib.request.urlopen(req, context=ctx) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 8192

        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total > 0:
                    percent = min(int(downloaded * 100 / total), 100)
                    bar_len = 40
                    filled = int(bar_len * percent / 100)
                    bar = chr(9608) * filled + chr(9617) * (bar_len - filled)
                    mb_done = downloaded / (1024 * 1024)
                    mb_total = total / (1024 * 1024)
                    print(
                        f"\r  [{bar}] {percent:3d}%  {mb_done:.0f}/{mb_total:.0f} MB",
                        end="",
                    )
        print()


# ---- Main ------------------------------------------------------------
def main():
    os.chdir(PROJECT_ROOT)

    if _check_existing():
        return

    zip_path = os.path.join(PROJECT_ROOT, ASSET_NAME)

    print("Downloading models from GitHub Releases...")
    print(f"  {DOWNLOAD_URL}\n")
    try:
        _download(DOWNLOAD_URL, zip_path)
    except urllib.request.HTTPError as e:
        print(f"\nHTTP {e.code}: {e.reason}")
        if e.code == 404:
            print("Release not found. Check TAG and ASSET_NAME in this script.")
        elif e.code == 401 or e.code == 403:
            print("Authentication failed. Set GITHUB_TOKEN for private repos.")
        sys.exit(1)
    except Exception as e:
        print(f"\nDownload failed: {e}")
        sys.exit(1)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(PROJECT_ROOT)

    os.remove(zip_path)

    print("\nDone! Models installed:")
    print("  models/distilbert/  (intent classifier)")
    print("  models/yolo/        (freshness detector)")


if __name__ == "__main__":
    main()
