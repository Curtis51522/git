from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _manifest_lines() -> set[str]:
    return {
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_runtime_manifest_pins_compatible_fastapi_stack() -> None:
    requirements = _manifest_lines()

    assert "fastapi==0.115.0" in requirements
    assert "starlette==0.38.6" in requirements
    assert "httpx==0.27.2" in requirements
    assert not any(line.startswith("langgraph-api") for line in requirements)


def test_launchers_check_dependency_consistency_before_startup() -> None:
    for filename in ("start.bat", "start_all.bat", "start_s5.bat"):
        launcher = (ROOT / filename).read_text(encoding="utf-8")

        assert "-m pip check" in launcher, filename
        assert "BakeryAI\\venv313\\Scripts\\python.exe" in launcher, filename
