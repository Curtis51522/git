from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_script(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8").lower()


def test_main_start_script_uses_project_venv() -> None:
    script = _read_script("start.bat")

    assert ".venv\\scripts\\python.exe" in script
    assert "python313" not in script
    assert "hermes" not in script
    assert "main.py" in script


def test_s5_start_script_uses_project_venv() -> None:
    script = _read_script("start_s5.bat")

    assert ".venv\\scripts\\python.exe" in script
    assert "hermes" not in script
    assert "python313" not in script
    assert "-m s5_agent.server" in script


def test_start_all_script_runs_main_and_s5_from_project_venv() -> None:
    script = _read_script("start_all.bat")

    assert ".venv\\scripts\\python.exe" in script
    assert "main.py" in script
    assert "-m s5_agent.server" in script
    assert "python313" not in script
    assert "hermes" not in script

