from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
START_SCRIPTS = ("start.bat", "start_all.bat", "start_s5.bat")


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _entries(path: str) -> list[str]:
    return [
        line.strip()
        for line in _read(path).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_development_requirements_include_runtime_and_pytest():
    path = ROOT / "requirements-dev.txt"

    assert path.exists()
    entries = _entries("requirements-dev.txt")
    assert "-r requirements.txt" in entries
    assert any(entry.lower().startswith("pytest==") for entry in entries)


def test_runtime_requirements_include_chinese_calendar():
    entries = {entry.lower() for entry in _entries("requirements.txt")}

    assert "chinese-calendar==1.11.0" in entries


def test_start_scripts_use_python_313_project_runtime():
    for script_name in START_SCRIPTS:
        content = _read(script_name).lower()

        assert 'cd /d "%~dp0"' in content
        assert "%localappdata%\\bakeryai\\venv313\\scripts\\python.exe" in content
        assert "py -3.13 -m venv" in content
        assert "sys.version_info[:2] == (3, 13)" in content
        assert "hermes" not in content
        assert "c:\\users" not in content
        assert "localhost" not in content


def test_start_all_uses_one_interpreter_for_both_services():
    content = _read("start_all.bat").lower()

    assert 'set "py=%bakery_python%"' in content
    assert content.count('start "') == 2
    assert "-m s5_agent.server" in content
    assert "main.py" in content


def test_environment_example_uses_explicit_loopback_address():
    content = _read(".env.example")

    assert "MYSQL_HOST=127.0.0.1" in content
    assert "MYSQL_PORT=3307" in content
    assert "MYSQL_USER=bakery_app" in content
    assert "localhost" not in content.lower()


def test_readme_documents_versioned_environment_setup():
    content = _read("README.md").lower()

    assert "python 3.13" in content
    assert "mysql community server 8.4" in content
    assert "py -3.13 -m venv" in content
    assert "-r requirements.txt" in content
    assert "-r requirements-dev.txt" in content
    assert "python -m venv .venv" not in content


def test_main_and_s5_import_in_the_same_interpreter():
    command = [
        sys.executable,
        "-c",
        "import main; import s5_agent.server; print('runtime-import-ok')",
    ]

    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "runtime-import-ok"


def test_main_uses_fastapi_lifespan_for_periodic_freshness():
    content = _read("main.py")

    assert "@app.on_event" not in content
    assert "lifespan=lifespan" in content
    assert "logger.exception" in content
