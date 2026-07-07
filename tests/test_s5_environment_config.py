from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _requirement_names(path: Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name = stripped.split("==", 1)[0].split(">=", 1)[0].split("<", 1)[0].strip().lower()
        names.add(name)
    return names


def test_s5_runtime_requirements_are_declared():
    requirements = ROOT / "requirements-s5.txt"

    assert requirements.exists()
    names = _requirement_names(requirements)

    assert {
        "fastapi",
        "uvicorn",
        "httpx",
        "pydantic",
        "python-dotenv",
        "mysql-connector-python",
        "langgraph",
        "pyyaml",
    }.issubset(names)


def test_project_requirements_include_yaml_loader_dependency():
    names = _requirement_names(ROOT / "requirements.txt")

    assert "pyyaml" in names


def test_s5_start_script_uses_project_virtual_environment():
    script = ROOT / "start_s5.bat"

    assert script.exists()
    content = script.read_text(encoding="utf-8").lower()

    assert "hermes" not in content
    assert "c:\\users" not in content
    assert ".venv\\scripts\\python.exe" in content
    assert "-m s5_agent.server" in content
