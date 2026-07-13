from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _requirement_names(path: Path, visited: set[Path] | None = None) -> set[str]:
    resolved = path.resolve()
    seen = set() if visited is None else visited
    if resolved in seen:
        return set()
    seen.add(resolved)

    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(("-r ", "--requirement ")):
            _, included = stripped.split(maxsplit=1)
            names.update(_requirement_names(path.parent / included, seen))
            continue
        name = stripped.split("==", 1)[0].split(">=", 1)[0].split("<", 1)[0].strip().lower()
        names.add(name)
    return names


def _requirement_entries(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


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


def test_s5_manifest_delegates_to_unified_project_requirements():
    assert _requirement_entries(ROOT / "requirements-s5.txt") == [
        "-r requirements.txt"
    ]


def test_s5_start_script_uses_python_313_project_runtime():
    script = ROOT / "start_s5.bat"

    assert script.exists()
    content = script.read_text(encoding="utf-8").lower()

    assert "hermes" not in content
    assert "c:\\users" not in content
    assert "%localappdata%\\bakeryai\\venv313\\scripts\\python.exe" in content
    assert "py -3.13 -m venv" in content
    assert "-m s5_agent.server" in content
