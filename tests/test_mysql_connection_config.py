import importlib.util
from pathlib import Path

import dotenv
import pytest


MYSQL_ENVIRONMENT_KEYS = (
    "MYSQL_HOST",
    "MYSQL_PORT",
    "MYSQL_USER",
    "MYSQL_PASSWORD",
    "MYSQL_DATABASE",
)

ROOT = Path(__file__).resolve().parents[1]


def _clear_mysql_environment(monkeypatch):
    for key in MYSQL_ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_mysql_config_loads_the_project_env_file_on_direct_import(monkeypatch):
    _clear_mysql_environment(monkeypatch)
    calls = []

    def fake_load_dotenv(path, *, override=False):
        calls.append((Path(path), override))
        monkeypatch.setenv("MYSQL_HOST", "127.0.0.1")
        monkeypatch.setenv("MYSQL_PORT", "3307")
        monkeypatch.setenv("MYSQL_USER", "bakery_app")
        monkeypatch.setenv("MYSQL_PASSWORD", "secret")
        monkeypatch.setenv("MYSQL_DATABASE", "bakery_ai")

    monkeypatch.setattr(dotenv, "load_dotenv", fake_load_dotenv)
    spec = importlib.util.spec_from_file_location(
        "mysql_config_direct_import_test",
        ROOT / "db" / "mysql_config.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert calls == [(ROOT / ".env", False)]
    assert module.get_mysql_connection_config() == {
        "host": "127.0.0.1",
        "port": 3307,
        "user": "bakery_app",
        "password": "secret",
        "database": "bakery_ai",
    }


def test_connection_config_uses_explicit_loopback_and_standard_port_by_default(
    monkeypatch,
):
    from db.mysql_config import get_mysql_connection_config

    _clear_mysql_environment(monkeypatch)

    assert get_mysql_connection_config() == {
        "host": "127.0.0.1",
        "port": 3306,
        "user": "root",
        "password": "",
        "database": "bakery_ai",
    }


def test_connection_config_parses_an_explicit_port(monkeypatch):
    from db.mysql_config import get_mysql_connection_config

    monkeypatch.setenv("MYSQL_HOST", "db.example.test")
    monkeypatch.setenv("MYSQL_PORT", "3307")
    monkeypatch.setenv("MYSQL_USER", "bakery_app")
    monkeypatch.setenv("MYSQL_PASSWORD", "secret")
    monkeypatch.setenv("MYSQL_DATABASE", "bakery_ai_build")

    assert get_mysql_connection_config() == {
        "host": "db.example.test",
        "port": 3307,
        "user": "bakery_app",
        "password": "secret",
        "database": "bakery_ai_build",
    }


@pytest.mark.parametrize("value", ["", "not-a-port", "0", "65536"])
def test_connection_config_rejects_invalid_ports(monkeypatch, value):
    from db.mysql_config import get_mysql_connection_config

    monkeypatch.setenv("MYSQL_PORT", value)

    with pytest.raises(ValueError, match="MYSQL_PORT"):
        get_mysql_connection_config()
