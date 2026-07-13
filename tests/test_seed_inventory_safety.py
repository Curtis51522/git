import importlib
import sys
from pathlib import Path

import pytest


MODULE_NAMES = (
    "scripts.seed_inventory",
)


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_seed_module_import_never_connects_to_database(
    module_name,
    monkeypatch,
):
    from db import mysql_client

    def unexpected_connection(*_args, **_kwargs):
        raise AssertionError("seed module import must not connect to the database")

    monkeypatch.setattr(mysql_client, "get_db", unexpected_connection)
    sys.modules.pop(module_name, None)

    importlib.import_module(module_name)


def test_seed_inventory_defaults_to_dry_run_without_database_access(
    monkeypatch,
    capsys,
):
    module = importlib.import_module("scripts.seed_inventory")

    def unexpected_connection(*_args, **_kwargs):
        raise AssertionError("dry run must not connect to the database")

    monkeypatch.setattr(module, "get_db", unexpected_connection)

    assert module.main([]) == 0
    assert "No database changes were made" in capsys.readouterr().out


def test_seed_inventory_requires_explicit_apply_and_database_confirmation():
    source = Path("scripts/seed_inventory.py").read_text(encoding="utf-8")

    assert '"--apply"' in source
    assert '"--confirm-database"' in source
    assert 'if __name__ == "__main__"' in source


def test_legacy_seed_inventory_entrypoint_is_removed():
    assert not Path("scripts/seed_inventory_new.py").exists()
