import asyncio
from datetime import datetime
from pathlib import Path

import pytest

import main as system_main
from api import freshness_service
from api.operation_clock import (
    get_operation_time,
    operation_now,
    operation_time_scope,
    parse_operation_time,
)


FRONTEND = Path("api/module4_frontend/static/index.html")


def test_operation_time_is_manager_replay_only_and_date_limited(monkeypatch):
    monkeypatch.setenv("BAKERY_OPERATION_REPLAY", "1")
    selected = parse_operation_time("2026-06-24T08:15:00")

    with operation_time_scope(selected):
        assert get_operation_time() == selected
        assert operation_now() == selected

    assert get_operation_time() is None

    with pytest.raises(ValueError, match="outside the allowed replay period"):
        parse_operation_time("2026-06-23T08:15:00")

    with pytest.raises(ValueError, match="outside the allowed replay period"):
        parse_operation_time("2026-07-25T08:15:00")


def test_operation_time_is_disabled_without_explicit_runtime_flag(monkeypatch):
    monkeypatch.delenv("BAKERY_OPERATION_REPLAY", raising=False)

    with pytest.raises(ValueError, match="not enabled"):
        parse_operation_time("2026-07-20T08:15:00")


def test_frontend_sends_selected_operation_time_with_normal_requests():
    source = FRONTEND.read_text(encoding="utf-8")

    assert 'id="operation-date"' in source
    assert 'id="operation-time"' in source
    assert "X-Operation-At" in source
    assert "2026-06-24" in source
    assert "2026-07-24" in source


def test_frontend_does_not_send_operation_time_with_login_requests():
    source = FRONTEND.read_text(encoding="utf-8")
    start = source.index("window.fetch=function")
    end = source.index("function canAccessPanel", start)
    block = source[start:end]

    assert "requestPath" in block
    assert "'/s4/login'" in block
    assert "requestPath!=='/s4/login'" in block


def test_frontend_keeps_browser_api_requests_same_origin():
    source = FRONTEND.read_text(encoding="utf-8")

    assert "replace('localhost','127.0.0.1')" not in source
    assert source.count("API=window.location.origin") >= 2


def test_manager_replay_uses_the_operation_date_as_today_for_attendance():
    source = FRONTEND.read_text(encoding="utf-8")
    start = source.index("function getLocalToday")
    end = source.index("function getSelectedOperationalDate", start)
    block = source[start:end]

    assert "role==='manager'&&token" in block
    assert "return getOperationDate();" in block


def test_replay_runtime_does_not_age_inventory_during_service_startup(monkeypatch):
    calls = []
    monkeypatch.setenv("BAKERY_OPERATION_REPLAY", "1")
    monkeypatch.setattr(freshness_service, "update_all_freshness", lambda: calls.append("updated"))

    async def run_lifespan():
        async with system_main.lifespan(None):
            await asyncio.sleep(0)

    asyncio.run(run_lifespan())

    assert calls == []
