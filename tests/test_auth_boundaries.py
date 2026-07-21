from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from jose import jwt
import pytest

import main as main_module
from api import module1_yolo, module2_forecast, module3_scheduling
from api.auth import create_access_token
from api.module4_frontend import bff
from config import settings as cfg
from main import app as system_app


def _app():
    app = FastAPI()
    app.include_router(module1_yolo.router)
    app.include_router(module2_forecast.router)
    app.include_router(module3_scheduling.router)
    app.include_router(bff.router)
    return app


def _route(router, path, method):
    return next(
        route
        for route in router.routes
        if route.path == path and method in route.methods
    )


def _dependency_names(route):
    return {dependency.call.__name__ for dependency in route.dependant.dependencies}


ANONYMOUS_ROUTES = {
    ("/ping", "GET"),
    ("/s4/login", "POST"),
}

STAFF_ROUTES = {
    ("/freshness/discounts", "GET"),
    ("/s1/batch_inventory", "GET"),
    ("/s1/checkout", "POST"),
    ("/s1/search", "GET"),
    ("/s3/attendance/punch", "POST"),
    ("/s4/beverages/options", "GET"),
    ("/s4/checkout/complete", "POST"),
    ("/s4/combo", "POST"),
    ("/s4/combo/select", "POST"),
    ("/s4/orders/receipt", "GET"),
    ("/s4/orders/today", "GET"),
    ("/s4/products", "GET"),
    ("/s5/{path:path}", "DELETE"),
    ("/s5/{path:path}", "GET"),
    ("/s5/{path:path}", "POST"),
    ("/s5/{path:path}", "PUT"),
}

MANAGER_ROUTES = {
    ("/freshness/update", "POST"),
    ("/s1/detection_logs", "GET"),
    ("/s1/inflow", "POST"),
    ("/s1/inflow/batch", "POST"),
    ("/s1/inflow/history", "GET"),
    ("/s1/inventory", "GET"),
    ("/s1/inventory_transactions", "GET"),
    ("/s2/accuracy", "GET"),
    ("/s2/business-events", "GET"),
    ("/s2/business-events", "POST"),
    ("/s2/business-events/{event_id}", "DELETE"),
    ("/s2/business-events/{event_id}", "PUT"),
    ("/s2/features/importance", "GET"),
    ("/s2/features/today", "GET"),
    ("/s2/forecast", "GET"),
    ("/s2/forecast/refresh", "GET"),
    ("/s2/sales_history", "GET"),
    ("/s3/attendance", "GET"),
    ("/s3/attendance/correct", "POST"),
    ("/s3/attendance/history", "GET"),
    ("/s3/eval", "GET"),
    ("/s3/kpi", "GET"),
    ("/s3/kpi/ranking", "GET"),
    ("/s3/materials", "GET"),
    ("/s3/plan/7day", "GET"),
    ("/s3/prep_acknowledge", "POST"),
    ("/s3/prep_checklist", "GET"),
    ("/s3/resync", "POST"),
    ("/s3/schedule", "GET"),
    ("/s3/sick", "POST"),
    ("/s3/solve", "POST"),
    ("/s3/swap", "POST"),
    ("/s3/unsick", "POST"),
    ("/s4/inventory/check", "POST"),
    ("/s4/inventory/check/history", "GET"),
    ("/s4/inventory/dashboard", "GET"),
    ("/s4/inventory/materials", "GET"),
    ("/s4/inventory/materials/theoretical", "GET"),
    ("/s4/inventory/restock", "POST"),
    ("/s4/inventory/restock/history", "GET"),
    ("/s4/inventory/stock-days-history", "GET"),
    ("/s4/inventory/wastage/summary", "GET"),
    ("/s4/orders/refund", "POST"),
    ("/s4/revenue/closing-loss", "GET"),
    ("/s4/revenue/daily", "GET"),
    ("/s4/revenue/historical", "GET"),
    ("/s4/revenue/hourly", "GET"),
}


def _token(role):
    return jwt.encode(
        {
            "sub": f"test-{role}",
            "role": role,
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        },
        cfg.JWT_SECRET,
        algorithm=cfg.JWT_ALGORITHM,
    )


def test_every_api_route_has_one_explicit_access_level():
    observed = {}
    for route in system_app.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in route.methods - {"HEAD", "OPTIONS"}:
            observed[(route.path, method)] = _dependency_names(route)

    expected = ANONYMOUS_ROUTES | STAFF_ROUTES | MANAGER_ROUTES
    assert set(observed) == expected

    for route_key in ANONYMOUS_ROUTES:
        assert observed[route_key].isdisjoint(
            {"get_current_user", "require_manager"}
        )
    for route_key in STAFF_ROUTES:
        assert "get_current_user" in observed[route_key]
        assert "require_manager" not in observed[route_key]
    for route_key in MANAGER_ROUTES:
        assert "require_manager" in observed[route_key]


def test_checkout_rejects_missing_token_before_payload_validation():
    client = TestClient(_app(), raise_server_exceptions=False)

    response = client.post("/s4/checkout/complete", json={"items": []})

    assert response.status_code == 401
    assert response.json() == {"detail": "Missing token"}


def test_direct_s1_deduction_route_is_not_exposed():
    assert not any(
        route.path == "/s1/deduct" and "POST" in route.methods
        for route in module1_yolo.router.routes
    )


def test_refund_requires_manager_before_payload_validation():
    client = TestClient(_app(), raise_server_exceptions=False)
    headers = {"Authorization": f"Bearer {_token('staff')}"}

    response = client.post("/s4/orders/refund", json={}, headers=headers)

    assert response.status_code == 403
    assert response.json() == {"detail": "Manager only"}


def test_business_event_mutation_requires_manager_before_database_access(monkeypatch):
    def unexpected_database_access(*_args, **_kwargs):
        raise AssertionError("authorization must run before database access")

    monkeypatch.setattr(module2_forecast, "get_db", unexpected_database_access)
    client = TestClient(_app(), raise_server_exceptions=False)
    headers = {"Authorization": f"Bearer {_token('staff')}"}

    response = client.post("/s2/business-events", json={}, headers=headers)

    assert response.status_code == 403
    assert response.json() == {"detail": "Manager only"}


def test_schedule_mutation_rejects_missing_token_before_solver_access(monkeypatch):
    def unexpected_solver_access(*_args, **_kwargs):
        raise AssertionError("authentication must run before solver access")

    monkeypatch.setattr(module3_scheduling, "_solve_impl", unexpected_solver_access)
    client = TestClient(_app(), raise_server_exceptions=False)

    response = client.post("/s3/solve", json={})

    assert response.status_code == 401
    assert response.json() == {"detail": "Missing token"}


@pytest.mark.parametrize(
    ("method", "path", "json_body"),
    [
        ("GET", "/s2/forecast", None),
        ("GET", "/s3/schedule", None),
        ("GET", "/s4/inventory/dashboard", None),
        ("GET", "/s4/revenue/daily", None),
        ("POST", "/s1/inflow/batch", {"items": []}),
        ("POST", "/s5/analyze/module", {}),
    ],
)
def test_staff_is_rejected_before_management_endpoint_work(
    method,
    path,
    json_body,
):
    client = TestClient(system_app, raise_server_exceptions=False)
    headers = {"Authorization": f"Bearer {_token('staff')}"}

    response = client.request(method, path, headers=headers, json=json_body)

    assert response.status_code == 403
    assert response.json() == {"detail": "Manager only"}


def test_production_auth_rejects_an_ephemeral_jwt_secret(monkeypatch):
    monkeypatch.setattr(cfg, "BAKERY_ENV", "production")
    monkeypatch.setattr(cfg, "JWT_SECRET_IS_EPHEMERAL", True)

    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        create_access_token("manager", "manager")


def test_production_auth_rejects_a_short_jwt_secret(monkeypatch):
    monkeypatch.setattr(cfg, "BAKERY_ENV", "production")
    monkeypatch.setattr(cfg, "JWT_SECRET_IS_EPHEMERAL", False)
    monkeypatch.setattr(cfg, "JWT_SECRET", "too-short")

    with pytest.raises(RuntimeError, match="at least 32"):
        create_access_token("manager", "manager")


def test_known_development_secrets_and_runtime_schema_bootstrap_are_removed():
    settings_source = Path("config/settings.py").read_text(encoding="utf-8")
    frontend_source = Path("api/module4_frontend/static/index.html").read_text(
        encoding="utf-8"
    )
    mysql_source = Path("db/mysql_client.py").read_text(encoding="utf-8")

    assert "dev-secret-change-me" not in settings_source
    assert 'value="hash123"' not in frontend_source
    assert "def init_db" not in mysql_source
    assert "def seed_defaults" not in mysql_source
    assert "def async_db" not in mysql_source
    assert "BAKERY_AUTO_INIT_DB" not in mysql_source
    assert "CREATE TABLE" not in mysql_source
    assert '"1" if BAKERY_ENV' not in settings_source
    assert "pwd_context.hash(req.password)" in Path(
        "api/module4_frontend/bff.py"
    ).read_text(encoding="utf-8")


def test_main_route_dependencies_match_role_boundaries():
    assert "require_manager" in _dependency_names(
        _route(system_app, "/freshness/update", "POST")
    )
    assert "get_current_user" in _dependency_names(
        _route(system_app, "/s5/{path:path}", "POST")
    )


def test_staff_can_use_s5_discount_support_endpoint(monkeypatch):
    class ProxyResponse:
        content = b'{"discounts": {}}'
        status_code = 200
        headers = {"content-type": "application/json"}

    class ProxyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return False

        async def request(self, **kwargs):
            assert kwargs["method"] == "POST"
            assert kwargs["url"] == "http://127.0.0.1:8001/discounts"
            return ProxyResponse()

    monkeypatch.setattr(
        main_module.httpx,
        "AsyncClient",
        lambda **_kwargs: ProxyClient(),
    )
    client = TestClient(system_app, raise_server_exceptions=False)
    headers = {"Authorization": f"Bearer {_token('staff')}"}

    response = client.post("/s5/discounts", json={"items": []}, headers=headers)

    assert response.status_code == 200
    assert response.json() == {"discounts": {}}


def test_frontend_applies_role_visibility_on_login_and_session_restore():
    source = Path("api/module4_frontend/static/index.html").read_text(
        encoding="utf-8"
    )
    restore = source[source.index("if(token&&role){") : source.index("var cartItems=[]")]
    login = source[source.index("async function doLogin") : source.index("function doLogout")]
    logout = source[source.index("function doLogout") : source.index("var _panelContainers")]
    show_panel = source[source.index("async function showPanel") : source.index("var _apiCache")]

    assert "applyRoleVisibility()" in restore
    assert "canAccessPanel(savedPanel)" in restore
    assert "sessionStorage.setItem('bakery_token',token)" in login
    assert "sessionStorage.setItem('bakery_role',role)" in login
    assert "sessionStorage.setItem('bakery_username',username)" in login
    assert "applyRoleVisibility()" in login
    assert "await showPanel('pos');loadPrices();" in login
    assert "showPanel('pos');loadStock()" not in login
    assert "sessionStorage.removeItem('bakery_token')" in logout
    assert "sessionStorage.removeItem('bakery_role')" in logout
    assert "sessionStorage.removeItem('bakery_username')" in logout
    assert "canAccessPanel(panel)" in show_panel
    assert show_panel.index("var container=getPanelContainer(panel)") < show_panel.index(
        "if(panel==='pos'){await loadStock();"
    )


def test_frontend_clears_password_after_login_and_logout():
    source = Path("api/module4_frontend/static/index.html").read_text(
        encoding="utf-8"
    )
    login = source[source.index("async function doLogin") : source.index("function doLogout")]
    logout = source[source.index("function doLogout") : source.index("var _panelContainers")]

    password_clear = "document.getElementById('password').value=''"
    assert password_clear in login
    assert password_clear in logout
    assert "document.getElementById('error-msg').textContent=''" in logout


def test_frontend_staff_view_does_not_load_management_data_or_controls():
    source = Path("api/module4_frontend/static/index.html").read_text(
        encoding="utf-8"
    )
    attendance = source[
        source.index("function renderAttendance") : source.index(
            "async function loadInvWastageRates"
        )
    ]

    assert '#dashboard[data-role="staff"] .manager-only' in source
    assert "if(role!=='manager')" in attendance
    assert "punchStaffAttendance()" in attendance
    assert "manager-only" in source[source.index("Fresh Batch Inflow") :]
    assert "manager-only" in source[source.index("function loadRecentOrders") :]


def test_frontend_protected_reads_send_authentication():
    source = Path("api/module4_frontend/static/index.html").read_text(
        encoding="utf-8"
    )

    assert "api('/s2/accuracy')" in source
    assert "api('/s4/revenue/daily')" in source
    assert (
        "fetch(API+'/s4/products',{headers:{'Authorization':'Bearer '+"
        "(typeof token==='undefined'?'':token)}})" in source
    )
    assert "api('/s1/batch_inventory',{force:true})" in source
    assert "fetch(API+'/s2/accuracy')" not in source
    assert "fetch(API+'/s1/batch_inventory')" not in source


def test_frontend_refreshes_stock_when_pos_is_opened():
    source = Path("api/module4_frontend/static/index.html").read_text(
        encoding="utf-8"
    )
    show_panel = source[
        source.index("async function showPanel") : source.index("var _apiCache")
    ]
    load_stock_start = source.index("function loadStock(){")
    load_stock = source[
        load_stock_start : source.index("function changeQty", load_stock_start)
    ]

    assert "if(panel==='pos'){await loadStock();" in show_panel
    assert "return api('/s1/batch_inventory',{force:true})" in load_stock
    assert "api('/s1/batch_inventory',{force:true})" in load_stock
