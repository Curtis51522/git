import asyncio

from fastapi.testclient import TestClient
from s5_agent.agents.inventory import InventoryAgent
from s5_agent.graph.registry import module_to_template, supported_templates
from s5_agent.graph.runner import run_s5_graph
from s5_agent.graph.state import S5Request
from s5_agent.graph.template_loader import load_template
from s5_agent.schemas.response import S5AnalysisResponse
from s5_agent.schemas.verification import VerificationReport
from s5_agent.server import app


def test_inventory_module_maps_to_inventory_diagnosis():
    assert module_to_template("inventory") == "inventory_diagnosis"


def test_promotion_mix_module_uses_langgraph_route():
    from s5_agent.server import LANGGRAPH_MODULES
    from s5_agent.graph.registry import module_to_template

    assert "promotion_mix" in LANGGRAPH_MODULES
    assert module_to_template("promotion_mix") == "promotion_mix_analysis"


def test_supported_templates_match_current_langgraph_modules():
    assert supported_templates() == [
        "inventory_diagnosis",
        "production_advice",
        "profit_root_cause",
        "promotion_mix_analysis",
        "wastage_root_cause",
    ]


def test_registry_rejects_non_langgraph_modules():
    try:
        module_to_template("schedule")
    except ValueError as exc:
        assert "Unsupported S5 module" in str(exc)
    else:
        raise AssertionError("schedule must not map to a legacy S5 template")


def test_load_inventory_diagnosis_template_returns_id_and_agents():
    template = load_template("inventory_diagnosis")

    assert template["id"] == "inventory_diagnosis"
    assert template["agents"] == ["InventoryAgent"]


def test_load_declared_non_inventory_templates():
    profit_template = load_template("profit_root_cause")
    production_template = load_template("production_advice")

    assert profit_template["id"] == "profit_root_cause"
    assert production_template["id"] == "production_advice"
    assert "summary" in profit_template["outputs"]
    assert "recommendations" in production_template["outputs"]


def test_run_s5_graph_accepts_declared_templates(monkeypatch):
    monkeypatch.setattr(InventoryAgent, "_query_db_freshness", lambda self, product_filter=None: None)
    request = S5Request(
        query="Analyze template compatibility",
        module="revenue",
        params={"product": "all"},
    )
    raw_inputs = {
        "inventory": {
            "inventory": [
                {
                    "product_name": "croissant",
                    "total_quantity": 12,
                    "batches": 1,
                    "selling_price": 5.9,
                }
            ]
        }
    }

    response = asyncio.run(
        run_s5_graph(
            "profit_root_cause",
            request,
            raw_inputs=raw_inputs,
        )
    )

    assert response.summary
    assert response.metadata["template"] == "profit_root_cause"
    assert response.verification_report.passed is True


def test_inventory_module_uses_langgraph_runner(monkeypatch):
    async def fake_run_s5_graph(template_id, request, raw_inputs=None):
        assert template_id == "inventory_diagnosis"
        assert request.query == "inventory"
        assert request.module == "inventory"
        assert request.params == {
            "date": "2026-07-07",
            "module": "inventory",
            "product": "all",
        }
        assert request.lang == "en"
        assert request.force_refresh is True
        assert raw_inputs is None
        return S5AnalysisResponse(
            summary="Inventory graph summary",
            verification_report=VerificationReport(passed=True),
            metadata={"template": template_id},
        )

    monkeypatch.setattr("s5_agent.server.run_s5_graph", fake_run_s5_graph)

    client = TestClient(app)
    response = client.post(
        "/analyze/module",
        json={
            "module": "inventory",
            "date": "2026-07-07",
            "lang": "en",
            "force_refresh": True,
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["summary"] == "Inventory graph summary"
    assert body["metadata"]["template"] == "inventory_diagnosis"
    assert "verification_report" in body


def test_analyze_module_routes_promotion_mix_to_langgraph(monkeypatch):
    captured = {}

    async def fake_run_s5_graph(template_id, request, raw_inputs=None):
        captured["template_id"] = template_id
        captured["module"] = request.module
        captured["raw_inputs"] = raw_inputs
        return S5AnalysisResponse(
            summary="Promotion mix response",
            verification_report=VerificationReport(passed=True),
            metadata={"template": template_id},
        )

    monkeypatch.setattr("s5_agent.server.run_s5_graph", fake_run_s5_graph)

    client = TestClient(app)
    response = client.post(
        "/analyze/module",
        json={
            "module": "promotion_mix",
            "date": "2026-06-30",
            "lang": "en",
            "force_refresh": True,
            "params": {},
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert captured == {
        "template_id": "promotion_mix_analysis",
        "module": "promotion_mix",
        "raw_inputs": None,
    }
    assert body["summary"] == "Promotion mix response"
    assert body["metadata"]["template"] == "promotion_mix_analysis"


def test_declared_modules_use_langgraph_runner(monkeypatch):
    calls = []

    async def fake_run_s5_graph(template_id, request, raw_inputs=None):
        calls.append((template_id, request, raw_inputs))
        return S5AnalysisResponse(
            summary=f"{template_id} graph summary",
            verification_report=VerificationReport(passed=True),
            metadata={"template": template_id},
        )

    monkeypatch.setattr("s5_agent.server.run_s5_graph", fake_run_s5_graph)

    client = TestClient(app)
    revenue_response = client.post(
        "/analyze/module",
        json={
            "module": "revenue",
            "date": "2026-07-07",
            "params": {"product": "croissant"},
            "lang": "zh-cn",
            "force_refresh": True,
        },
    )
    forecast_response = client.post(
        "/analyze/module",
        json={
            "module": "forecast",
            "date": "2026-07-08",
            "params": {"product": "bagel"},
            "lang": "en",
        },
    )

    assert revenue_response.status_code == 200
    assert forecast_response.status_code == 200
    assert revenue_response.json()["metadata"]["template"] == "profit_root_cause"
    assert forecast_response.json()["metadata"]["template"] == "production_advice"
    assert [call[0] for call in calls] == ["profit_root_cause", "production_advice"]
    assert calls[0][1].query == "revenue"
    assert calls[0][1].module == "revenue"
    assert calls[0][1].params == {
        "date": "2026-07-07",
        "module": "revenue",
        "product": "croissant",
    }
    assert calls[0][1].lang == "zh"
    assert calls[0][1].force_refresh is True
    assert calls[1][1].query == "forecast"
    assert calls[1][1].module == "forecast"
    assert calls[1][1].params == {
        "date": "2026-07-08",
        "module": "forecast",
        "product": "bagel",
    }
    assert calls[1][1].lang == "en"
    assert calls[1][2] is None


def test_unsupported_module_does_not_fall_back_to_legacy_dag(monkeypatch):
    async def fake_run_s5_graph(template_id, request, raw_inputs=None):
        raise AssertionError("unsupported modules must not call LangGraph")

    monkeypatch.setattr("s5_agent.server.run_s5_graph", fake_run_s5_graph)

    client = TestClient(app)
    response = client.post(
        "/analyze/module",
        json={
            "module": "schedule",
            "date": "2026-07-07",
            "lang": "en",
            "force_refresh": True,
        },
    )

    assert response.status_code == 400
    assert "Unsupported S5 module" in response.json()["detail"]


def test_legacy_analyze_and_template_routes_are_not_exposed():
    client = TestClient(app)

    analyze_response = client.post(
        "/analyze",
        json={"query": "legacy route should not be exposed", "params": {"date": "2026-07-07"}},
    )
    templates_response = client.get("/templates")

    assert analyze_response.status_code == 404
    assert templates_response.status_code == 404
