import asyncio
import csv
import json

from s5_agent.agents.inventory import InventoryAgent
from s5_agent.evaluation.experiment_runner import (
    export_experiment_rows,
    load_scenarios,
    run_experiment,
    summarize_experiment_rows,
)


def test_load_scenarios_returns_reproducible_cases():
    scenarios = load_scenarios()
    scenario_ids = {scenario["id"] for scenario in scenarios}

    assert {"inventory_shortage", "profit_low_margin", "production_overbuffer"} <= scenario_ids
    assert all("template" in scenario for scenario in scenarios)
    assert all("raw_inputs" in scenario for scenario in scenarios)


def test_run_experiment_returns_metric_rows(monkeypatch):
    monkeypatch.setattr(InventoryAgent, "_query_db_freshness", lambda self, product_filter=None: None)

    scenarios = [
        scenario
        for scenario in load_scenarios()
        if scenario["id"] in {"inventory_shortage", "profit_low_margin", "production_overbuffer"}
    ]
    rows = asyncio.run(run_experiment(scenarios))

    assert [row["scenario_id"] for row in rows] == [
        "inventory_shortage",
        "production_overbuffer",
        "profit_low_margin",
    ]
    assert all(row["variant"] == "proposed" for row in rows)
    assert all(row["template"] for row in rows)
    assert all("decision_quality_score" in row for row in rows)
    assert all("evidence_coverage" in row for row in rows)


def test_run_experiment_returns_rule_only_and_proposed_rows(monkeypatch):
    monkeypatch.setattr(InventoryAgent, "_query_db_freshness", lambda self, product_filter=None: None)

    scenarios = [
        scenario
        for scenario in load_scenarios()
        if scenario["id"] in {"inventory_shortage", "profit_low_margin", "production_overbuffer"}
    ]
    rows = asyncio.run(run_experiment(scenarios, variants=("rule_only", "proposed")))

    assert len(rows) == 6
    assert [row["variant"] for row in rows] == [
        "rule_only",
        "proposed",
        "rule_only",
        "proposed",
        "rule_only",
        "proposed",
    ]
    assert all("decision_quality_score" in row for row in rows)
    assert all(row["decision_quality_score"] >= 0 for row in rows)


def test_summarize_experiment_rows_reports_baseline_delta():
    rows = [
        {
            "scenario_id": "inventory_shortage",
            "variant": "rule_only",
            "decision_quality_score": 0.3,
            "evidence_coverage": 0.0,
            "unsupported_recommendation_rate": 1.0,
            "verification_passed": 0.0,
            "conflict_present": 0.0,
            "mean_agent_confidence": 0.4,
        },
        {
            "scenario_id": "inventory_shortage",
            "variant": "proposed",
            "decision_quality_score": 0.9,
            "evidence_coverage": 1.0,
            "unsupported_recommendation_rate": 0.0,
            "verification_passed": 1.0,
            "conflict_present": 0.0,
            "mean_agent_confidence": 0.8,
        },
    ]

    summary = summarize_experiment_rows(rows, baseline_name="rule_only")

    assert [row["variant"] for row in summary] == ["rule_only", "proposed"]
    assert summary[0]["decision_quality_delta"] == 0.0
    assert summary[1]["decision_quality_delta"] == 0.6


def test_export_experiment_rows_writes_json_and_csv(tmp_path):
    rows = [
        {
            "scenario_id": "inventory_shortage",
            "variant": "proposed",
            "template": "inventory_diagnosis",
            "decision_quality_score": 0.8,
            "evidence_coverage": 1.0,
        }
    ]
    json_path = tmp_path / "results.json"
    csv_path = tmp_path / "results.csv"

    export_experiment_rows(rows, json_path=json_path, csv_path=csv_path)

    assert json.loads(json_path.read_text(encoding="utf-8")) == rows
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        csv_rows = list(csv.DictReader(csv_file))
    assert csv_rows == [
        {
            "scenario_id": "inventory_shortage",
            "variant": "proposed",
            "template": "inventory_diagnosis",
            "decision_quality_score": "0.8",
            "evidence_coverage": "1.0",
        }
    ]
