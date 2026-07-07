from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from s5_agent.evaluation.baselines.rule_only import rule_only_response
from s5_agent.evaluation.metrics import evaluate_response
from s5_agent.graph.runner import run_s5_graph
from s5_agent.graph.state import S5Request

SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"


def load_scenarios(directory: str | Path | None = None) -> list[dict[str, Any]]:
    scenario_dir = Path(directory) if directory is not None else SCENARIO_DIR
    scenarios = []
    for scenario_path in sorted(scenario_dir.glob("*.json")):
        with scenario_path.open("r", encoding="utf-8") as scenario_file:
            scenarios.append(json.load(scenario_file))
    return sorted(scenarios, key=lambda scenario: scenario["id"])


def _metric_row(
    scenario: dict[str, Any],
    variant: str,
    response: dict[str, Any],
) -> dict[str, Any]:
    metrics = evaluate_response(response)
    return {
        "scenario_id": scenario["id"],
        "variant": variant,
        "module": scenario.get("module", ""),
        "template": scenario["template"],
        **metrics,
    }


async def _run_proposed_response(scenario: dict[str, Any]) -> dict[str, Any]:
    template_id = scenario["template"]
    request = S5Request(
        query=scenario["id"],
        module=scenario.get("module", ""),
        params=scenario.get("params", {}),
        lang=scenario.get("lang", "en"),
        force_refresh=True,
    )
    response = await run_s5_graph(
        template_id,
        request,
        raw_inputs=scenario.get("raw_inputs", {}),
    )
    return response.model_dump()


async def run_experiment(
    scenarios: list[dict[str, Any]] | None = None,
    variants: tuple[str, ...] = ("proposed",),
) -> list[dict[str, Any]]:
    selected_scenarios = scenarios if scenarios is not None else load_scenarios()
    rows = []
    for scenario in sorted(selected_scenarios, key=lambda item: item["id"]):
        for variant in variants:
            if variant == "rule_only":
                response = rule_only_response(scenario)
            elif variant == "proposed":
                response = await _run_proposed_response(scenario)
            else:
                raise ValueError(f"Unsupported experiment variant: {variant}")
            rows.append(_metric_row(scenario, variant, response))
    return rows


def _average(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(mean(float(row.get(key, 0.0)) for row in rows), 4)


def summarize_experiment_rows(
    rows: list[dict[str, Any]],
    baseline_name: str,
) -> list[dict[str, Any]]:
    variant_order = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        variant = row["variant"]
        if variant not in grouped:
            grouped[variant] = []
            variant_order.append(variant)
        grouped[variant].append(row)

    summary_rows = []
    for variant in variant_order:
        variant_rows = grouped[variant]
        summary_rows.append(
            {
                "variant": variant,
                "num_cases": len(variant_rows),
                "evidence_coverage": _average(variant_rows, "evidence_coverage"),
                "unsupported_recommendation_rate": _average(
                    variant_rows,
                    "unsupported_recommendation_rate",
                ),
                "verification_pass_rate": _average(variant_rows, "verification_passed"),
                "conflict_rate": _average(variant_rows, "conflict_present"),
                "mean_agent_confidence": _average(variant_rows, "mean_agent_confidence"),
                "decision_quality_score": _average(variant_rows, "decision_quality_score"),
            }
        )

    baseline = next(
        (row for row in summary_rows if row["variant"] == baseline_name),
        None,
    )
    baseline_score = baseline["decision_quality_score"] if baseline else 0.0
    for row in summary_rows:
        row["decision_quality_delta"] = round(
            row["decision_quality_score"] - baseline_score,
            4,
        )
    return summary_rows


def export_experiment_rows(
    rows: list[dict[str, Any]],
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    json_output = Path(json_path)
    csv_output = Path(csv_path)
    json_output.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with csv_output.open("w", encoding="utf-8", newline="") as csv_file:
        if not rows:
            return
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
