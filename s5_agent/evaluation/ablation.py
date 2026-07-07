from __future__ import annotations

from statistics import mean

from s5_agent.evaluation.metrics import (
    conflict_rate,
    evaluate_response,
    mean_agent_confidence,
    verification_pass_rate,
)


def _average_metric(rows: list[dict], metric_name: str) -> float:
    if not rows:
        return 0.0
    return round(mean(float(row.get(metric_name, 0.0)) for row in rows), 4)


def evaluate_variant_outputs(variant_name: str, responses: list[dict]) -> dict:
    response_metrics = [evaluate_response(response) for response in responses]
    return {
        "variant": variant_name,
        "num_cases": len(responses),
        "evidence_coverage": _average_metric(response_metrics, "evidence_coverage"),
        "unsupported_recommendation_rate": _average_metric(
            response_metrics,
            "unsupported_recommendation_rate",
        ),
        "verification_pass_rate": verification_pass_rate(responses),
        "conflict_rate": conflict_rate(responses),
        "mean_agent_confidence": mean_agent_confidence(responses),
        "decision_quality_score": _average_metric(
            response_metrics,
            "decision_quality_score",
        ),
    }


def compare_ablation_variants(
    variant_outputs: dict[str, list[dict]],
    baseline_name: str,
) -> list[dict]:
    rows = [
        evaluate_variant_outputs(variant_name, responses)
        for variant_name, responses in variant_outputs.items()
    ]
    baseline = next(
        (row for row in rows if row["variant"] == baseline_name),
        None,
    )
    baseline_score = baseline["decision_quality_score"] if baseline else 0.0
    for row in rows:
        row["decision_quality_delta"] = round(
            row["decision_quality_score"] - baseline_score,
            4,
        )
    return rows
