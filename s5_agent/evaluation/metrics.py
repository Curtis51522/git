from statistics import mean
from typing import Any


def _as_mapping(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _recommendations(response: dict) -> list[dict]:
    return [_as_mapping(item) for item in response.get("recommendations", [])]


def _verification_report(response: dict) -> dict:
    return _as_mapping(response.get("verification_report", {}))


def evidence_coverage(recommendations: list[dict]) -> float:
    if not recommendations:
        return 1.0

    supported = sum(1 for rec in recommendations if rec.get("evidence_ids"))
    return round(supported / len(recommendations), 4)


def unsupported_recommendation_rate(recommendations: list[dict]) -> float:
    if not recommendations:
        return 0.0

    unsupported = sum(1 for rec in recommendations if not rec.get("evidence_ids"))
    return round(unsupported / len(recommendations), 4)


def verification_pass_rate(responses: list[dict]) -> float:
    if not responses:
        return 0.0

    passed = 0
    for response in responses:
        report = _verification_report(_as_mapping(response))
        if report.get("passed") is True:
            passed += 1
    return round(passed / len(responses), 4)


def conflict_rate(responses: list[dict]) -> float:
    if not responses:
        return 0.0

    conflicted = 0
    for response in responses:
        report = _verification_report(_as_mapping(response))
        if report.get("conflicting_claims"):
            conflicted += 1
    return round(conflicted / len(responses), 4)


def mean_agent_confidence(responses: list[dict]) -> float:
    confidences = []
    for response in responses:
        response_map = _as_mapping(response)
        for output in response_map.get("agent_outputs", []):
            output_map = _as_mapping(output)
            confidence = output_map.get("confidence")
            if confidence is not None:
                confidences.append(float(confidence))
    if not confidences:
        return 0.0
    return round(mean(confidences), 4)


def evaluate_response(response: dict) -> dict:
    response_map = _as_mapping(response)
    recommendations = _recommendations(response_map)
    report = _verification_report(response_map)
    coverage = evidence_coverage(recommendations)
    unsupported_rate = unsupported_recommendation_rate(recommendations)
    verification_passed = 1.0 if report.get("passed") is True else 0.0
    conflict_present = 1.0 if report.get("conflicting_claims") else 0.0
    data_quality_warnings = report.get("data_quality_warnings", []) or []
    confidence = mean_agent_confidence([response_map])
    decision_quality_score = round(
        (
            coverage
            + verification_passed
            + confidence
            + (1.0 - unsupported_rate)
            + (1.0 - conflict_present)
        )
        / 5,
        4,
    )

    return {
        "evidence_coverage": coverage,
        "unsupported_recommendation_rate": unsupported_rate,
        "verification_passed": verification_passed,
        "conflict_present": conflict_present,
        "data_quality_warning_count": len(data_quality_warnings),
        "mean_agent_confidence": confidence,
        "decision_quality_score": decision_quality_score,
    }
