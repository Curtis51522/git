import asyncio
from types import SimpleNamespace

import pytest

from s5_agent.agents.product_mix import ProductMixAgent
from s5_agent.agents.promo import PromoAgent
from s5_agent.graph.builder import _promotion_mix_recommendations
from s5_agent.graph.runner import run_s5_graph
from s5_agent.graph.state import S5Request


def test_promo_agent_structures_discount_metrics():
    agent = PromoAgent("PromoAgent")
    raw = {
        "today_revenue": 1000,
        "today_discount": 80,
        "discount_rate": 0.08,
    }

    output = agent.analyze(raw, {"date": "2026-06-30"})

    assert output.agent == "PromoAgent"
    assert output.confidence >= 0.6
    assert output.attribution["metric"] == "discount_rate"
    assert output.attribution["deviation"] == pytest.approx(8.0)


def test_product_mix_agent_detects_top3_concentration():
    agent = ProductMixAgent("ProductMixAgent")
    raw = {
        "bread_ranking": [
            {"name": "macaron", "qty": 30, "revenue": 300, "profit": 240},
            {"name": "baguette", "qty": 20, "revenue": 200, "profit": 140},
            {"name": "croissant", "qty": 10, "revenue": 100, "profit": 70},
            {"name": "donut", "qty": 10, "revenue": 80, "profit": 48},
        ],
        "beverage_ranking": [
            {"name": "latte", "qty": 15, "revenue": 270, "profit": 180},
        ],
        "category": {"Bread": 680, "Beverages": 270},
        "total_bread_sku": 20,
        "week_avg": {"bread_avg": []},
    }

    output = agent.analyze(raw, {"date": "2026-06-30"})

    assert output.agent == "ProductMixAgent"
    assert output.attribution["metric"] == "product_mix"
    assert output.confidence >= 0.7
    assert "Top 3" in output.opinion


def test_promotion_mix_graph_returns_verified_response():
    response = asyncio.run(
        run_s5_graph(
            "promotion_mix_analysis",
            S5Request(
                query="promotion_mix",
                module="promotion_mix",
                params={"date": "2026-06-30", "module": "promotion_mix"},
                lang="en",
                force_refresh=True,
            ),
            raw_inputs={
                "promotion_signal": {"today_revenue": 1000, "today_discount": 0, "discount_rate": 0},
                "product_mix": {
                    "bread_ranking": [
                        {"name": "macaron", "qty": 30, "revenue": 300, "profit": 240},
                        {"name": "baguette", "qty": 20, "revenue": 200, "profit": 140},
                        {"name": "croissant", "qty": 10, "revenue": 100, "profit": 70},
                    ],
                    "beverage_ranking": [{"name": "latte", "qty": 15, "revenue": 270, "profit": 180}],
                    "category": {"Bread": 600, "Beverages": 270},
                    "total_bread_sku": 20,
                    "week_avg": {"bread_avg": []},
                },
            },
        )
    )

    assert response.verification_report.passed is True
    assert response.verification_report.unsupported_recommendations == []
    assert response.summary
    node_ids = {node.id for node in response.evidence_graph.nodes}
    recommendation_ids = {recommendation.id for recommendation in response.recommendations}
    assert "metric:discount_rate_pct" in node_ids
    assert "metric:top3_product_revenue_share_pct" in node_ids
    for recommendation_id in recommendation_ids:
        assert f"recommendation:{recommendation_id}" in node_ids
    assert any(
        output.agent_name in {"PromotionSignalAgent", "PromotionProductMixAgent", "PromotionDecisionAgent"}
        for output in response.agent_outputs
    )


def test_promotion_mix_recommends_no_broad_discount_when_stable():
    response = asyncio.run(
        run_s5_graph(
            "promotion_mix_analysis",
            S5Request(
                query="promotion_mix",
                module="promotion_mix",
                params={"date": "2026-06-30"},
                lang="en",
                force_refresh=True,
            ),
            raw_inputs={
                "promotion_signal": {"today_revenue": 2984, "today_discount": 0, "discount_rate": 0},
                "product_mix": {
                    "bread_ranking": [
                        {"name": "macaron", "qty": 59, "revenue": 590, "profit": 470},
                        {"name": "baguette", "qty": 20, "revenue": 350, "profit": 250},
                        {"name": "croissant", "qty": 18, "revenue": 304, "profit": 210},
                    ],
                    "beverage_ranking": [{"name": "latte", "qty": 15, "revenue": 270, "profit": 180}],
                    "category": {"Bread": 2324, "Beverages": 660},
                    "total_bread_sku": 20,
                    "week_avg": {"bread_avg": []},
                },
            },
        )
    )

    assert "broad" in response.summary.lower()
    assert any("broad discount" in rec.action.lower() for rec in response.recommendations)


def test_promotion_mix_flags_bread_concentration_with_natural_summary():
    response = asyncio.run(
        run_s5_graph(
            "promotion_mix_analysis",
            S5Request(
                query="promotion_mix",
                module="promotion_mix",
                params={"date": "2026-06-30"},
                lang="en",
                force_refresh=True,
            ),
            raw_inputs={
                "promotion_signal": {"today_revenue": 2984, "today_discount": 0, "discount_rate": 0},
                "product_mix": {
                    "bread_ranking": [
                        {"name": "macaron", "qty": 59, "revenue": 590, "profit": 470},
                        {"name": "baguette", "qty": 45, "revenue": 760, "profit": 520},
                        {"name": "croissant", "qty": 38, "revenue": 580, "profit": 390},
                        {"name": "melon_bread", "qty": 20, "revenue": 200, "profit": 130},
                    ],
                    "beverage_ranking": [{"name": "cold_brew", "qty": 8, "revenue": 658, "profit": 420}],
                    "category": {"Bread": 2326, "Beverages": 658},
                    "total_bread_sku": 30,
                    "week_avg": {"bread_avg": []},
                },
            },
        )
    )

    assert "|" not in response.summary
    assert "bread revenue is concentrated" in response.summary.lower()
    assert any("product_concentration" in output.risks for output in response.agent_outputs)
    bundle_recommendations = [
        rec for rec in response.recommendations if rec.id == "promotion_mid_tier_bundle"
    ]
    assert bundle_recommendations
    assert (
        bundle_recommendations[0].expected_impact
        == "Improves product-mix balance while protecting margin from broad discount erosion."
    )
    node_ids = {node.id for node in response.evidence_graph.nodes}
    assert "metric:top3_bread_revenue_share_pct" in node_ids


def test_promotion_mix_does_not_over_trigger_bundle_from_bread_only_share():
    response = asyncio.run(
        run_s5_graph(
            "promotion_mix_analysis",
            S5Request(
                query="promotion_mix",
                module="promotion_mix",
                params={"date": "2026-06-30"},
                lang="en",
                force_refresh=True,
            ),
            raw_inputs={
                "promotion_signal": {"today_revenue": 1600, "today_discount": 0, "discount_rate": 0},
                "product_mix": {
                    "bread_ranking": [
                        {"name": "macaron", "qty": 30, "revenue": 300, "profit": 240},
                        {"name": "baguette", "qty": 20, "revenue": 200, "profit": 140},
                        {"name": "croissant", "qty": 10, "revenue": 100, "profit": 70},
                    ],
                    "beverage_ranking": [
                        {"name": "latte", "qty": 80, "revenue": 800, "profit": 500},
                        {"name": "americano", "qty": 40, "revenue": 200, "profit": 120},
                    ],
                    "category": {"Bread": 600, "Beverages": 1000},
                    "total_bread_sku": 20,
                    "week_avg": {"bread_avg": []},
                },
            },
        )
    )

    recommendation_ids = {rec.id for rec in response.recommendations}
    node_values = {node.id: node.value for node in response.evidence_graph.nodes}
    assert node_values["metric:top3_product_revenue_share_pct"] == pytest.approx(37.5)
    assert "promotion_mid_tier_bundle" not in recommendation_ids


def test_promotion_mix_does_not_recommend_no_broad_discount_without_promo_evidence():
    recommendations = _promotion_mix_recommendations(
        {
            "promotion_product_mix": SimpleNamespace(
                claim="Top 3 products account for 38.0% of revenue.",
                metrics={"top3_product_revenue_share_pct": 38.0},
            )
        }
    )

    assert {rec.id for rec in recommendations}.isdisjoint({"promotion_no_broad_discount"})
