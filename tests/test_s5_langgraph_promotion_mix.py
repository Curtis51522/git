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


def test_promo_agent_fetch_preserves_order_basket_metrics(monkeypatch):
    monkeypatch.setattr(
        "s5_agent.agents.promo.fetch_dashboard_json",
        lambda *_args, **_kwargs: {
            "data": {
                "today_revenue": 2637.5,
                "today_discount": 92.1,
                "items_per_order": 4.78,
                "items_per_order_change": -27.8,
                "revenue_per_item": 12.27,
                "revenue_per_item_change": 1.0,
            }
        },
    )
    agent = PromoAgent("PromoAgent")

    result = asyncio.run(agent.fetch({"date": "2026-07-15"}))

    assert result["data"]["items_per_order"] == 4.78
    assert result["data"]["items_per_order_change"] == -27.8
    assert result["data"]["revenue_per_item"] == 12.27
    assert result["data"]["revenue_per_item_change"] == 1.0


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


def test_product_mix_agent_uses_total_bread_revenue_for_concentration():
    agent = ProductMixAgent("ProductMixAgent")
    raw = {
        "bread_ranking": [
            {"name": "macaron", "qty": 59, "revenue": 588, "profit": 470},
            {"name": "mantequilla", "qty": 37, "revenue": 333, "profit": 280},
            {"name": "bread_coconut", "qty": 32, "revenue": 314, "profit": 260},
            {"name": "melon_bread", "qty": 18, "revenue": 150, "profit": 110},
            {"name": "croissant", "qty": 12, "revenue": 85, "profit": 60},
        ],
        "beverage_ranking": [],
        "category": {"Bread": 2254.8, "Beverages": 0},
        "sold_bread_sku_count": 29,
        "total_bread_sku": 30,
        "week_avg": {"bread_avg": []},
    }

    output = agent.analyze(raw, {"date": "2026-06-30"})

    assert "Top 3 breads = 55% of revenue" in output.opinion
    assert "84%" not in output.opinion
    assert output.attribution["root_cause"] == "high_concentration"
    assert output.attribution["deviation"] == 5


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


def test_promotion_mix_uses_real_sold_sku_count_instead_of_top_five_length():
    response = asyncio.run(
        run_s5_graph(
            "promotion_mix_analysis",
            S5Request(
                query="promotion_mix",
                module="promotion_mix",
                params={"date": "2026-07-14"},
            ),
            raw_inputs={
                "promotion_signal": {
                    "today_revenue": 1000,
                    "today_discount": 0,
                    "expired_cost": 0,
                },
                "product_mix": {
                    "bread_ranking": [
                        {"name": f"bread_{index}", "qty": 10, "revenue": 50, "profit": 30}
                        for index in range(5)
                    ],
                    "beverage_ranking": [],
                    "category": {"Bread": 1000, "Beverages": 0},
                    "sold_bread_sku_count": 24,
                    "total_bread_sku": 30,
                    "week_avg": {"bread_avg": []},
                },
            },
        )
    )

    assert "24 of 30 tracked bread SKUs recorded sales" in response.summary
    assert "sold across" not in response.summary
    assert "Bread product mix is distributed" in response.summary
    assert "Bread revenue is concentrated" not in response.summary
    assert "reduce dependence on a small group" not in response.summary
    product_mix = next(
        output
        for output in response.agent_outputs
        if output.agent_name == "PromotionProductMixAgent"
    )
    assert product_mix.metrics["sold_bread_sku_count"] == 24


def test_promotion_mix_targets_high_margin_concentrated_unsold_product():
    response = asyncio.run(
        run_s5_graph(
            "promotion_mix_analysis",
            S5Request(
                query="promotion_mix",
                module="promotion_mix",
                params={"date": "2026-07-14"},
            ),
            raw_inputs={
                "promotion_signal": {
                    "today_revenue": 1000,
                    "today_discount": 10,
                    "expired_cost": 100,
                    "expired_products": [
                        {
                            "name": "Croissant",
                            "expired_qty": 8,
                            "expired_cost": 60,
                            "sold_qty": 32,
                            "margin_pct": 62.5,
                            "sell_through_pct": 80.0,
                            "loss_share_pct": 60.0,
                        }
                    ],
                },
                "product_mix": {
                    "bread_ranking": [
                        {"name": "croissant", "qty": 32, "revenue": 384, "profit": 240}
                    ],
                    "beverage_ranking": [],
                    "category": {"Bread": 1000, "Beverages": 0},
                    "sold_bread_sku_count": 12,
                    "total_bread_sku": 30,
                    "week_avg": {"bread_avg": []},
                },
            },
        )
    )

    recommendation_ids = {item.id for item in response.recommendations}
    assert "promotion_targeted_closing_bundle" in recommendation_ids
    assert "promotion_reduce_unsold_bake" not in recommendation_ids
    assert "10.0% of revenue" in response.summary
    assert "Croissant" in response.summary
    node_ids = {node.id for node in response.evidence_graph.nodes}
    assert "metric:expired_cost_revenue_pct" in node_ids
    assert "metric:expired_products" in node_ids


def test_promotion_mix_reduces_bake_when_unsold_product_lacks_promotion_room():
    response = asyncio.run(
        run_s5_graph(
            "promotion_mix_analysis",
            S5Request(
                query="promotion_mix",
                module="promotion_mix",
                params={"date": "2026-07-14"},
            ),
            raw_inputs={
                "promotion_signal": {
                    "today_revenue": 1000,
                    "today_discount": 0,
                    "expired_cost": 120,
                    "expired_products": [
                        {
                            "name": "Baguette",
                            "expired_qty": 20,
                            "expired_cost": 72,
                            "sold_qty": 5,
                            "margin_pct": 18.0,
                            "sell_through_pct": 20.0,
                            "loss_share_pct": 60.0,
                        }
                    ],
                },
                "product_mix": {
                    "bread_ranking": [
                        {"name": "baguette", "qty": 5, "revenue": 60, "profit": 11}
                    ],
                    "beverage_ranking": [],
                    "category": {"Bread": 1000, "Beverages": 0},
                    "sold_bread_sku_count": 8,
                    "total_bread_sku": 30,
                    "week_avg": {"bread_avg": []},
                },
            },
        )
    )

    recommendation_ids = {item.id for item in response.recommendations}
    assert "promotion_reduce_unsold_bake" in recommendation_ids
    assert "promotion_targeted_closing_bundle" not in recommendation_ids
    reduction = next(
        item
        for item in response.recommendations
        if item.id == "promotion_reduce_unsold_bake"
    )
    assert "Baguette" in reduction.action
    assert "discount" in reduction.rationale.lower()


def test_promotion_mix_explains_high_margin_low_sell_through_without_contradiction():
    response = asyncio.run(
        run_s5_graph(
            "promotion_mix_analysis",
            S5Request(
                query="promotion_mix",
                module="promotion_mix",
                params={"date": "2026-07-14"},
            ),
            raw_inputs={
                "promotion_signal": {
                    "today_revenue": 4180,
                    "today_discount": 62.7,
                    "expired_cost": 698.4,
                    "expired_products": [
                        {
                            "name": "Cream Horn",
                            "expired_qty": 16,
                            "expired_cost": 73.6,
                            "sold_qty": 4,
                            "margin_pct": 85.2,
                            "sell_through_pct": 20.0,
                            "loss_share_pct": 10.5,
                        }
                    ],
                },
                "product_mix": {
                    "bread_ranking": [
                        {"name": "macaron", "qty": 22, "revenue": 220, "profit": 176}
                    ],
                    "beverage_ranking": [
                        {"name": "cold_brew", "qty": 10, "revenue": 180, "profit": 120}
                    ],
                    "category": {"Bread": 3025, "Beverages": 1148},
                    "sold_bread_sku_count": 30,
                    "total_bread_sku": 30,
                    "week_avg": {"bread_avg": []},
                },
            },
        )
    )

    reduction = next(
        item
        for item in response.recommendations
        if item.id == "promotion_reduce_unsold_bake"
    )
    assert response.summary.count("\n\n") == 2
    summary_paragraphs = response.summary.split("\n\n")
    assert "Discount exposure" in summary_paragraphs[0]
    assert "Bread generated" in summary_paragraphs[1]
    assert "The practical decision" in summary_paragraphs[2]
    assert "limited margin room" not in response.summary
    assert "85.2% sold-product margin" in reduction.rationale
    assert "20.0% sell-through" in reduction.rationale
    assert "does not show that discounting would recover more value" in reduction.rationale
    assert "does not show enough margin" not in reduction.rationale


def test_promotion_mix_turns_basket_decline_into_targeted_pairing_test():
    response = asyncio.run(
        run_s5_graph(
            "promotion_mix_analysis",
            S5Request(
                query="promotion_mix",
                module="promotion_mix",
                params={"date": "2026-07-15"},
            ),
            raw_inputs={
                "promotion_signal": {
                    "today_revenue": 2637.5,
                    "today_discount": 92.1,
                    "discount_rate": 0.0349,
                    "expired_cost": 8.66,
                    "items_per_order": 4.78,
                    "items_per_order_change": -27.8,
                    "revenue_per_item": 12.27,
                    "revenue_per_item_change": 1.0,
                },
                "product_mix": {
                    "bread_ranking": [
                        {"name": "macaron", "qty": 12, "revenue": 168, "profit": 140},
                        {"name": "chocolate_cake", "qty": 6, "revenue": 127.6, "profit": 95},
                        {"name": "bread_coconut", "qty": 13, "revenue": 111.6, "profit": 85},
                    ],
                    "beverage_ranking": [
                        {"name": "latte", "qty": 6, "revenue": 108, "profit": 75}
                    ],
                    "category": {"Bread": 1833.4, "Beverages": 799.0},
                    "sold_bread_sku_count": 30,
                    "total_bread_sku": 30,
                    "week_avg": {"bread_avg": []},
                },
            },
        )
    )

    promo = next(
        output
        for output in response.agent_outputs
        if output.agent_name == "PromotionSignalAgent"
    )
    recommendation_ids = {item.id for item in response.recommendations}
    basket_recommendation = next(
        item
        for item in response.recommendations
        if item.id == "promotion_basket_bundle_test"
    )
    no_broad_discount = next(
        item
        for item in response.recommendations
        if item.id == "promotion_no_broad_discount"
    )

    assert "basket_size_weakness" in promo.risks
    assert "Items per order fell 27.8% to 4.78" in response.summary
    assert "revenue per item remained stable at \u00a512.27" in response.summary
    assert "promotion_basket_bundle_test" in recommendation_ids
    assert "promotion_no_broad_discount" in recommendation_ids
    assert "bread-and-beverage bundle" in basket_recommendation.action
    assert "2-3 completed trading days" in basket_recommendation.action
    assert "broad discount" in basket_recommendation.action
    assert "Keep broad discounts off the table" in no_broad_discount.action
    assert "unless traffic weakness persists" not in no_broad_discount.action
    assert basket_recommendation.evidence_ids == [
        "items_per_order",
        "items_per_order_change_pct",
        "revenue_per_item",
        "revenue_per_item_change_pct",
        "beverage_revenue_share_pct",
    ]
    node_ids = {node.id for node in response.evidence_graph.nodes}
    assert "metric:items_per_order" in node_ids
    assert "metric:items_per_order_change_pct" in node_ids
    assert "metric:revenue_per_item" in node_ids
    assert "metric:revenue_per_item_change_pct" in node_ids
