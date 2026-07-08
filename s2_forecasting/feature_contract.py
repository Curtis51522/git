"""Canonical S2 forecast-time feature contract.

This module owns the feature order used by S2 preprocessing, model training,
runtime inference, and API explanations. Keep this contract in sync with the
deployed Q50 conformal forecast model.

Business event fields are intentionally separated from the deployed daily
forecast input contract. They remain available as reserved scenario context for
new-product launches, competitor activity, promotion analysis, S5 reasoning,
and weekly event-aware experiments.
"""

FORECAST_FEATURES = [
    "product_id",
    "category",
    "daily_tickets",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "lag_1",
    "lag_7_avg",
    "lag_30_avg",
    "roll_std_7",
    "roll_std_14",
    "trend_7",
    "is_day1",
    "is_top3",
    "discount_pct",
    "is_member_day",
    "is_rainy",
    "temp_mean",
    "temp_range",
    "is_cold_day",
    "is_hot_day",
    "large_ratio",
    "cold_ratio",
    "sweetness_avg",
    "ice_avg",
    "temp_hot_ratio",
]

FEATURE_GROUPS = {
    "product_identity": ["product_id", "category"],
    "calendar": ["day_of_week", "month", "is_weekend", "is_holiday"],
    "lagged_history": [
        "lag_1",
        "lag_7_avg",
        "lag_30_avg",
        "roll_std_7",
        "roll_std_14",
        "trend_7",
    ],
    "inventory_context": ["is_day1", "is_top3"],
    "planned_operations": ["discount_pct", "is_member_day"],
    "weather": ["is_rainy", "temp_mean", "temp_range", "is_cold_day", "is_hot_day"],
    "traffic_proxy": ["daily_tickets"],
    "beverage_behavior_proxy": [
        "large_ratio",
        "cold_ratio",
        "sweetness_avg",
        "ice_avg",
        "temp_hot_ratio",
    ],
}

FEATURE_AVAILABILITY = {
    "product_identity": "known_before_forecast",
    "calendar": "known_before_forecast",
    "lagged_history": "past_observed_only",
    "inventory_context": "provided_by_inventory_before_forecast",
    "planned_operations": "planned_or_scenario_input",
    "weather": "forecast_or_historical_fallback",
    "traffic_proxy": "lagged_or_frozen_proxy",
    "beverage_behavior_proxy": "historical_proxy",
}

RESERVED_SCENARIO_FEATURES = {
    "is_new_product": {
        "label": "New product launch window",
        "description": (
            "Reserved planned-event flag for a product launch or launch promotion "
            "period. It is not part of the deployed 27-feature Q50 model."
        ),
        "availability": "reserved_scenario_input",
        "requires_model_retraining": True,
    },
    "is_competitor": {
        "label": "Nearby competitor opening or promotion response window",
        "description": (
            "Reserved planned-event flag for a nearby competitor opening, competitor "
            "promotion, or store response campaign period. It is not part of the "
            "deployed 27-feature Q50 model."
        ),
        "availability": "reserved_scenario_input",
        "requires_model_retraining": True,
    },
}

WEEKLY_RESERVED_SCENARIO_FEATURES = {
    "is_new_product_w": {
        "source_feature": "is_new_product",
        "label": "Weekly new product launch window",
        "description": (
            "Weekly aggregation of the reserved new-product launch context. "
            "It is kept for event-aware weekly experiments and is not part of "
            "the deployed daily 27-feature forecast contract."
        ),
    },
    "is_competitor_w": {
        "source_feature": "is_competitor",
        "label": "Weekly competitor activity window",
        "description": (
            "Weekly aggregation of the reserved competitor activity context. "
            "It is kept for event-aware weekly experiments and is not part of "
            "the deployed daily 27-feature forecast contract."
        ),
    },
}


def _build_metadata() -> dict:
    metadata = {}
    for group, features in FEATURE_GROUPS.items():
        for feature in features:
            metadata[feature] = {
                "group": group,
                "availability": FEATURE_AVAILABILITY[group],
            }
    return metadata


FEATURE_METADATA = _build_metadata()

missing_metadata = set(FORECAST_FEATURES) - set(FEATURE_METADATA)
if missing_metadata:
    raise RuntimeError(f"Missing S2 feature metadata: {sorted(missing_metadata)}")
