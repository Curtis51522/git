from pathlib import Path


INDEX_HTML = Path("api/module4_frontend/static/index.html")
BFF_SOURCE = Path("api/module4_frontend/bff.py")


def _function_source(html, name, next_name):
    start = html.index(name)
    end = html.index(next_name, start)
    return html[start:end]


def test_checkout_discount_is_clamped_to_server_validated_rate():
    from api.module4_frontend.bff import _resolve_checkout_discount

    resolved = _resolve_checkout_discount(
        item={"discount_rate": 0.25},
        allowed_dynamic={
            "discount_pct": 12,
            "source": "live_policy",
            "strategy": "diversify",
            "reason": "Live inventory pressure",
        },
        freshness_rate=0.0,
    )

    assert resolved == {
        "rate": 0.12,
        "source": "live_policy",
        "strategy": "diversify",
        "reason": "Live inventory pressure",
    }


def test_freshness_discount_wins_when_dynamic_rate_is_lower():
    from api.module4_frontend.bff import _resolve_checkout_discount

    resolved = _resolve_checkout_discount(
        item={"discount_rate": 0.12},
        allowed_dynamic={"discount_pct": 12, "source": "live_policy"},
        freshness_rate=0.2,
    )

    assert resolved["rate"] == 0.2
    assert resolved["source"] == "freshness"


def test_business_event_discount_is_clamped_to_server_validated_rate():
    from api.module4_frontend.bff import _resolve_checkout_discount

    resolved = _resolve_checkout_discount(
        item={"discount_rate": 0.25},
        allowed_dynamic={},
        freshness_rate=0.0,
        allowed_event={
            "discount_pct": 12,
            "source": "business_event",
            "strategy": "new_product_launch",
            "reason": "Active New Product Launch",
        },
    )

    assert resolved == {
        "rate": 0.12,
        "source": "business_event",
        "strategy": "new_product_launch",
        "reason": "Active New Product Launch",
    }


def test_checkout_loads_active_business_event_discounts_from_database():
    source = BFF_SOURCE.read_text(encoding="utf-8")
    checkout_source = source[source.index("async def checkout_complete") :]

    assert "_load_active_business_event_discounts" in source
    assert "event_discounts =" in checkout_source
    assert "allowed_event=event_discounts.get(product_name, {})" in checkout_source


def test_frontend_carries_dynamic_discount_metadata_to_checkout():
    html = INDEX_HTML.read_text(encoding="utf-8")
    confirm_payment = _function_source(html, "async function confirmPayment()", "function showReceipt")

    assert "discount_rate:ci.discount_rate||0" in confirm_payment
    assert "discount_source:ci.discount_source||''" in confirm_payment
    assert "discount_strategy:ci.discount_strategy||''" in confirm_payment
    assert "discount_reason:ci.discount_reason||''" in confirm_payment


def test_frontend_applies_active_business_event_discount_to_product_and_cart():
    html = INDEX_HTML.read_text(encoding="utf-8")
    quick_add = _function_source(
        html,
        "function quickAddBakery(name,fresh)",
        "function handleScan",
    )

    assert "getBusinessEventForProduct(name)" in quick_add
    assert "discountSource='business_event'" in quick_add
    assert "discount_source:discountSource" in quick_add
    assert "Math.max(discountRate,eventDiscount/100)" in html
    assert "Number(rc.discount_rate||0)" in html
    assert "rc.discount_source||" in html


def test_checkout_persists_applied_discount_in_order_items_and_receipt_json():
    source = BFF_SOURCE.read_text(encoding="utf-8")
    checkout_source = source[source.index("async def checkout_complete"):]

    assert "_fetch_validated_dynamic_discounts" in source
    assert "_resolve_checkout_discount" in source
    assert "discounted_unit_values(base_price, discount_rate)" in source
    assert '"line_total": float(priced_item["line_final"])' in checkout_source
    assert "float(priced_item[\"line_final\"]), line_profit" in checkout_source
    assert '"discount_source": resolved_discount["source"]' in source
    assert '"discount_strategy": resolved_discount["strategy"]' in source
    assert "INSERT INTO receipts" in source
    assert "json.dumps(receipt_items" in source


def test_receipt_renders_applied_discount_rate_and_source():
    html = INDEX_HTML.read_text(encoding="utf-8")
    show_receipt = _function_source(html, "function showReceipt", "function closePaymentModal")

    assert "it.discount_pct>0" in show_receipt
    assert "it.discount_source" in show_receipt
    assert "t(\"Discount\")" in show_receipt
