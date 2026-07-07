from pathlib import Path


INDEX_HTML = Path("api/module4_frontend/static/index.html")


def _html():
    return INDEX_HTML.read_text(encoding="utf-8")


def test_forecast_business_events_strip_is_rendered():
    html = _html()

    assert "fc-business-events" in html
    assert "fc-business-events-list" in html
    assert "openBusinessEventModal" in html
    assert "new_product_launch" in html
    assert "competitor_activity" in html


def test_business_events_modal_and_handlers_exist():
    html = _html()

    assert 'id="business-event-modal"' in html
    assert "function openBusinessEventModal" in html
    assert "function closeBusinessEventModal" in html
    assert "async function saveBusinessEvent" in html
    assert "async function loadBusinessEvents" in html
    assert "function renderBusinessEvents" in html
    assert "/s2/business-events" in html


def test_forecast_initial_render_loads_business_events():
    html = _html()
    render_start = html.index("function renderForecast")
    next_function = html.index("function getForecastPanelDate", render_start)
    render_body = html[render_start:next_function]

    assert "loadBusinessEvents();" in render_body


def test_business_event_chip_uses_english_separator():
    html = _html()

    assert "parts.join(' | ')" in html
    assert "parts.join(' \\u00B7 ')" not in html


def test_business_event_discount_input_guides_percent_values():
    html = _html()

    assert 'id="business-event-discount" type="number" min="0" max="100" step="1"' in html
    assert 'placeholder="e.g. 25 for 25%"' in html


def test_business_event_discount_normalizes_fractional_percent_input():
    html = _html()

    assert "function normalizeBusinessEventDiscount" in html
    assert "if(value>0&&value<=1)return value*100;" in html
    assert "discount_pct:normalizeBusinessEventDiscount(discountRaw)" in html


def test_business_event_save_refreshes_without_stale_cache():
    html = _html()

    assert "function clearApiCacheByPath" in html
    assert "clearApiCacheByPath('/s2/business-events')" in html
    assert "loadBusinessEvents(true)" in html


def test_business_event_load_can_bypass_get_cache():
    html = _html()

    assert "var force=opts.force===true;" in html
    assert "method==='GET'&&!force&&_apiCache" in html
    assert "api('/s2/business-events?date='+encodeURIComponent(sd),{force:!!force})" in html


def test_business_event_chip_exposes_delete_action():
    html = _html()

    assert "async function deleteBusinessEvent" in html
    assert "deleteBusinessEvent('+eventId+')" in html
    assert "Delete this business event?" in html


def test_business_event_delete_refreshes_without_stale_cache():
    html = _html()

    assert "api('/s2/business-events/'+encodeURIComponent(eventId),{method:'DELETE'})" in html
    assert "clearApiCacheByPath('/s2/business-events')" in html
    assert "loadBusinessEvents(true)" in html


def test_business_events_do_not_override_checkout_discount_logic():
    html = _html()

    assert "window._activeBusinessEvents" in html
    assert "event_discount_overrides" not in html
    assert "checkout_event_discount" not in html


def test_product_cards_show_event_discount_context_only():
    html = _html()

    assert "function getBusinessEventDiscountForProduct" in html
    assert "event-discount-tag" in html
    assert "Event:" in html
    assert "var eventDiscount=getBusinessEventDiscountForProduct(bk)" in html
    assert "discountRate=eventDiscount" not in html


def test_top3_request_includes_business_event_context_without_score_override():
    html = _html()

    assert "function activeBusinessEventContext" in html
    assert "comboBody.business_events=activeBusinessEventContext()" in html
    assert "business_event_score" not in html
    assert "business_event_boost" not in html


def test_inventory_stock_risk_ai_analysis_entry_exists():
    html = _html()

    assert 'id="inv-stock-s5-btn"' in html
    assert 'id="inv-stock-s5-result"' in html
    assert "runModuleS5Analysis(\\'inventory\\',\\'inv-date\\',\\'inv-stock-s5-result\\')" in html
    assert "Inventory AI analysis button removed" not in html


def test_revenue_promotion_mix_ai_analysis_entry_exists():
    html = _html()

    assert 'id="rev-s5-btn"' in html
    assert "rev-s5-result" in html
    assert "runModuleS5Analysis('revenue','rev-date','rev-s5-result')" in html
    assert "Promotion & Product Mix AI" in html
    assert 'id="rev-promotion-mix-s5-result"' in html
    assert "runModuleS5Analysis(\\'promotion_mix\\',\\'rev-date\\',\\'rev-promotion-mix-s5-result\\')" in html


def test_promotion_mix_evidence_labels_are_readable():
    labels = Path("api/module4_frontend/static/s5_analysis.js").read_text(encoding="utf-8")

    assert "promotionsignalagent_claim: 'Promotion signal'" in labels
    assert "promotionproductmixagent_claim: 'Product mix signal'" in labels
    assert "top3_bread_revenue_share_pct: 'Top 3 bread concentration'" in labels
