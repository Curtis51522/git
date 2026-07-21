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


def test_forecast_table_uses_business_friendly_uncertainty_cells():
    html = _html()

    assert "function renderForecastDemandCell" in html
    assert "uncertainty_level" in html
    assert "relative_width" in html
    assert "var labels={low:'Stable',medium:'Stable',high:'Flexible'}" in html
    assert "var labels={low:'Low',medium:'Medium',high:'High'}" not in html
    assert "medium:'Review'" not in html
    assert "renderForecastDemandCell(v)" in html
    assert "txt=lo+'-'+hi" not in html


def test_forecast_accuracy_bar_uses_range_hit_label():
    html = _html()

    assert "'Range Hit':'Range Hit'" in html
    assert "+t('Range Hit')+" in html


def test_forecast_supply_coverage_uses_bakery_demand_only():
    html = _html()

    assert "window._forecastBakeryDemandUnits=0" in html
    assert "var demand=Number(window._forecastBakeryDemandUnits||0)" in html
    assert "forecastBakeryDemand+=qty" in html
    assert "Object.prototype.hasOwnProperty.call(COFFEE_PRICES,pn)" in html
    assert "window._forecastBakeryDemandUnits=forecastBakeryDemand" in html


def test_forecast_demand_kpi_matches_s5_total_and_category_scope():
    html = _html()

    assert "t('7-Day Forecast Demand')" in html
    assert 'id="fc-kpi-demand-split"' in html
    assert "demandEl.textContent=Math.round(forecastDemand).toLocaleString()+' '+t('units')" in html
    assert "t('Bakery')+' '+Math.round(forecastBakeryDemand).toLocaleString()" in html
    assert "t('Beverages')+' '+Math.round(forecastBeverageDemand).toLocaleString()" in html
    assert "var forecastBeverageDemand=Math.max(forecastDemand-forecastBakeryDemand,0)" in html


def test_forecast_demand_kpi_translation_keys_exist():
    html = _html()

    assert "'7-Day Forecast Demand':'7-Day Forecast Demand'" in html
    assert "'Bakery':'Bakery'" in html
    assert "'7-Day Forecast Demand':'\\u0037\\u5929\\u9884\\u6D4B\\u9700\\u6C42'" in html
    assert "'Bakery':'\\u70D8\\u7119\\u4EA7\\u54C1'" in html


def test_forecast_kpi_units_and_material_restock_count_are_business_readable():
    html = _html()

    assert "forecastRevEl.textContent='\\u00A5'+formatMoney(forecastRevenue)" in html
    assert "planEl.textContent=Math.round(totalBake).toLocaleString()+' '+t('units')" in html
    assert "gapEl.textContent=Math.max(Math.round(demand-available),0).toLocaleString()+' '+t('units')" in html
    assert "t('Materials to Restock')" in html
    assert "matlEl.textContent=totalOrder.toLocaleString()+' '+t(totalOrder===1?'item':'items')" in html
    assert "matlEl.textContent=totalOrderQty" not in html


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


def test_inventory_top_consumption_chart_descends_from_top_to_bottom():
    html = _html()
    chart_start = html.index('var cc=document.getElementById("inv-consumption-chart")')
    chart_end = html.index("// Stock days table", chart_start)
    chart = html[chart_start:chart_end]

    assert 'yAxis:{type:"category",inverse:true' in chart


def test_inventory_visible_copy_uses_recipe_based_material_groups():
    html = _html()

    assert "'Baking Ingredients':'Baking Ingredients'" in html
    assert (
        "'Beverage Ingredients & Supplies':'Beverage Ingredients & Supplies'"
        in html
    )
    assert "'Packaging Supplies':'Packaging Supplies'" in html
    assert "d.beverage_materials||d.coffee_materials||[]" in html
    assert "d.packaging_materials||[]" in html
    assert "m.usage_scope" in html
    assert (
        "chart-panel chart-full\"><h4>'+t('Baking Ingredients')"
        in html
    )
    assert (
        "id=\"inv-beverage-table\"></div></div><div class=\"chart-panel\"><h4>'+t('Packaging Supplies')"
        in html
    )
    assert "totalBaking" not in html
    assert "totalCoffee" not in html
    assert "Baking Materials (kg/L)" not in html
    assert "Beverage Materials (kg/L)" not in html


def test_pos_drinks_have_lightweight_category_filters():
    html = _html()

    assert "'Drinks - Quick Order':'Drinks - Quick Order'" in html
    assert "'All Drinks':'All Drinks'" in html
    assert "'Tea & Milk Tea':'Tea & Milk Tea'" in html
    assert "'Other Drinks':'Other Drinks'" in html
    assert "var activeDrinkCategory='all';" in html
    assert "function setDrinkCategory(category)" in html
    assert 'id="pos-drink-filters"' in html
    assert 'id="pos-drink-grid"' in html
    assert "data-drink-category=\"'+cd.category+'\"" in html
    assert "category:'coffee'" in html
    assert "category:'tea_milk_tea'" in html
    assert "category:'other'" in html


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
    assert "expired_cost: 'Closing unsold-product cost'" in labels
    assert "expired_cost_revenue_pct: 'Closing loss share of revenue'" in labels
    assert "expired_products: 'Products discarded at closing'" in labels


def test_inventory_material_evidence_label_is_status_neutral():
    labels = Path("api/module4_frontend/static/s5_analysis.js").read_text(encoding="utf-8")

    assert "low_stock_material_count: 'Raw-material reorder status'" in labels
    assert "low_stock_material_count: 'Materials at or below reorder point'" not in labels


def test_s5_details_prioritize_recommendation_evidence():
    labels = Path("api/module4_frontend/static/s5_analysis.js").read_text(encoding="utf-8")

    assert "recommendationEvidenceIds.concat(agentEvidenceIds)" in labels
    assert "agentEvidenceIds.concat(recommendationEvidenceIds)" not in labels
    assert "profit_margin_before_expiry_pct: 'Profit margin before closing loss'" in labels


def test_s5_recommendation_supporting_text_is_readable():
    labels = Path("api/module4_frontend/static/s5_analysis.js").read_text(encoding="utf-8")

    assert "color:#6b5b4f;font-size:12px;line-height:1.5" in labels
    assert "color:#27ae60;font-size:12px;line-height:1.5" in labels


def test_revenue_trend_uses_revenue_title_and_currency_units():
    html = _html()
    chart_start = html.index("var tc=document.getElementById('rev-trend-chart')")
    chart_end = html.index("var oc=document.getElementById('rev-orders-chart')", chart_start)
    chart = html[chart_start:chart_end]

    assert "t('7-Day Revenue Trend')" in html
    assert "t('7-Day Profit Trend')" not in html
    assert "name:t('Revenue (CNY)')" in chart
    assert "return '\\u00A5'+value" in chart
    assert "seriesName+': \\u00A5'" in chart


def test_orders_and_atv_chart_formats_only_atv_as_currency():
    html = _html()
    chart_start = html.index("var oc=document.getElementById('rev-orders-chart')")
    chart_end = html.index("document.getElementById('rev-bread-ranking')", chart_start)
    chart = html[chart_start:chart_end]

    assert "name:t('Avg Ticket Value (CNY)')" in chart
    assert "return '\\u00A5'+value" in chart
    assert "seriesName+': \\u00A5'" in chart
    assert "seriesName+': '+value.toFixed(0)" in chart


def test_expired_session_returns_to_login_for_dashboard_and_s5_requests():
    html = _html()
    s5_source = Path("api/module4_frontend/static/s5_analysis.js").read_text(encoding="utf-8")

    assert "function handleUnauthorizedResponse(response)" in html
    assert "response.status!==401" in html
    assert "sessionStorage.removeItem('bakery_token')" in html
    assert "if(r.status===401&&handleUnauthorizedResponse(r))" in html
    assert "if(r.status===401&&handleUnauthorizedResponse(r))" in s5_source
    assert "Session expired. Please sign in again." in html
