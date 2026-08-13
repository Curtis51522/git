function injectS5Button(parentId, moduleId, dateSelector, resultDivId, style) {
  var parent = document.getElementById(parentId);
  if (!parent) return;
  style = style || '';
  var btn = document.createElement('button');
  btn.className = 'btn btn-sm';
  btn.style.cssText = 'margin-left:8px;background:#0071e3;color:#fff;border:none;border-radius:11px;padding:8px 16px;font-size:13px;font-weight:650;cursor:pointer' + (style ? ';' + style : '');
  btn.textContent = t('AI Analysis');
  btn.onclick = function() { runModuleS5Analysis(moduleId, dateSelector, resultDivId); };
  parent.appendChild(btn);
}

function injectS5ResultDiv(targetId, resultDivId) {
  var target = document.getElementById(targetId);
  if (!target) return;
  var div = document.createElement('div');
  div.id = resultDivId;
  div.style.cssText = 'display:none;background:#fbfbfd;border:1px solid rgba(29,29,31,0.09);border-radius:16px;padding:16px;margin-bottom:12px;box-shadow:0 14px 34px -30px rgba(17,17,20,0.42)';
  target.parentNode.insertBefore(div, target);
}

function getS5Lang() {
  var lang = localStorage.getItem('bakery_lang') || 'en';
  return lang === 'zh' ? 'zh' : 'en';
}

function formatS5TimeLabel(value) {
  var key = value || 'medium';
  return t(key) || key;
}

function escapeS5Html(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function formatS5Confidence(value) {
  var score = Number(value);
  if (!isFinite(score)) return 'N/A';
  return Math.round(score * 100) + '%';
}

function labelS5Token(value) {
  var labels = {
    forecast_uncertainty_hotspots: 'Demand uncertainty hotspots',
    forecast_volatility_risk: 'Large profit swing risk',
    material_shortage_risk: 'Low-stock material risk',
    material_data_gap: 'Raw-material stock data unavailable',
    overproduction_risk: 'Overproduction risk',
    stockout_risk: 'Stockout risk',
    widespread_low_stock_risk: 'Widespread low-stock risk',
    inventory_data_gap: 'Inventory data gap',
    inventory_expiry_risk: 'Day-1 stock risk',
    scenario_profit_gap: 'Profit swing between downside and expected demand',
    production_waste_rate_pct: 'Expected waste exposure',
    q90_shortage_units: 'High-demand capacity gap',
    production_total_bake: 'Planned bake units',
    supply_coverage_pct: 'Supply coverage rate',
    demand_gap_units: 'Forecast demand gap',
    total_available_units: 'Total available supply',
    material_low_count: 'Low-stock materials',
    material_critical_count: 'Critical material blockers',
    material_order_by_unit: 'Material order requirement by unit',
    material_stock_data_available: 'Raw-material stock data status',
    material_count_checked: 'Materials checked',
    wasted_material_count: 'Materials with recorded waste',
    total_waste_cost: 'Recorded waste cost',
    top_consumed_materials: 'Top consumed materials',
    yield_data_available: 'Production yield data availability',
    yield_total_units: 'Yield total units',
    material_wastage_risk: 'Material wastage risk',
    yield_data_gap: 'Production yield data gap',
    forecast_total_units: 'Forecast demand units',
    forecast_total_revenue: 'Forecast revenue',
    forecast_avg_interval_width: 'Demand uncertainty range',
    forecast_wape: 'Held-out historical forecast error',
    forecast_coverage: 'Forecast coverage',
    profit_margin_pct: 'Profit margin',
    revenue: 'Revenue',
    discount_total: 'Discount exposure',
    order_volume: 'Order volume',
    average_order_value: 'Average order value',
    revenue_trend_pct: 'Revenue trend',
    order_change_pct: 'Order change',
    average_order_value_change_pct: 'Average order value change',
    items_per_order: 'Items per order',
    items_per_order_change_pct: 'Items per order change',
    revenue_per_item: 'Revenue per item',
    revenue_per_item_change_pct: 'Revenue per item change',
    top_product_revenue_share_pct: 'Top product revenue share',
    top3_product_revenue_share_pct: 'Top product concentration',
    category_revenue_split: 'Category revenue split',
    peak_revenue_hour: 'Peak revenue hour',
    hourly_peak_revenue_share_pct: 'Peak-hour revenue share',
    low_revenue_hours: 'Low revenue hours',
    discount_rate_pct: 'Discount rate',
    revenue_decline: 'Revenue decline risk',
    basket_size_weakness: 'Basket size weakness',
    order_value_shift: 'Order value shift risk',
    low_sample_size: 'Low sample size',
    possible_data_gap: 'Possible data gap',
    product_concentration: 'Product concentration risk',
    hourly_revenue_concentration: 'Hourly revenue concentration risk',
    discount_margin_erosion: 'Discount margin erosion risk',
    promotionsignalagent_claim: 'Promotion signal',
    promotionproductmixagent_claim: 'Product mix signal',
    promotion_decision_basis: 'Promotion decision basis',
    top3_bread_revenue_share_pct: 'Top 3 bread concentration',
    bread_revenue_share_pct: 'Bread revenue share',
    beverage_revenue_share_pct: 'Beverage revenue share',
    expired_cost: 'Closing unsold-product cost',
    expired_cost_revenue_pct: 'Closing loss share of revenue',
    profit_margin_before_expiry_pct: 'Profit margin before closing loss',
    expired_products: 'Products discarded at closing',
    top_product_name: 'Top product',
    promotion_no_broad_discount: 'No broad discount decision',
    promotion_mid_tier_bundle: 'Targeted bundle opportunity',
    inventory_total: 'Finished-product stock',
    inventory_record_status: 'Inventory record status',
    zero_stock_product_count: 'Products with no stock',
    low_stock_product_count: 'Products with one unit left',
    thin_stock_product_share_pct: 'Products with zero or one unit',
    units_per_product: 'Average units per product',
    low_stock_material_count: 'Raw-material reorder status',
    inventory_recommendation_basis: 'Inventory recommendation basis'
  };
  var key = String(value || '');
  if (labels[key]) return labels[key];
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, function(ch) { return ch.toUpperCase(); });
}

function collectS5Details(data) {
  var outputs = Array.isArray(data.agent_outputs) ? data.agent_outputs : [];
  var confidenceTotal = 0;
  var confidenceCount = 0;
  var risks = [];
  var agentEvidenceIds = [];
  var recommendationEvidenceIds = [];

  for (var i = 0; i < outputs.length; i++) {
    var confidence = Number(outputs[i].confidence);
    if (isFinite(confidence)) {
      confidenceTotal += confidence;
      confidenceCount += 1;
    }
    var outputRisks = Array.isArray(outputs[i].risks) ? outputs[i].risks : [];
    for (var r = 0; r < outputRisks.length; r++) risks.push(outputRisks[r]);
    var outputEvidence = Array.isArray(outputs[i].evidence_items) ? outputs[i].evidence_items : [];
    for (var a = 0; a < outputEvidence.length; a++) {
      if (outputEvidence[a] && outputEvidence[a].id) agentEvidenceIds.push(outputEvidence[a].id);
    }
  }

  var recommendations = Array.isArray(data.recommendations) ? data.recommendations : [];
  for (var j = 0; j < recommendations.length; j++) {
    var ids = Array.isArray(recommendations[j].evidence_ids) ? recommendations[j].evidence_ids : [];
    for (var e = 0; e < ids.length; e++) recommendationEvidenceIds.push(ids[e]);
  }
  var evidenceIds = recommendationEvidenceIds.concat(agentEvidenceIds);

  return {
    confidence: confidenceCount ? confidenceTotal / confidenceCount : null,
    risks: Array.from(new Set(risks)).slice(0, 4),
    evidenceIds: Array.from(new Set(evidenceIds)).slice(0, 6)
  };
}

function renderS5Details(data) {
  var verification = data.verification_report || {};
  var details = collectS5Details(data || {});
  var passed = verification.passed === true;
  var statusText = passed ? 'Verified' : 'Needs review';
  var statusColor = passed ? '#1e8449' : '#b03a2e';
  var statusBg = passed ? '#eafaf1' : '#fdedec';
  var risks = details.risks.length ? details.risks.map(labelS5Token).join(', ') : 'None flagged';
  var evidence = details.evidenceIds.length ? details.evidenceIds.map(labelS5Token).join(', ') : 'Not linked';

  return '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:10px 0 12px 0;font-size:12px;color:#3d322b">' +
    '<div style="border:1px solid #eadfcc;background:#fffaf2;border-radius:6px;padding:8px"><div style="color:#7d6b5d;font-size:11px;margin-bottom:2px">' + t('Verification') + '</div><span style="display:inline-block;background:' + statusBg + ';color:' + statusColor + ';border-radius:3px;padding:2px 7px;font-weight:600">' + escapeS5Html(statusText) + '</span></div>' +
    '<div style="border:1px solid #eadfcc;background:#fffaf2;border-radius:6px;padding:8px"><div style="color:#7d6b5d;font-size:11px;margin-bottom:2px">' + t('Confidence') + '</div><strong>' + escapeS5Html(formatS5Confidence(details.confidence)) + '</strong></div>' +
    '<div style="border:1px solid #eadfcc;background:#fffaf2;border-radius:6px;padding:8px"><div style="color:#7d6b5d;font-size:11px;margin-bottom:2px">' + t('Risks') + '</div><span>' + escapeS5Html(risks) + '</span></div>' +
    '<div style="border:1px solid #eadfcc;background:#fffaf2;border-radius:6px;padding:8px"><div style="color:#7d6b5d;font-size:11px;margin-bottom:2px">' + t('Evidence') + '</div><span>' + escapeS5Html(evidence) + '</span></div>' +
    '</div>';
}

async function runModuleS5Analysis(moduleId, dateSelector, resultDivId) {
  var resDiv = document.getElementById(resultDivId);
  if (!resDiv) return;
  var forceRefresh = arguments[3] || false;
  resDiv.style.display = 'block';
  resDiv.innerHTML = '<div style="text-align:center;padding:24px;color:#0071e3"><span class="spinner"></span> ' + t('Analyzing...') + '</div>';
  try {
    var hdrs = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token};
    var dateEl = document.getElementById(dateSelector);
    var date = dateEl ? dateEl.value : '';
    var lang = getS5Lang();
    resDiv.setAttribute('data-s5-module', moduleId);
    resDiv.setAttribute('data-s5-date-selector', dateSelector);
    resDiv.setAttribute('data-s5-lang', lang);
    var r = await fetch(S5_API + '/analyze/module', {method: 'POST', headers: hdrs, body: JSON.stringify({module: moduleId, date: date, lang: lang, force_refresh: forceRefresh})});
    if (!r.ok) { if(r.status===401&&handleUnauthorizedResponse(r))return; var txt = await r.text(); throw new Error(txt); }
    var d = await r.json();
    var refreshBtn = forceRefresh ? '' : '<button onclick="runModuleS5Analysis(\'' + moduleId + '\', \'' + dateSelector + '\', \'' + resultDivId + '\', true)" title="' + t('Regenerate') + '" style="background:none;border:none;color:#0071e3;cursor:pointer;font-size:16px;margin-right:6px">&#x21bb;</button>';
    var html = '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid rgba(29,29,31,0.09)"><h4 style="margin:0;color:#1d1d1f;font-size:15px">' + t('AI Analysis') + (forceRefresh ? ' <span style="font-size:11px;color:#a65a00">(' + t('refreshed') + ')</span>' : '') + '</h4><div>' + refreshBtn + '<button onclick="document.getElementById(\'' + resultDivId + '\').style.display=\'none\'" style="background:none;border:none;color:#6e6e73;cursor:pointer;font-size:18px">&times;</button></div></div>';
    var summaryHtml = escapeS5Html(d.summary)
      .replace(/\n\n/g, '</p><p style="font-size:14px;color:#3d322b;line-height:1.8;margin-bottom:6px">')
      .replace(/\n/g, '<br>')
      .replace(/^(BOTTOM LINE|WHY THIS HAPPENED|WHAT TO DO)/gm, '<strong style="color:#1d1d1f;font-size:15px"></strong>');
    html += '<div style="margin-bottom:12px"><p style="font-size:14px;color:#3d322b;line-height:1.8;margin-bottom:6px">' + summaryHtml + '</p></div>';
    if (d.recommendations && d.recommendations.length > 0) {
      html += '<div><strong style="font-size:13px;color:#1d1d1f">' + t('Recommendations') + '</strong><div style="margin-top:6px">';
      for (var i = 0; i < Math.min(d.recommendations.length, 4); i++) {
        var rec = d.recommendations[i];
        var urgColor = rec.urgency === 'high' ? '#c0392b' : rec.urgency === 'medium' ? '#e67e22' : '#27ae60';
        var urgBg = rec.urgency === 'high' ? '#fdedec' : rec.urgency === 'medium' ? '#fef5e7' : '#eafaf1';
        html += '<div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:6px;font-size:13px;color:#3d322b"><span style="background:' + urgBg + ';color:' + urgColor + ';border-radius:3px;padding:2px 8px;font-size:10px;font-weight:600;white-space:nowrap;flex-shrink:0">' + escapeS5Html(formatS5TimeLabel(rec.time_horizon || rec.urgency || 'medium').toUpperCase()) + '</span><div><div>' + escapeS5Html(rec.action) + '</div>' + (rec.rationale ? '<div style="color:#6b5b4f;font-size:12px;line-height:1.5;margin-top:3px">' + escapeS5Html(rec.rationale) + '</div>' : '') + (rec.expected_impact ? '<div style="color:#27ae60;font-size:12px;line-height:1.5;margin-top:2px">' + escapeS5Html(rec.expected_impact) + '</div>' : '') + '</div></div>';
      }
      html += '</div></div>';
    }
    html += renderS5Details(d);
    resDiv.innerHTML = html;
  } catch (ex) {
    resDiv.innerHTML = '<div style="color:#c0392b;text-align:center;padding:12px">' + escapeS5Html(t('Analysis failed')) + ': ' + escapeS5Html(ex.message) + '</div>';
  }
}
