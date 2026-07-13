import json
import subprocess
from pathlib import Path


INDEX_HTML = Path("api/module4_frontend/static/index.html")


def _source_between(source, start_marker, end_marker):
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


def _pricing_values():
    source = INDEX_HTML.read_text(encoding="utf-8")
    defaults_source = _source_between(source, "function getDefaultBeverageOptions", "function roundPosMoney")
    legacy_normalizer_source = (
        _source_between(source, "function normalizeLegacyBeverageOption", "function formatCoffeeDetails")
        if "function normalizeLegacyBeverageOption" in source
        else ""
    )
    pricing_source = _source_between(source, "function roundPosMoney", "async function loadS5DiscountCache")
    script = """
const PRODUCT_PRICES = {croissant: 10};
const COFFEE_PRICES = {latte: 18};
const beverageOptionMap = {latte: {product_name: 'latte'}};
const DISCOUNT_RATE = 0.2;
""" + defaults_source + legacy_normalizer_source + pricing_source + """
const result = {
  beverage: getItemPricing({
    product_name: 'latte',
    tray_color: 'orange',
    discount_rate: 0,
    discount_source: 'dynamic',
    discount_strategy: 'clearance',
    discount_reason: 'stale'
  }),
  bakery: getItemPricing({product_name: 'croissant', tray_color: 'yellow', discount_rate: 0})
};
console.log(JSON.stringify(result));
"""
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _loaded_price_values():
    source = INDEX_HTML.read_text(encoding="utf-8")
    load_prices_source = _source_between(source, "function loadPrices", "function loadStock")
    script = r'''
var PRODUCT_PRICES = {croissant: 10};
var COFFEE_PRICES = {latte: 18};
var COFFEE_DRINKS = [{key: 'latte', price: '18.00'}];
var currentPanel = 'inventory';
var API = '';
function renderPOS() {}
function fetch() {
  return Promise.resolve({json: function() {
    return Promise.resolve({products: [
      {product_name: 'croissant', unit_price: 11.5},
      {product_name: 'latte', unit_price: 19.5}
    ]});
  }});
}
''' + load_prices_source + r'''
loadPrices();
setTimeout(function() {
  console.log(JSON.stringify({bakery: PRODUCT_PRICES.croissant, beverage: COFFEE_PRICES.latte, menu: COFFEE_DRINKS[0].price}));
}, 0);
'''
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _run_beverage_behavior(scenario):
    source = INDEX_HTML.read_text(encoding="utf-8")
    defaults_source = _source_between(source, "function getDefaultBeverageOptions", "function roundPosMoney")
    modal_source = _source_between(source, "var _coffeeModalKey", "function quickAddCoffee")
    save_source = _source_between(source, "function saveCartItem", "function confirmInflow")
    edit_source = _source_between(source, "function editCartItem", "function removeCartItem")
    script = r'''
const PRODUCT_PRICES = {croissant: 10};
const COFFEE_PRICES = {latte: 18};
const COFFEE_DRINKS = [{key: 'latte', name: 'Latte'}];
const DISCOUNT_RATE = 0.2;
var beverageOptionMap = {};
var beverageOptionsReady = true;
var cartItems = [];
var detections = [];
var hitlLog = [];
var inflowDets = [];
var s5DiscountCache = {};
var EDITING_IDX = -1;
var EDITING_DETECT = false;
var EDITING_INFLOW = false;
var alerts = [];
var renderCount = 0;

function makeElement(dataVal) {
  return {
    dataVal: dataVal || '', value: '', textContent: '', innerHTML: '', style: {}, disabled: false,
    classList: {
      values: {},
      add: function(name) { this.values[name] = true; },
      remove: function(name) { delete this.values[name]; },
      toggle: function(name, enabled) { if (enabled) this.add(name); else this.remove(name); },
      contains: function(name) { return !!this.values[name]; }
    },
    getAttribute: function(name) { return name === 'data-val' ? this.dataVal : ''; }
  };
}

const elements = {};
const buttons = {
  size: ['regular', 'large'].map(makeElement),
  temp: ['hot', 'cold'].map(makeElement),
  sugar: ['normal', 'less', 'slight', 'none'].map(makeElement),
  ice: ['normal', 'less', 'none'].map(makeElement)
};
const document = {
  getElementById: function(id) {
    if (!elements[id]) elements[id] = makeElement();
    return elements[id];
  },
  querySelector: function(selector) {
    if (selector === '.coffee-modal-add') return this.getElementById('coffee-modal-add');
    return null;
  },
  querySelectorAll: function(selector) {
    if (selector.indexOf('coffee-size-options') !== -1) return buttons.size;
    if (selector.indexOf('coffee-temp-options') !== -1) return buttons.temp;
    if (selector.indexOf('coffee-sugar-options') !== -1) return buttons.sugar;
    if (selector.indexOf('coffee-ice-options') !== -1) return buttons.ice;
    return [];
  }
};
function t(key) { return key; }
function alert(message) { alerts.push(message); }
function renderPOS() { renderCount += 1; }
function loadStock() {}
function closeEditModal() {
  document.getElementById('edit-modal').classList.remove('show');
  EDITING_IDX = -1;
  EDITING_DETECT = false;
  EDITING_INFLOW = false;
}
function capName(name) { return name; }
function expect(value, message) { if (!value) throw new Error(message); }
function config(overrides) {
  return Object.assign({
    product_name: 'latte',
    default_size: 'regular', default_temperature: 'hot', default_sugar: 'normal', default_ice: 'normal',
    allowed_sizes: ['regular', 'large'], allowed_temperatures: ['hot', 'cold'],
    allowed_sugar: ['normal', 'less', 'slight', 'none'], allowed_ice: ['normal', 'less', 'none']
  }, overrides || {});
}
''' + defaults_source + modal_source + save_source + edit_source + scenario
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _checkout_payload_values():
    source = INDEX_HTML.read_text(encoding="utf-8")
    defaults_source = _source_between(source, "function getDefaultBeverageOptions", "function roundPosMoney")
    pricing_source = _source_between(source, "function roundPosMoney", "async function loadS5DiscountCache")
    legacy_normalizer_source = (
        _source_between(source, "function normalizeLegacyBeverageOption", "function formatCoffeeDetails")
        if "function normalizeLegacyBeverageOption" in source
        else ""
    )
    payload_source = (
        _source_between(source, "function buildCheckoutItems", "async function confirmPayment")
        if "function buildCheckoutItems" in source
        else ""
    )
    payment_modal_source = _source_between(source, "function completePayment", "function selectPaymentMethod")
    payment_source = _source_between(source, "async function confirmPayment", "function showReceipt")
    script = r'''
const PRODUCT_PRICES = {};
const COFFEE_PRICES = {latte: 18};
var beverageOptionMap = {
  latte: {
    product_name: 'latte',
    default_size: 'regular', default_temperature: 'hot', default_sugar: 'normal', default_ice: 'none',
    allowed_sizes: ['regular', 'large'], allowed_temperatures: ['hot', 'cold'],
    allowed_sugar: ['normal', 'less', 'slight', 'none'], allowed_ice: ['normal', 'less', 'none']
  }
};
var beverageOptionsReady = true;
var cartItems = [];
var requests = [];
var API = '', token = 'test-token';
var dineType = 'dine_in';
var _payMethod = 'card', _payCashInput = 0, _payReceipt = null, _payTotal = 18;
var hitlLog = [], detections = [], lastScanResult = null, bundleRecs = [], stockMap = {latte: 1};
const elements = {};
function makeElement() { return {value: '', textContent: '', disabled: false, style: {}, classList: {add: function(){}, remove: function(){}}}; }
const document = {getElementById: function(id) { if (!elements[id]) elements[id] = makeElement(); return elements[id]; }};
function t(value) { return value; }
var alerts = [];
function alert(message) { alerts.push(message); }
function optionAllowed(allowed, value) { return (allowed || []).indexOf(value) !== -1; }
function clearPanelCache() {}
function loadStock() {}
function renderPOS() {}
function showReceipt() {}
function fetch(url, init) {
  requests.push({url: url, body: JSON.parse(init.body)});
  return Promise.resolve({ok: true, json: function() { return Promise.resolve({status: 'ok', receipt: {total: 18}, deducted: []}); }});
}
''' + defaults_source + pricing_source + legacy_normalizer_source + payload_source + payment_modal_source + payment_source + r'''
async function run() {
  cartItems = [{product_name: 'latte', quantity: 1, size: 'Large', temperature: 'COLD', sugar: 'LESS', ice_level: 'LESS'}];
  completePayment();
  var paymentTotal = _payTotal;
  await confirmPayment();
  var payment = requests[0].body.items[0];
  requests = [];
  cartItems = [{product_name: 'latte', quantity: 1, size: 'giant', temperature: 'cold', sugar: 'less', ice_level: 'less'}];
  var paymentOriginal = JSON.stringify(cartItems);
  await confirmPayment();
  var paymentRejected = {fetches: requests.length, unchanged: JSON.stringify(cartItems) === paymentOriginal, alert: alerts[alerts.length - 1]};
  requests = [];
  cartItems = [{product_name: 'latte', quantity: 1, size: 123, temperature: 'cold', sugar: 'less', ice_level: 'less'}];
  var paymentTypeOriginal = JSON.stringify(cartItems);
  await confirmPayment();
  var paymentTypeRejected = {fetches: requests.length, unchanged: JSON.stringify(cartItems) === paymentTypeOriginal, alert: alerts[alerts.length - 1]};
  requests = [];
  console.log(JSON.stringify({payment: payment, paymentTotal: paymentTotal, paymentRejected: paymentRejected, paymentTypeRejected: paymentTypeRejected}));
}
run().catch(function(error) { console.error(error.stack || error); process.exit(1); });
'''
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _checkout_error_value():
    source = INDEX_HTML.read_text(encoding="utf-8")
    defaults_source = _source_between(source, "function getDefaultBeverageOptions", "function roundPosMoney")
    payload_source = _source_between(source, "function buildCheckoutItems", "async function confirmPayment")
    payment_source = _source_between(source, "async function confirmPayment", "function showReceipt")
    script = r'''
const COFFEE_PRICES = {latte: 18};
var beverageOptionMap = {latte: {
  default_size: 'regular', default_temperature: 'hot', default_sugar: 'normal', default_ice: 'none',
  allowed_sizes: ['regular', 'large'], allowed_temperatures: ['hot', 'cold'],
  allowed_sugar: ['normal', 'less', 'slight', 'none'], allowed_ice: ['normal', 'less', 'none']
}};
var beverageOptionsReady = true;
var cartItems = [{product_name: 'latte', quantity: 1, size: 'regular', temperature: 'hot', sugar: 'normal', ice_level: 'none'}];
var API = '', token = 'token', dineType = 'dine_in', _payMethod = 'card', _payCashInput = 0;
var alerts = [], hitlLog = [], detections = [], lastScanResult = null, bundleRecs = [];
function t(value) { return value; }
function alert(value) { alerts.push(value); }
function optionAllowed(allowed, value) { return allowed.indexOf(value) !== -1; }
function fetch() { return Promise.resolve({ok: false, json: function() { return Promise.resolve({detail: 'Invalid sugar for latte: sweet'}); }}); }
function clearPanelCache() {}
function loadStock() {}
function renderPOS() {}
function showReceipt() {}
const document = {getElementById: function() { return {}; }};
''' + defaults_source + payload_source + payment_source + r'''
confirmPayment().then(function() {
  console.log(JSON.stringify({alert: alerts[0], cart: cartItems}));
}).catch(function(error) { console.error(error.stack || error); process.exit(1); });
'''
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _run_bundle_behavior(scenario):
    source = INDEX_HTML.read_text(encoding="utf-8")
    defaults_source = _source_between(source, "function getDefaultBeverageOptions", "function roundPosMoney")
    bundle_source = _source_between(source, "function addBundleToCart", "async function getScript")
    script = r'''
const PRODUCT_PRICES = {croissant: 10};
const COFFEE_PRICES = {latte: 18};
var beverageOptionMap = {latte: {
  default_size: 'regular', default_temperature: 'hot', default_sugar: 'normal', default_ice: 'none',
  allowed_sizes: ['regular', 'large'], allowed_temperatures: ['hot', 'cold'],
  allowed_sugar: ['normal', 'less', 'slight', 'none'], allowed_ice: ['normal', 'less', 'none']
}};
var beverageOptionsReady = true;
var cartItems = [], bundleRecs = [], freshnessMap = {}, alerts = [], renderCount = 0;
var window = {_bakeryFreshness: {croissant: 'Fresh'}};
function t(value) { return value; }
function alert(value) { alerts.push(value); }
function optionAllowed(allowed, value) { return (allowed || []).indexOf(value) !== -1; }
function renderPOS() { renderCount += 1; }
function expect(value, message) { if (!value) throw new Error(message); }
const document = {getElementById: function() { return {}; }};
''' + defaults_source + bundle_source + scenario
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _receipt_display_values():
    source = INDEX_HTML.read_text(encoding="utf-8")
    pricing_source = _source_between(source, "function roundPosMoney", "async function loadS5DiscountCache")
    legacy_normalizer_source = (
        _source_between(source, "function normalizeLegacyBeverageOption", "function formatCoffeeDetails")
        if "function normalizeLegacyBeverageOption" in source
        else ""
    )
    formatter_source = _source_between(
        source,
        "function formatCoffeeDetails",
        "function formatReceiptBeverageDetails",
    )
    receipt_formatter_source = (
        _source_between(source, "function formatReceiptBeverageDetails", "function applyI18n")
        if "function formatReceiptBeverageDetails" in source
        else ""
    )
    defaults_source = _source_between(source, "function getDefaultBeverageOptions", "function roundPosMoney")
    receipt_source = _source_between(source, "function showReceipt", "function closePaymentModal")
    history_source = _source_between(source, "function viewReceipt", "function setDineType")
    script = r'''
const PRODUCT_PRICES = {};
const COFFEE_PRICES = {latte: 18};
  var beverageOptionMap = {latte: {
    default_size: 'regular', default_temperature: 'hot', default_sugar: 'normal', default_ice: 'none',
    allowed_sizes: ['regular', 'large'], allowed_temperatures: ['hot', 'cold'],
    allowed_sugar: ['normal', 'less', 'slight', 'none'], allowed_ice: ['normal', 'less', 'none']
  }};
  var beverageOptionsReady = true;
var _payTotal = 0, _payMethod = 'card', cartItemsRaw = [];
var API = '', token = 'test-token';
const elements = {};
function makeElement() {
  return {textContent: '', innerHTML: '', classList: {add: function(){}, remove: function(){}}};
}
const document = {
  getElementById: function(id) { if (!elements[id]) elements[id] = makeElement(); return elements[id]; },
  body: {insertAdjacentHTML: function(position, html) { elements.history = {innerHTML: html}; }}
};
function t(value) { return value; }
function capName(value) { return value; }
function alert() {}
function optionAllowed(allowed, value) { return (allowed || []).indexOf(value) !== -1; }
function fetch() {
  return Promise.resolve({json: function() {
    return Promise.resolve({
      ticket_id: 'T-1', date: '2026-07-11', time: '10:00', dine_type: 'dine_in', state: 'paid',
      items: [{product_name: 'latte', quantity: 1, line_total: 21.24, size: 'large', temp: 'cold', sugar: 'less', ice: 'less'}],
      subtotal: 21.24, discount: 0, total: 21.24
    });
  }});
}
''' + defaults_source + pricing_source + legacy_normalizer_source + formatter_source + receipt_formatter_source + receipt_source + history_source + r'''
async function run() {
  showReceipt({items: [{product_name: 'latte', quantity: 1, line_total: 21.24, size: 'large', temp: 'cold', sugar: 'less', ice: 'less'}], total: 21.24});
  var receipt = elements['receipt-content'].innerHTML;
  cartItemsRaw = [{product_name: 'latte', quantity: 1, size: ' Large ', temperature: ' COLD ', sugar: ' LESS ', ice_level: ' LESS '}];
  _payTotal = 21.24;
  showReceipt();
  var fallback = elements['receipt-content'].innerHTML;
  viewReceipt('T-1');
  await new Promise(function(resolve) { setTimeout(resolve, 0); });
  var history = elements.history.innerHTML;
  showReceipt({items: [{product_name: 'latte', quantity: 1, line_total: 18}], total: 18});
  var legacyDefaults = elements['receipt-content'].innerHTML;
  beverageOptionMap = {};
  beverageOptionsReady = false;
  showReceipt({items: [{product_name: 'latte', quantity: 1, line_total: 18}], total: 18});
  var legacyOmitted = elements['receipt-content'].innerHTML;
  console.log(JSON.stringify({receipt: receipt, history: history, fallback: fallback, legacyDefaults: legacyDefaults, legacyOmitted: legacyOmitted}));
}
run().catch(function(error) { console.error(error.stack || error); process.exit(1); });
'''
    result = subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def test_frontend_loads_backend_beverage_capabilities():
    source = INDEX_HTML.read_text(encoding="utf-8")
    login_source = _source_between(source, "async function doLogin()", "async function api(")

    assert "function loadBeverageOptions" in source
    assert "'/s4/beverages/options'" in source
    assert "beverageOptionMap" in source
    assert "await loadBeverageOptions()" not in login_source
    assert "loadBeverageOptions().catch" in login_source


def test_top3_beverage_uses_validated_backend_defaults_without_deriving_ice():
    source = INDEX_HTML.read_text(encoding="utf-8")
    defaults_source = _source_between(
        source,
        "function getDefaultBeverageOptions(productName)",
        "function roundPosMoney",
    )

    assert "ice_level:config.default_ice" in defaults_source
    assert "config.default_temperature==='hot'" not in defaults_source
    assert "allowed_ice" in defaults_source


def test_top3_preflights_capabilities_and_normalizes_existing_beverage_fields():
    source = INDEX_HTML.read_text(encoding="utf-8")
    bundle_source = _source_between(source, "function addBundleToCart", "async function getScript")

    assert "normalizeBeverageOptions" in bundle_source
    assert "alert(t('Beverage options unavailable'));return;" in bundle_source
    assert "discount_rate:0" in bundle_source


def test_existing_top3_beverage_lines_clear_bakery_discount_metadata_only():
    source = INDEX_HTML.read_text(encoding="utf-8")
    bundle_source = _source_between(source, "function addBundleToCart", "async function getScript")
    bakery_branch = _source_between(bundle_source, "if(isBread){", "}else{\n        // Coffee")
    beverage_branch = _source_between(bundle_source, "if(found2){", "}else{\n            cartItems.push")

    assert "existingBeverage.discount_rate=0;" in beverage_branch
    assert "delete existingBeverage.discount_source;" in beverage_branch
    assert "delete existingBeverage.discount_strategy;" in beverage_branch
    assert "delete existingBeverage.discount_reason;" in beverage_branch
    assert "cartItems[j].discount_rate=dynamicDr;" in bakery_branch
    assert "cartItems[j].discount_source=dynamicSource;" in bakery_branch
    assert "delete cartItems[j].discount_source;" not in bakery_branch


def test_beverage_pricing_ignores_stale_tray_discount_while_bakery_keeps_discount_path():
    values = _pricing_values()

    assert values["beverage"] == {"base": 18, "discount": 0, "final": 18}
    assert values["bakery"] == {"base": 10, "discount": 2, "final": 8}


def test_frontend_menu_prices_update_from_backend_for_bakery_and_beverages():
    assert _loaded_price_values() == {"bakery": 11.5, "beverage": 19.5, "menu": 19.5}


def test_customization_surcharge_uses_cny_one_decimal_in_modal_and_translations():
    source = INDEX_HTML.read_text(encoding="utf-8")
    modal_source = _source_between(source, "<!-- Beverage Customization Modal -->", "<!-- Swap Modal -->")

    assert 'data-i18n="Large +&#165;3.0">Large +&#165;3.0' in modal_source
    assert "'Large +\\u00A53.0':'Large +\\u00A53.0'" in source
    assert "'Large +\\u00A53.0':'\\u5927\\u676f +\\u00A53.0'" in source
    assert "Large +RM3" not in source


def test_order_history_and_view_receipt_use_one_decimal_pos_money():
    source = INDEX_HTML.read_text(encoding="utf-8")
    renderer_source = _source_between(source, "function loadRecentOrders", "function setDineType")

    assert "roundPosMoney(o.total_amount).toFixed(1)" in renderer_source
    assert "roundPosMoney(it.line_total).toFixed(1)" in renderer_source
    assert "roundPosMoney(d.subtotal).toFixed(1)" in renderer_source
    assert "roundPosMoney(d.discount).toFixed(1)" in renderer_source
    assert "roundPosMoney(d.total).toFixed(1)" in renderer_source
    assert ".toFixed(2)" not in renderer_source


def test_beverage_customization_fails_closed_when_capabilities_are_unavailable():
    source = INDEX_HTML.read_text(encoding="utf-8")
    modal_source = _source_between(source, "function openCoffeeModal", "function selectCoffeeOption")

    assert "normalizeBeverageOptions(key,editItem||{})" in modal_source
    assert "alert(t('Beverage options unavailable'));return;" in modal_source


def test_top3_rationale_savings_branch_uses_shared_one_decimal_rounding():
    source = INDEX_HTML.read_text(encoding="utf-8")
    rationale_source = _source_between(source, "async function getScript", "function businessEventLabel")
    branch_start = rationale_source.index("if(b.savings>=2)")
    branch_end = rationale_source.index("if(pctMap.inventory>=15)", branch_start)
    savings_branch = rationale_source[branch_start:branch_end]

    assert "roundPosMoney(b.savings).toFixed(1)" in savings_branch
    assert ".toFixed(0)" not in savings_branch


def test_pos_and_top3_money_uses_one_decimal_while_revenue_keeps_two_decimals():
    source = INDEX_HTML.read_text(encoding="utf-8")
    bundle_cards = _source_between(source, "// Bundle", "// Coffee buttons")
    rationale_source = _source_between(source, "async function getScript", "function businessEventLabel")
    payment_source = _source_between(source, "function completePayment", "async function generateBundle")
    revenue_source = _source_between(source, "function loadRevenueData", "function onRevenueDateChange")

    assert "function roundPosMoney" in source
    assert "roundPosMoney(COFFEE_PRICES[cd.key]!==undefined?COFFEE_PRICES[cd.key]:cd.price).toFixed(1)" in source
    assert "t('Save RM')" not in bundle_cards
    assert "t('Save:')+' \\u00A5'+roundPosMoney(b.savings).toFixed(1)" in bundle_cards
    assert "roundPosMoney(b.savings||0).toFixed(1)" in rationale_source
    assert ".toFixed(2)" not in rationale_source
    assert "roundPosMoney(pr*item.quantity).toFixed(1)" in source
    assert "_payTotal=roundPosMoney(total)" in payment_source
    assert "_payTotal.toFixed(1)" in payment_source
    assert "roundPosMoney(it.line_total||0).toFixed(1)" in payment_source
    assert ".toFixed(2)" not in payment_source
    assert "m.avg_order.toFixed(2)" in revenue_source
    assert "r.revenue.toFixed(2)" in source


def test_active_checkout_path_includes_complete_beverage_options():
    payloads = _checkout_payload_values()

    expected = {"size": "large", "temperature": "cold", "sugar": "less", "ice_level": "less"}
    assert payloads["paymentTotal"] == 21
    assert {key: payloads["payment"][key] for key in expected} == expected
    for name in ("paymentRejected", "paymentTypeRejected"):
        assert payloads[name] == {
            "fetches": 0,
            "unchanged": True,
            "alert": "Beverage options unavailable",
        }


def test_checkout_surfaces_backend_validation_detail_without_clearing_cart():
    result = _checkout_error_value()

    assert result["alert"] == "Payment Failed: Invalid sugar for latte: sweet"
    assert result["cart"][0]["product_name"] == "latte"


def test_receipt_and_history_format_persisted_beverage_aliases_with_pos_money():
    displays = _receipt_display_values()

    for name in ("receipt", "history", "fallback"):
        display = displays[name]
        assert "Cold / Less Sweet / Less Ice" in display
        assert "21.2" in display
        assert "21.24" not in display


def test_receipt_fallback_keeps_structured_beverage_name_and_formats_once():
    fallback = _receipt_display_values()["fallback"]

    assert 'item-name">latte<div' in fallback
    assert fallback.count("Large / Cold / Less Sweet / Less Ice") == 1
    assert "Large latte (" not in fallback
    assert "21.0" in fallback
    assert "18.0" not in fallback


def test_legacy_receipt_uses_capability_defaults_or_omits_incomplete_summary():
    displays = _receipt_display_values()

    assert "Hot / Normal Sweet" in displays["legacyDefaults"]
    assert "Cold" not in displays["legacyDefaults"]
    assert "Hot /" not in displays["legacyOmitted"]
    assert "Cold /" not in displays["legacyOmitted"]


def test_receipt_node_harness_injects_each_formatter_once():
    test_source = Path(__file__).read_text(encoding="utf-8")

    assert (
        '"function formatCoffeeDetails",\n        "function formatReceiptBeverageDetails",'
        in test_source
    )


def test_beverage_cart_edit_routes_and_prefills_only_the_beverage_modal():
    result = _run_beverage_behavior(
        r'''
beverageOptionMap.latte = config({
  default_temperature: 'cold', default_sugar: 'less', default_ice: 'none',
  allowed_temperatures: ['cold']
});
cartItems = [{product_name: 'latte', quantity: 2, size: ' Large ', temperature: ' COLD ', sugar: ' LESS ', ice_level: ' NONE '}];
editCartItem(0);
expect(document.getElementById('coffee-modal').classList.contains('show'), 'beverage modal did not open');
expect(_coffeeModalEditIndex === 0, 'edit index was not retained');
expect(_coffeeModalSize === 'large' && _coffeeModalTemp === 'cold' && _coffeeModalSugar === 'less' && _coffeeModalIce === 'none', 'cart options were not prefilled');
expect(document.getElementById('edit-modal').classList.contains('show') === false, 'generic editor opened for beverage');
console.log(JSON.stringify({modal: document.getElementById('coffee-modal').classList.contains('show')}));
'''
    )

    assert result == {"modal": True}


def test_beverage_options_filter_reject_unsupported_values_and_transition_hot_and_cold_ice():
    result = _run_beverage_behavior(
        r'''
beverageOptionMap.latte = config({
  default_temperature: 'cold', default_ice: 'normal', allowed_temperatures: ['hot', 'cold'], allowed_ice: ['normal', 'none']
});
openCoffeeModal('latte');
expect(buttons.ice[1].style.display === 'none' && buttons.ice[1].disabled, 'unsupported ice control was available');
selectCoffeeOption('ice', 'less');
expect(_coffeeModalIce === 'normal', 'unsupported ice value was accepted');
selectCoffeeOption('temp', 'hot');
expect(_coffeeModalTemp === 'hot' && _coffeeModalIce === 'none', 'hot selection did not force no ice');
_coffeeModalIce = 'warm';
selectCoffeeOption('temp', 'cold');
expect(_coffeeModalTemp === 'cold' && _coffeeModalIce === 'normal', 'cold selection did not restore the configured valid ice');
console.log(JSON.stringify({ice: _coffeeModalIce, unsupportedHidden: buttons.ice[1].style.display === 'none'}));
'''
    )

    assert result == {"ice": "normal", "unsupportedHidden": True}


def test_unavailable_or_malformed_capabilities_leave_cart_unchanged_before_open_or_confirm():
    result = _run_beverage_behavior(
        r'''
cartItems = [{product_name: 'latte', quantity: 1, size: 'regular', temperature: 'hot', sugar: 'normal', ice_level: 'none', discount_rate: 0.4, discount_source: 'bakery'}];
var original = JSON.stringify(cartItems);
openCoffeeModal('latte', 0);
expect(JSON.stringify(cartItems) === original, 'unavailable open mutated the cart');
expect(document.getElementById('coffee-modal').classList.contains('show') === false, 'unavailable capability opened the modal');
expect(alerts[0] === 'Beverage options unavailable', 'unavailable capability did not show the unavailable message');
beverageOptionMap.latte = config({default_temperature: 'warm', allowed_temperatures: ['warm']});
openCoffeeModal('latte', 0);
expect(JSON.stringify(cartItems) === original, 'malformed open mutated the cart');
expect(document.getElementById('coffee-modal').classList.contains('show') === false, 'malformed capability opened the modal');
expect(alerts[1] === 'Beverage options unavailable', 'malformed capability did not show the unavailable message');
_coffeeModalKey = 'latte';
_coffeeModalEditIndex = 0;
_coffeeModalTemp = 'warm';
confirmCoffeeAdd();
expect(JSON.stringify(cartItems) === original, 'malformed confirm mutated the cart');
console.log(JSON.stringify({alerts: alerts.length, unchanged: JSON.stringify(cartItems) === original}));
'''
    )

    assert result == {"alerts": 3, "unchanged": True}


def test_top3_add_normalizes_legacy_values_and_fails_closed_before_any_cart_mutation():
    result = _run_bundle_behavior(
        r'''
bundleRecs = [{items: ['croissant', 'latte'], discount_rate: 0.2, discount_source: 's5'}];
cartItems = [
  {product_name: 'croissant', quantity: 3, tray_color: 'yellow'},
  {product_name: 'latte', quantity: 1, size: ' Large ', temperature: ' COLD ', sugar: ' LESS ', ice_level: ' LESS '}
];
addBundleToCart(0);
var latte = cartItems.filter(function(item) { return item.product_name === 'latte'; })[0];
expect(latte.size === 'large' && latte.temperature === 'cold' && latte.sugar === 'less' && latte.ice_level === 'less', 'legacy values were not canonicalized');

bundleRecs = [{items: ['croissant', 'latte'], discount_rate: 0.2, discount_source: 's5'}];
cartItems = [
  {product_name: 'croissant', quantity: 4, tray_color: 'yellow'},
  {product_name: 'latte', quantity: 1, size: {}, temperature: 'cold', sugar: 'less', ice_level: 'less'}
];
var original = JSON.stringify(cartItems);
var rendersBefore = renderCount;
addBundleToCart(0);
expect(JSON.stringify(cartItems) === original, 'malformed preflight mutated the cart');
expect(renderCount === rendersBefore, 'malformed preflight rendered a mutation');
expect(alerts[alerts.length - 1] === 'Beverage options unavailable', 'malformed preflight did not alert');
console.log(JSON.stringify({canonical: latte, unchanged: JSON.stringify(cartItems) === original}));
'''
    )

    assert result["canonical"]["size"] == "large"
    assert result["unchanged"] is True


def test_residual_frontend_checkout_defaults_and_translation_are_removed():
    source = INDEX_HTML.read_text(encoding="utf-8")
    drinks = _source_between(source, "var COFFEE_DRINKS", "async function doLogin")

    assert "async function checkout()" not in source
    assert "'Save RM':" not in source
    assert "default_size" not in drinks
    assert "default_temp" not in drinks
    assert "default_sugar" not in drinks
    assert "default_ice" not in drinks


def test_restored_session_refreshes_database_prices_before_rendering_saved_panel():
    source = INDEX_HTML.read_text(encoding="utf-8")
    restore = _source_between(source, "if(token&&role){", "var cartItems=[]")

    assert "loadPrices()" in restore
    assert restore.index("loadPrices()") < restore.index("showPanel(savedPanel)")


def test_beverage_edit_merges_only_identical_signatures_and_clears_discount_metadata():
    result = _run_beverage_behavior(
        r'''
beverageOptionMap.latte = config({default_temperature: 'cold', default_ice: 'none'});
cartItems = [
  {product_name: 'latte', quantity: 2, size: 'large', temperature: 'cold', sugar: 'less', ice_level: 'none'},
  {product_name: 'latte', quantity: 3, size: 'regular', temperature: 'hot', sugar: 'normal', ice_level: 'none', discount_rate: 0.4, discount_source: 'bakery', discount_strategy: 'clearance', discount_reason: 'stale'}
];
openCoffeeModal('latte', 1);
selectCoffeeOption('size', 'large');
selectCoffeeOption('temp', 'cold');
selectCoffeeOption('sugar', 'less');
selectCoffeeOption('ice', 'none');
confirmCoffeeAdd();
expect(cartItems.length === 1 && cartItems[0].quantity === 5, 'matching signatures did not merge');
expect(cartItems[0].discount_rate === 0 && !('discount_source' in cartItems[0]) && !('discount_strategy' in cartItems[0]) && !('discount_reason' in cartItems[0]), 'beverage metadata was not cleared');
cartItems = [
  {product_name: 'latte', quantity: 2, size: 'large', temperature: 'cold', sugar: 'normal', ice_level: 'none'},
  {product_name: 'latte', quantity: 3, size: 'regular', temperature: 'hot', sugar: 'normal', ice_level: 'none'}
];
openCoffeeModal('latte', 1);
selectCoffeeOption('temp', 'cold');
selectCoffeeOption('ice', 'none');
confirmCoffeeAdd();
expect(cartItems.length === 2, 'different signatures merged');
console.log(JSON.stringify({mergedQuantity: 5, separateLines: cartItems.length}));
'''
    )

    assert result == {"mergedQuantity": 5, "separateLines": 2}


def test_generic_bakery_edit_excludes_beverages_rejects_tampering_and_preserves_bakery_discount():
    result = _run_beverage_behavior(
        r'''
cartItems = [{product_name: 'croissant', quantity: 1, tray_color: 'yellow', freshness: 'Day-1', discount_rate: 0.3, discount_source: 'dynamic', discount_strategy: 'clearance', discount_reason: 'freshness'}];
var original = JSON.stringify(cartItems[0]);
editCartItem(0);
expect(document.getElementById('edit-name').innerHTML.indexOf('latte') === -1, 'generic bakery editor exposed beverages');
document.getElementById('edit-name').value = 'latte';
document.getElementById('edit-qty').value = '9';
document.getElementById('edit-freshness').value = 'auto';
saveCartItem();
expect(JSON.stringify(cartItems[0]) === original, 'tampered beverage selection changed bakery line');
expect(alerts[0] === 'Beverage options unavailable', 'tampered selection did not show the unavailable message');
document.getElementById('edit-name').value = 'croissant';
document.getElementById('edit-qty').value = '2';
saveCartItem();
expect(cartItems[0].quantity === 2 && cartItems[0].discount_rate === 0.3 && cartItems[0].discount_source === 'dynamic', 'generic bakery edit lost discount metadata');
console.log(JSON.stringify({quantity: cartItems[0].quantity, discount: cartItems[0].discount_rate}));
'''
    )

    assert result == {"quantity": 2, "discount": 0.3}


def test_refund_ui_records_reason_and_never_promises_restocking():
    source = INDEX_HTML.read_text(encoding="utf-8")
    refund_source = _source_between(
        source,
        "function refundOrder(ticketId)",
        "function viewReceipt(ticketId)",
    )

    assert "prompt(t('Refund reason prompt')" in refund_source
    assert "JSON.stringify({ticket_id:ticketId,reason:reason})" in refund_source
    assert "Return recorded as non-sellable." in refund_source
    assert "loadStock()" in refund_source
    assert "Items restocked" not in refund_source
    assert "This will restock all items" not in source


def test_mobile_header_keeps_navigation_and_account_controls_separate():
    source = INDEX_HTML.read_text(encoding="utf-8")
    mobile_css = _source_between(
        source,
        "@media (max-width:600px)",
        "</style>",
    )

    assert ".topnav{flex-wrap:wrap" in mobile_css
    assert ".topnav nav{order:2;flex-basis:100%" in mobile_css
    assert "#user-display{display:none}" in mobile_css


def test_legacy_s5_chat_surface_is_not_present_in_the_frontend():
    source = INDEX_HTML.read_text(encoding="utf-8")

    assert not (INDEX_HTML.parent / "_core.js").exists()
    assert "function renderAgent" not in source
    assert "function runAgentQuery" not in source
    assert "function runQuickAgent" not in source
    assert "S5_API+'/analyze'" not in source
    assert "agentState" not in source
    for legacy_key in (
        "AI Store Manager",
        "restock_q",
        "waste_q",
        "schedule_q",
        "audit_q",
        "Type a question",
        "Enter a question",
        "Agent Deliberation",
        "Plan Options",
        "Forecast Drivers",
    ):
        assert f"'{legacy_key}':" not in source
