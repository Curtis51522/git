import asyncio
import copy
import importlib
import json
import subprocess
import sys
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from tests import test_s4_checkout_transactions as checkout_support


class EventConnection:
    def __init__(self):
        self.events = []
        self.commit_count = 0
        self.rollback_count = 0
        self.closed = False
        self.cursors = []
        self.state = {"recommendation_events": []}
        self._committed_state = copy.deepcopy(self.state)

    def cursor(self, dictionary=False):
        cursor = EventCursor(self, dictionary=dictionary)
        self.cursors.append(cursor)
        return cursor

    def commit(self):
        self.commit_count += 1
        self._committed_state = copy.deepcopy(self.state)
        self.events.append(("commit", None))

    def rollback(self):
        self.rollback_count += 1
        self.state = copy.deepcopy(self._committed_state)
        self.events.append(("rollback", None))

    def close(self):
        self.closed = True


class EventCursor:
    def __init__(self, db, dictionary=False):
        self.db = db
        self.dictionary = dictionary
        self.lastrowid = None
        self.rowcount = 0
        self.closed = False

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split())
        params = tuple(params or ())
        self.rowcount = 0
        self.db.events.append((normalized, params))

        if normalized.startswith("INSERT INTO recommendation_events"):
            event_id = len(self.db.state["recommendation_events"]) + 1
            (
                request_id,
                operation_date,
                shown_at,
                rank_position,
                bakery_product,
                beverage_product,
                score,
                discount_rate,
                discount_source,
                discount_strategy,
            ) = params
            self.db.state["recommendation_events"].append(
                {
                    "id": event_id,
                    "request_id": request_id,
                    "operation_date": operation_date,
                    "shown_at": shown_at,
                    "rank_position": rank_position,
                    "bakery_product": bakery_product,
                    "beverage_product": beverage_product,
                    "score": score,
                    "discount_rate": discount_rate,
                    "discount_source": discount_source,
                    "discount_strategy": discount_strategy,
                    "selected_at": None,
                    "purchased_order_id": None,
                }
            )
            self.lastrowid = event_id
            self.rowcount = 1
        elif normalized.startswith(
            "UPDATE recommendation_events SET selected_at = %s"
        ):
            selected_at, event_id = params
            for event in self.db.state["recommendation_events"]:
                if event["id"] == event_id and event["selected_at"] is None:
                    event["selected_at"] = selected_at
                    self.rowcount = 1
                    break
        elif normalized.startswith(
            "UPDATE recommendation_events SET purchased_order_id = %s"
        ):
            order_id = params[0]
            operation_date = params[-1]
            event_ids = set(params[1:-1])
            for event in self.db.state["recommendation_events"]:
                if (
                    event["id"] in event_ids
                    and event["selected_at"] is not None
                    and event["purchased_order_id"] is None
                    and event["operation_date"] == operation_date
                ):
                    event["purchased_order_id"] = order_id
                    self.rowcount += 1

    def close(self):
        self.closed = True


class ComboCursor(EventCursor):
    def __init__(self, db, dictionary=False):
        super().__init__(db, dictionary=dictionary)
        self.rows = []

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split())
        if normalized.startswith(
            "SELECT product_name FROM products WHERE category='bakery'"
        ):
            if self.db.fail_on_bakery_read:
                raise RuntimeError("bakery read failed")
            self.db.events.append((normalized, tuple(params or ())))
            self.rows = [("macaron",)]
            return
        if normalized.startswith(
            "SELECT product_name, selling_price FROM products WHERE category='beverage'"
        ):
            self.db.events.append((normalized, tuple(params or ())))
            self.rows = [
                ("latte", 18.0),
                ("americano", 14.0),
                ("mocha", 22.0),
            ]
            return
        if (
            normalized.startswith("INSERT INTO recommendation_events")
            and self.db.fail_on_event_insert
        ):
            raise RuntimeError("recommendation event write failed")
        super().execute(sql, params)

    def fetchall(self):
        return list(self.rows)


class ComboConnection(EventConnection):
    def __init__(
        self,
        batches=None,
        fail_on_event_insert=False,
        fail_on_bakery_read=False,
    ):
        super().__init__()
        self.batches = list(batches or [])
        self.fail_on_event_insert = fail_on_event_insert
        self.fail_on_bakery_read = fail_on_bakery_read

    def cursor(self, dictionary=False):
        cursor = ComboCursor(self, dictionary=dictionary)
        self.cursors.append(cursor)
        return cursor


class ComboQuery:
    def __init__(self, db):
        self.db = db

    def select(self, _columns):
        return self

    def gt(self, _column, _value):
        return self

    def neq(self, _column, _value):
        return self

    def execute(self):
        return types.SimpleNamespace(data=copy.deepcopy(self.db.batches))


class RecommendationCheckoutCursor(checkout_support.RecordingCursor):
    def execute(self, sql, params=None):
        normalized = " ".join(sql.split())
        if normalized.startswith(
            "UPDATE recommendation_events SET purchased_order_id = %s"
        ):
            params = tuple(params or ())
            self.db.events.append((normalized, params))
            self.rowcount = self.db.recommendation_link_rowcount
            if self.rowcount == 1:
                self.db.state["recommendation_events"][0][
                    "purchased_order_id"
                ] = params[0]
            return
        super().execute(sql, params)


class RecommendationCheckoutConnection(checkout_support.RecordingConnection):
    def __init__(self, recommendation_link_rowcount=1):
        super().__init__()
        self.recommendation_link_rowcount = recommendation_link_rowcount
        self.state["recommendation_events"] = [
            {
                "id": 1,
                "selected_at": datetime(2026, 7, 18, 9, 31),
                "purchased_order_id": None,
            }
        ]
        self.initial_state = copy.deepcopy(self.state)

    def cursor(self, dictionary=False):
        cursor = RecommendationCheckoutCursor(self, dictionary=dictionary)
        self.cursors.append(cursor)
        return cursor


@pytest.fixture
def bff_module():
    mysql_stub = types.ModuleType("db.mysql_client")
    mysql_stub.get_db = lambda **_kwargs: None
    mysql_stub.q = lambda _db, _table: None
    sys.modules.pop("api.module4_frontend.bff", None)
    with patch.dict(sys.modules, {"db.mysql_client": mysql_stub}):
        module = importlib.import_module("api.module4_frontend.bff")
    yield module
    sys.modules.pop("api.module4_frontend.bff", None)


def _recommendations():
    return [
        {
            "product_name": "macaron",
            "coffee_key": "latte",
            "total_score": 0.82,
            "discount_rate": 0.1,
            "discount_source": "discount_override",
            "discount_strategy": None,
        },
        {
            "product_name": "macaron",
            "coffee_key": "americano",
            "total_score": 0.78,
            "discount_rate": 0.0,
            "discount_source": "freshness",
            "discount_strategy": None,
        },
        {
            "product_name": "macaron",
            "coffee_key": "mocha",
            "total_score": 0.71,
            "discount_rate": 0.0,
            "discount_source": "freshness",
            "discount_strategy": None,
        },
    ]


def _combo_stubs(monkeypatch, bff_module, db):
    calls = []

    def get_db(*, autocommit=True):
        calls.append(autocommit)
        assert autocommit is False
        return db

    freshness_stub = types.ModuleType("api.freshness_service")
    freshness_stub.update_all_freshness = lambda: None
    freshness_stub.get_discount_rate = lambda _freshness: 0.0
    pairing_stub = types.ModuleType("api.module4_frontend.pairing_llm")
    pairing_stub.get_pairing_matrix = lambda: {
        "macaron": {"latte": 0.9, "americano": 0.8, "mocha": 0.7}
    }

    monkeypatch.setattr(bff_module, "get_db", get_db)
    monkeypatch.setattr(bff_module, "q", lambda connection, _table: ComboQuery(connection))
    monkeypatch.setattr(bff_module, "get_product_prices", lambda: {"macaron": 10.0})
    return calls, freshness_stub, pairing_stub


def _combo_payload():
    return {"items": [{"product_name": "macaron", "quantity": 1}]}


def _source_between(source, start_marker, end_marker):
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


def _run_frontend_recommendation_behavior(scenario):
    source = Path("api/module4_frontend/static/index.html").read_text(
        encoding="utf-8"
    )
    payment_source = _source_between(
        source,
        "function buildCheckoutItems",
        "function showReceipt",
    )
    clear_source = _source_between(source, "function clearCart", "function printReceipt")
    add_bundle_source = _source_between(
        source,
        "async function addBundleToCart",
        "async function getScript",
    )
    selection_state_source = _source_between(
        source,
        "function resetRecommendationSelectionState",
        "var PRODUCT_PRICES",
    )
    script = r'''
const PRODUCT_PRICES = {croissant: 10};
const COFFEE_PRICES = {latte: 18};
var cartItems = [];
var bundleRecs = [];
var selectedRecommendationEventIds = new Set();
var pendingRecommendationSelections = new Map();
var recommendationSelectionGeneration = 0;
var checkoutInFlight = false;
var detections = [];
var lastScanResult = null;
var dineType = 'dine_in';
var freshnessMap = {croissant: 'Day-1'};
var token = 'test-token';
var API = '';
var _payMethod = 'card';
var _payCashInput = 0;
var _payReceipt = null;
var _payTotal = 28;
var hitlLog = [];
var alerts = [];
var renderCount = 0;
var checkoutRequests = [];
var selectionRequests = [];
var selectionDeferred = null;
var checkoutDeferred = null;
var window = {_bakeryFreshness: {croissant: 'Day-1'}};

function deferred() {
  var resolve;
  var promise = new Promise(function(done) { resolve = done; });
  return {promise: promise, resolve: resolve};
}
function response(ok, body) {
  return {ok: ok, json: function() { return Promise.resolve(body); }};
}
function makeElement() {
  return {
    disabled: false,
    value: '',
    textContent: '',
    style: {},
    classList: {add: function() {}, remove: function() {}, contains: function() { return true; }}
  };
}
var elements = {};
var document = {
  getElementById: function(id) {
    if (!elements[id]) elements[id] = makeElement();
    return elements[id];
  }
};
function t(value) { return value; }
function alert(message) { alerts.push(message); }
function confirm() { return true; }
function renderPOS() { renderCount += 1; }
function loadStock() {}
function clearPanelCache() {}
function showReceipt() {}
function roundPosMoney(value) { return Math.round(Number(value) * 100) / 100; }
function normalizeBeverageOptions(_name, item) {
  return {
    size: item.size || 'regular',
    temperature: item.temperature || 'hot',
    sugar: item.sugar || 'normal',
    ice_level: item.ice_level || 'none'
  };
}
function fetch(url, init) {
  if (url.indexOf('/s4/combo/select') !== -1) {
    selectionRequests.push(JSON.parse(init.body));
    return selectionDeferred.promise;
  }
  if (url.indexOf('/s4/checkout/complete') !== -1) {
    checkoutRequests.push(JSON.parse(init.body));
    return checkoutDeferred ? checkoutDeferred.promise : Promise.resolve(
      response(true, {status: 'ok', receipt: {total: 28}, deducted: []})
    );
  }
  throw new Error('Unexpected URL: ' + url);
}
''' + selection_state_source + payment_source + clear_source + add_bundle_source + scenario
    result = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _persisted_events(bff_module, db, when=datetime(2026, 7, 18, 9, 30)):
    cursor = db.cursor()
    recommendations = _recommendations()
    try:
        bff_module._persist_recommendation_events(
            cursor,
            request_id="REC-20260718-093000-test",
            operation_now_value=when,
            recommendations=recommendations,
        )
        db.commit()
    finally:
        cursor.close()
    return recommendations


def test_schema_has_recommendation_event_contract():
    schema = Path("schema.sql").read_text(encoding="utf-8")

    assert "CREATE TABLE `recommendation_events`" in schema
    assert "`request_id` varchar(50) NOT NULL" in schema
    assert "`selected_at` datetime DEFAULT NULL" in schema
    assert "`purchased_order_id` int(11) DEFAULT NULL" in schema


def test_combo_route_commits_ids_on_every_returned_recommendation(
    monkeypatch, bff_module
):
    db = ComboConnection(
        batches=[
            {
                "product_name": "macaron",
                "quantity": 5,
                "freshness_status": "Fresh",
            }
        ]
    )
    calls, freshness_stub, pairing_stub = _combo_stubs(
        monkeypatch, bff_module, db
    )

    with patch.dict(
        sys.modules,
        {
            "api.freshness_service": freshness_stub,
            "api.module4_frontend.pairing_llm": pairing_stub,
        },
    ):
        result = asyncio.run(bff_module.get_combo(_combo_payload()))

    recommendations = result["recommendations"]
    assert len(recommendations) == 3
    assert [item["recommendation_event_id"] for item in recommendations] == [
        1,
        2,
        3,
    ]
    assert len(db.state["recommendation_events"]) == 3
    assert calls == [False]
    assert db.commit_count == 1
    assert db.rollback_count == 0
    assert db.closed
    assert all(cursor.closed for cursor in db.cursors)


def test_combo_route_persistence_failure_returns_500_and_no_events(
    monkeypatch, bff_module
):
    db = ComboConnection(
        batches=[
            {
                "product_name": "macaron",
                "quantity": 5,
                "freshness_status": "Fresh",
            }
        ],
        fail_on_event_insert=True,
    )
    _, freshness_stub, pairing_stub = _combo_stubs(monkeypatch, bff_module, db)

    with patch.dict(
        sys.modules,
        {
            "api.freshness_service": freshness_stub,
            "api.module4_frontend.pairing_llm": pairing_stub,
        },
    ):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(bff_module.get_combo(_combo_payload()))

    assert exc_info.value.status_code == 500
    assert "persistence failed" in str(exc_info.value.detail).lower()
    assert db.state["recommendation_events"] == []
    assert db.commit_count == 0
    assert db.rollback_count == 1
    assert db.closed
    assert all(cursor.closed for cursor in db.cursors)


def test_combo_route_no_inventory_closes_connection(monkeypatch, bff_module):
    db = ComboConnection(batches=[])
    _, freshness_stub, pairing_stub = _combo_stubs(monkeypatch, bff_module, db)

    with patch.dict(
        sys.modules,
        {
            "api.freshness_service": freshness_stub,
            "api.module4_frontend.pairing_llm": pairing_stub,
        },
    ):
        result = asyncio.run(bff_module.get_combo(_combo_payload()))

    assert result == {"status": "ok", "recommendations": []}
    assert db.closed
    assert all(cursor.closed for cursor in db.cursors)


def test_combo_route_read_exception_closes_connection_and_cursor(
    monkeypatch, bff_module
):
    db = ComboConnection(fail_on_bakery_read=True)
    _, freshness_stub, pairing_stub = _combo_stubs(monkeypatch, bff_module, db)

    with patch.dict(
        sys.modules,
        {
            "api.freshness_service": freshness_stub,
            "api.module4_frontend.pairing_llm": pairing_stub,
        },
    ):
        with pytest.raises(RuntimeError, match="bakery read failed"):
            asyncio.run(bff_module.get_combo(_combo_payload()))

    assert db.closed
    assert all(cursor.closed for cursor in db.cursors)


def test_checkout_route_links_selected_event_in_full_transaction(
    monkeypatch, bff_module
):
    db = RecommendationCheckoutConnection(recommendation_link_rowcount=1)
    _, yolo_stub, freshness_stub = checkout_support._checkout_dependencies(
        monkeypatch, bff_module, db
    )
    monkeypatch.setattr(
        bff_module,
        "q",
        lambda connection, table: checkout_support.RecordingQuery(
            connection, table
        ),
    )
    payload = checkout_support._checkout_payload()
    payload["recommendation_event_ids"] = [1]

    with patch.dict(
        sys.modules,
        {
            "api.module1_yolo": yolo_stub,
            "api.freshness_service": freshness_stub,
        },
    ):
        result = asyncio.run(bff_module.checkout_complete(payload))

    assert result["status"] == "ok"
    assert db.state["recommendation_events"][0]["purchased_order_id"] == 101
    assert len(db.state["orders"]) == 1
    assert len(db.state["order_items"]) == 2
    assert len(db.state["payments"]) == 1
    assert len(db.state["receipts"]) == 1
    link_index = next(
        index
        for index, event in enumerate(db.events)
        if str(event[0]).startswith(
            "UPDATE recommendation_events SET purchased_order_id"
        )
    )
    commit_index = next(
        index for index, event in enumerate(db.events) if event[0] == "commit"
    )
    assert link_index < commit_index
    assert db.commit_count == 1
    assert db.rollback_count == 0


def test_checkout_route_event_rowcount_mismatch_rolls_back_every_write(
    monkeypatch, bff_module
):
    db = RecommendationCheckoutConnection(recommendation_link_rowcount=0)
    _, yolo_stub, freshness_stub = checkout_support._checkout_dependencies(
        monkeypatch, bff_module, db
    )
    monkeypatch.setattr(
        bff_module,
        "q",
        lambda connection, table: checkout_support.RecordingQuery(
            connection, table
        ),
    )
    payload = checkout_support._checkout_payload()
    payload["recommendation_event_ids"] = [1]

    with patch.dict(
        sys.modules,
        {
            "api.module1_yolo": yolo_stub,
            "api.freshness_service": freshness_stub,
        },
    ):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(bff_module.checkout_complete(payload))

    assert exc_info.value.status_code == 409
    assert db.commit_count == 0
    assert db.rollback_count == 1
    assert db.state == db.initial_state
    assert db.state["recommendation_events"][0]["purchased_order_id"] is None


def test_frontend_blocks_checkout_until_selection_resolves():
    result = _run_frontend_recommendation_behavior(
        r'''
(async function() {
  selectionDeferred = deferred();
  cartItems = [{
    product_name: 'latte', quantity: 1, size: 'large', temperature: 'cold',
    sugar: 'less', ice_level: 'less', discount_rate: 0
  }];
  bundleRecs = [{
    items: ['croissant', 'latte'], recommendation_event_id: 41,
    discount_rate: 0.2, discount_source: 's5_dynamic',
    discount_strategy: 'clearance', discount_reason: 'test'
  }];
  var addPromise = addBundleToCart(0);
  await Promise.resolve();
  var pendingBeforeCheckout = pendingRecommendationSelections.size;
  await confirmPayment();
  var checkoutBeforeSelection = checkoutRequests.length;
  selectionDeferred.resolve(response(true, {status: 'ok', recommendation_event_id: 41}));
  await addPromise;
  console.log(JSON.stringify({
    pendingBeforeCheckout: pendingBeforeCheckout,
    checkoutBeforeSelection: checkoutBeforeSelection,
    selected: Array.from(selectedRecommendationEventIds),
    cartLength: cartItems.length,
    beverage: cartItems.filter(function(item) { return item.product_name === 'latte'; })[0],
    pendingAfterSelection: pendingRecommendationSelections.size
  }));
})().catch(function(error) { console.error(error); process.exit(1); });
'''
    )

    assert result["pendingBeforeCheckout"] == 1
    assert result["checkoutBeforeSelection"] == 0
    assert result["selected"] == [41]
    assert result["cartLength"] == 2
    assert result["beverage"]["size"] == "large"
    assert result["beverage"]["temperature"] == "cold"
    assert result["beverage"]["sugar"] == "less"
    assert result["beverage"]["ice_level"] == "less"
    assert result["pendingAfterSelection"] == 0


def test_frontend_failed_selection_rolls_back_bundle_addition():
    result = _run_frontend_recommendation_behavior(
        r'''
(async function() {
  selectionDeferred = deferred();
  cartItems = [{
    product_name: 'latte', quantity: 1, size: 'large', temperature: 'cold',
    sugar: 'less', ice_level: 'less', discount_rate: 0
  }];
  var originalCart = JSON.stringify(cartItems);
  bundleRecs = [{
    items: ['croissant', 'latte'], recommendation_event_id: 42,
    discount_rate: 0.2, discount_source: 's5_dynamic',
    discount_strategy: 'clearance', discount_reason: 'test'
  }];
  var addPromise = addBundleToCart(0);
  await Promise.resolve();
  selectionDeferred.resolve(response(false, {detail: 'selection rejected'}));
  await addPromise;
  console.log(JSON.stringify({
    cartRestored: JSON.stringify(cartItems) === originalCart,
    selected: Array.from(selectedRecommendationEventIds),
    pending: pendingRecommendationSelections.size,
    alert: alerts[alerts.length - 1]
  }));
})().catch(function(error) { console.error(error); process.exit(1); });
'''
    )

    assert result["cartRestored"] is True
    assert result["selected"] == []
    assert result["pending"] == 0
    assert "selection rejected" in result["alert"]


def test_frontend_late_selection_response_cannot_reach_later_order():
    result = _run_frontend_recommendation_behavior(
        r'''
(async function() {
  selectionDeferred = deferred();
  cartItems = [{product_name: 'latte', quantity: 1, size: 'large'}];
  bundleRecs = [{
    items: ['croissant', 'latte'], recommendation_event_id: 43,
    discount_rate: 0.2
  }];
  var addPromise = addBundleToCart(0);
  await Promise.resolve();
  clearCart();
  selectionDeferred.resolve(response(true, {status: 'ok', recommendation_event_id: 43}));
  await addPromise;
  console.log(JSON.stringify({
    cartLength: cartItems.length,
    selected: Array.from(selectedRecommendationEventIds),
    pending: pendingRecommendationSelections.size
  }));
})().catch(function(error) { console.error(error); process.exit(1); });
'''
    )

    assert result == {"cartLength": 0, "selected": [], "pending": 0}


def test_frontend_guards_repeated_checkout_submission():
    result = _run_frontend_recommendation_behavior(
        r'''
(async function() {
  checkoutDeferred = deferred();
  cartItems = [{
    product_name: 'croissant', quantity: 1, freshness: 'Fresh',
    discount_rate: 0.2, discount_source: 's5_dynamic',
    discount_strategy: 'clearance', discount_reason: 'test'
  }];
  selectedRecommendationEventIds.add(44);
  var first = confirmPayment();
  var second = confirmPayment();
  await Promise.resolve();
  var requestsWhilePending = checkoutRequests.length;
  checkoutDeferred.resolve(response(true, {status: 'ok', receipt: {total: 8}, deducted: []}));
  await Promise.all([first, second]);
  console.log(JSON.stringify({
    requestsWhilePending: requestsWhilePending,
    checkoutInFlight: checkoutInFlight,
    cartLength: cartItems.length,
    submittedIds: checkoutRequests[0].recommendation_event_ids
  }));
})().catch(function(error) { console.error(error); process.exit(1); });
'''
    )

    assert result["requestsWhilePending"] == 1
    assert result["checkoutInFlight"] is False
    assert result["cartLength"] == 0
    assert result["submittedIds"] == [44]


def test_displayed_recommendations_receive_persisted_event_ids(bff_module):
    db = EventConnection()
    recommendations = _persisted_events(bff_module, db)

    assert [item["recommendation_event_id"] for item in recommendations] == [
        1,
        2,
        3,
    ]
    assert all(
        event["selected_at"] is None
        and event["purchased_order_id"] is None
        for event in db.state["recommendation_events"]
    )


def test_selected_recommendation_remains_unpurchased_until_checkout(
    monkeypatch, bff_module
):
    db = EventConnection()
    _persisted_events(bff_module, db)
    monkeypatch.setattr(bff_module, "get_db", lambda **_kwargs: db)

    result = asyncio.run(bff_module.select_combo({"recommendation_event_id": 1}))

    assert result == {"status": "ok", "recommendation_event_id": 1}
    event = db.state["recommendation_events"][0]
    assert event["selected_at"] is not None
    assert event["purchased_order_id"] is None
    assert db.commit_count == 2


def test_duplicate_recommendation_selection_is_rejected(monkeypatch, bff_module):
    db = EventConnection()
    _persisted_events(bff_module, db)
    monkeypatch.setattr(bff_module, "get_db", lambda **_kwargs: db)

    asyncio.run(bff_module.select_combo({"recommendation_event_id": 1}))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(bff_module.select_combo({"recommendation_event_id": 1}))

    assert exc_info.value.status_code == 404
    assert "already selected" in str(exc_info.value.detail).lower()
    assert db.rollback_count == 1


def test_selected_recommendation_is_linked_to_the_checkout_order(bff_module):
    db = EventConnection()
    _persisted_events(bff_module, db)
    db.state["recommendation_events"][0]["selected_at"] = datetime(
        2026, 7, 18, 9, 31
    )
    db.commit()
    cursor = db.cursor()

    bff_module._link_recommendation_events(
        cursor,
        recommendation_event_ids=[1],
        order_id=101,
        operation_date=datetime(2026, 7, 18).date(),
    )
    db.commit()

    assert db.state["recommendation_events"][0]["purchased_order_id"] == 101


def test_purchase_rejects_event_from_another_operation_date(bff_module):
    db = EventConnection()
    _persisted_events(bff_module, db)
    db.state["recommendation_events"][0]["selected_at"] = datetime(
        2026, 7, 18, 9, 31
    )
    db.commit()
    cursor = db.cursor()

    with pytest.raises(HTTPException) as exc_info:
        bff_module._link_recommendation_events(
            cursor,
            recommendation_event_ids=[1],
            order_id=101,
            operation_date=datetime(2026, 7, 19).date(),
        )

    assert exc_info.value.status_code == 409
    assert db.state["recommendation_events"][0]["purchased_order_id"] is None


def test_failed_checkout_rolls_back_recommendation_purchase_link(bff_module):
    db = EventConnection()
    _persisted_events(bff_module, db)
    db.state["recommendation_events"][0]["selected_at"] = datetime(
        2026, 7, 18, 9, 31
    )
    db.commit()
    cursor = db.cursor()

    bff_module._link_recommendation_events(
        cursor,
        recommendation_event_ids=[1],
        order_id=101,
        operation_date=datetime(2026, 7, 18).date(),
    )
    db.rollback()

    assert db.state["recommendation_events"][0]["purchased_order_id"] is None
    source = Path("api/module4_frontend/bff.py").read_text(encoding="utf-8")
    checkout = source[source.index("async def checkout_complete") :]
    assert checkout.index("_link_recommendation_events(") < checkout.index("db.commit()")


def test_checkout_rejects_duplicate_event_ids_and_keeps_is_top3_independent():
    source = Path("api/module4_frontend/bff.py").read_text(encoding="utf-8")
    checkout = source[source.index("async def checkout_complete") :]

    assert "recommendation_event_ids" in checkout
    assert "len(set(recommendation_event_ids))" in checkout
    assert checkout.index("len(set(recommendation_event_ids))") < checkout.index(
        "db = get_db(autocommit=False)"
    )
    assert "is_top3" not in checkout


def test_frontend_tracks_selected_ids_idempotently_and_clears_after_checkout():
    source = Path("api/module4_frontend/static/index.html").read_text(
        encoding="utf-8"
    )
    checkout = source[source.index("async function confirmPayment") : source.index("function showReceipt")]
    add_bundle = source[source.index("async function addBundleToCart") : source.index("async function getScript")]
    clear_cart = source[source.index("function clearCart") : source.index("function printReceipt")]

    assert "selectedRecommendationEventIds=new Set()" in source
    assert "recommendation_event_id:rc.recommendation_event_id" in source
    assert "selectedRecommendationEventIds.has" in add_bundle
    assert "API+'/s4/combo/select'" in add_bundle
    assert (
        'var recommendationEventIds=typeof selectedRecommendationEventIds!=="undefined"'
        in checkout
    )
    assert "recommendation_event_ids:recommendationEventIds" in checkout
    assert "resetRecommendationSelectionState()" in checkout
    assert "resetRecommendationSelectionState()" in clear_cart
