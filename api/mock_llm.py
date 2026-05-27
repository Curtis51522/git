"""Mock LLM responses for offline development and testing.
Allows full system integration testing without API costs.

Supports:
- DAG validation: valid DAGs + intentionally invalid DAGs for testing
- Verifier tiered testing: R1-R5 data shapes vs R6-R8 cross-table shapes
- Composer: summary + script modes
"""

import json
import time
from typing import Optional


# ======================================================================
# Mock DAGs
# ======================================================================

MOCK_DAG_VALID = {
    "dag": {
        "nodes": [
            {"id": "s2_forecast",  "type": "api", "endpoint": "/s2/forecast"},
            {"id": "s1_inventory", "type": "api", "endpoint": "/s1/batch_inventory"},
            {"id": "s3_capacity",  "type": "api", "endpoint": "/s3/capacity",
             "depends_on": ["s2_forecast"]},
        ],
        "edges": [
            ["s2_forecast", "s3_capacity"],
        ],
    }
}

# Intentionally invalid: missing endpoint
MOCK_DAG_NO_ENDPOINT = {
    "dag": {
        "nodes": [
            {"id": "bad_node", "type": "api"},
        ],
    }
}

# Intentionally invalid: depends_on references non-existent node
MOCK_DAG_BAD_DEP = {
    "dag": {
        "nodes": [
            {"id": "n1", "endpoint": "/s2/forecast"},
            {"id": "n2", "endpoint": "/s1/batch_inventory",
             "depends_on": ["n99"]},
        ],
    }
}

# Intentionally invalid: cycle
MOCK_DAG_CYCLE = {
    "dag": {
        "nodes": [
            {"id": "a", "endpoint": "/s2/forecast",    "depends_on": ["b"]},
            {"id": "b", "endpoint": "/s1/batch_inventory", "depends_on": ["a"]},
        ],
    }
}

# Intentionally invalid: endpoint not in whitelist
MOCK_DAG_BAD_ENDPOINT = {
    "dag": {
        "nodes": [
            {"id": "n1", "endpoint": "/evil/hack"},
        ],
    }
}


MOCK_DAG_MAP = {
    "stock_query":    MOCK_DAG_VALID,
    "waste_analysis": MOCK_DAG_VALID,
    "promo_eval":     MOCK_DAG_VALID,
    "schedule_audit": MOCK_DAG_VALID,
    # Special test keys
    "test_no_endpoint":  MOCK_DAG_NO_ENDPOINT,
    "test_bad_dep":      MOCK_DAG_BAD_DEP,
    "test_cycle":        MOCK_DAG_CYCLE,
    "test_bad_endpoint": MOCK_DAG_BAD_ENDPOINT,
}


# ======================================================================
# Mock Verifier responses
# ======================================================================

MOCK_VERIFIER_RESPONSE = {
    "passed": True,
    "audit_warnings": [],
    "rule_results": {
        "R1": True, "R2": True, "R3": True, "R4": True,
        "R5": True, "R6": True, "R7": True, "R8": True,
    },
}


# ======================================================================
# Mock Composer responses
# ======================================================================

MOCK_COMPOSER_RESPONSE = {
    "summary": (
        "Based on current inventory (12 units) and forecast (45 units), "
        "restock 33 Croissants. Production capacity is sufficient."
    ),
    "script": (
        "We have a great combo today - fresh Croissant paired with "
        "Iced Americano, perfect for this afternoon!"
    ),
}


# ======================================================================
# Mock functions
# ======================================================================

def mock_planner(intent: str, params: dict) -> dict:
    """Return a mock DAG.  Special intent keys trigger invalid DAGs for testing."""
    time.sleep(0.1)
    return MOCK_DAG_MAP.get(intent, MOCK_DAG_VALID)


def mock_verifier(result: dict) -> dict:
    time.sleep(0.1)
    return MOCK_VERIFIER_RESPONSE


def mock_composer(data: dict, mode: str = "summary") -> str:
    time.sleep(0.1)
    if mode == "script":
        return MOCK_COMPOSER_RESPONSE["script"]
    return MOCK_COMPOSER_RESPONSE["summary"]


def mock_intent_classifier(query: str) -> tuple:
    """Rule-based mock for offline testing."""
    keywords = {
        "stock":    "stock_query",
        "restock":  "stock_query",
        "waste":    "waste_analysis",
        "loss":     "waste_analysis",
        "promotion":"promo_eval",
        "discount": "promo_eval",
        "schedule": "schedule_audit",
        "shift":    "schedule_audit",
    }
    query_lower = query.lower()
    for keyword, intent in keywords.items():
        if keyword in query_lower:
            return intent, 0.95
    return "out_of_scope", 0.3


# ======================================================================
# Mock LLM class (for future real-API migration)
# ======================================================================

class MockLLM:
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock

    def call(self, model: str, prompt: str, **kwargs) -> str:
        if self.use_mock:
            return json.dumps(MOCK_DAG_VALID)
        raise NotImplementedError("Real API calls require API keys in .env")
