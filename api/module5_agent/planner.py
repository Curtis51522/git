import httpx
import json
from collections import deque
from typing import Dict, List, Tuple, Optional

from api.mock_llm import mock_planner


# -- Whitelist of valid data-source endpoints for S5 tool-calling ----------
ENDPOINT_WHITELIST = frozenset({
    "/s1/batch_inventory",
    "/s2/forecast",
    "/s3/schedule",
    "/s3/capacity",
    "/db/sql_query",
})

# -- Pre-defined DAG templates for confidence-gated routing ----------
# When DistilBERT confidence >= 0.95, Planner routes directly to these
# validated templates, skipping LLM entirely (zero latency, zero token cost).
# Design informed by: ToolLLM (Qin et al., arXiv 2023) DFSDT method,
# Gorilla (Patil et al., arXiv 2023) retrieval-over-generation principle,
# and MetaGPT (Hong et al., ICLR 2024) pre-defined SOP efficiency.

INTENT_TEMPLATES: Dict[str, dict] = {
    "stock_query": {
        "nodes": [
            {"id": "step_1", "endpoint": "/s2/forecast"},
            {"id": "step_2", "endpoint": "/s1/batch_inventory", "depends_on": ["step_1"]},
        ]
    },
    "waste_analysis": {
        "nodes": [
            {"id": "step_1", "endpoint": "/s2/forecast"},
            {"id": "step_2", "endpoint": "/s1/batch_inventory", "depends_on": ["step_1"]},
        ]
    },
    "schedule_audit": {
        "nodes": [
            {"id": "step_1", "endpoint": "/s3/schedule"},
            {"id": "step_2", "endpoint": "/s1/batch_inventory", "depends_on": ["step_1"]},
        ]
    },
    "cross_source_audit": {
        "nodes": [
            {"id": "step_1", "endpoint": "/s2/forecast"},
            {"id": "step_2", "endpoint": "/s1/batch_inventory", "depends_on": ["step_1"]},
            {"id": "step_3", "endpoint": "/s3/schedule", "depends_on": ["step_2"]},
        ]
    },
    "promo_eval": {
        "nodes": [
            {"id": "step_1", "endpoint": "/s1/batch_inventory"},
        ]
    },
    "sales_query": {
        "nodes": [
            {"id": "step_1", "endpoint": "/s2/forecast"},
        ]
    },
}

CONFIDENCE_THRESHOLD = 0.95



class DAGValidationError(Exception):
    """Raised when a Planner-generated DAG fails structural checks."""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.details = details or {}


def validate_dag(dag: dict) -> Tuple[bool, List[str]]:
    """Deterministic structural validation of a Planner-generated DAG.

    Checks performed (zero LLM cost, O(N+E) time):
    1. Required fields on every node (id, endpoint)
    2. Every ``depends_on`` reference resolves to an existing node id
    3. No duplicate node ids
    4. Endpoint whitelist check
    5. Cycle detection via Kahn's algorithm (topological sort)

    Returns (is_valid, error_messages).
    """
    errors: List[str] = []
    nodes = dag.get("nodes", [])

    if not nodes:
        errors.append("DAG has zero nodes -- nothing to execute.")
        return False, errors

    node_ids: set = set()
    node_map: Dict[str, dict] = {}

    # ---- Pass 1: per-node checks ----------------------------------------
    for i, node in enumerate(nodes):
        nid = node.get("id")
        ep  = node.get("endpoint")

        if not nid or not isinstance(nid, str):
            errors.append(f"Node[{i}] missing or invalid 'id' field.")
            continue
        if not ep or not isinstance(ep, str):
            errors.append(f"Node '{nid}' missing or invalid 'endpoint' field.")
        elif ep not in ENDPOINT_WHITELIST:
            errors.append(
                f"Node '{nid}' endpoint '{ep}' is not in the S5 whitelist."
            )

        if nid in node_ids:
            errors.append(f"Duplicate node id '{nid}'.")
        else:
            node_ids.add(nid)
            node_map[nid] = node

    if errors:
        return False, errors

    # ---- Pass 2: depends_on reference check + edge collection -----------
    in_degree: Dict[str, int] = {nid: 0 for nid in node_ids}
    adjacency: Dict[str, List[str]] = {nid: [] for nid in node_ids}

    for nid, node in node_map.items():
        deps = node.get("depends_on", [])
        if not isinstance(deps, list):
            errors.append(
                f"Node '{nid}' has non-list 'depends_on': {type(deps).__name__}"
            )
            continue
        for dep in deps:
            if dep not in node_map:
                errors.append(
                    f"Node '{nid}' depends_on '{dep}' which does not exist."
                )
            else:
                adjacency[dep].append(nid)
                in_degree[nid] += 1

    if errors:
        return False, errors

    # ---- Pass 3: cycle detection (Kahn's algorithm) ---------------------
    queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
    sorted_count = 0

    while queue:
        current = queue.popleft()
        sorted_count += 1
        for neighbor in adjacency.get(current, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if sorted_count != len(node_ids):
        remaining = [nid for nid, deg in in_degree.items() if deg > 0]
        errors.append(
            f"Cycle detected -- {len(remaining)} node(s) unreachable "
            f"after topological sort: {remaining}"
        )
        return False, errors

    return True, []


class PlannerAgent:
    """Planner Agent - generates and validates Dependency-Aware DAGs.

    Workflow:
    1. LLM (DeepSeek-V4-Pro) generates a raw DAG from intent + params.
    2. ``validate_dag()`` runs deterministic structural checks (O(N+E)).
    3. On failure a ``DAGValidationError`` is raised so the router can
       trigger self-reflection / retry or fall back to a canned DAG.
    """

    MAX_RETRIES = 2

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock


    # ------------------------------------------------------------------
    def plan(self, intent: str, params: dict, confidence: float) -> dict:
        """Confidence-gated routing: template for high-confidence intents,
        LLM generation for low-confidence or out_of_scope.

        This is the core of the Planner meta-reasoning capability:
        - confidence >= 0.95 + template exists -> zero-latency DAG
        - otherwise -> DeepSeek LLM generation with retry + fallback
        """
        if confidence >= CONFIDENCE_THRESHOLD and intent in INTENT_TEMPLATES:
            dag = INTENT_TEMPLATES[intent]
            valid, errors = validate_dag(dag)
            if valid:
                return dag
        return self.generate_dag(intent, params)

    # ------------------------------------------------------------------
    def generate_dag(self, intent: str, params: dict) -> dict:
        """Public entry-point: generate *and validate* a DAG."""
        if self.use_mock:
            raw = mock_planner(intent, params)
            dag = raw.get("dag", raw)
        else:
            dag = self._call_llm_for_dag(intent, params)

        # -- Deterministic validation -----------------------------------
        valid, errors = validate_dag(dag)
        if not valid:
            raise DAGValidationError(
                f"Planner DAG failed structural validation: {'; '.join(errors)}",
                details={"intent": intent, "errors": errors, "dag": dag},
            )
        return dag

    # ------------------------------------------------------------------
    def _call_llm_for_dag(self, intent: str, params: dict) -> dict:
        """Call DeepSeek to generate a DAG for the given intent."""
        import json
        from api.module5_agent.llm_client import call_deepseek

        prompt = f"""Generate a DAG (Directed Acyclic Graph) of API calls for this task.

Intent: {intent}
Params: {json.dumps(params)}

Available endpoints (DO NOT include ?params in endpoint field, use exact path only):
- GET /s2/forecast  (sales forecast, params: days, product)
- GET /s1/batch_inventory  (current stock)
- GET /s3/schedule  (staff schedule, params: date)

Return ONLY valid JSON:
{{"nodes": [
    {{"id": "step_1", "endpoint": "/s2/forecast"}},
    {{"id": "step_2", "endpoint": "/s1/batch_inventory", "depends_on": ["step_1"]}}
]}}

For stock_query: call /s2/forecast and /s1/batch_inventory.
For waste_analysis: call /s2/forecast and /s1/batch_inventory.
For schedule_audit: call /s3/schedule and /s1/batch_inventory.
For promo_eval: call /s1/batch_inventory.
For cross_source_audit: call /s2/forecast and /s1/batch_inventory and /s3/schedule.
For sales_query or schedule_query: call the most relevant single endpoint.

Return ONLY the JSON, no other text."""

        system = "You are a DAG planner. Return only valid JSON with nodes array. Each node has id and endpoint. Optional depends_on list."
        response = call_deepseek(prompt, system, max_tokens=500)

        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            dag = json.loads(response)
            if "nodes" in dag:
                return dag
            # Maybe the response is just the nodes array
            if isinstance(dag, list):
                return {"nodes": dag}
            return dag
        except (json.JSONDecodeError, IndexError):
            # Fallback to canned DAG
            from api.mock_llm import mock_planner
            return mock_planner(intent, params).get("dag", mock_planner(intent, params))
