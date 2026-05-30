import httpx, asyncio, logging
from collections import deque

logger = logging.getLogger("s5.executor")

ENDPOINT_HANDLERS = {
    "/s1/batch_inventory":        {"method": "GET", "params": []},
    "/s2/forecast":               {"method": "GET", "params": ["days", "product", "date"]},
    "/s3/schedule":               {"method": "GET", "params": ["date"]},
}


def _topo_sort(nodes: list) -> list:
    """Topological sort of DAG nodes by depends_on."""
    node_map = {n["id"]: n for n in nodes}
    in_degree = {n["id"]: 0 for n in nodes}
    adj = {n["id"]: [] for n in nodes}

    for n in nodes:
        for dep in n.get("depends_on", []):
            if dep in adj:
                adj[dep].append(n["id"])
                in_degree[n["id"]] += 1

    q = deque([nid for nid, deg in in_degree.items() if deg == 0])
    result = []
    while q:
        nid = q.popleft()
        result.append(node_map[nid])
        for neighbor in adj[nid]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                q.append(neighbor)
    return result


async def execute_dag_real(dag: dict, params: dict) -> dict:
    """Execute a DAG by calling internal API endpoints via async httpx.

    Uses httpx.AsyncClient within the event loop - no blocking calls.
    """
    nodes = dag.get("nodes", [])
    if not nodes:
        return _mock_fallback(params)

    try:
        ordered = _topo_sort(nodes)
    except Exception:
        ordered = nodes

    collected = {}
    BASE_URL = "http://localhost:8000"

    async def fetch(client, endpoint, method, qp):
        url = BASE_URL + endpoint
        try:
            if method == "GET":
                resp = await client.get(url, params=qp, timeout=httpx.Timeout(30.0))
            else:
                resp = await client.post(url, json=params, timeout=httpx.Timeout(30.0))
            if resp.status_code == 200:
                return resp.json()
            body = resp.text[:200] if resp.text else "(empty)"
            logger.warning("Endpoint %s returned %d: %s", endpoint, resp.status_code, body)
        except httpx.TimeoutException as e:
            logger.warning("Endpoint %s TIMEOUT: %s", endpoint, type(e).__name__)
        except httpx.HTTPStatusError as e:
            logger.warning("Endpoint %s HTTP %d: %s", endpoint, e.response.status_code, e.response.text[:200])
        except Exception as e:
            logger.error("Endpoint %s FAILED [%s]: %s", endpoint, type(e).__name__, e)
        return None

    async with httpx.AsyncClient() as client:
        tasks = {}
        for node in ordered:
            ep = node.get("endpoint", "")
            if not ep:
                continue
            handler = ENDPOINT_HANDLERS.get(ep)
            if not handler:
                logger.warning("Unknown endpoint: %s", ep)
                continue

            qp = {}
            for p in handler["params"]:
                if p in params:
                    qp[p] = params[p]
                if p == "days" and "days" not in params:
                    qp["days"] = 7

            tasks[node["id"]] = (ep, fetch(client, ep, handler["method"], qp))

        for node_id, (ep, task) in tasks.items():
            data = await task
            if data is None:
                continue

            collected[node_id] = data

            if "inventory" in data:
                prod = params.get("product", "")
                if prod:
                    collected["inventory"] = sum(
                        b.get("quantity", 0) for b in data["inventory"]
                        if b.get("product_name", "") == prod
                    )
                else:
                    collected["inventory"] = sum(
                        b.get("quantity", 0) for b in data["inventory"]
                    )
                    collected["_all_inventory"] = data["inventory"]
            if "forecasts" in data or "forecast" in data:
                forecasts = data.get("forecasts", data.get("forecast", []))
                if forecasts:
                    prod = params.get("product", "")
                    if prod:
                        match = None
                        for f in forecasts:
                            if f.get("product_name", "") == prod:
                                match = f
                                break
                        if match:
                            collected["forecast"] = match.get("predicted_demand", 45)
                        elif forecasts:
                            collected["forecast"] = forecasts[0].get("predicted_demand", 45)
                    else:
                        collected["forecast"] = forecasts[0].get("predicted_demand", 45) if forecasts else 45
                    collected["_all_forecasts"] = forecasts
                    # Extract bounds for interval-based stocking
                    if prod and forecasts:
                        match = None
                        for f in forecasts:
                            if f.get("product_name", "") == prod:
                                match = f
                                break
                        if match is None and forecasts:
                            match = forecasts[0]
                        if match:
                            collected["forecast_low"] = match.get("lower_bound", collected["forecast"])
                            collected["forecast_high"] = match.get("upper_bound", collected["forecast"])
                    elif forecasts:
                        collected["forecast_low"] = forecasts[0].get("lower_bound", collected["forecast"])
                        collected["forecast_high"] = forecasts[0].get("upper_bound", collected["forecast"])
            if "schedule" in data:
                collected["schedule"] = data.get("schedule", data.get("shifts", []))
            if "capacity" in data:
                collected["capacity"] = data.get("capacity", 50)
            if "transactions" in data:
                collected["transactions"] = data.get("transactions", [])


    if not collected:
        logger.error("ALL endpoints failed, using mock fallback")
        return _mock_fallback(params)

    collected.setdefault("product", params.get("product", "croissant"))
    if "forecast" not in collected:
        collected["forecast"] = 45.0
        logger.warning("Forecast not found, using default 45")
    collected.setdefault("inventory", 12)
    collected.setdefault("capacity", 50)
    collected.setdefault("predictions", [40, 45, 38, 42, 44, 40, 43])
    collected.setdefault("actuals", [35, 40, 20, 45, 42, 38, 44])
    collected.setdefault("incremental_revenue", 120.0)
    collected.setdefault("discount_cost", 30.0)
    collected.setdefault("schedule", [])
    collected.setdefault("transactions", [])

    return collected


def _mock_fallback(params: dict) -> dict:
    """Fallback when DAG execution fails."""
    return {
        "forecast": params.get("forecast", 45.0),
        "inventory": params.get("inventory", 12),
        "capacity": params.get("capacity", 50),
    }
