# Staffing Agent - reads S3 schedule + role counts
# Phase 3: dynamic confidence based on schedule data quality.
import httpx, logging, time
from typing import Dict, Any
from .base import BaseAgent
from s5_config.settings import S3_SCHEDULE_URL

logger = logging.getLogger("s5.agent.staffing")


class StaffingAgent(BaseAgent):
    def __init__(self):
        super().__init__("staffing")
        self._fetch_ok = False

    async def fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._fetch_ok = False
        try:
            today = params.get("date", time.strftime("%Y-%m-%d"))
            url = f"{S3_SCHEDULE_URL}?date={today}&days=1"
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=10)
                resp.raise_for_status()
                self._fetch_ok = True
                return resp.json()
        except Exception as e:
            logger.warning("Staffing fetch failed: %s", e)
            return {}

    def analyze(self, raw: Dict[str, Any], params: Dict[str, Any],
                history: str = "", key_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        # Use employee_summary from S3 for accurate hours per employee
        emp_summary = raw.get("employee_summary", {})
        role_counts = {}
        role_hours = {}
        for eid, info in emp_summary.items():
            role = info.get("role", "unknown")
            hours = info.get("hours", 0)
            role_counts[role] = role_counts.get(role, 0) + 1
            role_hours[role] = role_hours.get(role, 0) + hours

        bakers = role_counts.get("baker", 0)
        baristas = role_counts.get("barista", 0)
        cashiers = role_counts.get("cashier", 0)
        baker_hours = int(role_hours.get("baker", 0))

        # Dynamic confidence
        has_today = len(emp_summary) > 0
        has_required = bakers > 0 and cashiers > 0
        if self._fetch_ok and has_today and has_required:
            confidence = 0.95
        elif self._fetch_ok and has_today:
            confidence = 0.70
        elif self._fetch_ok:
            confidence = 0.40
        else:
            confidence = 0.10

        opinion = f"Staffing: {bakers} bakers ({baker_hours}h), {baristas} baristas, {cashiers} cashiers"

        constraints = []
        if bakers == 0:
            constraints.append("no bakers scheduled for today")
        if cashiers == 0:
            constraints.append("no cashiers scheduled for today")

        return {
            "opinion": opinion, "confidence": round(confidence, 2), "constraints": constraints,
            "data": {
                "bakers": bakers, "baker_hours": baker_hours,
                "baristas": baristas, "cashiers": cashiers,
                "role_counts": role_counts, "role_hours": role_hours,
            },
        }
