# Staffing Agent — schedule + labor constraints
import httpx, logging
from typing import Dict, Any
from .base import BaseAgent
from s5_config.settings import S3_SCHEDULE_URL, S3_KPI_URL

logger = logging.getLogger("s5.agent.staffing")

class StaffingAgent(BaseAgent):
    def __init__(self):
        super().__init__("staffing")

    async def fetch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        target_date = params.get("date", "")
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{S3_SCHEDULE_URL}?date={target_date}&days=1", timeout=10)
            resp.raise_for_status()
            schedule_data = resp.json()
        return schedule_data

    def analyze(self, raw: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        shifts = raw.get("schedule", [])

        role_counts = {}
        role_hours = {}
        for s in shifts:
            role = s.get("role", "")
            role_counts[role] = role_counts.get(role, 0) + 1
            slot = s.get("time_slot", "09:00-14:00")
            if "-" in slot:
                parts = slot.split("-")
                try:
                    h = int(parts[1].split(":")[0]) - int(parts[0].split(":")[0])
                    role_hours[role] = role_hours.get(role, 0) + h
                except Exception:
                    role_hours[role] = role_hours.get(role, 0) + 5

        bakers = role_counts.get("baker", 0)
        baristas = role_counts.get("barista", 0)
        cashiers = role_counts.get("cashier", 0)
        cleaners = role_counts.get("cleaner", 0)
        baker_hours = role_hours.get("baker", 6)

        constraints = []
        if bakers == 0:
            constraints.append("no bakers scheduled — cannot produce")
        if cashiers == 0:
            constraints.append("no cashiers scheduled — cannot open")

        return {
            "opinion": f"Staffing: {bakers} bakers ({baker_hours}h), {baristas} baristas, {cashiers} cashiers, {cleaners} cleaners",
            "confidence": 0.95,
            "constraints": constraints,
            "data": {
                "bakers": bakers,
                "baker_hours": baker_hours,
                "baristas": baristas,
                "cashiers": cashiers,
                "cleaners": cleaners,
                "role_counts": role_counts,
                "role_hours": role_hours,
            },
        }
