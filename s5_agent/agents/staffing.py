import os, sys, logging, httpx
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
from s5_agent.s5_config.settings import THRESHOLDS
from db.mysql_client import get_db
logger = logging.getLogger("s5.agent.staffing")

class StaffingAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_schedule", description="Get today schedule",
            parameters={"date": "string"}, primary=True, _handler=self._get_schedule))
        self.tools.register(Tool(name="get_attendance", description="Get attendance vs schedule",
            parameters={"date": "string"}, primary=False, fallback=True, _handler=self._get_attendance))

    async def _get_schedule(self, date: str = ""):
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                url = f"http://127.0.0.1:8002/s3/schedule"
                if date: url += f"?date={date}"
                r = await c.get(url)
                if r.status_code == 200:
                    d = r.json()
                    shifts = d.get("schedule", []) or d.get("shifts", []) or d.get("data", [])
                    if isinstance(shifts, list):
                        total = len(shifts)
                        emp_ids = set(s.get("employee_id") or s.get("employee","") for s in shifts)
                        return {"shifts": shifts, "total_staff": len(emp_ids), "total_shifts": total,
                                "employees": list(emp_ids), "raw": d}
        except Exception as e:
            logger.warning("Schedule fetch failed: %s", e)
        return {"shifts": [], "total_staff": 0, "total_shifts": 0, "note": "api_unavailable"}

    async def fetch(self, params):
        date_str = ""
        if isinstance(params, dict):
            date_str = str(params.get("date", ""))
        schedule_data = await self._get_schedule(date_str)
        total_headcount = 0
        try:
            db = get_db()
            cur = db.cursor()
            cur.execute("SELECT COUNT(DISTINCT emp_id) FROM attendance_records")
            total_headcount = cur.fetchone()[0]
        except Exception as e:
            logger.warning("Staffing headcount query failed: %s", e)
        schedule_data["total_headcount"] = total_headcount
        return {"success": True, "data": schedule_data, "tool": "staffing_db"}

    async def _get_attendance(self, date: str = ""):
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                url = f"http://127.0.0.1:8002/attendance"
                if date: url += f"?date={date}"
                r = await c.get(url)
                if r.status_code == 200:
                    d = r.json()
                    records = d.get("attendance", []) or d.get("data", []) or d.get("records", [])
                    if isinstance(records, list):
                        present = sum(1 for r2 in records if r2.get("punch_in"))
                        return {"records": records, "present": present, "total": len(records), "raw": d}
        except Exception as e:
            logger.warning("Attendance fetch failed: %s", e)
        return {"records": [], "present": 0, "total": 0, "note": "api_unavailable"}

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {}) if "data" in raw else raw
        staff = data.get("total_staff", 0)
        headcount = data.get("total_headcount", 0)
        hc_context = f" of {headcount} total" if headcount > 0 else ""
        if staff < THRESHOLDS["staffing_min_heads"]:
            return AgentOpinion(agent=self.name, opinion=f"Understaffed: only {staff}{hc_context} on shift",
                confidence=0.85, attribution={"metric": "staffing", "root_cause": "understaffed", "deviation": THRESHOLDS["staffing_min_heads"] - staff},
                recommendations=[{"action": "Consider calling backup staff", "urgency": "high", "projected_gain": 120, "ease": "low"}])
        return AgentOpinion(agent=self.name, opinion=f"{staff}{hc_context} staff on shift, adequate for current demand",
            confidence=0.85, attribution={"metric": "staffing", "root_cause": "adequate_staffing", "deviation": 0})
