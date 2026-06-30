import os, sys, logging, httpx, json, urllib.request
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
logger = logging.getLogger("s5.agent.attendance")

class AttendanceAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_attendance_log", description="Get today attendance records",
            parameters={"date": "string"}, primary=True, _handler=self._get_attendance))
        self.tools.register(Tool(name="get_punctuality", description="Get punctuality stats",
            parameters={"date": "string"}, primary=False, fallback=True, _handler=self._get_punctuality))

    async def _get_attendance(self, date: str = ""):
        return _fetch_attendance_api(date)

    async def _get_punctuality(self, date: str = ""):
        return await self._get_attendance(date)

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {}) if "data" in raw else raw
        absent = data.get("absent", 0)
        total = data.get("on_time", 0) + absent
        is_fallback = data.get("note") == "api_unavailable"
        no_data = (absent == 0 and total == 0) or is_fallback

        if no_data:
            date = ""
            if isinstance(params, dict):
                date = str(params.get("date", ""))
            logger.info("ATTEND_ANALYZE: no_data, params_date=%s", repr(date))
            if date:
                fresh = _fetch_attendance_api(date)
                absent = fresh.get("absent", 0)
                total = fresh.get("on_time", 0) + absent

        if absent > 0:
            return AgentOpinion(agent=self.name, opinion=f"{absent}/{total} employees absent today",
                confidence=0.9, attribution={"metric": "attendance", "root_cause": "staff_absence", "deviation": abs(absent)})
        if total > 0:
            return AgentOpinion(agent=self.name, opinion=f"All {total} staff present",
                confidence=0.9, attribution={"metric": "attendance", "root_cause": "full_attendance", "deviation": 0})
        return AgentOpinion(agent=self.name, opinion="No attendance data available",
            confidence=0.5, attribution={"metric": "attendance", "root_cause": "no_data", "deviation": 0})


def _fetch_attendance_api(date=""):
    """Synchronous API call (usable from analyze method)"""
    url = "http://127.0.0.1:8002/s3/attendance"
    if date:
        url += f"?date={date}"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            d = json.loads(r.read())
            today = d.get("today_attendance", {})
            records = today.get("employees", [])
            if not records:
                records = d.get("attendance", []) or d.get("data", []) or d.get("records", [])
            if isinstance(records, list):
                present = sum(1 for r2 in records if r2.get("punch_in") or r2.get("status") in ("on_time", "late", "present"))
                absent = sum(1 for r2 in records if not r2.get("punch_in") and r2.get("status") == "absent")
                return {"records": records, "on_time": present, "absent": absent, "raw": d}
    except Exception as e:
        logger.warning("ATTEND_API_SYNC: error=%s", e)
    return {"records": [], "on_time": 0, "absent": 0, "note": "api_unavailable"}