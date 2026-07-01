import os, sys, logging, json
from datetime import date as date_type

_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
from db.mysql_client import get_db

logger = logging.getLogger("s5.agent.attendance")

class AttendanceAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_attendance_log", description="Get today attendance records",
            parameters={"date": "string"}, primary=True, _handler=self._get_attendance))

    async def _get_attendance(self, date: str = ""):
        return _query_attendance_db(date)

    async def fetch(self, params):
        """Override: query DB directly to avoid async Tool pipeline issues."""
        date_str = ""
        if isinstance(params, dict):
            date_str = str(params.get("date", ""))
        data = _query_attendance_db(date_str)
        return {"success": True, "data": data, "tool": "attendance_db_direct"}

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {}) if "data" in raw else raw
        absent = data.get("absent", 0)
        total = data.get("on_time", 0) + absent

        if absent > 0:
            return AgentOpinion(agent=self.name, opinion=f"{absent}/{total} employees absent today",
                confidence=0.9, attribution={"metric": "attendance", "root_cause": "staff_absence", "deviation": abs(absent)})
        if total > 0:
            return AgentOpinion(agent=self.name, opinion=f"All {total} staff present",
                confidence=0.9, attribution={"metric": "attendance", "root_cause": "full_attendance", "deviation": 0})
        return AgentOpinion(agent=self.name, opinion="No attendance data available",
            confidence=0.5, attribution={"metric": "attendance", "root_cause": "no_data", "deviation": 0})


def _query_attendance_db(date_str=""):
    """Query attendance directly from MySQL."""
    try:
        db = get_db()
        cur = db.cursor(dictionary=True)
        if date_str:
            cur.execute("SELECT * FROM attendance_records WHERE date=%s", (date_str,))
        else:
            cur.execute("SELECT * FROM attendance_records WHERE date=CURDATE()")
        records = cur.fetchall()
        present = 0
        absent = 0
        result_records = []
        for r in records:
            rec = {
                "id": str(r.get("emp_id", "")),
                "name": str(r.get("emp_name", "")),
                "role": str(r.get("emp_role", "")),
                "status": str(r.get("status", "")),
                "punch_in": str(r.get("punch_in", "")),
                "punch_out": str(r.get("punch_out", "")),
            }
            result_records.append(rec)
            has_punch = bool(r.get("punch_in"))
            status = str(r.get("status", ""))
            if has_punch or status in ("on_time", "late", "present"):
                present += 1
            if not has_punch and status == "absent":
                absent += 1
        return {"records": result_records, "on_time": present, "absent": absent}
    except Exception as e:
        logger.warning("ATTEND_DB_ERROR: %s", e)
        return {"records": [], "on_time": 0, "absent": 0, "note": "db_error"}
