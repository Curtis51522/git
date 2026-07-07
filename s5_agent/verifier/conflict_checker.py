from __future__ import annotations

from s5_agent.schemas.agent_output import AgentOutput


def find_conflicting_claims(outputs: list[AgentOutput]) -> list[str]:
    claims_by_agent = {output.agent_name: output.claim.lower() for output in outputs}
    attendance_claim = claims_by_agent.get("AttendanceAgent", "")
    staffing_claim = claims_by_agent.get("StaffingAgent", "")

    if "absent" in attendance_claim and "adequate" in staffing_claim:
        return ["AttendanceAgent conflicts with StaffingAgent"]

    return []
