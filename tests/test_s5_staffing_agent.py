import asyncio

from s5_agent.agents.staffing import StaffingAgent


class FakeResponse:
    status_code = 200

    def json(self):
        return {
            "today_attendance": {
                "employees": [
                    {"employee_id": "E001", "punch_in": "06:00"},
                    {"employee_id": "E002", "punch_in": None},
                ]
            }
        }


class FakeAsyncClient:
    urls = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, url):
        self.urls.append(url)
        return FakeResponse()


def test_staffing_attendance_tool_uses_s3_attendance_endpoint(monkeypatch):
    FakeAsyncClient.urls = []
    monkeypatch.setattr("s5_agent.agents.staffing.httpx.AsyncClient", FakeAsyncClient)
    agent = StaffingAgent("StaffingAgent")

    result = asyncio.run(agent._get_attendance("2026-07-09"))

    assert FakeAsyncClient.urls == ["http://127.0.0.1:8002/s3/attendance?date=2026-07-09"]
    assert result["total"] == 2
