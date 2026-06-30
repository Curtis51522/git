import os, sys, httpx, logging
from collections import defaultdict
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
logger = logging.getLogger("s5.agent.hourly")

class HourlyPatternAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_hourly_sales", description="Get hourly sales breakdown",
            parameters={"date": "string"}, primary=True, _handler=self._get_hourly))

    async def _get_hourly(self, date: str = ""):
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                url = "http://127.0.0.1:8002/s4/revenue/hourly"
                if date: url += f"?date={date}"
                r = await c.get(url)
                if r.status_code == 200:
                    d = r.json()
                    data = d.get("data", {})
                    return {
                        "hours": data.get("hours", []),
                        "bread": data.get("bread", []),
                        "beverages": data.get("beverages", []),
                    }
        except Exception as e:
            logger.warning("Hourly fetch failed: %s", e)
        return {"hours": [], "bread": [], "beverages": []}

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {}) if "data" in raw else raw
        hours = data.get("hours", [])
        bread = data.get("bread", [])
        beverages = data.get("beverages", [])
        
        if not hours:
            return AgentOpinion(agent=self.name,
                opinion="No hourly sales data available for pattern analysis.",
                confidence=0.3,
                attribution={"metric": "hourly_pattern", "root_cause": "no_data", "deviation": 0})

        total = [bread[i] + beverages[i] for i in range(len(hours))]
        total_rev = sum(total)
        if total_rev == 0:
            return AgentOpinion(agent=self.name,
                opinion="No sales recorded across any hour today.",
                confidence=0.5,
                attribution={"metric": "hourly_pattern", "root_cause": "zero_sales", "deviation": 0})

        # Find peaks (hours contributing >12% of daily revenue)
        peaks = []
        dead_zones = []
        bev_spikes = []
        for i, h in enumerate(hours):
            pct = total[i] / total_rev * 100 if total_rev > 0 else 0
            if pct > 12:
                peaks.append((h, round(total[i]), round(pct)))
            if total[i] < 20 and i > 0:
                dead_zones.append(h)
            if beverages[i] > 80:
                bev_spikes.append((h, round(beverages[i])))

        # Build opinion
        parts = []
        if peaks:
            peak_str = ", ".join(f"{h} ({chr(165)}{v}, {p}%)" for h, v, p in peaks[:3])
            parts.append(f"Peak hours: {peak_str}")

        # Production timing: first peak determines bake deadline
        if peaks:
            first_peak_hour = int(peaks[0][0].split(":")[0])
            bake_by = first_peak_hour - 2
            if bake_by < 5: bake_by += 24
            parts.append(f"First bake deadline: {bake_by:02d}:00 to catch {peaks[0][0]} peak")

        if dead_zones:
            dz = dead_zones[:3]
            dz_str = ", ".join(dz)
            parts.append(f"Dead zones: {dz_str} (schedule prep/cleaning)")

        if bev_spikes:
            bs = bev_spikes[:2]
            bs_str = ", ".join(f"{h} ({chr(165)}{v})" for h, v in bs)
            parts.append(f"Beverage spikes: {bs_str}")

        opinion = " | ".join(parts) if parts else f"Steady sales across {len(hours)} hours, no strong patterns."

        recs = []
        if peaks:
            first_peak = peaks[0][0]
            fb = int(first_peak.split(":")[0]) - 2
            if fb < 5: fb += 24
            recs.append({
                "action": f"Complete first bake by {fb:02d}:00 to stock shelves before {first_peak} peak",
                "urgency": "high", "rationale": f"{first_peak} is the highest revenue hour"
            })
        if dead_zones:
            recs.append({
                "action": f"Schedule prep/cleaning during {dead_zones[0]}",
                "urgency": "medium", "rationale": f"Near-zero sales in this hour"
            })
        if bev_spikes:
            recs.append({
                "action": f"Staff barista fully on beverages from 30 min before {bev_spikes[0][0]}",
                "urgency": "medium", "rationale": f"Beverage revenue jumps to {chr(165)}{bev_spikes[0][1]} at {bev_spikes[0][0]}"
            })

        return AgentOpinion(agent=self.name, opinion=opinion, confidence=0.78,
            attribution={"metric": "hourly_pattern", "root_cause": "normal_pattern", "deviation": 0},
            evidence={"peaks": peaks, "dead_zones": dead_zones, "bev_spikes": bev_spikes},
            recommendations=recs)
