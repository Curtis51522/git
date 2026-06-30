import os, sys, httpx, logging
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)
from s5_agent.core.base import BaseAgent, AgentOpinion
from s5_agent.core.tool import Tool
logger = logging.getLogger("s5.agent.material_stock")

class MaterialStockAgent(BaseAgent):
    def _setup_tools(self):
        self.tools.register(Tool(name="get_stock", description="Get current stock for a material",
            parameters={"material_name": "string"}, primary=True, _handler=self._get_stock))
        self.tools.register(Tool(name="get_all_materials", description="Get all materials stock",
            parameters={}, primary=False, fallback=True, _handler=self._get_all_materials))

    async def _get_stock(self, material_name: str = ""):
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get("http://127.0.0.1:8002/s4/inventory/materials")
                if r.status_code == 200:
                    data = r.json()
                    mats = data.get("materials", [])
                    if material_name:
                        for m in mats:
                            if m.get("material_name") == material_name:
                                return m
                    return {"materials": mats}
        except Exception as e:
            logger.warning("Material fetch failed: %s", e)
        return {"materials": []}

    async def _get_all_materials(self):
        return await self._get_stock()

    def analyze(self, raw, params, context="", history="", key_metrics=None):
        data = raw.get("data", {})
        mats = data.get("materials", [])
        low = [m for m in mats if float(m.get("stock_quantity", 999)) <= float(m.get("reorder_point", 0))]
        if low:
            names = [m["material_name"] for m in low[:5]]
            return AgentOpinion(agent=self.name, opinion=f"Low stock: {', '.join(names)}",
                confidence=0.85, attribution={"metric": "material_stock", "root_cause": "low_stock", "deviation": -len(low)},
                recommendations=[{"action": f"Restock {n}", "urgency": "high", "projected_gain": 100, "ease": "high"} for n in names])
        return AgentOpinion(agent=self.name, opinion="All materials adequate", confidence=0.85,
            attribution={"metric": "material_stock", "root_cause": "adequate_stock", "deviation": 0})
