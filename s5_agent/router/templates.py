# s5_agent/router/templates.py
from s5_agent.core.dag import DAGTemplate, DAGNode

TEMPLATES = {
    "profit_root_cause": DAGTemplate(
        intent="profit_root_cause",
        description="Why is profit/revenue abnormal?",
        nodes=[
            DAGNode("ExternalFactorsAgent", phase=1),
            DAGNode("DemandAgent", phase=1),
            DAGNode("AttendanceAgent", phase=1),
            DAGNode("PricingAgent", phase=2, dependencies=["DemandAgent"]),
            DAGNode("StaffingAgent", phase=2, dependencies=["AttendanceAgent"]),
            DAGNode("ProfitAgent", phase=3, dependencies=["ExternalFactorsAgent","DemandAgent","PricingAgent","StaffingAgent"]),
            DAGNode("YieldAgent", phase=4, dependencies=["ProfitAgent"]),
            DAGNode("WastageAgent", phase=4, dependencies=["ProfitAgent"]),
        ]
    ),
    "wastage_root_cause": DAGTemplate(
        intent="wastage_root_cause",
        description="Why is wastage/spoilage high?",
        nodes=[
            DAGNode("WastageAgent", phase=1),
            DAGNode("MaterialStockAgent", phase=1),
            DAGNode("ProductStockAgent", phase=1),
            DAGNode("ProductionAgent", phase=2, dependencies=["WastageAgent","MaterialStockAgent","ProductStockAgent"]),
            DAGNode("YieldAgent", phase=3, dependencies=["ProductionAgent"]),
        ]
    ),
    "production_advice": DAGTemplate(
        intent="production_advice",
        description="How much to bake tomorrow?",
        nodes=[
            DAGNode("DemandAgent", phase=1),
            DAGNode("ProductStockAgent", phase=1),
            DAGNode("MaterialStockAgent", phase=1),
            DAGNode("ProductionAgent", phase=2, dependencies=["DemandAgent","ProductStockAgent","MaterialStockAgent"]),
        ]
    ),
    "inventory_diagnosis": DAGTemplate(
        intent="inventory_diagnosis",
        description="Stock level assessment",
        nodes=[
            DAGNode("MaterialStockAgent", phase=1),
            DAGNode("ProductStockAgent", phase=1),
            DAGNode("WastageAgent", phase=1),
        ]
    ),
    "staffing_diagnosis": DAGTemplate(
        intent="staffing_diagnosis",
        description="Schedule/staffing issues",
        nodes=[
            DAGNode("AttendanceAgent", phase=1),
            DAGNode("StaffingAgent", phase=1),
            DAGNode("DemandAgent", phase=1),
        ]
    ),
    "full_diagnosis": DAGTemplate(
        intent="full_diagnosis",
        description="Comprehensive system check",
        nodes=[
            DAGNode("ExternalFactorsAgent", phase=1),
            DAGNode("DemandAgent", phase=1),
            DAGNode("MaterialStockAgent", phase=1),
            DAGNode("ProductStockAgent", phase=1),
            DAGNode("WastageAgent", phase=1),
            DAGNode("AttendanceAgent", phase=1),
            DAGNode("ProductionAgent", phase=2, dependencies=["DemandAgent","MaterialStockAgent","WastageAgent"]),
            DAGNode("StaffingAgent", phase=2, dependencies=["AttendanceAgent"]),
            DAGNode("PricingAgent", phase=2, dependencies=["DemandAgent"]),
            DAGNode("PromoAgent", phase=2),
            DAGNode("YieldAgent", phase=2, dependencies=["ProductionAgent"]),
            DAGNode("ProfitAgent", phase=3, dependencies=["ProductionAgent","StaffingAgent","PricingAgent","PromoAgent","ExternalFactorsAgent"]),
        ]
    ),
    "promo_evaluation": DAGTemplate(
        intent="promo_evaluation",
        description="Promotion effectiveness",
        nodes=[
            DAGNode("PromoAgent", phase=1),
            DAGNode("PricingAgent", phase=1),
            DAGNode("ProfitAgent", phase=2, dependencies=["PromoAgent","PricingAgent"]),
        ]
    ),
}

def get_template(intent: str) -> DAGTemplate:
    return TEMPLATES.get(intent)
