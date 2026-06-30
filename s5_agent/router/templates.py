# s5_agent/router/templates.py
from s5_agent.core.dag import DAGTemplate, DAGNode

TEMPLATES = {
    "profit_root_cause": DAGTemplate(
        intent="profit_root_cause",
        description="Why is profit/revenue abnormal? (L1 card agents -> L2 cross-card -> L3 synthesis)",
        nodes=[
            # L1: Card-level analysis (parallel)
            DAGNode("TrendAgent", phase=1),
            DAGNode("FeatureSensitivityAgent", phase=1),
            DAGNode("HourlyPatternAgent", phase=1),
            DAGNode("ProductMixAgent", phase=1),
            DAGNode("ExternalFactorsAgent", phase=1),
            DAGNode("DemandAgent", phase=1),
            DAGNode("AttendanceAgent", phase=1),
            # L2: Cross-card analysis
            DAGNode("MetricConflictAgent", phase=2, dependencies=["TrendAgent","DemandAgent","ProductMixAgent","ProfitAgent"]),
            DAGNode("CausalChainAgent", phase=2, dependencies=["FeatureSensitivityAgent","ExternalFactorsAgent","TrendAgent"]),
            DAGNode("CrossRiskAgent", phase=2, dependencies=["ProductMixAgent","FeatureSensitivityAgent","ProfitAgent"]),
            # L3: Domain synthesis agents
            DAGNode("PricingAgent", phase=3, dependencies=["DemandAgent","MetricConflictAgent"]),
            DAGNode("StaffingAgent", phase=3, dependencies=["AttendanceAgent"]),
            DAGNode("ProfitAgent", phase=4, dependencies=["ExternalFactorsAgent","DemandAgent","PricingAgent","StaffingAgent","CausalChainAgent","CrossRiskAgent"]),
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
        description="Comprehensive system check with L2 cross-card analysis",
        nodes=[
            DAGNode("TrendAgent", phase=1),
            DAGNode("FeatureSensitivityAgent", phase=1),
            DAGNode("HourlyPatternAgent", phase=1),
            DAGNode("ProductMixAgent", phase=1),
            DAGNode("ExternalFactorsAgent", phase=1),
            DAGNode("DemandAgent", phase=1),
            DAGNode("MaterialStockAgent", phase=1),
            DAGNode("ProductStockAgent", phase=1),
            DAGNode("WastageAgent", phase=1),
            DAGNode("AttendanceAgent", phase=1),
            DAGNode("MetricConflictAgent", phase=2, dependencies=["TrendAgent","DemandAgent","ProductMixAgent","ProfitAgent"]),
            DAGNode("CausalChainAgent", phase=2, dependencies=["FeatureSensitivityAgent","ExternalFactorsAgent","TrendAgent"]),
            DAGNode("CrossRiskAgent", phase=2, dependencies=["ProductMixAgent","FeatureSensitivityAgent","ProfitAgent"]),
            DAGNode("ProductionAgent", phase=3, dependencies=["DemandAgent","MaterialStockAgent","WastageAgent"]),
            DAGNode("StaffingAgent", phase=3, dependencies=["AttendanceAgent"]),
            DAGNode("PricingAgent", phase=3, dependencies=["DemandAgent","MetricConflictAgent"]),
            DAGNode("PromoAgent", phase=3),
            DAGNode("YieldAgent", phase=3, dependencies=["ProductionAgent"]),
            DAGNode("ProfitAgent", phase=4, dependencies=["ProductionAgent","StaffingAgent","PricingAgent","PromoAgent","ExternalFactorsAgent","CausalChainAgent","CrossRiskAgent"]),
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
