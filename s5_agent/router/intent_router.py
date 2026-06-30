# s5_agent/router/intent_router.py — Routing layer for S5 queries
# Phase 1: keyword+rule hybrid (paper phase: replace with fine-tuned DistilBERT)
import re, logging
from typing import Tuple

logger = logging.getLogger("s5.router")

INTENT_PATTERNS = {
    "profit_root_cause": {
        "en": [r"\bprofit\b", r"\brevenue\b", r"\bmargin\b", r"\bearn", r"\bmoney", r"\bincome\b"],
        "zh": [r"利润|赚钱|盈利|亏|收入|收益|利润率"],
    },
    "wastage_root_cause": {
        "en": [r"\bwast(e|age)\b", r"\bspoil", r"\bloss\b", r"\bscrap\b"],
        "zh": [r"损耗|浪费|损失|报废|消耗"],
    },
    "production_advice": {
        "en": [r"\bbake\b", r"\bproduc(e|tion)\b", r"\bmake\b", r"\bprepare\b", r"\btomorrow\b"],
        "zh": [r"烘焙|生产|做多少|准备|明天|制作|烤"],
    },
    "inventory_diagnosis": {
        "en": [r"\bstock\b", r"\binventory\b", r"\bsupply\b", r"\bmaterial\b", r"\braw\b"],
        "zh": [r"库存|原材料|物料|存货|备货|材料"],
    },
    "staffing_diagnosis": {
        "en": [r"\bstaff\b", r"\bschedule\b", r"\bshift\b", r"\bworker\b", r"\bemployee\b"],
        "zh": [r"排班|员工|人手|人力|值班|上班|人够"],
    },
    "full_diagnosis": {
        "en": [r"\bfull\b", r"\bcomprehensive\b", r"\ball\b", r"\boverall\b", r"\bcheck\b", r"\breport\b", r"\bstatus\b"],
        "zh": [r"全面|整体|总览|综合|检查|报告|全部|所有"],
    },
    "promo_evaluation": {
        "en": [r"\bpromo", r"\bdiscount\b", r"\bcampaign\b", r"\bmarketing\b", r"\boffer\b"],
        "zh": [r"促销|折扣|活动|优惠|推广|营销"],
    },
}

def route_intent(query: str) -> Tuple[str, float]:
    query_lower = query.lower().strip()
    scores = {}
    for intent, patterns in INTENT_PATTERNS.items():
        score = 0
        for lang_patterns in patterns.values():
            for pat in lang_patterns:
                if re.search(pat, query_lower):
                    score += 1
        scores[intent] = score
    if not scores or max(scores.values()) == 0:
        return "full_diagnosis", 0.3
    best = max(scores, key=scores.get)
    confidence = min(scores[best] / 5.0, 0.95)
    logger.info("Intent: %s (confidence=%.2f)", best, confidence)
    return best, confidence
