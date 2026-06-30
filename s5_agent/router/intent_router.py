# s5_agent/router/intent_router.py
import os, re, logging, json
from typing import Tuple

logger = logging.getLogger("s5.router")

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intent_model")
_model = None
_tokenizer = None
_label_map = None

INTENT_PATTERNS = {
    "profit_root_cause": {"en": [r"\bprofit\b", r"\brevenue\b", r"\bmargin\b", r"\bearn", r"\bmoney"], "zh": [r"利润|赚钱|盈利|亏|收入"]},
    "wastage_root_cause": {"en": [r"\bwast(e|age)\b", r"\bspoil", r"\bloss\b"], "zh": [r"损耗|浪费|损失|报废"]},
    "production_advice": {"en": [r"\bbake\b", r"\bproduc", r"\btomorrow\b"], "zh": [r"烘焙|生产|做多少|明天|制作"]},
    "inventory_diagnosis": {"en": [r"\bstock\b", r"\binventory\b", r"\bmaterial\b"], "zh": [r"库存|原材料|物料|存货"]},
    "staffing_diagnosis": {"en": [r"\bstaff\b", r"\bschedule\b", r"\bshift\b"], "zh": [r"排班|员工|人手|人力|值班"]},
    "full_diagnosis": {"en": [r"\bfull\b", r"\boverall\b", r"\bcheck\b", r"\breport\b"], "zh": [r"全面|整体|综合|检查|报告"]},
    "promo_evaluation": {"en": [r"\bpromo", r"\bdiscount\b", r"\bcampaign\b"], "zh": [r"促销|折扣|活动|优惠"]},
}

def _load_model():
    global _model, _tokenizer, _label_map
    if _model is not None:
        return True
    try:
        if os.path.exists(_MODEL_PATH):
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            _tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH)
            _model = AutoModelForSequenceClassification.from_pretrained(_MODEL_PATH)
            with open(os.path.join(_MODEL_PATH, "label_map.json"), "r") as f:
                _label_map = json.load(f)
            logger.info("DistilBERT intent model loaded")
            return True
    except Exception as e:
        logger.warning("Failed to load DistilBERT model: %s, falling back to keyword router", e)
    return False

def route_intent(query: str):
    query_lower = query.lower().strip()
    if _load_model():
        try:
            import torch
            inputs = _tokenizer(query, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                outputs = _model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                pred_idx = torch.argmax(probs).item()
                confidence = float(probs[pred_idx])
                intent = _label_map["id2label"][str(pred_idx)]
                if confidence > 0.5:
                    return intent, confidence
        except Exception as e:
            logger.warning("Model inference failed: %s", e)
    scores = {}
    for intent, patterns in INTENT_PATTERNS.items():
        score = 0
        for lang_pats in patterns.values():
            for pat in lang_pats:
                if re.search(pat, query_lower):
                    score += 1
        scores[intent] = score
    if not scores or max(scores.values()) == 0:
        return "full_diagnosis", 0.3
    best = max(scores, key=scores.get)
    return best, min(scores[best] / 5.0, 0.95)
