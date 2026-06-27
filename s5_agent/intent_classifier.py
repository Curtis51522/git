# Intent Classifier - DistilBERT fine-tuned on bakery queries
# Primary intent router for S5. Keyword rules as fallback.
import os, json, logging
from typing import Tuple

logger = logging.getLogger("s5.intent")

INTENT_LABELS = [
    "stock_query", "waste_analysis", "promo_eval",
    "schedule_audit", "cross_source_audit", "profit_analysis",
    "comparison_analysis", "out_of_scope",
]

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "distilbert_intent")
_MODEL = None
_TOKENIZER = None

_KEYWORD_RULES = {
    "stock_query": [
        "stock", "restock", "inventory", "replenish", "bake", "prepare",
        "how many", "how is the", "what is my stock", "how is", "tell me about",
        "forecast", "demand",
        "report", "summary", "what about",
        "doing", "issues", "update", "berapa", "banyak", "esok", "stok",
        "bakar", "sediakan", "semua barang", "semua stok",
        "top product", "best selling", "most popular", "give me",
    ],
    "waste_analysis": [
        "waste", "loss", "expired", "expiring", "expiry", "spoilage", "throw", "why",
        "bazir", "buang", "membazir", "rosak", "busuk",
    ],
    "promo_eval": [
        "promo", "discount", "sale", "markdown", "deal", "bundle",
        "run a", "apply",
    ],
    "schedule_audit": [
        "schedule", "shift", "staff", "anomal", "who", "swap", "sick", "leave",
        "roster", "staff issues", "schedule issues", "shift issues",
        "coverage gap", "understaff", "kerja", "cuti", "siapa",
    ],
    "cross_source_audit": [
        "health check", "audit", "full", "cross", "all product",
        "store health", "overview", "masalah kedai", "check kedai",
        "apa masalah", "semua ok", "everyting ok",
    ],
    "profit_analysis": [
        "profit", "margin", "revenue", "cost", "earn", "income",
        "untung", "rugi", "pendapatan", "jualan", "hasil", "keuntungan",
        "belanja", "modal", "perbelanjaan",
    ],
    "comparison_analysis": [
        "compare", "versus", " vs ", "which is better", "which one",
        "between", "sell better", "sells best", "sells more",
        "top 3", "ranking", "rank",
    ],
}


def _load_model():
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return True
    try:
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
        if os.path.exists(_MODEL_DIR):
            _MODEL = DistilBertForSequenceClassification.from_pretrained(_MODEL_DIR)
            _TOKENIZER = DistilBertTokenizer.from_pretrained(_MODEL_DIR)
            logger.info("DistilBERT intent classifier loaded from %s", _MODEL_DIR)
            return True
        else:
            logger.warning("DistilBERT model not found at %s, using keyword fallback", _MODEL_DIR)
            return False
    except ImportError:
        logger.warning("transformers not installed, using keyword fallback")
        return False
    except Exception as e:
        logger.warning("DistilBERT load failed: %s, using keyword fallback", e)
        return False


def _keyword_classify(query: str) -> Tuple[str, float]:
    """Pure keyword classification, returns (intent, keyword_density).
    Profit-force: Malay/English profit words auto-boost profit_analysis score."""
    ql = query.lower()
    scores = {k: sum(1 for w in kws if w in ql) for k, kws in _KEYWORD_RULES.items()}

    # Profit-force: explicit profit words override generic counters like "berapa"
    _PROFIT_FORCE = ["untung", "rugi", "keuntungan", "margin", "profit", "revenue", "income"]
    if any(w in ql for w in _PROFIT_FORCE):
        scores["profit_analysis"] = max(scores["profit_analysis"], scores.get("stock_query", 0) + 1)

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best, scores[best]
    return "stock_query", 0


def classify_intent(query: str) -> Tuple[str, float]:
    """Classify query intent. Returns (intent_label, confidence).
    
    Primary: DistilBERT (if model loaded)
    Cross-check: when DistilBERT confidence < 0.85, validate against keywords;
                 if keywords strongly disagree (>3:1 ratio), override to keyword result.
    Fallback: keyword rules (if DistilBERT unavailable)
    """
    if _load_model():
        try:
            import torch
            inputs = _TOKENIZER(query, return_tensors="pt", truncation=True, max_length=64, padding=True)
            with torch.no_grad():
                outputs = _MODEL(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                max_prob, max_idx = probs.max(dim=0)
                intent = INTENT_LABELS[max_idx.item()]
                confidence = round(max_prob.item(), 4)

            # Cross-check with keywords: run whenever keywords disagree with DistilBERT.
            # Two trigger conditions:
            #   (a) DistilBERT confidence < 0.85 — any keyword disagreement counts
            #   (b) Keywords have signal (score >= 1) but DistilBERT's own intent has zero keyword match
            kw_intent, kw_score = _keyword_classify(query)
            if kw_score > 0 and kw_intent != intent:
                db_kw_score = sum(1 for w in _KEYWORD_RULES.get(intent, []) if w in query.lower())
                should_override = False

                if confidence < 0.85 and kw_score >= 3 and kw_score >= db_kw_score * 3:
                    should_override = True  # low confidence + strong keyword disagreement
                elif db_kw_score == 0 and kw_score >= 1:
                    should_override = True  # DistilBERT has zero keyword support, keywords do
                elif kw_score >= 2 and kw_score >= db_kw_score * 2:
                    should_override = True  # keywords strongly outvote DistilBERT keywords

                if should_override:
                    logger.info("Keyword override: DistilBERT=%s(%.2f,kw=%d) → keywords=%s(kw=%d)",
                                intent, confidence, db_kw_score, kw_intent, kw_score)
                    return kw_intent, min(confidence, 0.75)

            return intent, confidence
        except Exception as e:
            logger.warning("DistilBERT inference failed: %s, falling back to keywords", e)

    # Keyword-only fallback
    kw_intent, kw_score = _keyword_classify(query)
    if kw_score > 0:
        return kw_intent, 0.7
    return "stock_query", 0.5
