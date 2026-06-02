# Intent Classifier - hybrid DistilBERT + keyword fallback
# Loads fine-tuned DistilBERT from models/distilbert/.
# Falls back to keyword rules when model unavailable or confidence < threshold.
import math, os, logging
from typing import Tuple

logger = logging.getLogger("s5.intent")

INTENT_LABELS = [
    "stock_query", "waste_analysis", "promo_eval",
    "schedule_audit", "cross_source_audit", "profit_analysis", "out_of_scope",
]

_KEYWORD_RULES = {
    "stock_query": [
        "stock", "restock", "inventory", "replenish", "bake", "prepare",
        "how many", "how is the", "what is my stock", "how is", "tell me about", "berapa", "banyak", "esok", "stok", "bakar", "sediakan",
    ],
    "waste_analysis": [
        "waste", "loss", "expired", "expiring", "expiry", "spoilage", "throw", "why",
        "buang", "bazir", "rosak", "pembaziran", "rugi",
    ],
    "promo_eval": [
        "promo", "promotion", "discount", "marketing", "effective", "combo",
        "diskaun", "tawaran", "jualan",
    ],
    "schedule_audit": [
        "schedule", "shift", "staffing", "roster", "anomal", "enough staff",
        "who is working", "baker", "bakers", "barista", "cashier", "cleaner",
        "who working", "staff today", "staff tomorrow", "working today",
        "working tomorrow", "on duty", "staffed", "understaffed", "overstaffed",
        "enough baker", "jadual", "syif", "kerja", "pekerja", "semak jadual",
    ],
    "cross_source_audit": [
        "audit everything", "full audit", "health check", "cross check",
        "any problem", "any issue", "diagnostics", "overview", "sweep", "kpi",
        "operations report", "all system", "any alert", "risks today",
        "integrity check", "consistency check", "compliance check",
        "expiry check", "dashboard summary",
    ],
    "profit_analysis": [
        "profit", "revenue", "margin", "income", "earnings", "earn",
        "make money", "untung", "pendapatan", "keuntungan", "jualan", "berapa untung", "margin untung",
        "sales revenue", "gross profit", "net profit", "how much profit",
        "how much revenue", "cost analysis", "profit breakdown", "product margin",
    ],
}

# Path to DistilBERT model relative to s5-agent-brain/
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "distilbert")
INTENT_CONFIDENCE_THRESHOLD = 0.75


class IntentClassifier:
    """Hybrid intent classifier: DistilBERT model + keyword fallback."""

    def __init__(self):
        self.labels = INTENT_LABELS
        self.threshold = INTENT_CONFIDENCE_THRESHOLD
        self._model = None
        self._tokenizer = None
        self._device = None
        self._model_loaded = False
        self._try_load_model()

    def classify(self, query: str) -> Tuple[str, float]:
        """Return (intent_label, confidence_float)."""
        if self._model_loaded:
            intent, conf = self._classify_dl(query)
            if not math.isnan(conf) and conf >= self.threshold:
                kw_intent, kw_conf = self._classify_keywords(query)
                if kw_intent != "out_of_scope" and kw_intent != intent and kw_conf >= 0.60:
                    return kw_intent, kw_conf
                if intent == "out_of_scope" and kw_intent != "out_of_scope":
                    return kw_intent, kw_conf
                return intent, conf
        return self._classify_keywords(query)

    def _try_load_model(self):
        try:
            import torch
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
            if not os.path.exists(os.path.join(_MODEL_DIR, "config.json")):
                logger.info("DistilBERT model not found at %s, using keyword fallback", _MODEL_DIR)
                return
            self._tokenizer = DistilBertTokenizer.from_pretrained(_MODEL_DIR)
            self._model = DistilBertForSequenceClassification.from_pretrained(_MODEL_DIR)
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
            self._model.eval()
            self._model_loaded = True
            logger.info("DistilBERT intent classifier loaded (%s)", self._device)
        except Exception as e:
            logger.warning("DistilBERT load failed: %s, using keyword fallback", e)
            self._model_loaded = False

    def _classify_dl(self, query: str) -> Tuple[str, float]:
        import torch
        import torch.nn.functional as F
        inputs = self._tokenizer(query, return_tensors="pt", truncation=True, max_length=128, padding=True)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0]
        top_idx = int(torch.argmax(probs).item())
        confidence = float(probs[top_idx].item())
        return self.labels[top_idx], confidence

    def _classify_keywords(self, query: str) -> Tuple[str, float]:
        q = query.lower()
        best_intent = "out_of_scope"
        best_score = 0.0
        for intent, keywords in _KEYWORD_RULES.items():
            matches = sum(1 for kw in keywords if kw in q)
            if matches > 0:
                score = min(0.92, 0.5 + matches * 0.15)
                if score > best_score:
                    best_score = score
                    best_intent = intent
        if best_intent == "out_of_scope":
            return "out_of_scope", 0.3
        return best_intent, best_score


# Singleton
_classifier: IntentClassifier = None

def get_classifier() -> IntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier


def classify_intent(query: str) -> Tuple[str, float]:
    """Convenience function: classify query and return (intent, confidence)."""
    return get_classifier().classify(query)