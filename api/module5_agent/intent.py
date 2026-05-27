"""
Intent Classifier for S5 -- hybrid DistilBERT + keyword fallback.

Loads the fine-tuned DistilBERT model if available (models/distilbert/).
Falls back to keyword rules when:
- Model not trained yet
- PyTorch/transformers not installed
- Any loading error

Intent labels: stock_query, waste_analysis, promo_eval, schedule_audit, out_of_scope
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import INTENT_CONFIDENCE_THRESHOLD

INTENT_LABELS = ["stock_query", "waste_analysis", "promo_eval", "schedule_audit", "cross_source_audit", "out_of_scope"]

# Keyword fallback rules (used when DistilBERT model is unavailable)
_KEYWORD_RULES = {
    "stock_query":     ["stock", "restock", "inventory", "replenish", "bake", "prepare", "how many", "stok", "bakar", "sediakan"],
    "waste_analysis":  ["waste", "loss", "expired", "spoilage", "throw", "why", "buang", "bazir", "rosak"],
    "promo_eval":      ["promo", "promotion", "discount", "marketing", "effective", "combo", "diskaun", "tawaran", "jualan"],
    "schedule_audit":  ["schedule", "shift", "staffing", "roster", "anomal", "enough staff", "who is working", "baker", "bakers", "barista", "cashier", "cleaner", "who working", "staff today", "staff tomorrow", "working today", "working tomorrow", "on duty", "staffed", "understaffed", "overstaffed", "enough baker", "jadual", "syif", "kerja"],
    "cross_source_audit": ["audit everything", "full audit", "health check", "cross check", "any problem", "any issue", "diagnostics", "overview", "sweep", "kpi", "operations report", "all system", "any alert", "risks today", "integrity check", "consistency check", "compliance check", "expiry check", "dashboard summary"],
}


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

    # ------------------------------------------------------------------
    def classify(self, query: str) -> tuple:
        """Return (intent_label, confidence_float).
        
        Uses DistilBERT if model is loaded AND confidence >= threshold.
        Falls back to keyword rules otherwise.
        
        Post-processing: if keyword rules strongly conflict with DL result,
        the keyword classifier wins (DL model has limited training data).
        """
        if self._model_loaded:
            intent, conf = self._classify_dl(query)
            import math
            if not math.isnan(conf) and conf >= self.threshold:
                kw_intent, kw_conf = self._classify_keywords(query)
                # If keywords disagree AND are confident, override DL
                if kw_intent != "out_of_scope" and kw_intent != intent and kw_conf >= 0.7:
                    return kw_intent, kw_conf
                # If DL says out_of_scope but keywords match
                if intent == "out_of_scope" and kw_intent != "out_of_scope":
                    return kw_intent, kw_conf
                return intent, conf
        return self._classify_keywords(query)

    # ------------------------------------------------------------------
    # Deep-learning path
    # ------------------------------------------------------------------
    def _try_load_model(self):
        """Attempt to load the fine-tuned DistilBERT model."""
        try:
            import torch
            from transformers import (
                DistilBertTokenizer,
                DistilBertForSequenceClassification,
            )
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "models", "distilbert",
            )
            if not os.path.exists(os.path.join(model_dir, "config.json")):
                return

            self._tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
            self._model = DistilBertForSequenceClassification.from_pretrained(model_dir)
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
            self._model.eval()
            self._model_loaded = True
        except Exception:
            self._model_loaded = False

    def _classify_dl(self, query: str) -> tuple:
        import torch
        import torch.nn.functional as F
        inputs = self._tokenizer(
            query, return_tensors="pt", truncation=True, max_length=128, padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0]
        top_idx = int(torch.argmax(probs).item())
        confidence = float(probs[top_idx].item())
        return self.labels[top_idx], confidence

    # ------------------------------------------------------------------
    # Keyword fallback
    # ------------------------------------------------------------------
    def _classify_keywords(self, query: str) -> tuple:
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
_intent_classifier: IntentClassifier = None

def get_classifier() -> IntentClassifier:
    global _intent_classifier
    if _intent_classifier is None:
        _intent_classifier = IntentClassifier()
    return _intent_classifier
