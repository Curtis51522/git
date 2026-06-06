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
        "status", "check", "show", "forecast", "demand", "selling", "look",
        "report", "summary", "what about", "should i", "do i", "need to",
        "doing", "issues", "update", "berapa", "banyak", "esok", "stok",
        "bakar", "sediakan",
    ],
    "waste_analysis": [
        "waste", "loss", "expired", "expiring", "expiry", "spoilage", "throw", "why",
    ],
    "promo_eval": [
        "promo", "discount", "off", "sale", "markdown", "deal", "bundle",
        "should i", "do i", "run a", "apply",
    ],
    "schedule_audit": [
        "schedule", "shift", "staff", "anomal", "who", "swap", "sick", "leave",
        "roster", "staff issues", "schedule issues", "shift issues",
        "coverage gap", "understaff",
    ],
    "cross_source_audit": [
        "health check", "audit", "full", "cross", "all product",
        "store health", "overview",
    ],
    "profit_analysis": [
        "profit", "margin", "revenue", "cost", "earn", "income",
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


def classify_intent(query: str) -> Tuple[str, float]:
    """Classify query intent. Returns (intent_label, confidence).
    
    Primary: DistilBERT (if model loaded)
    Fallback: keyword rules
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
                return intent, confidence
        except Exception as e:
            logger.warning("DistilBERT inference failed: %s, falling back to keywords", e)

    # Keyword fallback
    ql = query.lower()
    scores = {k: sum(1 for w in kws if w in ql) for k, kws in _KEYWORD_RULES.items()}
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best, 0.7
    return "stock_query", 0.5


def train_intent_classifier(training_data_path: str = None, output_dir: str = None):
    """Fine-tune DistilBERT on bakery query intent data."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        DistilBertForSequenceClassification, DistilBertTokenizer,
        Trainer, TrainingArguments,
    )

    if training_data_path is None:
        training_data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "s5_config", "training_data.json")
    if output_dir is None:
        output_dir = _MODEL_DIR

    with open(training_data_path, "r") as f:
        data = json.load(f)

    texts = [d["text"] for d in data]
    labels = [INTENT_LABELS.index(d["intent"]) for d in data]

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(INTENT_LABELS))

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=64)

    class IntentDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)

    dataset = IntentDataset(encodings, labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        save_strategy="epoch",
        logging_steps=5,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("DistilBERT intent classifier saved to %s", output_dir)
    return model, tokenizer
