# s5_agent/router/train_intent.py — Generate training data + train DistilBERT
import os, sys, json, httpx, logging
_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PARENT not in sys.path: sys.path.insert(0, _PARENT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("s5.train")

INTENTS = {
    "profit_root_cause": "Why is profit/revenue/margin abnormal? Lower than expected earnings.",
    "wastage_root_cause": "Why is wastage/spoilage/loss high? Materials being wasted.",
    "production_advice": "How much should we bake/produce tomorrow? Production planning.",
    "inventory_diagnosis": "Check stock/inventory/materials level. Do we have enough?",
    "staffing_diagnosis": "Check schedule/shift/staffing. Are we under/over staffed?",
    "full_diagnosis": "Comprehensive/overall system check. Full report on everything.",
    "promo_evaluation": "How effective was the promotion/discount/campaign? Marketing performance.",
}

STYLES_ZH = ["formal", "colloquial", "inverted", "dialectal", "code_switched", "terse"]
STYLES_EN = ["formal", "colloquial", "inverted", "manglish"]

def generate_training_data(api_key: str, output_path: str):
    llm_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1") + "/chat/completions"
    key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
    if not key:
        logger.error("No API key found")
        return

    all_data = []

    for intent, desc in INTENTS.items():
        logger.info("Generating for intent: %s", intent)

        # Chinese: 6 styles x 60 = 360
        for style in STYLES_ZH:
            prompt = f"""Generate 30 Chinese bakery-related queries for intent classification.
Intent: {intent} - {desc}
Style: {style}
Requirements: natural, varied sentence structures, different lengths (2-15 words).
Return ONLY JSON: {{"queries": ["query1", ...]}} (exactly 60 items)"""
            queries = _call_llm(llm_url, key, prompt, intent)
            batch_seen = set()
            added = 0
            for q in queries:
                k = q.strip().lower()
                if k not in batch_seen:
                    batch_seen.add(k)
                    all_data.append({"text": q.strip(), "label": intent})
                    added += 1
            logger.info("  [zh:%s] %d added", style, added)

        # English: 4 styles x 50 = 200
        for style in STYLES_EN:
            prompt = f"""Generate 25 English bakery-related queries for intent classification.
Intent: {intent} - {desc}
Style: {style}
Requirements: natural, varied sentence structures, different lengths (3-12 words).
Return ONLY JSON: {{"queries": ["query1", ...]}} (exactly 50 items)"""
            queries = _call_llm(llm_url, key, prompt, intent)
            batch_seen = set()
            added = 0
            for q in queries:
                k = q.strip().lower()
                if k not in batch_seen:
                    batch_seen.add(k)
                    all_data.append({"text": q.strip(), "label": intent})
                    added += 1
            logger.info("  [en:%s] %d added", style, added)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d training examples to %s", len(all_data), output_path)
    return all_data

def _call_llm(llm_url, key, prompt, intent):
    try:
        r = httpx.post(llm_url, json={
            "model": "deepseek-chat", "temperature": 0.8,
            "messages": [{"role": "user", "content": prompt}]
        }, headers={"Authorization": f"Bearer {key}"}, timeout=60)
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"]
            text = text.strip()
            queries = _extract_queries(text)
            if queries:
                return queries
            logger.warning("  Could not extract queries for %s from: %s...", intent, text[:200])
        else:
            logger.warning("  LLM error for %s: %d", intent, r.status_code)
    except Exception as e:
        logger.error("  Failed for %s: %s", intent, e)
    return []

def _extract_queries(text):
    """Robust JSON extraction from LLM output with multiple fallback strategies."""
    json_str = text
    # 1) Strip markdown code fences
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        json_str = m.group(1).strip()
    # 2) Find JSON object or array start
    for bracket in ["{", "["]:
        start = json_str.find(bracket)
        if start >= 0:
            json_str = json_str[start:]
            break
    # 3) Try parsing; if fails, try to fix common JSON issues
    attempts = [
        json_str,
        json_str.replace("\n", " ").replace("\r", ""),
        re.sub(r",\s*}", "}", json_str),
        re.sub(r",\s*]", "]", json_str),
    ]
    for attempt in attempts:
        try:
            data = json.loads(attempt)
            if isinstance(data, dict) and "queries" in data:
                return data["queries"]
            if isinstance(data, list):
                if all(isinstance(x, str) for x in data):
                    return data
                if len(data) > 0 and isinstance(data[0], dict) and "queries" in data[0]:
                    return data[0]["queries"]
        except (json.JSONDecodeError, ValueError):
            continue
    # 4) Last resort: regex extract all double-quoted strings with length >= 2
    texts = re.findall(r'"([^"]{2,})"', json_str)
    if texts and len(texts) >= 5:
        return texts
    return []

def train_model(data_path: str, model_output_path: str):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    from datasets import Dataset
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = sorted(set(d["label"] for d in data))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    for d in data:
        d["label_id"] = label2id[d["label"]]

    # 85/15 split
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    split = int(len(data) * 0.85)
    train_data = [data[i] for i in indices[:split]]
    test_data = [data[i] for i in indices[split:]]

    logger.info("Train: %d, Test: %d, Labels: %d", len(train_data), len(test_data), len(labels))

    model_name = "distilbert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    def tokenize(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

    train_ds = Dataset.from_list(train_data).map(tokenize, batched=True, remove_columns=["text", "label"])
    test_ds = Dataset.from_list(test_data).map(tokenize, batched=True, remove_columns=["text", "label"])
    train_ds = train_ds.rename_column("label_id", "label")
    test_ds = test_ds.rename_column("label_id", "label")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="macro")}

    training_args = TrainingArguments(
        output_dir=model_output_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds,
        eval_dataset=test_ds, processing_class=tokenizer, compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    logger.info("Final metrics: %s", metrics)

    model.save_pretrained(model_output_path)
    tokenizer.save_pretrained(model_output_path)

    with open(os.path.join(model_output_path, "label_map.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f)

    logger.info("Model saved to %s", model_output_path)
    return metrics

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true", help="Generate training data via LLM")
    ap.add_argument("--train", action="store_true", help="Train DistilBERT")
    ap.add_argument("--all", action="store_true", help="Generate + Train")
    ap.add_argument("--data_path", default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "intent_train.json"))
    ap.add_argument("--model_path", default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "intent_model"))
    ap.add_argument("--api_key", default="")
    args = ap.parse_args()

    if args.generate or args.all:
        generate_training_data(args.api_key, args.data_path)
    if args.train or args.all:
        train_model(args.data_path, args.model_path)
