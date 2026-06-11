"""Train DistilBERT-based intent classifier for S5 IntentClassifier.
Fine-tunes distilbert-base-multilingual-cased on labeled intent queries."""
import os, sys, json
import numpy as np
import torch
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import INTENT_LABELS, MODEL_CACHE_DIR, CACHE_MANIFEST

MODEL_NAME = "distilbert-base-multilingual-cased"
OUTPUT_DIR = os.path.join(MODEL_CACHE_DIR, "distilbert")
LABEL2ID = {label: i for i, label in enumerate(INTENT_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# Load training data
DATA_PATH = os.path.join(os.path.dirname(__file__), "intent_data.json")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    TRAINING_DATA = json.load(f)
print(f"Loaded {len(TRAINING_DATA)} training examples")

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding=True, truncation=True, max_length=128)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

def train_and_save():
    texts, labels = zip(*TRAINING_DATA)
    label_ids = [LABEL2ID[l] for l in labels]
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, label_ids, test_size=0.2, random_state=42, stratify=label_ids
    )
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})
    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(INTENT_LABELS),
        id2label=ID2LABEL, label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=8,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Update cache manifest
    cache = {}
    if os.path.exists(CACHE_MANIFEST):
        with open(CACHE_MANIFEST, "r") as f:
            cache = json.load(f)
    cache["distilbert"] = {"path": OUTPUT_DIR, "model": MODEL_NAME, "num_labels": len(INTENT_LABELS)}
    os.makedirs(os.path.dirname(CACHE_MANIFEST), exist_ok=True)
    with open(CACHE_MANIFEST, "w") as f:
        json.dump(cache, f, indent=2, default=str)

    # Final eval + classification report
    preds_output = trainer.predict(val_ds)
    preds = np.argmax(preds_output.predictions, axis=-1)
    print("\n=== Classification Report ===")
    print(classification_report(val_labels, preds, target_names=INTENT_LABELS))
    print(f"\nModel saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    train_and_save()
