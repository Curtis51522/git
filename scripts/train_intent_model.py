"""
DistilBERT Intent Classifier — Training with 1142 synthetic samples.

Strategy: 2-phase transfer learning
- Phase 1: Frozen DistilBERT base, train classifier head only
- Phase 2: Unfreeze, fine-tune with lower LR
- Optuna tunes: learning_rate, batch_size, dropout, num_epochs
"""

import json, logging, os, sys
from datetime import datetime
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback,
)
import optuna

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "models", "distilbert")
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("intent_train")

INTENT_LABELS = ["stock_query","waste_analysis","promo_eval","schedule_audit",
                 "cross_source_audit","profit_analysis","out_of_scope"]
NUM_LABELS = len(INTENT_LABELS)
ID2LABEL = dict(enumerate(INTENT_LABELS))
LABEL2ID = {l: i for i, l in enumerate(INTENT_LABELS)}

# ---------------------------------------------------------------------------
# Load generated training data
# ---------------------------------------------------------------------------
data_path = os.path.join(MODEL_DIR, "training_data.json")
with open(data_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Balance classes — downsample majority classes to ~120 per class max
from collections import Counter
balanced = []
per_class = Counter()
MAX_PER_CLASS = 120
for text, intent in raw_data:
    if per_class[intent] < MAX_PER_CLASS:
        balanced.append((text, intent))
        per_class[intent] += 1

logger.info("Training data: %d samples (balanced at %d/class)", len(balanced), MAX_PER_CLASS)
for intent in INTENT_LABELS:
    logger.info("  %s: %d samples", intent, per_class[intent])

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]), truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def objective(trial, train_texts, train_labels, val_texts, val_labels, tokenizer):
    lr = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    num_epochs = trial.suggest_int("num_epochs", 5, 15)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=NUM_LABELS,
        id2label=ID2LABEL, label2id=LABEL2ID,
    )
    model.config.seq_classif_dropout = dropout
    model.config.attention_dropout = dropout

    # Freeze base
    for param in model.distilbert.parameters():
        param.requires_grad = False

    train_ds = IntentDataset(train_texts, train_labels, tokenizer)
    val_ds = IntentDataset(val_texts, val_labels, tokenizer)

    args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "checkpoints"),
        num_train_epochs=num_epochs, per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size, learning_rate=lr,
        warmup_steps=10, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1",
        logging_steps=50, report_to="none", disable_tqdm=True,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer.train()
    return trainer.evaluate()["eval_f1"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=" * 60)
    logger.info("DistilBERT Intent Classifier — %d balanced samples", len(balanced))
    logger.info("=" * 60)

    texts = [item[0] for item in balanced]
    labels = [LABEL2ID[item[1]] for item in balanced]

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)
    logger.info("Split: train=%d, val=%d, test=%d", len(train_texts), len(val_texts), len(test_texts))

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # ---- Optuna ----
    logger.info("Optuna hyperparameter search (10 trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: objective(t, train_texts, train_labels, val_texts, val_labels, tokenizer),
        n_trials=10, timeout=900,
    )
    best = study.best_trial.params
    logger.info("Best: %s (F1=%.4f)", best, study.best_value)

    # ---- Final training: Phase 1 frozen ----
    logger.info("Phase 1: frozen base training...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=NUM_LABELS,
        id2label=ID2LABEL, label2id=LABEL2ID,
    )
    model.config.seq_classif_dropout = best["dropout"]
    model.config.attention_dropout = best["dropout"]
    for param in model.distilbert.parameters():
        param.requires_grad = False

    train_ds = IntentDataset(train_texts, train_labels, tokenizer)
    val_ds = IntentDataset(val_texts, val_labels, tokenizer)

    args1 = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "checkpoints"),
        num_train_epochs=best["num_epochs"], per_device_train_batch_size=best["batch_size"],
        per_device_eval_batch_size=best["batch_size"], learning_rate=best["learning_rate"],
        warmup_steps=10, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1",
        logging_steps=50, report_to="none",
    )
    trainer = Trainer(model=model, args=args1, train_dataset=train_ds, eval_dataset=val_ds,
                      compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=5)])
    trainer.train()

    # ---- Phase 2: unfreeze + fine-tune ----
    logger.info("Phase 2: unfreezing + fine-tuning...")
    for param in model.distilbert.parameters():
        param.requires_grad = True
    args2 = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "checkpoints_ft"),
        num_train_epochs=max(3, best["num_epochs"] // 3),
        per_device_train_batch_size=best["batch_size"],
        per_device_eval_batch_size=best["batch_size"],
        learning_rate=best["learning_rate"] * 0.1,
        warmup_steps=5, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1",
        logging_steps=50, report_to="none",
    )
    trainer = Trainer(model=model, args=args2, train_dataset=train_ds, eval_dataset=val_ds,
                      compute_metrics=compute_metrics)
    trainer.train()

    # ---- Evaluate ----
    logger.info("Evaluating on test set (%d samples)...", len(test_texts))
    test_ds = IntentDataset(test_texts, test_labels, tokenizer)
    preds_out = trainer.predict(test_ds)
    preds = np.argmax(preds_out.predictions, axis=-1)
    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average="weighted")
    logger.info("Test Accuracy: %.4f", acc)
    logger.info("Test F1: %.4f", f1)
    logger.info("Report:\n%s", classification_report(test_labels, preds, target_names=INTENT_LABELS))

    # ---- Save ----
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "num_labels": NUM_LABELS, "labels": INTENT_LABELS,
        "best_hyperparams": best, "best_val_f1": study.best_value,
        "test_accuracy": acc, "test_f1": f1,
        "train_samples": len(train_texts), "val_samples": len(val_texts),
        "test_samples": len(test_texts), "strategy": "2phase_frozen_then_finetune",
    }
    with open(os.path.join(MODEL_DIR, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Model saved. Metadata: %s", json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
