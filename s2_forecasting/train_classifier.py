#!/usr/bin/env python
"""
XGBoost Binary Classifier -- Experiment 2: High-Demand Prediction
===================================================================
Predicts whether tomorrow exceeds the product-specific 70th percentile
of historical demand, following the M5 competition classification paradigm.

Problem Formulation
-------------------
Given feature vector x_t (same 13 features as regression baseline):
  x_t = [product_id, temp_mean, temp_max, temp_min, humidity, precipitation,
         day_of_week, month, is_weekend, is_holiday, lag_1, lag_7_avg, lag_30_avg]

Binary label construction (no data leakage):
  threshold_i = P70({y_t : product_id = i, t in train})
  z_t = 1  if  y_t > threshold_{product_id(t)}  else  0

Model: XGBoost Classifier with objective = binary:logistic
  p_t_pred = P(z_t = 1 | x_t) = 1 / (1 + exp(-f(x_t)))
  Loss: Binary Cross-Entropy
    L = -(1/n) * sum [ z_t * log(p_t_pred) + (1 - z_t) * log(1 - p_t_pred) ]

  Class imbalance handled via scale_pos_weight = n_neg / n_pos.
  Default probability threshold = 0.5 for hard predictions.

Data Split (identical to regression baseline)
----------------------------------------------
  Train: 2021-01-01 to 2022-12-31  (2 years)
  Val:   2023-01-01 to 2023-06-30  (6 months, used for tuning)
  Test:  2023-07-01 to 2023-12-31  (6 months, final evaluation)

  Label thresholds computed from TRAIN SET ONLY to prevent data leakage.

Hyperparameter Tuning (5-fold CV on Train+Val, scoring=roc_auc)
-----------------------------------------------------------------
  Parameter         | Range              | Rationale
  ------------------|--------------------|-------------------------
  n_estimators      | [100, 200, 300]    | Prevent overfit on weak signal
  max_depth         | [3, 5, 7]          | Shallow trees for noisy labels
  learning_rate     | [0.01, 0.05, 0.1]  | Moderate learning rates
  subsample         | [0.8, 1.0]         | Row sampling for regularisation
  colsample_bytree  | [0.8, 1.0]         | Column sampling
  reg_alpha         | [0, 0.1]           | L1 regularisation
  reg_lambda        | [1, 1.5]           | L2 regularisation

Evaluation Metrics
------------------
  Accuracy  = (TP + TN) / (TP + TN + FP + FN)
  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)
  F1        = 2 * (Precision * Recall) / (Precision + Recall)
  ROC-AUC   = Area under the Receiver Operating Characteristic curve

  where TP/TN/FP/FN are defined with respect to the "high-demand" (z=1) class.

References
----------
  [1] Makridakis et al. (2022). M5 accuracy competition: Results, findings
      and conclusions. International Journal of Forecasting, 38(4), 1346-1364.
  [2] Chen & Guestrin (2016). XGBoost: A scalable tree boosting system.
      Proceedings of the 22nd ACM SIGKDD, 785-794.

Reproducibility
---------------
  random_state = 42
  numpy random seed = 42
  P70 thresholds saved in classifier_best_params.json

Outputs
-------
  outputs/classifier_model.pkl        : trained XGBoost classifier
  outputs/classifier_metrics.json      : train/test/gap metrics
  outputs/classifier_best_params.json  : best params + P70 thresholds
  outputs/classifier_per_product.csv   : per-SKU accuracy / ROC-AUC
  outputs/classifier_confusion.png     : confusion matrix heatmap

Author: Bakery AI System
""";

import os, warnings, json, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================
# CONFIGURATION
# ============================================================
import os as _os
BASE_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA_DIR = _os.path.join(BASE_DIR, "data")
OUT_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = [
    "product_id", "daily_tickets", "day_of_week", "month",
    "is_weekend", "is_holiday", "days_to_next_holiday",
    "lag_1", "lag_7_avg", "lag_30_avg",
    "roll_std_7", "roll_std_14", "trend_7", "category",
]
TARGET = "quantity"

CV_FOLDS = 5
N_JOBS = -1

# ============================================================
# LOAD DATA
# ============================================================
def load_data():
    """Load pre-split CSV files from data/ directory."""
    import pandas as pd
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    train = pd.read_csv(os.path.join(data_dir, "xgboost_train.csv"))
    val = pd.read_csv(os.path.join(data_dir, "xgboost_val.csv"))
    test = pd.read_csv(os.path.join(data_dir, "xgboost_test.csv"))
    print(f"Loaded from CSVs: Train={len(train)}  Val={len(val)}  Test={len(test)}")
    return train, val, test


def get_xy(df):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


# ============================================================
# BINARY LABEL: quantity > P70(product)
# ============================================================
def build_labels(train_df, val_df, test_df, percentile=70):
    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()

    thresholds = {}
    for pid in sorted(train_df["product_id"].unique()):
        sub = train_df[train_df["product_id"] == pid]
        thresh = np.percentile(sub[TARGET].values, percentile)
        thresholds[int(pid)] = round(float(thresh), 1)

    for df in [train_df, val_df, test_df]:
        df["label"] = (df[TARGET] > df["product_id"].map(thresholds)).astype(int)

    print(f"\n  P{percentile} thresholds (training set):")
    for pid, t in sorted(thresholds.items()):
        print(f"    product {pid:3d}:  > {t:.1f}  ->  high-demand")

    return train_df, val_df, test_df, thresholds


# ============================================================
# METRICS
# ============================================================
def compute_metrics(y_true, y_pred, y_proba, prefix=""):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = np.nan
    return {
        f"{prefix}Accuracy":  round(acc, 4),
        f"{prefix}Precision": round(prec, 4),
        f"{prefix}Recall":    round(rec, 4),
        f"{prefix}F1":        round(f1, 4),
        f"{prefix}ROC_AUC":   round(auc, 4),
    }


def print_metrics(metrics, title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:24s}: {v:>10.4f}")


# ============================================================
# TUNING
# ============================================================
PARAM_GRID_CLS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_alpha": [0, 0.1],
    "reg_lambda": [1, 1.5],
}


def tune(train_df, val_df):
    print("\n--- Tuning (5-fold CV) ---")
    combined = pd.concat([train_df, val_df], ignore_index=True)
    X, _ = get_xy(combined)
    y_label = combined["label"].values

    n_neg = (y_label == 0).sum()
    n_pos = (y_label == 1).sum()
    spw = n_neg / n_pos if n_pos > 0 else 1
    print(f"  neg={n_neg}  pos={n_pos}  scale_pos_weight={spw:.2f}")

    base = xgb.XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=N_JOBS,
        verbosity=0,
    )

    grid = GridSearchCV(
        base, PARAM_GRID_CLS,
        cv=CV_FOLDS,
        scoring="roc_auc",
        n_jobs=N_JOBS,
        verbose=1,
    )
    grid.fit(X, y_label)

    print(f"\n  Best: {grid.best_params_}")
    print(f"  Best CV ROC-AUC: {grid.best_score_:.4f}")
    return grid.best_params_


# ============================================================
# TRAIN FINAL
# ============================================================
def train_final(train_df, best_params):
    print("\n--- Training Final Classifier ---")
    X, _ = get_xy(train_df)
    y_label = train_df["label"].values

    n_neg = (y_label == 0).sum()
    n_pos = (y_label == 1).sum()
    spw = n_neg / n_pos if n_pos > 0 else 1

    model = xgb.XGBClassifier(
        **best_params,
        objective="binary:logistic",
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=N_JOBS,
        verbosity=0,
    )
    model.fit(X, y_label)

    joblib.dump(model, os.path.join(OUT_DIR, "classifier_model.pkl"))
    print("  Saved -> outputs/classifier_model.pkl")
    return model


# ============================================================
# PER-PRODUCT
# ============================================================
def per_product_metrics(test_df, y_pred, y_proba):
    print("\n--- Per-Product Accuracy ---")
    test_df = test_df.copy()
    test_df["pred"] = y_pred
    test_df["proba"] = y_proba

    results = []
    for pid in sorted(test_df["product_id"].unique()):
        sub = test_df[test_df["product_id"] == pid]
        yt = sub["label"].values
        yp = sub["pred"].values
        ypr = sub["proba"].values
        if len(yt) < 5:
            continue
        acc = accuracy_score(yt, yp)
        try:
            auc = roc_auc_score(yt, ypr)
        except ValueError:
            auc = np.nan
        results.append({
            "product_id": pid, "samples": len(yt),
            "pos": int(yt.sum()), "Accuracy": round(acc, 4),
            "ROC_AUC": round(auc, 4),
        })

    df_p = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    print(df_p.to_string(index=False))
    df_p.to_csv(os.path.join(OUT_DIR, "classifier_per_product.csv"), index=False)
    return df_p


# ============================================================
# CONFUSION MATRIX
# ============================================================
def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["Normal", "High Demand"])
    plt.yticks([0, 1], ["Normal", "High Demand"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     fontsize=14, fontweight="bold",
                     color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "classifier_confusion.png"), dpi=150)
    plt.close()
    print("  confusion matrix saved.")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  XGBoost Binary Classifier -- High-Demand Prediction")
    print("=" * 60)

    train, val, test = load_data()
    print(f"Train={len(train)}  Val={len(val)}  Test={len(test)}")

    train, val, test, thresholds = build_labels(train, val, test, percentile=70)
    best_params = tune(train, val)
    with open(os.path.join(OUT_DIR, "classifier_best_params.json"), "w") as f:
        json.dump({"params": best_params, "percentile": 70, "thresholds": thresholds}, f, indent=2, default=str)

    model = train_final(train, best_params)

    # Train eval
    X_train, _ = get_xy(train)
    y_train_label = train["label"].values
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    train_metrics = compute_metrics(y_train_label, y_train_pred, y_train_proba, prefix="train_")
    print_metrics(train_metrics, "CLASSIFIER (Training)")

    # Test eval
    X_test, _ = get_xy(test)
    y_test_label = test["label"].values
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test_label, y_test_pred, y_test_proba, prefix="test_")
    print_metrics(test_metrics, "CLASSIFIER (Test)")

    # Gap
    gap = {}
    for k in ["Accuracy","Precision","Recall","F1","ROC_AUC"]:
        gap[f"gap_{k}"] = round(test_metrics[f"test_{k}"] - train_metrics[f"train_{k}"], 4)
    print_metrics(gap, "TRAIN-TEST GAP")

    per_product_metrics(test, y_test_pred, y_test_proba)
    plot_confusion(y_test_label, y_test_pred)

    all_metrics = {**train_metrics, **test_metrics, **gap, "best_params": str(best_params)}
    with open(os.path.join(OUT_DIR, "classifier_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("  Done. Outputs -> s2_forecasting/outputs/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
