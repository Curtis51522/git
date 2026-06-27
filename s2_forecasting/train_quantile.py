#!/usr/bin/env python
"""
XGBoost Quantile Regression -- Experiment 3: Probabilistic Demand Forecasting
================================================================================
Trains three independent XGBoost quantile regressors (alpha = 0.10, 0.50, 0.90)
to output prediction intervals for inventory decisions.

Motivation: Point predictions ("5.3 units") are unreliable at daily SKU level.
Quantile regression produces calibrated intervals ("2--12 units, 80% confidence"),
enabling practical stock-level recommendations.

Problem Formulation
-------------------
Given feature vector x_t (same 13 features as regression baseline):
  x_t = [product_id, temp_mean, temp_max, temp_min, humidity, precipitation,
         day_of_week, month, is_weekend, is_holiday, lag_1, lag_7_avg, lag_30_avg]

For each quantile level alpha in {0.10, 0.50, 0.90}, train:
  q_t_pred^(alpha) = f_alpha(x_t)

such that:
  P( y_t <= q_t_pred^(alpha) ) approx alpha

Model: XGBoost Regressor with objective = reg:quantileerror
  Pinball (Quantile) Loss:
    rho_alpha(u) = alpha * max(u, 0)  +  (1 - alpha) * max(-u, 0)
    L_alpha = (1/n) * sum_{t=1}^{n} rho_alpha( y_t - q_t_pred^(alpha) )

  For alpha = 0.5, pinball loss reduces to MAE/2 (median regression).
  For alpha = 0.9, the loss penalises under-prediction 9x more than over-prediction,
    producing conservative (upper-bound) forecasts suitable for stock decisions.

Hyperparameter Tuning (5-fold CV on Q50, shared with Q10/Q90)
------------------------------------------------------------------
  Tune on median (Q50) model only; reuse best params for Q10 and Q90.
  Rationale: tree-structure hyperparams (max_depth, subsample, etc.) are
  quantile-agnostic. Only the quantile_alpha differs.

  Parameter         | Range              | Rationale
  ------------------|--------------------|-------------------------
  n_estimators      | [100, 200, 300]    | Prevent overfit
  max_depth         | [3, 5, 7]          | Shallow for noisy daily data
  learning_rate     | [0.01, 0.05, 0.1]  | Standard range
  subsample         | [0.8, 1.0]         | Row sampling
  colsample_bytree  | [0.8, 1.0]         | Column sampling
  reg_alpha         | [0, 0.1]           | L1 regularisation
  reg_lambda        | [1, 1.5]           | L2 regularisation

Data Split (identical to regression baseline)
----------------------------------------------
  Train: 2021-01-01 to 2022-12-31  (2 years)
  Val:   2023-01-01 to 2023-06-30  (6 months, combined with train for fitting)
  Test:  2023-07-01 to 2023-12-31  (6 months, final evaluation)

Evaluation Metrics
------------------
  Coverage (alpha_low, alpha_high):
    fraction of test samples where y_t in [q_pred^(alpha_low), q_pred^(alpha_high)]
    - Q10-Q90 target: 0.80 (80% prediction interval)
    - Q10-Q50 target: 0.40
    - Q50-Q90 target: 0.40

  Interval Width = mean( q_pred^(alpha_high) - q_pred^(alpha_low) )
    Narrower intervals with correct coverage indicate better calibration.

POS Application
---------------
  Customer-facing output for product i on day t:
    "Low (Q10):    2 units"
    "Expected (Q50):  5 units"
    "Recommend (Q90): 12 units (90% confidence)"

References
----------
  [1] Lim et al. (2021). Temporal Fusion Transformer for Multi-horizon Time
      Series Forecasting. International Journal of Forecasting, 37(4), 1748-1764.
  [2] Salinas et al. (2020). DeepAR: Probabilistic forecasting with autoregressive
      recurrent networks. International Journal of Forecasting, 36(3), 1181-1191.
  [3] Koenker & Hallock (2001). Quantile Regression. Journal of Economic
      Perspectives, 15(4), 143-156.
  [4] Chen & Guestrin (2016). XGBoost: A scalable tree boosting system.
      Proceedings of the 22nd ACM SIGKDD, 785-794.

Reproducibility
---------------
  random_state = 42
  numpy random seed = 42
  sklearn version >= 1.0
  xgboost version >= 1.7

Outputs
-------
  outputs/quantile_model_q10.pkl    : XGBoost quantile regressor (alpha=0.10)
  outputs/quantile_model_q50.pkl    : XGBoost quantile regressor (alpha=0.50)
  outputs/quantile_model_q90.pkl    : XGBoost quantile regressor (alpha=0.90)
  outputs/quantile_best_params.json  : GridSearchCV best hyperparameters
  outputs/quantile_metrics.json      : coverage rates + interval widths
  outputs/quantile_per_product.csv   : per-SKU coverage analysis

Author: Bakery AI System
"""

import os, warnings, json, joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
import xgboost as xgb

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================
# CONFIG
# ============================================================
import os as _os
BASE_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA_DIR = _os.path.join(BASE_DIR, "data")
OUT_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = [
    "product_id", "temp_mean", "temp_max", "temp_min",
    "humidity", "precipitation", "day_of_week", "month",
    "is_weekend", "is_holiday", "lag_1", "lag_7_avg", "lag_30_avg",
]
TARGET = "quantity"
QUANTILES = [0.10, 0.50, 0.90]
N_JOBS = -1


# ============================================================
# LOAD
# ============================================================
def load_data():
    """Run full preprocessing pipeline; return train/val/test DataFrames."""
    from preprocess import run_preprocessing
    train, val, test = run_preprocessing(verbose=True)
    return train, val, test


def get_xy(df):
    return df[FEATURES].copy(), df[TARGET].copy()


# ============================================================
# HYPERPARAMETER TUNING (on Q50 only, shared with Q10/Q90)
# ============================================================
PARAM_GRID_QUANTILE = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_alpha": [0, 0.1],
    "reg_lambda": [1, 1.5],
}


def tune_quantile(train_df, val_df):
    """GridSearchCV on Q50 model; best params shared with Q10/Q90."""
    print("\n--- Tuning Q50 (5-fold CV) ---")
    combined = pd.concat([train_df, val_df], ignore_index=True)
    X, y = get_xy(combined)

    base = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=0.50,
        random_state=42,
        n_jobs=N_JOBS,
        verbosity=0,
    )

    grid = GridSearchCV(
        base, PARAM_GRID_QUANTILE,
        cv=5,
        scoring="neg_mean_absolute_error",  # pinball(0.5) = MAE/2
        n_jobs=N_JOBS,
        verbose=1,
    )
    grid.fit(X, y)

    print(f"\n  Best params: {grid.best_params_}")
    print(f"  Best CV pinball loss: {-grid.best_score_:.4f}")

    return grid.best_params_


# ============================================================
# TRAIN ONE QUANTILE MODEL
# ============================================================
def train_quantile(train_df, val_df, quantile_alpha, best_params):
    print(f"\n--- Training Q{int(quantile_alpha*100)} ---")
    combined = pd.concat([train_df, val_df], ignore_index=True)
    X, y = get_xy(combined)

    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile_alpha,
        **best_params,
        random_state=42,
        n_jobs=N_JOBS,
        verbosity=0,
    )
    model.fit(X, y)

    fname = os.path.join(OUT_DIR, f"quantile_model_q{int(quantile_alpha*100)}.pkl")
    joblib.dump(model, fname)
    print(f"  Saved -> {fname}")
    return model


# ============================================================
# EVALUATE
# ============================================================
def coverage_metrics(y_true, y_lower, y_upper):
    """Fraction of true values falling within [y_lower, y_upper]."""
    in_interval = (y_true >= y_lower) & (y_true <= y_upper)
    return in_interval.mean()


def interval_width(y_lower, y_upper):
    return np.mean(y_upper - y_lower)


def evaluate(test_df, models):
    X_test, y_test = get_xy(test_df)

    preds = {}
    for q, model in models.items():
        raw = model.predict(X_test)
        preds[q] = np.maximum(raw, 0)

    # Q10-Q90 coverage (should be ~80%)
    cov_80 = coverage_metrics(y_test, preds[0.10], preds[0.90])
    # Q10-Q50 coverage (should be ~40%)
    cov_40_low = coverage_metrics(y_test, preds[0.10], preds[0.50])
    # Q50-Q90 coverage (should be ~40%)
    cov_40_high = coverage_metrics(y_test, preds[0.50], preds[0.90])

    width_80 = interval_width(preds[0.10], preds[0.90])
    width_50 = interval_width(preds[0.10], preds[0.50])

    metrics = {
        "coverage_Q10_Q90": round(float(cov_80), 4),
        "coverage_Q10_Q50": round(float(cov_40_low), 4),
        "coverage_Q50_Q90": round(float(cov_40_high), 4),
        "interval_width_Q10_Q90": round(float(width_80), 2),
        "interval_width_Q10_Q50": round(float(width_50), 2),
    }

    print(f"\n  Coverage Q10-Q90:  {cov_80:.4f}  (target 0.80)")
    print(f"  Coverage Q10-Q50:  {cov_40_low:.4f}  (target 0.40)")
    print(f"  Coverage Q50-Q90:  {cov_40_high:.4f}  (target 0.40)")
    print(f"  80% interval width: {width_80:.2f}")
    print(f"  50% interval width: {width_50:.2f}")

    return metrics, preds, y_test


# ============================================================
# PER-PRODUCT COVERAGE
# ============================================================
def per_product_coverage(test_df, preds, y_test):
    print("\n--- Per-Product Coverage (Q10-Q90) ---")
    test_df = test_df.copy()
    test_df["y_true"] = y_test
    test_df["q10"] = preds[0.10]
    test_df["q90"] = preds[0.90]

    results = []
    for pid in sorted(test_df["product_id"].unique()):
        sub = test_df[test_df["product_id"] == pid]
        yt = sub["y_true"].values
        ql = sub["q10"].values
        qu = sub["q90"].values
        if len(yt) < 5:
            continue
        cov = coverage_metrics(yt, ql, qu)
        w = interval_width(ql, qu)
        results.append({
            "product_id": pid, "samples": len(yt),
            "coverage_80": round(cov, 4),
            "avg_width": round(w, 2),
        })

    df_p = pd.DataFrame(results).sort_values("coverage_80", ascending=False)
    print(df_p.to_string(index=False))
    df_p.to_csv(os.path.join(OUT_DIR, "quantile_per_product.csv"), index=False)
    return df_p


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  XGBoost Quantile Regression -- Q10 / Q50 / Q90")
    print("=" * 60)

    train, val, test = load_data()
    print(f"Train={len(train)}  Val={len(val)}  Test={len(test)}")

    best_params = tune_quantile(train, val)
    with open(os.path.join(OUT_DIR, "quantile_best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2, default=str)

    models = {}
    for q in QUANTILES:
        models[q] = train_quantile(train, val, q, best_params)

    print("\n--- Evaluation on Test Set ---")
    metrics, preds, y_test = evaluate(test, models)
    per_product_coverage(test, preds, y_test)

    metrics["best_params"] = str(best_params)
    with open(os.path.join(OUT_DIR, "quantile_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("  Done. Outputs -> s2_forecasting/outputs/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
