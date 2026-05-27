"""
XGBoost sales-forecast training -- one model per product.

Supports two data sources:
1. Supabase  (--source supabase)   pulls inventory_transactions, builds daily aggregates
2. CSV       (--source csv --data path.csv)

Output:  models/xgboost/{product_name}_model.json  (6 files for 6 products)
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from pathlib import Path
from typing import Optional, Dict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    MODEL_CACHE_DIR, CACHE_MANIFEST, PRODUCT_TYPES,
    MIN_TRAINING_DAYS,
)
from db.supabase_client import get_supabase
from api.weather import get_weather

MODEL_DIR = "models/xgboost"


# ======================================================================
# Data loading
# ======================================================================

def fetch_supabase_data(min_days: int = MIN_TRAINING_DAYS) -> pd.DataFrame:
    """Pull outflow transactions from Supabase and build daily sales aggregates."""
    supabase = get_supabase()
    r = supabase.table("inventory_transactions") \
        .select("*") \
        .eq("transaction_type", "outflow") \
        .execute()

    if not r.data:
        raise ValueError(
            "No outflow transactions found in Supabase. "
            "Run the POS checkout flow first to generate sales data."
        )

    df = pd.DataFrame(r.data)
    df["transaction_date"] = pd.to_datetime(df["transaction_time"]).dt.date
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    # Aggregate to daily sales per product
    daily = df.groupby(["transaction_date", "product_name"]) \
              .agg(quantity_sold=("quantity", "sum")) \
              .reset_index()

    days_count = (daily["transaction_date"].max() - daily["transaction_date"].min()).days
    if days_count < min_days:
        print(
            f"[XGBoost] WARNING: only {days_count} days of data available "
            f"(minimum {min_days} recommended). Forecasts may be unreliable."
        )

    print(f"[XGBoost] Loaded {len(daily)} daily-sales rows from Supabase "
          f"({daily['product_name'].nunique()} products, {days_count} days)")
    return daily


def load_csv(path: str) -> pd.DataFrame:
    """Load from CSV.  Must have columns: transaction_date, product_name, quantity_sold."""
    df = pd.read_csv(path, parse_dates=["transaction_date"])
    required = {"transaction_date", "product_name", "quantity_sold"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {required}")
    return df[list(required)]


# ======================================================================
# Feature engineering
# ======================================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features to a daily-sales DataFrame."""
    df = df.copy()
    df["day_of_week"]     = df["transaction_date"].dt.dayofweek
    df["is_weekend"]      = df["day_of_week"].isin([5, 6]).astype(int)
    df["day_of_month"]    = df["transaction_date"].dt.day
    df["month"]           = df["transaction_date"].dt.month
    df["discount_rate"]   = 0.0
    df["is_public_holiday"] = 0
    df["temperature"]     = df["transaction_date"].apply(lambda d: get_weather(d)[0])
    df["rainfall"]        = df["transaction_date"].apply(lambda d: get_weather(d)[1])
    return df


# ======================================================================
# Per-product training
# ======================================================================

def train_one_product(
    product_df: pd.DataFrame,
    product_name: str,
) -> Dict:
    """Train an XGBoost model for a single product."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = build_features(product_df).sort_values("transaction_date")

    feature_cols = [
        "day_of_week", "is_weekend", "day_of_month", "month",
        "discount_rate", "is_public_holiday", "temperature", "rainfall",
    ]
    X = df[feature_cols].fillna(0)
    y = df["quantity_sold"]

    n_samples = len(df)
    if n_samples < 20:
        print(f"  [{product_name}] only {n_samples} rows -- using simple mean fallback")
        mean_val = float(y.mean())
        model_path = os.path.join(MODEL_DIR, f"{product_name}_model.json")
        model = xgb.XGBRegressor(n_estimators=10, max_depth=2)
        model.fit(X, y)
        model.save_model(model_path)
        return {"product": product_name, "samples": n_samples, "fallback_mean": mean_val}

    tscv = TimeSeriesSplit(n_splits=min(3, n_samples // 10))
    metrics_list = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = np.maximum(model.predict(X_test), 0)

        if len(y_test) > 0 and y_test.sum() > 0:
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae  = mean_absolute_error(y_test, y_pred)
            wape = np.sum(np.abs(y_test - y_pred)) / np.sum(y_test) * 100
            metrics_list.append({"MAPE": mape, "RMSE": rmse, "MAE": mae, "WAPE": wape})

    # Final model on all data
    final_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    final_model.fit(X, y)
    model_path = os.path.join(MODEL_DIR, f"{product_name}_model.json")
    final_model.save_model(model_path)

    if metrics_list:
        avg = {k: round(np.mean([m[k] for m in metrics_list]), 2) for k in metrics_list[0]}
    else:
        avg = {"MAPE": 0, "RMSE": 0, "MAE": 0, "WAPE": 0}

    print(f"  [{product_name}] samples={n_samples} MAPE={avg['MAPE']}% WAPE={avg['WAPE']}%")
    return {"product": product_name, "samples": n_samples, "metrics": avg}


def train_all_products(df: pd.DataFrame) -> Dict:
    """Train one model per product found in the data."""
    products_in_data = df["product_name"].unique()
    results = {}

    for product in products_in_data:
        if product not in PRODUCT_TYPES:
            print(f"  [{product}] skipping -- not in PRODUCT_TYPES")
            continue
        product_df = df[df["product_name"] == product].copy()
        if len(product_df) == 0:
            continue
        results[product] = train_one_product(product_df, product)

    return results


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["supabase", "csv"], default="supabase")
    parser.add_argument("--data", help="CSV path (required for --source csv)")
    args = parser.parse_args()

    if args.source == "csv":
        if not args.data:
            parser.error("--data is required for --source csv")
        df = load_csv(args.data)
    else:
        df = fetch_supabase_data()

    results = train_all_products(df)
    print(f"\n[XGBoost] Trained {len(results)} product models -> {MODEL_DIR}/")
