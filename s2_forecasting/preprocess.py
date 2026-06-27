#!/usr/bin/env python
"""
S2 Preprocessing Module
========================
Importable module: `from preprocess import run_preprocessing`
Returns (train_df, val_df, test_df) ready for model training.

Stages:
  1. Load raw ticket-level CSV
  2. EDA (summary statistics)
  3. Aggregate to daily per-product
  4. Weather (Open-Meteo historical, Guangzhou)
  5. Chinese holidays (chinese-calendar)
  6. Lag features (lag_1, lag_7_avg, lag_30_avg)
  7. Train/val/test split by date
"""

import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import os as _os
BASE_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA_DIR = _os.path.join(BASE_DIR, "data")
RAW_CSV = os.path.join(DATA_DIR, "bakery_sales_raw.csv")
GZ_LAT = 23.1291
GZ_LON = 113.2644


# Data split dates (time-series, no shuffle)
SPLIT_DATE = "2023-01-01"    # train -> val
SPLIT_DATE_2 = "2023-07-01"  # val -> test

FEATURE_COLS = [
    "product_id", "temp_mean", "temp_max", "temp_min",
    "humidity", "precipitation", "day_of_week", "month",
    "is_weekend", "is_holiday", "lag_1", "lag_7_avg", "lag_30_avg",
]
TARGET_COL = "quantity"


def run_preprocessing(verbose=True):
    """
    Full S2 preprocessing pipeline.

    Returns: (train_df, val_df, test_df) as DataFrames with feature columns.
    """
    p = print if verbose else lambda *a, **kw: None

    # ---- Stage 1: Load raw ----
    p("=" * 50)
    p("  STAGE 1: Load Raw Ticket-Level Data")
    p("=" * 50)
    df_raw = pd.read_csv(RAW_CSV, parse_dates=["date"])
    p(f"  Rows: {len(df_raw):,}  |  Tickets: {df_raw['ticket_id'].nunique():,}")
    p(f"  Products: {df_raw['product_name'].nunique()}  |  Date: {df_raw['date'].min().date()} to {df_raw['date'].max().date()}")

    # ---- Stage 2: EDA ----
    p("\n" + "=" * 50)
    p("  STAGE 2: EDA")
    p("=" * 50)
    daily_total = df_raw.groupby("date")["quantity"].sum()
    p(f"  Daily units: mean={daily_total.mean():.0f}  min={daily_total.min():.0f}  max={daily_total.max():.0f}  std={daily_total.std():.0f}")
    top5 = df_raw.groupby("product_name")["quantity"].sum().nlargest(5)
    p(f"  Top products: {', '.join(f'{n}({int(v)})' for n,v in top5.items())}")

    # ---- Stage 3: Aggregate to daily ----
    p("\n" + "=" * 50)
    p("  STAGE 3: Daily Aggregation")
    p("=" * 50)
    daily = df_raw.groupby(["date", "product_name"])["quantity"].sum().reset_index()
    all_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    all_products = sorted(daily["product_name"].unique())
    full_index = pd.MultiIndex.from_product([all_dates, all_products], names=["date", "product_name"])
    daily_full = pd.DataFrame(index=full_index).reset_index()
    daily_full = daily_full.merge(daily, on=["date", "product_name"], how="left")
    daily_full["quantity"] = daily_full["quantity"].fillna(0).astype(int)
    pid_map = {p: i for i, p in enumerate(all_products)}
    daily_full["product_id"] = daily_full["product_name"].map(pid_map)
    p(f"  Rows: {len(daily_full):,}  |  Zero-sales: {(daily_full['quantity']==0).mean()*100:.1f}%")

    # ---- Stage 4: Weather ----
    p("\n" + "=" * 50)
    p("  STAGE 4: Weather (Guangzhou, Open-Meteo)")
    p("=" * 50)
    weather = _fetch_weather(daily_full["date"].min(), daily_full["date"].max())
    daily_full["date_str"] = daily_full["date"].dt.strftime("%Y-%m-%d")
    for col, default in [("temp_mean", 22.0), ("temp_max", 26.0), ("temp_min", 18.0),
                          ("precipitation", 0.0), ("humidity", 75.0)]:
        daily_full[col] = daily_full["date_str"].map(lambda d: weather.get(d, {}).get(col, default))
    p(f"  temp: {daily_full['temp_mean'].min():.0f}-{daily_full['temp_mean'].max():.0f}C  |  precip>0: {(daily_full['precipitation']>0).mean()*100:.0f}%")

    # ---- Stage 5: Holidays ----
    p("\n" + "=" * 50)
    p("  STAGE 5: Chinese Holidays")
    p("=" * 50)
    from chinese_calendar import is_holiday as chinese_holiday
    holiday_dates = {}
    for d in pd.date_range(daily_full["date"].min(), daily_full["date"].max()):
        try:
            holiday_dates[d.strftime("%Y-%m-%d")] = int(chinese_holiday(d))
        except:
            holiday_dates[d.strftime("%Y-%m-%d")] = 0
    daily_full["day_of_week"] = daily_full["date"].dt.dayofweek
    daily_full["month"] = daily_full["date"].dt.month
    daily_full["is_weekend"] = (daily_full["day_of_week"] >= 5).astype(int)
    daily_full["is_holiday"] = daily_full["date_str"].map(holiday_dates).fillna(0).astype(int)
    p(f"  Holiday dates: {daily_full[daily_full['is_holiday']==1]['date_str'].nunique()}")

    # ---- Stage 6: Lag features ----
    p("\n" + "=" * 50)
    p("  STAGE 6: Lag Features")
    p("=" * 50)
    df = daily_full.sort_values(["product_id", "date"]).copy()
    df["lag_1"] = df.groupby("product_id")["quantity"].shift(1).fillna(0)
    df["lag_7_avg"] = df.groupby("product_id")["quantity"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()).fillna(0)
    df["lag_30_avg"] = df.groupby("product_id")["quantity"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=1).mean()).fillna(0)
    p(f"  lag_1 mean={df['lag_1'].mean():.1f}  lag_7_avg mean={df['lag_7_avg'].mean():.1f}  lag_30_avg mean={df['lag_30_avg'].mean():.1f}")

    # ---- Stage 7: Split ----
    p("\n" + "=" * 50)
    p("  STAGE 7: Train / Val / Test Split")
    p("=" * 50)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    output_cols = ["date"] + FEATURE_COLS + [TARGET_COL]
    train_df = df[df["date"] < SPLIT_DATE][output_cols].copy()
    val_df = df[(df["date"] >= SPLIT_DATE) & (df["date"] < SPLIT_DATE_2)][output_cols].copy()
    test_df = df[df["date"] >= SPLIT_DATE_2][output_cols].copy()
    p(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")
    p(f"  Train range: {train_df['date'].min()} to {train_df['date'].max()}")
    p(f"  Test range:  {test_df['date'].min()} to {test_df['date'].max()}")

    p("\n" + "=" * 50)
    p("  PREPROCESSING DONE")
    p("=" * 50)
    return train_df, val_df, test_df


def _fetch_weather(start_date, end_date):
    """Fetch Guangzhou weather from Open-Meteo. Falls back to existing CSV."""
    weather = {}
    try:
        import httpx
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={GZ_LAT}&longitude={GZ_LON}"
            f"&start_date={start_date.strftime('%Y-%m-%d')}"
            f"&end_date={end_date.strftime('%Y-%m-%d')}"
            f"&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,"
            f"precipitation_sum,relative_humidity_2m_mean"
            f"&timezone=Asia/Shanghai"
        )
        resp = httpx.get(url, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()["daily"]
        for i, d in enumerate(data["time"]):
            weather[d] = {
                "temp_mean": round(data["temperature_2m_mean"][i] or 22.0, 1),
                "temp_max": round(data["temperature_2m_max"][i] or 26.0, 1),
                "temp_min": round(data["temperature_2m_min"][i] or 18.0, 1),
                "precipitation": round(data["precipitation_sum"][i] or 0.0, 1),
                "humidity": round(data["relative_humidity_2m_mean"][i] or 75.0, 1),
            }
        print(f"  Fetched {len(weather)} days from Open-Meteo")
    except Exception as e:
        print(f"  Open-Meteo failed ({e}), using fallback...")
        fallback = pd.read_csv(os.path.join(DATA_DIR, "training_data_45products.csv"))
        wdf = fallback[["date", "temp_mean", "temp_max", "temp_min", "humidity", "precipitation"]].drop_duplicates()
        for _, row in wdf.iterrows():
            weather[row["date"]] = {
                "temp_mean": row["temp_mean"], "temp_max": row["temp_max"],
                "temp_min": row["temp_min"], "precipitation": row["precipitation"],
                "humidity": row["humidity"],
            }
        print(f"  Extracted {len(weather)} days from fallback")
    return weather


# Standalone run: save CSVs to disk
if __name__ == "__main__":
    train_df, val_df, test_df = run_preprocessing(verbose=True)
    for name, df in [("xgboost_train", train_df), ("xgboost_val", val_df), ("xgboost_test", test_df)]:
        path = os.path.join(DATA_DIR, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved: {path}  ({len(df):,} rows)")
