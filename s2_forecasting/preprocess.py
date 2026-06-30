#!/usr/bin/env python
"""S2 Preprocessing Module — Clean Rewrite (2026-06-30)
=========================================================
Feature set (17 features):
  product_id, category, daily_tickets, day_of_week, month,
  is_weekend, is_holiday (pure public holiday, not weekend),
  lag_1, lag_7_avg, lag_30_avg, is_day1, is_top3, discount_pct,
  is_member_day, is_new_product, is_competitor, is_rainy

Data split (time-series, no shuffle):
  Train: 2023-01-01 to 2024-12-31
  Val:   2025-01-01 to 2025-06-30
  Test:  2025-07-01 to 2026-06-29
"""

import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_CSV = os.path.join(DATA_DIR, "bakery_sales_raw.csv")

SPLIT_DATE = "2025-01-01"
SPLIT_DATE_2 = "2025-07-01"

FEATURE_COLS = [
    "product_id", "category", "daily_tickets", "day_of_week", "month",
    "is_weekend", "is_holiday",
    "lag_1", "lag_7_avg", "lag_30_avg", "roll_std_7", "roll_std_14", "trend_7",
    "is_day1", "is_top3", "discount_pct",
    "is_member_day", "is_new_product", "is_competitor", "is_rainy",
    "temp_mean", "temp_range", "is_cold_day", "is_hot_day",
    "large_ratio", "cold_ratio", "sweetness_avg", "ice_avg", "temp_hot_ratio",
]
TARGET_COL = "quantity"


def run_preprocessing(verbose=True):
    p = print if verbose else lambda *a, **kw: None

    # ---- Stage 1: Load raw ----
    p("=" * 50)
    p("  STAGE 1: Load Raw Data")
    p("=" * 50)
    df_raw = pd.read_csv(RAW_CSV, parse_dates=["date"])
    for col in ["is_day1", "is_top3", "discount_pct", "is_member_day", "is_new_product", "is_competitor", "is_rainy"]:
        if col not in df_raw.columns:
            df_raw[col] = 0
        df_raw[col] = df_raw[col].fillna(0)
    p(f"  Rows: {len(df_raw):,}  |  Tickets: {df_raw['ticket_id'].nunique():,}")
    p(f"  Products: {df_raw['product_name'].nunique()}  |  Date: {df_raw['date'].min().date()} to {df_raw['date'].max().date()}")
    p(f"  Day-1 rows: {(df_raw['is_day1']==1).sum():,}  |  Top-3 rows: {(df_raw['is_top3']==1).sum():,}")

    # ---- Stage 2: EDA ----
    p()
    p("=" * 50)
    p("  STAGE 2: EDA")
    p("=" * 50)
    daily_total = df_raw.groupby("date")["quantity"].sum()
    p(f"  Daily units: mean={daily_total.mean():.0f}  std={daily_total.std():.0f}")

    # ---- Stage 3: Aggregate to daily ----
    p()
    p("=" * 50)
    p("  STAGE 3: Daily Aggregation")
    p("=" * 50)
    daily = df_raw.groupby(["date", "product_name"]).agg(
        quantity=("quantity", "sum"),
        is_day1=("is_day1", "max"),
        is_top3=("is_top3", "max"),
        discount_pct=("discount_pct", "max"),
        is_member_day=("is_member_day", "max"),
        is_new_product=("is_new_product", "max"),
        is_competitor=("is_competitor", "max"),
        is_rainy=("is_rainy", "max"),
        large_cnt=("beverage_size", lambda x: (x == 'large').sum()),
        bev_cnt=("beverage_size", lambda x: (x != '').sum()),
        cold_cnt=("beverage_temp", lambda x: (x == 'cold').sum()),
        sugar_sum=("beverage_sweetness", lambda x: x.map({'normal':3,'less':2,'slight':1,'none':0}).sum()),
        sugar_cnt=("beverage_sweetness", lambda x: (x != '').sum()),
        ice_sum=("beverage_ice", lambda x: x.map({'normal':2,'less':1,'none':0}).sum()),
        ice_cnt=("beverage_ice", lambda x: (x != '').sum()),
    ).reset_index()

    all_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    all_products = sorted(daily["product_name"].unique())
    full_index = pd.MultiIndex.from_product([all_dates, all_products], names=["date", "product_name"])
    daily_full = pd.DataFrame(index=full_index).reset_index()
    daily_full = daily_full.merge(daily, on=["date", "product_name"], how="left")
    daily_full["quantity"] = daily_full["quantity"].fillna(0).astype(int)
    for col in ["is_day1", "is_top3", "discount_pct", "is_member_day", "is_new_product", "is_competitor", "is_rainy"]:
        daily_full[col] = daily_full[col].fillna(0)
    for col in ["large_cnt", "bev_cnt", "cold_cnt", "sugar_sum", "sugar_cnt", "ice_sum", "ice_cnt"]:
        daily_full[col] = daily_full[col].fillna(0)

    pid_map = {p: i for i, p in enumerate(all_products)}
    daily_full["product_id"] = daily_full["product_name"].map(pid_map)
    daily_full["category"] = (daily_full["product_id"] >= 30).astype(int)

    # Daily ticket count (traffic proxy)
    daily_tickets = df_raw.groupby("date")["ticket_id"].nunique().reset_index()
    daily_tickets.columns = ["date", "daily_tickets"]
    daily_full = daily_full.merge(daily_tickets, on="date", how="left")
    daily_full["daily_tickets"] = daily_full["daily_tickets"].fillna(0).astype(int)

    p(f"  Rows: {len(daily_full):,}  |  Zero-sales: {(daily_full['quantity']==0).mean()*100:.1f}%")
    p(f"  Daily tickets: mean={daily_full['daily_tickets'].mean():.0f}")

    # ---- Stage 3b: Weather features ----
    p()
    p("=" * 50)
    p("  STAGE 3b: Guangzhou Weather Features")
    p("=" * 50)
    weather_path = os.path.join(DATA_DIR, "guangzhou_weather.csv")
    if os.path.exists(weather_path):
        weather = pd.read_csv(weather_path, parse_dates=["date"])
        weather["date"] = weather["date"].dt.strftime("%Y-%m-%d")
        daily_full["date_str"] = daily_full["date"].dt.strftime("%Y-%m-%d")
        daily_full = daily_full.merge(weather, left_on="date_str", right_on="date", how="left", suffixes=("", "_w"))
        daily_full["temp_mean"] = daily_full["temp_mean"].fillna(20)
        daily_full["temp_range"] = daily_full["temp_max"].fillna(28) - daily_full["temp_min"].fillna(22)
        daily_full["is_cold_day"] = (daily_full["temp_mean"] < 15).astype(int)
        daily_full["is_hot_day"] = (daily_full["temp_mean"] > 25).astype(int)
        p(f"  Weather merged: temp_mean {daily_full['temp_mean'].mean():.1f}C, cold days {(daily_full['is_cold_day']==1).mean()*100:.1f}%, hot days {(daily_full['is_hot_day']==1).mean()*100:.1f}%")
    
        # Derived beverage aggregate features
        daily_full['large_ratio'] = np.where(daily_full['bev_cnt'] > 0, daily_full['large_cnt'] / daily_full['bev_cnt'], 0)
        daily_full['cold_ratio'] = np.where(daily_full['bev_cnt'] > 0, daily_full['cold_cnt'] / daily_full['bev_cnt'], 0)
        daily_full['sweetness_avg'] = np.where(daily_full['sugar_cnt'] > 0, daily_full['sugar_sum'] / daily_full['sugar_cnt'], 0)
        daily_full['ice_avg'] = np.where(daily_full['ice_cnt'] > 0, daily_full['ice_sum'] / daily_full['ice_cnt'], 0)
        # temp_hot_ratio: continuous sigmoid for hot-drink tendency
        daily_full['temp_hot_ratio'] = 0.15 + 0.70 / (1 + np.exp((daily_full['temp_mean'] - 22) / 4))
        p(f"  Beverage features: large_ratio mean={daily_full['large_ratio'].mean():.2f}, cold_ratio mean={daily_full['cold_ratio'].mean():.2f}")
        p(f"  temp_hot_ratio: mean={daily_full['temp_hot_ratio'].mean():.2f}, range=[{daily_full['temp_hot_ratio'].min():.2f}, {daily_full['temp_hot_ratio'].max():.2f}]")
    else:
        p("  WARNING: guangzhou_weather.csv not found, using defaults")
        daily_full["temp_mean"] = 20
        daily_full["temp_range"] = 6
        daily_full["is_cold_day"] = 0
        daily_full["is_hot_day"] = 0

    # ---- Stage 4: Holidays ----
    p()
    p("=" * 50)
    p("  STAGE 4: Chinese Holidays")
    p("=" * 50)
    daily_full["date_str"] = daily_full["date"].dt.strftime("%Y-%m-%d")
    from chinese_calendar import is_holiday as chinese_holiday
    holiday_dates = {}
    for d in pd.date_range(daily_full["date"].min(), daily_full["date"].max()):
        try:
            holiday_dates[d.strftime("%Y-%m-%d")] = int(chinese_holiday(d))
        except Exception:
            holiday_dates[d.strftime("%Y-%m-%d")] = 0
    daily_full["day_of_week"] = daily_full["date"].dt.dayofweek
    daily_full["month"] = daily_full["date"].dt.month
    daily_full["is_weekend"] = (daily_full["day_of_week"] >= 5).astype(int)
    # Split: is_holiday = public holiday on weekday only (not weekend)
    daily_full["is_holiday_raw"] = daily_full["date_str"].map(holiday_dates).fillna(0).astype(int)
    daily_full["is_holiday"] = ((daily_full["is_holiday_raw"] == 1) & (daily_full["day_of_week"] < 5)).astype(int)
    daily_full = daily_full.drop(columns=["is_holiday_raw"])
    p(f"  Holiday dates: {daily_full[daily_full['is_holiday']==1]['date_str'].nunique()}")

    # ---- Stage 5: Lag features ----
    p()
    p("=" * 50)
    p("  STAGE 5: Lag Features")
    p("=" * 50)
    df = daily_full.sort_values(["product_id", "date"]).copy()
    for lag_days, col_name in [(1, "lag_1"), (7, "lag_7_avg"), (30, "lag_30_avg")]:
        if col_name == "lag_1":
            df[col_name] = df.groupby("product_id")["quantity"].shift(1).fillna(0)
        else:
            window = lag_days
            df[col_name] = df.groupby("product_id")["quantity"].transform(
                lambda x, w=window: x.shift(1).rolling(w, min_periods=1).mean()).fillna(0)

    # Rolling std (volatility)
    df["roll_std_7"] = df.groupby("product_id")["quantity"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=2).std()).fillna(0)
    df["roll_std_14"] = df.groupby("product_id")["quantity"].transform(
        lambda x: x.shift(1).rolling(14, min_periods=3).std()).fillna(0)
    # Trend
    df["trend_7"] = df["lag_1"] - df["lag_7_avg"]

    for col in ["lag_1", "lag_7_avg", "lag_30_avg", "roll_std_7", "roll_std_14", "trend_7"]:
        p(f"  {col} mean={df[col].mean():.1f}")

    # ---- Stage 6: Split ----
    p()
    p("=" * 50)
    p("  STAGE 6: Train / Val / Test Split")
    p("=" * 50)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    output_cols = ["date"] + FEATURE_COLS + [TARGET_COL]
    train_df = df[df["date"] < SPLIT_DATE][output_cols].copy()
    val_df = df[(df["date"] >= SPLIT_DATE) & (df["date"] < SPLIT_DATE_2)][output_cols].copy()
    test_df = df[df["date"] >= SPLIT_DATE_2][output_cols].copy()
    p(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")
    p(f"  Train: {train_df['date'].min()} to {train_df['date'].max()}")
    p(f"  Test:  {test_df['date'].min()} to {test_df['date'].max()}")

    p()
    p("  PREPROCESSING DONE")
    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = run_preprocessing(verbose=True)
    for name, df in [("xgboost_train", train_df), ("xgboost_val", val_df), ("xgboost_test", test_df)]:
        path = os.path.join(DATA_DIR, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved: {path}  ({len(df):,} rows)")
