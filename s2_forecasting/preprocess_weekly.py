#!/usr/bin/env python
"""S2 Weekly Aggregation Preprocessing (2026-06-30)
===================================================
Aggregates daily bakery_sales_raw.csv to weekly level.
Features adapted for weekly prediction:
  product_id, category, weekly_tickets, week_of_year, month,
  is_holiday_week, lag_1w, lag_4w_avg, lag_8w_avg,
  is_day1_w, is_top3_w, discount_pct_w,
  is_member_week, is_new_product_w, is_competitor_w, is_rainy_w, quantity_w

Split (time-series):
  Train: 2023-01-01 to 2024-12-31
  Val:   2025-01-01 to 2025-06-30 (26 weeks)
  Test:  2025-07-01 to 2026-06-29 (26 weeks)
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

WEEKLY_FEATURES = [
    "product_id", "category", "weekly_tickets", "week_of_year", "month",
    "is_holiday_week",
    "lag_1w", "lag_4w_avg", "lag_8w_avg",
    "is_day1_w", "is_top3_w", "discount_pct_w",
    "is_member_week", "is_new_product_w", "is_competitor_w", "is_rainy_w",
]
TARGET_COL = "quantity_w"


def run_weekly_preprocessing(verbose=True):
    p = print if verbose else lambda *a, **kw: None

    # ---- Stage 1: Load raw ----
    p("=" * 50)
    p("  STAGE 1: Load Raw Data")
    p("=" * 50)
    df_raw = pd.read_csv(RAW_CSV, parse_dates=["date"])
    for col in ["is_day1", "is_top3", "discount_pct", "is_member_day",
                 "is_new_product", "is_competitor", "is_rainy"]:
        if col not in df_raw.columns:
            df_raw[col] = 0
        df_raw[col] = df_raw[col].fillna(0)
    p(f"  Rows: {len(df_raw):,}  |  Tickets: {df_raw['ticket_id'].nunique():,}")
    p(f"  Products: {df_raw['product_name'].nunique()}")
    p(f"  Date range: {df_raw['date'].min().date()} to {df_raw['date'].max().date()}")

    # ---- Stage 2: Weekly Aggregation ----
    p()
    p("=" * 50)
    p("  STAGE 2: Weekly Aggregation")
    p("=" * 50)

    # Create week identifier (ISO week: Mon-Sun)
    df_raw["week_start"] = df_raw["date"] - pd.to_timedelta(df_raw["date"].dt.dayofweek, unit="D")
    df_raw["week_start"] = df_raw["week_start"].dt.strftime("%Y-%m-%d")

    weekly = df_raw.groupby(["week_start", "product_name"]).agg(
        quantity_w=("quantity", "sum"),
        is_day1_w=("is_day1", "max"),
        is_top3_w=("is_top3", "max"),
        discount_pct_w=("discount_pct", "mean"),
        is_member_week=("is_member_day", "max"),
        is_new_product_w=("is_new_product", "max"),
        is_competitor_w=("is_competitor", "max"),
        is_rainy_w=("is_rainy", "mean"),
    ).reset_index()

    weekly["week_start"] = pd.to_datetime(weekly["week_start"])

    # Fill all product x week combinations with 0
    all_weeks = sorted(weekly["week_start"].unique())
    all_products = sorted(weekly["product_name"].unique())
    full_index = pd.MultiIndex.from_product(
        [all_weeks, all_products], names=["week_start", "product_name"])
    weekly_full = pd.DataFrame(index=full_index).reset_index()
    weekly_full = weekly_full.merge(weekly, on=["week_start", "product_name"], how="left")
    weekly_full["quantity_w"] = weekly_full["quantity_w"].fillna(0).astype(int)
    for col in ["is_day1_w", "is_top3_w", "discount_pct_w", "is_member_week",
                 "is_new_product_w", "is_competitor_w", "is_rainy_w"]:
        weekly_full[col] = weekly_full[col].fillna(0)

    # Product ID and category
    pid_map = {p: i for i, p in enumerate(all_products)}
    weekly_full["product_id"] = weekly_full["product_name"].map(pid_map)
    weekly_full["category"] = (weekly_full["product_id"] >= 30).astype(int)

    # Weekly ticket count
    weekly_tickets = df_raw.groupby("week_start")["ticket_id"].nunique().reset_index()
    weekly_tickets.columns = ["week_start", "weekly_tickets"]
    weekly_tickets["week_start"] = pd.to_datetime(weekly_tickets["week_start"])
    weekly_full = weekly_full.merge(weekly_tickets, on="week_start", how="left")
    weekly_full["weekly_tickets"] = weekly_full["weekly_tickets"].fillna(0).astype(int)

    p(f"  Rows: {len(weekly_full):,}  |  Weeks: {weekly_full['week_start'].nunique()}")
    p(f"  Zero-sales weeks: {(weekly_full['quantity_w']==0).mean()*100:.1f}%")

    # ---- Stage 3: Calendar Features ----
    p()
    p("=" * 50)
    p("  STAGE 3: Calendar Features")
    p("=" * 50)

    weekly_full["week_of_year"] = weekly_full["week_start"].dt.isocalendar().week.astype(int)
    weekly_full["month"] = weekly_full["week_start"].dt.month

    # Holiday week: week contains a Chinese holiday
    from chinese_calendar import is_holiday as chinese_holiday
    holiday_weeks = set()
    for d in pd.date_range(weekly_full["week_start"].min(),
                            weekly_full["week_start"].max() + pd.Timedelta(days=6)):
        try:
            if chinese_holiday(d):
                ws = (d - pd.Timedelta(days=d.dayofweek)).strftime("%Y-%m-%d")
                holiday_weeks.add(ws)
        except Exception:
            pass
    weekly_full["ws_str"] = weekly_full["week_start"].dt.strftime("%Y-%m-%d")
    weekly_full["is_holiday_week"] = weekly_full["ws_str"].isin(holiday_weeks).astype(int)
    p(f"  Holiday weeks: {weekly_full[weekly_full['is_holiday_week']==1]['ws_str'].nunique()}")

    # ---- Stage 4: Weekly Lag Features ----
    p()
    p("=" * 50)
    p("  STAGE 4: Weekly Lag Features")
    p("=" * 50)

    df = weekly_full.sort_values(["product_id", "week_start"]).copy()
    # lag_1w: previous week
    df["lag_1w"] = df.groupby("product_id")["quantity_w"].shift(1).fillna(0)
    # lag_4w_avg: average of last 4 weeks
    df["lag_4w_avg"] = df.groupby("product_id")["quantity_w"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean()).fillna(0)
    # lag_8w_avg: average of last 8 weeks
    df["lag_8w_avg"] = df.groupby("product_id")["quantity_w"].transform(
        lambda x: x.shift(1).rolling(8, min_periods=1).mean()).fillna(0)

    for col in ["lag_1w", "lag_4w_avg", "lag_8w_avg"]:
        p(f"  {col}: mean={df[col].mean():.1f}")

    # ---- Stage 5: Split ----
    p()
    p("=" * 50)
    p("  STAGE 5: Train / Val / Test Split")
    p("=" * 50)

    df["week_start"] = df["week_start"].dt.strftime("%Y-%m-%d")
    df = df.drop(columns=["ws_str"], errors="ignore")

    output_cols = ["week_start"] + WEEKLY_FEATURES + [TARGET_COL]
    train_df = df[df["week_start"] < SPLIT_DATE][output_cols].copy()
    val_df = df[(df["week_start"] >= SPLIT_DATE) & (df["week_start"] < SPLIT_DATE_2)][output_cols].copy()
    test_df = df[df["week_start"] >= SPLIT_DATE_2][output_cols].copy()

    p(f"  Train: {len(train_df):,}  ({train_df['week_start'].min()} to {train_df['week_start'].max()})")
    p(f"  Val:   {len(val_df):,}  ({val_df['week_start'].min()} to {val_df['week_start'].max()})")
    p(f"  Test:  {len(test_df):,}  ({test_df['week_start'].min()} to {test_df['week_start'].max()})")

    p()
    p("  WEEKLY PREPROCESSING DONE")
    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = run_weekly_preprocessing(verbose=True)
    for name, df in [("weekly_train", train_df), ("weekly_val", val_df), ("weekly_test", test_df)]:
        path = os.path.join(DATA_DIR, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved: {path}  ({len(df):,} rows)")
