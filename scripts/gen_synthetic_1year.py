# -*- coding: utf-8 -*-
"""Synthetic 1-year sales dataset generator.
Bootstrap-sampled from real 47-day DB data, stratified by day_of_week + weather.
Monday = shop closed (0 sales). All 17 features included. No hand-crafted formulas.
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
import holidays
from datetime import date, timedelta
from collections import defaultdict
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PRODUCTS = ['croissant','donut','chiffon','bread_coconut','bread_roll','croissant_chocolate']
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

_my_holidays = holidays.MY()

KL_CLIMATE = {
    1:(27.3,226.7,80),2:(27.8,192.8,80),3:(28.1,270.4,80),
    4:(28.1,301.5,82),5:(28.5,229.9,81),6:(28.4,145.8,80),
    7:(28.0,165.2,79),8:(28.0,174.3,79),9:(27.7,220.3,81),
    10:(27.5,283.8,82),11:(27.1,355.8,84),12:(27.1,280.6,83),
}

def is_ramadan(d):
    try:
        from hijri_converter import convert
        h = convert.Gregorian(d.year,d.month,d.day).to_hijri()
        return 1 if h.month == 9 else 0
    except: return 0

def get_weather(dt):
    m = dt.month; t, r, h = KL_CLIMATE[m]
    temp = round(t + np.random.normal(0, 1.5), 1)
    rain = round(max(0, r + np.random.normal(0, r * 0.15)), 1)
    hum = round(np.clip(h + np.random.normal(0, 3), 60, 95), 1)
    if rain > 250: wt = 'rainy'
    elif rain > 180: wt = 'cloudy'
    else: wt = 'sunny'
    return {
        'temperature': temp, 'rainfall': rain, 'humidity': hum,
        'is_rainy': 1 if rain > 100 else 0,
        'weather_sunny': 1 if wt=='sunny' else 0,
        'weather_cloudy': 1 if wt=='cloudy' else 0,
        'weather_rainy': 1 if wt=='rainy' else 0,
        'weather_storm': 0, 'weather_type': wt,
    }

def load_real_sales():
    from db.mysql_client import get_db
    db = get_db(); c = db.cursor(dictionary=True)
    c.execute(
        "SELECT product_name, DATE(transaction_time) as dt, SUM(quantity) as qty "
        "FROM inventory_transactions WHERE transaction_type=%s "
        "AND product_name IN (%s,%s,%s,%s,%s,%s) "
        "GROUP BY product_name, DATE(transaction_time) ORDER BY dt",
        ('outflow',)+tuple(PRODUCTS)
    )
    rows = c.fetchall()
    buckets = defaultdict(list)
    for r in rows:
        d = r['dt']; m = d.month
        _, rain, _ = KL_CLIMATE[m]
        if rain > 250: wt = 'rainy'
        elif rain > 180: wt = 'cloudy'
        else: wt = 'sunny'
        buckets[(r['product_name'], d.weekday(), wt)].append(float(r['qty']))
    dow_buckets = defaultdict(list)
    for r in rows:
        dow_buckets[(r['product_name'], r['dt'].weekday())].append(float(r['qty']))
    return buckets, dow_buckets

def generate_sales(product, dow, weather_type, buckets, dow_buckets):
    key = (product, dow, weather_type)
    if key in buckets and len(buckets[key]) >= 3:
        return int(round(np.random.choice(buckets[key])))
    dow_key = (product, dow)
    if dow_key in dow_buckets and len(dow_buckets[dow_key]) >= 3:
        return int(round(np.random.choice(dow_buckets[dow_key])))
    all_vals = [v for (p, _, _), vals in buckets.items() for v in vals if p == product]
    if all_vals:
        return int(round(np.random.choice(all_vals)))
    return 0

def _calendar_lag(sales_lookup, prod, fd, days_back):
    target = fd - timedelta(days=days_back)
    for _ in range(4):
        key = target.strftime('%Y-%m-%d')
        if (prod, key) in sales_lookup:
            return sales_lookup[(prod, key)]
        target -= timedelta(days=1)
    return 0.0

def _calendar_rolling_7d(sales_lookup, prod, fd):
    vals = []
    for offset in range(1, 8):
        target = fd - timedelta(days=offset)
        key = target.strftime('%Y-%m-%d')
        if (prod, key) in sales_lookup:
            vals.append(sales_lookup[(prod, key)])
    return float(sum(vals) / len(vals)) if vals else 0.0

def main():
    print("Loading real 47-day sales patterns...")
    buckets, dow_buckets = load_real_sales()
    total_real = sum(len(v) for v in buckets.values())
    print(f"  Real data: {total_real} points across {len(buckets)} buckets")

    start = date(2025, 6, 1)
    end = date(2026, 5, 31)
    all_dates = []
    d = start
    while d <= end:
        all_dates.append(d)
        d += timedelta(days=1)
    print(f"  Date range: {start} to {end} ({len(all_dates)} days)")

    records = []
    sales_lookup = {}
    for d in all_dates:
        dow = d.weekday()
        w = get_weather(d)
        for prod in PRODUCTS:
            sales = 0 if dow == 0 else generate_sales(prod, dow, w['weather_type'], buckets, dow_buckets)
            key = d.strftime('%Y-%m-%d')
            sales_lookup[(prod, key)] = float(sales)
            records.append({
                'date': d, 'product': prod, 'sales': sales,
                'day_of_week': dow, 'is_weekend': 1 if dow >= 5 else 0,
                'day_of_month': d.day, 'month': d.month,
                'is_public_holiday': 1 if d in _my_holidays else 0,
                'is_ramadan': is_ramadan(d),
                'temperature': w['temperature'], 'rainfall': w['rainfall'],
                'humidity': w['humidity'], 'is_rainy': w['is_rainy'],
                'weather_sunny': w['weather_sunny'], 'weather_cloudy': w['weather_cloudy'],
                'weather_rainy': w['weather_rainy'], 'weather_storm': w['weather_storm'],
                'lag_1': 0.0, 'lag_7': 0.0, 'rolling_7d_mean': 0.0,
            })

    df = pd.DataFrame(records)
    print(f"  Generated {len(df)} rows")

    print("Computing lag features (calendar-day logic)...")
    for i, row in df.iterrows():
        prod = row['product']
        d = row['date']
        df.at[i, 'lag_1'] = _calendar_lag(sales_lookup, prod, d, 1)
        df.at[i, 'lag_7'] = _calendar_lag(sales_lookup, prod, d, 7)
        df.at[i, 'rolling_7d_mean'] = _calendar_rolling_7d(sales_lookup, prod, d)

    non_monday = df[df['day_of_week'] != 0]
    print(f"\n=== Dataset Summary ===")
    print(f"Total rows: {len(df)} (365d x 6 products)")
    print(f"Monday (closed) rows: {len(df[df['day_of_week'] == 0])}")
    print(f"\nPer-product mean sales (non-Monday):")
    for prod in PRODUCTS:
        pdf = df[(df['product'] == prod) & (df['day_of_week'] != 0)]
        print(f"  {prod}: mean={pdf['sales'].mean():.1f}, std={pdf['sales'].std():.1f}")

    features = [
        'day_of_week','is_weekend','day_of_month','month','is_public_holiday',
        'is_ramadan','temperature','rainfall','humidity','is_rainy',
        'weather_sunny','weather_cloudy','weather_rainy','weather_storm',
        'lag_1','lag_7','rolling_7d_mean',
    ]
    all_ok = all(f in df.columns for f in features)
    print(f"\nAll 17 features present: {all_ok}")
    print(f"Any NaN in features: {df[features].isna().any().any()}")

    out_path = os.path.join(ROOT, 'data', 'synthetic_sales_1year.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"\n=== First 5 rows ===")
    print(df.head(5).to_string())

if __name__ == '__main__':
    main()
