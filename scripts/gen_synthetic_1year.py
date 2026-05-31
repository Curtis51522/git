# -*- coding: utf-8 -*-
"""Synthetic 1-year sales dataset with random daily weather.
Bootstrap sales from real DB, daily weather drawn from realistic distributions."""

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
    """Random daily weather within monthly climatology bounds."""
    m = dt.month; t_mean, r_mean, h_mean = KL_CLIMATE[m]

    # Temperature: normal around monthly mean, +/- 3C
    temp = round(np.clip(np.random.normal(t_mean, 1.8), t_mean-4, t_mean+4), 1)

    # Rainfall: exponential distribution (most days dry, occasional heavy)
    # Scale such that mean matches monthly average / 30
    daily_rate = r_mean / 30.0
    rain = round(np.random.exponential(daily_rate), 1)

    # Humidity: normal around mean
    hum = round(np.clip(np.random.normal(h_mean, 5), 60, 98), 1)

    # Weather type from daily rain amount
    if rain > 15:   wt = 'rainy'
    elif rain > 3:  wt = 'cloudy'
    else:           wt = 'sunny'

    return {
        'temperature': temp, 'rainfall': rain, 'humidity': hum,
        'is_rainy': 1 if rain > 5 else 0,
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
    dow_buckets = defaultdict(list)
    for r in rows:
        dow_buckets[(r['product_name'], r['dt'].weekday())].append(float(r['qty']))
    return dow_buckets

def generate_sales(product, dow, dow_buckets):
    if dow == 0: return 0
    key = (product, dow)
    if key in dow_buckets and len(dow_buckets[key]) >= 3:
        return int(round(np.random.choice(dow_buckets[key])))
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
    print("Loading real sales patterns...")
    dow_buckets = load_real_sales()
    print(f"  {sum(len(v) for v in dow_buckets.values())} data points")

    start = date(2025, 6, 1)
    end = date(2026, 5, 31)
    all_dates = [start + timedelta(days=i) for i in range((end-start).days+1)]
    print(f"  Date range: {start} to {end} ({len(all_dates)} days)")

    records = []
    sales_lookup = {}
    weather_counts = {'sunny':0, 'cloudy':0, 'rainy':0}

    for d in all_dates:
        dow = d.weekday()
        w = get_weather(d)
        weather_counts[w['weather_type']] += 1
        for prod in PRODUCTS:
            sales = generate_sales(prod, dow, dow_buckets)
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
    print(f"  Weather: {weather_counts}")
    print(f"  Generated {len(df)} rows")

    print("Computing lag features...")
    for i, row in df.iterrows():
        prod = row['product']; d = row['date']
        df.at[i, 'lag_1'] = _calendar_lag(sales_lookup, prod, d, 1)
        df.at[i, 'lag_7'] = _calendar_lag(sales_lookup, prod, d, 7)
        df.at[i, 'rolling_7d_mean'] = _calendar_rolling_7d(sales_lookup, prod, d)

    non_monday = df[df['day_of_week'] != 0]
    print(f"\nPer-product mean sales (non-Monday):")
    for prod in PRODUCTS:
        pdf = df[(df['product']==prod) & (df['day_of_week'] != 0)]
        print(f"  {prod}: mean={pdf['sales'].mean():.1f}, std={pdf['sales'].std():.1f}")

    features = [
        'day_of_week','is_weekend','day_of_month','month','is_public_holiday',
        'is_ramadan','temperature','rainfall','humidity','is_rainy',
        'weather_sunny','weather_cloudy','weather_rainy','weather_storm',
        'lag_1','lag_7','rolling_7d_mean',
    ]
    print(f"\nAll 17 features: {all(f in df.columns for f in features)}")

    out_path = os.path.join(ROOT, 'data', 'synthetic_sales_1year.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}\n")

    print("Sample weather variation (first 10 days of January):")
    jan = df[(df['product']=='croissant') & (df['month']==1)].head(10)
    for _, r in jan.iterrows():
        d = r['date'].strftime('%Y-%m-%d') if hasattr(r['date'], 'strftime') else str(r['date'])
        wt = 'rainy' if r['weather_rainy'] else ('cloudy' if r['weather_cloudy'] else 'sunny')
        print(f"  {d}: {wt:7s} temp={r['temperature']:.1f}C rain={r['rainfall']:.1f}mm hum={r['humidity']:.0f}%")

if __name__=='__main__':
    main()