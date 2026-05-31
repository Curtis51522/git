# -*- coding: utf-8 -*-
"""Synthetic 1-year sales dataset with random daily weather.
Bootstrap from real DB, stratified by (product x day_of_week x weather_type)."""

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
RANDOM_SEED = 77
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
    m = dt.month; t_mean, r_mean, h_mean = KL_CLIMATE[m]
    temp = round(np.clip(np.random.normal(t_mean, 1.8), t_mean-4, t_mean+4), 1)
    daily_rate = r_mean / 30.0
    rain = round(np.random.exponential(daily_rate), 1)
    hum = round(np.clip(np.random.normal(h_mean, 5), 60, 98), 1)
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
    # Stratified: (product, dow, weather_type) -> [sales]
    wt_buckets = defaultdict(list)
    # Fallback: (product, dow) -> [sales]
    dow_buckets = defaultdict(list)
    for r in rows:
        d = r['dt']; m = d.month
        _, rain_mean, _ = KL_CLIMATE[m]
        if rain_mean > 250: wt = 'rainy'
        elif rain_mean > 180: wt = 'cloudy'
        else: wt = 'sunny'
        wt_buckets[(r['product_name'], d.weekday(), wt)].append(float(r['qty']))
        dow_buckets[(r['product_name'], d.weekday())].append(float(r['qty']))
    return wt_buckets, dow_buckets

def generate_sales(product, dow, weather_type, wt_buckets, dow_buckets):
    if dow == 0: return 0
    key = (product, dow, weather_type)
    if key in wt_buckets and len(wt_buckets[key]) >= 2:
        return int(round(np.random.choice(wt_buckets[key])))
    # Fallback to day-of-week only
    dow_key = (product, dow)
    if dow_key in dow_buckets and len(dow_buckets[dow_key]) >= 2:
        return int(round(np.random.choice(dow_buckets[dow_key])))
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
    wt_buckets, dow_buckets = load_real_sales()
    print(f"  {sum(len(v) for v in wt_buckets.values())} points in {len(wt_buckets)} (product,dow,weather) buckets")

    start = date(2025, 6, 1)
    end = date(2026, 5, 31)
    all_dates = [start + timedelta(days=i) for i in range((end-start).days+1)]
    print(f"  Date range: {start} to {end} ({len(all_dates)} days)")

    records = []; sales_lookup = {}; wc = {'sunny':0, 'cloudy':0, 'rainy':0}

    for d in all_dates:
        dow = d.weekday()
        w = get_weather(d)
        wc[w['weather_type']] += 1
        for prod in PRODUCTS:
            sales = generate_sales(prod, dow, w['weather_type'], wt_buckets, dow_buckets)
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
    print(f"  Weather: {wc}")
    print(f"  Generated {len(df)} rows")
    print("Computing lag features...")
    for i, row in df.iterrows():
        prod = row['product']; d = row['date']
        df.at[i, 'lag_1'] = _calendar_lag(sales_lookup, prod, d, 1)
        df.at[i, 'lag_7'] = _calendar_lag(sales_lookup, prod, d, 7)
        df.at[i, 'rolling_7d_mean'] = _calendar_rolling_7d(sales_lookup, prod, d)

    # Show weather-sales relationship
    for prod in ['croissant','donut']:
        print(f'\n{prod} sales by weather:')
    out_path = os.path.join(ROOT, 'data', 'synthetic_sales_1year.csv')
    df.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')

if __name__=='__main__':
    main()