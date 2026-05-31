# -*- coding: utf-8 -*-
"""Formula-free XGBoost: 100% real data from DB, climate, calendar.
Lag features use calendar-day logic matching inference in module2_forecast.py."""

import sys, os, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import holidays
from datetime import date, timedelta
from sklearn.metrics import mean_absolute_error, r2_score
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MODEL_DIR = os.path.join(ROOT, 'models', 'xgboost')
os.makedirs(MODEL_DIR, exist_ok=True)

PRODUCTS = ['croissant','donut','chiffon','bread_coconut','bread_roll','croissant_chocolate']
RANDOM_SEED = 42

FEATURES = [
    'day_of_week','is_weekend','day_of_month','month','is_public_holiday',
    'is_ramadan','temperature','rainfall','humidity','is_rainy',
    'weather_sunny','weather_cloudy','weather_rainy','weather_storm',
    'lag_1','lag_7','rolling_7d_mean',
]

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
    if r > 250: wt = 'rainy'
    elif r > 180: wt = 'cloudy'
    else: wt = 'sunny'
    return {
        'temperature': float(t), 'rainfall': float(r), 'humidity': float(h),
        'is_rainy': 1 if r > 100 else 0,
        'weather_sunny': 1 if wt=='sunny' else 0,
        'weather_cloudy': 1 if wt=='cloudy' else 0,
        'weather_rainy': 1 if wt=='rainy' else 0,
        'weather_storm': 0,
    }

def _calendar_lag(sales_lookup, prod, fd, days_back):
    """Get sales from `days_back` calendar days before fd, backing up up to 4 days."""
    target = fd - timedelta(days=days_back)
    for _ in range(4):
        key = target.strftime('%Y-%m-%d')
        if (prod, key) in sales_lookup:
            return sales_lookup[(prod, key)]
        target -= timedelta(days=1)
    return 0.0

def _calendar_rolling_7d(sales_lookup, prod, fd):
    """Average sales over the 7 calendar days before fd."""
    vals = []
    for offset in range(1, 8):
        target = fd - timedelta(days=offset)
        key = target.strftime('%Y-%m-%d')
        if (prod, key) in sales_lookup:
            vals.append(sales_lookup[(prod, key)])
    return float(sum(vals) / len(vals)) if vals else 0.0

def build_dataset():
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
    records = []
    for r in rows:
        d = r['dt']; w = get_weather(d)
        records.append({
            'date': d, 'product': r['product_name'], 'sales': float(r['qty']),
            'day_of_week': d.weekday(), 'is_weekend': 1 if d.weekday()>=5 else 0,
            'day_of_month': d.day, 'month': d.month,
            'is_public_holiday': 1 if d in _my_holidays else 0,
            'is_ramadan': is_ramadan(d),
            'temperature': w['temperature'], 'rainfall': w['rainfall'],
            'humidity': w['humidity'], 'is_rainy': w['is_rainy'],
            'weather_sunny': w['weather_sunny'], 'weather_cloudy': w['weather_cloudy'],
            'weather_rainy': w['weather_rainy'], 'weather_storm': w['weather_storm'],
            'lag_1': 0.0, 'lag_7': 0.0, 'rolling_7d_mean': 0.0,
        })
    df = pd.DataFrame(records).sort_values(['product','date']).reset_index(drop=True)
    # Build calendar-day-based lags (matches inference logic in module2_forecast.py)
    sales_lookup = {}
    for _, row in df.iterrows():
        d = row['date']
        key = d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)
        sales_lookup[(row['product'], key)] = float(row['sales'])
    for i, row in df.iterrows():
        prod = row['product']
        d = row['date']
        fd = d if hasattr(d, 'date') else d
        df.at[i, 'lag_1'] = _calendar_lag(sales_lookup, prod, fd, 1)
        df.at[i, 'lag_7'] = _calendar_lag(sales_lookup, prod, fd, 7)
        df.at[i, 'rolling_7d_mean'] = _calendar_rolling_7d(sales_lookup, prod, fd)
    return df

def main():
    print('Formula-free training: DB sales + climate data + calendar')
    df = build_dataset()
    print(f'Dataset: {len(df)} rows, {len(df.date.unique())} dates')
    dates = sorted(df['date'].unique())
    test_dates = set(dates[-8:]); val_dates = set(dates[-16:-8])
    train = df[~df['date'].isin(test_dates|val_dates)]
    val = df[df['date'].isin(val_dates)]
    test = df[df['date'].isin(test_dates)]
    print(f'Train/Val/Test: {len(train)}/{len(val)}/{len(test)} rows')
    for prod in PRODUCTS:
        print(f'{prod}...')
        for lbl, subset in [('train',train),('val',val),('test',test)]:
            pdf = subset[subset['product']==prod]
            X = pdf[FEATURES]; y = pdf['sales']
            if len(X) < 3: continue
            if lbl == 'train':
                model = xgb.XGBRegressor(
                    max_depth=3, learning_rate=0.05, n_estimators=200,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                    reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED,
                    objective='reg:squarederror'
                )
                model.fit(X, y)
            preds = model.predict(X)
            mae = mean_absolute_error(y, preds)
            mape = np.mean(np.abs((y-preds)/y))*100 if len(y)>0 else 0
            r2 = r2_score(y, preds)
            print(f'  {lbl}: MAE={mae:.1f} MAPE={mape:.1f}% R2={r2:.3f} n={len(y)}')
        model.save_model(os.path.join(MODEL_DIR, f'{prod}_model.json'))
    with open(os.path.join(MODEL_DIR,'feature_columns.json'),'w') as f:
        json.dump(FEATURES, f, indent=2)
    print('Done')

if __name__=='__main__':
    main()
