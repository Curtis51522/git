# -*- coding: utf-8 -*-
"""XGBoost training from synthetic 1-year CSV. No formulas, pure model learning."""

import sys, os, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import date, timedelta
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MODEL_DIR = os.path.join(ROOT, 'models', 'xgboost')
DATA_PATH = os.path.join(ROOT, 'data', 'synthetic_sales_1year.csv')
os.makedirs(MODEL_DIR, exist_ok=True)

PRODUCTS = ['croissant','donut','chiffon','bread_coconut','bread_roll','croissant_chocolate']
RANDOM_SEED = 42

FEATURES = [
    'day_of_week','is_weekend','day_of_month','month','is_public_holiday',
    'is_ramadan','temperature','rainfall','humidity','is_rainy',
    'weather_sunny','weather_cloudy','weather_rainy','weather_storm',
    'lag_1','lag_7','rolling_7d_mean',
]

def main():
    print(f'Loading: {DATA_PATH}')
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    # Drop Monday rows (closed, zero sales) - model should learn from real sales days only
    df = df[df['day_of_week'] != 0].copy()
    print(f'Non-Monday rows: {len(df)} ({len(df)/6:.0f} days x 6 products)')

    # Time-based split: last 60 days = test, 60 before that = val, rest = train
    dates = sorted(df['date'].unique())
    test_dates  = set(dates[-60:])
    val_dates   = set(dates[-120:-60])
    train_dates = set(dates[:-120])
    train = df[df['date'].isin(train_dates)]
    val   = df[df['date'].isin(val_dates)]
    test  = df[df['date'].isin(test_dates)]
    print(f'Train/Val/Test: {len(train)}/{len(val)}/{len(test)} rows')

    for prod in PRODUCTS:
        print(f'{prod}...')
        for lbl, subset in [('train',train),('val',val),('test',test)]:
            pdf = subset[subset['product']==prod]
            X = pdf[FEATURES]; y = pdf['sales']
            if len(X) < 3: continue
            if lbl == 'train':
                model = xgb.XGBRegressor(
                    max_depth=4, learning_rate=0.03, n_estimators=300,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                    reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED,
                    objective='reg:squarederror'
                )
                model.fit(X, y)
            preds = model.predict(X)
            mae  = mean_absolute_error(y, preds)
            mape = np.mean(np.abs((y-preds)/y))*100 if len(y)>0 else 0
            r2   = r2_score(y, preds)
            print(f'  {lbl}: MAE={mae:.1f} MAPE={mape:.1f}% R2={r2:.3f} n={len(y)}')
        model.save_model(os.path.join(MODEL_DIR, f'{prod}_model.json'))

    with open(os.path.join(MODEL_DIR,'feature_columns.json'),'w') as f:
        json.dump(FEATURES, f, indent=2)
    print('Done - all 6 models saved.')

if __name__=='__main__':
    main()
