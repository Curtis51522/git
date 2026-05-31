# -*- coding: utf-8 -*-
"""XGBoost quantile regression: P2.5 / P50 / P97.5 per product = 18 models.
Prediction intervals from model-native quantiles — no hand-crafted coefficients."""

import sys, os, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MODEL_DIR = os.path.join(ROOT, 'models', 'xgboost')
DATA_PATH = os.path.join(ROOT, 'data', 'synthetic_sales_1year.csv')
os.makedirs(MODEL_DIR, exist_ok=True)

PRODUCTS = ['croissant','donut','chiffon','bread_coconut','bread_roll','croissant_chocolate']
QUANTILES = {'lower': 0.025, 'median': 0.5, 'upper': 0.975}
RANDOM_SEED = 42

FEATURES = [
    'day_of_week','is_weekend','day_of_month','month','is_public_holiday',
    'is_ramadan','temperature','rainfall','humidity','is_rainy',
    'weather_sunny','weather_cloudy','weather_rainy','weather_storm',
    'lag_1','lag_7','rolling_7d_mean',
]

XGB_PARAMS = dict(
    max_depth=4, learning_rate=0.03, n_estimators=300,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED,
)

def main():
    print(f'Loading: {DATA_PATH}')
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df = df[df['day_of_week'] != 0].copy()
    print(f'Non-Monday rows: {len(df)} ({len(df)/6:.0f} days x 6 products)')

    dates = sorted(df['date'].unique())
    test_dates  = set(dates[-60:])
    val_dates   = set(dates[-120:-60])
    train_dates = set(dates[:-120])
    train = df[df['date'].isin(train_dates)]
    val   = df[df['date'].isin(val_dates)]
    test  = df[df['date'].isin(test_dates)]
    print(f'Train/Val/Test: {len(train)}/{len(val)}/{len(test)} rows\n')

    for prod in PRODUCTS:
        for qname, qval in QUANTILES.items():
            pdf = train[train['product']==prod]
            X, y = pdf[FEATURES], pdf['sales']
            model = xgb.XGBRegressor(
                objective='reg:quantileerror',
                quantile_alpha=qval,
                **XGB_PARAMS,
            )
            model.fit(X, y)

            # Evaluate on all splits
            results = []
            for lbl, subset in [('train',train),('val',val),('test',test)]:
                spdf = subset[subset['product']==prod]
                Xs, ys = spdf[FEATURES], spdf['sales']
                preds = model.predict(Xs)
                mae  = mean_absolute_error(ys, preds)
                r2   = r2_score(ys, preds)
                results.append(f'{lbl} MAE={mae:.1f} R2={r2:.3f}')

            suffix = 'lower' if qname == 'lower' else ('upper' if qname == 'upper' else 'median')
            path = os.path.join(MODEL_DIR, f'{prod}_{suffix}_model.json')
            model.save_model(path)
            print(f'{prod} [{qname} q={qval}] {", ".join(results)}')

    with open(os.path.join(MODEL_DIR,'feature_columns.json'),'w') as f:
        json.dump(FEATURES, f, indent=2)
    print('\nDone - 18 quantile models saved.')

if __name__=='__main__':
    main()
