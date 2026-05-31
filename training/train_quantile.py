# -*- coding: utf-8 -*-
"""XGBoost median-only training with GridSearchCV hyperparameter tuning.
No prediction intervals - data too small for reliable quantile regression."""

import sys, os, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
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

PARAM_GRID = {
    'max_depth': [2, 3, 4],
    'learning_rate': [0.01, 0.03, 0.05],
    'n_estimators': [100, 200],
    'min_child_weight': [5, 10],
}

def main():
    print(f'Loading: {DATA_PATH}')
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df = df[df['day_of_week'] != 0].copy()
    print(f'Non-Monday rows: {len(df)}')

    dates = sorted(df['date'].unique())
    test_dates  = set(dates[-60:])
    val_dates   = set(dates[-120:-60])
    train = df[df['date'].isin(set(dates[:-120]))]
    val   = df[df['date'].isin(val_dates)]
    test  = df[df['date'].isin(test_dates)]
    print(f'Train/Val/Test: {len(train)}/{len(val)}/{len(test)} rows\n')

    tscv = TimeSeriesSplit(n_splits=3)
    best_params_all = {}
    results_all = {}

    for prod in PRODUCTS:
        pdf = train[train['product']==prod]
        X, y = pdf[FEATURES], pdf['sales']

        base = xgb.XGBRegressor(
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED,
        )
        grid = GridSearchCV(
            base, PARAM_GRID, scoring='neg_mean_absolute_error',
            cv=tscv, n_jobs=1,
        )
        grid.fit(X, y)
        best = grid.best_params_
        best_params_all[prod] = best

        model = xgb.XGBRegressor(
            max_depth=best['max_depth'],
            learning_rate=best['learning_rate'],
            n_estimators=best['n_estimators'],
            min_child_weight=best['min_child_weight'],
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=RANDOM_SEED,
        )
        model.fit(X, y)

        # Evaluate
        prod_results = {}
        for lbl, subset in [('train',train),('val',val),('test',test)]:
            spdf = subset[subset['product']==prod]
            Xs, ys = spdf[FEATURES], spdf['sales']
            preds = model.predict(Xs)
            mae  = mean_absolute_error(ys, preds)
            mape = np.mean(np.abs((ys-preds)/ys))*100 if len(ys)>0 else 0
            r2   = r2_score(ys, preds)
            prod_results[lbl] = {'MAE': round(mae,1), 'MAPE': round(mape,1), 'R2': round(r2,3)}
            print(f'{prod} CV MAE={-grid.best_score_:.1f} | {lbl} MAE={mae:.1f} MAPE={mape:.1f}% R2={r2:.3f}')

        results_all[prod] = prod_results
        model.save_model(os.path.join(MODEL_DIR, f'{prod}_model.json'))
        print(f'  best: {best}\n')

    with open(os.path.join(MODEL_DIR,'feature_columns.json'),'w') as f:
        json.dump(FEATURES, f, indent=2)
    with open(os.path.join(MODEL_DIR,'best_params.json'),'w') as f:
        json.dump(best_params_all, f, indent=2)
    print('Done - 6 median models saved.\n')
    print('Accuracy reference:')
    for prod in PRODUCTS:
        r = results_all[prod]['test']
        print(f'  {prod}: Test MAE={r["MAE"]}, MAPE={r["MAPE"]}%, R2={r["R2"]}')

if __name__=='__main__':
    main()