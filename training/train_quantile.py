# -*- coding: utf-8 -*-
"""XGBoost training with GridSearchCV hyperparameter tuning.
Comparison models: LinearRegression, RandomForest, XGBoost.
Prediction intervals use test-set MAE."""

import sys, os, json, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings('ignore', category=UserWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MODEL_DIR = os.path.join(ROOT, 'models', 'xgboost')
DATA_PATH = os.path.join(ROOT, 'data', 'synthetic_sales_1year.csv')
os.makedirs(MODEL_DIR, exist_ok=True)

from config.settings import PRODUCT_TYPES as PRODUCTS
RANDOM_SEED = 42

from config.settings import FORECAST_FEATURE_COLS as FEATURES

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
    with open(os.path.join(MODEL_DIR,"test_metrics.json"),"w") as f:
        json.dump(results_all, f, indent=2)
    # --- Comparison models: Linear Regression & Random Forest ---
    # RF param grid for GridSearchCV (LR has no hyperparams to tune)
    RF_PARAM_GRID = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 8],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [2, 4],
    }
    
    comparison_results = {}
    rf_best_params_all = {}
    for prod in PRODUCTS:
        pdf_train = train[train["product"]==prod]
        X_train, y_train = pdf_train[FEATURES], pdf_train["sales"]
        
        tpdf = test[test["product"]==prod]
        X_test, y_test = tpdf[FEATURES], tpdf["sales"]
        
        prod_comp = {}
        
        # Linear Regression (no hyperparams, OLS closed-form)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_preds = np.maximum(lr.predict(X_test), 0)
        lr_mae = mean_absolute_error(y_test, lr_preds)
        lr_mape = np.mean(np.abs((y_test - lr_preds) / y_test)) * 100 if len(y_test) > 0 else 0
        lr_r2 = r2_score(y_test, lr_preds)
        prod_comp["LinearRegression"] = {"MAE": round(lr_mae,1), "MAPE": round(lr_mape,1), "R2": round(lr_r2,3)}
        
        # Random Forest with GridSearchCV
        rf_base = RandomForestRegressor(random_state=RANDOM_SEED)
        rf_grid = GridSearchCV(
            rf_base, RF_PARAM_GRID, scoring="neg_mean_absolute_error",
            cv=tscv, n_jobs=1,
        )
        rf_grid.fit(X_train, y_train)
        rf_best = rf_grid.best_estimator_
        rf_best_params_all[prod] = rf_grid.best_params_
        rf_preds = np.maximum(rf_best.predict(X_test), 0)
        rf_mae = mean_absolute_error(y_test, rf_preds)
        rf_mape = np.mean(np.abs((y_test - rf_preds) / y_test)) * 100 if len(y_test) > 0 else 0
        rf_r2 = r2_score(y_test, rf_preds)
        prod_comp["RandomForest"] = {"MAE": round(rf_mae,1), "MAPE": round(rf_mape,1), "R2": round(rf_r2,3)}
        
        # XGBoost (reload for fair comparison on same test split)
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(os.path.join(MODEL_DIR, f"{prod}_model.json"))
        xgb_preds_arr = np.maximum(xgb_model.predict(X_test), 0)
        xgb_mae = mean_absolute_error(y_test, xgb_preds_arr)
        xgb_mape = np.mean(np.abs((y_test - xgb_preds_arr) / y_test)) * 100 if len(y_test) > 0 else 0
        xgb_r2 = r2_score(y_test, xgb_preds_arr)
        prod_comp["XGBoost"] = {"MAE": round(xgb_mae,1), "MAPE": round(xgb_mape,1), "R2": round(xgb_r2,3)}
        
        comparison_results[prod] = prod_comp
        best_model = min(prod_comp, key=lambda k: prod_comp[k]["MAE"])
        print(f"{prod} comparison: LR MAE={lr_mae:.1f}, RF MAE={rf_mae:.1f}, XGB MAE={xgb_mae:.1f} -> Best: {best_model}")
    
    with open(os.path.join(MODEL_DIR, "model_comparison.json"), "w") as f:
        json.dump(comparison_results, f, indent=2)
    with open(os.path.join(MODEL_DIR, "rf_best_params.json"), "w") as f:
        json.dump(rf_best_params_all, f, indent=2)
    print("\nModel comparison saved to model_comparison.json\n")

    print('Done - 6 median models saved.\n')
    print('Accuracy reference:')
    for prod in PRODUCTS:
        r = results_all[prod]['test']
        print(f'  {prod}: Test MAE={r["MAE"]}, MAPE={r["MAPE"]}%, R2={r["R2"]}')

if __name__=='__main__':
    main()