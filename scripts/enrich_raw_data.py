#!/usr/bin/env python
"""Data Enrichment — Fixed."""
import pandas as pd, numpy as np, os

np.random.seed(42)
BASE = r'C:\Users\Curtis\Desktop\learningmaterials\SEMESTER3\bakery-ai-system'
DATA = os.path.join(BASE, 'data')

print("Loading...")
raw = pd.read_csv(os.path.join(DATA, 'bakery_sales_raw.csv'))
raw['date'] = pd.to_datetime(raw['date'])
weather = pd.read_csv(os.path.join(DATA, 'guangzhou_weather.csv'), parse_dates=['date'])

BREAD = ['apple_pie','bagel','baguette','bread_coconut','bread_roll','brioche','brownie','chiffon','chocolate_cake','chocopie','cookie','cornbread','cream_horn','croissant','croissant_chocolate','donut','eggtart','flatbread','macaron','mantequilla','melon_bread','muffin','pancake','pandesal','pizza_bread','pullman','soboru_bread','sourdough','stickbread','tostada']
HOT = ['americano','cappuccino','caramel_macchiato','chai_latte','espresso','flat_white','hot_chocolate','latte','mocha','earl_grey','english_breakfast']
COLD = ['cold_brew','lemonade','matcha_latte','milk_tea']
bm = raw['product_name'].isin(BREAD)
hm = raw['product_name'].isin(HOT)
cm = raw['product_name'].isin(COLD)

# Merge weather
raw['ds'] = raw['date'].dt.strftime('%Y-%m-%d')
weather['ds'] = weather['date'].dt.strftime('%Y-%m-%d')
weather['rain_flag'] = (weather['precipitation'] > 1.0).astype(int)
raw = raw.merge(weather[['ds','temp_mean','temp_max','temp_min','rain_flag']], on='ds', how='left')
raw['temp_mean'] = raw['temp_mean'].fillna(22)
raw['rain_flag'] = raw['rain_flag'].fillna(0).astype(int)
raw['is_rainy'] = raw['rain_flag']
raw.drop(columns=['rain_flag'], inplace=True)
print(f"Merged: {len(raw):,} rows, rainy={raw['is_rainy'].mean()*100:.1f}%")

# R1: Weather
print("R1: Weather coupling...")
raw['mult'] = 1.0
raw.loc[bm & (raw['temp_mean'] < 15), 'mult'] = 1.08
raw.loc[bm & (raw['temp_mean'] > 32), 'mult'] = 0.92
raw.loc[bm & (raw['is_rainy'] == 1), 'mult'] *= 0.88
raw.loc[hm & (raw['temp_mean'] < 12), 'mult'] = 1.45
raw.loc[hm & (raw['temp_mean'].between(12, 14.99)), 'mult'] = 1.30
raw.loc[hm & (raw['temp_mean'].between(15, 17.99)), 'mult'] = 1.10
raw.loc[hm & (raw['temp_mean'] > 30), 'mult'] = 0.70
raw.loc[hm & (raw['temp_mean'].between(25, 30)), 'mult'] = 0.82
raw.loc[hm & (raw['is_rainy'] == 1), 'mult'] *= 1.08
raw.loc[cm & (raw['temp_mean'] > 32), 'mult'] = 1.65
raw.loc[cm & (raw['temp_mean'].between(28, 32)), 'mult'] = 1.40
raw.loc[cm & (raw['temp_mean'].between(25, 27.99)), 'mult'] = 1.20
raw.loc[cm & (raw['temp_mean'] < 15), 'mult'] = 0.55
raw.loc[cm & (raw['temp_mean'].between(15, 17.99)), 'mult'] = 0.75
raw.loc[cm & (raw['is_rainy'] == 1), 'mult'] *= 0.82
raw['quantity'] = (raw['quantity'] * raw['mult']).round().clip(lower=1).astype(int)

# R2: Top-3
print("R2: Top-3...")
daily_qty = raw.groupby(['ds', 'product_name'])['quantity'].sum().reset_index()
daily_qty['rank'] = daily_qty.groupby('ds')['quantity'].rank(method='dense', ascending=False)
top3 = daily_qty[daily_qty['rank'] <= 3].copy()
n3 = len(top3)
top3['discount'] = np.where(top3['rank'] == 1, np.random.uniform(0.18, 0.25, n3),
                   np.where(top3['rank'] == 2, np.random.uniform(0.12, 0.18, n3),
                            np.random.uniform(0.08, 0.14, n3)))
top3['boost'] = np.where(top3['rank'] == 1, 1.28, np.where(top3['rank'] == 2, 1.18, 1.10))
raw = raw.merge(top3[['ds','product_name','discount','boost']], on=['ds','product_name'], how='left')
raw['is_top3'] = raw['discount'].notna().astype(int)
raw['discount_pct'] = raw['discount'].fillna(raw.get('discount_pct', 0)).astype(float)
raw['boost'] = raw['boost'].fillna(1.0)
raw['quantity'] = (raw['quantity'] * raw['boost']).round().clip(lower=1).astype(int)
raw.drop(columns=['discount','boost'], inplace=True)
print(f"  Top-3: {raw['is_top3'].mean()*100:.1f}% coverage")

# R3: Member day
print("R3: Member day...")
raw['day'] = raw['date'].dt.day
raw['is_member_day'] = (raw['day'].isin([8,18,28])).astype(int) | raw.get('is_member_day', 0).astype(int)
mem = raw['day'].isin([8,18,28])
n_mem = mem.sum()
if n_mem > 0:
    boost_mem = np.random.uniform(1.15, 1.30, n_mem)
    raw.loc[mem, 'quantity'] = (raw.loc[mem, 'quantity'] * boost_mem).round().clip(lower=1).astype(int)
print(f"  Member: {raw['is_member_day'].mean()*100:.1f}% coverage")

# R4: Seasonality
print("R4: Seasonality...")
raw['month'] = raw['date'].dt.month
MB = {1:1.05,2:1.02,3:1.00,4:0.98,5:0.96,6:0.94,7:0.92,8:0.93,9:0.97,10:1.02,11:1.06,12:1.10}
MH = {1:1.35,2:1.25,3:1.10,4:1.00,5:0.85,6:0.72,7:0.65,8:0.68,9:0.82,10:0.95,11:1.15,12:1.35}
MC = {1:0.55,2:0.60,3:0.75,4:1.00,5:1.25,6:1.45,7:1.55,8:1.50,9:1.30,10:1.05,11:0.80,12:0.55}
raw['seas'] = 1.0
for m, v in MB.items(): raw.loc[bm & (raw['month']==m), 'seas'] = v
for m, v in MH.items(): raw.loc[hm & (raw['month']==m), 'seas'] = v
for m, v in MC.items(): raw.loc[cm & (raw['month']==m), 'seas'] = v
raw['quantity'] = (raw['quantity'] * raw['seas']).round().clip(lower=1).astype(int)

# Save
print("Saving...")
cols = ['date','time','ticket_id','product_name','quantity','unit_price_cny',
        'is_rainy','is_member_day','is_competitor','is_new_product','is_day1','is_top3','discount_pct']
raw['date'] = raw['date'].dt.strftime('%Y-%m-%d')
raw = raw.sort_values(['date','time']).reset_index(drop=True)
raw['ticket_id'] = range(1, len(raw)+1)
raw[cols].to_csv(os.path.join(DATA, 'bakery_sales_raw.csv'), index=False)

print(f"Saved: {len(raw):,} rows, {raw['quantity'].sum():,} units")
for c in ['is_rainy','is_member_day','is_top3','discount_pct']:
    print(f"  {c}: {(raw[c]>0).mean()*100:.1f}% coverage")
print("Done.")
