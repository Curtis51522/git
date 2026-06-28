#!/usr/bin/env python
"""
Generate Raw Ticket-Level Sales Data — Guangzhou Bakery-Cafe (2021-2023)
=========================================================================
Output format matches French Bakery original:
  date, time, ticket_id, product_name, quantity, unit_price_cny

Source:
  - 9 products: real French Bakery ticket-level data (EUR → CNY)
  - 21 products: synthetic, generated from template product ticket patterns
  - 15 drinks: synthetic, with realistic coffee-shop time distribution

Usage:
  python scripts/generate_raw_sales.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import random, os, sys

np.random.seed(42)
random.seed(42)

# ============================================================
# CONFIGURATION
# ============================================================
import os as _os
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_DATA_DIR = _os.path.join(_os.path.dirname(_SCRIPT_DIR), "data")
FRENCH_CSV = _os.path.join(_DATA_DIR, "french_bakery_sales.csv")
OUTPUT = _os.path.join(_DATA_DIR, "bakery_sales_raw.csv")
START_DATE = "2021-01-01"
END_DATE = "2023-12-31"
EUR_TO_CNY = 7.8

# 9 products from French Bakery (real ticket-level data)
REAL_PRODUCTS = {
    "croissant":           "CROISSANT",
    "baguette":            "BAGUETTE",
    "croissant_chocolate": "PAIN AU CHOCOLAT",
    "cookie":              "COOKIE",
    "brioche":             "BRIOCHE",
    "brownie":             "BROWNIES",
    "macaron":             "MACARON",
    "apple_pie":           "CHAUSSON AUX POMMES",
    "chocolate_cake":      "FONDANT CHOCOLAT",
}

# 21 synthetic bread products — template + scale
SYNTHETIC_PRODUCTS = {
    "donut":          {"template": "CROISSANT",          "scale": 0.35, "price": 6.00},
    "eggtart":        {"template": "TARTELETTE",         "scale": 0.25, "price": 7.00},
    "cream_horn":     {"template": "CROISSANT",          "scale": 0.18, "price": 13.00},
    "bread_coconut":  {"template": "PAIN AUX RAISINS",   "scale": 0.22, "price": 10.00},
    "melon_bread":    {"template": "BRIOCHE",            "scale": 0.28, "price": 10.00},
    "bread_roll":     {"template": "PAIN",               "scale": 0.90, "price": 6.50},
    "pizza_bread":    {"template": "FORMULE SANDWICH",   "scale": 0.15, "price": 16.00},
    "chiffon":        {"template": "FINANCIER X5",       "scale": 0.50, "price": 18.00},
    "soboru_bread":   {"template": "BRIOCHE",            "scale": 0.20, "price": 11.00},
    "chocopie":       {"template": "COOKIE",             "scale": 0.30, "price": 10.00},
    "stickbread":     {"template": "FICELLE",            "scale": 0.60, "price": 5.50},
    "pandesal":       {"template": "PAIN",               "scale": 0.40, "price": 5.00},
    "sourdough":      {"template": "CAMPAGNE",           "scale": 0.35, "price": 22.00},
    "cornbread":      {"template": "PAIN",               "scale": 0.25, "price": 7.00},
    "flatbread":      {"template": "PAIN",               "scale": 0.20, "price": 8.00},
    "mantequilla":    {"template": "BRIOCHE",            "scale": 0.30, "price": 7.50},
    "muffin":         {"template": "COOKIE",             "scale": 0.50, "price": 12.00},
    "pancake":        {"template": "CROISSANT",          "scale": 0.22, "price": 13.00},
    "pullman":        {"template": "PAIN",               "scale": 0.30, "price": 9.00},
    "tostada":        {"template": "PAIN",               "scale": 0.18, "price": 10.00},
    "bagel":          {"template": "BRIOCHE",            "scale": 0.15, "price": 12.00},
}

# 15 drinks — generated independently
DRINKS = {
    "latte":             18,   # price in CNY
    "americano":         15,
    "cappuccino":        18,
    "mocha":             22,
    "espresso":          12,
    "flat_white":        20,
    "caramel_macchiato": 22,
    "cold_brew":         18,
    "hot_chocolate":     15,
    "matcha_latte":      18,
    "milk_tea":          14,
    "chai_latte":        16,
    "earl_grey":         10,
    "english_breakfast": 10,
    "lemonade":          12,
}

# All 30 bread product names for baseline filling
PRODUCTS_BREAD = [
    "croissant", "baguette", "croissant_chocolate", "cookie", "brioche",
    "brownie", "macaron", "apple_pie", "chocolate_cake",
    "donut", "eggtart", "cream_horn", "bread_coconut", "melon_bread",
    "bread_roll", "pizza_bread", "chiffon", "soboru_bread", "chocopie",
    "stickbread", "pandesal", "sourdough", "cornbread", "flatbread",
    "mantequilla", "muffin", "pancake", "pullman", "tostada", "bagel",
]

# Bread prices map (from products table)
BREAD_PRICES = {
    "croissant": 8.58, "baguette": 7.02, "croissant_chocolate": 9.36,
    "cookie": 6.24, "brioche": 13.26, "brownie": 15.60,
    "macaron": 18.72, "apple_pie": 10.92, "chocolate_cake": 19.50,
    "donut": 6.00, "eggtart": 7.00, "cream_horn": 13.00,
    "bread_coconut": 10.00, "melon_bread": 10.00, "bread_roll": 6.50,
    "pizza_bread": 16.00, "chiffon": 18.00, "soboru_bread": 11.00,
    "chocopie": 10.00, "stickbread": 5.50, "pandesal": 5.00,
    "sourdough": 22.00, "cornbread": 7.00, "flatbread": 8.00,
    "mantequilla": 7.50, "muffin": 12.00, "pancake": 13.00,
    "pullman": 9.00, "tostada": 10.00, "bagel": 12.00,
}

REAL_PRICES = {
    "croissant": 8.58, "baguette": 7.02, "croissant_chocolate": 9.36,
    "cookie": 6.24, "brioche": 13.26, "brownie": 15.60,
    "macaron": 18.72, "apple_pie": 10.92, "chocolate_cake": 19.50,
}

# ============================================================
# LOAD FRENCH BAKERY
# ============================================================
print("Loading French Bakery raw data...")
fb = pd.read_csv(FRENCH_CSV)
fb["date"] = pd.to_datetime(fb["date"])
fb = fb[fb["date"] >= pd.Timestamp(START_DATE)]  # Clip to 2021+
fb["unit_price_eur"] = fb["unit_price"].str.replace(",", ".").str.extract(r"(\d+\.?\d*)", expand=False).astype(float)

# Convert EUR to CNY
fb["unit_price_cny"] = round(fb["unit_price_eur"] * EUR_TO_CNY, 2)

print(f"  French Bakery rows: {len(fb):,}")
print(f"  Date range: {fb['date'].min().date()} to {fb['date'].max().date()}")
print(f"  Unique articles: {fb['article'].nunique()}")

# ============================================================
# PART 1: Extract 9 real products (ticket-level)
# ============================================================
print("\n=== Extracting 9 real products ===")
real_rows = []
ticket_counter = 1_000_000  # Start ticket IDs high to avoid conflict

for prod_name, french_name in REAL_PRODUCTS.items():
    sub = fb[fb["article"] == french_name].copy()
    for _, row in sub.iterrows():
        real_rows.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "time": row["time"],
            "ticket_id": int(row["ticket_number"]),
            "product_name": prod_name,
            "quantity": int(row["Quantity"]),
            "unit_price_cny": round(row["unit_price_eur"] * EUR_TO_CNY, 2),
            "source": "french_bakery_real",
        })
    print(f"  {prod_name:25s} <- {french_name:25s}: {len(sub):,} ticket rows")

# Extend beyond French Bakery end date (2022-09) to 2023-12
fb_end = fb["date"].max()
fb_start = fb["date"].min()
print(f"\n  French Bakery covers: {fb_start.date()} to {fb_end.date()}")
print(f"  Extending to {END_DATE} with seasonal repeat...")

# Build daily lookup for real products (already extracted)
daily_real = defaultdict(lambda: defaultdict(int))
for r in real_rows:
    daily_real[r["date"]][r["product_name"]] += r["quantity"]

# Extend: for each date beyond fb_end, repeat year-1 pattern with noise
current = fb_end + timedelta(days=1)
end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
while current <= end_dt:
    date_str = current.strftime("%Y-%m-%d")
    ref_date = current.replace(year=2021)  # Use 2021 as reference year
    ref_str = ref_date.strftime("%Y-%m-%d")
    
    if ref_str in daily_real and current.weekday() < 7:
        for prod_name, ref_qty in daily_real[ref_str].items():
            if ref_qty == 0:
                continue
            # Add noise: random walk from 2021 baseline
            years_diff = current.year - 2021
            growth = 1.0 + years_diff * 0.05  # 5% annual growth
            noise = np.random.normal(1.0, 0.12)
            qty = max(0, int(ref_qty * growth * noise))
            if qty == 0:
                continue
            
            # Split into 1-3 tickets for that product on that day
            num_tickets = min(qty, np.random.randint(1, min(4, qty+1)))
            remaining = qty
            for t in range(num_tickets):
                if t == num_tickets - 1:
                    ticket_qty = remaining
                else:
                    ticket_qty = np.random.randint(1, max(2, remaining - (num_tickets - t - 1)))
                    ticket_qty = min(ticket_qty, remaining)
                remaining -= ticket_qty
                
                # Random time between 07:00-19:00
                hour = np.random.randint(7, 19)
                minute = np.random.randint(0, 60)
                time_str = f"{hour:02d}:{minute:02d}"
                
                # Get price from existing real rows
                price = 8.0  # fallback
                for r in real_rows:
                    if r["product_name"] == prod_name:
                        price = r["unit_price_cny"]
                        break
                
                ticket_counter += 1
                real_rows.append({
                    "date": date_str,
                    "time": time_str,
                    "ticket_id": ticket_counter,
                    "product_name": prod_name,
                    "quantity": ticket_qty,
                    "unit_price_cny": price,
                    "source": "extended",
                })
    current += timedelta(days=1)

print(f"  Total real + extended rows: {len(real_rows):,}")

# ============================================================
# PART 2: Generate 21 synthetic products (ticket-level)
# ============================================================
print("\n=== Generating 21 synthetic products ===")

# First build daily quantities from template scaling
synthetic_rows = []

for prod_name, cfg in SYNTHETIC_PRODUCTS.items():
    template = cfg["template"]
    scale = cfg["scale"]
    price = cfg["price"]
    
    # Get template daily quantities from French Bakery
    template_daily = defaultdict(int)
    for _, row in fb.iterrows():
        if row["article"] == template:
            d = row["date"].strftime("%Y-%m-%d")
            template_daily[d] += int(row["Quantity"])
    
    if not template_daily:
        print(f"  WARNING: No template data for {template}, skipping {prod_name}")
        continue
    
    # Fill all dates
    current = datetime.strptime(START_DATE, "%Y-%m-%d")
    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")
        
        # Get template quantity for this date (or use seasonal repeat)
        if date_str in template_daily:
            base_qty = template_daily[date_str]
        else:
            ref_date = current.replace(year=2021)
            ref_str = ref_date.strftime("%Y-%m-%d")
            base_qty = template_daily.get(ref_str, 0)
        
        noise = np.random.normal(1.0, 0.12)
        qty = max(0, int(base_qty * scale * noise))
        
        if qty > 0:
            # Split into tickets
            num_tickets = min(qty, np.random.randint(1, min(5, qty+1)))
            remaining = qty
            for t in range(num_tickets):
                if t == num_tickets - 1:
                    ticket_qty = remaining
                else:
                    ticket_qty = np.random.randint(1, max(2, remaining - (num_tickets - t - 1)))
                    ticket_qty = min(ticket_qty, remaining)
                remaining -= ticket_qty
                
                hour = np.random.randint(7, 19)
                minute = np.random.randint(0, 60)
                
                ticket_counter += 1
                synthetic_rows.append({
                    "date": date_str,
                    "time": f"{hour:02d}:{minute:02d}",
                    "ticket_id": ticket_counter,
                    "product_name": prod_name,
                    "quantity": ticket_qty,
                    "unit_price_cny": price,
                    "source": "synthetic",
                })
        
        current += timedelta(days=1)
    
    day_count = len(set(r["date"] for r in synthetic_rows if r["product_name"]==prod_name))
    total_qty = sum(r["quantity"] for r in synthetic_rows if r["product_name"]==prod_name)
    print(f"  {prod_name:25s}: {total_qty:>8,} units, {day_count:>4} selling days")

# ============================================================
# PART 3: Generate 15 drinks (ticket-level)
# ============================================================
print("\n=== Generating 15 drinks ===")
drink_rows = []

# Drink demand patterns: coffee peaks 07:00-10:00 and 14:00-16:00
# Base daily quantities (realistic for a small bakery-cafe in Guangzhou)
drink_base_daily = {
    "latte": 10, "americano": 10, "cappuccino": 7, "mocha": 5,
    "espresso": 5, "flat_white": 3, "caramel_macchiato": 3, "cold_brew": 5,
    "hot_chocolate": 4, "matcha_latte": 5, "milk_tea": 4,
    "chai_latte": 3, "earl_grey": 3, "english_breakfast": 3, "lemonade": 3,
}

current = datetime.strptime(START_DATE, "%Y-%m-%d")
while current <= end_dt:
    date_str = current.strftime("%Y-%m-%d")
    weekday = current.weekday()
    month = current.month
    
    # Weekend boost (1.2x), summer boost for cold drinks (1.3x)
    for drink_name, base_qty in drink_base_daily.items():
        # Seasonal adjustment
        seasonal = 1.0
        if drink_name in ("cold_brew", "lemonade") and month in (6, 7, 8, 9):
            seasonal = 1.1  # Summer cold drink boost (mild)
        if drink_name in ("hot_chocolate", "mocha") and month in (12, 1, 2):
            seasonal = 1.1  # Winter hot drink boost (mild)
        
        # Weekend effect
        weekend_mult = 1.05 if weekday >= 5 else 1.0
        
        noise = np.random.normal(1.0, 0.12)
        qty = max(0, int(base_qty * seasonal * weekend_mult * noise))
        
        if qty == 0:
            continue
        
        # Coffee drinks use different time distribution
        is_coffee = drink_name in ("latte", "americano", "cappuccino", "mocha",
                                    "espresso", "flat_white", "caramel_macchiato")
        
        num_tickets = min(qty, np.random.randint(1, min(4, qty+1)))
        remaining = qty
        for t in range(num_tickets):
            if t == num_tickets - 1:
                ticket_qty = remaining
            else:
                ticket_qty = np.random.randint(1, max(2, remaining - (num_tickets - t - 1)))
                ticket_qty = min(ticket_qty, remaining)
            remaining -= ticket_qty
            
            # Time: coffee peaks 07:00-10:00, drinks spread throughout day
            if is_coffee:
                # 60% chance morning peak, 40% afternoon
                if np.random.random() < 0.6:
                    hour = np.random.randint(7, 11)
                else:
                    hour = np.random.randint(13, 17)
            else:
                hour = np.random.randint(8, 20)
            minute = np.random.randint(0, 60)
            
            ticket_counter += 1
            drink_rows.append({
                "date": date_str,
                "time": f"{hour:02d}:{minute:02d}",
                "ticket_id": ticket_counter,
                "product_name": drink_name,
                "quantity": ticket_qty,
                "unit_price_cny": DRINKS[drink_name],
                "source": "synthetic_drink",
            })
    
    current += timedelta(days=1)

for drink_name in sorted(DRINKS.keys()):
    day_count = len(set(r["date"] for r in drink_rows if r["product_name"]==drink_name))
    total_qty = sum(r["quantity"] for r in drink_rows if r["product_name"]==drink_name)
    print(f"  {drink_name:25s}: {total_qty:>8,} cups, {day_count:>4} selling days")

# ============================================================
# POST-PROCESSING: Per-product daily bread baseline
# ============================================================
print("\n=== Ensuring per-product daily bread baseline ===")
# Track existing bread quantities per date per product
from collections import defaultdict
daily_prod_qty = defaultdict(lambda: defaultdict(int))
for r in real_rows + synthetic_rows:
    daily_prod_qty[r["date"]][r["product_name"]] += r["quantity"]

# Per-product minimum: each bread product sells at least 1 unit/day (weekday), 0.7/day avg
# High-volume products get higher baseline
high_volume = {"croissant", "baguette", "croissant_chocolate", "bread_roll", "donut", "pancake"}
medium_volume = {"cookie", "brioche", "apple_pie", "sourdough", "stickbread", "chocopie", "muffin"}

current = datetime.strptime(START_DATE, "%Y-%m-%d")
added_bread = 0
while current <= end_dt:
    date_str = current.strftime("%Y-%m-%d")
    weekday = current.weekday()
    weekend_mult = 1.3 if weekday >= 5 else 1.0
    
    for prod in PRODUCTS_BREAD:
        existing = daily_prod_qty[date_str].get(prod, 0)
        # Determine base minimum per product
        if prod in high_volume:
            base = max(1, int(np.random.normal(3, 1) * weekend_mult))
        elif prod in medium_volume:
            base = max(1, int(np.random.normal(2, 0.7) * weekend_mult))
        else:
            # Low volume: 70% chance of 1 unit, 30% chance of 0
            base = 1 if random.random() < (0.7 * weekend_mult) else 0
        
        shortfall = max(0, base - existing)
        if shortfall <= 0:
            continue
        
        hour = np.random.randint(7, 18)
        minute = np.random.randint(0, 60)
        ticket_counter += 1
        price = REAL_PRICES.get(prod) or BREAD_PRICES.get(prod, 10.0)
        synthetic_rows.append({
            "date": date_str,
            "time": f"{hour:02d}:{minute:02d}",
            "ticket_id": ticket_counter,
            "product_name": prod,
            "quantity": shortfall,
            "unit_price_cny": price,
            "source": "bread_baseline",
        })
        added_bread += shortfall
    
    current += timedelta(days=1)
print(f"  Added {added_bread} bread units (per-product baseline)")

# ============================================================
# MERGE AND SAVE
# ============================================================
print("\n=== Merging and saving ===")
all_rows = real_rows + synthetic_rows + drink_rows
df_out = pd.DataFrame(all_rows)
df_out = df_out.sort_values(["date", "time"]).reset_index(drop=True)

# Ensure column order matches French Bakery format
df_out = df_out[["date", "time", "ticket_id", "product_name", "quantity", "unit_price_cny"]]

df_out.to_csv(OUTPUT, index=False)

print(f"\nSaved: {len(df_out):,} rows to {OUTPUT}")
print(f"Columns: {list(df_out.columns)}")
print(f"Date range: {df_out['date'].min()} to {df_out['date'].max()}")
print(f"Unique products: {df_out['product_name'].nunique()}")
print(f"Unique tickets: {df_out['ticket_id'].nunique():,}")
print(f"Total units: {df_out['quantity'].sum():,}")

print("\n=== Sample rows ===")
print(df_out.head(10).to_string(index=False))

print("\nDone.")
