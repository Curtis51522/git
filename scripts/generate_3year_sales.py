import pandas as pd
import numpy as np
import json, os, sys
from datetime import datetime, timedelta
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================
FRENCH_CSV = r"C:\Users\Curtis\Desktop\learningmaterials\SEMESTER3\bakery-ai-system\data\french_bakery_sales.csv"
OUTPUT_DIR = r"C:\Users\Curtis\Desktop\learningmaterials\SEMESTER3\bakery-ai-system\data"

# 9 products with real French Bakery data
REAL_PRODUCTS = {
    "croissant": "CROISSANT",
    "baguette": "BAGUETTE",
    "croissant_chocolate": "PAIN AU CHOCOLAT",
    "cookie": "COOKIE",
    "brioche": "BRIOCHE",
    "brownie": "BROWNIES",
    "macaron": "MACARON",
    "apple_pie": "CHAUSSON AUX POMMES",
    "chocolate_cake": "FONDANT CHOCOLAT",
}

# 21 products to generate synthetically, each mapped to a French Bakery template
# Template = a product with similar sales pattern (seasonality, weekday effect)
SYNTHETIC_PRODUCTS = {
    "donut": {"template": "CROISSANT", "scale": 0.35, "price_myr": 4.50},
    "eggtart": {"template": "TARTELETTE", "scale": 0.25, "price_myr": 3.50},
    "cream_horn": {"template": "CROISSANT", "scale": 0.18, "price_myr": 5.00},
    "bread_coconut": {"template": "PAIN AUX RAISINS", "scale": 0.22, "price_myr": 4.00},
    "melon_bread": {"template": "BRIOCHE", "scale": 0.28, "price_myr": 4.50},
    "bread_roll": {"template": "PAIN", "scale": 0.90, "price_myr": 3.50},
    "pizza_bread": {"template": "FORMULE SANDWICH", "scale": 0.15, "price_myr": 5.50},
    "chiffon": {"template": "FINANCIER X5", "scale": 0.50, "price_myr": 6.00},
    "soboru_bread": {"template": "BRIOCHE", "scale": 0.20, "price_myr": 4.50},
    "chocopie": {"template": "COOKIE", "scale": 0.30, "price_myr": 5.00},
    "stickbread": {"template": "FICELLE", "scale": 0.60, "price_myr": 3.00},
    "pandesal": {"template": "PAIN", "scale": 0.40, "price_myr": 2.50},
    "sourdough": {"template": "CAMPAGNE", "scale": 0.35, "price_myr": 8.00},
    "cheesecake": {"template": "FLAN", "scale": 0.40, "price_myr": 10.00},
    "cupcake": {"template": "COOKIE", "scale": 0.55, "price_myr": 6.50},
    "tiramisu": {"template": "TARTELETTE", "scale": 0.20, "price_myr": 12.00},
    "waffle": {"template": "CROISSANT", "scale": 0.15, "price_myr": 5.50},
    "red_velvet_cake": {"template": "FINANCIER X5", "scale": 0.12, "price_myr": 14.00},
    "carrot_cake": {"template": "FINANCIER X5", "scale": 0.15, "price_myr": 11.00},
    "churros": {"template": "CROISSANT", "scale": 0.25, "price_myr": 5.00},
    "creme_brulee": {"template": "FLAN", "scale": 0.18, "price_myr": 9.00},
}

# Target date range: 3 years (Jan 2021 - Dec 2023)
# French Bakery covers 2021-01 to 2022-09
# We extend to 2023-12 by repeating seasonal patterns
START_DATE = "2021-01-01"
END_DATE = "2023-12-31"

# ============================================================
# Load French Bakery data
# ============================================================
print("Loading French Bakery CSV...")
df = pd.read_csv(FRENCH_CSV)
df["date"] = pd.to_datetime(df["date"])
df["unit_price_num"] = df["unit_price"].str.replace(",", ".").str.replace(" \u20ac", "", regex=False).str.strip().astype(float)

# Aggregate to daily sales per product
daily = df.groupby(["date", "article"]).agg(
    quantity=("Quantity", "sum"),
    revenue=("unit_price_num", lambda x: (df.loc[x.index, "Quantity"] * df.loc[x.index, "unit_price_num"]).sum())
).reset_index()

daily["avg_price"] = daily["revenue"] / daily["quantity"]

# Build lookup: date -> product -> {quantity, avg_price}
sales_lookup = defaultdict(dict)
for _, row in daily.iterrows():
    d = row["date"].strftime("%Y-%m-%d")
    p = row["article"]
    sales_lookup[d][p] = {"qty": int(row["quantity"]), "price": round(row["avg_price"], 2)}

# French Bakery date range
fb_start = df["date"].min()
fb_end = df["date"].max()
fb_days = (fb_end - fb_start).days
print(f"French Bakery range: {fb_start.date()} to {fb_end.date()} ({fb_days} days)")

# ============================================================
# Build daily sales for REAL products (9 products)
# ============================================================
def get_daily_sales(product_name, french_name, start, end):
    """Get daily sales for a product, extended to full 3-year range."""
    records = []
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        if date_str in sales_lookup and french_name in sales_lookup[date_str]:
            entry = sales_lookup[date_str][french_name]
            records.append({
                "date": date_str,
                "product_name": product_name,
                "quantity": entry["qty"],
                "unit_price_eur": entry["price"],
                "source": "real",
            })
        else:
            records.append({
                "date": date_str,
                "product_name": product_name,
                "quantity": 0,
                "unit_price_eur": 0,
                "source": "real",
            })
        current += timedelta(days=1)
    return records

# For dates beyond French Bakery range, repeat with seasonal noise
def extend_seasonal(records, fb_end_date):
    """Extend records beyond French Bakery end date by repeating year-1 pattern."""
    # Get 1 year of pattern (2021)
    pattern = {}
    for r in records:
        d = datetime.strptime(r["date"], "%Y-%m-%d")
        if d.year == 2021:
            key = d.strftime("%m-%d")
            pattern[key] = r["quantity"]
    
    for r in records:
        d = datetime.strptime(r["date"], "%Y-%m-%d")
        if d > fb_end_date and r["quantity"] == 0:
            key = d.strftime("%m-%d")
            base_qty = pattern.get(key, 0)
            # Add 5-15% growth + noise
            noise = np.random.normal(1.0, 0.15)
            r["quantity"] = max(0, int(base_qty * noise))
            r["source"] = "extended"

# ============================================================
# Build daily sales for SYNTHETIC products (21 products)
# ============================================================
def generate_synthetic(product_name, template_name, scale, price_myr, start, end):
    """Generate synthetic daily sales from a template product's pattern."""
    records = []
    template_prices = {}
    
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        qty = 0
        if date_str in sales_lookup and template_name in sales_lookup[date_str]:
            template_qty = sales_lookup[date_str][template_name]["qty"]
            # Apply scale + noise
            noise = np.random.normal(1.0, 0.20)
            qty = max(0, int(template_qty * scale * noise))
            template_prices[date_str] = sales_lookup[date_str][template_name]["price"]
        
        records.append({
            "date": date_str,
            "product_name": product_name,
            "quantity": qty,
            "unit_price_myr": price_myr,
            "source": "synthetic",
        })
        current += timedelta(days=1)
    
    # Extend beyond French Bakery range for synthetic too
    pattern = {}
    for r in records:
        d = datetime.strptime(r["date"], "%Y-%m-%d")
        if d.year == 2021:
            key = d.strftime("%m-%d")
            pattern[key] = r["quantity"]
    
    # Monday = closed
    for r in records:
        d = datetime.strptime(r["date"], "%Y-%m-%d")
        if d.weekday() == 0:
            r["quantity"] = 0
        elif r["quantity"] == 0 and d > fb_end:
            key = d.strftime("%m-%d")
            base_qty = pattern.get(key, 0)
            if base_qty > 0:
                noise = np.random.normal(1.0, 0.15)
                r["quantity"] = max(0, int(base_qty * noise))
    
    return records

# ============================================================
# Generate all data
# ============================================================
start = datetime.strptime(START_DATE, "%Y-%m-%d")
end = datetime.strptime(END_DATE, "%Y-%m-%d")

all_records = []

print("\n=== Real products (9) ===")
for prod_key, french_name in REAL_PRODUCTS.items():
    records = get_daily_sales(prod_key, french_name, start, end)
    extend_seasonal(records, fb_end)
    total_qty = sum(r["quantity"] for r in records)
    nonzero_days = sum(1 for r in records if r["quantity"] > 0)
    print(f"  {prod_key:>22}: {total_qty:>8,} units, {nonzero_days:>4} selling days")
    all_records.extend(records)

print("\n=== Synthetic products (21) ===")
for prod_key, cfg in SYNTHETIC_PRODUCTS.items():
    records = generate_synthetic(prod_key, cfg["template"], cfg["scale"], cfg["price_myr"], start, end)
    total_qty = sum(r["quantity"] for r in records)
    nonzero_days = sum(1 for r in records if r["quantity"] > 0)
    print(f"  {prod_key:>22}: {total_qty:>8,} units, {nonzero_days:>4} selling days (template: {cfg['template']}, scale: {cfg['scale']})")
    all_records.extend(records)

# ============================================================
# Save output
# ============================================================
df_out = pd.DataFrame(all_records)
df_out = df_out.sort_values(["date", "product_name"])

# CSV
csv_path = os.path.join(OUTPUT_DIR, "bakery_sales_3year.csv")
df_out.to_csv(csv_path, index=False)
print(f"\nSaved {len(df_out):,} rows to {csv_path}")

# Summary
print(f"\n=== Summary ===")
print(f"Total products: {df_out['product_name'].nunique()}")
print(f"Date range: {df_out['date'].min()} to {df_out['date'].max()}")
print(f"Total units sold: {df_out['quantity'].sum():,}")
for src, grp in df_out.groupby("source"):
    print(f"  {src}: {len(grp):>8,} rows, {grp['quantity'].sum():>10,} units")

# Per-product stats
print(f"\n=== Per-Product Summary ===")
for prod in sorted(df_out["product_name"].unique()):
    pdf = df_out[df_out["product_name"] == prod]
    total = pdf["quantity"].sum()
    avg_daily = pdf[pdf["quantity"] > 0]["quantity"].mean()
    print(f"  {prod:>22}: {total:>8,} total, {avg_daily:>6.1f} avg daily")
