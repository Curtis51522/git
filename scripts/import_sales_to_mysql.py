#!/usr/bin/env python
"""Import bakery_sales_raw.csv into MySQL with real product costs from recipes."""
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.mysql_client import get_db
from s3_scheduling.scheduler import PRODUCT_RECIPES, _DEFAULT_RECIPE

# ============================================================
# INGREDIENT MARKET PRICES (CNY per gram or ml, Guangzhou 2024)
# ============================================================
INGREDIENT_PRICES = {
    "flour_g": 0.008,       # 8 CNY/kg
    "butter_g": 0.060,      # 60 CNY/kg
    "sugar_g": 0.006,       # 6 CNY/kg
    "egg_whole_g": 0.020,   # ~1 CNY per 50g egg
    "egg_yolk_g": 0.030,
    "egg_white_g": 0.015,
    "milk_ml": 0.012,       # 12 CNY/L
    "chocolate_g": 0.080,   # 80 CNY/kg
}

# Coffee/drink material costs (CNY per cup)
DRINK_COSTS = {
    "latte": 2.50, "americano": 1.80, "cappuccino": 2.80,
    "mocha": 3.20, "espresso": 1.50, "flat_white": 2.60,
    "caramel_macchiato": 3.00, "cold_brew": 3.00,
    "hot_chocolate": 2.00, "matcha_latte": 3.50,
    "milk_tea": 1.80, "chai_latte": 2.20,
    "earl_grey": 0.80, "english_breakfast": 0.80, "lemonade": 0.60,
}

def calc_bread_cost(product_name):
    """Calculate material cost from recipe."""
    recipe = PRODUCT_RECIPES.get(product_name, _DEFAULT_RECIPE)
    cost = 0
    for ing, price in INGREDIENT_PRICES.items():
        cost += recipe.get(ing, 0) * price
    return round(cost, 2)

def main():
    db = get_db()
    cur = db.cursor()

    # Read CSV
    print("Loading CSV...")
    df = pd.read_csv("data/bakery_sales_raw.csv")
    df["date"] = pd.to_datetime(df["date"])
    print(f"  {len(df)} rows, {df['ticket_id'].nunique()} tickets")

    # 1. Create tables
    print("Creating tables...")
    cur.execute("SET FOREIGN_KEY_CHECKS=0")
    cur.execute("DROP TABLE IF EXISTS order_items")
    cur.execute("DROP TABLE IF EXISTS orders")
    cur.execute("DROP TABLE IF EXISTS products")
    cur.execute("DROP TABLE IF EXISTS payments")
    cur.execute("SET FOREIGN_KEY_CHECKS=1")

    cur.execute("""
        CREATE TABLE products (
            id INT AUTO_INCREMENT PRIMARY KEY,
            product_name VARCHAR(50) UNIQUE NOT NULL,
            category VARCHAR(20) NOT NULL,
            unit_price DECIMAL(8,2) NOT NULL,
            material_cost DECIMAL(8,2) DEFAULT 0
        )
    """)
    cur.execute("""
        CREATE TABLE orders (
            id INT AUTO_INCREMENT PRIMARY KEY,
            ticket_id VARCHAR(20) NOT NULL,
            order_date DATE NOT NULL,
            order_time TIME,
            total_amount DECIMAL(10,2) DEFAULT 0,
            total_profit DECIMAL(10,2) DEFAULT 0,
            item_count INT DEFAULT 0,
            payment_method VARCHAR(10) DEFAULT 'cash',
            INDEX idx_date (order_date)
        )
    """)
    cur.execute("""
        CREATE TABLE order_items (
            id INT AUTO_INCREMENT PRIMARY KEY,
            order_id INT,
            product_name VARCHAR(50),
            quantity INT DEFAULT 1,
            unit_price DECIMAL(8,2),
            material_cost DECIMAL(8,2),
            line_total DECIMAL(10,2),
            line_profit DECIMAL(10,2),
            FOREIGN KEY (order_id) REFERENCES orders(id),
            INDEX idx_order (order_id)
        )
    """)
    cur.execute("""
        CREATE TABLE payments (
            id INT AUTO_INCREMENT PRIMARY KEY,
            order_id INT,
            amount DECIMAL(10,2),
            payment_method VARCHAR(10),
            payment_date DATE,
            FOREIGN KEY (order_id) REFERENCES orders(id)
        )
    """)

    # 2. Insert products
    print("Inserting products...")
    COFFEE = {"latte","americano","cappuccino","mocha","espresso","flat_white",
              "caramel_macchiato","cold_brew","hot_chocolate","matcha_latte",
              "milk_tea","chai_latte","earl_grey","english_breakfast","lemonade"}

    avg_prices = df.groupby("product_name")["unit_price_cny"].mean().to_dict()
    products_inserted = 0
    for pname in sorted(avg_prices.keys()):
        cat = "coffee" if pname in COFFEE else "bakery"
        price = round(avg_prices[pname], 2)
        if cat == "bakery":
            cost = calc_bread_cost(pname)
        else:
            cost = DRINK_COSTS.get(pname, 1.50)
        cur.execute(
            "INSERT INTO products (product_name, category, unit_price, material_cost) VALUES (%s,%s,%s,%s)",
            (pname, cat, price, cost)
        )
        products_inserted += 1
    print(f"  {products_inserted} products inserted")

    # Build cost lookup
    cur.execute("SELECT product_name, material_cost FROM products")
    cost_map = {row[0]: float(row[1]) for row in cur.fetchall()}

    # 3. Group by ticket to create orders
    print("Creating orders (grouping by ticket_id)...")
    tickets = df.groupby(["date", "ticket_id"]).agg(
        order_time=("time", "first"),
        items=("product_name", list),
        qtys=("quantity", list),
        uprices=("unit_price_cny", list)
    ).reset_index()

    order_count = 0
    item_count = 0
    batch_size = 5000

    for i, row in tickets.iterrows():
        order_date = row["date"].strftime("%Y-%m-%d")
        order_time = str(row["order_time"])
        ticket_id = str(int(row["ticket_id"]))

        total = 0
        profit = 0
        n_items = 0

        for j in range(len(row["items"])):
            pname = row["items"][j]
            qty = int(row["qtys"][j])
            uprice = float(row["uprices"][j])
            cost = cost_map.get(pname, uprice * 0.30)
            line_total = qty * uprice
            line_profit = line_total - (qty * cost)
            total += line_total
            profit += line_profit
            n_items += qty

        cur.execute(
            "INSERT INTO orders (ticket_id, order_date, order_time, total_amount, total_profit, item_count) VALUES (%s,%s,%s,%s,%s,%s)",
            (ticket_id, order_date, order_time, round(total, 2), round(profit, 2), n_items)
        )
        order_id = cur.lastrowid

        for j in range(len(row["items"])):
            pname = row["items"][j]
            qty = int(row["qtys"][j])
            uprice = float(row["uprices"][j])
            cost = cost_map.get(pname, uprice * 0.30)
            line_total = qty * uprice
            line_profit = line_total - (qty * cost)
            cur.execute(
                "INSERT INTO order_items (order_id, product_name, quantity, unit_price, material_cost, line_total, line_profit) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (order_id, pname, qty, uprice, cost, round(line_total, 2), round(line_profit, 2))
            )
            item_count += 1

        order_count += 1
        if order_count % 10000 == 0:
            print(f"  {order_count} orders, {item_count} items...")
            db.commit()

    db.commit()
    print(f"Done: {order_count} orders, {item_count} items imported")
    print(f"  Date range: {df['date'].min().date()} ~ {df['date'].max().date()}")

if __name__ == "__main__":
    main()
