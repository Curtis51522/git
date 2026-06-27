import pandas as pd, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.mysql_client import get_db
from s3_scheduling.scheduler import PRODUCT_RECIPES, _DEFAULT_RECIPE

IP = {"flour_g":0.008,"butter_g":0.060,"sugar_g":0.006,"egg_whole_g":0.020,"egg_yolk_g":0.030,"egg_white_g":0.015,"milk_ml":0.012,"chocolate_g":0.080}
DC = {"latte":2.50,"americano":1.80,"cappuccino":2.80,"mocha":3.20,"espresso":1.50,"flat_white":2.60,"caramel_macchiato":3.00,"cold_brew":3.00,"hot_chocolate":2.00,"matcha_latte":3.50,"milk_tea":1.80,"chai_latte":2.20,"earl_grey":0.80,"english_breakfast":0.80,"lemonade":0.60}
COFFEE = {"latte","americano","cappuccino","mocha","espresso","flat_white","caramel_macchiato","cold_brew","hot_chocolate","matcha_latte","milk_tea","chai_latte","earl_grey","english_breakfast","lemonade"}

db = get_db()
cur = db.cursor()

# Drop
for t in ["order_items","payments","orders"]:
    cur.execute(f"DROP TABLE IF EXISTS {t}")
print("Dropped")

# Orders
cur.execute("""
CREATE TABLE orders (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ticket_id VARCHAR(20) NOT NULL,
    order_date DATE NOT NULL,
    order_time TIME,
    cashier_id VARCHAR(10) DEFAULT NULL,
    subtotal DECIMAL(10,2) DEFAULT 0,
    discount_total DECIMAL(10,2) DEFAULT 0,
    total_amount DECIMAL(10,2) DEFAULT 0,
    total_profit DECIMAL(10,2) DEFAULT 0,
    item_count INT DEFAULT 0,
    state VARCHAR(20) DEFAULT 'paid',
    INDEX idx_order_date (order_date),
    INDEX idx_ticket (ticket_id)
)""")

# Payments
cur.execute("""
CREATE TABLE payments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    payment_method VARCHAR(10) DEFAULT NULL,
    payment_date DATE,
    transaction_id VARCHAR(50),
    is_change BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE,
    INDEX idx_payment_order (order_id)
)""")

# Order items
cur.execute("""
CREATE TABLE order_items (
    id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT NOT NULL,
    product_name VARCHAR(50),
    quantity INT DEFAULT 1,
    unit_price DECIMAL(8,2),
    discount_rate DECIMAL(5,3) DEFAULT 0,
    line_total DECIMAL(10,2),
    line_profit DECIMAL(10,2) DEFAULT 0,
    freshness VARCHAR(10) DEFAULT 'Fresh',
    coffee_temp VARCHAR(10) DEFAULT NULL,
    coffee_ice VARCHAR(10) DEFAULT NULL,
    coffee_sugar VARCHAR(10) DEFAULT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE,
    INDEX idx_item_order (order_id)
)""")
print("Tables created")

# Cost map
cur.execute("SELECT product_name, material_cost FROM products")
cost_map = {r[0]: float(r[1]) for r in cur.fetchall()}

# Load and group
df = pd.read_csv("data/bakery_sales_raw.csv")
print(f"Loaded {len(df)} rows")

tk = df.groupby(["date","ticket_id"]).agg(
    order_time=("time","first"),
    items=("product_name",list),
    qtys=("quantity",list),
    prices=("unit_price_cny",list)
).reset_index()
print(f"{len(tk)} tickets")

# Bulk insert orders
BATCH = 5000
order_rows = []
for i, row in tk.iterrows():
    od = str(row["date"])[:10]
    ot = str(row["order_time"])
    tid = str(int(row["ticket_id"]))
    sub = total = prof = n = 0
    for j in range(len(row["items"])):
        pn = row["items"][j]; q = int(row["qtys"][j]); pr = float(row["prices"][j])
        c = cost_map.get(pn, pr*0.30); lt = q*pr; lp = lt - q*c
        sub += lt; prof += lp; n += q
    order_rows.append((tid, od, ot, round(sub,2), 0.0, round(sub,2), round(prof,2), n))

    if len(order_rows) >= BATCH:
        cur.executemany(
            "INSERT INTO orders (ticket_id, order_date, order_time, subtotal, discount_total, total_amount, total_profit, item_count) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            order_rows
        )
        db.commit()
        print(f"  Orders: {i+1}/{len(tk)}")
        order_rows = []

if order_rows:
    cur.executemany(
        "INSERT INTO orders (ticket_id, order_date, order_time, subtotal, discount_total, total_amount, total_profit, item_count) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
        order_rows
    )
    db.commit()
print(f"Orders done: {len(tk)}")

# Payments - only insert amount, skip payment_method (POS records real payment methods)
cur.execute("SELECT id, total_amount, order_date FROM orders ORDER BY id")
order_data = cur.fetchall()
pay_rows = []
for oid, total, odate in order_data:
    pay_rows.append((oid, float(total), None, str(odate)))
    if len(pay_rows) >= BATCH:
        cur.executemany("INSERT INTO payments (order_id, amount, payment_method, payment_date) VALUES (%s,%s,%s,%s)", pay_rows)
        db.commit(); pay_rows = []
if pay_rows:
    cur.executemany("INSERT INTO payments (order_id, amount, payment_method, payment_date) VALUES (%s,%s,%s,%s)", pay_rows)
    db.commit()
print("Payments done")

# Items via join
cur.execute("SELECT id, ticket_id FROM orders")
tid_to_oid = {str(r[1]): r[0] for r in cur.fetchall()}
item_rows = []
for i, row in df.iterrows():
    tid = str(int(row["ticket_id"]))
    oid = tid_to_oid.get(tid)
    if not oid: continue
    pn = row["product_name"]; q = int(row["quantity"]); pr = float(row["unit_price_cny"])
    c = cost_map.get(pn, pr*0.30); lt = q*pr; lp = lt - q*c
    item_rows.append((oid, pn, q, pr, round(lt,2), round(lp,2)))
    if len(item_rows) >= BATCH:
        cur.executemany("INSERT INTO order_items (order_id, product_name, quantity, unit_price, line_total, line_profit) VALUES (%s,%s,%s,%s,%s,%s)", item_rows)
        db.commit(); item_rows = []
        print(f"  Items: {i+1}/{len(df)}")

if item_rows:
    cur.executemany("INSERT INTO order_items (order_id, product_name, quantity, unit_price, line_total, line_profit) VALUES (%s,%s,%s,%s,%s,%s)", item_rows)
    db.commit()
print("Items done")

# Verify
cur.execute("SELECT COUNT(*) FROM orders"); print("Orders:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM payments"); print("Payments:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM order_items"); print("Items:", cur.fetchone()[0])
cur.execute("SELECT MIN(order_date), MAX(order_date) FROM orders"); print("Range:", cur.fetchone())
print("ALL DONE")
