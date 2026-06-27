import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(r"C:\Users\Curtis\Desktop\learningmaterials\SEMESTER3\bakery-ai-system\scripts\seed_inventory.py"))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
from db.mysql_client import get_db
from datetime import datetime, timedelta

db = get_db(); cur = db.cursor()
cur.execute("DELETE FROM batch_inventory")
cur.execute("DELETE FROM inventory_transactions")
print("Cleared")

cur.execute("SELECT product_name FROM products WHERE category='bakery'")
breads = [r[0] for r in cur.fetchall()]
print(f"Seeding {len(breads)} breads")

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
today = datetime.now().strftime("%Y%m%d")

for b in breads:
    fid = f"BATCH-{b}-F-{today}"
    cur.execute("INSERT INTO batch_inventory (batch_id,product_name,quantity,production_time,tray_color,freshness_status,quantity_initial,quantity_remaining,sales_area) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",(fid,b,20,now,"green","Fresh",20,20,"Fresh Area"))
    cur.execute("INSERT INTO inventory_transactions (batch_id,product_name,quantity,transaction_type,freshness_status,unit_price) VALUES (%s,%s,%s,%s,%s,%s)",(fid,b,20,"inflow","Fresh",0))
    did = f"BATCH-{b}-D1-{today}"
    cur.execute("INSERT INTO batch_inventory (batch_id,product_name,quantity,production_time,tray_color,freshness_status,quantity_initial,quantity_remaining,sales_area) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",(did,b,5,yesterday,"orange","Day-1",5,5,"Day-1 Area"))
    cur.execute("INSERT INTO inventory_transactions (batch_id,product_name,quantity,transaction_type,freshness_status,unit_price) VALUES (%s,%s,%s,%s,%s,%s)",(did,b,5,"inflow","Day-1",0))

db.commit()
cur.execute("SELECT COUNT(*) FROM batch_inventory")
print(f"Rows: {cur.fetchone()[0]}")
cur.execute("SELECT freshness_status,SUM(quantity_remaining) FROM batch_inventory GROUP BY freshness_status")
for r in cur.fetchall(): print(f"  {r[0]}: {r[1]}")
print("Done")
