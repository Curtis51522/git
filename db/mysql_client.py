"""
MySQL database client -- drop-in replacement for Supabase / SQLite.
All modules use the same q(db, table) API so switching databases
only requires changing the import in each module.
"""
import mysql.connector, threading, json, os
from datetime import datetime

DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "bakery_ai"),
}

_local = threading.local()

def get_db():
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = mysql.connector.connect(**DB_CONFIG, autocommit=True)
    return _local.conn

def init_db():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS " + DB_CONFIG["database"])
    cursor.execute("USE " + DB_CONFIG["database"])
    
    tables = [
        """CREATE TABLE IF NOT EXISTS users (
            username VARCHAR(50) PRIMARY KEY,
            password_hash VARCHAR(255) NOT NULL DEFAULT 'hash123',
            role VARCHAR(20) NOT NULL DEFAULT 'staff'
        )""",
        """CREATE TABLE IF NOT EXISTS employees (
            id VARCHAR(10) PRIMARY KEY,
            name VARCHAR(50) NOT NULL,
            skills JSON NOT NULL DEFAULT ('["bakery"]'),
            min_hours_per_week FLOAT NOT NULL DEFAULT 15.0,
            max_hours_per_week FLOAT NOT NULL DEFAULT 40.0,
            available TINYINT NOT NULL DEFAULT 1,
            rest_days_per_week INT NOT NULL DEFAULT 1,
            unavailable_dates JSON NOT NULL DEFAULT ('[]')
        )""",
        """CREATE TABLE IF NOT EXISTS shift_schedule (
            id INT AUTO_INCREMENT PRIMARY KEY,
            schedule_date DATE NOT NULL,
            time_slot VARCHAR(20) NOT NULL,
            employee_id VARCHAR(10),
            employee_name VARCHAR(50),
            role VARCHAR(30) DEFAULT 'bakery',
            staff_count INT NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS batch_inventory (
            batch_id VARCHAR(50) PRIMARY KEY,
            product_name VARCHAR(50) NOT NULL,
            quantity INT NOT NULL DEFAULT 0,
            production_time DATETIME NOT NULL,
            tray_color VARCHAR(20) DEFAULT 'green',
            freshness_status VARCHAR(30) DEFAULT 'Fresh',
            quantity_initial INT,
            quantity_remaining INT,
            sales_area VARCHAR(30) DEFAULT 'Fresh Area'
        )""",
        """CREATE TABLE IF NOT EXISTS inventory_transactions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            transaction_type VARCHAR(20) NOT NULL,
            batch_id VARCHAR(50),
            product_name VARCHAR(50) NOT NULL,
            quantity INT NOT NULL,
            unit_price FLOAT DEFAULT 0,
            discount_applied FLOAT DEFAULT 0,
            freshness_status VARCHAR(30),
            transaction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
    ]
    
    for t in tables:
        cursor.execute(t)
    db.commit()

def seed_defaults(db):
    cursor = db.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO users(username,password_hash,role) VALUES(%s,%s,%s)", ("manager","hash123","manager"))
        cursor.execute("INSERT INTO users(username,password_hash,role) VALUES(%s,%s,%s)", ("staff1","hash123","staff"))
    
    cursor.execute("SELECT COUNT(*) FROM employees")
    if cursor.fetchone()[0] == 0:
        emps = [
            ("E001","Ali",    '["bakery","coffee","cashier"]',15,40,1,1,"[]"),
            ("E002","Mei",    '["bakery","coffee","cashier"]',15,40,1,1,"[]"),
            ("E003","Raj",    '["bakery","coffee","cashier"]',15,40,1,1,"[]"),
            ("E004","Siti",   '["bakery","coffee","cashier"]',15,40,1,1,"[]"),
            ("E005","Ahmad",  '["bakery","coffee","cashier"]',15,40,1,1,"[]"),
            ("E006","Priya",  '["bakery","coffee","cashier"]',15,40,1,1,"[]"),
        ]
        cursor.executemany(
            "INSERT INTO employees(id,name,skills,min_hours_per_week,max_hours_per_week,available,rest_days_per_week,unavailable_dates) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)",
            emps
        )
    db.commit()


class FakeResponse:
    def __init__(self, data):
        self.data = data

def q(db, table):
    return QueryBuilder(db, table)

class QueryBuilder:
    def __init__(self, db, table):
        self.db = db
        self.table = table
        self._select = "*"
        self._where = []
        self._params = []
        self._order = None
        self._limit = None
        self._insert_data = None
        self._update_data = None
        self._delete = False

    def select(self, columns="*"):
        self._select = columns; return self
    def eq(self, col, val):
        self._where.append(f"{col} = %s" if not col.startswith('') else f"{col} = %s")
        self._params.append(val); return self
    def neq(self, col, val):
        self._where.append(f'{col} != %s'); self._params.append(val); return self
    def gt(self, col, val):
        self._where.append(f"{col} > %s"); self._params.append(val); return self
    def gte(self, col, val):
        self._where.append(f"{col} >= %s"); self._params.append(val); return self
    def lte(self, col, val):
        self._where.append(f"{col} <= %s"); self._params.append(val); return self
    def order(self, col, desc=False):
        self._order = f"ORDER BY {col} {'DESC' if desc else 'ASC'}"; return self
    def limit(self, n):
        self._limit = f"LIMIT {n}"; return self
    def insert(self, data):
        self._insert_data = data; return self
    def update(self, data):
        self._update_data = data; return self
    def delete(self):
        self._delete = True; return self

    def execute(self):
        if self._insert_data:
            if isinstance(self._insert_data, list):
                return self._exec_insert_many()
            return self._exec_insert()
        if self._update_data:
            return self._exec_update()
        if self._delete:
            return self._exec_delete()
        return self._exec_select()

    def _exec_select(self):
        where = ("WHERE " + " AND ".join(self._where)) if self._where else ""
        order = self._order or ""
        limit = self._limit or ""
        sql = f"SELECT {self._select} FROM {self.table} {where} {order} {limit}"
        cursor = self.db.cursor(dictionary=True)
        cursor.execute(sql, self._params)
        rows = cursor.fetchall()
        # Convert non-serializable types
        result = []
        for r in rows:
            d = {}
            for k, v in r.items():
                if isinstance(v, datetime):
                    d[k] = v.strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(v, bytes):
                    d[k] = v.decode()
                else:
                    d[k] = v
            result.append(d)
        return FakeResponse(result)

    def _exec_insert(self):
        cols = ", ".join(f"{k}" for k in self._insert_data)
        ph = ", ".join("%s" for _ in self._insert_data)
        vals = list(self._insert_data.values())
        sql = f"INSERT INTO {self.table} ({cols}) VALUES ({ph})"
        cursor = self.db.cursor()
        cursor.execute(sql, vals)
        self.db.commit()
        return FakeResponse([self._insert_data])

    def _exec_insert_many(self):
        cursor = self.db.cursor()
        for row in self._insert_data:
            cols = ", ".join(f"{k}" for k in row)
            ph = ", ".join("%s" for _ in row)
            cursor.execute(f"INSERT INTO {self.table} ({cols}) VALUES ({ph})", list(row.values()))
        self.db.commit()
        return FakeResponse(self._insert_data)

    def _exec_update(self):
        where = ("WHERE " + " AND ".join(self._where)) if self._where else ""
        sets = ", ".join(f"{k} = %s" for k in self._update_data)
        vals = list(self._update_data.values()) + self._params
        sql = f"UPDATE {self.table} SET {sets} {where}"
        cursor = self.db.cursor()
        cursor.execute(sql, vals)
        self.db.commit()
        return FakeResponse([self._update_data])

    def _exec_delete(self):
        where = ("WHERE " + " AND ".join(self._where)) if self._where else ""
        sql = f"DELETE FROM {self.table} {where}"
        cursor = self.db.cursor()
        cursor.execute(sql, self._params)
        self.db.commit()
        return FakeResponse([])

# Auto-init
init_db()
seed_defaults(get_db())
print("MySQL ready:", DB_CONFIG["database"])
