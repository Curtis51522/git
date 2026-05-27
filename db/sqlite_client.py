"""
SQLite database client - replaces Supabase for local operation.
Drop-in replacement: same table names, same column names as Supabase schema.
"""
import sqlite3, os, threading
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bakery.db")
_local = threading.local()

def get_db():
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn

def init_db():
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL DEFAULT 'hash123',
            role TEXT NOT NULL DEFAULT 'staff'
        );
        CREATE TABLE IF NOT EXISTS employees (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            skills TEXT NOT NULL DEFAULT '["bakery"]',
            min_hours_per_week REAL NOT NULL DEFAULT 15.0,
            max_hours_per_week REAL NOT NULL DEFAULT 40.0,
            available INTEGER NOT NULL DEFAULT 1,
            rest_days_per_week INTEGER NOT NULL DEFAULT 1,
            unavailable_dates TEXT NOT NULL DEFAULT '[]'
        );
        CREATE TABLE IF NOT EXISTS shift_schedule (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            schedule_date TEXT NOT NULL,
            time_slot TEXT NOT NULL,
            employee_id TEXT,
            employee_name TEXT,
            role TEXT DEFAULT 'bakery',
            staff_count INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS batch_inventory (
            batch_id TEXT PRIMARY KEY,
            product_name TEXT NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 0,
            production_time TEXT NOT NULL,
            tray_color TEXT DEFAULT 'green',
            freshness_status TEXT DEFAULT 'Fresh',
            quantity_initial INTEGER,
            quantity_remaining INTEGER,
            sales_area TEXT DEFAULT 'Fresh Area'
        );
        CREATE TABLE IF NOT EXISTS inventory_transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_type TEXT NOT NULL,
            batch_id TEXT,
            product_name TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL DEFAULT 0,
            discount_applied REAL DEFAULT 0,
            freshness_status TEXT,
            transaction_time TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    db.commit()

def seed_defaults(db):
    if db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        db.execute("INSERT INTO users(username,password_hash,role) VALUES(?,?,?)", ("manager", "hash123", "manager"))
        db.execute("INSERT INTO users(username,password_hash,role) VALUES(?,?,?)", ("staff1", "hash123", "staff"))
    if db.execute("SELECT COUNT(*) FROM employees").fetchone()[0] == 0:
        emps = [
            ("E001","Ali",   '["bakery","coffee","cashier"]',15,40,1,1,"[]"),
            ("E002","Mei",   '["bakery","coffee","cashier"]',15,40,1,1,"[]"),
            ("E003","Raj",   '["bakery","coffee","cashier"]',15,40,1,1,"[]"),
            ("E004","Siti",  '["bakery","coffee","cashier"]',15,40,1,1,"[]"),
            ("E005","Ahmad", '["bakery","coffee","cashier"]',15,40,1,1,"[]"),
            ("E006","Priya", '["bakery","coffee","cashier"]',15,40,1,1,"[]"),
        ]
        db.executemany("INSERT INTO employees(id,name,skills,min_hours_per_week,max_hours_per_week,available,rest_days_per_week,unavailable_dates) VALUES(?,?,?,?,?,?,?,?)", emps)
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
        self._where.append(f"{col} = ?"); self._params.append(val); return self
    def gt(self, col, val):
        self._where.append(f"{col} > ?"); self._params.append(val); return self
    def gte(self, col, val):
        self._where.append(f"{col} >= ?"); self._params.append(val); return self
    def lte(self, col, val):
        self._where.append(f"{col} <= ?"); self._params.append(val); return self
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
        rows = self.db.execute(sql, self._params).fetchall()
        return FakeResponse([dict(r) for r in rows])

    def _exec_insert(self):
        cols = ", ".join(self._insert_data.keys())
        ph = ", ".join("?" * len(self._insert_data))
        vals = list(self._insert_data.values())
        self.db.execute(f"INSERT INTO {self.table} ({cols}) VALUES ({ph})", vals)
        self.db.commit()
        return FakeResponse([self._insert_data])

    def _exec_insert_many(self):
        for row in self._insert_data:
            cols = ", ".join(row.keys())
            ph = ", ".join("?" * len(row))
            self.db.execute(f"INSERT INTO {self.table} ({cols}) VALUES ({ph})", list(row.values()))
        self.db.commit()
        return FakeResponse(self._insert_data)

    def _exec_update(self):
        where = ("WHERE " + " AND ".join(self._where)) if self._where else ""
        sets = ", ".join(f"{k} = ?" for k in self._update_data)
        vals = list(self._update_data.values()) + self._params
        self.db.execute(f"UPDATE {self.table} SET {sets} {where}", vals)
        self.db.commit()
        return FakeResponse([self._update_data])

    def _exec_delete(self):
        where = ("WHERE " + " AND ".join(self._where)) if self._where else ""
        self.db.execute(f"DELETE FROM {self.table} {where}", self._params)
        self.db.commit()
        return FakeResponse([])

# Auto-init
init_db()
seed_defaults(get_db())
print('SQLite ready:', DB_PATH)
