"""
MySQL database client -- drop-in replacement for Supabase / SQLite.
All modules use the same q(db, table) API so switching databases
only requires changing the import in each module.
"""
import mysql.connector
from datetime import datetime

from db.mysql_config import get_mysql_connection_config

DB_CONFIG = get_mysql_connection_config()

def get_db(*, autocommit=True):
    return mysql.connector.connect(**DB_CONFIG, autocommit=autocommit)

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
        self._where.append(f"{col} = %s")
        self._params.append(val); return self
    def neq(self, col, val):
        self._where.append(f'{col} != %s'); self._params.append(val); return self
    def gt(self, col, val):
        self._where.append(f"{col} > %s"); self._params.append(val); return self
    def gte(self, col, val):
        self._where.append(f"{col} >= %s"); self._params.append(val); return self
    def lte(self, col, val):
        self._where.append(f"{col} <= %s"); self._params.append(val); return self
    def lt(self, col, val):
        self._where.append(f"{col} < %s"); self._params.append(val); return self
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
        try:
            cursor.execute(sql, self._params)
            rows = cursor.fetchall()
        finally:
            cursor.close()
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
        try:
            cursor.execute(sql, vals)
        finally:
            cursor.close()
        self._commit_if_autocommit()
        return FakeResponse([self._insert_data])

    def _exec_insert_many(self):
        cursor = self.db.cursor()
        try:
            for row in self._insert_data:
                cols = ", ".join(f"{k}" for k in row)
                ph = ", ".join("%s" for _ in row)
                cursor.execute(f"INSERT INTO {self.table} ({cols}) VALUES ({ph})", list(row.values()))
        finally:
            cursor.close()
        self._commit_if_autocommit()
        return FakeResponse(self._insert_data)

    def _exec_update(self):
        where = ("WHERE " + " AND ".join(self._where)) if self._where else ""
        sets = ", ".join(f"{k} = %s" for k in self._update_data)
        vals = list(self._update_data.values()) + self._params
        sql = f"UPDATE {self.table} SET {sets} {where}"
        cursor = self.db.cursor()
        try:
            cursor.execute(sql, vals)
        finally:
            cursor.close()
        self._commit_if_autocommit()
        return FakeResponse([self._update_data])

    def _exec_delete(self):
        where = ("WHERE " + " AND ".join(self._where)) if self._where else ""
        sql = f"DELETE FROM {self.table} {where}"
        cursor = self.db.cursor()
        try:
            cursor.execute(sql, self._params)
        finally:
            cursor.close()
        self._commit_if_autocommit()
        return FakeResponse([])

    def _commit_if_autocommit(self):
        if getattr(self.db, "autocommit", True):
            self.db.commit()
