"""Explicit, transaction-safe inventory seed command."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.mysql_client import DB_CONFIG, get_db


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Replace finished-goods inventory with explicit demo seed data."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the destructive inventory replacement.",
    )
    parser.add_argument(
        "--confirm-database",
        default="",
        help="Database name that must exactly match the configured target.",
    )
    parser.add_argument("--fresh-quantity", type=int, default=20)
    parser.add_argument("--day1-quantity", type=int, default=5)
    return parser, parser.parse_args(argv)


def seed_inventory(db, fresh_quantity=20, day1_quantity=5):
    if fresh_quantity < 0 or day1_quantity < 0:
        raise ValueError("Seed quantities must be non-negative")

    cursor = db.cursor()
    try:
        cursor.execute("DELETE FROM batch_inventory")
        cursor.execute("DELETE FROM inventory_transactions")
        cursor.execute(
            "SELECT product_name FROM products WHERE category = 'bakery' "
            "ORDER BY product_name"
        )
        products = [row[0] for row in cursor.fetchall()]

        current = datetime.now()
        current_text = current.strftime("%Y-%m-%d %H:%M:%S")
        previous_text = (current - timedelta(days=1)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        date_key = current.strftime("%Y%m%d")

        for product_name in products:
            rows = (
                (
                    f"BATCH-{product_name}-F-{date_key}",
                    fresh_quantity,
                    current_text,
                    "green",
                    "Fresh",
                    "Fresh Area",
                ),
                (
                    f"BATCH-{product_name}-D1-{date_key}",
                    day1_quantity,
                    previous_text,
                    "orange",
                    "Day-1",
                    "Day-1 Area",
                ),
            )
            for batch_id, quantity, produced_at, color, freshness, area in rows:
                cursor.execute(
                    "INSERT INTO batch_inventory "
                    "(batch_id, product_name, quantity, production_time, "
                    "tray_color, freshness_status, quantity_initial, "
                    "quantity_remaining, sales_area) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        batch_id,
                        product_name,
                        quantity,
                        produced_at,
                        color,
                        freshness,
                        quantity,
                        quantity,
                        area,
                    ),
                )
                cursor.execute(
                    "INSERT INTO inventory_transactions "
                    "(batch_id, product_name, quantity, transaction_type, "
                    "freshness_status, unit_price) VALUES (%s,%s,%s,%s,%s,%s)",
                    (
                        batch_id,
                        product_name,
                        quantity,
                        "inflow",
                        freshness,
                        0,
                    ),
                )
        db.commit()
        return {
            "products": len(products),
            "batches": len(products) * 2,
        }
    except Exception:
        db.rollback()
        raise
    finally:
        cursor.close()


def main(argv=None, connect=None):
    parser, args = parse_args(argv)
    if not args.apply:
        print("Dry run only. No database changes were made.")
        print(
            "Use --apply --confirm-database "
            f"{DB_CONFIG['database']} to replace inventory explicitly."
        )
        return 0
    if args.confirm_database != DB_CONFIG["database"]:
        parser.error("--confirm-database must match the configured database")

    connector = connect or get_db
    db = connector(autocommit=False)
    try:
        summary = seed_inventory(
            db,
            fresh_quantity=args.fresh_quantity,
            day1_quantity=args.day1_quantity,
        )
    finally:
        db.close()
    print(
        f"Seeded {summary['products']} products and {summary['batches']} batches."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
