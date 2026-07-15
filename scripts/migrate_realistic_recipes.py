import argparse
import json
from decimal import Decimal
from pathlib import Path

from scripts.recipe_catalog import calculate_material_costs, load_catalog


DEFAULT_CATALOG_PATH = Path("data/reference/product_recipe_catalog.json")
MONEY_PRECISION = Decimal("0.01")


def ensure_track_inventory_column(connection):
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT 1
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = %s
              AND COLUMN_NAME = %s
            LIMIT 1
            """,
            ("raw_materials", "track_inventory"),
        )
        if cursor.fetchone() is None:
            cursor.execute(
                """
                ALTER TABLE raw_materials
                ADD COLUMN track_inventory TINYINT(1) NOT NULL DEFAULT 1
                """
            )
    finally:
        cursor.close()


def validate_database_products(connection, catalog):
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT product_name
            FROM products
            WHERE LOWER(category) = 'bakery'
            ORDER BY product_name
            """
        )
        database_names = {row["product_name"] for row in cursor.fetchall()}
    finally:
        cursor.close()

    catalog_names = {row["product_name"] for row in catalog["products"]}
    if database_names != catalog_names:
        missing = sorted(database_names - catalog_names)
        unknown = sorted(catalog_names - database_names)
        raise ValueError(
            "Catalog product mismatch: "
            f"missing_from_catalog={missing}, unknown_in_database={unknown}"
        )


def upsert_materials(connection, catalog):
    sql = """
        INSERT INTO raw_materials (
            material_name,
            stock_quantity,
            unit,
            unit_price,
            category,
            reorder_point,
            track_inventory
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            unit = VALUES(unit),
            unit_price = VALUES(unit_price),
            category = VALUES(category),
            reorder_point = VALUES(reorder_point),
            track_inventory = VALUES(track_inventory)
    """
    cursor = connection.cursor()
    try:
        for material in catalog["materials"]:
            cursor.execute(
                sql,
                (
                    material["material_name"],
                    Decimal(material["opening_stock"]),
                    material["unit"],
                    Decimal(material["unit_price"]),
                    material["category"],
                    Decimal(material["reorder_point"]),
                    int(material["track_inventory"]),
                ),
            )
    finally:
        cursor.close()


def replace_product_recipes(connection, catalog):
    delete_sql = "DELETE FROM product_recipes WHERE product_name = %s"
    insert_sql = """
        INSERT INTO product_recipes (
            product_name,
            material_name,
            quantity_per_unit
        ) VALUES (%s, %s, %s)
    """
    cursor = connection.cursor()
    try:
        for product in catalog["products"]:
            product_name = product["product_name"]
            cursor.execute(delete_sql, (product_name,))
            cursor.executemany(
                insert_sql,
                [
                    (
                        product_name,
                        ingredient["material_name"],
                        Decimal(ingredient["quantity_per_unit"]),
                    )
                    for ingredient in product["ingredients"]
                ],
            )
    finally:
        cursor.close()


def update_material_costs(connection, catalog):
    costs = calculate_material_costs(catalog)
    cursor = connection.cursor()
    try:
        for product_name, cost in costs.items():
            cursor.execute(
                "UPDATE products SET material_cost = %s WHERE product_name = %s",
                (cost.quantize(MONEY_PRECISION), product_name),
            )
    finally:
        cursor.close()


def validate_database_state(connection, catalog):
    expected_recipes = {
        (product["product_name"], ingredient["material_name"]): (
            Decimal(ingredient["quantity_per_unit"]),
            ingredient["unit"],
        )
        for product in catalog["products"]
        for ingredient in product["ingredients"]
    }

    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT
                pr.product_name,
                pr.material_name,
                pr.quantity_per_unit,
                rm.unit
            FROM product_recipes pr
            JOIN raw_materials rm ON rm.material_name = pr.material_name
            JOIN products p ON p.product_name = pr.product_name
            WHERE LOWER(p.category) = 'bakery'
            ORDER BY pr.product_name, pr.material_name
            """
        )
        actual_recipes = {
            (row["product_name"], row["material_name"]): (
                Decimal(str(row["quantity_per_unit"])),
                row["unit"],
            )
            for row in cursor.fetchall()
        }
        if actual_recipes != expected_recipes:
            raise ValueError("Migrated recipe validation failed")

        cursor.execute(
            """
            SELECT product_name, material_cost
            FROM products
            WHERE LOWER(category) = 'bakery'
            ORDER BY product_name
            """
        )
        actual_costs = {
            row["product_name"]: Decimal(str(row["material_cost"])).quantize(
                MONEY_PRECISION
            )
            for row in cursor.fetchall()
        }
    finally:
        cursor.close()

    expected_costs = {
        product_name: cost.quantize(MONEY_PRECISION)
        for product_name, cost in calculate_material_costs(catalog).items()
    }
    if actual_costs != expected_costs:
        raise ValueError("Migrated material-cost validation failed")


def migrate(connection, catalog_path=DEFAULT_CATALOG_PATH):
    catalog = load_catalog(catalog_path)

    # MySQL DDL commits implicitly, so schema repair is completed before the
    # atomic data migration begins.
    ensure_track_inventory_column(connection)
    if getattr(connection, "in_transaction", False):
        connection.commit()
    connection.start_transaction()
    try:
        validate_database_products(connection, catalog)
        upsert_materials(connection, catalog)
        replace_product_recipes(connection, catalog)
        update_material_costs(connection, catalog)
        validate_database_state(connection, catalog)
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def summarize_catalog(catalog_path=DEFAULT_CATALOG_PATH):
    catalog = load_catalog(catalog_path)
    return {
        "catalog_path": str(catalog_path),
        "product_count": len(catalog["products"]),
        "material_count": len(catalog["materials"]),
        "recipe_row_count": sum(
            len(product["ingredients"]) for product in catalog["products"]
        ),
        "source_count": len(
            {
                product["source_url"]
                for product in catalog["products"]
            }
        ),
        "untracked_materials": sorted(
            material["material_name"]
            for material in catalog["materials"]
            if not material["track_inventory"]
        ),
    }


def main(argv=None, connection_factory=None):
    parser = argparse.ArgumentParser(
        description="Load the validated realistic bakery recipe catalog into MySQL."
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG_PATH,
        help="Path to the recipe catalog JSON file.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the catalog without connecting to the database. This is the default.",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Apply the validated catalog to the database.",
    )
    args = parser.parse_args(argv)

    report = summarize_catalog(args.catalog)
    report["mode"] = "apply" if args.apply else "validate-only"
    report["applied"] = False

    if args.apply:
        if connection_factory is None:
            from db.mysql_client import get_db

            connection_factory = get_db

        connection = connection_factory(autocommit=False)
        try:
            migrate(connection, args.catalog)
        finally:
            connection.close()
        report["applied"] = True

    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
