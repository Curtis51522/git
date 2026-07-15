import argparse
import json
from datetime import date, datetime
from decimal import Decimal, InvalidOperation


QUANTITY_PRECISION = Decimal("0.000001")


def _decimal(value, label):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid decimal for {label}") from exc
    if not number.is_finite() or number < 0:
        raise ValueError(f"Invalid decimal for {label}")
    return number


def _production_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("production_date must use YYYY-MM-DD format") from exc


def _production_time(value):
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"Invalid production time: {value}") from exc


def _production_reference(production_time):
    return "production:" + production_time.strftime("%Y%m%d%H%M%S%f")


def calculate_expected_outflows(batch_rows, recipe_rows):
    recipes_by_product = {}
    for row in recipe_rows:
        product_name = str(row["product_name"])
        material_name = str(row["material_name"])
        product_recipes = recipes_by_product.setdefault(product_name, {})
        if material_name in product_recipes:
            raise ValueError(
                f"Duplicate recipe material for {product_name}: {material_name}"
            )
        product_recipes[material_name] = row

    expected = {}
    for batch in batch_rows:
        product_name = str(batch["product_name"])
        quantity = _decimal(batch["quantity_initial"], f"{product_name} batch quantity")
        if quantity <= 0 or quantity != quantity.to_integral_value():
            raise ValueError(f"Invalid batch quantity for {product_name}")
        if product_name not in recipes_by_product:
            raise ValueError(f"Missing product recipe: {product_name}")

        created_at = _production_time(batch["production_time"])
        reference = _production_reference(created_at)
        group = expected.setdefault(reference, {})
        for material_name, recipe in recipes_by_product[product_name].items():
            per_unit = _decimal(
                recipe["quantity_per_unit"],
                f"{product_name}/{material_name} quantity_per_unit",
            )
            if per_unit <= 0:
                raise ValueError(
                    f"Invalid recipe quantity for {product_name}/{material_name}"
                )
            wastage = _decimal(recipe["wastage_pct"], f"{product_name} wastage")
            unit = str(recipe["unit"])
            category = str(recipe["category"])
            tracked = bool(recipe["track_inventory"])

            required = per_unit * quantity
            if tracked and unit.lower() != "pcs" and category.lower() != "packaging":
                required *= Decimal("1") + wastage
            required = required.quantize(QUANTITY_PRECISION)

            existing = group.get(material_name)
            if existing is not None:
                if (
                    existing["unit"] != unit
                    or existing["track_inventory"] != tracked
                ):
                    raise ValueError(f"Inconsistent material metadata: {material_name}")
                existing["quantity"] = (
                    existing["quantity"] + required
                ).quantize(QUANTITY_PRECISION)
            else:
                group[material_name] = {
                    "quantity": required,
                    "unit": unit,
                    "track_inventory": tracked,
                    "created_at": created_at,
                }
    return expected


def _existing_totals(existing_outflows):
    totals = {}
    for row in existing_outflows:
        material_name = str(row["material_name"])
        quantity = _decimal(row["quantity"], f"{material_name} old outflow")
        totals[material_name] = (
            totals.get(material_name, Decimal("0")) + quantity
        ).quantize(QUANTITY_PRECISION)
    return totals


def _expected_totals(expected_outflows):
    totals = {}
    for group in expected_outflows.values():
        for material_name, row in group.items():
            totals[material_name] = (
                totals.get(material_name, Decimal("0")) + row["quantity"]
            ).quantize(QUANTITY_PRECISION)
    return totals


def calculate_stock_adjustments(stocks, existing_outflows, expected_outflows):
    old_totals = _existing_totals(existing_outflows)
    expected_totals = _expected_totals(expected_outflows)
    adjustments = {}
    for material_name in sorted(set(old_totals) | set(expected_totals)):
        stock = stocks.get(material_name)
        if stock is None:
            raise ValueError(f"Missing raw material stock: {material_name}")
        if not bool(stock["track_inventory"]):
            continue

        current = _decimal(
            stock["stock_quantity"], f"{material_name} current stock"
        ).quantize(QUANTITY_PRECISION)
        old = old_totals.get(material_name, Decimal("0.000000"))
        expected = expected_totals.get(material_name, Decimal("0.000000"))
        new_stock = (current + old - expected).quantize(QUANTITY_PRECISION)
        if new_stock < 0:
            raise ValueError(
                f"Reconciliation would create negative stock for {material_name}"
            )
        if old != expected:
            adjustments[material_name] = {
                "old_outflow": old,
                "expected_outflow": expected,
                "current_stock": current,
                "new_stock": new_stock,
            }
    return adjustments


def _load_reconciliation_state(connection, production_date, lock):
    date_value = _production_date(production_date).isoformat()
    lock_clause = " FOR UPDATE" if lock else ""
    cursor = connection.cursor(dictionary=True)
    try:
        cursor.execute(
            """
            SELECT product_name, quantity_initial, production_time
            FROM batch_inventory
            WHERE DATE(production_time) = %s
            ORDER BY production_time, product_name
            """
            + lock_clause,
            (date_value,),
        )
        batches = cursor.fetchall()
        if not batches:
            raise ValueError(f"No production batches found for {date_value}")

        product_names = sorted({str(row["product_name"]) for row in batches})
        placeholders = ",".join(["%s"] * len(product_names))
        cursor.execute(
            f"""
            SELECT
                p.product_name,
                p.wastage_pct,
                pr.material_name,
                pr.quantity_per_unit,
                rm.unit,
                rm.category,
                rm.track_inventory
            FROM products p
            JOIN product_recipes pr ON pr.product_name = p.product_name
            JOIN raw_materials rm ON rm.material_name = pr.material_name
            WHERE p.product_name IN ({placeholders})
            ORDER BY p.product_name, pr.material_name
            """,
            product_names,
        )
        recipes = cursor.fetchall()

        cursor.execute(
            """
            SELECT id, material_name, quantity, unit, reference, created_at
            FROM material_transactions
            WHERE transaction_type = 'outflow'
              AND reference LIKE 'production:%'
              AND DATE(created_at) = %s
            ORDER BY reference, material_name, id
            """
            + lock_clause,
            (date_value,),
        )
        existing_outflows = cursor.fetchall()

        material_names = sorted(
            {str(row["material_name"]) for row in recipes}
            | {str(row["material_name"]) for row in existing_outflows}
        )
        material_placeholders = ",".join(["%s"] * len(material_names))
        cursor.execute(
            f"""
            SELECT material_name, stock_quantity, track_inventory
            FROM raw_materials
            WHERE material_name IN ({material_placeholders})
            ORDER BY material_name
            """
            + lock_clause,
            material_names,
        )
        stocks = {
            str(row["material_name"]): row for row in cursor.fetchall()
        }
    finally:
        cursor.close()

    return {
        "batches": batches,
        "recipes": recipes,
        "existing_outflows": existing_outflows,
        "stocks": stocks,
    }


def _changed_outflow_materials(existing_outflows, expected_outflows):
    old_totals = _existing_totals(existing_outflows)
    expected_totals = _expected_totals(expected_outflows)
    return sorted(
        material_name
        for material_name in set(old_totals) | set(expected_totals)
        if old_totals.get(material_name, Decimal("0.000000"))
        != expected_totals.get(material_name, Decimal("0.000000"))
    )


def _build_report(production_date, state, expected_outflows, adjustments):
    changed_outflows = _changed_outflow_materials(
        state["existing_outflows"], expected_outflows
    )
    return {
        "production_date": _production_date(production_date).isoformat(),
        "batch_rows": len(state["batches"]),
        "batch_units": sum(int(row["quantity_initial"]) for row in state["batches"]),
        "production_group_count": len(expected_outflows),
        "existing_outflow_rows": len(state["existing_outflows"]),
        "expected_outflow_rows": sum(
            len(group) for group in expected_outflows.values()
        ),
        "changed_material_count": len(adjustments),
        "changed_outflow_material_count": len(changed_outflows),
        "changed_outflow_materials": changed_outflows,
        "stock_adjustments": adjustments,
    }


def _apply_reconciliation(
    connection,
    production_date,
    expected_outflows,
    adjustments,
):
    date_value = _production_date(production_date).isoformat()
    cursor = connection.cursor()
    try:
        for material_name, adjustment in adjustments.items():
            cursor.execute(
                """
                UPDATE raw_materials
                SET stock_quantity = %s
                WHERE material_name = %s AND track_inventory = 1
                """,
                (adjustment["new_stock"], material_name),
            )
            if cursor.rowcount != 1:
                raise ValueError(
                    f"Material stock changed during reconciliation: {material_name}"
                )

        cursor.execute(
            """
            DELETE FROM material_transactions
            WHERE transaction_type = 'outflow'
              AND reference LIKE 'production:%'
              AND DATE(created_at) = %s
            """,
            (date_value,),
        )

        insert_sql = """
            INSERT INTO material_transactions (
                material_name,
                transaction_type,
                quantity,
                unit,
                reference,
                created_at
            ) VALUES (%s, %s, %s, %s, %s, %s)
        """
        for reference, group in sorted(expected_outflows.items()):
            for material_name, row in sorted(group.items()):
                cursor.execute(
                    insert_sql,
                    (
                        material_name,
                        "outflow",
                        row["quantity"],
                        row["unit"],
                        reference,
                        row["created_at"],
                    ),
                )
    finally:
        cursor.close()


def _outflow_signature(rows):
    signature = {}
    for row in rows:
        key = (str(row["reference"]), str(row["material_name"]))
        value = (
            _decimal(row["quantity"], f"{key} quantity").quantize(
                QUANTITY_PRECISION
            ),
            str(row["unit"]),
        )
        signature[key] = value
    return signature


def _expected_signature(expected_outflows):
    return {
        (reference, material_name): (row["quantity"], row["unit"])
        for reference, group in expected_outflows.items()
        for material_name, row in group.items()
    }


def reconcile(connection, production_date, dry_run=True):
    production_date = _production_date(production_date)
    if dry_run:
        state = _load_reconciliation_state(connection, production_date, False)
        expected = calculate_expected_outflows(state["batches"], state["recipes"])
        adjustments = calculate_stock_adjustments(
            state["stocks"], state["existing_outflows"], expected
        )
        return _build_report(production_date, state, expected, adjustments)

    connection.start_transaction()
    try:
        state = _load_reconciliation_state(connection, production_date, True)
        expected = calculate_expected_outflows(state["batches"], state["recipes"])
        adjustments = calculate_stock_adjustments(
            state["stocks"], state["existing_outflows"], expected
        )
        report = _build_report(production_date, state, expected, adjustments)
        _apply_reconciliation(
            connection,
            production_date,
            expected,
            adjustments,
        )

        final_state = _load_reconciliation_state(connection, production_date, False)
        if _outflow_signature(final_state["existing_outflows"]) != _expected_signature(
            expected
        ):
            raise ValueError("Production material reconciliation audit failed")
        connection.commit()
        report["applied"] = True
        return report
    except Exception:
        connection.rollback()
        raise


def _json_default(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser(
        description="Reconcile date-scoped bakery production material outflows."
    )
    parser.add_argument("--date", required=True, help="Production date in YYYY-MM-DD format.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the reconciliation. The default is a read-only dry-run.",
    )
    args = parser.parse_args()

    from db.mysql_client import get_db

    connection = get_db(autocommit=False)
    try:
        report = reconcile(connection, args.date, dry_run=not args.apply)
        report["applied"] = bool(args.apply)
        print(json.dumps(report, indent=2, default=_json_default))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
