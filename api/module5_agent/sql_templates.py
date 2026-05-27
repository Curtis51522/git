"""
Parameterized SQL templates for the 4 high-frequency S5 business scenarios.

Design rationale:
- LLM-generated SQL has inherent accuracy risk (~80% EX).
- For the 4 standard queries the SQL shape is known and stable.
- We pre-write parameterized templates so the Composer fills in values
  rather than generating SQL from scratch.
- Non-standard / ad-hoc queries still fall back to LLM SQL generation.
"""

from typing import Dict, Optional


# ======================================================================
# SQL Templates
# ======================================================================

SQL_TEMPLATES: Dict[str, str] = {

    # ------------------------------------------------------------------
    # Scenario 1: Smart Stock Preparation
    # "How many Croissants should I prepare for tomorrow?"
    # ------------------------------------------------------------------
    "stock_query": """
        WITH forecast AS (
            SELECT predicted_quantity AS forecast_qty
            FROM sales_forecasts
            WHERE product_name = %(product_name)s
              AND forecast_date = %(target_date)s
            LIMIT 1
        ),
        current_stock AS (
            SELECT COALESCE(SUM(quantity_remaining), 0) AS stock_qty
            FROM batch_inventory
            WHERE product_name = %(product_name)s
              AND freshness_status != 'expired'
        )
        SELECT
            f.forecast_qty,
            s.stock_qty,
            GREATEST(f.forecast_qty - s.stock_qty, 0) AS suggested_restock
        FROM forecast f, current_stock s;
    """,

    # ------------------------------------------------------------------
    # Scenario 2: Waste Root-Cause Analysis
    # "Why was waste so high this week?"
    # ------------------------------------------------------------------
    "waste_analysis": """
        SELECT
            d.transaction_date,
            d.actual_sales,
            f.predicted_quantity AS forecast_sales,
            (f.predicted_quantity - d.actual_sales) AS deviation,
            CASE WHEN f.predicted_quantity > 0
                 THEN ROUND((f.predicted_quantity - d.actual_sales)
                            / f.predicted_quantity * 100, 1)
                 ELSE 0
            END AS deviation_pct,
            sch.headcount
        FROM daily_sales d
        LEFT JOIN sales_forecasts f
            ON d.product_name = f.product_name
           AND d.transaction_date = f.forecast_date
        LEFT JOIN daily_schedule sch
            ON d.transaction_date = sch.schedule_date
        WHERE d.transaction_date BETWEEN %(start_date)s AND %(end_date)s
          AND d.product_name = %(product_name)s
        ORDER BY d.transaction_date;
    """,

    # ------------------------------------------------------------------
    # Scenario 3: Promotion Effectiveness Evaluation
    # "Was the combo promotion effective?"
    # ------------------------------------------------------------------
    "promo_eval": """
        WITH promo_window AS (
            SELECT
                product_name,
                SUM(quantity) AS promo_sales,
                COUNT(DISTINCT transaction_date) AS promo_days,
                SUM(discount_amount) AS total_discount
            FROM inventory_transactions
            WHERE transaction_date BETWEEN %(promo_start)s AND %(promo_end)s
              AND product_name = %(product_name)s
              AND transaction_type = 'sale'
            GROUP BY product_name
        ),
        baseline AS (
            SELECT
                product_name,
                SUM(quantity) AS baseline_sales,
                COUNT(DISTINCT transaction_date) AS baseline_days
            FROM inventory_transactions
            WHERE transaction_date BETWEEN %(baseline_start)s AND %(baseline_end)s
              AND product_name = %(product_name)s
              AND transaction_type = 'sale'
            GROUP BY product_name
        )
        SELECT
            p.product_name,
            p.promo_sales,
            b.baseline_sales,
            ROUND(p.promo_sales::numeric / NULLIF(p.promo_days, 0), 1) AS promo_daily_avg,
            ROUND(b.baseline_sales::numeric / NULLIF(b.baseline_days, 0), 1) AS baseline_daily_avg,
            ROUND(
                (p.promo_sales::numeric / NULLIF(p.promo_days, 0)
                 - b.baseline_sales::numeric / NULLIF(b.baseline_days, 0))
                / NULLIF(b.baseline_sales::numeric / NULLIF(b.baseline_days, 0), 0) * 100,
                1
            ) AS lift_pct,
            p.total_discount
        FROM promo_window p
        LEFT JOIN baseline b ON p.product_name = b.product_name;
    """,

    # ------------------------------------------------------------------
    # Scenario 4: Scheduling Compliance Audit
    # "Are there any scheduling anomalies?"
    # ------------------------------------------------------------------
    "schedule_audit": """
        SELECT
            sch.schedule_date,
            sch.hour,
            sch.headcount,
            COUNT(txn.id) AS transaction_count
        FROM daily_schedule sch
        LEFT JOIN inventory_transactions txn
            ON sch.schedule_date = txn.transaction_date
           AND EXTRACT(HOUR FROM txn.transaction_time) = sch.hour
        WHERE sch.schedule_date BETWEEN %(start_date)s AND %(end_date)s
        GROUP BY sch.schedule_date, sch.hour, sch.headcount
        ORDER BY sch.schedule_date, sch.hour;
    """,
}


# ======================================================================
# Lookup helpers
# ======================================================================

def get_template(intent: str) -> Optional[str]:
    """Return the parameterized SQL template for a given intent, or None."""
    return SQL_TEMPLATES.get(intent)


def list_templates() -> list:
    """Return all supported template intents."""
    return list(SQL_TEMPLATES.keys())
