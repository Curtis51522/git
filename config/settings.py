import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BAKERY_ENV = os.getenv("BAKERY_ENV", "development").strip().lower()
_JWT_SECRET_FROM_ENV = os.getenv("JWT_SECRET", "").strip()
JWT_SECRET_IS_EPHEMERAL = not bool(_JWT_SECRET_FROM_ENV)
JWT_SECRET = _JWT_SECRET_FROM_ENV or secrets.token_urlsafe(48)
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 480
ALLOW_LEGACY_PLAINTEXT_LOGIN = os.getenv(
    "BAKERY_ALLOW_LEGACY_PLAINTEXT_LOGIN",
    "0",
) == "1"
S5_BASE_URL = os.getenv("S5_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
S5_DISCOUNT_URL = os.getenv("S5_DISCOUNT_URL", f"{S5_BASE_URL}/discounts")
S5_DISCOUNT_TIMEOUT_SECONDS = float(os.getenv("S5_DISCOUNT_TIMEOUT_SECONDS", "5"))
YOLO_MODEL_PATH = str(_PROJECT_ROOT / "models" / "yolo" / "best.pt")
YOLO_CONFIDENCE_THRESHOLD = 0.5
BAKERY_PRODUCT_TYPES = [
    "apple_pie", "bagel", "baguette", "bread_coconut", "bread_roll", "brioche",
    "brownie", "chiffon", "chocolate_cake", "chocopie", "cookie", "cornbread",
    "cream_horn", "croissant", "croissant_chocolate", "donut", "eggtart",
    "flatbread", "macaron", "mantequilla", "melon_bread", "muffin", "pancake",
    "pandesal", "pizza_bread", "pullman", "soboru_bread", "sourdough",
    "stickbread", "tostada",
]
BEVERAGE_PRODUCT_TYPES = [
    "americano", "cappuccino", "caramel_macchiato", "chai_latte", "cold_brew",
    "earl_grey", "english_breakfast", "espresso", "flat_white", "hot_chocolate",
    "latte", "lemonade", "matcha_latte", "milk_tea", "mocha",
]
PRODUCT_TYPES = BAKERY_PRODUCT_TYPES + BEVERAGE_PRODUCT_TYPES

INTENT_CONFIDENCE_THRESHOLD = 0.75
MAX_RETRIES = 2
COLD_START_WEEKS = 4
MIN_TRAINING_DAYS = 30

INTENT_LABELS = ["stock_query", "waste_analysis", "promo_eval", "schedule_audit", "cross_source_audit", "profit_analysis", "out_of_scope"]
# Coffee-to-bakery demand ratio for staffing estimation
COFFEE_DEMAND_RATIO = 0.6
# Default production capacity fallback (per product values in products.daily_capacity)
PRODUCTION_CAPACITY = 50  # fallback only, prefer get_capacity()
def get_capacity(product_name: str) -> int:
    """Read daily_capacity from products table, fallback to 50."""
    try:
        from db.mysql_client import get_db
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT daily_capacity FROM products WHERE product_name = %s", (product_name,))
        row = cur.fetchone()
        cur.close()
        return int(row[0]) if row else PRODUCTION_CAPACITY
    except Exception:
        return PRODUCTION_CAPACITY
# --- Verifier thresholds ---
BAKER_UNITS_PER_HOUR = 15        # units a baker can produce per hour
BAKER_HOURS_PER_SHIFT = 8        # hours per shift for capacity calc
STOCKOUT_THRESHOLD = 0.2         # inventory/forecast below this = stockout risk
OVERSTOCK_THRESHOLD = 2.0        # inventory/forecast above this = overstock
FORECAST_CHANGE_NORMAL_PCT = 15  # forecast change % within this = normal
# Demand level absolute thresholds (daily total units)
DEMAND_HIGH_THRESHOLD = 280
DEMAND_LOW_THRESHOLD = 230
