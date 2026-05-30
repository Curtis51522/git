import os
from dotenv import load_dotenv
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 480
YOLO_MODEL_PATH = "models/yolo/best.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.5
TRAY_GREEN_THRESHOLD = 120
TRAY_ORANGE_BLUE_MAX = 80
TRAY_YELLOW_CHANNEL_MIN = 150
TRAY_YELLOW_BLUE_MAX = 80
TRAY_RED_CHANNEL_MIN = 150
TRAY_RED_GREEN_MAX = 100
TRAY_RED_BLUE_MAX = 80
TRAY_BBOX_PADDING = 15

PRODUCT_TYPES = [
    "donut",
    "croissant",
    "bread_coconut",
    "bread_roll",
    "chiffon",
    "croissant_chocolate",
]

COFFEE_TYPES = [
    "Latte", "Americano", "Cappuccino", "Cold Brew",
    "Espresso", "Flat White", "Mocha",
]
FRESHNESS_STATES = ["Fresh", "Day-1", "Day-2", "Discount"]
FORECAST_FEATURE_COLS = [
    "day_of_week", "is_weekend", "day_of_month", "month",
    "discount_rate", "is_public_holiday", "is_ramadan",
    "temperature", "rainfall", "humidity", "is_rainy",
    "weather_sunny", "weather_cloudy", "weather_rainy", "weather_storm",
    "freshness_Fresh", "freshness_Day-1", "freshness_Day-2", "freshness_Discount",
    "lag_1", "lag_7", "rolling_7d_mean",
]

INTENT_CONFIDENCE_THRESHOLD = 0.75
MAX_RETRIES = 2
MODEL_CACHE_DIR = "models"
CACHE_MANIFEST = "models/cache.json"
COLD_START_WEEKS = 4
MIN_TRAINING_DAYS = 30

INTENT_LABELS = ["stock_query", "waste_analysis", "promo_eval", "schedule_audit", "cross_source_audit", "out_of_scope"]
