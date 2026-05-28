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
INTENT_CONFIDENCE_THRESHOLD = 0.75
MAX_RETRIES = 2
MODEL_CACHE_DIR = "models"
CACHE_MANIFEST = "models/cache.json"
COLD_START_WEEKS = 4
MIN_TRAINING_DAYS = 30

INTENT_LABELS = ["stock_query", "waste_analysis", "promo_eval", "schedule_audit", "cross_source_audit", "out_of_scope"]
