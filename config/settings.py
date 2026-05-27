import os
from dotenv import load_dotenv
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
PLANNER_MODEL = "deepseek-chat"
VERIFIER_MODEL = "qwen2.5-72b-instruct"
COMPOSER_MODEL = "gpt-4o-mini"
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 480
YOLO_MODEL_PATH = "models/yolo/best.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.5
TRAY_GREEN_THRESHOLD = 150
TRAY_ORANGE_BLUE_MAX = 80

# 6 bakery product types matching the Roboflow bakery-hbr5t dataset
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
