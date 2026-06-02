# s5-agent-brain config
# Independent multi-agent service - decoupled from main bakery system

PORT = 8001
HOST = "0.0.0.0"

# LLM
LLM_MODEL = "deepseek-chat"
LLM_API_KEY = None  # read from environment or main config
LLM_BASE_URL = "https://api.deepseek.com/v1"

# Product list - must match S1/S2
PRODUCT_NAMES = ["croissant", "donut", "chiffon", "bread_roll", "bread_coconut", "croissant_chocolate"]

# External service endpoints (S1-S3 on :8002)
S1_INVENTORY_URL = "http://localhost:8002/s1/inventory"
S2_FORECAST_URL  = "http://localhost:8002/s2/forecast"
S3_SCHEDULE_URL  = "http://localhost:8002/s3/schedule"
S3_KPI_URL       = "http://localhost:8002/s3/kpi"
S4_COMBO_URL     = "http://localhost:8002/s4/combo"

# Agent parallelism
AGENT_TIMEOUT = 15  # seconds per agent call
