from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
from enum import Enum

class FreshnessStatus(str, Enum):
    FRESH = "Fresh"
    DAY1 = "Day-1"
    DAY2 = "Day-2"
    DISCOUNT = "Discount"

class SalesArea(str, Enum):
    FRESH = "Fresh Area"
    DAY1_DISCOUNT = "Day-1 Discount Area"
    EXPIRED = "Expired"

class TransactionType(str, Enum):
    INFLOW = "inflow"
    OUTFLOW = "outflow"

# ---------------------------------------------------------------
# S1 / YOLO
# ---------------------------------------------------------------
class YOLOResult(BaseModel):
    product_name: str
    confidence: float
    bbox: list[float] = []
    quantity: int = 1
    tray_color: str = "unknown"
    freshness_status: FreshnessStatus = FreshnessStatus.FRESH

class DeductRequest(BaseModel):
    items: list[dict] = Field(..., description="[{'product_name': 'Croissant', 'quantity': 2}, ...]")
    unit_price: float = 0.0
    discount_applied: bool = False

class DeductResponse(BaseModel):
    status: str
    deducted: list[dict] = []
    errors: list[str] = []

# ---------------------------------------------------------------
# Batch Inventory
# ---------------------------------------------------------------
class BatchInventory(BaseModel):
    batch_id: str
    product_name: str
    quantity: int = Field(ge=0)
    production_time: datetime
    freshness_status: FreshnessStatus = FreshnessStatus.FRESH
    sales_area: SalesArea = SalesArea.FRESH
    tray_color: Optional[str] = None

class InventoryTransaction(BaseModel):
    transaction_type: TransactionType
    batch_id: str
    product_name: str
    quantity: int
    unit_price: Optional[float] = None
    discount_applied: bool = False
    freshness_status: Optional[str] = None
    transaction_time: datetime = Field(default_factory=datetime.now)

# ---------------------------------------------------------------
# Search
# ---------------------------------------------------------------
class ImageSearchResult(BaseModel):
    product_name: str
    batch_id: str
    quantity: int = 0
    freshness_status: str = "Fresh"
    sales_area: str = "Fresh Area"
    production_time: str = ""

# ---------------------------------------------------------------
# S4 / Auth
# ---------------------------------------------------------------
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    username: str
    role: str

class UserProfile(BaseModel):
    username: str
    role: str
class SalesForecast(BaseModel):
    forecast_date: str
    product_name: str
    freshness_status: str
    predicted_demand: int
    lower_bound: int
    upper_bound: int
    confidence: str
class ComboScore(BaseModel):
    product_name: str
    flavor_pairing: float = 0.0
    discount_value: float = 0.0
    freshness: float = 0.0
    inventory_pressure: float = 0.0
    order_context_match: float = 0.0
    total_score: float = 0.0
    tray_color: Optional[str] = None

class UserRole(str, Enum):
    STAFF = "staff"
    MANAGER = "manager"
