import os
from datetime import date, datetime


FIXED_BUSINESS_DATE_ENV = "BAKERY_FIXED_BUSINESS_DATE"


def _fixed_business_date() -> date | None:
    value = os.getenv(FIXED_BUSINESS_DATE_ENV, "").strip()
    if not value:
        return None
    if len(value) != 10:
        raise ValueError(f"{FIXED_BUSINESS_DATE_ENV} must use YYYY-MM-DD format")
    try:
        selected = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"{FIXED_BUSINESS_DATE_ENV} must use YYYY-MM-DD format"
        ) from exc
    if selected.isoformat() != value:
        raise ValueError(f"{FIXED_BUSINESS_DATE_ENV} must use YYYY-MM-DD format")
    return selected


def business_date_is_fixed() -> bool:
    return _fixed_business_date() is not None


def operation_now(default_factory=None) -> datetime:
    current = default_factory() if default_factory is not None else datetime.now()
    selected = _fixed_business_date()
    if selected is None:
        return current
    return current.replace(
        year=selected.year,
        month=selected.month,
        day=selected.day,
    )
