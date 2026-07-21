import os
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime


REPLAY_START = datetime(2026, 6, 24, 0, 0, 0)
REPLAY_END = datetime(2026, 7, 24, 23, 59, 59)

_operation_time = ContextVar("bakery_operation_time", default=None)


def parse_operation_time(value: str) -> datetime:
    if os.getenv("BAKERY_OPERATION_REPLAY") != "1":
        raise ValueError("Operation replay is not enabled")
    try:
        selected = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Operation time must use ISO local date-time format") from exc
    if selected.tzinfo is not None:
        selected = selected.replace(tzinfo=None)
    if not REPLAY_START <= selected <= REPLAY_END:
        raise ValueError("Operation time is outside the allowed replay period")
    return selected


def get_operation_time():
    return _operation_time.get()


def operation_now(default_factory=None) -> datetime:
    selected = get_operation_time()
    if selected is not None:
        return selected
    return default_factory() if default_factory is not None else datetime.now()


@contextmanager
def operation_time_scope(selected: datetime):
    token = _operation_time.set(selected)
    try:
        yield selected
    finally:
        _operation_time.reset(token)
