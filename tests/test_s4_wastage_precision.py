import asyncio

import pytest
from fastapi import HTTPException

from api.module4_frontend import bff
from api.module4_frontend.bff import (
    _calculate_material_wastage,
    _normalize_material_wastage,
)


def test_rounding_variance_does_not_create_negative_wastage():
    result = _calculate_material_wastage(
        reference_stock=8.137,
        restocked=0,
        actual_stock=7.413,
        theoretical_consumed=0.7245,
    )

    assert result == (0.724, 0.725, 0.0, 0.0)


def test_real_material_loss_remains_visible():
    result = _calculate_material_wastage(
        reference_stock=8.137,
        restocked=0,
        actual_stock=7.4,
        theoretical_consumed=0.7245,
    )

    assert result == (0.737, 0.725, 0.012, 0.0166)


def test_zero_consumption_display_precision_does_not_create_phantom_waste():
    negative_residue = _calculate_material_wastage(
        reference_stock=5.6004,
        restocked=0,
        actual_stock=5.601,
        theoretical_consumed=0,
    )
    positive_residue = _calculate_material_wastage(
        reference_stock=7.7956,
        restocked=0,
        actual_stock=7.795,
        theoretical_consumed=0,
    )

    assert negative_residue == (0.0, 0.0, 0.0, 0.0)
    assert positive_residue == (0.0, 0.0, 0.0, 0.0)


def test_legacy_negative_wastage_is_normalized_for_readers():
    assert _normalize_material_wastage(-0.001, -0.0014) == (0.0, 0.0)


class _FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, sql, params=None):
        return None

    def fetchall(self):
        return self.rows


class _FakeDb:
    def __init__(self, rows):
        self.cursor_instance = _FakeCursor(rows)

    def cursor(self):
        return self.cursor_instance


class _InventoryCheckCursor:
    def __init__(self):
        self.current_row = None
        self.executed_sql = []

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split()).lower()
        self.executed_sql.append((normalized, params))
        if "select unit, track_inventory from raw_materials" in normalized:
            self.current_row = ("pcs", 1)
        else:
            self.current_row = None

    def fetchone(self):
        return self.current_row


class _InventoryCheckDb:
    def __init__(self):
        self.cursor_instance = _InventoryCheckCursor()
        self.committed = False

    def cursor(self):
        return self.cursor_instance

    def commit(self):
        self.committed = True


def test_inventory_check_uses_stored_precision_for_wastage_equation(monkeypatch):
    fake_db = _InventoryCheckDb()
    fake_db.cursor_instance.current_row = None
    original_execute = fake_db.cursor_instance.execute

    def execute_with_measured_unit(sql, params=None):
        original_execute(sql, params)
        if "select unit, track_inventory from raw_materials" in " ".join(sql.split()).lower():
            fake_db.cursor_instance.current_row = ("kg", 1)

    fake_db.cursor_instance.execute = execute_with_measured_unit
    monkeypatch.setattr(bff, "get_db", lambda: fake_db)
    monkeypatch.setattr(
        bff,
        "_get_theoretical",
        lambda _material_name: (
            7.2596,
            0.536,
            0.0,
            7.7954,
            "2026-07-01 00:00:00",
        ),
    )

    result = asyncio.run(
        bff.inventory_check(
            {
                "check_date": "2026-07-01",
                "counts": [
                    {"material_name": "Tomato Sauce", "actual_stock": 7.259}
                ],
            }
        )
    )

    insert_params = next(
        params
        for sql, params in fake_db.cursor_instance.executed_sql
        if sql.startswith("insert into material_wastage_log")
    )
    assert insert_params[2:] == (7.26, 7.259, 0.536, 0.537, 0.001, 0.0019)
    assert result["results"][0]["wastage_qty"] == 0.001


def test_inventory_check_rejects_fractional_piece_counts(monkeypatch):
    fake_db = _InventoryCheckDb()
    monkeypatch.setattr(bff, "get_db", lambda: fake_db)
    monkeypatch.setattr(
        bff,
        "_get_theoretical",
        lambda _material_name: (10.0, 0.0, 0.0, 10.0, "2026-07-01 00:00:00"),
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            bff.inventory_check(
                {
                    "check_date": "2026-07-01",
                    "counts": [
                        {"material_name": "Cup Regular", "actual_stock": 9.5}
                    ],
                }
            )
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Piece quantities must be whole numbers"
    assert fake_db.committed is False
    assert not any(
        sql.startswith("insert into material_wastage_log")
        or sql.startswith("update raw_materials")
        for sql, _params in fake_db.cursor_instance.executed_sql
    )


def test_wastage_history_normalizes_legacy_negative_values(monkeypatch):
    fake_db = _FakeDb(
        [
            (
                3019,
                "Tomato Sauce",
                "2026-07-20",
                7.413,
                7.413,
                0.725,
                0.724,
                -0.001,
                -0.0014,
                "2026-07-20 05:05:00",
                "kg",
            )
        ]
    )
    monkeypatch.setattr(bff, "get_db", lambda: fake_db)

    result = asyncio.run(bff.inventory_check_history())

    assert result["history"][0]["wastage_qty"] == 0.0
    assert result["history"][0]["wastage_rate"] == 0.0


def test_wastage_summary_normalizes_legacy_negative_values(monkeypatch):
    fake_db = _FakeDb(
        [("Tomato Sauce", 0.725, -0.001, -0.0014, "2026-07-20", "kg")]
    )
    monkeypatch.setattr(bff, "get_db", lambda: fake_db)

    result = asyncio.run(bff.wastage_summary(date="2026-07-20"))

    assert result["summary"][0]["wastage_qty"] == 0.0
    assert result["summary"][0]["wastage_rate"] == 0.0
