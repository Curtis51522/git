from pathlib import Path


FRONTEND = Path("api/module4_frontend/static/index.html")


def test_monthly_kpi_ranking_uses_business_friendly_columns():
    source = FRONTEND.read_text(encoding="utf-8")
    start = source.index("async function loadKPIRanking")
    end = source.index("async function loadAttendance", start)
    block = source[start:end]

    assert "t('Band')" in block
    assert "t('Strength')" in block
    assert "t('Watch')" in block
    assert "t('Pctl')" not in block


def test_monthly_kpi_ranking_maps_percentile_to_band():
    source = FRONTEND.read_text(encoding="utf-8")
    start = source.index("async function loadKPIRanking")
    end = source.index("async function loadAttendance", start)
    block = source[start:end]

    assert "Excellent" in block
    assert "Strong" in block
    assert "Stable" in block
    assert "Needs Attention" in block
