from pathlib import Path


INDEX_HTML = (
    Path(__file__).resolve().parents[1]
    / "api"
    / "module4_frontend"
    / "static"
    / "index.html"
)


def test_freshness_ratio_uses_fixed_side_labels_without_pie_callouts():
    source = INDEX_HTML.read_text(encoding="utf-8")

    assert 'id="inv-fresh-ratio-left"' in source
    assert 'id="inv-fresh-ratio-right"' in source
    assert 'fresh-ratio-connector fresh-ratio-connector-left' in source
    assert 'fresh-ratio-connector fresh-ratio-connector-right' in source
    assert "<span>'+t('Fresh')+': <strong>0%</strong></span>" in source
    assert "<span>'+t('Day-1')+': <strong>0%</strong></span>" in source
    assert 'label:{show:false},labelLine:{show:false}' in source
    assert 'document.getElementById("inv-fresh-ratio-left")' in source
    assert 'document.getElementById("inv-fresh-ratio-right")' in source


def test_freshness_ratio_percentages_handle_zero_total():
    source = INDEX_HTML.read_text(encoding="utf-8")

    assert "var freshPct=ratioTotal>0?d.fresh_total/ratioTotal*100:0;" in source
    assert "var day1Pct=ratioTotal>0?d.day1_total/ratioTotal*100:0;" in source
