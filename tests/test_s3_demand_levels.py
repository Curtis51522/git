from api.module3_scheduling import _classify_demand_level


def test_relative_demand_level_uses_weekday_baseline():
    assert _classify_demand_level(121, 100) == "high"
    assert _classify_demand_level(100, 100) == "normal"
    assert _classify_demand_level(84, 100) == "low"


def test_relative_demand_level_falls_back_without_baseline():
    assert _classify_demand_level(450, None) == "high"
    assert _classify_demand_level(250, None) == "normal"
    assert _classify_demand_level(120, None) == "low"
