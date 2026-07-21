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


def test_attendance_views_show_complete_status_and_period_metrics():
    source = FRONTEND.read_text(encoding="utf-8")
    start = source.index("function attendanceStatusMeta")
    end = source.index("async function loadMaterials", start)
    block = source[start:end]

    assert "early_leave" in block
    assert "late_and_early_leave" in block
    assert "t('Early Leave')" in block
    assert "t('Late & Early Leave')" in block
    assert "t('Attendance Rate')" in block
    assert "t('On-Time Rate')" in block
    assert "t('Shift Completion')" in block
    assert "attendance_display" in block
    assert "punctuality_rate" in block
    assert "shift_completion_rate" in block
    assert "Att%" not in block


def test_manager_attendance_punch_uses_an_in_page_pin_dialog():
    source = FRONTEND.read_text(encoding="utf-8")
    start = source.index("function punchEmployee")
    end = source.index("async function loadKPIRanking", start)
    block = source[start:end]

    assert 'id="attendance-pin-modal"' in source
    assert 'id="attendance-manager-pin"' in source
    assert 'id="attendance-manager-pin-result"' in source
    assert "function submitManagerAttendancePunch" in block
    assert "encodeURIComponent(empId)" in block
    assert "encodeURIComponent(pin)" in block
    assert "_apiCache={};" in block
    assert "prompt(" not in block
    assert "alert(" not in block


def test_manager_can_correct_current_or_historical_attendance_with_a_reason():
    source = FRONTEND.read_text(encoding="utf-8")
    start = source.index("function attendanceStatusMeta")
    end = source.index("async function loadMaterials", start)
    block = source[start:end]

    assert 'id="attendance-correction-modal"' in source
    assert 'id="attendance-correction-in"' in source
    assert 'id="attendance-correction-out"' in source
    assert 'id="attendance-correction-reason"' in source
    assert "function openAttendanceCorrection" in source
    assert "function submitAttendanceCorrection" in source
    assert "'/s3/attendance/correct'" in source
    assert "t('Correct')" in block
    assert "correction_reason" in block


def test_schedule_and_attendance_share_the_selected_operational_date():
    source = FRONTEND.read_text(encoding="utf-8")
    schedule_start = source.index("function renderSchedule")
    schedule_end = source.index("async function generateSchedule", schedule_start)
    schedule_block = source[schedule_start:schedule_end]
    attendance_start = source.index("function renderAttendance")
    attendance_end = source.index("function formatInventoryHistoryTime", attendance_start)
    attendance_block = source[attendance_start:attendance_end]

    assert "function getSelectedOperationalDate" in source
    assert "function setSelectedOperationalDate" in source
    assert "sessionStorage.getItem('bakery_operational_date')" in source
    assert "onScheduleDateChange(this.value)" in schedule_block
    assert "getSelectedOperationalDate()" in schedule_block
    assert "onAttendanceDateChange(this.value)" in attendance_block
    assert "getSelectedOperationalDate()" in attendance_block
    assert "showTodayAttendance()" in attendance_block


def test_schedule_generation_reloads_persisted_rows_without_stale_get_cache():
    source = FRONTEND.read_text(encoding="utf-8")
    start = source.index("async function generateSchedule")
    end = source.index("function daysToSunday", start)
    block = source[start:end]

    assert "clearApiCacheByPath('/s3/schedule')" in block
    assert "clearApiCacheByPath('/s3/kpi')" in block
    assert "await loadSchedule();" in block
    assert "await loadKPI();" in block
