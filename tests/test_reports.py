"""reports 的个人日报与班级日报测试。"""
from __future__ import annotations

import pytest

from src.posture_classify import PostureLabel as P
from src.reports import class_report, daily_report


def test_daily_report_fields_and_ratio():
    labels = [P.DESK_LYING, P.SIT_UPRIGHT, P.BEND, P.ABSENCE, P.MISSING, P.SIT_UPRIGHT]
    rep = daily_report("p01", labels, phase="intervention", reminders=2,
                       session_start=0.0, session_end=60.0)
    assert rep["participant_id"] == "p01"
    assert rep["phase"] == "intervention"
    assert rep["total_frames"] == 6
    assert rep["valid_frames"] == 4
    assert rep["bad_frames"] == 2
    assert rep["bad_ratio"] == pytest.approx(0.5)
    assert rep["posture_counts"]["desk_lying"] == 1
    assert rep["posture_counts"]["sit_upright"] == 2
    assert rep["posture_counts"]["absence"] == 1
    assert rep["posture_counts"]["missing"] == 1
    assert rep["reminders"] == 2
    assert rep["session_start"] == 0.0
    assert rep["session_end"] == 60.0


def test_daily_report_empty():
    rep = daily_report("p02", [])
    assert rep["total_frames"] == 0
    assert rep["valid_frames"] == 0
    assert rep["bad_frames"] == 0
    assert rep["bad_ratio"] == 0.0


def test_class_report_aggregates():
    r1 = daily_report("p01", [P.BEND, P.SIT_UPRIGHT])    # valid 2, bad 1
    r2 = daily_report("p02", [P.DESK_LYING, P.ABSENCE])  # valid 1, bad 1
    cls = class_report([r1, r2])
    assert cls["n_participants"] == 2
    assert cls["participants"] == ["p01", "p02"]
    assert cls["total_valid_frames"] == 3
    assert cls["total_bad_frames"] == 2
    assert cls["class_bad_ratio"] == pytest.approx(2 / 3)


def test_class_report_empty():
    cls = class_report([])
    assert cls["n_participants"] == 0
    assert cls["class_bad_ratio"] == 0.0
