"""statistics 的平滑与占比测试（ADR-0006）。"""
from __future__ import annotations

import pytest

from src.posture_classify import PostureLabel as P
from src.statistics import compute_ratio, smooth


def _entries(labels, fps=30):
    return [(i / fps, lab) for i, lab in enumerate(labels)]


def test_smooth_empty():
    assert smooth([]) == []


def test_smooth_stable_all_sit_upright():
    entries = _entries([P.SIT_UPRIGHT] * 10)
    assert smooth(entries) == [P.SIT_UPRIGHT] * 10


def test_smooth_removes_single_frame_jitter():
    labels = [P.SIT_UPRIGHT] * 5 + [P.BEND] + [P.SIT_UPRIGHT] * 5
    result = smooth(_entries(labels))
    assert result[5] == P.SIT_UPRIGHT  # 单帧 bend 被众数覆盖
    assert all(l == P.SIT_UPRIGHT for l in result)


def test_smooth_preserves_sustained_bad():
    labels = [P.BEND] * 35  # 持续 1 秒以上
    result = smooth(_entries(labels))
    assert all(l == P.BEND for l in result)


def test_ratio_empty():
    assert compute_ratio([]) == 0.0


def test_ratio_excludes_absence_and_missing():
    labels = [P.DESK_LYING, P.SIT_UPRIGHT, P.BEND, P.ABSENCE, P.MISSING]
    # 有效 3 帧，不良 2 帧 → 2/3
    assert compute_ratio(labels) == pytest.approx(2 / 3)


def test_ratio_all_absent_or_missing():
    assert compute_ratio([P.ABSENCE, P.MISSING]) == 0.0


def test_ratio_all_good():
    assert compute_ratio([P.SIT_UPRIGHT, P.FORWARD_READ]) == 0.0


def test_ratio_all_bad():
    assert compute_ratio([P.DESK_LYING, P.BEND]) == 1.0
