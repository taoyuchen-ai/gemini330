"""statistics 的平滑与占比测试（ADR-0006）。"""
from __future__ import annotations

import pytest

from src.posture_classify import PostureLabel as P
from src.statistics import compute_missing_ratio, compute_ratio, smooth


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


def test_smooth_protects_missing_frames():
    """ADR-0006：MISSING 帧不被众数覆盖（缺失不计入分母，必须独立保留）。"""
    labels = [P.SIT_UPRIGHT] * 5 + [P.MISSING] + [P.SIT_UPRIGHT] * 5
    result = smooth(_entries(labels))
    assert result[5] == P.MISSING
    assert all(l == P.SIT_UPRIGHT for i, l in enumerate(result) if i != 5)


def test_smooth_protects_absence_frames():
    """ADR-0007：ABSENCE 帧不被众数覆盖（离座状态独立标识，不计入分母）。"""
    labels = [P.BEND] * 5 + [P.ABSENCE] + [P.BEND] * 5
    result = smooth(_entries(labels))
    assert result[5] == P.ABSENCE
    assert all(l == P.BEND for i, l in enumerate(result) if i != 5)


def test_smooth_excludes_protected_from_window():
    """有效姿态帧的窗口排除 MISSING/ABSENCE，避免污染众数。"""
    # 中间帧 SIT_UPRIGHT，左右各 2 个 MISSING
    # 若 MISSING 进窗口：4 MISSING vs 1 SIT_UPRIGHT → 众数 MISSING（错）
    # 若 MISSING 排除：窗口只 1 SIT_UPRIGHT → 众数 SIT_UPRIGHT（对）
    labels = [P.MISSING, P.MISSING, P.SIT_UPRIGHT, P.MISSING, P.MISSING]
    result = smooth(_entries(labels))
    assert result[2] == P.SIT_UPRIGHT


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


# ---------- compute_missing_ratio（ADR-0006/0007）----------

def test_missing_ratio_empty():
    """空列表 → 0.0。"""
    assert compute_missing_ratio([]) == 0.0


def test_missing_ratio_no_missing():
    """无 MISSING/ABSENCE → 0.0。"""
    assert compute_missing_ratio([P.SIT_UPRIGHT, P.BEND, P.DESK_LYING]) == 0.0


def test_missing_ratio_all_missing():
    """全 MISSING/ABSENCE → 1.0。"""
    assert compute_missing_ratio([P.MISSING, P.ABSENCE]) == 1.0


def test_missing_ratio_mixed():
    """混合：2/4 = 0.5。"""
    labels = [P.DESK_LYING, P.SIT_UPRIGHT, P.MISSING, P.ABSENCE]
    assert compute_missing_ratio(labels) == pytest.approx(0.5)
