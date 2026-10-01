"""depth_distance 的深度采样与距离计算测试（ADR-0002）。"""
from __future__ import annotations

import numpy as np
import pytest

from src.depth_distance import head_desk_distance, sample_depth


def test_head_desk_distance_none_inputs():
    assert head_desk_distance(None, 0.5) is None
    assert head_desk_distance(0.5, None) is None


def test_head_desk_distance_value():
    assert head_desk_distance(0.60, 0.80) == pytest.approx(0.20)


def test_sample_depth_returns_meters():
    frame = np.full((4, 4), 600, dtype=np.uint16)  # 600 mm
    assert sample_depth(frame, 0.5, 0.5) == pytest.approx(0.60)


def test_sample_depth_zero_is_none():
    frame = np.zeros((4, 4), dtype=np.uint16)
    assert sample_depth(frame, 0.5, 0.5) is None


def test_sample_depth_clamps_coords():
    frame = np.full((4, 4), 500, dtype=np.uint16)
    assert sample_depth(frame, -0.5, 1.5) == pytest.approx(0.50)


def test_sample_depth_none_frame():
    assert sample_depth(None, 0.5, 0.5) is None
