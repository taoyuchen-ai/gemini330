"""posture_classify.classify 的分支测试（ADR-0005）。"""
from __future__ import annotations

import math

import pytest

from src.posture_classify import (
    LEFT_EAR,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    NOSE,
    RIGHT_EAR,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    PostureLabel,
    classify,
)


THR = {
    "rgb": {
        "trunk_thigh_angle_bend_threshold": 100,
        "trunk_lean_back_threshold": 20,
        "head_yaw_turn_threshold": 25,
        "trunk_lateral_lean_threshold": 15,
    },
    "depth": {
        "forward_read_min": 0.20,
        "forward_read_max": 0.45,
        "desk_lying_threshold": 0.20,
    },
}


def _kp(overrides=None):
    """构造默认坐正骨架（33 元素 list）。"""
    kp = [None] * 33
    defaults = {
        NOSE: (0.5, 0.30, 0.10),
        LEFT_EAR: (0.47, 0.32, 0.05),
        RIGHT_EAR: (0.53, 0.32, 0.05),
        LEFT_SHOULDER: (0.45, 0.40, 0.00),
        RIGHT_SHOULDER: (0.55, 0.40, 0.00),
        LEFT_HIP: (0.46, 0.60, 0.00),
        RIGHT_HIP: (0.54, 0.60, 0.00),
        LEFT_KNEE: (0.47, 0.78, 0.10),
        RIGHT_KNEE: (0.53, 0.78, 0.10),
    }
    for i, v in defaults.items():
        kp[i] = v
    if overrides:
        for i, v in overrides.items():
            kp[i] = v
    return kp


def test_missing_when_keypoints_none():
    assert classify(None, 0.35, THR) == PostureLabel.MISSING


def test_missing_when_core_keypoint_absent():
    kp = _kp({NOSE: None})
    assert classify(kp, 0.35, THR) == PostureLabel.MISSING


def test_sit_upright_default():
    assert classify(_kp(), 0.35, THR) == PostureLabel.SIT_UPRIGHT


def test_desk_lying_when_head_too_close():
    assert classify(_kp(), 0.15, THR) == PostureLabel.DESK_LYING


def test_bend_when_trunk_thigh_angle_small():
    overrides = {
        LEFT_SHOULDER: (0.50, 0.50, 0.15),
        RIGHT_SHOULDER: (0.50, 0.50, 0.15),
        LEFT_KNEE: (0.47, 0.75, 0.10),
        RIGHT_KNEE: (0.53, 0.75, 0.10),
    }
    assert classify(_kp(overrides), 0.50, THR) == PostureLabel.BEND


def test_lean_back_when_trunk_backward():
    overrides = {
        LEFT_SHOULDER: (0.45, 0.40, -0.15),
        RIGHT_SHOULDER: (0.55, 0.40, -0.15),
    }
    assert classify(_kp(overrides), 0.35, THR) == PostureLabel.LEAN_BACK


def test_turn_side_via_lateral_lean():
    overrides = {
        LEFT_SHOULDER: (0.65, 0.40, 0.00),
        RIGHT_SHOULDER: (0.65, 0.40, 0.00),
    }
    assert classify(_kp(overrides), 0.35, THR) == PostureLabel.TURN_SIDE


def test_turn_side_via_head_yaw():
    overrides = {NOSE: (0.60, 0.30, 0.10)}
    assert classify(_kp(overrides), 0.35, THR) == PostureLabel.TURN_SIDE


def test_forward_read_when_leaning_and_distance_in_range():
    overrides = {
        LEFT_SHOULDER: (0.45, 0.45, 0.15),
        RIGHT_SHOULDER: (0.55, 0.45, 0.15),
    }
    assert classify(_kp(overrides), 0.30, THR) == PostureLabel.FORWARD_READ
