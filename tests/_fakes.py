"""共享测试 fakes（ADR-0006/0007/0008 集成测试用）。

集中 test_pipeline.py 与 test_study_integration.py 重复的：
THR / _sit_upright_kps / _bend_kps / FakeCamera / FakePose / FakeClock / FakeDotii。

符合 project_memory 约定「Test fakes should be centralized in shared test utilities」。
"""
from __future__ import annotations

import numpy as np

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
    "reminder": {
        "sustained_seconds": 30,
        "min_interval_seconds": 60,
        "track_phase_disabled": True,
    },
    "absence": {"no_skeleton_seconds": 30},
}


def _sit_upright_kps():
    kp = [None] * 33
    kp[NOSE] = (0.5, 0.30, 0.10)
    kp[LEFT_EAR] = (0.47, 0.32, 0.05)
    kp[RIGHT_EAR] = (0.53, 0.32, 0.05)
    kp[LEFT_SHOULDER] = (0.45, 0.40, 0.00)
    kp[RIGHT_SHOULDER] = (0.55, 0.40, 0.00)
    kp[LEFT_HIP] = (0.46, 0.60, 0.00)
    kp[RIGHT_HIP] = (0.54, 0.60, 0.00)
    kp[LEFT_KNEE] = (0.47, 0.78, 0.10)
    kp[RIGHT_KNEE] = (0.53, 0.78, 0.10)
    return kp


def _bend_kps():
    kp = _sit_upright_kps()
    kp[LEFT_SHOULDER] = (0.50, 0.50, 0.15)
    kp[RIGHT_SHOULDER] = (0.50, 0.50, 0.15)
    kp[LEFT_KNEE] = (0.47, 0.75, 0.10)
    kp[RIGHT_KNEE] = (0.53, 0.75, 0.10)
    return kp


class FakeCamera:
    def __init__(self, n_frames):
        self._color = np.zeros((4, 4, 3), dtype=np.uint8)
        self._n = n_frames

    def grab(self, timeout_ms=2000):
        if self._n <= 0:
            return None, None
        self._n -= 1
        return self._color, None  # depth=None → 不用深度


class FakePose:
    def __init__(self, kps):
        self._kps = list(kps)

    def infer(self, color):
        if not self._kps:
            return [None] * 33
        return self._kps.pop(0)


class FakeClock:
    def __init__(self, step=1.0):
        self.t = 0.0
        self.step = step

    def __call__(self):
        v = self.t
        self.t += self.step
        return v


class FakeDotii:
    def __init__(self):
        self.calls = []

    def __call__(self, expression):
        self.calls.append(expression)
