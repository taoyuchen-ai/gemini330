"""pipeline 集成测试（用 fake camera/pose/clock/dotii）。

ADR-0006/0007 端到端：分类→落盘→提醒→日报。
"""
from __future__ import annotations

import numpy as np
import pytest

from src.pipeline import run_session
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
from src.storage import Storage


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


def _storage(tmp_path):
    return Storage(str(tmp_path / "p.db"), flush_interval_seconds=9999)


def test_pipeline_sit_upright_no_reminder(tmp_path):
    n = 5
    cam = FakeCamera(n)
    pose = FakePose([_sit_upright_kps() for _ in range(n)])
    dotii = FakeDotii()
    rep = run_session(cam, pose, _storage(tmp_path), THR,
                      participant_id="p01", phase="intervention",
                      dotii_show=dotii, clock=FakeClock(1.0), max_frames=n)
    assert rep["bad_ratio"] == 0.0
    assert rep["reminders"] == 0
    assert dotii.calls == []


def test_pipeline_triggers_fail_after_30s(tmp_path):
    n = 35
    cam = FakeCamera(n)
    pose = FakePose([_bend_kps() for _ in range(n)])
    dotii = FakeDotii()
    rep = run_session(cam, pose, _storage(tmp_path), THR,
                      participant_id="p01", phase="intervention",
                      dotii_show=dotii, clock=FakeClock(1.0), max_frames=n)
    assert rep["reminders"] == 1
    assert dotii.calls.count("fail") == 1
    assert rep["bad_ratio"] == 1.0


def test_pipeline_no_double_reminder_within_60s(tmp_path):
    n = 70
    cam = FakeCamera(n)
    pose = FakePose([_bend_kps() for _ in range(n)])
    dotii = FakeDotii()
    rep = run_session(cam, pose, _storage(tmp_path), THR,
                      participant_id="p01", phase="intervention",
                      dotii_show=dotii, clock=FakeClock(1.0), max_frames=n)
    assert rep["reminders"] == 1  # 30s 触发一次，60s 间隔内不再触发


def test_pipeline_triggers_twice_after_90s(tmp_path):
    n = 95
    cam = FakeCamera(n)
    pose = FakePose([_bend_kps() for _ in range(n)])
    dotii = FakeDotii()
    rep = run_session(cam, pose, _storage(tmp_path), THR,
                      participant_id="p01", phase="intervention",
                      dotii_show=dotii, clock=FakeClock(1.0), max_frames=n)
    assert rep["reminders"] == 2  # 30s + 90s


def test_pipeline_recovery_sends_idle(tmp_path):
    n = 35
    # 前 5 帧 bad，后 30 帧 sit_upright
    kps = [_bend_kps() for _ in range(5)] + [_sit_upright_kps() for _ in range(30)]
    cam = FakeCamera(n)
    pose = FakePose(kps)
    dotii = FakeDotii()
    rep = run_session(cam, pose, _storage(tmp_path), THR,
                      participant_id="p01", phase="intervention",
                      dotii_show=dotii, clock=FakeClock(1.0), max_frames=n)
    assert "idle" in dotii.calls
    assert rep["bad_ratio"] < 1.0


def test_pipeline_absence_after_30s_no_skeleton(tmp_path):
    n = 35
    cam = FakeCamera(n)
    pose = FakePose([[None] * 33 for _ in range(n)])  # 全无骨架
    dotii = FakeDotii()
    rep = run_session(cam, pose, _storage(tmp_path), THR,
                      participant_id="p01", phase="intervention",
                      dotii_show=dotii, clock=FakeClock(1.0), max_frames=n)
    # 前 30 帧 MISSING，之后 ABSENCE
    assert rep["posture_counts"].get("absence", 0) > 0
    assert rep["posture_counts"].get("missing", 0) > 0
    assert rep["valid_frames"] == 0  # absence+missing 全排除


def test_pipeline_no_reminder_in_followup(tmp_path):
    n = 40
    cam = FakeCamera(n)
    pose = FakePose([_bend_kps() for _ in range(n)])
    dotii = FakeDotii()
    rep = run_session(cam, pose, _storage(tmp_path), THR,
                      participant_id="p01", phase="followup",
                      dotii_show=dotii, clock=FakeClock(1.0), max_frames=n)
    assert rep["reminders"] == 0
    assert dotii.calls == []


def test_pipeline_records_correction_time_after_reminder_recovery(tmp_path):
    """B2：提醒触发后恢复 good → 记录从提醒到纠正的耗时。

    前 35 帧 bend：t=30 触发提醒（last_reminder=30）。
    第 36 帧 sit_upright（t=35）恢复 good → correction = 35 - 30 = 5。
    """
    n = 45
    kps = [_bend_kps() for _ in range(35)] + [_sit_upright_kps() for _ in range(10)]
    cam = FakeCamera(n)
    pose = FakePose(kps)
    dotii = FakeDotii()
    rep = run_session(cam, pose, _storage(tmp_path), THR,
                      participant_id="p01", phase="intervention",
                      dotii_show=dotii, clock=FakeClock(1.0), max_frames=n)
    assert rep["reminders"] == 1
    assert dotii.calls.count("fail") == 1
    assert len(rep["correction_times"]) == 1
    assert rep["correction_times"][0] == pytest.approx(5.0)


def test_pipeline_no_correction_time_when_bad_recovers_without_reminder(tmp_path):
    """B2：bad 持续 <30s 恢复（未触发提醒）→ 不记录纠正耗时。"""
    n = 10
    # 前 5 帧 bend（<30s，不触发提醒），后 5 帧 sit_upright
    kps = [_bend_kps() for _ in range(5)] + [_sit_upright_kps() for _ in range(5)]
    cam = FakeCamera(n)
    pose = FakePose(kps)
    dotii = FakeDotii()
    rep = run_session(cam, pose, _storage(tmp_path), THR,
                      participant_id="p01", phase="intervention",
                      dotii_show=dotii, clock=FakeClock(1.0), max_frames=n)
    assert rep["reminders"] == 0
    assert rep["correction_times"] == []
