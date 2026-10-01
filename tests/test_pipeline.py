"""pipeline 集成测试（用 fake camera/pose/clock/dotii）。

ADR-0006/0007 端到端：分类→落盘→提醒→日报。
"""
from __future__ import annotations

import pytest

from src.pipeline import run_session
from src.storage import Storage
from tests._fakes import (
    THR,
    FakeCamera,
    FakeClock,
    FakeDotii,
    FakePose,
    _bend_kps,
    _sit_upright_kps,
)


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


def test_pipeline_applies_smoothing_to_labels(tmp_path):
    """ADR-0005：单帧分类后用 1 秒窗移动众数平滑。

    单帧 BEND 夹在 SIT_UPRIGHT 间应被平滑成 SIT_UPRIGHT，
    daily_report.bad_ratio 反映平滑后结果（0 不良）。
    用 FakeClock 步长 1/30 秒模拟真实 fps=30 → 1 秒窗覆盖 ±15 帧。
    """
    n = 11
    kps = ([_sit_upright_kps() for _ in range(5)]
           + [_bend_kps()]
           + [_sit_upright_kps() for _ in range(5)])
    cam = FakeCamera(n)
    pose = FakePose(kps)
    dotii = FakeDotii()
    rep = run_session(cam, pose, _storage(tmp_path), THR,
                      participant_id="p01", phase="intervention",
                      dotii_show=dotii, clock=FakeClock(1/30), max_frames=n)
    # 单帧 BEND 在 1 秒窗众数下被覆盖 → 0 不良
    assert rep["bad_ratio"] == 0.0
    assert rep["bad_frames"] == 0


# ---------- Alerter 接入（ADR-0007 PI 报警机制）----------

def test_pipeline_alerts_on_camera_reconnect_exhausted(tmp_path):
    """ADR-0007：相机重连耗尽（CameraReconnectError）时调 alerter。

    相机不可恢复 → log_exception + alert("camera_reconnect_exhausted")，
    提前退出主循环并生成空日报（避免吞异常）。
    """
    from src.exceptions import CameraReconnectError
    from tests.test_alerter import FakeAlerter

    class DeadCamera:
        def grab(self, timeout_ms=2000):
            raise CameraReconnectError("相机重连 3 次仍失败")

    alerter = FakeAlerter()
    rep = run_session(DeadCamera(), FakePose([]), _storage(tmp_path), THR,
                      participant_id="p01", phase="intervention",
                      clock=FakeClock(1.0), max_frames=5, alerter=alerter)
    # 相机重连耗尽 → alert 一次，kind 为 camera_reconnect_exhausted
    kinds = [k for k, _ in alerter.alerts]
    assert "camera_reconnect_exhausted" in kinds
    # 相机死 → 主循环立即退出，日报无帧
    assert rep["total_frames"] == 0


def test_pipeline_alerts_on_dotii_failure_threshold(tmp_path):
    """ADR-0007：Dotii 连续失败 ≥ dotii_failure_alert_threshold 时调 alerter。

    35 帧 bend（clock step=1.0）：t=30 触发 should_remind，dotii 失败 →
    last_reminder 仍为 None → 后续帧每帧都重试 dotii → 5 次连续失败时报警。
    """
    from src.exceptions import DotiiAPIError
    from tests.test_alerter import FakeAlerter

    def failing_dotii(expression):
        raise DotiiAPIError("Dotii HTTP 失联")

    thr = dict(THR)
    thr["integrity"] = {"dotii_failure_alert_threshold": 5}

    n = 35
    cam = FakeCamera(n)
    pose = FakePose([_bend_kps() for _ in range(n)])
    alerter = FakeAlerter()
    rep = run_session(cam, pose, _storage(tmp_path), thr,
                      participant_id="p01", phase="intervention",
                      dotii_show=failing_dotii, clock=FakeClock(1.0), max_frames=n,
                      alerter=alerter)
    # Dotii 连续失败 ≥ 5 → alert kind=dotii_failure_threshold
    kinds = [k for k, _ in alerter.alerts]
    assert "dotii_failure_threshold" in kinds
    # 失败的 dotii 调用不增加 reminders
    assert rep["reminders"] == 0
