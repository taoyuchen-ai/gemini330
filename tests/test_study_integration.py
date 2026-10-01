"""研究主分析集成测试（ADR-0008 端到端）。

闭环：daily_reports → phase_columns → analyze_three_phase
以及 run_study 多被试×3 阶段编排。
"""
from __future__ import annotations

import pytest

from src.posture_classify import PostureLabel as P
from src.reports import daily_report, phase_columns
from src.stats_analysis import analyze_three_phase


# ---------- phase_columns ----------

def _rep(pid, phase, bad_ratio, valid=100):
    """构造一个 daily_report dict 用于 phase_columns 测试。"""
    return {
        "participant_id": pid,
        "phase": phase,
        "bad_ratio": bad_ratio,
        "valid_frames": valid,
        "bad_frames": int(bad_ratio * valid),
    }


def test_phase_columns_empty():
    pids, groups = phase_columns([])
    assert pids == []
    assert groups == [[], [], []]


def test_phase_columns_single_participant_three_phases():
    reports = [
        _rep("p01", "baseline", 0.50),
        _rep("p01", "intervention", 0.30),
        _rep("p01", "followup", 0.35),
    ]
    pids, groups = phase_columns(reports)
    assert pids == ["p01"]
    assert len(groups) == 3
    assert groups[0] == [pytest.approx(0.50)]
    assert groups[1] == [pytest.approx(0.30)]
    assert groups[2] == [pytest.approx(0.35)]


def test_phase_columns_aggregates_multiple_days_per_phase():
    # 同一被试同一阶段多日 → 取均值
    reports = [
        _rep("p01", "baseline", 0.40),
        _rep("p01", "baseline", 0.60),  # 均值 0.50
        _rep("p01", "intervention", 0.20),
        _rep("p01", "intervention", 0.40),  # 均值 0.30
        _rep("p01", "followup", 0.30),
        _rep("p01", "followup", 0.40),  # 均值 0.35
    ]
    pids, groups = phase_columns(reports)
    assert pids == ["p01"]
    assert groups[0] == [pytest.approx(0.50)]
    assert groups[1] == [pytest.approx(0.30)]
    assert groups[2] == [pytest.approx(0.35)]


def test_phase_columns_multiple_participants_aligned():
    reports = [
        _rep("p01", "baseline", 0.50),
        _rep("p02", "baseline", 0.40),
        _rep("p01", "intervention", 0.30),
        _rep("p02", "intervention", 0.20),
        _rep("p01", "followup", 0.35),
        _rep("p02", "followup", 0.25),
    ]
    pids, groups = phase_columns(reports)
    assert set(pids) == {"p01", "p02"}
    assert len(groups[0]) == 2
    assert len(groups[1]) == 2
    assert len(groups[2]) == 2
    # 列对齐：同 index = 同被试
    p01_idx = pids.index("p01")
    p02_idx = pids.index("p02")
    assert groups[0][p01_idx] == pytest.approx(0.50)
    assert groups[0][p02_idx] == pytest.approx(0.40)
    assert groups[1][p01_idx] == pytest.approx(0.30)
    assert groups[2][p02_idx] == pytest.approx(0.25)


def test_phase_columns_drops_participant_missing_in_any_phase():
    # p02 缺 followup → 整体剔除
    reports = [
        _rep("p01", "baseline", 0.50),
        _rep("p02", "baseline", 0.40),
        _rep("p01", "intervention", 0.30),
        _rep("p02", "intervention", 0.20),
        _rep("p01", "followup", 0.35),
        # p02 followup 缺失
    ]
    pids, groups = phase_columns(reports)
    assert pids == ["p01"]
    assert all(len(g) == 1 for g in groups)


def test_phase_columns_phase_order_fixed_regardless_of_input():
    # 故意打乱输入顺序，输出阶段顺序仍为 baseline/intervention/followup
    reports = [
        _rep("p01", "followup", 0.35),
        _rep("p01", "intervention", 0.30),
        _rep("p01", "baseline", 0.50),
    ]
    pids, groups = phase_columns(reports)
    assert groups[0] == [pytest.approx(0.50)]  # baseline
    assert groups[1] == [pytest.approx(0.30)]  # intervention
    assert groups[2] == [pytest.approx(0.35)]  # followup


def test_phase_columns_unknown_phase_raises():
    reports = [_rep("p01", "unknown_phase", 0.50)]
    with pytest.raises(ValueError):
        phase_columns(reports)


# ---------- run_study 端到端 ----------

def test_run_study_end_to_end_two_participants_three_phases(tmp_path):
    """2 被试 × 3 阶段端到端：fake camera/pose → run_study → analyze_three_phase 结构。"""
    from src.pipeline import run_study
    from src.storage import Storage

    # 用 test_pipeline 的 fakes（最小复制避免跨文件耦合）
    import numpy as np
    from src.posture_classify import (
        LEFT_EAR, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER,
        NOSE, RIGHT_EAR, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER,
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
            return self._color, None

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

    # 研究设计：3 被试 × 3 阶段 × 5 帧/会话
    # baseline 全 bend（高 bad_ratio），intervention 全 sit_upright（0），followup 一半 bend
    sessions = []
    expected_pids = ["p01", "p02", "p03"]
    for pid in expected_pids:
        sessions.append({"participant_id": pid, "phase": "baseline",
                         "kps_seq": [_bend_kps() for _ in range(5)]})
        sessions.append({"participant_id": pid, "phase": "intervention",
                         "kps_seq": [_sit_upright_kps() for _ in range(5)]})
        sessions.append({"participant_id": pid, "phase": "followup",
                         "kps_seq": [_bend_kps() for _ in range(2)] +
                                    [_sit_upright_kps() for _ in range(3)]})

    storage = Storage(str(tmp_path / "study.db"), flush_interval_seconds=9999)

    # session_runner：单 session 配置 → daily_report（封装 run_session + fakes）
    def session_runner(sess):
        n = len(sess["kps_seq"])
        cam = FakeCamera(n)
        pose = FakePose(sess["kps_seq"])
        # clock 重置避免跨 session 累积
        clk = FakeClock(1.0)
        from src.pipeline import run_session
        return run_session(cam, pose, storage, THR,
                          participant_id=sess["participant_id"],
                          phase=sess["phase"],
                          clock=clk, max_frames=n)

    result = run_study(
        sessions_config=sessions,
        session_runner=session_runner,
    )

    # 验证结构
    assert "daily_reports" in result
    assert "analysis" in result
    assert "n_subjects" in result
    assert result["n_subjects"] == 3
    assert len(result["daily_reports"]) == 9  # 3×3

    analysis = result["analysis"]
    assert set(analysis.keys()) == {
        "friedman", "post_hoc", "bonferroni", "eta_squared",
        "kendalls_w", "n_subjects", "k_phases",
    }
    # baseline 全 bend → bad_ratio ≈ 1.0；intervention 全 sit_upright → ≈ 0.0
    # 3 阶段差异显著（小样本下 Friedman χ² > 0）
    chi2, p = analysis["friedman"]
    assert chi2 > 0


def test_run_study_returns_empty_when_no_sessions(tmp_path):
    from src.pipeline import run_study

    result = run_study(
        sessions_config=[],
        session_runner=lambda sess: {},
    )
    assert result["daily_reports"] == []
    assert result["n_subjects"] == 0
    assert result["analysis"] is None  # 无数据时不调用 stats
