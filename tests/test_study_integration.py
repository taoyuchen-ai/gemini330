"""研究主分析集成测试（ADR-0008 端到端）。

闭环：daily_reports → phase_columns → analyze_three_phase
以及 run_study 多被试×3 阶段编排。
"""
from __future__ import annotations

import pytest

from src.posture_classify import PostureLabel as P
from src.reports import daily_report, phase_columns
from src.stats_analysis import analyze_three_phase
from src.phases import ThreePhase


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
    pids, phases = phase_columns([])
    assert pids == []
    assert phases == ThreePhase([], [], [])


def test_phase_columns_single_participant_three_phases():
    reports = [
        _rep("p01", "baseline", 0.50),
        _rep("p01", "intervention", 0.30),
        _rep("p01", "followup", 0.35),
    ]
    pids, phases = phase_columns(reports)
    assert pids == ["p01"]
    assert phases.baseline == [pytest.approx(0.50)]
    assert phases.intervention == [pytest.approx(0.30)]
    assert phases.followup == [pytest.approx(0.35)]


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
    pids, phases = phase_columns(reports)
    assert pids == ["p01"]
    assert phases.baseline == [pytest.approx(0.50)]
    assert phases.intervention == [pytest.approx(0.30)]
    assert phases.followup == [pytest.approx(0.35)]


def test_phase_columns_multiple_participants_aligned():
    reports = [
        _rep("p01", "baseline", 0.50),
        _rep("p02", "baseline", 0.40),
        _rep("p01", "intervention", 0.30),
        _rep("p02", "intervention", 0.20),
        _rep("p01", "followup", 0.35),
        _rep("p02", "followup", 0.25),
    ]
    pids, phases = phase_columns(reports)
    assert set(pids) == {"p01", "p02"}
    assert len(phases.baseline) == 2
    assert len(phases.intervention) == 2
    assert len(phases.followup) == 2
    # 列对齐：同 index = 同被试
    p01_idx = pids.index("p01")
    p02_idx = pids.index("p02")
    assert phases.baseline[p01_idx] == pytest.approx(0.50)
    assert phases.baseline[p02_idx] == pytest.approx(0.40)
    assert phases.intervention[p01_idx] == pytest.approx(0.30)
    assert phases.followup[p02_idx] == pytest.approx(0.25)


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
    pids, phases = phase_columns(reports)
    assert pids == ["p01"]
    assert all(len(col) == 1 for col in phases)


def test_phase_columns_phase_order_fixed_regardless_of_input():
    # 故意打乱输入顺序，输出阶段顺序仍为 baseline/intervention/followup
    reports = [
        _rep("p01", "followup", 0.35),
        _rep("p01", "intervention", 0.30),
        _rep("p01", "baseline", 0.50),
    ]
    pids, phases = phase_columns(reports)
    assert phases.baseline == [pytest.approx(0.50)]  # baseline
    assert phases.intervention == [pytest.approx(0.30)]  # intervention
    assert phases.followup == [pytest.approx(0.35)]  # followup


def test_phase_columns_unknown_phase_raises():
    reports = [_rep("p01", "unknown_phase", 0.50)]
    with pytest.raises(ValueError):
        phase_columns(reports)


# ---------- run_study 端到端 ----------

def test_run_study_end_to_end_two_participants_three_phases(tmp_path):
    """2 被试 × 3 阶段端到端：fake camera/pose → run_study → analyze_three_phase 结构。"""
    from src.pipeline import run_session, run_study
    from src.storage import Storage
    from tests._fakes import (
        THR, FakeCamera, FakeClock, FakePose, _bend_kps, _sit_upright_kps,
    )

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


def _fake_report(pid, phase, bad_ratio, correction_times=None):
    """构造完整 daily_report dict 用于 run_study 探索性分析测试。"""
    return {
        "participant_id": pid,
        "phase": phase,
        "session_start": 0.0,
        "session_end": 60.0,
        "total_frames": 100,
        "valid_frames": 100,
        "bad_frames": int(bad_ratio * 100),
        "bad_ratio": bad_ratio,
        "posture_counts": {},
        "reminders": 0,
        "correction_times": list(correction_times) if correction_times else [],
    }


def test_run_study_with_screening_returns_exploratory_spearman(tmp_path):
    """B3：screening CSV + 3 被试 → spearman_vision/questionnaire 非 None；
    correction_stats 聚合所有 correction_times。"""
    from src.pipeline import run_study

    pids = ["p01", "p02", "p03"]
    # baseline bad_ratio 与 vision 完全反相关 → rho=-1.0
    baseline_ratios = [0.50, 0.30, 0.40]
    vision = [0.4, 0.7, 0.5]
    quest = [0.5, 0.8, 0.6]

    reports = []
    for i, pid in enumerate(pids):
        reports.append(_fake_report(pid, "baseline", baseline_ratios[i]))
        reports.append(_fake_report(pid, "intervention", 0.20))
        reports.append(_fake_report(pid, "followup", 0.25,
                                    correction_times=[5.0, 6.0]))

    csv_path = tmp_path / "screening.csv"
    csv_path.write_text(
        "participant_id,vision_score,questionnaire_score\n" +
        "\n".join(f"{p},{v},{q}" for p, v, q in zip(pids, vision, quest)) + "\n",
        encoding="utf-8")

    result = run_study(
        sessions_config=reports,
        session_runner=lambda s: s,
        screening_path=str(csv_path),
    )

    assert "exploratory" in result
    expl = result["exploratory"]
    assert expl["spearman_vision"] is not None
    rho_v, _ = expl["spearman_vision"]
    assert rho_v == pytest.approx(-1.0)  # 完全反相关
    assert expl["spearman_questionnaire"] is not None
    # 3 被试 × 2 correction_times（仅 followup 阶段有）= 6 个值
    assert expl["correction_stats"]["n"] == 6
    assert expl["correction_stats"]["mean"] == pytest.approx(5.5)


def test_run_study_without_screening_returns_none_spearman(tmp_path):
    """B3：未提供 screening_path → spearman 为 None；correction_stats 仍聚合。"""
    from src.pipeline import run_study

    pids = ["p01", "p02", "p03"]
    reports = []
    for i, pid in enumerate(pids):
        reports.append(_fake_report(pid, "baseline", 0.50 - i * 0.10))
        reports.append(_fake_report(pid, "intervention", 0.20))
        reports.append(_fake_report(pid, "followup", 0.25,
                                    correction_times=[3.0]))

    result = run_study(
        sessions_config=reports,
        session_runner=lambda s: s,
        # 不传 screening_path
    )

    expl = result["exploratory"]
    assert expl["spearman_vision"] is None
    assert expl["spearman_questionnaire"] is None
    assert expl["correction_stats"]["n"] == 3  # 3 被试 × 1 correction_time


def test_run_study_no_correction_times_returns_none_correction_stats(tmp_path):
    """B3：无 correction_times → correction_stats 为 None。"""
    from src.pipeline import run_study

    pids = ["p01", "p02", "p03"]
    reports = []
    for i, pid in enumerate(pids):
        reports.append(_fake_report(pid, "baseline", 0.50 - i * 0.10))
        reports.append(_fake_report(pid, "intervention", 0.20))
        reports.append(_fake_report(pid, "followup", 0.25))  # 无 correction_times

    result = run_study(
        sessions_config=reports,
        session_runner=lambda s: s,
    )

    expl = result["exploratory"]
    assert expl["correction_stats"] is None
