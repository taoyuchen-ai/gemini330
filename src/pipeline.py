"""主入口 pipeline：串联相机→推理→分类→深度→落盘→提醒→日报。

ADR-0001/0002/0005/0006/0007 整体编排。
不存图像，仅 33 关键点衍生标签 + 头-桌面距离。
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import yaml

from src.alerter import Alerter, NoopAlerter
from src.depth_distance import head_desk_distance, sample_depth
from src.dotii_reminder import show_fail, show_idle
from src.exceptions import CameraReconnectError, DotiiAPIError, MediaPipeTimeoutError
from src.phases import Phase
from src.posture_classify import NOSE, BAD_POSTURES, PostureLabel, classify
from src.reminder_policy import is_bad, should_remind
from src.reports import daily_report, phase_columns
from src.screening import load_screening_csv
from src.statistics import smooth
from src.storage import Storage
from src.stats_analysis import (
    analyze_three_phase,
    descriptive_stats,
    spearman_corr,
)


def load_thresholds(path: str = "config/thresholds.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _head_desk(kps, depth, desk_depth_m) -> Optional[float]:
    if depth is None or desk_depth_m is None:
        return None
    nose = kps[NOSE] if kps and len(kps) > NOSE else None
    if nose is None:
        return None
    hd = sample_depth(depth, nose[0], nose[1])
    return head_desk_distance(hd, desk_depth_m)


def run_session(camera,
                pose,
                storage: Storage,
                thresholds: dict,
                participant_id: str,
                phase: Phase,
                desk_depth_m: Optional[float] = None,
                dotii_show: Optional[Callable[[str], None]] = None,
                clock: Optional[Callable[[], float]] = None,
                max_frames: Optional[int] = None,
                alerter: Optional[Alerter] = None) -> dict:
    """主循环。

    Args:
        camera: 已 start 的 CameraCapture（或 fake），提供 grab() -> (color, depth)。
        pose: 已 start 的 PoseInference（或 fake），提供 infer(color) -> kps。
        storage: Storage 实例。
        thresholds: 顶层配置 dict。
        participant_id, phase: 被试与阶段。
        desk_depth_m: 桌面参考深度（米），None 表示不用深度。
        dotii_show: callable(expression) 触发 Dotii；None 表示无提醒通道。
        clock: 返回当前秒数的 callable；默认 time.time。
        max_frames: 最多处理帧数（测试用）；None 表示不限。
        alerter: PI 报警器（ADR-0007）；None 时使用 NoopAlerter 静默。

    Returns:
        daily_report dict。相机重连耗尽时提前退出，返回空日报。
    """
    if clock is None:
        clock = time.time
    if alerter is None:
        alerter = NoopAlerter()
    absence_thr = thresholds["absence"]["no_skeleton_seconds"]
    smooth_window = thresholds.get("smoothing", {}).get("window_seconds", 1.0)
    dotii_alert_thr = thresholds.get(
        "integrity", {}).get("dotii_failure_alert_threshold", 5)

    sid = storage.create_session(participant_id, phase)
    bad_since: Optional[float] = None
    last_reminder: Optional[float] = None
    last_label: Optional[PostureLabel] = None
    last_seen: Optional[float] = None
    no_skeleton_since: Optional[float] = None
    reminders = 0
    dotii_failure_count = 0
    labels: list[PostureLabel] = []
    entries: list[tuple[float, PostureLabel]] = []
    correction_times: list[float] = []
    t0 = clock()

    frame_idx = 0
    while max_frames is None or frame_idx < max_frames:
        frame_idx += 1
        now = clock()
        try:
            color, depth = camera.grab()
        except CameraReconnectError as e:
            # ADR-0007：相机重连耗尽 → 异常落盘 + PI 报警 + 提前退出
            storage.log_exception("camera_reconnect_exhausted", str(e),
                                  session_id=sid)
            alerter.alert("camera_reconnect_exhausted", str(e))
            break
        if color is None:
            kps = [None] * 33
        else:
            try:
                kps = pose.infer(color)
            except MediaPipeTimeoutError as e:
                # ADR-0007：MediaPipe 超时独立落盘到 exceptions 表
                storage.log_exception("mediapipe_timeout", str(e), session_id=sid)
                kps = [None] * 33

        has_skeleton = bool(kps and kps[NOSE] is not None)
        if has_skeleton:
            last_seen = now
            no_skeleton_since = None
        else:
            if no_skeleton_since is None:
                no_skeleton_since = now
        # 离座判定（连续无骨架 ≥ absence_thr）
        if not has_skeleton and no_skeleton_since is not None \
                and (now - no_skeleton_since) >= absence_thr:
            label = PostureLabel.ABSENCE
        else:
            label = classify(kps, _head_desk(kps, depth, desk_depth_m), thresholds)

        storage.insert_frame(sid, now, label)
        labels.append(label)
        entries.append((now, label))

        # 提醒逻辑（followup 阶段 should_remind 内部关闭）
        if is_bad(label):
            if bad_since is None:
                bad_since = now
            if dotii_show is not None and should_remind(
                    bad_since, now, last_reminder, phase, thresholds):
                try:
                    dotii_show("fail")
                    last_reminder = now
                    reminders += 1
                    dotii_failure_count = 0  # 成功 → 重置连续失败计数
                except DotiiAPIError as e:
                    # ADR-0007：Dotii API 异常独立落盘 + 连续失败计数
                    storage.log_exception("dotii_api_error", str(e), session_id=sid)
                    dotii_failure_count += 1
                    if dotii_failure_count >= dotii_alert_thr:
                        alerter.alert(
                            "dotii_failure_threshold",
                            f"Dotii 连续失败 {dotii_failure_count} 次（阈值 {dotii_alert_thr}）")
        else:
            if bad_since is not None and last_label in BAD_POSTURES \
                    and dotii_show is not None:
                try:
                    dotii_show("idle")
                    dotii_failure_count = 0  # 成功 → 重置
                except DotiiAPIError as e:
                    # ADR-0007：Dotii idle 提醒异常独立落盘
                    storage.log_exception("dotii_api_error", str(e), session_id=sid)
                    dotii_failure_count += 1
                    if dotii_failure_count >= dotii_alert_thr:
                        alerter.alert(
                            "dotii_failure_threshold",
                            f"Dotii 连续失败 {dotii_failure_count} 次（阈值 {dotii_alert_thr}）")
            # B2：本次 bad 周期内触发过提醒（last_reminder >= bad_since）
            # 且当前恢复 good → 记录从提醒到纠正的耗时
            if bad_since is not None and last_reminder is not None \
                    and last_reminder >= bad_since:
                correction_times.append(now - last_reminder)
            bad_since = None
        last_label = label

    storage.flush()
    # ADR-0005：单帧分类后用 1 秒窗移动众数平滑，避免单帧误判入统计。
    # 提醒/离座判定用原始 label（已落盘）；统计口径用平滑后 labels。
    smoothed_labels = smooth(entries, window_seconds=smooth_window)
    return daily_report(participant_id, smoothed_labels, phase=phase,
                        reminders=reminders,
                        correction_times=correction_times,
                        session_start=t0, session_end=clock(),
                        max_missing_ratio=thresholds.get(
                            "integrity", {}).get("max_missing_ratio"))


def run_study(sessions_config: list[dict],
              session_runner: Callable[[dict], dict],
              screening_path: Optional[str] = None) -> dict:
    """研究主分析编排（ADR-0008 端到端）。

    串联多被试 × 3 阶段（baseline/intervention/followup）会话，
    聚合为 Friedman 列对齐格式后调用 analyze_three_phase。
    探索性分析（ADR-0008）：若提供 screening_path，对齐 baseline bad_ratio
    与视力筛查/体态问卷得分做 Spearman 相关；聚合所有 correction_times
    做描述性统计。

    Args:
        sessions_config: 每项至少含 participant_id 与 phase；
                         其他字段由 session_runner 解释（如 kps_seq、max_frames）。
        session_runner: callable(sess_config) -> daily_report dict。
                         调用方负责构造 camera/pose/storage/thresholds 等依赖。
        screening_path: 可选，视力筛查 + 体态问卷 CSV 路径
                        （participant_id,vision_score,questionnaire_score）。

    Returns:
        {
            "daily_reports": [daily_report, ...],
            "analysis": analyze_three_phase 输出 dict 或 None（无数据）,
            "n_subjects": int,
            "exploratory": {
                "spearman_vision": (rho, p) | None,
                "spearman_questionnaire": (rho, p) | None,
                "correction_stats": dict | None,
            },
        }
    """
    daily_reports = [session_runner(s) for s in sessions_config]
    if not daily_reports:
        return {"daily_reports": [], "analysis": None, "n_subjects": 0,
                "exploratory": _exploratory_analysis([], [], [], screening_path)}

    # ADR-0006/0007：缺失比例 > 阈值的会话整体剔除，不进入 phase_columns
    # 与 _exploratory_analysis。daily_reports 仍全部返回以保留可见性。
    included = [r for r in daily_reports if not r.get("excluded", False)]
    pids, phases = phase_columns(included)
    if not pids:
        # 有日报但被试未在 3 阶段全出现 → 无可分析列
        return {"daily_reports": daily_reports, "analysis": None, "n_subjects": 0,
                "exploratory": _exploratory_analysis(included, [], [], screening_path)}

    if len(pids) < 3:
        # Friedman 需 ≥3 被试：剔除后不足则跳过主分析，但仍返回被试数与探索性分析
        return {"daily_reports": daily_reports, "analysis": None,
                "n_subjects": len(pids),
                "exploratory": _exploratory_analysis(included, pids,
                                                     phases.baseline, screening_path)}

    analysis = analyze_three_phase(phases)
    exploratory = _exploratory_analysis(included, pids,
                                        phases.baseline, screening_path)
    return {"daily_reports": daily_reports, "analysis": analysis,
            "n_subjects": len(pids), "exploratory": exploratory}


def _exploratory_analysis(daily_reports: list[dict],
                          pids: list[str],
                          baseline_ratios: list[float],
                          screening_path: Optional[str]) -> dict:
    """ADR-0008 探索性分析编排。

    - Spearman 相关：baseline bad_ratio × vision_score / questionnaire_score
      （需 screening_path 且对齐后 ≥3 样本，否则返回 None）
    - 描述性统计：所有 daily_reports 的 correction_times 聚合
      （空则返回 None）
    """
    spearman_vision: Optional[tuple[float, float]] = None
    spearman_questionnaire: Optional[tuple[float, float]] = None

    if screening_path and pids:
        records = load_screening_csv(screening_path)
        rec_by_pid = {r.participant_id: r for r in records}
        # 对齐：pids 顺序中在 screening 出现的被试
        aligned = [(br, rec_by_pid[pid].vision_score,
                    rec_by_pid[pid].questionnaire_score)
                   for pid, br in zip(pids, baseline_ratios)
                   if pid in rec_by_pid]
        if len(aligned) >= 3:
            bad_ratios = [a[0] for a in aligned]
            vision_scores = [a[1] for a in aligned]
            quest_scores = [a[2] for a in aligned]
            spearman_vision = spearman_corr(bad_ratios, vision_scores)
            spearman_questionnaire = spearman_corr(bad_ratios, quest_scores)

    # 聚合所有 correction_times
    all_correction_times: list[float] = []
    for r in daily_reports:
        all_correction_times.extend(r.get("correction_times", []))
    correction_stats = descriptive_stats(all_correction_times) \
        if all_correction_times else None

    return {
        "spearman_vision": spearman_vision,
        "spearman_questionnaire": spearman_questionnaire,
        "correction_stats": correction_stats,
    }
    analysis = analyze_three_phase(phases)
    exploratory = _exploratory_analysis(included, pids,
                                        phases.baseline, screening_path)
    return {"daily_reports": daily_reports, "analysis": analysis,
            "n_subjects": len(pids), "exploratory": exploratory}


def _exploratory_analysis(daily_reports: list[dict],
                          pids: list[str],
                          baseline_ratios: list[float],
                          screening_path: Optional[str]) -> dict:
    """ADR-0008 探索性分析编排。

    - Spearman 相关：baseline bad_ratio × vision_score / questionnaire_score
      （需 screening_path 且对齐后 ≥3 样本，否则返回 None）
    - 描述性统计：所有 daily_reports 的 correction_times 聚合
      （空则返回 None）
    """
    spearman_vision: Optional[tuple[float, float]] = None
    spearman_questionnaire: Optional[tuple[float, float]] = None

    if screening_path and pids:
        records = load_screening_csv(screening_path)
        rec_by_pid = {r.participant_id: r for r in records}
        # 对齐：pids 顺序中在 screening 出现的被试
        aligned = [(br, rec_by_pid[pid].vision_score,
                    rec_by_pid[pid].questionnaire_score)
                   for pid, br in zip(pids, baseline_ratios)
                   if pid in rec_by_pid]
        if len(aligned) >= 3:
            bad_ratios = [a[0] for a in aligned]
            vision_scores = [a[1] for a in aligned]
            quest_scores = [a[2] for a in aligned]
            spearman_vision = spearman_corr(bad_ratios, vision_scores)
            spearman_questionnaire = spearman_corr(bad_ratios, quest_scores)

    # 聚合所有 correction_times
    all_correction_times: list[float] = []
    for r in daily_reports:
        all_correction_times.extend(r.get("correction_times", []))
    correction_stats = descriptive_stats(all_correction_times) \
        if all_correction_times else None

    return {
        "spearman_vision": spearman_vision,
        "spearman_questionnaire": spearman_questionnaire,
        "correction_stats": correction_stats,
    }
