"""日报生成（个人日报 / 班级日报）。

字段按 Q12：被试 ID、阶段、总帧、有效帧、不良帧、不良占比、
各姿势计数、提醒次数、会话起止。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence

from src.posture_classify import BAD_POSTURES, PostureLabel
from src.statistics import compute_ratio

# ADR-0008 三阶段固定顺序（baseline → intervention → followup）
PHASE_ORDER: tuple[str, ...] = ("baseline", "intervention", "followup")


def daily_report(participant_id: str,
                 labels: Sequence[PostureLabel],
                 phase: str = "",
                 reminders: int = 0,
                 correction_times: Sequence[float] | None = None,
                 session_start: float | None = None,
                 session_end: float | None = None) -> dict:
    """生成个人日报。"""
    counts = Counter(l.value for l in labels)
    valid = [l for l in labels
             if l not in (PostureLabel.ABSENCE, PostureLabel.MISSING)]
    bad = sum(1 for l in valid if l in BAD_POSTURES)
    return {
        "participant_id": participant_id,
        "phase": phase,
        "session_start": session_start,
        "session_end": session_end,
        "total_frames": len(labels),
        "valid_frames": len(valid),
        "bad_frames": bad,
        "bad_ratio": compute_ratio(labels),
        "posture_counts": dict(counts),
        "reminders": reminders,
        "correction_times": list(correction_times) if correction_times else [],
    }


def class_report(reports: Sequence[dict]) -> dict:
    """聚合多个个人日报为班级日报。"""
    total_valid = sum(r["valid_frames"] for r in reports)
    total_bad = sum(r["bad_frames"] for r in reports)
    total_reminders = sum(r["reminders"] for r in reports)
    return {
        "n_participants": len(reports),
        "participants": [r["participant_id"] for r in reports],
        "total_valid_frames": total_valid,
        "total_bad_frames": total_bad,
        "class_bad_ratio": (total_bad / total_valid) if total_valid else 0.0,
        "total_reminders": total_reminders,
    }


def phase_columns(reports: Sequence[dict]) -> tuple[list[str], list[list[float]]]:
    """聚合 daily_reports 为 Friedman 列对齐格式（ADR-0008）。

    被试内前后测：每位被试每阶段一个 mean bad_ratio。
    - 同 (participant_id, phase) 多日 → 取均值
    - 任一阶段缺失该被试 → 整体剔除
    - 阶段顺序固定为 PHASE_ORDER = (baseline, intervention, followup)

    Args:
        reports: daily_report dict 序列。

    Returns:
        (participant_ids, [phase1_ratios, phase2_ratios, phase3_ratios])
        每组按 participant_ids 顺序对齐（同 index = 同被试）。
        无数据返回 ([], [[], [], []])。

    Raises:
        ValueError: 报告含 PHASE_ORDER 之外的阶段。
    """
    # 按 (participant, phase) 收集 bad_ratio 列表
    by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    participants_per_phase: dict[str, set[str]] = {p: set() for p in PHASE_ORDER}
    for r in reports:
        phase = r["phase"]
        if phase not in participants_per_phase:
            raise ValueError(
                f"未知阶段 '{phase}'，期望 {PHASE_ORDER} 之一")
        pid = r["participant_id"]
        by_pair[(pid, phase)].append(r["bad_ratio"])
        participants_per_phase[phase].add(pid)

    # 取每阶段都出现的被试交集
    common = set.intersection(*participants_per_phase.values()) \
        if participants_per_phase else set()
    pids = sorted(common)

    # 每阶段每被试取均值
    groups: list[list[float]] = []
    for phase in PHASE_ORDER:
        col = [sum(by_pair[(pid, phase)]) / len(by_pair[(pid, phase)])
               for pid in pids]
        groups.append(col)
    return pids, groups
