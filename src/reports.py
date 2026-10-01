"""日报生成（个人日报 / 班级日报）。

字段按 Q12：被试 ID、阶段、总帧、有效帧、不良帧、不良占比、
各姿势计数、提醒次数、会话起止。
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

from src.posture_classify import BAD_POSTURES, PostureLabel
from src.statistics import compute_ratio


def daily_report(participant_id: str,
                 labels: Sequence[PostureLabel],
                 phase: str = "",
                 reminders: int = 0,
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
