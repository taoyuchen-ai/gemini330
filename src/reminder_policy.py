"""提醒触发逻辑（ADR-0007 + config/thresholds.yaml reminder）。

不良姿势连续持续 ≥ sustained_seconds 触发 Dotii 提醒；
两次提醒间最短间隔 min_interval_seconds；
跟踪阶段（followup）关掉提醒验证内化效应。
"""
from __future__ import annotations

from typing import Optional

from src.posture_classify import BAD_POSTURES, PostureLabel


def is_bad(label: PostureLabel) -> bool:
    """是否为不良姿势（趴桌 / 弯腰）。"""
    return label in BAD_POSTURES


def should_remind(bad_since: Optional[float],
                  now: float,
                  last_reminder: Optional[float],
                  phase: str,
                  thresholds: dict) -> bool:
    """是否应触发提醒。

    Args:
        bad_since: 当前不良持续开始时间（秒），None 表示当前非不良。
        now: 当前时间（秒）。
        last_reminder: 上次触发提醒时间（秒），None 表示从未触发。
        phase: "baseline" / "intervention" / "followup"。
        thresholds: 顶层配置 dict，含 "reminder" 子项。

    Returns:
        True 表示应触发提醒。
    """
    cfg = thresholds["reminder"]
    if cfg.get("track_phase_disabled", True) and phase == "followup":
        return False
    if bad_since is None:
        return False
    if now - bad_since < cfg["sustained_seconds"]:
        return False
    if last_reminder is not None and \
            (now - last_reminder) < cfg["min_interval_seconds"]:
        return False
    return True
