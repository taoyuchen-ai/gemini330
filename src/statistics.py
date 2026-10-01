"""统计口径（ADR-0006）。

1 秒窗移动众数平滑 + 不良占比公式：
  占比 = (趴桌帧 + 弯腰帧) / (有效帧 − 离座 − 缺失)
不插值；离座/缺失不计入分母。
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

from src.posture_classify import BAD_POSTURES, PostureLabel


def smooth(entries: Sequence[tuple[float, PostureLabel]],
           window_seconds: float = 1.0) -> list[PostureLabel]:
    """1 秒窗移动众数平滑。

    Args:
        entries: [(timestamp_seconds, label), ...] 按时间升序。
        window_seconds: 窗长（秒），中心对称 ±window/2。

    Returns:
        平滑后 label 列表，与 entries 等长。
    """
    n = len(entries)
    if n == 0:
        return []
    half = window_seconds / 2.0
    result: list[PostureLabel] = []
    for i, (t, _) in enumerate(entries):
        lo, hi = t - half, t + half
        window = [lab for (tt, lab) in entries if lo <= tt <= hi]
        if not window:
            result.append(entries[i][1])
            continue
        result.append(Counter(window).most_common(1)[0][0])
    return result


def compute_ratio(labels: Sequence[PostureLabel]) -> float:
    """不良姿势占比。

    分母 = 有效帧（排除离座 ABSENCE 与缺失 MISSING）。
    分子 = 不良帧（趴桌 + 弯腰）。
    无有效帧返回 0.0。
    """
    total = len(labels)
    if total == 0:
        return 0.0
    valid = [l for l in labels
             if l not in (PostureLabel.ABSENCE, PostureLabel.MISSING)]
    if not valid:
        return 0.0
    bad = sum(1 for l in valid if l in BAD_POSTURES)
    return bad / len(valid)
