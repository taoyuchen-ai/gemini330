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

    ADR-0005：单帧分类后用 1 秒窗移动众数平滑，避免单帧误判入统计。
    ADR-0006/0007：MISSING 与 ABSENCE 帧受保护——
      - 当前帧为 MISSING/ABSENCE → 保留原标签（缺失与离座不计入分母）
      - 有效姿态帧的众数窗口排除 MISSING/ABSENCE，避免污染

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
    protected = {PostureLabel.MISSING, PostureLabel.ABSENCE}
    result: list[PostureLabel] = []
    for i, (t, lab) in enumerate(entries):
        if lab in protected:
            result.append(lab)
            continue
        lo, hi = t - half, t + half
        window = [l for (tt, l) in entries
                  if lo <= tt <= hi and l not in protected]
        if not window:
            result.append(lab)
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


def compute_missing_ratio(labels: Sequence[PostureLabel]) -> float:
    """缺失帧占比（ADR-0006/0007）。

    分子 = MISSING + ABSENCE；分母 = 总帧数。
    用于 ADR-0007 缺失比例 > 阈值时会话整体剔除。
    空 list 返回 0.0（避免除零）。
    """
    total = len(labels)
    if total == 0:
        return 0.0
    missing = sum(1 for l in labels
                  if l in (PostureLabel.MISSING, PostureLabel.ABSENCE))
    return missing / total
