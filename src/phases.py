"""实验阶段类型（ADR-0008 三阶段）。

被试内前后测固定 3 阶段：baseline / intervention / followup。
Phase 用 Literal 类型约束，消除 Primitive Obsession（phase:str）。
ThreePhase 封装三阶段列对齐数据，消除 baseline/intervention/followup 三元组 data clump。
"""
from __future__ import annotations

from typing import Literal, NamedTuple, Sequence

# ADR-0008 三阶段固定顺序（baseline → intervention → followup）
PHASE_ORDER: tuple[str, ...] = ("baseline", "intervention", "followup")

# 阶段标识符类型约束。None 表示"无阶段"（如纯函数构造的日报不绑定阶段）。
Phase = Literal["baseline", "intervention", "followup"]


class ThreePhase(NamedTuple):
    """三阶段被试内数据（baseline/intervention/followup 列对齐）。

    每字段为该阶段所有被试的测量值序列，列对齐（同 index = 同被试）。
    消除 baseline/intervention/followup 三元组 data clump（ADR-0008 Standards S2）。
    """
    baseline: Sequence[float]
    intervention: Sequence[float]
    followup: Sequence[float]

    @property
    def n_subjects(self) -> int:
        return len(self.baseline)

    @property
    def k_phases(self) -> int:
        return 3
