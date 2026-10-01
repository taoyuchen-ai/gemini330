"""视力筛查 + 体态问卷数据加载（ADR-0008 探索性分析）。

ADR-0001：视力/脊柱数据走「现状相关性」路径，使用学校现有筛查数据做关联分析。
本模块只负责加载与校验 CSV，不做统计计算（统计在 stats_analysis.spearman_corr）。

CSV 三列：participant_id, vision_score, questionnaire_score
（vision_score 越高代表视力越好；questionnaire_score 越高代表体态自评越好）。
"""
from __future__ import annotations

import csv

from typing import NamedTuple


class ScreeningRecord(NamedTuple):
    """单被试的视力筛查 + 体态问卷得分。"""
    participant_id: str
    vision_score: float
    questionnaire_score: float


# 期望的 CSV 列顺序（参与 run_study 探索性分析对齐）
_EXPECTED_COLUMNS: tuple[str, ...] = (
    "participant_id", "vision_score", "questionnaire_score",
)


def load_screening_csv(path: str) -> list[ScreeningRecord]:
    """从 CSV 加载筛查记录。

    Args:
        path: CSV 文件路径。首行需为 header，列顺序为
              participant_id, vision_score, questionnaire_score。

    Returns:
        ScreeningRecord 列表（按文件行序）。空文件（仅 header）返回 []。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: header 列与期望不符（缺失或多余）。
    """
    records: list[ScreeningRecord] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return records  # 完全空文件
        if tuple(header) != _EXPECTED_COLUMNS:
            raise ValueError(
                f"CSV header 列不符：期望 {_EXPECTED_COLUMNS}，实际 {tuple(header)}")
        for lineno, row in enumerate(reader, start=2):
            if not row:  # 跳过空行
                continue
            if len(row) != 3:
                raise ValueError(
                    f"第 {lineno} 行列数不符：期望 3，实际 {len(row)}")
            pid, vision, quest = row
            records.append(ScreeningRecord(
                participant_id=pid,
                vision_score=float(vision),
                questionnaire_score=float(quest),
            ))
    return records
