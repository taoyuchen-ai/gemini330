"""screening 数据加载测试（ADR-0008 探索性分析：视力筛查 + 体态问卷）。

CSV 三列：participant_id, vision_score, questionnaire_score。
数据走「学校现有筛查数据」（ADR-0001），本模块只负责加载与校验。
"""
from __future__ import annotations

import pytest

from src.screening import ScreeningRecord, load_screening_csv


def _write_csv(path, rows):
    """辅助：写入 CSV 文本（rows 为行字符串列表，含 header）。"""
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_load_screening_csv_basic(tmp_path):
    p = tmp_path / "screening.csv"
    _write_csv(p, [
        "participant_id,vision_score,questionnaire_score",
        "p01,1.0,0.8",
        "p02,0.9,0.7",
        "p03,0.7,0.6",
    ])
    recs = load_screening_csv(str(p))
    assert len(recs) == 3
    assert recs[0] == ScreeningRecord("p01", 1.0, 0.8)
    assert recs[1].participant_id == "p02"
    assert recs[2].questionnaire_score == pytest.approx(0.6)


def test_load_screening_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_screening_csv(str(tmp_path / "nonexistent.csv"))


def test_load_screening_csv_missing_column(tmp_path):
    p = tmp_path / "bad.csv"
    _write_csv(p, [
        "participant_id,vision_score",  # 缺 questionnaire_score
        "p01,1.0",
    ])
    with pytest.raises(ValueError):
        load_screening_csv(str(p))


def test_load_screening_csv_extra_column(tmp_path):
    p = tmp_path / "extra.csv"
    _write_csv(p, [
        "participant_id,vision_score,questionnaire_score,extra",
        "p01,1.0,0.8,x",
    ])
    with pytest.raises(ValueError):
        load_screening_csv(str(p))


def test_load_screening_csv_header_only(tmp_path):
    p = tmp_path / "empty.csv"
    _write_csv(p, ["participant_id,vision_score,questionnaire_score"])
    assert load_screening_csv(str(p)) == []
