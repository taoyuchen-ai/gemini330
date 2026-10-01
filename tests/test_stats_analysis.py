"""stats_analysis 的统计检验测试（ADR-0008）。

非参数方法：
- Friedman 检验（3 阶段一次性差异）
- Wilcoxon 符号秩（事后成对比较）
- Bonferroni 校正多重比较
- η² 效应量（Kendall's W 转 η²）
- Spearman 相关（探索性）
- 描述性统计（提醒后纠正耗时）

约束：6 人被试内前后测，3 阶段（baseline / intervention / followup）。
"""
from __future__ import annotations

import math

import pytest

from src.stats_analysis import (
    analyze_three_phase,
    bonferroni,
    descriptive_stats,
    eta_squared_friedman,
    friedman_test,
    kendalls_w,
    spearman_corr,
    wilcoxon_signed_rank,
)


# ---------- Friedman ----------

def test_friedman_empty_raises():
    with pytest.raises(ValueError):
        friedman_test([])


def test_friedman_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        friedman_test([[0.1, 0.2, 0.3], [0.1, 0.2]])  # 第二组长度不一致


def test_friedman_identical_groups_chi2_zero():
    g = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    chi2, p = friedman_test([g, g, g])
    assert chi2 == 0.0
    assert p == 1.0


def test_friedman_known_small_dataset():
    # 3 被试 × 3 阶段，已知秩次可手算
    # baseline = [0.40, 0.50, 0.60]
    # intervention = [0.20, 0.30, 0.40]
    # followup = [0.30, 0.40, 0.50]
    # 每个被试内排名：baseline=3, intervention=1, followup=2
    # R_baseline=9, R_intervention=3, R_followup=6
    # chi2 = 12/(N*k*(k+1)) * Σ(R_j - R̄)² , N=3, k=3, R̄=N*(k+1)/2=6
    # Σ(R-6)² = 9+0+9 = 18
    # chi2 = 12/(3*3*4) * 18 = 12/36*18 = 6.0
    baseline = [0.40, 0.50, 0.60]
    intervention = [0.20, 0.30, 0.40]
    followup = [0.30, 0.40, 0.50]
    chi2, p = friedman_test([baseline, intervention, followup])
    assert chi2 == pytest.approx(6.0, abs=1e-9)
    assert 0.0 < p < 0.1  # 小样本下 p 应在边界


# ---------- Wilcoxon 符号秩 ----------

def test_wilcoxon_identical_series():
    x = [0.5, 0.4, 0.3, 0.6, 0.5, 0.4]
    w, p = wilcoxon_signed_rank(x, x)
    # 完全相同 → 所有差为 0 → scipy 抛 ValueError，本实现返回 (0.0, 1.0)
    assert w == 0.0
    assert p == 1.0


def test_wilcoxon_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        wilcoxon_signed_rank([0.1, 0.2, 0.3], [0.1, 0.2])


def test_wilcoxon_known_small_dataset():
    # before = [0.50, 0.45, 0.40, 0.55, 0.48, 0.52]
    # after  = [0.30, 0.25, 0.20, 0.35, 0.28, 0.32]
    # 差 d = after - before 全为负
    # |d|  = [0.20, 0.20, 0.20, 0.20, 0.20, 0.20]，秩 1..6 平均后 = 3.5
    # 负秩和 W- = 3.5*6 = 21, W+ = 0
    # scipy.stats.wilcoxon 默认返回 min(W+, W-) = 0
    before = [0.50, 0.45, 0.40, 0.55, 0.48, 0.52]
    after = [0.30, 0.25, 0.20, 0.35, 0.28, 0.32]
    w, p = wilcoxon_signed_rank(before, after)
    assert w == 0.0
    assert p < 0.05


# ---------- Bonferroni ----------

def test_bonferroni_basic_correction():
    pvals = [0.01, 0.04, 0.10]
    out = bonferroni(pvals, alpha=0.05)
    assert out[0][0] == pytest.approx(0.03)
    assert out[0][1] is True
    assert out[1][0] == pytest.approx(0.12)
    assert out[1][1] is False
    assert out[2][0] == pytest.approx(0.30)
    assert out[2][1] is False


def test_bonferroni_caps_at_one():
    out = bonferroni([0.5, 0.6, 0.7])
    assert all(cp <= 1.0 for cp, _ in out)


def test_bonferroni_empty():
    assert bonferroni([]) == []


# ---------- Kendall's W 与 η² ----------

def test_kendalls_w_identical_zero():
    g = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    w = kendalls_w([g, g, g])
    assert w == 0.0


def test_kendalls_w_strictly_ordered_one():
    # 每个被试 3 阶段排名完全一致 → W=1
    # baseline > followup > intervention（每个被试内一致）
    baseline = [0.60, 0.50, 0.40, 0.55, 0.45, 0.35]
    intervention = [0.10, 0.05, 0.0, 0.05, 0.0, -0.05]
    followup = [0.30, 0.25, 0.20, 0.25, 0.20, 0.15]
    w = kendalls_w([baseline, intervention, followup])
    assert w == pytest.approx(1.0, abs=1e-9)


def test_eta_squared_friedman_known():
    # 与 test_friedman_known_small_dataset 同数据，chi2 = 6.0, N=3, k=3
    # Tomczak & Tomczak (2014) 对 Friedman 的 η² 公式：
    #   η² = χ²_F / [N*(k-1)] = 6.0 / (3*2) = 1.0
    # 该公式在数学上等于 Kendall's W（ADR-0008：Kendall's W 转 η²）
    baseline = [0.40, 0.50, 0.60]
    intervention = [0.20, 0.30, 0.40]
    followup = [0.30, 0.40, 0.50]
    eta = eta_squared_friedman([baseline, intervention, followup])
    assert eta == pytest.approx(1.0, abs=1e-9)


# ---------- Spearman ----------

def test_spearmon_perfect_monotonic():
    x = [1, 2, 3, 4, 5, 6]
    y = [10, 20, 30, 40, 50, 60]
    rho, p = spearman_corr(x, y)
    assert rho == pytest.approx(1.0)
    assert p < 0.01


def test_spearmon_inverse_monotonic():
    x = [1, 2, 3, 4, 5, 6]
    y = [60, 50, 40, 30, 20, 10]
    rho, p = spearman_corr(x, y)
    assert rho == pytest.approx(-1.0)


def test_spearmon_length_mismatch_raises():
    with pytest.raises(ValueError):
        spearman_corr([1, 2, 3], [1, 2])


# ---------- 描述性统计 ----------

def test_descriptive_stats_basic():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    d = descriptive_stats(samples)
    assert d["n"] == 6
    assert d["mean"] == pytest.approx(3.5)
    assert d["median"] == pytest.approx(3.5)
    assert d["std"] > 0
    assert d["q1"] == pytest.approx(2.25)
    assert d["q3"] == pytest.approx(4.75)
    assert d["iqr"] == pytest.approx(2.5)
    assert d["min"] == 1.0
    assert d["max"] == 6.0


def test_descriptive_stats_empty_raises():
    with pytest.raises(ValueError):
        descriptive_stats([])


# ---------- analyze_three_phase（主分析入口） ----------

def test_analyze_three_phase_returns_full_structure():
    rng_baseline = [0.50, 0.45, 0.40, 0.55, 0.48, 0.52]
    rng_intervention = [0.30, 0.25, 0.20, 0.35, 0.28, 0.32]
    rng_followup = [0.35, 0.30, 0.25, 0.40, 0.33, 0.37]
    out = analyze_three_phase(rng_baseline, rng_intervention, rng_followup)
    assert set(out.keys()) == {
        "friedman", "post_hoc", "bonferroni", "eta_squared",
        "kendalls_w", "n_subjects", "k_phases",
    }
    assert out["n_subjects"] == 6
    assert out["k_phases"] == 3
    chi2, p = out["friedman"]
    assert chi2 > 0
    assert 0.0 <= p <= 1.0
    # 3 个成对比较
    assert len(out["post_hoc"]) == 3
    assert len(out["bonferroni"]) == 3
    labels = [pair["comparison"] for pair in out["post_hoc"]]
    assert "baseline_vs_intervention" in labels
    assert "baseline_vs_followup" in labels
    assert "intervention_vs_followup" in labels
    # η² 在 [0, 1] 区间（理论上；小样本可能 >1，按 cap 处理）
    assert out["eta_squared"] >= 0.0
