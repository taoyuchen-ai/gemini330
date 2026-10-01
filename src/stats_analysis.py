"""统计分析方法（ADR-0008）。

被试内前后测，3 阶段（baseline / intervention / followup），6 人焦点被试。
主结局：不良坐姿时长占比（连续型）。非参数方法（小样本 + 不假设正态）。

主分析：
- Friedman 检验：3 阶段一次性差异
- 事后 Wilcoxon 符号秩：成对比较
- Bonferroni 校正：多重比较
- η² 效应量：基于 Friedman χ²

敏感性分析（贝叶斯）：ADR-0008 标注"可选"，本模块不实现；后续可补 stats_bayes.py。

探索性分析：
- Spearman 相关：不良占比 × 视力筛查 / 体态问卷得分
- 描述性统计：提醒触发后纠正耗时（秒）
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import stats


# ---------- Friedman 检验 ----------

def friedman_test(groups: Sequence[Sequence[float]]) -> tuple[float, float]:
    """Friedman 检验：多阶段（k≥2）被试内差异。

    Args:
        groups: [[被试1_阶段1, 被试2_阶段1, ...], [被试1_阶段2, ...], ...]
                shape = (k_phases, n_subjects)，列对齐被试。

    Returns:
        (chi2_statistic, p_value)。
    """
    if len(groups) < 2:
        raise ValueError("Friedman 需 ≥2 组")
    n = len(groups[0])
    if n == 0:
        raise ValueError("每组至少 1 个被试")
    for g in groups:
        if len(g) != n:
            raise ValueError("各组被试数必须一致（被试内设计列对齐）")
    chi2, p = stats.friedmanchisquare(*groups)
    # 完全相同组（所有秩并列）→ scipy 除零返回 NaN；兜底为无差异
    if math.isnan(chi2):
        return 0.0, 1.0
    return float(chi2), float(p)


# ---------- Wilcoxon 符号秩 ----------

def wilcoxon_signed_rank(before: Sequence[float],
                         after: Sequence[float]) -> tuple[float, float]:
    """Wilcoxon 符号秩检验：成对比较。

    Args:
        before, after: 等长的被试内两组测量。

    Returns:
        (W_statistic, p_value)。
        完全相同（所有差为 0）→ scipy 抛 ValueError，本实现兜底返回 (0.0, 1.0)。
    """
    if len(before) != len(after):
        raise ValueError("before/after 长度必须一致")
    if len(before) == 0:
        raise ValueError("数据不能为空")
    diffs = np.asarray(before, dtype=float) - np.asarray(after, dtype=float)
    if np.all(diffs == 0):
        return 0.0, 1.0
    result = stats.wilcoxon(before, after)
    return float(result.statistic), float(result.pvalue)


# ---------- Bonferroni 校正 ----------

def bonferroni(p_values: Sequence[float],
               alpha: float = 0.05) -> list[tuple[float, bool]]:
    """Bonferroni 多重比较校正。

    Args:
        p_values: 原始 p 值序列。
        alpha: 显著性水平（默认 0.05）。

    Returns:
        [(corrected_p, reject), ...] 与输入等长。
        corrected_p = min(p * m, 1.0)，reject = corrected_p < alpha。
    """
    m = len(p_values)
    if m == 0:
        return []
    out: list[tuple[float, bool]] = []
    for p in p_values:
        cp = min(p * m, 1.0)
        out.append((cp, cp < alpha))
    return out


# ---------- Kendall's W 与 η² ----------

def _kendalls_w_from_chi2(chi2: float, n: int, k: int) -> float:
    """由已算的 Friedman χ² 计算 Kendall's W（避免重算 χ²）。

    W = χ²_F / [N*(k-1)]，N = 被试数，k = 阶段数。
    完全一致 W=1.0；无一致 W=0.0。
    """
    if n * (k - 1) == 0:
        return 0.0
    return chi2 / (n * (k - 1))


def _eta_squared_from_chi2(chi2: float, n: int, k: int) -> float:
    """由已算的 Friedman χ² 计算 η²（避免重算 χ²）。

    Tomczak & Tomczak (2014) 对 Friedman 检验的公式：
        η² = χ²_F / [N*(k-1)]
    该公式在数学上等于 Kendall's W（ADR-0008：Kendall's W 转 η²）。范围 [0, 1]。
    """
    if n * (k - 1) == 0:
        return 0.0
    return chi2 / (n * (k - 1))


def kendalls_w(groups: Sequence[Sequence[float]]) -> float:
    """Kendall's W 一致性系数。

    W = χ²_F / [N*(k-1)]，N = 被试数，k = 阶段数。
    完全一致 W=1.0；无一致 W=0.0。
    """
    n = len(groups[0])
    k = len(groups)
    chi2, _ = friedman_test(groups)
    return _kendalls_w_from_chi2(chi2, n, k)


def eta_squared_friedman(groups: Sequence[Sequence[float]]) -> float:
    """Friedman η² 效应量。

    采用 Tomczak & Tomczak (2014) 对 Friedman 检验的公式：
        η² = χ²_F / [N*(k-1)]

    N = 被试数，k = 阶段数。该公式在数学上等于 Kendall's W
    （ADR-0008：报告效应量 η²，Kendall's W 转 η²）。范围 [0, 1]。
    """
    n = len(groups[0])
    k = len(groups)
    chi2, _ = friedman_test(groups)
    return _eta_squared_from_chi2(chi2, n, k)


# ---------- Spearman 相关 ----------

def spearman_corr(x: Sequence[float],
                  y: Sequence[float]) -> tuple[float, float]:
    """Spearman ρ 秩相关。

    Args:
        x, y: 等长测量序列。

    Returns:
        (rho, p_value)。
    """
    if len(x) != len(y):
        raise ValueError("x/y 长度必须一致")
    if len(x) < 3:
        raise ValueError("Spearman 需 ≥3 对样本")
    result = stats.spearmanr(x, y)
    if np.isnan(result.statistic):
        # 常数序列 → scipy 返回 NaN；定义为无相关
        return 0.0, 1.0
    return float(result.statistic), float(result.pvalue)


# ---------- 描述性统计 ----------

def descriptive_stats(samples: Sequence[float]) -> dict:
    """描述性统计（提醒后纠正耗时等）。

    Returns:
        {n, mean, median, std, min, max, q1, q3, iqr}
        std 用样本标准差（ddof=1）；n<2 时 std=0.0。
    """
    arr = np.asarray(samples, dtype=float)
    n = arr.size
    if n == 0:
        raise ValueError("samples 不能为空")
    std = float(np.std(arr, ddof=1)) if n >= 2 else 0.0
    q1 = float(np.percentile(arr, 25, method="linear"))
    q3 = float(np.percentile(arr, 75, method="linear"))
    return {
        "n": n,
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": std,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }


# ---------- 主分析入口 ----------

def analyze_three_phase(baseline: Sequence[float],
                        intervention: Sequence[float],
                        followup: Sequence[float],
                        alpha: float = 0.05) -> dict:
    """三阶段被试内主分析（ADR-0008 主分析入口）。

    流程：
    1. Friedman 检验 3 阶段差异
    2. 3 个事后 Wilcoxon 符号秩（baseline×intervention / baseline×followup / intervention×followup）
    3. Bonferroni 校正 3 个 p 值
    4. η² 效应量
    5. Kendall's W

    Args:
        baseline, intervention, followup: 等长（n_subjects）的不良占比序列。

    Returns:
        {friedman: (chi2, p), post_hoc: [...], bonferroni: [...],
         eta_squared: float, kendalls_w: float,
         n_subjects: int, k_phases: int}
    """
    n = len(baseline)
    if not (len(intervention) == n == len(followup)):
        raise ValueError("三阶段被试数必须一致")
    if n < 3:
        raise ValueError("Friedman 需 ≥3 被试")

    groups = [list(baseline), list(intervention), list(followup)]
    chi2, p_friedman = friedman_test(groups)

    pairs = [
        ("baseline_vs_intervention", baseline, intervention),
        ("baseline_vs_followup", baseline, followup),
        ("intervention_vs_followup", intervention, followup),
    ]
    post_hoc = []
    raw_pvals = []
    for label, b, a in pairs:
        w, p_w = wilcoxon_signed_rank(b, a)
        post_hoc.append({
            "comparison": label,
            "w_statistic": w,
            "p_value": p_w,
        })
        raw_pvals.append(p_w)

    corrected = bonferroni(raw_pvals, alpha=alpha)
    bonferroni_out = [
        {"comparison": post_hoc[i]["comparison"],
         "corrected_p": corrected[i][0],
         "reject": corrected[i][1]}
        for i in range(len(corrected))
    ]

    return {
        "friedman": (chi2, p_friedman),
        "post_hoc": post_hoc,
        "bonferroni": bonferroni_out,
        "eta_squared": _eta_squared_from_chi2(chi2, n, len(groups)),
        "kendalls_w": _kendalls_w_from_chi2(chi2, n, len(groups)),
        "n_subjects": n,
        "k_phases": 3,
    }
