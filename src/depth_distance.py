"""深度判定头-桌面距离（ADR-0002）。

用 Orbbec aligned depth 流（mm 单位）取头部像素深度，
与现场标定的桌面参考深度作差，得到头-桌面距离（米）。
用于区分前倾读写 vs 趴桌。
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def sample_depth(depth_frame: np.ndarray,
                 x_norm: float, y_norm: float) -> Optional[float]:
    """从 aligned depth frame 取 (x,y) 归一化像素深度（米）。

    depth_frame: HxW 数组，值为毫米（0 表示无效深度）。
    x_norm, y_norm: MediaPipe 归一化坐标 [0,1]。
    """
    if depth_frame is None or x_norm is None or y_norm is None:
        return None
    h, w = depth_frame.shape[:2]
    ix = int(min(max(x_norm, 0.0), 1.0) * (w - 1))
    iy = int(min(max(y_norm, 0.0), 1.0) * (h - 1))
    v = float(depth_frame[iy, ix])
    if v <= 0:
        return None
    return v / 1000.0  # mm → m


def head_desk_distance(head_depth_m: Optional[float],
                       desk_depth_m: Optional[float]) -> Optional[float]:
    """头-桌面距离（米）。任一为 None 返回 None。"""
    if head_depth_m is None or desk_depth_m is None:
        return None
    return abs(desk_depth_m - head_depth_m)
