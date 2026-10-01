"""姿态分类（ADR-0005）。

6 类坐姿 + 离座/缺失：
  坐正 / 前倾读写 / 趴桌(不良) / 弯腰(不良) / 后仰(中性) / 转头侧身(中性)

不良 = 趴桌 + 弯腰
判定优先级（高到低）：
  1. 缺失帧（核心关键点不全）
  2. 转头侧身（中性）
  3. 后仰（中性）
  4. 趴桌（深度 < threshold，不良）
  5. 弯腰（躯干-大腿角 < threshold，不良）
  6. 前倾读写（头-桌面距离在可接受区间 + 躯干前倾）
  7. 坐正
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Optional

import numpy as np

# MediaPipe BlazePose 33 关键点索引
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26


class PostureLabel(str, Enum):
    SIT_UPRIGHT = "sit_upright"
    FORWARD_READ = "forward_read"
    DESK_LYING = "desk_lying"
    BEND = "bend"
    LEAN_BACK = "lean_back"
    TURN_SIDE = "turn_side"
    ABSENCE = "absence"
    MISSING = "missing"


# 不良姿势集合（计入分子）
BAD_POSTURES = frozenset({PostureLabel.DESK_LYING, PostureLabel.BEND})
# 中性姿势集合（不计入不良）
NEUTRAL_POSTURES = frozenset({PostureLabel.LEAN_BACK, PostureLabel.TURN_SIDE})


def _vec(keypoints, idx) -> Optional[np.ndarray]:
    """取关键点为 np.array([x,y,z])，缺失返回 None。

    MediaPipe 归一化坐标，y 朝下；z 缺失补 0。
    """
    if keypoints is None:
        return None
    kp = keypoints[idx]
    if kp is None:
        return None
    arr = np.asarray(kp, dtype=float).ravel()
    if arr.size < 2:
        return None
    x, y = float(arr[0]), float(arr[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    z = float(arr[2]) if arr.size >= 3 and math.isfinite(float(arr[2])) else 0.0
    return np.array([x, y, z], dtype=float)


def _mid(a, b):
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def _joint_angle(a, b, c) -> Optional[float]:
    """a-b-c 关节角（度），任一缺失返回 None。"""
    if a is None or b is None or c is None:
        return None
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _lean_angle(trunk_vec) -> Optional[float]:
    """躯干向量与垂直向上方向的夹角（度）。"""
    if trunk_vec is None:
        return None
    n = float(np.linalg.norm(trunk_vec))
    if n < 1e-9:
        return None
    cos = float(np.clip(-trunk_vec[1] / n, -1.0, 1.0))
    return math.degrees(math.acos(cos))


def _lateral_angle(trunk_vec) -> Optional[float]:
    """躯干在 x 方向的侧偏角（度）。"""
    if trunk_vec is None:
        return None
    y = abs(float(trunk_vec[1]))
    if y < 1e-9:
        return 90.0
    return math.degrees(math.atan2(abs(float(trunk_vec[0])), y))


def _head_yaw(nose, l_ear, r_ear) -> Optional[float]:
    """头部偏航角（度）：鼻相对双耳中点的横向偏转。"""
    if nose is None or l_ear is None or r_ear is None:
        return None
    mid_x = (l_ear[0] + r_ear[0]) / 2.0
    ear_dx = abs(r_ear[0] - l_ear[0])
    if ear_dx < 1e-9:
        return None
    offset = abs(nose[0] - mid_x) / ear_dx
    return math.degrees(math.atan(offset))


def classify(keypoints, head_desk_distance, thresholds) -> PostureLabel:
    """单帧姿态分类。

    Args:
        keypoints: 长度 33 的 list，每元素 (x,y,z) 或 None（MediaPipe 归一化坐标，y 朝下）。
        head_desk_distance: 头-桌面距离（米），None 表示无深度流。
        thresholds: dict，含 rgb/depth 子项（见 config/thresholds.yaml）。

    Returns:
        PostureLabel
    """
    rgb = thresholds["rgb"]
    depth = thresholds["depth"]

    nose = _vec(keypoints, NOSE)
    l_sh = _vec(keypoints, LEFT_SHOULDER)
    r_sh = _vec(keypoints, RIGHT_SHOULDER)
    l_hip = _vec(keypoints, LEFT_HIP)
    r_hip = _vec(keypoints, RIGHT_HIP)
    l_kn = _vec(keypoints, LEFT_KNEE)
    r_kn = _vec(keypoints, RIGHT_KNEE)
    l_ear = _vec(keypoints, LEFT_EAR)
    r_ear = _vec(keypoints, RIGHT_EAR)

    sh_mid = _mid(l_sh, r_sh)
    hip_mid = _mid(l_hip, r_hip)
    knee_mid = _mid(l_kn, r_kn)

    # 1. 缺失帧
    if any(c is None for c in (sh_mid, hip_mid, knee_mid, nose, l_ear, r_ear)):
        return PostureLabel.MISSING

    trunk_vec = sh_mid - hip_mid

    # 2. 转头侧身（中性）
    lateral = _lateral_angle(trunk_vec)
    yaw = _head_yaw(nose, l_ear, r_ear)
    if (lateral is not None and lateral > rgb["trunk_lateral_lean_threshold"]) or \
       (yaw is not None and yaw > rgb["head_yaw_turn_threshold"]):
        return PostureLabel.TURN_SIDE

    # 3. 后仰（中性）
    lean = _lean_angle(trunk_vec)
    if lean is not None and trunk_vec[2] < 0 and lean > rgb["trunk_lean_back_threshold"]:
        return PostureLabel.LEAN_BACK

    # 4. 趴桌（不良）
    if head_desk_distance is not None and head_desk_distance < depth["desk_lying_threshold"]:
        return PostureLabel.DESK_LYING

    # 5. 弯腰（不良）
    trunk_thigh = _joint_angle(sh_mid, hip_mid, knee_mid)
    if trunk_thigh is not None and trunk_thigh < rgb["trunk_thigh_angle_bend_threshold"]:
        return PostureLabel.BEND

    # 6. 前倾读写
    if head_desk_distance is not None and trunk_vec[2] > 0:
        if depth["forward_read_min"] <= head_desk_distance <= depth["forward_read_max"]:
            return PostureLabel.FORWARD_READ

    # 7. 坐正
    return PostureLabel.SIT_UPRIGHT
