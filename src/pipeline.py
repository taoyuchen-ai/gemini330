"""主入口 pipeline：串联相机→推理→分类→深度→落盘→提醒→日报。

ADR-0001/0002/0005/0006/0007 整体编排。
不存图像，仅 33 关键点衍生标签 + 头-桌面距离。
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import yaml

from src.depth_distance import head_desk_distance, sample_depth
from src.dotii_reminder import show_fail, show_idle
from src.exceptions import DotiiAPIError, MediaPipeTimeoutError
from src.posture_classify import NOSE, BAD_POSTURES, PostureLabel, classify
from src.reminder_policy import is_bad, should_remind
from src.reports import daily_report
from src.storage import Storage


def load_thresholds(path: str = "config/thresholds.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _head_desk(kps, depth, desk_depth_m) -> Optional[float]:
    if depth is None or desk_depth_m is None:
        return None
    nose = kps[NOSE] if kps and len(kps) > NOSE else None
    if nose is None:
        return None
    hd = sample_depth(depth, nose[0], nose[1])
    return head_desk_distance(hd, desk_depth_m)


def run_session(camera,
                pose,
                storage: Storage,
                thresholds: dict,
                participant_id: str,
                phase: str,
                desk_depth_m: Optional[float] = None,
                dotii_show: Optional[Callable[[str], None]] = None,
                clock: Optional[Callable[[], float]] = None,
                max_frames: Optional[int] = None) -> dict:
    """主循环。

    Args:
        camera: 已 start 的 CameraCapture（或 fake），提供 grab() -> (color, depth)。
        pose: 已 start 的 PoseInference（或 fake），提供 infer(color) -> kps。
        storage: Storage 实例。
        thresholds: 顶层配置 dict。
        participant_id, phase: 被试与阶段。
        desk_depth_m: 桌面参考深度（米），None 表示不用深度。
        dotii_show: callable(expression) 触发 Dotii；None 表示无提醒通道。
        clock: 返回当前秒数的 callable；默认 time.time。
        max_frames: 最多处理帧数（测试用）；None 表示不限。

    Returns:
        daily_report dict。
    """
    if clock is None:
        clock = time.time
    absence_thr = thresholds["absence"]["no_skeleton_seconds"]

    sid = storage.create_session(participant_id, phase)
    bad_since: Optional[float] = None
    last_reminder: Optional[float] = None
    last_label: Optional[PostureLabel] = None
    last_seen: Optional[float] = None
    no_skeleton_since: Optional[float] = None
    reminders = 0
    labels: list[PostureLabel] = []
    t0 = clock()

    frame_idx = 0
    while max_frames is None or frame_idx < max_frames:
        frame_idx += 1
        now = clock()
        color, depth = camera.grab()
        if color is None:
            kps = [None] * 33
        else:
            try:
                kps = pose.infer(color)
            except MediaPipeTimeoutError:
                kps = [None] * 33

        has_skeleton = bool(kps and kps[NOSE] is not None)
        if has_skeleton:
            last_seen = now
            no_skeleton_since = None
        else:
            if no_skeleton_since is None:
                no_skeleton_since = now
        # 离座判定（连续无骨架 ≥ absence_thr）
        if not has_skeleton and no_skeleton_since is not None \
                and (now - no_skeleton_since) >= absence_thr:
            label = PostureLabel.ABSENCE
        else:
            label = classify(kps, _head_desk(kps, depth, desk_depth_m), thresholds)

        storage.insert_frame(sid, now, label)
        labels.append(label)

        # 提醒逻辑（followup 阶段 should_remind 内部关闭）
        if is_bad(label):
            if bad_since is None:
                bad_since = now
            if dotii_show is not None and should_remind(
                    bad_since, now, last_reminder, phase, thresholds):
                try:
                    dotii_show("fail")
                    last_reminder = now
                    reminders += 1
                except DotiiAPIError:
                    pass
        else:
            if bad_since is not None and last_label in BAD_POSTURES \
                    and dotii_show is not None:
                try:
                    dotii_show("idle")
                except DotiiAPIError:
                    pass
            bad_since = None
        last_label = label

    storage.flush()
    return daily_report(participant_id, labels, phase=phase, reminders=reminders,
                        session_start=t0, session_end=clock())
