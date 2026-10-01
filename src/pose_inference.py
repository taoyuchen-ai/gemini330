"""MediaPipe BlazePose 33 关键点推理（ADR-0005）。

GPU 推理 30 fps；单帧超 timeout_ms 抛 MediaPipeTimeoutError 并跳过本帧。
不存图像，仅返回 33 关键点 (x,y,z,visibility)。
"""
from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from src.exceptions import MediaPipeTimeoutError

NUM_LANDMARKS = 33


class PoseInference:
    def __init__(self, model_complexity: int = 1,
                 timeout_ms: int = 100,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.model_complexity = model_complexity
        self.timeout_ms = timeout_ms
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._pose = None
        self._mp = None

    def start(self) -> None:
        import mediapipe as mp
        self._mp = mp
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def infer(self, color_rgb: Optional[np.ndarray]) -> Sequence[Optional[tuple]]:
        """推理一帧。

        Args:
            color_rgb: HxWx3 uint8 RGB 图像（MediaPipe 要求 RGB）。

        Returns:
            长度 33 list，每元素 (x,y,z,visibility) 或 None。
            超时/失败抛 MediaPipeTimeoutError（调用方应跳过本帧）。
        """
        if self._pose is None or color_rgb is None:
            return [None] * NUM_LANDMARKS
        start = time.perf_counter()
        try:
            result = self._pose.process(color_rgb)
        except Exception as e:
            raise MediaPipeTimeoutError(f"mediapipe inference failed: {e}")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms > self.timeout_ms:
            raise MediaPipeTimeoutError(
                f"mediapipe timeout: {elapsed_ms:.0f}ms > {self.timeout_ms}ms"
            )
        if result.pose_landmarks is None:
            return [None] * NUM_LANDMARKS
        kps: list[Optional[tuple]] = []
        for lm in result.pose_landmarks.landmark:
            kps.append((lm.x, lm.y, lm.z, lm.visibility))
        return kps

    def close(self) -> None:
        if self._pose is not None:
            try:
                self._pose.close()
            except Exception:
                pass
            self._pose = None
        self._mp = None
