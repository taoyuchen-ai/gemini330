"""Orbbec Gemini335 相机采集（ADR-0002/0007）。

pyorbbecsdk aligned depth+color，30 fps。
相机掉线重连 reconnect_attempts 次；耗尽抛 CameraReconnectError。

API 细节按 pyorbbecsdk 实际版本，部署前需现场联调确认。
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from src.exceptions import CameraReconnectError


class CameraCapture:
    def __init__(self, fps: int = 30, width: int = 640, height: int = 480,
                 reconnect_attempts: int = 3, reconnect_delay: float = 5.0):
        self.fps = fps
        self.width = width
        self.height = height
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self._pipeline = None
        self._align = None

    def start(self) -> None:
        from pyorbbecsdk import Align, Config, OBFormat, OBStream, Pipeline
        config = Config()
        config.enable_stream(OBStream.COLOR, self.width, self.height,
                             OBFormat.RGB, self.fps)
        config.enable_stream(OBStream.DEPTH, self.width, self.height,
                             OBFormat.Z16, self.fps)
        self._pipeline = Pipeline()
        self._pipeline.enable_frame_sync()
        self._pipeline.start(config)
        self._align = Align(OBStream.COLOR)  # depth 对齐到 color

    def _restart(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
        self.start()

    def grab(self, timeout_ms: int = 2000) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """返回 (color_rgb HxWx3 uint8, depth_mm HxW uint16)；失败抛 CameraReconnectError。"""
        if self._pipeline is None:
            return None, None
        last_err: Optional[Exception] = None
        for attempt in range(self.reconnect_attempts + 1):
            try:
                frameset = self._pipeline.waitForFrames(timeout_ms)
                if frameset is None:
                    raise RuntimeError("empty frameset")
                color = self._frame_to_array(frameset.colorFrame(), 3)
                depth = self._frame_to_array(frameset.depthFrame(), 1)
                if self._align is not None:
                    aligned = self._align.process(frameset)
                    depth = self._frame_to_array(aligned.depthFrame(), 1)
                return color, depth
            except Exception as e:  # 包含 OBError
                last_err = e
                if attempt >= self.reconnect_attempts:
                    raise CameraReconnectError(
                        f"camera reconnect failed after {self.reconnect_attempts} attempts: {last_err}"
                    )
                time.sleep(self.reconnect_delay)
                self._restart()
        return None, None  # unreachable

    @staticmethod
    def _frame_to_array(frame, channels: int) -> Optional[np.ndarray]:
        if frame is None:
            return None
        data = frame.data() if callable(getattr(frame, "data", None)) else getattr(frame, "data", None)
        if data is None:
            return None
        h, w = frame.height(), frame.width()
        dtype = np.uint8 if channels == 3 else np.uint16
        arr = np.frombuffer(bytes(data), dtype=dtype)
        if channels == 3:
            arr = arr.reshape(h, w, 3)
        else:
            arr = arr.reshape(h, w)
        return np.ascontiguousarray(arr)

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        self._align = None
