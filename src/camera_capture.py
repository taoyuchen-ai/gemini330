"""Orbbec Gemini335 相机采集（ADR-0002/0007）。

pyorbbecsdk V2 API（pip 包名 pyorbbecsdk2，import 名仍是 pyorbbecsdk）。
profile-based Config + HW_MODE D2C 对齐，30 fps。
相机掉线重连 reconnect_attempts 次；耗尽抛 CameraReconnectError。

实测依据（真机 Gemini335, FW 1.4.60, USB3.2）：
  - color 640x480 RGB@30 原生支持（profile [203]）
  - depth 640x480 Y16@30 原生支持（profile [15]）；depth_scale=1.0 mm/unit
  - HW_MODE D2C 已对齐 depth 到 color 视角（depth 与 color 同尺寸），
    无需软件 AlignFilter
  - wait_for_frames 接受位置参数 int（非 timeout_ms= 关键字）
  - frame.get_data() 直接返回 ndarray；兼容 memoryview/bytes 路径
  - depth Y16 值即毫米（不是 Z16；代码历史版本请求 Z16 是错的）

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
        self._config = None

    def start(self) -> None:
        from pyorbbecsdk import (
            Config, OBAlignMode, OBFormat, OBSensorType, Pipeline,
        )
        pipe = Pipeline()
        color_profile = self._find_profile(
            pipe, OBSensorType.COLOR_SENSOR,
            self.width, self.height, OBFormat.RGB, self.fps)
        if color_profile is None:
            raise RuntimeError(
                f"color profile {self.width}x{self.height} RGB@{self.fps} 不支持")
        depth_profile = self._find_profile(
            pipe, OBSensorType.DEPTH_SENSOR,
            self.width, self.height, OBFormat.Y16, self.fps)
        if depth_profile is None:
            raise RuntimeError(
                f"depth profile {self.width}x{self.height} Y16@{self.fps} 不支持")

        config = Config()
        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)
        # D2C 对齐：HW_MODE 优先（实测可用，硬件对齐）；失败 fallback SW_MODE
        try:
            config.set_align_mode(OBAlignMode.HW_MODE)
        except Exception:
            config.set_align_mode(OBAlignMode.SW_MODE)

        pipe.enable_frame_sync()
        pipe.start(config)
        self._pipeline = pipe
        self._config = config

    @staticmethod
    def _find_profile(pipe, sensor_type, width, height, fmt, fps):
        """从 pipe.get_stream_profile_list(sensor_type) 找匹配的 video profile。"""
        profiles = pipe.get_stream_profile_list(sensor_type)
        n = profiles.get_count()
        for i in range(n):
            p = profiles.get_stream_profile_by_index(i)
            if not p.is_video_stream_profile():
                continue
            vp = p.as_video_stream_profile()
            if (vp.get_width() == width and vp.get_height() == height
                    and vp.get_format() == fmt and vp.get_fps() == fps):
                return vp
        return None

    def _restart(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
        self.start()

    def grab(self, timeout_ms: int = 2000) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """返回 (color_rgb HxWx3 uint8, depth_mm HxW uint16)；失败抛 CameraReconnectError。

        depth 值即毫米（Y16 格式，depth_scale=1.0 mm/unit）。
        HW_MODE D2C 已对齐 depth 到 color 视角，无需软件 AlignFilter。
        """
        if self._pipeline is None:
            return None, None
        last_err: Optional[Exception] = None
        for attempt in range(self.reconnect_attempts + 1):
            try:
                frameset = self._pipeline.wait_for_frames(timeout_ms)
                if frameset is None:
                    raise RuntimeError("empty frameset")
                color = self._frame_to_array(frameset.get_color_frame(), 3)
                depth = self._frame_to_array(frameset.get_depth_frame(), 1)
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
        # V2: frame.get_data() 通常直接返回 ndarray（实测）；
        # 兼容 memoryview/bytes 路径以防不同版本行为差异。
        data = frame.get_data()
        if data is None:
            return None
        dtype = np.uint8 if channels == 3 else np.uint16
        if isinstance(data, np.ndarray):
            arr = data if data.dtype == dtype else data.view(dtype).ravel()
        else:
            arr = np.frombuffer(bytes(data), dtype=dtype)
        h, w = frame.get_height(), frame.get_width()
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
        self._config = None
