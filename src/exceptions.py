"""异常定义（ADR-0007）。

相机重连 / 离座 / Dotii API 失联 / MediaPipe 超时 / 数据缺失。
"""


class Gemini330Error(Exception):
    """所有 gemini330 异常基类。"""


class CameraReconnectError(Gemini330Error):
    """相机重连 3 次仍失败。"""


class AbsenceDetectedError(Gemini330Error):
    """连续无骨架 30 秒，判定离座。"""


class DotiiAPIError(Gemini330Error):
    """Dotii-Display HTTP 接口失联（超时/重试耗尽）。"""


class MediaPipeTimeoutError(Gemini330Error):
    """MediaPipe 推理超时，本帧跳过。"""


class MissingDataError(Gemini330Error):
    """缺失帧比例超过阈值，会话整体剔除。"""
