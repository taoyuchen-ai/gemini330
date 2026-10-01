"""Dotii-Display HTTP 触发（ADR-0004）。

向 ESP32-S3-Touch-AMOLED-1.75（默认 127.0.0.1:8787）发 HTTP 请求
切换屏幕表情：不良时 fail 表情，恢复时 idle。
超时 1 秒，重试 1 次（ADR-0007）；连续失联转异常。
"""
from __future__ import annotations

from typing import Optional

import requests

from src.exceptions import DotiiAPIError

DEFAULT_BASE_URL = "http://127.0.0.1:8787"
DEFAULT_TIMEOUT = 1
DEFAULT_RETRIES = 1


def trigger_expression(expression: str,
                       base_url: str = DEFAULT_BASE_URL,
                       timeout: int = DEFAULT_TIMEOUT,
                       retries: int = DEFAULT_RETRIES) -> bool:
    """触发 Dotii-Display 表情。

    Args:
        expression: 'fail' / 'idle' / 'smile' 等。
        base_url: Dotii HTTP 基址。
        timeout: 单次请求超时（秒）。
        retries: 失败重试次数（总请求 = retries + 1）。

    Returns:
        True 表示成功。失败抛 DotiiAPIError。
    """
    last_err: Optional[Exception] = None
    url = f"{base_url}/api/face"
    for _ in range(retries + 1):
        try:
            r = requests.post(url, json={"expression": expression},
                               timeout=timeout)
            r.raise_for_status()
            return True
        except requests.RequestException as e:
            last_err = e
    raise DotiiAPIError(f"Dotii API failed for '{expression}': {last_err}")


def show_fail(base_url: str = DEFAULT_BASE_URL, **kw) -> bool:
    """不良提醒：fail 表情。"""
    return trigger_expression("fail", base_url=base_url, **kw)


def show_idle(base_url: str = DEFAULT_BASE_URL, **kw) -> bool:
    """恢复：idle 表情。"""
    return trigger_expression("idle", base_url=base_url, **kw)
