"""dotii_reminder 的 HTTP 触发测试（ADR-0004/0007）。

用 monkeypatch 替换 requests.post，避免真实网络。
"""
from __future__ import annotations

import pytest

from src import dotii_reminder
from src.dotii_reminder import show_fail, show_idle, trigger_expression
from src.exceptions import DotiiAPIError


class _FakeResponse:
    def __init__(self, status=200):
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}")


def test_trigger_success(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append((url, json))
        return _FakeResponse(200)

    monkeypatch.setattr(dotii_reminder.requests, "post", fake_post)
    assert trigger_expression("fail") is True
    assert calls[0][1] == {"expression": "fail"}


def test_trigger_retries_then_raises(monkeypatch):
    import requests

    def fake_post(url, json=None, timeout=None):
        raise requests.Timeout("timeout")

    monkeypatch.setattr(dotii_reminder.requests, "post", fake_post)
    with pytest.raises(DotiiAPIError):
        trigger_expression("fail", retries=1)


def test_trigger_succeeds_on_retry(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            import requests
            raise requests.Timeout("timeout")
        return _FakeResponse(200)

    monkeypatch.setattr(dotii_reminder.requests, "post", fake_post)
    assert trigger_expression("idle", retries=1) is True
    assert len(calls) == 2


def test_show_fail_and_idle_helpers(monkeypatch):
    received = []

    def fake_post(url, json=None, timeout=None):
        received.append(json["expression"])
        return _FakeResponse(200)

    monkeypatch.setattr(dotii_reminder.requests, "post", fake_post)
    assert show_fail() is True
    assert show_idle() is True
    assert received == ["fail", "idle"]
