"""Alerter 抽象接口与 NoopAlerter 测试（ADR-0007 PI 报警机制）。"""
from __future__ import annotations

from src.alerter import Alerter, NoopAlerter


def test_noop_alerter_can_be_instantiated():
    """NoopAlerter 可直接实例化。"""
    a = NoopAlerter()
    assert isinstance(a, Alerter)


def test_noop_alerter_alert_does_not_raise():
    """NoopAlerter.alert 不抛异常（生产部署时子类化绑定具体通道）。"""
    a = NoopAlerter()
    a.alert("camera_reconnect_exhausted", "相机重连耗尽")
    a.alert("dotii_failure_threshold", "Dotii 连续失败 5 次")


class FakeAlerter(Alerter):
    """测试用 alerter：收集所有调用。"""

    def __init__(self):
        self.alerts: list[tuple[str, str]] = []

    def alert(self, kind: str, message: str) -> None:
        self.alerts.append((kind, message))


def test_fake_alerter_collects_calls():
    """FakeAlerter 收集 alert 调用供 pipeline 测试断言。"""
    a = FakeAlerter()
    a.alert("k1", "m1")
    a.alert("k2", "m2")
    assert a.alerts == [("k1", "m1"), ("k2", "m2")]
