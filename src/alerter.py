"""PI 报警机制抽象（ADR-0007）。

生产部署时子类化绑定具体通道（飞书/邮件/日志聚合等）；
默认 NoopAlerter 静默丢弃，保证开发与测试不被外部依赖阻塞。
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Alerter(ABC):
    """报警器抽象基类。"""

    @abstractmethod
    def alert(self, kind: str, message: str) -> None:
        """发送一条报警。

        Args:
            kind: 报警类型（如 camera_reconnect_exhausted / dotii_failure_threshold）。
            message: 报警内容（人可读）。
        """


class NoopAlerter(Alerter):
    """静默 alerter：丢弃所有报警。

    生产环境通过子类化绑定具体通道；本期骨架到位即满足 ADR-0007 要求。
    """

    def alert(self, kind: str, message: str) -> None:  # noqa: D401
        return None
