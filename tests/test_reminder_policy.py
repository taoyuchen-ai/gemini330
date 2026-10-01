"""reminder_policy.should_remind 分支测试（ADR-0007）。"""
from __future__ import annotations

import pytest

from src.posture_classify import PostureLabel
from src.reminder_policy import is_bad, should_remind


THR = {
    "reminder": {
        "sustained_seconds": 30,
        "min_interval_seconds": 60,
        "track_phase_disabled": True,
    }
}


def test_is_bad_for_desk_lying_and_bend():
    assert is_bad(PostureLabel.DESK_LYING)
    assert is_bad(PostureLabel.BEND)
    assert not is_bad(PostureLabel.SIT_UPRIGHT)
    assert not is_bad(PostureLabel.LEAN_BACK)


def test_trigger_after_sustained_30s():
    assert should_remind(bad_since=0.0, now=30.0, last_reminder=None,
                         phase="intervention", thresholds=THR) is True


def test_no_trigger_before_30s():
    assert should_remind(bad_since=0.0, now=29.0, last_reminder=None,
                         phase="intervention", thresholds=THR) is False


def test_no_trigger_within_min_interval():
    assert should_remind(bad_since=0.0, now=30.0, last_reminder=29.0,
                         phase="intervention", thresholds=THR) is False


def test_trigger_after_min_interval_elapsed():
    assert should_remind(bad_since=0.0, now=90.0, last_reminder=30.0,
                         phase="intervention", thresholds=THR) is True


def test_no_trigger_when_not_bad():
    assert should_remind(bad_since=None, now=100.0, last_reminder=None,
                         phase="intervention", thresholds=THR) is False


def test_no_trigger_in_followup_phase():
    assert should_remind(bad_since=0.0, now=100.0, last_reminder=None,
                         phase="followup", thresholds=THR) is False
