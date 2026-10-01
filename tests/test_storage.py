"""Storage 的建表/插入/落盘测试（ADR-0007）。"""
from __future__ import annotations

import pytest

from src.posture_classify import PostureLabel
from src.storage import Storage


def test_wal_mode(tmp_path):
    s = Storage(str(tmp_path / "t.db"))
    mode = s.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    s.close()


def test_create_session_returns_id(tmp_path):
    s = Storage(str(tmp_path / "t.db"))
    sid = s.create_session("p01", "baseline", start_time=0.0)
    assert isinstance(sid, int) and sid > 0
    s.close()


def test_buffer_not_visible_before_flush(tmp_path):
    s = Storage(str(tmp_path / "t.db"), flush_interval_seconds=9999)
    sid = s.create_session("p01", "baseline", start_time=0.0)
    s.insert_frame(sid, 1.0, PostureLabel.BEND, 0.15)
    assert s.count_frames_by_label(sid) == {}
    s.close()


def test_flush_persists_frames(tmp_path):
    s = Storage(str(tmp_path / "t.db"), flush_interval_seconds=9999)
    sid = s.create_session("p01", "baseline", start_time=0.0)
    s.insert_frame(sid, 1.0, PostureLabel.BEND, 0.15)
    s.insert_frame(sid, 2.0, PostureLabel.SIT_UPRIGHT, 0.35)
    s.insert_frame(sid, 3.0, PostureLabel.DESK_LYING, 0.10)
    s.flush()
    counts = s.count_frames_by_label(sid)
    assert counts == {"bend": 1, "sit_upright": 1, "desk_lying": 1}
    s.close()


def test_close_flushes_remaining_buffer(tmp_path):
    s = Storage(str(tmp_path / "t.db"), flush_interval_seconds=9999)
    sid = s.create_session("p01", "baseline", start_time=0.0)
    s.insert_frame(sid, 1.0, PostureLabel.BEND)
    s.close()
    s2 = Storage(str(tmp_path / "t.db"))
    assert s2.count_frames_by_label(sid) == {"bend": 1}
    s2.close()
