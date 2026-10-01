"""SQLite 落盘（ADR-0007）。

WAL 模式；每分钟 flush 一次缓冲帧。
不存图像，仅骨架坐标衍生标签 + 头-桌面距离。
"""
from __future__ import annotations

import sqlite3
import time
from typing import Optional

from src.posture_classify import PostureLabel


class Storage:
    def __init__(self, db_path: str, flush_interval_seconds: int = 60):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()
        self._flush_interval = flush_interval_seconds
        self._last_flush = time.time()
        self._buffer: list[tuple] = []

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY,
                participant_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                start_time REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS frames (
                session_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                label TEXT NOT NULL,
                head_desk_distance REAL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );
            CREATE INDEX IF NOT EXISTS idx_frames_session
                ON frames(session_id);
            """
        )
        self.conn.commit()

    def create_session(self, participant_id: str, phase: str,
                       start_time: Optional[float] = None) -> int:
        t = start_time if start_time is not None else time.time()
        cur = self.conn.execute(
            "INSERT INTO sessions(participant_id, phase, start_time) VALUES(?,?,?)",
            (participant_id, phase, t),
        )
        self.conn.commit()
        return cur.lastrowid

    def insert_frame(self, session_id: int, timestamp: float,
                     label: PostureLabel,
                     head_desk_distance: Optional[float] = None) -> None:
        self._buffer.append(
            (session_id, timestamp, label.value, head_desk_distance)
        )
        if time.time() - self._last_flush >= self._flush_interval:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            self._last_flush = time.time()
            return
        self.conn.executemany(
            "INSERT INTO frames(session_id, timestamp, label, head_desk_distance) "
            "VALUES(?,?,?,?)",
            self._buffer,
        )
        self.conn.commit()
        self._buffer.clear()
        self._last_flush = time.time()

    def count_frames_by_label(self, session_id: int) -> dict[str, int]:
        cur = self.conn.execute(
            "SELECT label, COUNT(*) FROM frames WHERE session_id=? GROUP BY label",
            (session_id,),
        )
        return {row[0]: row[1] for row in cur.fetchall()}

    def close(self) -> None:
        self.flush()
        self.conn.close()
