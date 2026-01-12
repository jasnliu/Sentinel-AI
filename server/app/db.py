from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class DbPaths:
    db_path: str
    images_dir: str


def get_paths(data_dir: str) -> DbPaths:
    db_path = os.path.join(data_dir, "alerts.sqlite3")
    images_dir = os.path.join(data_dir, "images")
    return DbPaths(db_path=db_path, images_dir=images_dir)


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          device_id TEXT NOT NULL,
          timestamp_ms INTEGER NOT NULL,
          received_ms INTEGER NOT NULL,
          confidence REAL NOT NULL,
          consecutive_hits INTEGER NOT NULL,
          lat REAL,
          lon REAL,
          image_path TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_received_ms ON alerts(received_ms)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_device_id ON alerts(device_id)")
    conn.commit()


def insert_alert(
    conn: sqlite3.Connection,
    *,
    device_id: str,
    timestamp_ms: int,
    received_ms: int,
    confidence: float,
    consecutive_hits: int,
    lat: Optional[float],
    lon: Optional[float],
) -> int:
    cur = conn.execute(
        """
        INSERT INTO alerts(device_id, timestamp_ms, received_ms, confidence, consecutive_hits, lat, lon, image_path)
        VALUES(?, ?, ?, ?, ?, ?, ?, NULL)
        """,
        (device_id, timestamp_ms, received_ms, confidence, consecutive_hits, lat, lon),
    )
    conn.commit()
    return int(cur.lastrowid)


def set_image_path(conn: sqlite3.Connection, alert_id: int, image_path: str) -> None:
    conn.execute("UPDATE alerts SET image_path = ? WHERE id = ?", (image_path, alert_id))
    conn.commit()


def fetch_alert(conn: sqlite3.Connection, alert_id: int) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
    if row is None:
        return None
    return dict(row)


def list_alerts(conn: sqlite3.Connection, limit: int) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM alerts ORDER BY received_ms DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def delete_alerts_older_than(conn: sqlite3.Connection, cutoff_received_ms: int) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM alerts WHERE received_ms < ?",
        (cutoff_received_ms,),
    ).fetchall()
    conn.execute("DELETE FROM alerts WHERE received_ms < ?", (cutoff_received_ms,))
    conn.commit()
    return [dict(r) for r in rows]

