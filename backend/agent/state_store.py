"""SQLite shared store for the PRAETOR information layer (Builder B).

Implements the StateStore contract in PIPELINE_CONTRACT.md exactly, plus one
service-layer extra: `save_reading()` — TickResult carries no raw signals, so
the service loop persists each verified PinnReading payload here and
`get_sensor_history()` serves signals+pinn from that table (oldest → newest).

stdlib sqlite3 only. One connection, check_same_thread=False, guarded by a
threading.Lock — good enough for a demo loop + a handful of API readers.
Pydantic objects are stored as JSON columns next to indexed scalar columns.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional, Union

from .config import settings
from .schemas import Advisory, AdvisoryStatus, OperatorOverride, TickResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id  TEXT NOT NULL,
    department  TEXT NOT NULL,
    epoch       INTEGER NOT NULL,
    at          REAL NOT NULL,
    signals     TEXT NOT NULL,
    pinn        TEXT NOT NULL,
    note        TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_readings_machine ON readings(machine_id, id);
CREATE INDEX IF NOT EXISTS ix_readings_epoch   ON readings(epoch);

CREATE TABLE IF NOT EXISTS ticks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id  TEXT NOT NULL,
    department  TEXT NOT NULL,
    epoch       INTEGER NOT NULL,
    risk        TEXT NOT NULL DEFAULT '',
    at          REAL NOT NULL,
    json        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ticks_machine ON ticks(machine_id, id);
CREATE INDEX IF NOT EXISTS ix_ticks_dept    ON ticks(department, id);
CREATE INDEX IF NOT EXISTS ix_ticks_epoch   ON ticks(epoch);

CREATE TABLE IF NOT EXISTS advisories (
    id          TEXT PRIMARY KEY,
    machine_id  TEXT NOT NULL,
    department  TEXT NOT NULL,
    epoch       INTEGER NOT NULL,
    severity    TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  REAL NOT NULL,
    json        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_adv_status  ON advisories(status, created_at);
CREATE INDEX IF NOT EXISTS ix_adv_machine ON advisories(machine_id);

CREATE TABLE IF NOT EXISTS overrides (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    advisory_id TEXT NOT NULL,
    decision    TEXT NOT NULL,
    at          REAL NOT NULL,
    json        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ovr_advisory ON overrides(advisory_id);

CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    kind    TEXT NOT NULL,
    at      REAL NOT NULL,
    json    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_kind ON events(kind, id);
"""


def _status_value(status: Union[AdvisoryStatus, str, None]) -> Optional[str]:
    if status is None:
        return None
    return status.value if isinstance(status, AdvisoryStatus) else str(status)


class StateStore:
    """Thread-safe SQLite store shared by pipeline, API and operator agent."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._path = db_path or settings.db_path
        self._lock = threading.Lock()
        try:
            self._conn = self._open(self._path)
        except sqlite3.OperationalError as e:
            # Some mounted/network filesystems refuse SQLite locking ("disk
            # I/O error"). Demo must not die on stage — fall back to tempdir.
            fallback = str(Path(tempfile.gettempdir()) / "praetor_state.db")
            print(f"[state_store] cannot open {self._path} ({e}); "
                  f"falling back to {fallback}")
            self._path = fallback
            self._conn = self._open(fallback)

    @staticmethod
    def _open(path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        conn.commit()
        return conn

    # ------------------------------------------------------------- lifecycle
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------- raw reads
    def save_reading(self, payload: dict[str, Any]) -> None:
        """Persist one verified PinnReading payload (service-layer extra —
        TickResult has no signals, so sensor history lives here)."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO readings(machine_id, department, epoch, at, signals, pinn, note)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    str(payload.get("machine_id", "?")),
                    str(payload.get("department", "?")),
                    int(payload.get("epoch", -1)),
                    float(payload.get("timestamp", time.time())),
                    json.dumps(payload.get("signals", {})),
                    json.dumps(payload.get("pinn", {})),
                    str(payload.get("note", "")),
                ),
            )
            self._conn.commit()

    # ------------------------------------------------------------- contract
    def save_tick(self, tick: TickResult) -> None:
        risk = ""
        if tick.triage is not None:
            risk = tick.triage.risk.value
        elif tick.rejected_reason:
            risk = "REJECTED"
        with self._lock:
            self._conn.execute(
                "INSERT INTO ticks(machine_id, department, epoch, risk, at, json)"
                " VALUES (?,?,?,?,?,?)",
                (tick.machine_id, tick.department, tick.epoch, risk,
                 time.time(), tick.model_dump_json()),
            )
            self._conn.commit()

    def save_advisory(self, adv: Advisory) -> None:
        """Upsert by id — the pipeline saves once, override handlers re-save."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO advisories(id, machine_id, department, epoch, severity,"
                " status, created_at, json) VALUES (?,?,?,?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET"
                " severity=excluded.severity, status=excluded.status, json=excluded.json",
                (adv.id, adv.machine_id, adv.department, adv.epoch,
                 adv.severity.value, adv.status.value, adv.created_at,
                 adv.model_dump_json()),
            )
            self._conn.commit()

    def save_override(self, ov: OperatorOverride) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO overrides(advisory_id, decision, at, json) VALUES (?,?,?,?)",
                (ov.advisory_id, ov.decision, ov.at, ov.model_dump_json()),
            )
            self._conn.commit()

    def get_advisories(self, status: Union[AdvisoryStatus, str, None] = None,
                       limit: int = 50) -> list[Advisory]:
        sv = _status_value(status)
        with self._lock:
            if sv:
                rows = self._conn.execute(
                    "SELECT json FROM advisories WHERE status=?"
                    " ORDER BY created_at DESC, rowid DESC LIMIT ?", (sv, limit)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT json FROM advisories ORDER BY created_at DESC, rowid DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [Advisory.model_validate_json(r["json"]) for r in rows]

    def get_advisory(self, advisory_id: str) -> Optional[Advisory]:
        with self._lock:
            row = self._conn.execute(
                "SELECT json FROM advisories WHERE id=?", (advisory_id,)
            ).fetchone()
        return Advisory.model_validate_json(row["json"]) if row else None

    def get_sensor_history(self, machine_id: str, limit: int = 40) -> list[dict]:
        """Signals + pinn dicts of the most recent readings, oldest → newest."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT machine_id, department, epoch, at, signals, pinn, note"
                " FROM readings WHERE machine_id=? ORDER BY id DESC LIMIT ?",
                (machine_id, limit),
            ).fetchall()
        out = [
            {
                "machine_id": r["machine_id"],
                "department": r["department"],
                "epoch": r["epoch"],
                "at": r["at"],
                "signals": json.loads(r["signals"]),
                "pinn": json.loads(r["pinn"]),
                "note": r["note"],
            }
            for r in rows
        ]
        out.reverse()  # oldest → newest
        return out

    def get_overrides(self, limit: int = 20) -> list[OperatorOverride]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT json FROM overrides ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [OperatorOverride.model_validate_json(r["json"]) for r in rows]

    def department_snapshot(self) -> dict[str, dict]:
        """Per-department view of the LAST tick: risk label, epoch, machine."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT department, machine_id, epoch, risk, at FROM ticks"
                " WHERE id IN (SELECT MAX(id) FROM ticks GROUP BY department)"
            ).fetchall()
        return {
            r["department"]: {
                "machine_id": r["machine_id"],
                "risk": r["risk"] or "UNKNOWN",
                "epoch": r["epoch"],
                "at": r["at"],
            }
            for r in rows
        }

    def log_event(self, kind: str, data: dict) -> None:
        """Flight-recorder row — everything the SSE hub publishes lands here too."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO events(kind, at, json) VALUES (?,?,?)",
                (kind, time.time(), json.dumps(data, default=str)),
            )
            self._conn.commit()

    def get_events(self, kind: Optional[str] = None, limit: int = 100) -> list[dict]:
        with self._lock:
            if kind:
                rows = self._conn.execute(
                    "SELECT kind, at, json FROM events WHERE kind=? ORDER BY id DESC LIMIT ?",
                    (kind, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT kind, at, json FROM events ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
        return [{"kind": r["kind"], "at": r["at"], "data": json.loads(r["json"])} for r in rows]
