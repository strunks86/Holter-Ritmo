"""Registro local de pacientes/exámenes y auditoría (SQLite).

No es una base clínica multiusuario: es el índice local del prototipo.
El archivo vive en `data/holterai.db` y nunca sale del equipo.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


def default_db() -> Path:
    env = os.environ.get("HOLTERIA_DB")
    if env:
        return Path(env)
    base = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    return base / "HolterIA" / "holterai.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_code TEXT UNIQUE,
    name TEXT DEFAULT '',
    sex TEXT DEFAULT '',
    age INTEGER,
    dob TEXT DEFAULT '',
    notes TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS exams (
    exam_id TEXT PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(id),
    source_path TEXT NOT NULL,
    started_at TEXT DEFAULT '',
    duration_s REAL DEFAULT 0,
    n_beats INTEGER DEFAULT 0,
    status TEXT DEFAULT 'analyzed',
    report_path TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exam_id TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_exams_patient ON exams(patient_id);
CREATE INDEX IF NOT EXISTS idx_audit_exam ON audit(exam_id);
"""


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    p = Path(db_path) if db_path else default_db()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def upsert_patient(
    *,
    patient_code: str,
    name: str = "",
    sex: str = "",
    age: int | None = None,
    dob: str = "",
    notes: str = "",
    db_path: str | Path | None = None,
) -> int:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT id FROM patients WHERE patient_code = ?", (patient_code,)
        ).fetchone()
        now = _now()
        if row:
            conn.execute(
                "UPDATE patients SET name=?, sex=?, age=?, dob=?, notes=?, updated_at=? WHERE id=?",
                (name, sex, age, dob, notes, now, row["id"]),
            )
            return int(row["id"])
        cur = conn.execute(
            "INSERT INTO patients(patient_code,name,sex,age,dob,notes,created_at,updated_at)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (patient_code, name, sex, age, dob, notes, now, now),
        )
        last = cur.lastrowid
        if last is None:
            raise RuntimeError("no se pudo insertar el paciente")
        return int(last)


def upsert_exam(
    *,
    exam_id: str,
    source_path: str,
    patient_id: int | None = None,
    started_at: str = "",
    duration_s: float = 0,
    n_beats: int = 0,
    status: str = "analyzed",
    report_path: str = "",
    db_path: str | Path | None = None,
) -> None:
    with _connect(db_path) as conn:
        now = _now()
        conn.execute(
            "INSERT INTO exams(exam_id,patient_id,source_path,started_at,duration_s,"
            "n_beats,status,report_path,created_at,updated_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(exam_id) DO UPDATE SET patient_id=excluded.patient_id,"
            " source_path=excluded.source_path, started_at=excluded.started_at,"
            " duration_s=excluded.duration_s, n_beats=excluded.n_beats,"
            " status=excluded.status, report_path=excluded.report_path,"
            " updated_at=excluded.updated_at",
            (
                exam_id,
                patient_id,
                source_path,
                started_at,
                float(duration_s),
                int(n_beats),
                status,
                report_path,
                now,
                now,
            ),
        )


def log_audit(
    exam_id: str,
    action: str,
    detail: str = "",
    db_path: str | Path | None = None,
) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO audit(exam_id,action,detail,created_at) VALUES(?,?,?,?)",
            (exam_id, action, detail[:500], _now()),
        )


def list_exams(db_path: str | Path | None = None) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT e.*, p.patient_code, p.name AS patient_name, p.sex, p.age"
            " FROM exams e LEFT JOIN patients p ON p.id = e.patient_id"
            " ORDER BY e.updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_patient(patient_code: str, db_path: str | Path | None = None) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM patients WHERE patient_code = ?", (patient_code,)
        ).fetchone()
        return dict(row) if row else None


def list_audit(
    exam_id: str | None = None,
    limit: int = 200,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        if exam_id:
            rows = conn.execute(
                "SELECT * FROM audit WHERE exam_id = ? ORDER BY id DESC LIMIT ?",
                (exam_id, max(1, min(int(limit), 2000))),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM audit ORDER BY id DESC LIMIT ?",
                (max(1, min(int(limit), 2000)),),
            ).fetchall()
        return [dict(r) for r in rows]
