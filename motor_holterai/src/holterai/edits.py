"""Persistencia local de correcciones de latidos y notas por examen.

El sidecar vive junto al WFDB: `<examen>.edits.json`. No contiene datos
personales; solo índices de muestra, clases y notas escritas por el usuario.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

VALID = {"N", "S", "V", "F", "Q"}


def edits_path(exam: str | Path) -> Path:
    env = os.environ.get("HOLTERIA_EDITS_DIR")
    if env:
        return Path(env) / (Path(exam).stem + ".edits.json")
    p = Path(exam)
    return p.with_suffix(".edits.json")


def load_edits(exam: str | Path) -> dict[str, Any]:
    p = edits_path(exam)
    if not p.is_file():
        return {"beats": {}, "deleted": [], "inserted": [], "notes": "", "history": []}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"beats": {}, "deleted": [], "inserted": [], "notes": "", "history": []}
    if not isinstance(raw, dict):
        return {"beats": {}, "deleted": [], "inserted": [], "notes": "", "history": []}
    data = cast(dict[str, Any], raw)
    if not isinstance(data.get("beats"), dict):
        data["beats"] = {}
    data.setdefault("deleted", [])
    data.setdefault("inserted", [])
    data.setdefault("notes", "")
    data.setdefault("history", [])
    return data


def save_edits(exam: str | Path, analysis: Any, history: list[Any] | None = None) -> Path:
    """Guarda solo latidos editados y notas; devuelve el sidecar."""
    beats = {
        str(b.idx): b.cls
        for b in getattr(analysis, "beats", [])
        if getattr(b, "edited", False) and b.cls in VALID
    }
    payload = {
        "schema": "holterai.edits/1",
        "beats": beats,
        "deleted": list(getattr(analysis, "deleted_idxs", [])),
        "inserted": list(getattr(analysis, "inserted_beats", [])),
        "events": [e.to_dict() for e in getattr(analysis, "events", [])],
        "notes": getattr(analysis, "notes", "") or "",
        "diary": getattr(analysis, "diary", "") or "",
        "history": history or [],
    }
    p = edits_path(exam)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def apply_edits(analysis: Any, exam: str | Path) -> dict[str, Any]:
    """Aplica el sidecar a un Analysis recién calculado."""
    data = load_edits(exam)
    deleted = {int(x) for x in data.get("deleted", []) if str(x).lstrip("-").isdigit()}
    if deleted:
        analysis.beats = [b for b in getattr(analysis, "beats", []) if b.idx not in deleted]
    inserted = 0
    for raw in data.get("inserted", []):
        try:
            idx = int(raw["idx"])
            t = float(raw["t"])
            cls = str(raw["cls"])
        except (KeyError, TypeError, ValueError):
            continue
        if cls not in VALID:
            continue
        from holterai.legacy.app.core.models import Beat

        analysis.beats.append(Beat(idx=idx, t=t, cls=cls, conf=0.0, edited=True))
        inserted += 1
    if inserted:
        analysis.beats.sort(key=lambda b: b.t)
        for i, b in enumerate(analysis.beats):
            b.rr_prev = b.t - analysis.beats[i - 1].t if i else 0.0
            b.rr_next = analysis.beats[i + 1].t - b.t if i + 1 < len(analysis.beats) else 0.0
    by_idx = {b.idx: b for b in getattr(analysis, "beats", [])}
    changed = 0
    for raw_idx, cls in data.get("beats", {}).items():
        try:
            idx = int(raw_idx)
        except ValueError:
            continue
        b = by_idx.get(idx)
        if b is not None and cls in VALID and b.cls != cls:
            b.cls = cls
            b.edited = True
            changed += 1
    if data.get("notes"):
        analysis.notes = str(data["notes"])
    if data.get("diary"):
        analysis.diary = str(data["diary"])
    if data.get("events") is not None:
        from holterai.legacy.app.core.models import Event

        events = []
        for raw_event in data.get("events", []):
            if not isinstance(raw_event, dict):
                continue
            payload_event = {
                k: raw_event[k]
                for k in ("kind", "start_s", "end_s", "severity", "detail", "beats")
                if k in raw_event
            }
            try:
                events.append(Event(**payload_event))
            except (TypeError, ValueError):
                continue
        analysis.events = events
    return {
        "changed": changed,
        "deleted": len(deleted),
        "inserted": inserted,
        "notes": bool(data.get("notes")),
        "path": str(edits_path(exam)),
    }
