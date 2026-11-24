"""Exportación de un reporte a CSV/JSON para revisión o análisis externo."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def export_report(
    report_json: str | Path,
    out_dir: str | Path,
    formats: tuple[str, ...] = ("csv", "json"),
) -> dict[str, str]:
    """Exporta beats/eventos/resumen. No modifica el reporte original."""
    src = Path(report_json)
    payload = json.loads(src.read_text(encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}

    beats = payload.get("beats", []) or []
    beat_rows = [
        {
            "idx": b.get("idx"),
            "t_s": b.get("t"),
            "clase": b.get("cls"),
            "conf": b.get("conf"),
            "rr_prev_s": b.get("rr_prev"),
            "rr_next_s": b.get("rr_next"),
            "editado": b.get("edited", False),
        }
        for b in beats
    ]
    events = payload.get("events", []) or []
    event_rows = [
        {
            "tipo": e.get("kind"),
            "inicio_s": e.get("start_s"),
            "fin_s": e.get("end_s"),
            "duracion_s": e.get("duration_s"),
            "severidad": e.get("severity"),
            "detalle": e.get("detail"),
        }
        for e in events
    ]

    if "csv" in formats:
        bp = out / "beats.csv"
        _write_csv(
            bp,
            beat_rows,
            ["idx", "t_s", "clase", "conf", "rr_prev_s", "rr_next_s", "editado"],
        )
        ep = out / "events.csv"
        _write_csv(
            ep,
            event_rows,
            ["tipo", "inicio_s", "fin_s", "duracion_s", "severidad", "detalle"],
        )
        files["beats_csv"] = str(bp)
        files["events_csv"] = str(ep)

    if "json" in formats:
        summary = {
            "exam_id": payload.get("exam_id"),
            "schema": payload.get("schema"),
            "summary": payload.get("summary", {}),
            "counts": payload.get("counts", {}),
            "quality": payload.get("quality", {}),
            "hscribe": payload.get("hscribe", {}),
            "validation": payload.get("validation", {}),
            "edits": payload.get("edits", {}),
            "disclaimer": payload.get("disclaimer"),
        }
        sp = out / "summary.json"
        sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        files["summary_json"] = str(sp)

    return files
