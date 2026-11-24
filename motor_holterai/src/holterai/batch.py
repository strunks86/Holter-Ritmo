"""Procesamiento por lotes de exámenes HScribe, sin intervención del agente.

Escanea un examen o un repositorio `FDrepo`, exporta y analiza cada examen
completo, y escribe un `batch.json` con el resultado. Los exámenes parciales
se registran como error y no detienen el lote.

La ventana exportada se define con `start_s` y `seconds` (0 = registro
completo) y queda anotada tanto en el `.hea` como en el `report.json`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from holterai.ingest_hscribe import export_hscribe
from holterai.prototype import analyze_exam


def is_exam(path: Path) -> bool:
    """True si `path` es un examen standalone o un Exam_X con slots."""
    if (path / "Samples.dat").is_file():
        return True
    return any((slot / "Samples.dat").is_file() for slot in path.glob("slot_*"))


def find_exams(root: str | Path) -> list[Path]:
    """Devuelve exámenes completos bajo `root`, ordenados."""
    p = Path(root)
    if is_exam(p):
        return [p]
    if not p.is_dir():
        return []
    return [ex for ex in sorted(p.glob("Exam_*")) if ex.is_dir() and is_exam(ex)]


def tag_report_window(report: str | Path, *, start_s: float, seconds: int) -> bool:
    """Anota en el `report.json` de qué ventana del examen salió la señal.

    `analyze_exam` escribe el reporte; la ventana solo la conoce el lote, así
    que se agrega acá. Si el reporte no se puede leer, no se toca.
    """
    path = Path(report)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    payload["export"] = {"start_s": start_s, "seconds": seconds}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def process_batch(
    root: str | Path,
    *,
    dll_dir: str | Path,
    out: str | Path = "data/reports",
    interim: str | Path = "data/interim",
    seconds: int = 600,
    start_s: float = 0.0,
    limit: int = 0,
    skip_existing: bool = False,
    exporter: str | Path = "ingest/export180.exe",
) -> dict[str, Any]:
    """Procesa un examen o un repositorio y devuelve el resumen del lote."""
    out_dir = Path(out)
    interim_dir = Path(interim)
    out_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    exams = find_exams(root)
    if limit and limit > 0:
        exams = exams[:limit]

    results: list[dict[str, Any]] = []
    for exam in exams:
        name = exam.name
        report = out_dir / name / "report.json"
        if skip_existing and report.is_file():
            results.append({"exam": name, "status": "skipped", "report": str(report)})
            continue
        try:
            hea = export_hscribe(
                exam,
                interim_dir / name,
                seconds=seconds,
                start_s=start_s,
                dll_dir=dll_dir,
                exporter=exporter,
            )
            result = analyze_exam(hea, out_dir)
            report = result["files"].get("json")
            if report:
                tag_report_window(report, start_s=start_s, seconds=seconds)
            results.append(
                {
                    "exam": name,
                    "status": "ok",
                    "wfdb": str(hea),
                    "start_s": start_s,
                    "seconds": seconds,
                    "files": result["files"],
                    "n_beats": len(result["payload"].get("beats", [])),
                }
            )
        except Exception as exc:  # un examen malo no detiene el lote
            results.append(
                {
                    "exam": name,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    summary = {
        "schema": "holterai.batch/1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(Path(root)),
        "seconds": seconds,
        "start_s": start_s,
        "n_exams": len(exams),
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "errors": sum(1 for r in results if r["status"] == "error"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "results": results,
        "disclaimer": "No es un dispositivo médico; requiere revisión del cardiólogo.",
    }
    (out_dir / "batch.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
