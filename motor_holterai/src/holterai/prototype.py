"""Prototipo funcional end-to-end sobre el pipeline legado portado.

No es un dispositivo médico: genera un borrador de reporte para revisión.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from holterai.edits import apply_edits
from holterai.hscribe_beats import (
    crop_reference,
    match_reference,
    read_beats,
    start_s_from_header,
    summarize,
)
from holterai.legacy.app.pipeline import Options, analyze
from holterai.legacy.app.report import report as legacy_report
from holterai.legacy.loader import load_wfdb16
from holterai.report_basic import write_html, write_index, write_pdf, write_strip
from holterai.report_config import load_report_config

DISCLAIMER = (
    "Borrador automático de apoyo a la lectura. No es un dispositivo médico, "
    "no emite diagnóstico y requiere revisión del cardiólogo."
)


def analyze_exam(
    exam: str | Path,
    out_dir: str | Path = "data/reports",
    formats: tuple[str, ...] = ("docx", "html", "pdf"),
) -> dict[str, Any]:
    """Analiza un WFDB y escribe report.json + reportes legibles."""
    wfdb_path = Path(exam)
    out = Path(out_dir) / wfdb_path.stem
    out.mkdir(parents=True, exist_ok=True)

    rec = load_wfdb16(wfdb_path)
    try:
        from holterai.config import load_criteria

        qtc_formula = load_criteria().qtc_formula
    except Exception:  # sin criterios legibles se usa Bazett
        qtc_formula = "bazett"
    opts = Options(
        pause_s=2.0,
        brady_bpm=50.0,
        tachy_bpm=120.0,
        af_threshold=0.55,
        af_min_s=30.0,
        use_ecgfounder=False,
        pacing_mode="auto",
        crt_mode="auto",
        qtc_formula=qtc_formula,
    )
    an = analyze(rec, opts)
    an.notes = DISCLAIMER
    applied = apply_edits(an, wfdb_path)

    payload = an.to_dict(include_beats=True)
    payload["schema"] = "holterai.prototype/1"
    payload["exam_id"] = wfdb_path.stem
    payload["software"] = {"name": "HolterIA", "version": "0.1.0-prototype"}
    payload["generated_at"] = datetime.now().isoformat(timespec="seconds")
    payload["disclaimer"] = DISCLAIMER
    payload["edits"] = applied
    # Hasta calibrar las clases con etiquetas del hospital, no se publican
    # eventos clínicos (VT/FA/bigeminismo): el modelo sobreclasifica V.
    payload["events"] = []
    payload["events_note"] = "Eventos clínicos suprimidos: la clasificación no está validada."
    beats_sidecar = wfdb_path.with_suffix(".beats.dat")
    if beats_sidecar.is_file():
        try:
            start_s = start_s_from_header(wfdb_path)
            all_beats = read_beats(beats_sidecar)
            window = crop_reference(
                all_beats,
                start_s * 1000.0 - 200.0,
                (start_s + rec.duration_s) * 1000.0 + 200.0,
            )
            payload["hscribe"] = summarize(window)
            payload["hscribe"]["exam_total_n"] = len(all_beats)
            payload["hscribe"]["start_s"] = start_s
            # Se/PPV/F1 solo en las ventanas de 60 s donde HScribe tiene
            # referencia; el tiempo sin referencia se informa aparte.
            payload["validation"] = match_reference(
                window, an.beats, start_s=start_s, duration_s=rec.duration_s
            )
        except Exception as exc:  # el reporte no debe caerse por metadata
            payload["hscribe"] = {"error": f"{type(exc).__name__}: {exc}"}
    report_json = out / "report.json"
    report_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    files: dict[str, str] = {"json": str(report_json)}
    try:
        generated = legacy_report.generate(an, formats=list(formats))
        for key in ("docx", "html", "pdf"):
            src = generated.get(key)
            if not src:
                continue
            suffix = Path(src).suffix
            dst = out / f"report{suffix}"
            shutil.copyfile(src, dst)
            files[key] = str(dst)
    except Exception as exc:  # el reporte JSON no debe perderse
        files["report_error"] = f"{type(exc).__name__}: {exc}"

    strip = write_strip(
        rec.samples,
        rec.fs,
        [b.t for b in an.beats],
        out / "strip.png",
        start_s=min(30.0, max(1.0, rec.duration_s / 4)),
        duration_s=10.0,
    )
    report_cfg = load_report_config()
    write_html(payload, out / "report.html", strip, config=report_cfg)
    write_pdf(payload, out / "report.pdf", strip, config=report_cfg)
    files["html"] = str(out / "report.html")
    files["pdf"] = str(out / "report.pdf")
    files["strip"] = str(strip)
    files["index"] = str(write_index(out.parent))

    return {"payload": payload, "files": files, "out_dir": str(out)}


def _fallback_html(payload: dict[str, Any]) -> str:
    counts = payload.get("counts", {})
    summary = payload.get("summary", {})
    rows = "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in counts.items())
    return f"""<!doctype html><html lang="es"><meta charset="utf-8">
<title>HolterIA — {payload.get("exam_id", "")}</title>
<body style="font-family:system-ui;margin:2rem">
<h1>HolterIA — borrador automático</h1>
<p><b>No es un dispositivo médico.</b> Requiere revisión del cardiólogo.</p>
<h2>Conteos</h2><ul>{rows}</ul>
<h2>Resumen</h2><pre>{json.dumps(summary, indent=2, ensure_ascii=False)[:4000]}</pre>
</body></html>"""
