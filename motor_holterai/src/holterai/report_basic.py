"""Reporte HTML/PDF mínimo a partir de `report.json` (sin dependencias nuevas)."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from holterai.report_config import ReportConfig, load_report_config, logo_data_uri


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return html.escape(str(value))


def _clinical_key(key: str) -> bool:
    """Métricas clínicas del modelo que no se publican hasta validar."""
    return key.startswith(("pvc_", "pac_", "af_")) or key in {
        "lown",
        "ventricular",
        "pacing",
        "n_pauses",
        "n_af_episodes",
        "n_no_organized",
        "n_asystole",
    }


def _num(value: Any, digits: int = 0) -> str:
    """Número con decimales fijos; guion largo si falta."""
    if value is None:
        return "—"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def qtc_table_html(summary: dict[str, Any]) -> str:
    """Sección de QT/QTc corregido (exploratoria) a partir del resumen."""
    by = summary.get("qtc_by_formula") or {}
    if not isinstance(by, dict) or not by:
        return ""
    selected = str(summary.get("qtc_formula") or "bazett")
    rows = []
    for name, st in by.items():
        if not isinstance(st, dict):
            continue
        over = st.get("minutes_over") or {}
        mark = " (indicada)" if name == selected else ""
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(st.get('label') or name))}{mark}</td>"
            f"<td>{_num(st.get('qtc_min_ms'), 1)}</td>"
            f"<td>{_num(st.get('qtc_mean_ms'), 1)}</td>"
            f"<td>{_num(st.get('qtc_max_ms'), 1)}</td>"
            f"<td>{fmt_minute(st.get('qtc_max_minute'))}</td>"
            f"<td>{_num(over.get('450'))}</td>"
            f"<td>{_num(over.get('480'))}</td>"
            f"<td>{_num(over.get('500'))}</td>"
            "</tr>"
        )
    chosen = by.get(selected) if isinstance(by.get(selected), dict) else {}
    excluded = (
        f"Excluidos: FA {_num(chosen.get('excluded_af'))} latidos · "
        f"FC &gt; 120 lpm {_num(chosen.get('excluded_hr'))} latidos · "
        f"minutos analizados {_num(chosen.get('minutes_analyzed'))}."
    )
    return (
        "<h2>QT/QTc corregido (exploratorio)</h2>"
        '<p class="warn">Medición exploratoria: QT estimado por retorno de la onda T '
        "y fórmulas poblacionales. No es diagnóstico ni sustituye la medición manual. "
        "Los minutos se cuentan por mediana del QTc de cada minuto.</p>"
        "<table><tr><th>Fórmula</th><th>QTc mín (ms)</th><th>QTc medio (ms)</th>"
        "<th>QTc máx (ms)</th><th>Minuto del máximo</th><th>min &gt; 450 ms</th>"
        "<th>min &gt; 480 ms</th><th>min &gt; 500 ms</th></tr>"
        + "".join(rows)
        + "</table>"
        f"<p>Fórmula indicada en criterios: <b>{html.escape(selected)}</b>. {excluded}</p>"
    )


def fmt_minute(minute: Any) -> str:
    """Minuto del registro en formato h:mm (o guion si falta)."""
    if minute is None:
        return "—"
    try:
        m = int(minute)
    except (TypeError, ValueError):
        return html.escape(str(minute))
    return f"{m // 60}:{m % 60:02d}"


def write_index(root: Path) -> Path:
    """Índice simple de reportes para `holterai serve`."""
    links = []
    for p in sorted(root.glob("*/report.html")):
        rel = p.relative_to(root).as_posix()
        links.append(f'<li><a href="{html.escape(rel)}">{html.escape(p.parent.name)}</a></li>')
    body = (
        '<!doctype html><html lang="es"><meta charset="utf-8">'
        '<title>HolterIA — reportes</title><body style="font-family:system-ui;margin:2rem">'
        "<h1>HolterIA — reportes locales</h1>"
        "<p>Borradores automáticos; no son diagnóstico.</p><ul>"
        + "".join(links)
        + "</ul></body></html>"
    )
    path = root / "index.html"
    path.write_text(body, encoding="utf-8")
    return path


def write_html(
    payload: dict[str, Any],
    path: Path,
    strip: Path | None = None,
    config: ReportConfig | None = None,
) -> Path:
    cfg = config or load_report_config()
    counts = payload.get("counts", {})
    summary = payload.get("summary", {})
    meta = payload.get("meta", {}) or {}
    pat = meta.get("patient", {}) or {}
    patient_bits = [
        str(pat.get("name") or ""),
        f"ID {pat['patient_id']}" if pat.get("patient_id") else "",
        str(pat.get("sex") or ""),
        f"{pat['age']} años" if pat.get("age") else "",
    ]
    patient_line = " · ".join(x for x in patient_bits if x)
    rows = "".join(f"<tr><td>{_fmt(k)}</td><td>{_fmt(v)}</td></tr>" for k, v in counts.items())
    summary_rows = "".join(
        f"<tr><td>{_fmt(k)}</td><td>{_fmt(v)}</td></tr>"
        for k, v in summary.items()
        if not isinstance(v, (dict, list)) and not _clinical_key(k)
    )
    qtc_html = qtc_table_html(summary)
    events = payload.get("events", [])
    events_html = "".join(
        f"<tr><td>{_fmt(e.get('kind'))}</td><td>{_fmt(e.get('start_s'))}</td>"
        f"<td>{_fmt(e.get('end_s'))}</td><td>{_fmt(e.get('detail'))}</td></tr>"
        for e in events[:200]
    )
    if events:
        events_section = (
            "<h2>Eventos candidatos (experimental, no validados)</h2>"
            "<table><tr><th>Tipo</th><th>Inicio (s)</th><th>Fin (s)</th>"
            f"<th>Detalle</th></tr>{events_html}</table>"
        )
    else:
        events_section = f'<p class="warn">{_fmt(payload.get("events_note", "Sin eventos"))}</p>'
    strip_html = (
        f'<h2>Tira de ejemplo</h2><img src="{html.escape(strip.name)}" '
        'style="max-width:100%;border:1px solid #dde3ea">'
        if strip is not None and strip.exists()
        else ""
    )
    hs = payload.get("hscribe", {})
    hs_html = ""
    if isinstance(hs, dict) and hs.get("n_beats"):
        hs_labels = hs.get("labels", {})
        hs_rows = "".join(
            f"<tr><td>{_fmt(k)}</td><td>{_fmt(v)}</td></tr>" for k, v in hs_labels.items()
        )
        hs_html = (
            "<h2>HScribe (metadatos del examen)</h2>"
            f"<p>Latidos en la ventana: {_fmt(hs.get('n_beats'))} · "
            f"total del examen: {_fmt(hs.get('exam_total_n', '—'))}</p>"
            f"<table><tr><th>Label HScribe</th><th>n</th></tr>{hs_rows}</table>"
            '<p class="muted">Labels crudos de HScribe; no se interpretan como '
            "diagnóstico. No equivalen a las clases N/S/V/F/Q del prototipo.</p>"
        )
    val = payload.get("validation", {})
    val_html = ""
    if isinstance(val, dict) and val.get("reference_n"):
        val_html = (
            "<h2>Validación técnica contra HScribe</h2>"
            '<p class="warn">La clasificación de latidos y los eventos candidatos '
            "NO están validados clínicamente. No usar PVC/VT/FA de este borrador "
            "como hallazgo.</p>"
            f"<p>QRS emparejado ±150 ms: {_fmt(val.get('matched'))} / "
            f"{_fmt(val.get('reference_n'))} · Sensibilidad "
            f"{_fmt(val.get('sensitivity'))} · PPV {_fmt(val.get('ppv'))} · "
            f"mediana Δt {_fmt(val.get('median_dt_ms'))} ms</p>"
        )
    if isinstance(val, dict) and val.get("reference_n"):
        cov = val.get("coverage", 0.0)
        aligned = "sí" if val.get("reference_valid") else "no"
        val_html += (
            f"<p>Cobertura de la referencia: {_fmt(cov)} · "
            f"Referencia alineada: {aligned}</p>"
        )
    brand_bits = "".join(f"<div>{html.escape(x)}</div>" for x in (cfg.clinic, cfg.doctor) if x)
    brand = f'<div class="brand">{brand_bits}</div>' if brand_bits else ""
    logo = logo_data_uri(cfg.logo)
    logo_html = f'<img class="logo" src="{logo}" alt="logo">' if logo else ""
    footer_bits = " · ".join(x for x in (cfg.signature, cfg.footer) if x)
    footer = f'<p class="footer">{html.escape(footer_bits)}</p>' if footer_bits else ""
    diary = html.escape(str(payload.get("diary", "") or ""))
    diary_section = (
        f"<h2>Diario del paciente</h2><p>{diary}</p>" if diary else ""
    )
    body = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>HolterIA — {html.escape(str(payload.get("exam_id", "")))}</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:32px;color:#182028;max-width:1100px}}
h1{{font-size:26px;margin-bottom:2px}} h2{{margin-top:28px}}
table{{border-collapse:collapse;width:100%;margin:8px 0}}
td,th{{border-bottom:1px solid #dde3ea;padding:4px 6px;text-align:left;font-size:14px}}
.warn{{background:#fff4e5;border-left:4px solid #e08b00;padding:10px}}
.muted{{color:#6e7a88;font-size:12px}}
.brand{{float:right;text-align:right;font-size:13px;color:#3d4a58}}
.logo{{max-height:56px;max-width:220px;display:block;margin-left:auto}}
.footer{{margin-top:28px;border-top:1px solid #dde3ea;padding-top:8px;color:#56616e;font-size:12px}}
</style></head><body>
{logo_html}
{brand}
<h1>HolterIA — borrador automático</h1>
<p><b>{html.escape(patient_line) if patient_line else "Paciente: —"}</b></p>
<p class="warn">{html.escape(str(payload.get("disclaimer", "")))}</p>
<p class="muted">Examen: {html.escape(str(payload.get("exam_id", "")))} ·
{html.escape(str(payload.get("generated_at", "")))} · {html.escape(str(payload.get("software", {})))}</p>
{strip_html}
{hs_html}
{val_html}
<h2>Resumen</h2><table>{summary_rows}</table>
{qtc_html}
<h2>Conteo de latidos</h2><table><tr><th>Clase</th><th>n</th></tr>{rows}</table>
{events_section}
{diary_section}
<p class="muted">Este reporte no interpreta ni diagnostica. La lectura clínica es del cardiólogo.</p>
{footer}
</body></html>"""
    path.write_text(body, encoding="utf-8")
    return path


def write_pdf(
    payload: dict[str, Any],
    path: Path,
    strip: Path | None = None,
    config: ReportConfig | None = None,
) -> Path:
    cfg = config or load_report_config()
    counts = payload.get("counts", {})
    summary = payload.get("summary", {})
    meta = payload.get("meta", {}) or {}
    pat = meta.get("patient", {}) or {}
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        head = " · ".join(x for x in (cfg.clinic, cfg.doctor) if x)
        if head:
            fig.text(0.08, 0.965, head, fontsize=9, color="#3d4a58")
        fig.text(0.08, 0.94, "HolterIA — borrador automático", fontsize=18, weight="bold")
        fig.text(0.08, 0.915, str(payload.get("exam_id", "")), fontsize=11)
        pline = " · ".join(
            x
            for x in (
                str(pat.get("name") or ""),
                f"ID {pat['patient_id']}" if pat.get("patient_id") else "",
                str(pat.get("sex") or ""),
                f"{pat['age']} años" if pat.get("age") else "",
            )
            if x
        )
        if pline:
            fig.text(0.08, 0.905, pline, fontsize=9, color="#3d4a58")
        fig.text(
            0.08,
            0.885,
            str(payload.get("disclaimer", "")),
            fontsize=8,
            wrap=True,
            color="#8a4b00",
        )
        lines = [
            f"{k}: {_fmt(v)}"
            for k, v in summary.items()
            if not isinstance(v, (dict, list)) and not _clinical_key(k)
        ]
        fig.text(0.08, 0.84, "\n".join(lines[:28]), fontsize=8, va="top", family="monospace")
        ctext = "   ".join(f"{k}={v}" for k, v in counts.items())
        fig.text(0.08, 0.40, "Latidos: " + ctext, fontsize=10)
        diary = str(payload.get("diary", "") or "").strip()
        if diary:
            fig.text(0.08, 0.24, "Diario: " + diary[:1200], fontsize=8, wrap=True)
        hs = payload.get("hscribe", {})
        if isinstance(hs, dict) and hs.get("n_beats"):
            hs_text = "HScribe: n={} labels={}".format(hs.get("n_beats"), hs.get("labels", {}))
            fig.text(0.08, 0.36, hs_text, fontsize=8, family="monospace")
        val = payload.get("validation", {})
        if isinstance(val, dict) and val.get("reference_n"):
            fig.text(
                0.08,
                0.32,
                "Validacion QRS: Se={} PPV={} matched={}/{}".format(
                    val.get("sensitivity"),
                    val.get("ppv"),
                    val.get("matched"),
                    val.get("reference_n"),
                ),
                fontsize=8,
                family="monospace",
            )
        qtc = summary.get("qtc") if isinstance(summary.get("qtc"), dict) else {}
        if qtc:
            over = qtc.get("minutes_over") or {}

            def _p(v: Any) -> str:
                return "-" if v is None else str(v)

            fig.text(
                0.08,
                0.29,
                "QTc exploratorio ({}): min {} med {} max {} ms | "
                "min>450/480/500: {}/{}/{}".format(
                    qtc.get("formula", summary.get("qtc_formula", "bazett")),
                    _p(qtc.get("qtc_min_ms")),
                    _p(qtc.get("qtc_mean_ms")),
                    _p(qtc.get("qtc_max_ms")),
                    _p(over.get("450", 0)),
                    _p(over.get("480", 0)),
                    _p(over.get("500", 0)),
                ),
                fontsize=7,
                family="monospace",
            )
        fig.text(
            0.08,
            0.27,
            "No es un dispositivo médico. Requiere revisión del cardiólogo.",
            fontsize=8,
        )
        foot = " · ".join(x for x in (cfg.signature, cfg.footer) if x)
        if foot:
            fig.text(0.08, 0.06, foot, fontsize=8, color="#56616e")
        pdf.savefig(fig)
        plt.close(fig)
        if strip is not None and strip.exists():
            img = plt.imread(strip)
            fig2 = plt.figure(figsize=(8.27, 11.69))
            ax = fig2.add_axes((0.04, 0.35, 0.92, 0.55))
            ax.imshow(img)
            ax.axis("off")
            fig2.text(0.08, 0.25, "Tira de ejemplo (10 s)", fontsize=10)
            pdf.savefig(fig2)
            plt.close(fig2)
    return path


def write_strip(
    samples: Any,
    fs: float,
    beats_t: Any,
    path: Path,
    start_s: float = 5.0,
    duration_s: float = 10.0,
) -> Path:
    """Dibuja una tira de 3 canales con marcas R y la guarda como PNG."""
    import numpy as np

    a = max(0, int(start_s * fs))
    b = min(len(samples), int((start_s + duration_s) * fs))
    seg = np.asarray(samples[a:b], dtype=float)
    t = np.arange(seg.shape[0]) / fs + start_s
    fig, axes = plt.subplots(seg.shape[1], 1, figsize=(11, 2.2 * seg.shape[1]), sharex=True)
    if seg.shape[1] == 1:
        axes = [axes]
    for ch in range(seg.shape[1]):
        axes[ch].plot(t, seg[:, ch], lw=0.7, color="#1f4e79")
        axes[ch].set_ylabel(f"canal {ch + 1} (mV)", fontsize=8)
        axes[ch].tick_params(labelsize=7)
        for bt in beats_t:
            if start_s <= bt <= start_s + duration_s:
                axes[ch].axvline(bt, color="#c0392b", lw=0.5, alpha=0.7)
    axes[-1].set_xlabel("tiempo (s)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
