"""Generación del informe (DOCX + PDF/HTML) con gráficos.

El informe es el consumidor más estricto del registro de funciones
(`configs/features.yaml`): solo muestra funciones en estado `visible`.
El registro y su puerta viven aquí porque este módulo es quien debe
respetarlo; el servidor los reexporta por `/api/features` para la interfaz.
"""
from __future__ import annotations

import html as html_mod
import os
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

from holterai.legacy.app.core import config
from holterai.legacy.app.core.models import Analysis
from holterai.legacy.app.core.logging import get_logger
from holterai.legacy.app.report import plotting as P
from holterai.legacy.app.signal import hrv as hrv_mod
from holterai.report_config import load_report_config

log = get_logger(__name__)

CLASS_NAMES = {"N": "Normal", "S": "Supraventricular", "V": "Ventricular",
               "F": "Fusión", "Q": "No clasificable"}
EVENT_NAMES = {
    "pause": "Pausa", "brady": "Bradicardia", "tachy": "Taquicardia",
    "afib": "Fibrilación auricular", "vt_run": "Racha ventricular",
    "bigeminy": "Bigeminismo", "svt": "Taquicardia supraventricular",
    "st_depression": "Depresión ST", "st_elevation": "Elevación ST",
}
DEFAULT_SECTIONS = {"resumen", "hrv", "strips", "horaria", "eventos", "st", "conclusiones"}


# ---------------------------------------------------------------------------
# Registro de funciones (configs/features.yaml)
#
# Estados: visible = se puede mostrar e informar; experimental = se muestra en
# la interfaz con franja de aviso pero NO se informa; hidden = no se muestra.
# ---------------------------------------------------------------------------
VISIBLE = "visible"
EXPERIMENTAL = "experimental"
HIDDEN = "hidden"
FEATURE_STATUSES = (VISIBLE, EXPERIMENTAL, HIDDEN)
FEATURES_ENV = "HOLTERIA_FEATURES"
FEATURES_FILE = config.APP_ROOT / "configs" / "features.yaml"
UI_KEYS = ("tabs", "panes", "ids", "selectors")

# Funciones que este módulo puede llegar a mostrar. La puerta decide si se
# incluyen según el estado; la prueba unitaria comprueba que todas existan en
# el registro, que puedan informarse y que ninguna no visible se cuele.
REPORT_FEATURES = (
    "report.generate",
    "report.narrative",
    "manage.patient",
    "analysis.qrs_detect",
    "analysis.heart_rate",
    "analysis.hourly",
    "analysis.hrv",
    "beat.classification",
    "beat.ectopy_ventricular",
    "beat.ectopy_supraventricular",
    "rhythm.pause",
    "rhythm.brady_tachy",
    "rhythm.af",
    "rhythm.vt_svt",
    "rhythm.pacing",
    "view.strips",
)

# Tipo de evento -> función que lo respalda. Un evento desconocido se trata
# como anotación del revisor.
EVENT_FEATURES = {
    "pause": "rhythm.pause",
    "brady": "rhythm.brady_tachy",
    "tachy": "rhythm.brady_tachy",
    "afib": "rhythm.af",
    "vt_run": "rhythm.vt_svt",
    "svt": "rhythm.vt_svt",
    "bigeminy": "beat.ectopy_ventricular",
    "couplet": "beat.ectopy_ventricular",
    "triplet": "beat.ectopy_ventricular",
    "st_depression": "measure.st",
    "st_elevation": "measure.st",
}
MANUAL_EVENT_FEATURE = "report.narrative"

_ID_RX = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class Feature:
    """Una función del registro con su estado, su interfaz y su evidencia."""

    id: str
    name: str
    area: str
    status: str
    report: bool = False
    reason: str = ""
    evidence: tuple[str, ...] = ()
    report_markers: tuple[str, ...] = ()
    ui: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def is_visible(self) -> bool:
        return self.status == VISIBLE

    def ui_list(self, key: str) -> tuple[str, ...]:
        return self.ui.get(key, ())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "area": self.area,
            "status": self.status,
            "reason": self.reason,
            "report": self.report,
            "report_markers": list(self.report_markers),
            "ui": {k: list(v) for k, v in self.ui.items()},
            "evidence": list(self.evidence),
        }


_CACHE: dict[str, tuple[float, dict[str, Feature]]] = {}


def features_path(path: str | Path | None = None) -> Path:
    """Ruta del registro: argumento, variable de entorno o `configs/`."""
    if path is not None:
        return Path(path)
    env = os.environ.get(FEATURES_ENV, "").strip()
    if env:
        return Path(env)
    return FEATURES_FILE


def load_features(path: str | Path | None = None) -> dict[str, Feature]:
    """Carga y valida el registro de funciones (con caché por fecha)."""
    p = features_path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no existe el registro de funciones: {p}")
    stamp = p.stat().st_mtime
    key = str(p.resolve())
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    feats = _parse_registry(raw, p)
    _CACHE[key] = (stamp, feats)
    return feats


def _text_list(value: Any, what: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not str(v).strip() for v in value):
        raise ValueError(f"{what} debe ser una lista de textos")
    return tuple(str(v).strip() for v in value)


def _parse_registry(raw: Any, path: Path) -> dict[str, Feature]:
    if not isinstance(raw, dict):
        raise ValueError(f"el registro no es un mapeo: {path}")
    entries = raw.get("features")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"el registro no tiene lista 'features': {path}")
    out: dict[str, Feature] = {}
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError(f"entrada no válida en {path}: {item!r}")
        fid = str(item.get("id", "") or "").strip()
        if not _ID_RX.match(fid):
            raise ValueError(f"id inválido en {path}: {fid!r}")
        if fid in out:
            raise ValueError(f"id repetido en {path}: {fid}")
        status = str(item.get("status", "") or "").strip()
        if status not in FEATURE_STATUSES:
            raise ValueError(f"estado inválido para {fid}: {status!r}")
        name = str(item.get("name", "") or "").strip()
        area = str(item.get("area", "") or "").strip()
        if not name or not area:
            raise ValueError(f"faltan 'name' o 'area' en {fid}")
        reason = str(item.get("reason", "") or "").strip()
        if status != VISIBLE and not reason:
            raise ValueError(f"falta 'reason' para {fid} ({status})")
        do_report = item.get("report", False)
        if not isinstance(do_report, bool):
            raise ValueError(f"'report' debe ser booleano en {fid}")
        ui_raw = item.get("ui") or {}
        if not isinstance(ui_raw, dict):
            raise ValueError(f"'ui' debe ser un mapeo en {fid}")
        unknown = set(ui_raw) - set(UI_KEYS)
        if unknown:
            raise ValueError(f"claves ui desconocidas en {fid}: {sorted(unknown)}")
        ui: dict[str, tuple[str, ...]] = {}
        for k in UI_KEYS:
            vals = _text_list(ui_raw.get(k), f"ui.{k} de {fid}")
            if vals:
                ui[k] = vals
        out[fid] = Feature(
            id=fid,
            name=name,
            area=area,
            status=status,
            report=do_report,
            reason=reason,
            evidence=_text_list(item.get("evidence"), f"evidence de {fid}"),
            report_markers=_text_list(item.get("report_markers"), f"report_markers de {fid}"),
            ui=ui,
        )
    return out


def visible_ids(features: Mapping[str, Feature]) -> frozenset[str]:
    """Identificadores en estado visible."""
    return frozenset(f.id for f in features.values() if f.is_visible)


class FeatureGate:
    """Decide si una función se puede mostrar y deja traza de lo decidido.

    `on()` devuelve True solo para funciones visibles; un id desconocido se
    trata como oculto (falla cerrado) y queda registrado en `skipped`.
    """

    def __init__(self, features: Mapping[str, Feature]) -> None:
        self.features = features
        self.rendered: set[str] = set()
        self.skipped: set[str] = set()

    def status(self, fid: str) -> str:
        feat = self.features.get(fid)
        return feat.status if feat is not None else HIDDEN

    def on(self, fid: str) -> bool:
        ok = self.status(fid) == VISIBLE
        (self.rendered if ok else self.skipped).add(fid)
        return ok


def ui_plan(features: Mapping[str, Feature]) -> dict[str, list[str]]:
    """Traduce el registro a selectores CSS para la interfaz.

    `hide` son elementos que se ocultan (estado hidden), `banners` paneles que
    llevan la franja "exploratorio, no validado", `tags` controles concretos
    que se marcan como exploratorios y `hidden_tabs` pestañas que no deben
    poder activarse.
    """
    hide: list[str] = []
    banners: list[str] = []
    tags: list[str] = []
    exp_tabs: list[str] = []
    hidden_tabs: list[str] = []

    def add(bucket: list[str], value: str) -> None:
        if value and value not in bucket:
            bucket.append(value)

    for feat in features.values():
        tabs = feat.ui_list("tabs")
        panes = feat.ui_list("panes")
        ids = feat.ui_list("ids")
        sels = feat.ui_list("selectors")
        if feat.status == HIDDEN:
            for tab in tabs:
                add(hide, f'[data-tab="{tab}"]')
                add(hidden_tabs, tab)
            for pane in panes:
                add(hide, f"#{pane}")
                if pane.startswith("pane-"):
                    tab = pane[len("pane-"):]
                    add(hide, f'[data-tab="{tab}"]')
                    add(hidden_tabs, tab)
            for element in ids:
                add(hide, f"#{element}")
            for sel in sels:
                add(hide, sel)
        elif feat.status == EXPERIMENTAL:
            for tab in tabs:
                add(exp_tabs, tab)
            for pane in panes:
                add(banners, f"#{pane}")
            for element in ids:
                add(tags, f"#{element}")
            for sel in sels:
                add(tags, sel)
    return {
        "hide": hide,
        "banners": banners,
        "tags": tags,
        "exp_tabs": exp_tabs,
        "hidden_tabs": hidden_tabs,
    }


def _fmt_dur(seconds: float) -> str:
    s = int(max(0, seconds))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


QTC_THRESHOLDS: tuple[int, ...] = (450, 480, 500)


def _fmt_ms(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.1f}"
    return str(value)


def _fmt_qtc_minute(minute: object) -> str:
    if minute is None:
        return "—"
    try:
        m = int(minute)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    return f"{m // 60}:{m % 60:02d}"


# ---------------------------------------------------------------------------
# Onda P, PR y morfología del QRS (T33)
# ---------------------------------------------------------------------------
def _pr_rows(pw: dict) -> list[tuple[str, str, str]]:
    """(medida, ms, hora) del PR mínimo, mediano y máximo."""
    if not pw or pw.get("pr_median_ms") is None:
        return []
    return [
        ("PR mínimo", _fmt_ms(pw.get("pr_min_ms")), str(pw.get("pr_min_hour") or "—")),
        ("PR mediano", _fmt_ms(pw.get("pr_median_ms")),
         str(pw.get("pr_median_hour") or "—")),
        ("PR máximo", _fmt_ms(pw.get("pr_max_ms")), str(pw.get("pr_max_hour") or "—")),
    ]


def _pwave_sentence(pw: dict) -> str:
    """Frase con la P presente/ausente y la cobertura de minutos evaluables."""
    if not pw or "n_evaluable" not in pw:
        return ""
    n_eval = int(pw.get("n_evaluable") or 0)
    n_min = int(pw.get("n_minutes") or 0)
    if n_eval == 0:
        return ("Onda P: ningún minuto evaluable "
                f"({n_min} minutos; revisar calidad de señal).")
    txt = (f"Onda P: presente en el {pw.get('p_present_pct', 0):.1f} % de los minutos "
           f"con P evaluable ({pw.get('n_present', 0)} de {pw.get('n_minutes_with_p', 0)}). "
           f"Minutos medidos: {n_eval} de {n_min} "
           f"({pw.get('evaluable_pct', 0):.1f} %); el resto queda como no evaluable.")
    reasons = pw.get("no_evaluable_reasons") or {}
    if reasons:
        names = {"calidad": "calidad de señal", "pocos_latidos": "pocos latidos",
                 "borde": "borde del registro", "qrs_no_medible": "QRS no medible",
                 "sin_latidos": "sin latidos"}
        detalle = ", ".join(f"{names.get(k, k)} {v}" for k, v in sorted(reasons.items()))
        txt += f" Motivos: {detalle}."
    return txt


def _qrs_morph_sentence(qm: dict) -> str:
    """Frase con QRS ancho sí/no, patrón, persistencia y % del tiempo."""
    if not qm or "n_evaluable" not in qm:
        return ""
    n_eval = int(qm.get("n_evaluable") or 0)
    if n_eval == 0:
        return "QRS: sin minutos evaluables para medir la duración."
    pct = float(qm.get("wide_pct_time") or 0.0)
    med = qm.get("median_width_ms")
    if not qm.get("n_wide_minutes"):
        return (f"QRS ancho (≥{qm.get('wide_threshold_ms', 120):.0f} ms): no. "
                f"Mediana {_fmt_ms(med)} ms en {n_eval} minutos evaluables "
                f"(incompletos 110-119 ms en el {qm.get('incomplete_pct_time', 0):.1f} %).")
    kinds = {
        "br_persistente": "persistente",
        "br_intermitente": "intermitente",
        "br_frecuencia_dependiente": "dependiente de la frecuencia cardíaca",
        "morfologia_variable": "con morfología variable",
        "aberrancia_probable": "en minutos aislados (aberrancia probable)",
    }
    txt = (f"QRS ancho (≥{qm.get('wide_threshold_ms', 120):.0f} ms): sí, en el "
           f"{pct:.1f} % del tiempo ({qm.get('n_wide_minutes', 0)} de {n_eval} minutos). "
           f"Patrón dominante: {qm.get('pattern_name', qm.get('pattern', '—'))} "
           f"({kinds.get(str(qm.get('kind')), qm.get('kind', '—'))}). "
           f"Mediana {_fmt_ms(med)} ms.")
    if qm.get("rate_dependent") and qm.get("wide_hr") and qm.get("narrow_hr"):
        txt += (f" Frecuencia con QRS ancho {qm['wide_hr']:.0f} lpm frente a "
                f"{qm['narrow_hr']:.0f} lpm con QRS estrecho.")
    if qm.get("missing"):
        txt += " Limitación: " + "; ".join(str(x) for x in qm["missing"]) + "."
    return txt


def _pwave_docx(doc, s: dict) -> None:
    """Sección de onda P, PR y morfología del QRS del informe DOCX."""
    pw = s.get("pwave") or {}
    morph = s.get("qrs_morph") or {}
    if not pw and not morph:
        return
    doc.add_heading("Conducción auriculoventricular y QRS", level=1)
    clock = pw.get("clock") or ""
    note = f" Horas en {clock}." if clock else ""
    first = _pwave_sentence(pw)
    if first:
        doc.add_paragraph(first + note)
    second = _qrs_morph_sentence(morph)
    if second:
        doc.add_paragraph(second)
    rows = _pr_rows(pw)
    if rows:
        t = doc.add_table(rows=1, cols=3)
        t.style = "Light List Accent 1"
        for i, head in enumerate(("Medida", "PR (ms)", "Hora")):
            t.rows[0].cells[i].text = head
            t.rows[0].cells[i].paragraphs[0].runs[0].bold = True
        for row in rows:
            cells = t.add_row().cells
            for i, value in enumerate(row):
                cells[i].text = str(value)
        doc.add_paragraph(
            f"PR medido en {pw.get('n_minutes_pr', 0)} minutos "
            f"(p5-p95 {_fmt_ms(pw.get('pr_p5_ms'))}-{_fmt_ms(pw.get('pr_p95_ms'))} ms)."
        )
    p = doc.add_paragraph(
        "Método: onda P por plantilla del minuto con control par/impar y resta de la T "
        "previa; PR desde el inicio de la P de cada latido hasta el inicio del QRS del "
        "latido promediado del minuto; duración del QRS por unión multicanal y patrón de "
        "rama sobre ese mismo latido promediado. Medición exploratoria: no sustituye la "
        "lectura del ECG por un profesional."
    )
    p.runs[0].font.size = Pt(9)


def _qtc_rows(summary: dict) -> list[tuple[str, ...]]:
    """Filas de QTc (exploratorio): fórmula, mín, medio, máx, minuto y minutos sobre umbral."""
    by = summary.get("qtc_by_formula") or {}
    if not isinstance(by, dict):
        return []
    selected = str(summary.get("qtc_formula") or "bazett")
    rows: list[tuple[str, ...]] = []
    for name, st in by.items():
        if not isinstance(st, dict):
            continue
        over = st.get("minutes_over") or {}
        label = str(st.get("label") or name)
        if name == selected:
            label += " (indicada)"
        rows.append((
            label,
            _fmt_ms(st.get("qtc_min_ms")),
            _fmt_ms(st.get("qtc_mean_ms")),
            _fmt_ms(st.get("qtc_max_ms")),
            _fmt_qtc_minute(st.get("qtc_max_minute")),
            "/".join(str(over.get(str(t), 0)) for t in QTC_THRESHOLDS),
        ))
    return rows


def _soffice() -> str | None:
    candidates = [
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/LibreOffice/program/soffice.exe"),
        "/usr/bin/soffice",
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


def _libreoffice_kit() -> tuple[str, str] | None:
    """CLI de LibreOffice Kit incluido en algunos entornos (DeepSeek Harness)."""
    node = Path(os.environ.get(
        "HOLTERIA_NODE",
        r"C:\Users\CokeNote\AppData\Local\Programs\DeepSeek Harness\resources"
        r"\runtime\primary-runtime\dependencies\node\bin\node.exe",
    ))
    cli = Path(os.environ.get(
        "HOLTERIA_LO_CLI",
        r"C:\Users\CokeNote\AppData\Local\Programs\DeepSeek Harness\resources"
        r"\app.asar.unpacked\dsh\node_modules\@deepseek-ai\libreoffice-kit\lib\cli.js",
    ))
    if node.exists() and cli.exists():
        return str(node), str(cli)
    return None


# ---------------------------------------------------------------------------
# Selección de tiras de ejemplo
# ---------------------------------------------------------------------------
def _strip_candidates(an: Analysis, lead: int, window_s: float = 10.0,
                      gate: FeatureGate | None = None):
    """Devuelve tiras representativas: normal, ectopia, FA, pausa.

    Cada tipo de tira se asocia a la función que lo respalda; solo se
    devuelven las de funciones visibles según `gate`.
    """
    if gate is None:
        gate = FeatureGate(load_features())
    rec = an.recording
    beats = an.beats
    if not beats or not gate.on("view.strips"):
        return []
    out = []
    t_all = np.array([b.t for b in beats])
    cls = np.array([b.cls for b in beats])

    def window_at(t):
        a = max(0.0, min(rec.duration_s - window_s, t - window_s / 2))
        return a, a + window_s

    # 1) tramo normal (solo latidos N, FC 55-95): exige clasificación
    rr = np.array([b.rr_prev for b in beats])
    hr = np.where(rr > 0, 60.0 / np.maximum(rr, 1e-6), np.nan)
    if gate.on("beat.classification"):
        for i, b in enumerate(beats):
            if not (55 < (hr[i] if np.isfinite(hr[i]) else 0) < 95):
                continue
            a = max(0.0, min(rec.duration_s - window_s, b.t - window_s / 2))
            sub = [x for x in beats if a <= x.t <= a + window_s]
            if len(sub) >= 6 and all(x.cls == "N" for x in sub):
                out.append(("Ritmo sinusal (ejemplo)", (a, a + window_s)))
                break

    # 2) ectopia ventricular
    vs = np.flatnonzero(cls == "V")
    if vs.size and gate.on("beat.ectopy_ventricular"):
        k = vs[len(vs) // 2]
        out.append(("Ectopia ventricular", window_at(float(t_all[k]))))

    # 3) fibrilación auricular
    if an.rhythm and gate.on("rhythm.af"):
        seg = max(an.rhythm, key=lambda s: s.end_s - s.start_s)
        out.append(("Fibrilación auricular", window_at((seg.start_s + seg.end_s) / 2)))

    # 4) pausa
    pauses = [e for e in an.events if e.kind == "pause"]
    if pauses and gate.on("rhythm.pause"):
        e = max(pauses, key=lambda x: x.duration_s)
        out.append((f"Pausa de {e.duration_s:.1f} s", window_at((e.start_s + e.end_s) / 2)))
    return out


# ---------------------------------------------------------------------------
# Informe
# ---------------------------------------------------------------------------
def generate(an: Analysis, formats: list[str] | None = None,
             sections: list[str] | None = None) -> dict:
    """Genera el informe y devuelve las rutas y la traza de funciones usadas."""
    formats = formats or ["docx", "pdf"]
    sec = set(sections) if sections else set(DEFAULT_SECTIONS)
    gate = FeatureGate(load_features())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = config.REPORTS_DIR / f"informe_{stamp}"
    imgs = _render_images(an, base, gate)
    docx_path = base.with_suffix(".docx")
    _build_docx(an, docx_path, imgs, sec, gate)
    result: dict[str, Any] = {"docx": str(docx_path)}

    if "pdf" in formats:
        pdf = _to_pdf(docx_path)
        if pdf:
            result["pdf"] = str(pdf)
    if "html" in formats:
        html = _build_html(an, base, imgs, gate)
        result["html"] = str(html)
    result["reports_dir"] = str(config.REPORTS_DIR)
    # Traza para la prueba de aceptación: lo mostrado debe ser siempre visible.
    result["features_used"] = sorted(gate.rendered)
    result["features_skipped"] = sorted(gate.skipped)
    return result


def _render_images(an: Analysis, base: Path,
                   gate: FeatureGate | None = None) -> dict:
    if gate is None:
        gate = FeatureGate(load_features())
    imgs: dict[str, Path] = {}
    img_dir = base.parent / (base.name + "_img")
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1) tiras de ECG
    lead = int(an.quality.get("lead", 0))
    for i, (title, (t0, t1)) in enumerate(_strip_candidates(an, lead, gate=gate)):
        a, seg = an.recording.segment(t0, t1)
        if seg.size == 0:
            continue
        sub = [b for b in an.beats if t0 <= b.t <= t1]
        img = P.ecg_strip(seg, an.recording.fs, t0, width=1700, height=430,
                          leads=an.recording.leads, beats=[b.to_dict() for b in sub],
                          title=title)
        imgs[f"strip_{i}"] = P.save_png(img, img_dir / f"strip_{i}.png")

    # 2) tendencia de FC por hora
    from holterai.legacy.app.signal.rhythm import hourly_rr_stats
    t, hr = hourly_rr_stats(an.beats, an.recording.duration_s)
    if t.size > 1 and gate.on("analysis.hourly"):
        imgs["trend"] = P.save_png(
            P.trend_chart(t, hr, width=1200, height=260,
                          title="Frecuencia cardíaca (mediana por hora)", ylabel="lpm"),
            img_dir / "trend.png")

    # 3) dispersión RR
    rr = np.array([b.rr_prev for b in an.beats[1:]], dtype=float)
    rr = rr[(rr > 0.2) & (rr < 3.0)]
    if rr.size > 10 and gate.on("analysis.heart_rate"):
        imgs["scatter"] = P.save_png(
            P.scatter_chart(rr[:-1], rr[1:], width=760, height=420,
                            title="Dispersión RR (n vs n+1)",
                            xlabel="RR n (s)", ylabel="RR n+1 (s)"),
            img_dir / "scatter.png")

    # 4) barras por clase
    if gate.on("beat.classification"):
        counts = an.class_counts()
        labels = [f"{c}" for c in ("N", "S", "V", "Q")]
        vals = [counts.get(c, 0) for c in labels]
        imgs["bars"] = P.save_png(
            P.bar_chart(labels, vals, width=620, height=260,
                        title="Latidos por clase",
                        colors=[P.CLASS_COLORS[c] for c in labels]),
            img_dir / "bars.png")
    return imgs


def _build_docx(an: Analysis, path: Path, imgs: dict,
                sec: set[str] | None = None,
                gate: FeatureGate | None = None) -> Path:
    sec = sec or set(DEFAULT_SECTIONS)
    if gate is None:
        gate = FeatureGate(load_features())
    features = gate.features
    doc = Document()
    cfg = load_report_config()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10)

    # El documento en sí es la función report.generate.
    gate.on("report.generate")
    title = doc.add_heading("Informe de análisis Holter", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    if cfg.clinic or cfg.doctor:
        head = doc.add_paragraph(" · ".join(x for x in (cfg.clinic, cfg.doctor) if x))
        head.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    if cfg.logo:
        try:
            doc.add_picture(cfg.logo, width=Cm(3.5))
        except Exception:
            pass
    sub = doc.add_paragraph()
    r = sub.add_run(f"{config.APP_NAME} v{config.VERSION} · "
                    f"{datetime.now().strftime('%d/%m/%Y %H:%M')}")
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    warn = doc.add_paragraph()
    rw = warn.add_run(config.DISCLAIMER)
    rw.bold = True
    rw.font.size = Pt(9)
    rw.font.color.rgb = RGBColor(0xB0, 0x2A, 0x37)

    meta = an.recording.meta_dict()
    doc.add_heading("Datos del registro", level=1)
    t = doc.add_table(rows=0, cols=2)
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    rows = []
    if gate.on("manage.patient"):
        rows += [
            ("Paciente", meta["patient"].get("name") or "—"),
            ("Identificador", meta["patient"].get("patient_id") or "—"),
            ("Edad / sexo",
             f"{meta['patient'].get('age') or '—'} / {meta['patient'].get('sex') or '—'}"),
        ]
    rows += [
        ("Archivo", Path(meta["source"]).name),
        ("Formato", meta["format"]),
        ("Duración", f"{_fmt_dur(meta['duration_s'])} ({meta['duration_s']/3600:.1f} h)"),
        ("Derivaciones", ", ".join(meta["leads"])),
        ("Frecuencia de muestreo", f"{meta['fs']:g} Hz"),
        ("Inicio", meta.get("start_time") or "—"),
    ]
    for k, v in rows:
        cells = t.add_row().cells
        cells[0].text = k
        cells[1].text = str(v)
    for row in t.rows:
        row.cells[0].width = Cm(4.2)

    s = an.summary
    doc.add_heading("Resumen", level=1)
    entries: list[tuple[str, str]] = []
    if gate.on("analysis.qrs_detect"):
        entries.append(("Latidos analizados", f"{s.get('n_beats', 0):,}".replace(",", ".")))
    if gate.on("analysis.heart_rate"):
        entries += [
            ("FC media", f"{s.get('hr_mean', 0):.0f} lpm"),
            ("FC mínima (p1)", f"{s.get('hr_min', 0):.0f} lpm"),
            ("FC máxima (p99)", f"{s.get('hr_max', 0):.0f} lpm"),
        ]
    if gate.on("beat.ectopy_ventricular"):
        entries.append(("Carga de PVC", f"{s.get('pvc_burden_pct', 0):.2f} %"))
    if gate.on("beat.ectopy_supraventricular"):
        entries.append(("Carga de PAC", f"{s.get('pac_burden_pct', 0):.2f} %"))
    if gate.on("rhythm.af"):
        entries.append(("Carga de FA", f"{s.get('af_burden_pct', 0):.2f} %"))
    if gate.on("rhythm.pause"):
        entries.append(("Pausas > umbral", f"{s.get('n_pauses', 0)}"))
    if entries:
        t2 = doc.add_table(rows=0, cols=4)
        t2.style = "Light List Accent 1"
        for i in range(0, len(entries), 2):
            left = entries[i]
            right = entries[i + 1] if i + 1 < len(entries) else ("", "")
            cells = t2.add_row().cells
            cells[0].text, cells[1].text = left
            cells[2].text, cells[3].text = right
            if cells[0].text:
                cells[0].paragraphs[0].runs[0].bold = True
            if cells[2].text:
                cells[2].paragraphs[0].runs[0].bold = True

    qtc_rows = _qtc_rows(s) if "resumen" in sec else []
    if qtc_rows:
        doc.add_heading("QT/QTc corregido (exploratorio)", level=1)
        p = doc.add_paragraph(
            "Medición exploratoria automática: QT estimado por retorno de la onda T y "
            "fórmulas poblacionales. No sustituye la medición manual del cardiólogo. "
            "Se excluyen latidos en FA y con FC > 120 lpm; los minutos se cuentan por la "
            f"mediana del QTc de cada minuto. Fórmula indicada: {s.get('qtc_formula', 'bazett')}."
        )
        p.runs[0].font.size = Pt(9)
        tq = doc.add_table(rows=1, cols=6)
        tq.style = "Light List Accent 1"
        for i, h in enumerate(("Fórmula", "QTc mín (ms)", "QTc medio (ms)", "QTc máx (ms)",
                               "Minuto del máximo", "min > 450/480/500")):
            tq.rows[0].cells[i].text = h
            tq.rows[0].cells[i].paragraphs[0].runs[0].bold = True
        for row in qtc_rows:
            cells = tq.add_row().cells
            for i, value in enumerate(row):
                cells[i].text = str(value)
        tq.rows[0].cells[0].width = Cm(6.0)

    # onda P, PR y morfología del QRS (T33): igual que QT/ST, se informa con la
    # sección "resumen" (no hay todavía id propio en configs/features.yaml)
    if "resumen" in sec and (s.get("pwave") or s.get("qrs_morph")):
        _pwave_docx(doc, s)

    hrv = s.get("hrv", {})
    if hrv and "hrv" in sec and gate.on("analysis.hrv"):
        for line in hrv_mod.summary_lines(hrv):
            doc.add_paragraph(line)
        ventanas = s.get("hrv_windows") or {}
        for line in hrv_mod.day_night_lines(s.get("hrv_day"), s.get("hrv_night"), ventanas):
            doc.add_paragraph(line)

    lown = s.get("lown") or {}
    if lown and gate.on("beat.ectopy_ventricular"):
        doc.add_paragraph(
            f"Ectopia ventricular: {s.get('counts', {}).get('V', 0)} PVC "
            f"({s.get('pvc_per_hour', 0)}/h), "
            f"{s.get('pvc_morphologies', 1)} morfología(s), "
            f"parejas {s.get('ventricular', {}).get('couplets', 0)}, "
            f"tríos {s.get('ventricular', {}).get('triplets', 0)}, "
            f"R-on-T {s.get('ventricular', {}).get('r_on_t', 0)}. "
            f"Grado de Lown: {lown.get('label', '—')}."
        )

    pac = s.get("pacing") or {}
    if pac and gate.on("rhythm.pacing"):
        if pac.get("spikes_detected"):
            txt = (f"Marcapasos: {pac.get('pct_paced', 0)} % de latidos estimulados "
                   f"({pac.get('n_spikes', 0)} espigas, frecuencia "
                   f"{pac.get('pacing_rate') or '—'} lpm)")
            if pac.get("capture_failures"):
                txt += f", {pac['capture_failures']} espigas sin captura"
            if pac.get("crt_suspected"):
                txt += (". Resincronización: "
                        + str(pac.get("note") or "biventricular probable"))
            doc.add_paragraph(txt + ".")
        else:
            doc.add_paragraph("Marcapasos: " + str(pac.get("note") or
                                                   "sin espigas detectables") + ".")

    hourly = s.get("hourly") or []
    if hourly and "horaria" in sec:
        doc.add_heading("Resumen por hora", level=1)
        columns: list[tuple[str, Any]] = [
            ("Hora", lambda row: f"{row['hour']:02d}:00"),
            ("Latidos", lambda row: f"{row['n']}"),
            ("FC media", lambda row: f"{row['hr_mean']:.0f}" if row["hr_mean"] else "—"),
            ("FC mín–máx", lambda row: (f"{row['hr_min']:.0f}–{row['hr_max']:.0f}"
                                        if row["hr_min"] else "—")),
        ]
        if gate.on("beat.ectopy_ventricular"):
            columns.append(("PVC", lambda row: f"{row['pvc']}"))
        if gate.on("beat.ectopy_supraventricular"):
            columns.append(("PAC", lambda row: f"{row['pac']}"))
        if gate.on("rhythm.af"):
            columns.append(("FA (prob.)", lambda row: f"{row['af_pct']:.0f}"))
        th = doc.add_table(rows=1, cols=len(columns))
        th.style = "Light Grid Accent 1"
        for i, (head, _fn) in enumerate(columns):
            th.rows[0].cells[i].text = head
            th.rows[0].cells[i].paragraphs[0].runs[0].bold = True
        for row in hourly:
            c = th.add_row().cells
            for i, (_head, fn) in enumerate(columns):
                c[i].text = fn(row)

    stb = s.get("st") or {}
    if stb and "st" in sec:
        _st_docx(doc, stb)

    # conclusiones redactadas por el revisor
    notes = (getattr(an, "notes", "") or "").strip()
    if "conclusiones" in sec and gate.on("report.narrative"):
        doc.add_heading("Conclusiones y comentarios", level=1)
        if notes:
            for line in notes.splitlines():
                doc.add_paragraph(line)
        else:
            pr = doc.add_paragraph("Sin conclusiones redactadas. "
                                   "Puede escribirlas en el visor antes de generar el informe.")
            pr.runs[0].italic = True

    diary = (getattr(an, "diary", "") or "").strip()
    if diary and gate.on("report.narrative"):
        doc.add_heading("Diario del paciente", level=1)
        for line in diary.splitlines():
            doc.add_paragraph(line)

    if imgs.get("trend") and gate.on("analysis.hourly"):
        doc.add_heading("Tendencia de frecuencia cardíaca", level=1)
        doc.add_picture(str(imgs["trend"]), width=Cm(16.5))

    if "strips" in sec and gate.on("view.strips"):
        doc.add_heading("Tiras de ejemplo", level=1)
        for key in ("strip_0", "strip_1", "strip_2", "strip_3"):
            if key in imgs:
                doc.add_picture(str(imgs[key]), width=Cm(17))
                doc.add_paragraph()

    if gate.on("beat.classification"):
        doc.add_heading("Distribución de latidos", level=1)
        if imgs.get("bars"):
            doc.add_picture(str(imgs["bars"]), width=Cm(9))
        counts = an.class_counts()
        total = max(1, sum(counts.values()))
        p = doc.add_paragraph()
        p.add_run(" · ".join(
            f"{c}: {counts.get(c, 0)} ({100*counts.get(c,0)/total:.1f} %)"
            for c in ("N", "S", "V", "Q")))
    if imgs.get("scatter") and gate.on("analysis.heart_rate"):
        doc.add_picture(str(imgs["scatter"]), width=Cm(10))

    if "eventos" in sec:
        shown = [e for e in an.events
                 if gate.on(EVENT_FEATURES.get(e.kind, MANUAL_EVENT_FEATURE))]
        doc.add_heading("Eventos detectados", level=1)
        if shown:
            t3 = doc.add_table(rows=1, cols=4)
            t3.style = "Light Grid Accent 1"
            hdr = t3.rows[0].cells
            for i, h in enumerate(("Tipo", "Inicio", "Duración", "Detalle")):
                hdr[i].text = h
                hdr[i].paragraphs[0].runs[0].bold = True
            for e in shown[:80]:
                cells = t3.add_row().cells
                cells[0].text = EVENT_NAMES.get(e.kind, e.kind)
                cells[1].text = _fmt_dur(e.start_s)
                cells[2].text = f"{e.duration_s:.0f} s" if e.duration_s >= 1 else "<1 s"
                cells[3].text = e.detail
        else:
            doc.add_paragraph("No se detectaron eventos relevantes.")

    doc.add_heading("Métodos y limitaciones", level=1)
    mi = an.model_info
    doc.add_paragraph(
        f"Modelo de latidos: {mi.get('beat_model', '—')}. "
        f"Modelo de ritmo: {mi.get('rhythm_model', '—')}. "
        "La detección de QRS es una variante de Pan-Tompkins con umbral relativo "
        "a la energía local; la clasificación de latidos usa una red convolucional "
        "1-D y la de ritmo una regresión logística sobre características de la "
        "serie RR."
    )
    doc.add_paragraph(
        "Limitaciones: herramienta de investigación y docencia, no validada "
        "clínicamente. El rendimiento depende de la calidad de la señal, de la "
        "derivación utilizada y de la población. Las anotaciones deben ser "
        "revisadas por un profesional antes de cualquier uso asistencial. "
        "El segmento ST se mide en un único punto fijo (J+60 o J+80 ms) sobre "
        "una sola derivación y es exploratorio."
    )
    omitted = [f for f in features.values() if f.report and not f.is_visible]
    if omitted and gate.on("report.generate"):
        doc.add_heading("Funciones no informadas", level=1)
        doc.add_paragraph(
            "El informe solo presenta funciones en estado visible "
            "(configs/features.yaml). Quedan fuera "
            f"{len(omitted)} funciones exploratorias o no disponibles: "
            + ", ".join(f.id for f in omitted) + "."
        )
    foot = " · ".join(x for x in (cfg.signature, cfg.footer) if x)
    if foot:
        fp = doc.add_paragraph(foot)
        fp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    doc.save(str(path))
    return path


def _st_docx(doc, stb: dict) -> None:
    """Sección ST del informe: criterio usado y tabla de episodios."""
    doc.add_heading("Segmento ST (exploratorio)", level=1)
    lead = stb.get("lead_name") or "—"
    j_label = stb.get("j_label") or f"J+{stb.get('j_ms', 60)} ms"
    doc.add_paragraph(
        f"Medición {j_label} en la derivación {lead}. "
        f"Umbrales: depresión {stb.get('depression_uv', 100):g} µV, "
        f"elevación {stb.get('elevation_uv', 200):g} µV, "
        f"duración mínima {stb.get('min_duration_s', 60):g} s. "
        f"Latidos medidos: {stb.get('n_measured', 0)} · "
        f"desviaciones {stb.get('n_depressions', 0)} depresión / "
        f"{stb.get('n_elevations', 0)} elevación · "
        f"{stb.get('n_episodes', 0)} episodios."
    )
    eps = stb.get("episodes") or []
    if eps:
        ts = doc.add_table(rows=1, cols=5)
        ts.style = "Light Grid Accent 1"
        for i, h in enumerate(("Tipo", "Canal", "Inicio", "Duración", "Desviación máx")):
            ts.rows[0].cells[i].text = h
            ts.rows[0].cells[i].paragraphs[0].runs[0].bold = True
        for ep in eps[:60]:
            c = ts.add_row().cells
            c[0].text = EVENT_NAMES.get(str(ep.get("kind")), str(ep.get("kind")))
            c[1].text = str(ep.get("lead_name") or "—")
            c[2].text = _fmt_dur(float(ep.get("start_s", 0.0)))
            c[3].text = f"{float(ep.get('duration_s', 0.0)):.0f} s"
            c[4].text = f"{float(ep.get('max_uv', 0.0)):+.0f} µV"
    else:
        doc.add_paragraph("Sin episodios de ST con los umbrales configurados.")
    pr = doc.add_paragraph(
        "ST es una medición exploratoria (punto J fijo, una sola derivación): "
        "no está validada clínicamente y no sustituye la revisión del "
        "electrocardiograma ni el análisis del latido medio."
    )
    if pr.runs:
        pr.runs[0].italic = True


def _to_pdf(docx_path: Path) -> Path | None:
    pdf = docx_path.with_suffix(".pdf")
    soffice = _soffice()
    if soffice:
        try:
            subprocess.run(
                [soffice, "--headless", "--norestore", "--convert-to", "pdf",
                 "--outdir", str(docx_path.parent), str(docx_path)],
                check=True, timeout=180, capture_output=True,
            )
            return pdf if pdf.exists() else None
        except Exception as exc:
            log.warning("LibreOffice falló: %s", exc)
    kit = _libreoffice_kit()
    if kit:
        node, cli = kit
        try:
            subprocess.run([node, cli, "convert", "--input", str(docx_path),
                            "--output", str(pdf)],
                           check=True, timeout=180, capture_output=True)
            return pdf if pdf.exists() else None
        except Exception as exc:
            log.warning("LibreOffice Kit falló: %s", exc)
    log.info("sin conversor a PDF; se entrega DOCX/HTML")
    return None


def _build_html(an: Analysis, base: Path, imgs: dict,
                gate: FeatureGate | None = None) -> Path:
    if gate is None:
        gate = FeatureGate(load_features())
    features = gate.features
    s = an.summary
    meta = an.recording.meta_dict()
    pairs = [
        ("Paciente", meta["patient"].get("name") or "—"),
        ("Archivo", Path(meta["source"]).name),
        ("Duración", f"{meta['duration_s']/3600:.1f} h"),
        ("Derivaciones", ", ".join(meta["leads"])),
    ]
    if gate.on("analysis.qrs_detect"):
        pairs.append(("Latidos", s.get("n_beats", 0)))
    if gate.on("analysis.heart_rate"):
        pairs.append(("FC media / mín / máx",
                      f"{s.get('hr_mean',0):.0f} / {s.get('hr_min',0):.0f} / "
                      f"{s.get('hr_max',0):.0f} lpm"))
    if (gate.on("beat.ectopy_ventricular") or gate.on("beat.ectopy_supraventricular")
            or gate.on("rhythm.af")):
        pairs.append(("Carga PVC / PAC / FA",
                      f"{s.get('pvc_burden_pct',0):.2f} % / "
                      f"{s.get('pac_burden_pct',0):.2f} % / "
                      f"{s.get('af_burden_pct',0):.2f} %"))
    if gate.on("rhythm.pause"):
        pairs.append(("Pausas", s.get("n_pauses", 0)))
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in pairs)
    omite = ", ".join(f.id for f in features.values() if f.report and not f.is_visible)
    aviso = (f"<p class='muted'>El informe solo presenta funciones en estado visible "
             f"(configs/features.yaml). Fuera: {omite}.</p>") if omite else ""
    shown_events = [e for e in an.events
                    if gate.on(EVENT_FEATURES.get(e.kind, MANUAL_EVENT_FEATURE))]
    imgs_html = "".join(
        f'<figure><img src="{p.name}"><figcaption>{k}</figcaption></figure>'
        for k, p in imgs.items() if p.parent == base.parent / (base.name + "_img")
    )
    strips_html = f"<h2>Tiras de ejemplo</h2>{imgs_html}" if gate.on("view.strips") else ""
    events_html = "".join(
        f"<li>{EVENT_NAMES.get(e.kind, e.kind)} — {_fmt_dur(e.start_s)} "
        f"({e.duration_s:.0f} s)</li>" for e in shown_events[:60])
    qtc_rows = _qtc_rows(s)
    qtc_html = ""
    if qtc_rows:
        body = "".join(
            "<tr>" + "".join(f"<td>{html_mod.escape(str(v))}</td>" for v in row) + "</tr>"
            for row in qtc_rows
        )
        sel = html_mod.escape(str(s.get("qtc_formula", "bazett")))
        qtc_html = (
            "<h2>QT/QTc corregido (exploratorio)</h2>"
            '<p class="muted">QT automático y fórmulas poblacionales; no sustituye la '
            "medición manual. Se excluyen FA y FC &gt; 120 lpm. "
            f"Fórmula indicada: {sel}.</p>"
            "<table><tr><th>Fórmula</th><th>QTc mín (ms)</th><th>QTc medio (ms)</th>"
            "<th>QTc máx (ms)</th><th>Minuto del máximo</th>"
            "<th>min &gt; 450/480/500</th></tr>"
            f"{body}</table>"
        )
    stb = s.get("st") or {}
    st_html = ""
    if stb:
        eps = stb.get("episodes") or []
        st_rows = "".join(
            f"<tr><td>{EVENT_NAMES.get(str(e.get('kind')), e.get('kind'))}</td>"
            f"<td>{e.get('lead_name') or '—'}</td><td>{_fmt_dur(float(e.get('start_s', 0.0)))}</td>"
            f"<td>{float(e.get('duration_s', 0.0)):.0f} s</td>"
            f"<td>{float(e.get('max_uv', 0.0)):+.0f} µV</td></tr>"
            for e in eps[:60]
        )
        st_html = (
            "<h2>Segmento ST (exploratorio)</h2>"
            f"<p>{stb.get('j_label', 'J+60 ms')} · derivación {stb.get('lead_name') or '—'} · "
            f"umbrales {stb.get('depression_uv', 100):g}/{stb.get('elevation_uv', 200):g} µV · "
            f"{stb.get('n_episodes', 0)} episodios</p>"
            + (f"<table><tr><th>Tipo</th><th>Canal</th><th>Inicio</th><th>Duración</th>"
               f"<th>Desviación máx</th></tr>{st_rows}</table>" if eps else "")
            + '<p class="muted">Medición exploratoria, no validada clínicamente.</p>'
        )
    hrv_html = ""
    if s.get("hrv"):
        lineas = hrv_mod.summary_lines(s["hrv"]) + hrv_mod.day_night_lines(
            s.get("hrv_day"), s.get("hrv_night"), s.get("hrv_windows") or {})
        hrv_html = ('<h2>Variabilidad de la frecuencia cardíaca</h2><ul>'
                    + "".join(f"<li>{ln}</li>" for ln in lineas) + "</ul>")
    pwave_html = ""
    pw, morph = s.get("pwave") or {}, s.get("qrs_morph") or {}
    if pw or morph:
        filas = "".join(
            "<tr>" + "".join(f"<td>{html_mod.escape(str(v))}</td>" for v in row) + "</tr>"
            for row in _pr_rows(pw)
        )
        parrafos = "".join(
            f"<p>{html_mod.escape(txt)}</p>"
            for txt in (_pwave_sentence(pw), _qrs_morph_sentence(morph)) if txt
        )
        tabla = (f"<table><tr><th>Medida</th><th>PR (ms)</th><th>Hora</th></tr>{filas}"
                 "</table>" if filas else "")
        pwave_html = ("<h2>Conducción auriculoventricular y QRS</h2>"
                      f"{parrafos}{tabla}"
                      '<p class="muted">Medición exploratoria (onda P por plantilla del '
                      "minuto, PR al inicio del QRS del latido promediado). No sustituye "
                      "la lectura del ECG.</p>")
    html = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<title>Informe HolterIA</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:32px;color:#182028;max-width:1100px}}
h1{{font-size:26px;margin-bottom:2px}} .muted{{color:#6e7a88;font-size:13px}}
.warn{{background:#fdecee;border-left:4px solid #b02a37;padding:10px 14px;margin:16px 0;font-weight:600;color:#7a1c26}}
table{{border-collapse:collapse;margin:12px 0}} td{{border-bottom:1px solid #e6eaee;padding:6px 14px 6px 0}}
td:first-child{{color:#6e7a88;width:220px}}
figure{{margin:18px 0}} img{{max-width:100%;border:1px solid #e6eaee}}
figcaption{{color:#6e7a88;font-size:12px;margin-top:4px}}
</style></head><body>
<h1>Informe de análisis Holter</h1>
<div class="muted">HolterIA v{config.VERSION} · {datetime.now().strftime('%d/%m/%Y %H:%M')}</div>
<div class="warn">{config.DISCLAIMER}</div>
<table>{rows}</table>
{aviso}
{qtc_html}
{st_html}
{pwave_html}
{hrv_html}
{strips_html}
<h2>Eventos</h2>
<ul>{events_html}</ul>
<p class="muted">{config.DISCLAIMER}</p>
</body></html>"""
    path = base.with_suffix(".html")
    path.write_text(html, encoding="utf-8")
    return path

