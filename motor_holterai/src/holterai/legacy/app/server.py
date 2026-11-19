"""Servidor HTTP local de HolterIA (sin dependencias externas)."""
from __future__ import annotations

import json
import math
import mimetypes
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from holterai import store
from holterai.edits import apply_edits, save_edits
from holterai.hscribe_beats import read_beats
from holterai.report_config import ReportConfig, load_report_config, save_report_config
from holterai.legacy.app.core import config
from holterai.legacy.app.core.logging import get_logger
from holterai.legacy.app.core.models import Analysis, Beat, Event, Recording
from holterai.legacy.app.ingest import base as ingest
from holterai.legacy.app.pipeline import Options, analyze, apply_st
from holterai.legacy.app.report import report as report_mod
from holterai.legacy.app.signal import artifact as artifact_mod
from holterai.legacy.app.signal import st as ST
from holterai.legacy.app.signal.filters import display_filter, display_filters

log = get_logger(__name__)

# Validación contra HScribe (matriz bxb)
AAMI_CLASSES = ("N", "S", "V", "F", "Q")
REF_MISSED = "AUSENTE"      # HScribe tiene un latido y el motor no
OURS_EXTRA = "SIN_REF"      # el motor marca un latido y HScribe no
DEFAULT_TOL_S = 0.150       # ancho de la ventana de emparejamiento (EC57)
REF_DENSITY_MIN = 0.7       # densidad mínima de HScribe frente a su propio RR
QUALITY_OK_MIN = 0.7        # fracción de minutos buenos para ventana "buena"
SIDECAR_SUFFIX = ".beats.dat"


class Session:
    """Estado de la aplicación (un registro activo a la vez)."""

    def __init__(self):
        self.lock = threading.RLock()
        self.recording: Recording | None = None
        self.analysis: Analysis | None = None
        self.state = "idle"          # idle | loading | analyzing | done | error
        self.message = ""
        self.progress = 0.0
        self.error = ""
        self.thread: threading.Thread | None = None
        self.overview: dict | None = None
        self._xf: np.ndarray | None = None
        self._lead: int = 0
        self._history: list[dict] = []
        self._last_opts: dict = {}
        self.inverted: set[int] = set()
        self._quality_cache_key: tuple | None = None
        self._quality_cache: dict | None = None
        self._mask_cache_key: tuple | None = None
        self._mask_cache = None

    # -- utilidades de revisión ---------------------------------------
    def _ensure_xf(self):
        if self._xf is None and self.recording is not None:
            from holterai.legacy.app.signal.filters import preprocess
            lead = int(self.analysis.quality.get("lead", 0)) if self.analysis else 0
            x = np.asarray(self.recording.samples[:, lead], dtype=np.float64)
            self._xf = preprocess(x, self.recording.fs, "display")
            self._lead = lead
        return self._xf, self._lead

    def counts(self) -> dict:
        if self.analysis is None:
            return {}
        return self.analysis.class_counts()

    def _save_edits(self, action: str = "edit") -> None:
        try:
            if self.analysis is not None and self.recording is not None and self.recording.source:
                save_edits(self.recording.source, self.analysis, self._history)
                store.log_audit(
                    Path(self.recording.source).stem,
                    action,
                    f"{len(self.analysis.beats)} latidos",
                )
        except Exception:
            log.exception("no se pudieron guardar las correcciones")

    def quality_map(self, bin_s: int = 60) -> dict:
        """Mapa de calidad por canal y por minuto con la máscara de artefactos.

        Antes se decidía con un umbral fijo de ruido (``noise < 0,22``) que
        marcaba malo cualquier registro con línea de base movida. Ahora manda
        ``signal/artifact.py``: los índices por ventana de 2 s (``hf_uv``,
        ``flat``, ``sat``, ``step``, ``kurt``, ``sqi``, ``corr``, ``disc``,
        ``motion``) se agregan al minuto y la ventana es buena si menos de la
        mitad de sus ventanas de 2 s están malas.
        """
        with self.lock:
            if self.recording is None:
                raise RuntimeError("no hay registro")
            rec = self.recording
            fs = rec.fs
            n = rec.n_samples
            duration = rec.duration_s
            try:
                mask = self._artifact_mask()
            except Exception as exc:  # nunca tumbar la UI por la calidad
                log.warning("no se pudo calcular la máscara de artefactos: %s", exc)
                mask = None
            step = max(1, int(round(bin_s * fs)))
            leads = []
            if mask is not None and mask.n_windows:
                win_s = mask.win_s
                nb = max(1, int(np.ceil(duration / float(bin_s))))
                edges = np.arange(nb + 1) * (float(bin_s) / win_s)
                for j in range(rec.n_leads):
                    bad = mask.bad[:, j] if j < mask.n_leads else np.zeros(mask.n_windows, bool)
                    hf = (mask.metrics["hf_uv"][:, j] if j < mask.n_leads
                          else np.zeros(mask.n_windows, np.float32))
                    if j < mask.n_leads:
                        disc = (mask.reasons[:, j]
                                & np.uint16(1 << artifact_mod.CRITERIA.index("disc"))) != 0
                        sat = (mask.reasons[:, j]
                               & np.uint16(1 << artifact_mod.CRITERIA.index("sat"))) != 0
                    else:
                        disc = sat = np.zeros(mask.n_windows, bool)
                    vals = []
                    for b in range(nb):
                        a0 = int(min(edges[b], mask.n_windows))
                        a1 = int(min(edges[b + 1], mask.n_windows))
                        if a1 <= a0:
                            continue
                        bads = bad[a0:a1]
                        ok = float(bads.mean()) < 0.5
                        vals.append({
                            "t": round(b * bin_s, 1),
                            "noise": round(float(np.median(hf[a0:a1])) / 1000.0, 4),
                            "flat": bool(disc[a0:a1].mean() >= 0.5),
                            "saturated": bool(sat[a0:a1].mean() >= 0.5),
                            "unreadable": bool(mask.unreadable[a0:a1].mean() >= 0.5),
                            "bad_pct": round(100.0 * float(bads.mean()), 1),
                            "ok": bool(ok),
                        })
                    leads.append({"name": rec.leads[j] if j < len(rec.leads) else f"L{j+1}",
                                  "bins": vals})
            else:
                for j in range(rec.n_leads):
                    vals = []
                    for a in range(0, n, step):
                        seg = np.asarray(rec.samples[a:a + step, j], dtype=np.float32)
                        if seg.size < 3:
                            continue
                        vals.append({"t": round(a / fs, 1), "noise": 0.0,
                                     "flat": False, "saturated": False,
                                     "unreadable": False, "bad_pct": 0.0,
                                     "ok": True})
                    leads.append({"name": rec.leads[j] if j < len(rec.leads) else f"L{j+1}",
                                  "bins": vals})
        out = {"duration_s": round(duration, 1), "bin_s": bin_s, "leads": leads}
        if mask is not None and mask.n_windows:
            out.update({
                "unreadable_pct": round(100.0 * float(mask.unreadable.mean()), 2),
                "unreadable_s": round(float(mask.unreadable_s), 1),
                "per_lead_bad_pct": [round(float(v), 1) for v in mask.bad_pct],
                "unreadable_spans": mask.spans[:200],
                "source": "artifact",
            })
        return out

    def _artifact_mask(self):
        """Máscara de artefactos del registro activo (cacheada)."""
        if self.recording is None:
            raise RuntimeError("no hay registro")
        rec = self.recording
        key = (rec.source, rec.n_samples, float(rec.fs))
        if self._mask_cache_key == key and self._mask_cache is not None:
            return self._mask_cache
        mask = artifact_mod.quality_mask(rec.samples, rec.fs)
        self._mask_cache_key = key
        self._mask_cache = mask
        return mask

    def _quality_bins(self, lead: int = 0) -> dict:
        """Minutos buenos/malos de la derivación analizada (cacheado)."""
        if self.recording is None:
            return {"bin_s": 60.0, "ok": []}
        key = (self.recording.source, int(lead), self.recording.n_samples)
        if self._quality_cache_key == key and self._quality_cache is not None:
            return self._quality_cache
        q = self.quality_map(60)
        leads = q.get("leads") or []
        bins = leads[min(max(0, int(lead)), len(leads) - 1)]["bins"] if leads else []
        out = {"bin_s": float(q.get("bin_s", 60.0)),
               "ok": [bool(b.get("ok")) for b in bins]}
        self._quality_cache_key = key
        self._quality_cache = out
        return out

    def validation(self, *, window_s: float = 300.0, quality: str = "all",
                   kind: str = "all", cls: str = "", label: str = "",
                   limit: int = 500, offset_s: float = 0.0,
                   tol_ms: float = DEFAULT_TOL_S * 1000.0) -> dict:
        """Compara el examen contra el `Beats.dat` de HScribe (matriz bxb)."""
        with self.lock:
            an = self.analysis
            rec = self.recording
            if an is None:
                raise RuntimeError("no hay análisis")
            source = rec.source if rec is not None else ""
            duration = rec.duration_s if rec is not None else 0.0
            lead = int((an.quality or {}).get("lead", 0) or 0)
            ot = [float(b.t) for b in an.beats]
            oc = [str(b.cls) for b in an.beats]
            oi = [int(b.idx) for b in an.beats]
            of = [float(b.conf) for b in an.beats]
            qbins = self._quality_bins(lead)
        path = find_reference(source)
        if path is None:
            return {
                "ok": True, "available": False,
                "reason": "sin Beats.dat de HScribe para este examen",
                "expected": f"{Path(source).stem}{SIDECAR_SUFFIX}" if source else "",
            }
        beats = read_beats(path)
        if not beats:
            return {
                "ok": True, "available": False,
                "reason": f"{path.name} no tiene latidos legibles",
                "expected": path.name,
            }
        ref_t = [b.time_ms / 1000.0 for b in beats]
        ref_l = [int(b.label) for b in beats]
        out = build_validation(
            duration_s=duration, ref_times=ref_t, ref_labels=ref_l,
            our_times=ot, our_classes=oc, our_idxs=oi, our_confs=of,
            window_s=window_s, quality=quality, kind=kind, cls=cls, label=label,
            limit=limit, offset_s=offset_s, tol_s=float(tol_ms) / 1000.0,
            ok_bins=qbins["ok"], bin_s=float(qbins["bin_s"]),
        )
        out["reference"] = {
            "path": path.name,
            "n": len(beats),
            "labels": {str(k): sum(1 for x in ref_l if x == k)
                       for k in sorted(set(ref_l))},
            "first_s": round(ref_t[0], 3) if ref_t else None,
            "last_s": round(ref_t[-1], 3) if ref_t else None,
        }
        out["lead"] = lead
        return out

    def templates(self, cls_name: str = "V") -> list[dict]:
        """Agrupa por morfología los latidos de una clase."""
        from holterai.legacy.app.signal import features as FT
        from holterai.legacy.app.signal.rhythm import cluster_windows
        with self.lock:
            if self.analysis is None:
                raise RuntimeError("no hay análisis")
            xf, _ = self._ensure_xf()
            rec = self.recording
            sel = [b for b in self.analysis.beats if b.cls == cls_name]
            if len(sel) < 5:
                return []
            idx = np.array([b.idx for b in sel], dtype=np.int64)
            wins = FT.extract_windows_batch(xf, rec.fs, idx)
            labels = cluster_windows(wins, thr=0.85, min_frac=0.04)
        out = []
        for k in range(labels.max() + 1 if labels.size and labels.max() >= 0 else 0):
            members = np.flatnonzero(labels == k)
            if members.size < 3:
                continue
            out.append({
                "id": int(k),
                "n": int(members.size),
                "cls": cls_name,
                "examples": [int(sel[i].idx) for i in members[:400]],
                "times": [round(float(sel[i].t), 1) for i in members[:5]],
            })
        out.sort(key=lambda c: -c["n"])
        return out

    def split_template(self, idxs: list[int], thr: float = 0.94) -> list[dict]:
        """Divide una plantilla en subgrupos más finos para revisión."""
        from holterai.legacy.app.signal import features as FT
        from holterai.legacy.app.signal.rhythm import cluster_windows
        with self.lock:
            if self.analysis is None:
                raise RuntimeError("no hay análisis")
            wanted = set(int(i) for i in idxs)
            sel = [b for b in self.analysis.beats if b.idx in wanted]
            if len(sel) < 4:
                return []
            xf, _ = self._ensure_xf()
            rec = self.recording
            arr = np.array([b.idx for b in sel], dtype=np.int64)
            wins = FT.extract_windows_batch(xf, rec.fs, arr)
            labels = cluster_windows(wins, thr=float(thr), min_frac=0.02)
        out = []
        for k in range(labels.max() + 1 if labels.size and labels.max() >= 0 else 0):
            members = np.flatnonzero(labels == k)
            if members.size < 2:
                continue
            out.append({
                "id": int(k), "n": int(members.size), "cls": sel[0].cls,
                "examples": [int(sel[i].idx) for i in members[:400]],
                "times": [round(float(sel[i].t), 1) for i in members[:5]],
            })
        out.sort(key=lambda c: -c["n"])
        return out

    def apply_template(self, idxs: list[int], cls: str) -> dict:
        with self.lock:
            if self.analysis is None:
                raise RuntimeError("no hay análisis")
            wanted = set(int(i) for i in idxs)
            hist = []
            for b in self.analysis.beats:
                if b.idx in wanted and b.cls != cls:
                    hist.append((b.idx, b.cls))
                    b.cls = cls
            if hist:
                self._history.append({"kind": "class", "changes": hist})
            self._save_edits("template")
            return {"changed": len(hist), "counts": self.analysis.class_counts()}

    def undo(self) -> dict:
        with self.lock:
            # las inversiones de canal tienen su propio "deshacer"
            while self._history and self._history[-1].get("kind") not in (
                    "class", "delete", "insert"):
                self._history.pop()
            if not self._history or self.analysis is None:
                return {"changed": 0, "counts": self.counts()}
            op = self._history.pop()
            changed = 0
            if op.get("kind") == "class":
                by_idx = {b.idx: b for b in self.analysis.beats}
                for idx, old in op.get("changes", []):
                    if idx in by_idx:
                        by_idx[idx].cls = old
                        by_idx[idx].edited = True
                        changed += 1
            elif op.get("kind") == "delete":
                data = op.get("beat") or {}
                if data:
                    beat = Beat(**data)
                    self.analysis.beats.append(beat)
                    self.analysis.beats.sort(key=lambda b: b.t)
                    if beat.idx in self.analysis.deleted_idxs:
                        self.analysis.deleted_idxs.remove(beat.idx)
                    changed = 1
            elif op.get("kind") == "insert":
                idx = int(op.get("idx", -1))
                before = len(self.analysis.beats)
                self.analysis.beats = [b for b in self.analysis.beats if b.idx != idx]
                self.analysis.inserted_beats = [
                    x for x in self.analysis.inserted_beats if int(x.get("idx", -1)) != idx
                ]
                changed = before - len(self.analysis.beats)
            if changed:
                for i, b in enumerate(self.analysis.beats):
                    b.rr_prev = b.t - self.analysis.beats[i - 1].t if i else 0.0
                    b.rr_next = (
                        self.analysis.beats[i + 1].t - b.t
                        if i + 1 < len(self.analysis.beats)
                        else 0.0
                    )
            self._save_edits("undo")
            return {"changed": changed, "counts": self.analysis.class_counts()}

    # -- carga ---------------------------------------------------------
    def load(self, path: str, fs: float | None = None) -> dict:
        with self.lock:
            self.state = "loading"
            self.progress = 0.0
            self.message = "Leyendo archivo"
            self.error = ""
            self.analysis = None
        try:
            t0 = time.time()
            rec = ingest.load(path, fs=fs)
            with self.lock:
                self.recording = rec
                self.inverted = set()
                self._xf = None
                self.overview = _build_overview(rec)
                self.state = "idle"
                self.progress = 1.0
                self.message = f"Cargado en {time.time()-t0:.1f}s"
            return rec.meta_dict()
        except Exception as exc:
            log.exception("error cargando %s", path)
            with self.lock:
                self.state = "error"
                self.error = str(exc)
                self.message = "Error de carga"
            raise

    # -- análisis ------------------------------------------------------
    def start_analysis(self, opts_dict: dict) -> None:
        with self.lock:
            if self.recording is None:
                raise RuntimeError("no hay registro cargado")
            if self.state == "analyzing":
                return
            self.state = "analyzing"
            self.progress = 0.0
            self.message = "Iniciando"
            self.error = ""
            opts = _options_from_dict(opts_dict)
            self._last_opts = dict(opts_dict or {})
            rec = self.recording

        def _work():
            try:
                def progress(msg, frac):
                    with self.lock:
                        self.message = msg
                        self.progress = float(frac)
                res = analyze(rec, opts, progress)
                applied = apply_edits(res, rec.source)
                exam_id = Path(rec.source).stem
                patient_id = None
                if rec.patient.patient_id or rec.patient.name:
                    patient_id = store.upsert_patient(
                        patient_code=rec.patient.patient_id or exam_id,
                        name=rec.patient.name,
                        sex=rec.patient.sex,
                        age=rec.patient.age,
                        notes=rec.patient.notes,
                    )
                store.upsert_exam(
                    exam_id=exam_id,
                    source_path=rec.source,
                    patient_id=patient_id,
                    duration_s=rec.duration_s,
                    n_beats=len(res.beats),
                )
                with self.lock:
                    self.analysis = res
                    self.state = "done"
                    self.progress = 1.0
                    self.message = f"Análisis completado en {res.elapsed_s:.1f}s"
                    if applied.get("changed"):
                        self.message += f" · {applied['changed']} correcciones"
            except Exception as exc:
                log.exception("error en análisis")
                with self.lock:
                    self.state = "error"
                    self.error = str(exc)
                    self.message = "Error de análisis"

        self.thread = threading.Thread(target=_work, daemon=True)
        self.thread.start()

    def invert_lead(self, lead: int, undo: bool = False) -> dict:
        """Invierte (o restaura) la polaridad de un canal y reanaliza.

        Nunca se aplica sola: la pide el revisor desde el aviso de polaridad.
        `undo=True` devuelve el canal a su polaridad original.
        """
        with self.lock:
            rec = self.recording
            if rec is None:
                raise RuntimeError("no hay registro cargado")
            if self.state == "analyzing":
                raise RuntimeError("espera a que termine el análisis")
            j = int(lead)
            if not 0 <= j < rec.n_leads:
                raise ValueError("canal fuera de rango")
            # pedir dos veces lo mismo no vuelve a invertir la señal
            if undo == (j not in self.inverted):
                return {"ok": True, "changed": False, "lead": j,
                        "inverted": sorted(self.inverted), "leads": list(rec.leads)}
            if not rec.samples.flags.writeable:
                rec.samples = np.array(rec.samples, dtype=np.float32, copy=True)
            rec.samples[:, j] *= -1.0
            if undo:
                self.inverted.discard(j)
            else:
                self.inverted.add(j)
            self._xf = None
            self._history.append({"kind": "invert", "lead": j, "undo": bool(undo)})
            opts = dict(self._last_opts)
        self.start_analysis(opts)
        return {"ok": True, "changed": True, "lead": j,
                "inverted": sorted(self.inverted), "leads": list(rec.leads)}

    # -- resultados ----------------------------------------------------
    def status(self) -> dict:
        with self.lock:
            return {
                "state": self.state,
                "progress": round(self.progress, 3),
                "message": self.message,
                "error": self.error,
                "has_recording": self.recording is not None,
                "has_analysis": self.analysis is not None,
                "inverted_leads": sorted(self.inverted),
            }

    def result(self, include_beats: bool = True) -> dict:
        with self.lock:
            if self.analysis is None:
                raise RuntimeError("no hay análisis")
            d = self.analysis.to_dict(include_beats=include_beats)
            d["status"] = self.status()
            return d

    def arrhythmogram(self) -> dict:
        """Arritmiagrama por tipo y acoplamiento de V del análisis actual."""
        with self.lock:
            if self.analysis is None:
                raise RuntimeError("no hay análisis")
            if self.recording is not None:
                duration = self.recording.duration_s
            else:
                duration = float(self.analysis.summary.get("duration_h") or 0.0) * 3600.0
            return build_arrhythmogram(self.analysis, duration)

    def beats_page(self, start: int, count: int) -> dict:
        with self.lock:
            if self.analysis is None:
                raise RuntimeError("no hay análisis")
            beats = self.analysis.beats
            chunk = beats[start:start + count]
            return {
                "total": len(beats),
                "start": start,
                "beats": [b.to_dict() for b in chunk],
            }

    # -- ST exploratorio -----------------------------------------------
    def st_block(self) -> dict:
        with self.lock:
            if self.analysis is None:
                raise RuntimeError("no hay análisis")
            return dict(self.analysis.summary.get("st") or {})

    def recompute_st(self, overrides: dict | None = None) -> dict:
        """Recalcula los episodios ST sin volver a detectar latidos.

        Con el mismo punto J y el mismo canal solo se reaplican los umbrales
        sobre los valores `st_uv` guardados por latido; si cambia el punto J o
        el canal se vuelve a medir ST sobre la señal filtrada, pero no se
        vuelve a detectar ni a clasificar ningún QRS.
        """
        ov = {k: v for k, v in (overrides or {}).items() if v is not None}
        if "j_ms" in ov:  # la UI manda `j_ms`; los criterios usan `st_j_ms`
            ov["st_j_ms"] = ov.pop("j_ms")
        raw = _load_criteria()
        raw.update(ov)
        crit = ST.StCriteria.from_criteria(raw)
        with self.lock:
            analysis = self.analysis
            rec = self.recording
            if analysis is None:
                raise RuntimeError("no hay análisis")
            current = analysis.summary.get("st") or {}
            lead = int(raw.get("st_lead", current.get("lead", analysis.quality.get("lead", 0))))
            n_leads = rec.n_leads if rec is not None else 1
            if not 0 <= lead < n_leads:
                lead = int(analysis.quality.get("lead", 0))
            remeasure = (
                int(current.get("j_ms", -1)) != int(crit.j_ms)
                or int(current.get("lead", -1)) != lead
            )
            if remeasure:
                if rec is None:
                    raise RuntimeError("sin registro para volver a medir el ST")
                xf, cached_lead = self._ensure_xf()
                if xf is None or int(cached_lead) != lead:
                    from holterai.legacy.app.signal.filters import preprocess

                    xf = preprocess(np.asarray(rec.samples[:, lead], dtype=np.float64),
                                    rec.fs, "display")
                ST.remeasure(analysis.beats, xf, rec.fs, crit, lead=lead)
            name = rec.leads[lead] if rec is not None and lead < len(rec.leads) else ""
            block = apply_st(analysis, crit, lead=lead, lead_name=name)
            events = [e.to_dict() for e in analysis.events
                      if e.kind in ("st_depression", "st_elevation")]
            self._save_edits("st")
        return {"st": block, "events": events, "remeasured": bool(remeasure)}

    def waveform(self, t0: float, t1: float, max_points: int,
                 notch_hz=0.0, lowpass_hz=0.0) -> dict:
        """Envolvente de la señal para dibujar.

        ``notch_hz`` (0/50/60) y ``lowpass_hz`` (0/25/30/35) se aplican a una
        copia de la señal: solo afectan la vista, nunca el análisis ni las
        medidas de ST/QT. El registro y el resultado quedan intactos.
        """
        with self.lock:
            rec = self.recording
            analysis = self.analysis
        if rec is None:
            raise RuntimeError("no hay registro")
        t0 = max(0.0, float(t0))
        t1 = min(rec.duration_s, float(t1))
        if t1 <= t0:
            t1 = min(rec.duration_s, t0 + 10.0)
        a, seg = rec.segment(t0, t1)
        notch_hz, lowpass_hz = display_filters(notch_hz, lowpass_hz)
        view = display_filter(seg, rec.fs, notch_hz, lowpass_hz)
        payload = _envelope_payload(rec, view, a, max_points, analysis, t0, t1)
        payload["filter"] = {"notch_hz": notch_hz, "lowpass_hz": lowpass_hz,
                             "scope": "display"}
        return payload

    def edit_beat(self, idx: int, cls: str) -> dict:
        with self.lock:
            if self.analysis is None:
                raise RuntimeError("no hay análisis")
            for b in self.analysis.beats:
                if b.idx == idx:
                    if b.cls != cls:
                        self._history.append(
                            {"kind": "class", "changes": [[b.idx, b.cls]]}
                        )
                    b.cls = cls
                    b.edited = True
                    self._save_edits("beat")
                    return b.to_dict()
        raise KeyError("latido no encontrado")

    def delete_beat(self, idx: int) -> dict:
        with self.lock:
            if self.analysis is None:
                raise RuntimeError("no hay análisis")
            before = len(self.analysis.beats)
            target = next((b for b in self.analysis.beats if b.idx == idx), None)
            if target is None:
                raise KeyError("latido no encontrado")
            self._history.append({"kind": "delete", "beat": target.to_dict()})
            self.analysis.beats = [b for b in self.analysis.beats if b.idx != idx]
            if len(self.analysis.beats) == before:
                raise KeyError("latido no encontrado")
            if idx not in self.analysis.deleted_idxs:
                self.analysis.deleted_idxs.append(idx)
            self._save_edits("delete")
            return {"changed": 1, "counts": self.analysis.class_counts()}

    def insert_beat(self, t: float, cls: str) -> dict:
        with self.lock:
            if self.analysis is None:
                raise RuntimeError("no hay análisis")
            if cls not in {"N", "S", "V", "F", "Q"}:
                raise ValueError("clase inválida")
            fs = self.recording.fs if self.recording else 180.0
            idx = int(round(t * fs))
            beat = Beat(idx=idx, t=float(t), cls=cls, conf=0.0, edited=True)
            self.analysis.beats.append(beat)
            self.analysis.beats.sort(key=lambda b: b.t)
            for i, b in enumerate(self.analysis.beats):
                b.rr_prev = b.t - self.analysis.beats[i - 1].t if i else 0.0
                b.rr_next = (
                    self.analysis.beats[i + 1].t - b.t
                    if i + 1 < len(self.analysis.beats)
                    else 0.0
                )
            self.analysis.inserted_beats.append({"idx": idx, "t": float(t), "cls": cls})
            self._history.append({"kind": "insert", "idx": idx})
            self._save_edits("insert")
            return {"beat": beat.to_dict(), "counts": self.analysis.class_counts()}

    def save_events(self, rows: list[dict]) -> dict:
        with self.lock:
            if self.analysis is None:
                raise RuntimeError("no hay análisis")
            events: list[Event] = []
            for row in rows:
                kind = str(row.get("kind", "")).strip()
                if not kind:
                    continue
                try:
                    start = float(row.get("start_s", row.get("time", 0)))
                    end = float(row.get("end_s", start))
                except (TypeError, ValueError):
                    continue
                events.append(
                    Event(
                        kind=kind,
                        start_s=start,
                        end_s=max(start, end),
                        severity=str(row.get("severity", "info")),
                        detail=str(row.get("detail", "")),
                        beats=[int(x) for x in row.get("beats", []) if str(x).isdigit()],
                    )
                )
            events.sort(key=lambda e: e.start_s)
            self.analysis.events = events
            self._save_edits("events")
            return {"events": len(events)}


# ---------------------------------------------------------------------------
# Serialización de forma de onda
# ---------------------------------------------------------------------------
def _envelope_payload(rec: Recording, seg: np.ndarray, a: int, max_points: int,
                      analysis: Analysis | None, t0: float, t1: float) -> dict:
    n, L = seg.shape
    max_points = max(64, min(int(max_points), 8000))
    payload = {"t0": round(t0, 4), "t1": round(t1, 4), "fs": rec.fs,
               "leads": list(rec.leads), "mode": "line", "data": []}
    if n == 0:
        return payload

    k = max(1, int(np.ceil(n / max_points)))
    m = n // k
    if m * k < n:
        seg = seg[: m * k]
    dense = np.asarray(seg, dtype=np.float32).reshape(m, k, L)
    lo = dense.min(axis=1)
    hi = dense.max(axis=1)
    payload["mode"] = "envelope" if k > 1 else "line"
    payload["data"] = [
        {"min": np.round(lo[:, j], 4).tolist(),
         "max": np.round(hi[:, j], 4).tolist()} for j in range(L)
    ]
    payload["decimation"] = int(k)

    if analysis is not None:
        t_start = t0
        t_end = t0 + n / rec.fs
        marks = []
        for b in analysis.beats:
            if b.t < t_start or b.t > t_end:
                continue
            marks.append({"idx": b.idx, "t": round(b.t - t_start, 4),
                          "cls": b.cls, "conf": round(b.conf, 2),
                          "edited": b.edited})
        payload["beats"] = marks
    return payload


def _build_overview(rec: Recording, points: int = 1600) -> dict:
    """Envolvente de todo el registro para la barra de navegación."""
    n = rec.n_samples
    k = max(1, int(np.ceil(n / points)))
    m = n // k
    seg = np.asarray(rec.samples[: m * k], dtype=np.float32).reshape(m, k, rec.n_leads)
    lo = seg.min(axis=1)
    hi = seg.max(axis=1)
    return {
        "duration_s": rec.duration_s,
        "n": int(m),
        "leads": list(rec.leads),
        "min": np.round(lo, 3).tolist(),
        "max": np.round(hi, 3).tolist(),
    }


# ---------------------------------------------------------------------------
# Arritmiagrama y acoplamiento ventricular
# ---------------------------------------------------------------------------
ARRHYTHMOGRAM_ROWS: tuple[tuple[str, str], ...] = (
    ("pause", "Pausa"),
    ("brady", "Bradicardia"),
    ("tachy", "Taquicardia"),
    ("afib", "FA"),
    ("vt_run", "TV"),
    ("bigeminy", "Bigeminismo"),
)
COUPLING_BIN_MS = 20.0
COUPLING_MIN_MS = 200.0
COUPLING_MAX_MS = 2500.0


def build_arrhythmogram(analysis: Analysis, duration_s: float) -> dict:
    """Arma las filas del arritmiagrama y el histograma de acoplamiento de V.

    Solo lee el análisis existente. El acoplamiento usa la misma definición que
    el resumen (`rhythm.ventricular_events`): V precedido por un latido no V y
    RR previo entre 200 y 2500 ms.
    """
    duration = max(0.0, float(duration_s or 0.0))
    events = list(analysis.events or [])
    rows: list[dict] = []
    for kind, label in ARRHYTHMOGRAM_ROWS:
        picked = sorted((e for e in events if e.kind == kind), key=lambda e: e.start_s)
        items = []
        for event in picked:
            start = max(0.0, float(event.start_s))
            end = max(start, float(event.end_s))
            items.append({
                "start_s": round(start, 3),
                "end_s": round(end, 3),
                "duration_s": round(end - start, 3),
                "severity": event.severity,
                "detail": event.detail,
                "beat": int(event.beats[0]) if event.beats else None,
            })
        rows.append({
            "kind": kind,
            "label": label,
            "n": len(items),
            "total_s": round(sum(x["duration_s"] for x in items), 3),
            "max_s": round(max((x["duration_s"] for x in items), default=0.0), 3),
            "events": items,
        })

    beats = list(analysis.beats or [])
    points: list[tuple[float, int, float]] = []
    for i in range(1, len(beats)):
        beat = beats[i]
        if beat.cls != "V" or beats[i - 1].cls == "V":
            continue
        value = float(beat.rr_prev) * 1000.0
        if not (COUPLING_MIN_MS < value < COUPLING_MAX_MS):
            continue
        points.append((value, int(beat.idx), float(beat.t)))
    points.sort(key=lambda p: p[0])

    bins: list[dict] = []
    start_ms: float | None = None
    end_ms: float | None = None
    if points:
        start_ms = math.floor(points[0][0] / COUPLING_BIN_MS) * COUPLING_BIN_MS
        n_bins = int(math.floor((points[-1][0] - start_ms) / COUPLING_BIN_MS)) + 1
        end_ms = start_ms + n_bins * COUPLING_BIN_MS
        counts = [0] * n_bins
        first: list[tuple[int, float] | None] = [None] * n_bins
        for value, idx, t in points:
            k = max(0, min(n_bins - 1, int((value - start_ms) // COUPLING_BIN_MS)))
            counts[k] += 1
            if first[k] is None:
                first[k] = (idx, t)
        for k in range(n_bins):
            hit = first[k]
            bins.append({
                "start_ms": round(start_ms + k * COUPLING_BIN_MS, 1),
                "end_ms": round(start_ms + (k + 1) * COUPLING_BIN_MS, 1),
                "n": counts[k],
                "idx": hit[0] if hit else None,
                "t_s": round(hit[1], 3) if hit else None,
            })

    return {
        "duration_s": round(duration, 3),
        "rows": rows,
        "coupling": {
            "bin_ms": COUPLING_BIN_MS,
            "n": len(points),
            "median_ms": round(points[len(points) // 2][0], 1) if points else None,
            "mean_ms": round(sum(p[0] for p in points) / len(points), 1) if points else None,
            "min_ms": round(points[0][0], 1) if points else None,
            "max_ms": round(points[-1][0], 1) if points else None,
            "start_ms": round(start_ms, 1) if start_ms is not None else None,
            "end_ms": round(end_ms, 1) if end_ms is not None else None,
            "bins": bins,
        },
    }


def _options_from_dict(d: dict) -> Options:
    o = Options()
    if not d:
        d = {}
    o.use_ecgfounder = bool(d.get("use_ecgfounder", o.use_ecgfounder))
    o.ecgfounder_stride_s = float(d.get("ecgfounder_stride_s", o.ecgfounder_stride_s))
    o.pause_s = float(d.get("pause_s", o.pause_s))
    o.brady_bpm = float(d.get("brady_bpm", o.brady_bpm))
    o.tachy_bpm = float(d.get("tachy_bpm", o.tachy_bpm))
    o.af_threshold = float(d.get("af_threshold", o.af_threshold))
    o.af_min_s = float(d.get("af_min_s", o.af_min_s))
    o.force_lead = int(d.get("lead", o.force_lead))
    o.pacing_mode = str(d.get("pacing_mode", o.pacing_mode))
    o.crt_mode = str(d.get("crt_mode", o.crt_mode))
    crit = _load_criteria()
    o.pause_s = float(crit.get("pause_ms", 2000)) / 1000.0
    o.brady_bpm = float(crit.get("bradycardia_bpm", 50))
    o.tachy_bpm = float(crit.get("tachycardia_bpm", 120))
    o.qtc_formula = str(crit.get("qtc_formula", "bazett"))
    st = ST.StCriteria.from_criteria(crit)
    o.st_j_ms = int(st.j_ms)
    o.st_depression_uv = float(st.depression_uv)
    o.st_elevation_uv = float(st.elevation_uv)
    o.st_min_s = float(st.min_s)
    return o


def _criteria_path() -> Path:
    return config.DATA_DIR / "criteria.json"


def _criteria_defaults() -> dict:
    try:
        from holterai.config import load_criteria
        return load_criteria().model_dump()
    except Exception:
        return {
            "sv_prematurity_pct": 25.0, "pause_ms": 2000,
            "st_depression_uv": 100, "st_elevation_uv": 200,
            "st_j_ms": 60, "st_min_duration_s": 60,
            "tachycardia_bpm": 120, "bradycardia_bpm": 50,
            "min_tachy_brady_duration_s": 180, "vt_bpm": 100,
            "vt_min_beats": 3, "svt_bpm": 100, "svt_min_beats": 3,
            "af_auto_detect": True, "hr_scope": "all_beats",
            "exclude_pause_from_hr": False, "pacemaker_analysis": False,
            "hrv_scope": "normal", "arrhythmia_start_s": 0,
            "pacemaker_rate_bpm": 60, "pause_scope": "all_beats",
            "qtc_formula": "bazett",
        }


def _load_criteria() -> dict:
    out = _criteria_defaults()
    try:
        out.update(json.loads(_criteria_path().read_text(encoding="utf-8")))
    except Exception:
        pass
    return out


def _visible_events(events: list[dict]) -> list[dict]:
    """Descarta eventos de funciones no visibles (misma regla que el informe)."""
    try:
        gate = report_mod.FeatureGate(report_mod.load_features())
    except Exception:
        log.exception("sin registro de funciones; se devuelven todos los eventos")
        return events
    out = []
    for ev in events:
        fid = report_mod.EVENT_FEATURES.get(
            str(ev.get("kind", "")), report_mod.MANUAL_EVENT_FEATURE)
        if gate.on(fid):
            out.append(ev)
    return out


# ---------------------------------------------------------------------------
# Validación contra HScribe: matriz latido a latido (bxb)
# ---------------------------------------------------------------------------
def _normalize_stem(text: str) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def find_reference(source: str | Path | None) -> Path | None:
    """Busca el `Beats.dat` de HScribe (`<examen>.beats.dat`) del registro.

    Orden: junto al registro, en `HOLTERIA_SIDECARS` y en `data/sidecars` o
    `data/interim`. La comparación de nombres ignora mayúsculas y separadores.
    """
    if not source:
        return None
    src = Path(source)
    roots: list[Path] = []
    env = (os.environ.get("HOLTERIA_SIDECARS") or "").strip()
    if env:
        roots.append(Path(env))
    roots += [config.DATA_DIR / "sidecars", config.DATA_DIR / "interim"]
    name = f"{src.stem}{SIDECAR_SUFFIX}"
    for cand in [src.with_suffix(SIDECAR_SUFFIX), src.with_name(name),
                 *(root / name for root in roots)]:
        if cand.is_file():
            return cand
    want = _normalize_stem(src.stem)
    seen: set[str] = set()
    for folder in [src.parent, *roots]:
        try:
            key = str(folder.resolve()).lower()
            if key in seen or not folder.is_dir():
                continue
            seen.add(key)
            for p in sorted(folder.iterdir())[:5000]:
                if not (p.is_file() and p.name.lower().endswith(SIDECAR_SUFFIX)):
                    continue
                stem = p.name[: -len(SIDECAR_SUFFIX)]
                if _normalize_stem(stem) == want:
                    return p
        except OSError:
            continue
    return None


def match_times(reference, detected, tol_s: float = DEFAULT_TOL_S) -> tuple[np.ndarray, np.ndarray]:
    """Empareja dos series ordenadas por vecino más cercano dentro de `tol_s`.

    Cada latido se usa una vez. Devuelve los índices (en `reference` y en
    `detected`) de los pares encontrados.
    """
    ref = np.asarray(reference, dtype=float)
    det = np.asarray(detected, dtype=float)
    if ref.size == 0 or det.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    used = np.zeros(det.size, dtype=bool)
    pairs_i: list[int] = []
    pairs_j: list[int] = []
    for i, t in enumerate(ref):
        k = int(np.searchsorted(det, t - tol_s, side="left"))
        best = -1
        while k < det.size and det[k] <= t + tol_s:
            if not used[k] and (best < 0 or abs(det[k] - t) < abs(det[best] - t)):
                best = k
            k += 1
        if best >= 0:
            used[best] = True
            pairs_i.append(i)
            pairs_j.append(int(best))
    return np.asarray(pairs_i, dtype=int), np.asarray(pairs_j, dtype=int)


def _median_rr(times: np.ndarray) -> float:
    """RR mediano de la referencia, ignorando huecos y latidos absurdos."""
    if times.size < 3:
        return 0.8
    d = np.diff(np.sort(times))
    d = d[(d >= 0.25) & (d <= 3.0)]
    if d.size == 0:
        return 0.8
    return float(np.median(d))


def _quality_frac(ok_bins, bin_s: float, t0: float, t1: float) -> float | None:
    if not len(ok_bins):
        return None
    b0 = max(0, int(t0 // bin_s))
    b1 = min(len(ok_bins), max(b0 + 1, int(math.ceil(t1 / bin_s))))
    vals = ok_bins[b0:b1]
    if not vals:
        return None
    return float(sum(1 for v in vals if v)) / float(len(vals))


def build_validation(
    *,
    duration_s: float,
    ref_times,
    ref_labels,
    our_times,
    our_classes,
    our_idxs=None,
    our_confs=None,
    window_s: float = 300.0,
    quality: str = "all",
    kind: str = "all",
    cls: str = "",
    label: str = "",
    limit: int = 500,
    offset_s: float = 0.0,
    tol_s: float = DEFAULT_TOL_S,
    ok_bins=(),
    bin_s: float = 60.0,
) -> dict:
    """Matriz bxb, ventanas válidas y discrepancias entre HScribe y el motor.

    Solo cuentan las ventanas con referencia válida (HScribe con al menos el
    70 % de los latidos esperados según su propio RR). El filtro por calidad se
    aplica encima. No interpreta clínicamente: solo compara tiempos y etiquetas.
    """
    window_s = max(10.0, min(3600.0, float(window_s)))
    tol_s = max(0.02, min(1.0, float(tol_s)))
    limit = max(1, min(5000, int(limit)))
    quality = (quality or "all").lower()
    kind = (kind or "all").lower()
    cls = (cls or "").upper()
    label = str(label or "")

    rt = np.asarray(ref_times, dtype=float) - float(offset_s)
    rl = np.asarray(ref_labels, dtype=int)
    ot = np.asarray(our_times, dtype=float)
    oc = np.asarray([str(c) for c in our_classes], dtype=object)
    oi = np.asarray(our_idxs if our_idxs is not None else np.arange(ot.size), dtype=int)
    of = np.asarray(our_confs if our_confs is not None else np.zeros(ot.size), dtype=float)
    if rt.size != rl.size:
        raise ValueError("referencia y etiquetas con tamaños distintos")
    if not (ot.size == oc.size == oi.size == of.size):
        raise ValueError("latidos del motor con campos inconsistentes")
    duration_s = max(0.0, float(duration_s))
    if duration_s <= 0.0:
        duration_s = max(float(rt.max()) if rt.size else 0.0,
                         float(ot.max()) if ot.size else 0.0)
    # fuera del registro no se compara nada (latidos con marca de tiempo absurda)
    keep_r = np.isfinite(rt) & (rt >= -0.5) & (rt <= duration_s + 0.2)
    keep_o = np.isfinite(ot) & (ot >= -0.5) & (ot <= duration_s + 0.2)
    ignored = {"ref": int((~keep_r).sum()), "ours": int((~keep_o).sum())}
    rt, rl = rt[keep_r], rl[keep_r]
    ot, oc, oi, of = ot[keep_o], oc[keep_o], oi[keep_o], of[keep_o]
    order = np.argsort(rt, kind="stable")
    rt, rl = rt[order], rl[order]
    order = np.argsort(ot, kind="stable")
    ot, oc, oi, of = ot[order], oc[order], oi[order], of[order]

    # alineación: residuo de los pares (la referencia se toma tal cual, sin
    # desplazamientos grandes que solo calzan por azar)
    gi, gj = match_times(rt, ot, tol_s)
    diffs = (ot[gj] - rt[gi]) if gi.size else np.zeros(0)
    alignment = {
        "offset_s": round(float(offset_s), 3),
        "median_dt_ms": round(float(np.median(diffs)) * 1000.0, 1) if diffs.size else None,
        "pairs": int(gi.size),
    }

    n_win = max(1, int(math.ceil(duration_s / window_s)))
    ref_rr = _median_rr(rt)
    all_labels = sorted({int(x) for x in rl})
    rows = [str(x) for x in all_labels] + [OURS_EXTRA]
    extra_cls = sorted({str(x) for x in oc if str(x) not in AAMI_CLASSES})
    cols = list(AAMI_CLASSES) + extra_cls + [REF_MISSED]
    counts = {r: {c: 0 for c in cols} for r in rows}
    windows: list[dict] = []
    disc: list[dict] = []
    totals = {"tp": 0, "fn": 0, "fp": 0, "n_ref": 0, "n_ours": 0,
              "used": 0, "valid": 0, "used_s": 0.0}
    used_labels: set[str] = set()
    max_rows = 5000

    for w in range(n_win):
        t0 = w * window_s
        t1 = min(duration_s, t0 + window_s)
        if t1 <= t0:
            continue
        a = int(np.searchsorted(rt, t0, side="left"))
        b = int(np.searchsorted(rt, t1, side="left"))
        c = int(np.searchsorted(ot, t0, side="left"))
        d = int(np.searchsorted(ot, t1, side="left"))
        n_ref, n_ours = b - a, d - c
        pi, pj = match_times(rt[a:b], ot[c:d], tol_s)
        tp = int(pi.size)
        fn, fp = n_ref - tp, n_ours - tp
        expected = (t1 - t0) / ref_rr if ref_rr > 0 else 0.0
        density = (n_ref / expected) if expected > 0 else 0.0
        valid = n_ref >= 2 and density >= REF_DENSITY_MIN
        qfrac = _quality_frac(ok_bins, bin_s, t0, t1)
        qlabel = "sin datos" if qfrac is None else ("buena" if qfrac >= QUALITY_OK_MIN else "mala")
        qok = quality == "all" or (quality == "good" and qfrac is not None
                                   and qfrac >= QUALITY_OK_MIN) \
            or (quality == "bad" and (qfrac is None or qfrac < QUALITY_OK_MIN))
        used = bool(valid and qok)
        if len(windows) < max_rows:
            windows.append({
                "index": w, "t0": round(t0, 2), "t1": round(t1, 2),
                "n_ref": n_ref, "n_ours": n_ours, "tp": tp, "fn": fn, "fp": fp,
                "se": round(tp / n_ref, 4) if n_ref else None,
                "ppv": round(tp / n_ours, 4) if n_ours else None,
                "ref_expected": round(expected, 1), "ref_density": round(density, 3),
                "valid": bool(valid), "quality": qlabel,
                "quality_ok_frac": None if qfrac is None else round(qfrac, 3),
                "used": used,
            })
        totals["valid"] += int(valid)
        if not used:
            continue
        totals["used"] += 1
        totals["used_s"] += t1 - t0
        totals["tp"] += tp
        totals["fn"] += fn
        totals["fp"] += fp
        totals["n_ref"] += n_ref
        totals["n_ours"] += n_ours
        pair_ref = {int(x): int(y) for x, y in zip(pi, pj)}
        pair_our = set(pair_ref.values())
        for k in range(n_ref):
            row = str(int(rl[a + k]))
            used_labels.add(row)
            j = pair_ref.get(k)
            if j is not None:
                counts[row][str(oc[c + j])] += 1
            else:
                counts[row][REF_MISSED] += 1
                disc.append({
                    "kind": "fn", "t": round(float(rt[a + k]), 3),
                    "cls": "", "label": row, "conf": None, "idx": None,
                    "window": w, "quality": qlabel,
                })
        for k in range(n_ours):
            if k in pair_our:
                continue
            counts[OURS_EXTRA][str(oc[c + k])] += 1
            disc.append({
                "kind": "fp", "t": round(float(ot[c + k]), 3),
                "cls": str(oc[c + k]), "label": None,
                "conf": round(float(of[c + k]), 3), "idx": int(oi[c + k]),
                "window": w, "quality": qlabel,
            })

    kept_rows = [r for r in rows if r == OURS_EXTRA or r in used_labels]
    matrix = {
        "rows": kept_rows,
        "cols": cols,
        "counts": [[counts[r][c] for c in cols] for r in kept_rows],
        "row_totals": [sum(counts[r][c] for c in cols) for r in kept_rows],
        "col_totals": [sum(counts[r][c] for r in kept_rows) for c in cols],
    }

    def _keep(item: dict) -> bool:
        if kind in ("fp", "fn") and item["kind"] != kind:
            return False
        if cls and item["kind"] == "fp" and item["cls"] != cls:
            return False
        if label and item["kind"] == "fn" and item["label"] != label:
            return False
        return True

    filtered = [x for x in disc if _keep(x)]
    filtered.sort(key=lambda x: x["t"])
    n_fp = sum(1 for x in filtered if x["kind"] == "fp")
    n_fn = len(filtered) - n_fp
    used_s = totals["used_s"]
    summary = {
        "n_ref": totals["n_ref"], "n_ours": totals["n_ours"],
        "tp": totals["tp"], "fn": totals["fn"], "fp": totals["fp"],
        "se": round(totals["tp"] / totals["n_ref"], 4) if totals["n_ref"] else None,
        "ppv": round(totals["tp"] / totals["n_ours"], 4) if totals["n_ours"] else None,
        "coverage": round(totals["tp"] / max(1, min(totals["n_ref"], totals["n_ours"])), 4),
        "fp_per_min": round(totals["fp"] / (used_s / 60.0), 3) if used_s > 0 else None,
        "windows_total": n_win, "windows_valid": totals["valid"],
        "windows_used": totals["used"], "used_s": round(used_s, 1),
        "windows_listed": len(windows), "windows_truncated": n_win > len(windows),
        "ref_n_total": int(rt.size), "ours_n_total": int(ot.size),
        "ignored_out_of_range": ignored,
        "ref_rr_ms": round(ref_rr * 1000.0, 1),
    }
    return {
        "ok": True,
        "available": True,
        "summary": summary,
        "alignment": alignment,
        "matrix": matrix,
        "windows": windows,
        "discrepancies": filtered[:limit],
        "discrepancies_total": len(filtered),
        "discrepancies_fp": n_fp,
        "discrepancies_fn": n_fn,
        "filters": {
            "window_s": window_s, "quality": quality, "kind": kind, "cls": cls,
            "label": label, "limit": limit, "tol_ms": round(tol_s * 1000.0, 1),
            "offset_s": round(float(offset_s), 3),
        },
    }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "HolterIA/1.0"
    session: Session = None  # type: ignore
    static_dir: Path = config.STATIC_DIR

    # -- utilidades ----------------------------------------------------
    def _send_json(self, obj, code: int = 200):
        body = json.dumps(obj, ensure_ascii=False, default=_json_default).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_file(self, path: Path):
        if not path.exists() or not path.is_file():
            self._send_json({"error": "no encontrado"}, 404)
            return
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_bytes(self, data: bytes, ctype: str = "image/png"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def log_message(self, fmt, *args):  # silencio en consola
        log.debug(fmt, *args)

    # -- rutas ---------------------------------------------------------
    def do_GET(self):
        try:
            u = urlparse(self.path)
            q = parse_qs(u.query)
            p = u.path
            if p in ("/", "/index.html"):
                self._send_file(self.static_dir / "index.html")
            elif p.startswith("/static/"):
                rel = p[len("/static/"):]
                target = (self.static_dir / rel).resolve()
                if self.static_dir.resolve() not in target.parents and target != self.static_dir.resolve():
                    self._send_json({"error": "ruta inválida"}, 403)
                    return
                self._send_file(target)
            elif p == "/sw.js":
                self._send_file(self.static_dir / "sw.js")
            elif p.startswith("/reports/"):
                name = Path(p).name
                self._send_file(config.REPORTS_DIR / name)
            elif p == "/api/status":
                self._send_json(self.session.status())
            elif p == "/api/version":
                self._send_json({"name": config.APP_NAME, "version": config.VERSION,
                                 "disclaimer": config.DISCLAIMER})
            elif p == "/api/features":
                # Registro de funciones: estados, evidencia y plan de interfaz.
                try:
                    feats = report_mod.load_features()
                    plan = report_mod.ui_plan(feats)
                    self._send_json({
                        "ok": True,
                        "statuses": list(report_mod.FEATURE_STATUSES),
                        "features": [f.to_dict() for f in feats.values()],
                        "ui": plan,
                        "policy": "El informe solo informa funciones en estado visible.",
                    })
                except Exception as exc:
                    log.exception("registro de funciones")
                    self._send_json({"ok": False, "error": str(exc)}, 500)
            elif p == "/api/export/beats.csv":
                import csv
                import io
                with self.session.lock:
                    analysis = self.session.analysis
                if analysis is None:
                    self._send_json({"error": "no hay análisis"}, 400)
                    return
                buf = io.StringIO()
                w = csv.writer(buf)
                w.writerow(["idx", "t_s", "clase", "conf", "rr_prev_s",
                            "width_ms", "st_uv", "qt_ms", "artifact"])
                for b in analysis.beats:
                    f = b.features or {}
                    w.writerow([b.idx, round(b.t, 3), b.cls, round(b.conf, 3),
                                round(b.rr_prev, 3), f.get("width_ms"),
                                f.get("st_uv"), f.get("qt_ms"), int(bool(f.get("artifact")))])
                self._send_bytes(buf.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8")
            elif p == "/api/result":
                inc = q.get("beats", ["1"])[0] not in ("0", "false")
                res = self.session.result(include_beats=inc)
                res["events"] = _visible_events(res.get("events") or [])
                self._send_json(res)
            elif p == "/api/arrhythmogram":
                try:
                    payload = self.session.arrhythmogram()
                except RuntimeError as exc:
                    self._send_json({"ok": False, "error": str(exc)}, 400)
                else:
                    self._send_json({"ok": True, **payload})
            elif p == "/api/beats":
                start = int(q.get("start", ["0"])[0])
                count = min(5000, int(q.get("count", ["500"])[0]))
                self._send_json(self.session.beats_page(start, count))
            elif p == "/api/waveform":
                t0 = float(q.get("t0", ["0"])[0])
                t1 = float(q.get("t1", ["10"])[0])
                mp = int(q.get("points", ["2000"])[0])
                # filtros de pantalla: red 0/50/60 Hz y paso bajo 0/25/30/35 Hz
                notch = q.get("notch", ["0"])[0]
                lowpass = q.get("lp", ["0"])[0]
                self._send_json(self.session.waveform(t0, t1, mp, notch_hz=notch,
                                                      lowpass_hz=lowpass))
            elif p == "/api/overview":
                self._send_json(self.session.overview or {})
            elif p == "/api/settings":
                self._send_json(config.load_settings())
            elif p == "/api/criteria":
                self._send_json({"ok": True, "criteria": _load_criteria()})
            elif p == "/api/st":
                block = self.session.st_block() if self.session.analysis is not None else {}
                self._send_json({"ok": True, "st": block,
                                 "criteria": _load_criteria()})
            elif p == "/api/audit":
                exam = (q.get("exam", [""])[0] or "").strip()
                self._send_json(
                    {"ok": True, "audit": store.list_audit(exam or None)}
                )
            elif p == "/api/prior":
                exam = (q.get("exam", [""])[0] or "").strip()
                exams = store.list_exams()
                cur = next((e for e in exams if e.get("exam_id") == exam), None)
                if cur is None or cur.get("patient_id") is None:
                    self._send_json({"ok": True, "prior": []})
                else:
                    prior = [
                        e for e in exams
                        if e.get("patient_id") == cur.get("patient_id")
                        and e.get("exam_id") != exam
                    ]
                    self._send_json({"ok": True, "prior": prior})
            elif p == "/api/models":
                from holterai.model_registry import list_models

                self._send_json({
                    "ok": True,
                    "models": [
                        {"name": m.name, "installed": m.installed}
                        for m in list_models()
                    ],
                })
            elif p == "/api/patient":
                rec = self.session.recording
                self._send_json({"ok": True, "patient": rec.patient.to_dict() if rec else {}})
            elif p == "/api/report_config":
                self._send_json({"ok": True, "config": load_report_config().to_dict()})
            elif p == "/api/accel":
                from holterai.legacy.app.ml import providers
                pref = str(config.load_settings().get("accel", "auto"))
                self._send_json({"ok": True, **providers.describe(pref)})
            elif p == "/api/beatpng":
                try:
                    idx = int(q.get("idx", ["0"])[0])
                    win = float(q.get("w", ["2.4"])[0])
                    with self.session.lock:
                        rec = self.session.recording
                        an = self.session.analysis
                    if rec is None:
                        raise RuntimeError("sin registro")
                    fs = rec.fs
                    t0 = max(0.0, idx / fs - win / 2.0)
                    _, seg = rec.segment(t0, t0 + win)
                    lead = int(an.quality.get("lead", 0)) if an else 0
                    sub = [b for b in (an.beats if an else []) if t0 <= b.t <= t0 + win]
                    from holterai.legacy.app.report import plotting as P
                    col = seg[:, [lead]] if seg.ndim > 1 else seg
                    img = P.ecg_strip(col, fs, t0, width=360, height=96,
                                      leads=[rec.leads[lead]],
                                      beats=[b.to_dict() for b in sub], paper=True)
                    from io import BytesIO
                    buf = BytesIO()
                    img.save(buf, format="PNG", optimize=True)
                    self._send_bytes(buf.getvalue())
                except Exception as exc:
                    self._send_json({"error": str(exc)}, 400)
            elif p == "/api/templatepng":
                try:
                    idxs = [int(x) for x in q.get("idx", [""])[0].split(",") if x.strip()][:200]
                    from io import BytesIO
                    import matplotlib
                    matplotlib.use("Agg")
                    from matplotlib.figure import Figure
                    from matplotlib.backends.backend_agg import FigureCanvasAgg
                    import numpy as np
                    from holterai.legacy.app.signal import features as FT
                    with self.session.lock:
                        rec = self.session.recording
                        if rec is None:
                            raise RuntimeError("sin registro")
                        xf, _ = self.session._ensure_xf()
                        fs = rec.fs
                    wins = FT.extract_windows_batch(xf, fs, np.array(idxs, dtype=np.int64))
                    fig = Figure(figsize=(6.4, 3.2), dpi=110)
                    ax = fig.add_subplot(111)
                    for w in wins:
                        ax.plot(np.linspace(-0.25, 0.45, w.size), w, color="#7f8c9a", alpha=0.12, linewidth=0.7)
                    if len(wins):
                        ax.plot(np.linspace(-0.25, 0.45, wins.shape[1]), wins.mean(axis=0),
                                color="#e74c3c", linewidth=1.8, label="promedio")
                    ax.set_xlabel("s respecto al R"); ax.set_ylabel("mV normalizado")
                    ax.grid(alpha=0.2); ax.legend(loc="upper right", fontsize=8)
                    fig.tight_layout()
                    buf = BytesIO(); FigureCanvasAgg(fig).print_png(buf)
                    self._send_bytes(buf.getvalue())
                except Exception as exc:
                    self._send_json({"error": str(exc)}, 400)
            elif p == "/api/templates":
                cls = q.get("cls", ["V"])[0]
                self._send_json({"ok": True, "clusters": self.session.templates(cls)})
            elif p == "/api/quality":
                self._send_json({"ok": True, **self.session.quality_map(int(q.get("bin", ["60"])[0]))})
            elif p == "/api/validation":
                with self.session.lock:
                    if self.session.analysis is None:
                        self._send_json({"ok": False, "available": False,
                                         "error": "no hay análisis"}, 400)
                        return
                try:
                    params = {
                        "window_s": float(q.get("window", ["300"])[0]),
                        "quality": (q.get("quality", ["all"])[0] or "all"),
                        "kind": (q.get("kind", ["all"])[0] or "all"),
                        "cls": (q.get("cls", [""])[0] or ""),
                        "label": (q.get("label", [""])[0] or ""),
                        "limit": int(q.get("limit", ["500"])[0]),
                        "offset_s": float(q.get("offset", ["0"])[0]),
                        "tol_ms": float(q.get("tol", ["150"])[0]),
                    }
                except (TypeError, ValueError) as exc:
                    self._send_json({"ok": False, "error": f"parámetro inválido: {exc}"}, 400)
                    return
                self._send_json(self.session.validation(**params))
            elif p == "/api/recent":
                self._send_json({"ok": True, "items": _recent_list()})
            elif p == "/api/worklist":
                self._send_json({"ok": True, "items": _worklist()})
            else:
                self._send_json({"error": "ruta no encontrada"}, 404)
        except Exception as exc:
            log.exception("GET %s", self.path)
            self._send_json({"error": str(exc)}, 500)

    def do_POST(self):
        try:
            u = urlparse(self.path)
            p = u.path
            body = self._body()
            if p == "/api/import":
                path = body.get("path", "")
                fs = body.get("fs")
                if str(path).strip().lower() in ("demo", "ejemplo", "example"):
                    from holterai.legacy.app.tools.demo import build as build_demo
                    path = str(build_demo())
                meta = self.session.load(path, fs=float(fs) if fs else None)
                _recent_add(str(path), meta)
                self._send_json({"ok": True, "meta": meta,
                                 "overview": self.session.overview})
            elif p == "/api/analyze":
                self.session.start_analysis(body.get("options", {}))
                self._send_json({"ok": True})
            elif p == "/api/notes":
                text = str(body.get("text", ""))
                with self.session.lock:
                    if self.session.analysis is not None:
                        self.session.analysis.notes = text
                        self.session._save_edits("notes")
                self._send_json({"ok": True, "notes": text})
            elif p == "/api/diary":
                text = str(body.get("text", ""))
                with self.session.lock:
                    if self.session.analysis is not None:
                        self.session.analysis.diary = text
                        self.session._save_edits("diary")
                self._send_json({"ok": True, "diary": text})
            elif p == "/api/beat":
                out = self.session.edit_beat(int(body["idx"]), str(body["cls"]))
                self._send_json({"ok": True, "beat": out})
            elif p == "/api/beat/delete":
                out = self.session.delete_beat(int(body["idx"]))
                self._send_json({"ok": True, **out})
            elif p == "/api/beat/insert":
                out = self.session.insert_beat(float(body["t"]), str(body["cls"]))
                self._send_json({"ok": True, **out})
            elif p == "/api/events":
                out = self.session.save_events(list(body.get("events", [])))
                self._send_json({"ok": True, **out})
            elif p == "/api/probe":
                info = ingest.probe(body.get("path", ""))
                self._send_json({"ok": True, "probe": info})
            elif p == "/api/undo":
                self._send_json({"ok": True, **self.session.undo()})
            elif p == "/api/invert_lead":
                out = self.session.invert_lead(int(body["lead"]),
                                               bool(body.get("undo", False)))
                self._send_json(out)
            elif p == "/api/templates/apply":
                out = self.session.apply_template(body.get("idx", []), str(body.get("cls", "N")))
                self._send_json({"ok": True, **out})
            elif p == "/api/templates/split":
                self._send_json({"ok": True, "clusters": self.session.split_template(
                    body.get("idx", []), float(body.get("thr", 0.94))
                )})
            elif p == "/api/export/annotations":
                with self.session.lock:
                    analysis = self.session.analysis
                if analysis is None:
                    self._send_json({"error": "no hay análisis"}, 400)
                    return
                from holterai.legacy.app.ingest import wfdb as wfdb_io
                name = Path(analysis.recording.source).stem or "estudio"
                out = config.DATA_DIR / "exports" / f"{name}_holteria.atr"
                out.parent.mkdir(parents=True, exist_ok=True)
                wfdb_io.write_annotations(
                    out, [b.idx for b in analysis.beats],
                    [b.cls for b in analysis.beats])
                self._send_json({"ok": True, "path": str(out),
                                 "n": len(analysis.beats)})
            elif p == "/api/export/beats.csv":
                import csv
                import io
                with self.session.lock:
                    analysis = self.session.analysis
                if analysis is None:
                    self._send_json({"error": "no hay análisis"}, 400)
                    return
                buf = io.StringIO()
                w = csv.writer(buf)
                w.writerow(["idx", "t_s", "clase", "conf", "rr_prev_s",
                            "width_ms", "st_uv", "qt_ms", "artifact"])
                for b in analysis.beats:
                    f = b.features or {}
                    w.writerow([b.idx, round(b.t, 3), b.cls, round(b.conf, 3),
                                round(b.rr_prev, 3), f.get("width_ms"),
                                f.get("st_uv"), f.get("qt_ms"), int(bool(f.get("artifact")))])
                self._send_bytes(buf.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8")
            elif p == "/api/report":
                with self.session.lock:
                    analysis = self.session.analysis
                if analysis is None:
                    self._send_json({"error": "no hay análisis"}, 400)
                    return
                paths = report_mod.generate(analysis, formats=body.get("formats"),
                                            sections=body.get("sections"))
                self._send_json({"ok": True, **paths})
            elif p == "/api/settings":
                cur = config.load_settings()
                cur.update(body or {})
                config.save_settings(cur)
                self._send_json({"ok": True, "settings": config.load_settings()})
            elif p == "/api/criteria":
                cur = _load_criteria()
                cur.update(body or {})
                _criteria_path().write_text(
                    json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                self._send_json({"ok": True, "criteria": cur})
            elif p == "/api/st/recompute":
                out = self.session.recompute_st(body or {})
                self._send_json({"ok": True, **out})
            elif p == "/api/patient":
                rec = self.session.recording
                if rec is None:
                    self._send_json({"ok": False, "error": "sin estudio"}, 400)
                    return
                code = str(body.get("patient_id") or rec.patient.patient_id or
                           Path(rec.source).stem)
                rec.patient.patient_id = code
                rec.patient.name = str(body.get("name", rec.patient.name) or "")
                rec.patient.sex = str(body.get("sex", rec.patient.sex) or "")
                try:
                    rec.patient.age = int(body["age"]) if body.get("age") not in (None, "") else None
                except (TypeError, ValueError):
                    rec.patient.age = None
                rec.patient.notes = str(body.get("notes", rec.patient.notes) or "")
                rec.patient.dob = str(body.get("dob", rec.patient.dob) or "")
                pid = store.upsert_patient(
                    patient_code=code, name=rec.patient.name, sex=rec.patient.sex,
                    age=rec.patient.age, dob=str(body.get("dob", "") or ""),
                    notes=rec.patient.notes,
                )
                store.upsert_exam(
                    exam_id=Path(rec.source).stem, source_path=rec.source,
                    patient_id=pid, duration_s=rec.duration_s,
                    n_beats=len(self.session.analysis.beats) if self.session.analysis else 0,
                )
                self._send_json({"ok": True, "patient": rec.patient.to_dict()})
            elif p == "/api/report_config":
                cfg = ReportConfig(
                    clinic=str(body.get("clinic", "")),
                    doctor=str(body.get("doctor", "")),
                    signature=str(body.get("signature", "")),
                    footer=str(body.get("footer", "")),
                    logo=str(body.get("logo", "")),
                )
                save_report_config(cfg)
                self._send_json({"ok": True, "config": cfg.to_dict()})
            elif p == "/api/quit":
                self._send_json({"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self._send_json({"error": "ruta no encontrada"}, 404)
        except Exception as exc:
            log.exception("POST %s", self.path)
            self._send_json({"error": str(exc)}, 400)


def _recent_list() -> list[dict]:
    try:
        return json.loads((config.DATA_DIR / "recent.json").read_text(encoding="utf-8"))
    except Exception:
        return []


def _worklist() -> list[dict]:
    """Reportes generados + exámenes registrados en SQLite."""
    registered = {e["exam_id"]: e for e in store.list_exams()}
    out: list[dict] = []
    root = config.REPORTS_DIR
    for report in sorted(root.glob("*/report.json")):
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = payload.get("meta", {}) or {}
        source = str(meta.get("source") or "")
        src = Path(source)
        if source and not src.is_absolute():
            src = config.APP_ROOT / src
        if not source or not src.is_file():
            continue
        summary = payload.get("summary", {}) or {}
        exam_id = payload.get("exam_id", report.parent.name)
        db = registered.pop(exam_id, None)
        out.append(
            {
                "exam_id": exam_id,
                "path": str(src),
                "report": str(report),
                "html": str(report.with_suffix(".html")),
                "n_beats": payload.get("n_beats", 0),
                "duration_h": summary.get("duration_h"),
                "generated_at": payload.get("generated_at"),
                "registered": db is not None,
                "patient": {
                    "code": db.get("patient_code") if db else "",
                    "name": db.get("patient_name") if db else "",
                    "sex": db.get("sex") if db else "",
                    "age": db.get("age") if db else None,
                },
            }
        )

    for exam_id, db in registered.items():
        src = Path(str(db.get("source_path") or ""))
        if not src.is_file():
            continue
        out.append(
            {
                "exam_id": exam_id,
                "path": str(src),
                "report": db.get("report_path") or "",
                "html": "",
                "n_beats": db.get("n_beats", 0),
                "duration_h": round(float(db.get("duration_s") or 0) / 3600.0, 2),
                "generated_at": db.get("updated_at"),
                "registered": True,
                "patient": {
                    "code": db.get("patient_code") or "",
                    "name": db.get("patient_name") or "",
                    "sex": db.get("sex") or "",
                    "age": db.get("age"),
                },
            }
        )
    return out


def _recent_add(path: str, meta: dict) -> None:
    try:
        items = [i for i in _recent_list() if i.get("path") != path]
        items.insert(0, {
            "path": path,
            "name": (meta.get("patient", {}) or {}).get("name") or Path(path).name,
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "duration_h": round(float(meta.get("duration_s", 0)) / 3600.0, 2),
        })
        (config.DATA_DIR / "recent.json").write_text(
            json.dumps(items[:25], ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def serve(host: str | None = None, port: int = 0,
          open_browser: bool | None = None) -> tuple[ThreadingHTTPServer, str]:
    host = host or config.SETTINGS.get("host", "127.0.0.1")
    session = Session()
    Handler.session = session
    httpd = ThreadingHTTPServer((host, port), Handler)
    actual_port = httpd.server_address[1]
    url = f"http://{host}:{actual_port}"
    log.info("HolterIA escuchando en %s", url)
    return httpd, url

