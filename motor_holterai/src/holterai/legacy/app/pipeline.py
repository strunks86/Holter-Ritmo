"""Pipeline de análisis Holter: señal -> latidos -> modelos -> eventos.

Ordena los pasos de forma que un registro de 24-48 h se procese por
bloques, sin cargar más de lo necesario en memoria.
"""
from __future__ import annotations

import functools
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from holterai.carga_ectopia import cargar_criterios, resumen_carga
from holterai.legacy.app.core.logging import get_logger

log = get_logger(__name__)
from holterai.legacy.app.core.models import Analysis, Beat, Event, Recording, RhythmSegment
from holterai.legacy.app.ml.calibration import Calibration
from holterai.legacy.app.ml.net import CLASSES, BeatNet
from holterai.legacy.app.ml.rhythm_model import RhythmModel
from holterai.legacy.app.signal import artifact as artifact_mod
from holterai.legacy.app.signal import af as fa_reglas
from holterai.legacy.app.signal import atrial
from holterai.legacy.app.signal import features as FT
from holterai.legacy.app.signal import pwave as pwave_mod
from holterai.legacy.app.signal import qrs_morph as qm
from holterai.legacy.app.signal import rhythm
from holterai.legacy.app.signal import st as ST
from holterai.legacy.app.signal.filters import preprocess
from holterai.legacy.app.signal.qrs import (detect_qrs, detect_qrs_multilead, lead_score,
                                           rank_leads)
from holterai.legacy.app.signal import hrv as hrv_mod
from holterai.legacy.app.signal import multilead
from holterai.legacy.app.signal import pacing
from holterai.legacy.app.signal import qt as qt_mod
from holterai.legacy.app.signal import svpb
from holterai.legacy.app.signal.rhythm import (AF_FEATURES, BASE_EVENT_KINDS,
                               BASE_RHYTHM_NAMES, af_probabilities, base_rate_hourly,
                               base_rhythm_analysis, baseline_fraction,
                               bigeminy_events, count_pvc_morphologies,
                               ectopic_runs, gap_events, hourly_rr_stats,
                               hourly_summary, lown_grade,
                               no_organized_rhythm_events, no_signal_events,
                               pause_events, qrs_polarity,
                               rate_events, ritmo_beats, ritmo_mask,
                               rolling_percentile, rr_features,
                               s_salva_layer, supraventricular_events,
                               ventricular_events)


# T173: parámetros de fusión con T158 (ajustados en registros de validación de
# AFDB/LTAFDB de T158; ver resultados/T173.md)
T158_V_VETO = 0.0
T158_S_VETO = 0.0
T158_AF_WEIGHT = 0.0


@dataclass
class Options:
    chunk_s: float = 600.0
    overlap_s: float = 5.0
    use_ecgfounder: bool = True
    ecgfounder_stride_s: float = 30.0
    # T173: modelo para FA/PVC/PAC. "actual" = reglas + modelos de latido y RR (sin
    # ECGFounder en las decisiones); "t158" = ECGFounder ajustado de T158 (3 canales)
    # que veta V/S en ventanas sin PVC/PAC y se promedia con la FA del RR.
    # Flutter sigue siempre con el detector actual.
    ecgf_model: str = field(
        default_factory=lambda: os.environ.get("HOLTERIA_ECGF_MODEL", "actual"))
    t158_stride_s: float = 5.0
    t158_v_veto: float = T158_V_VETO     # V -> N si p(PVC) de sus ventanas < umbral
    t158_s_veto: float = T158_S_VETO     # S -> N si p(PAC) de sus ventanas < umbral
    t158_af_weight: float = T158_AF_WEIGHT  # peso de p(FA) de T158 frente a la del RR
    pause_s: float = 2.0
    brady_bpm: float = 50.0
    tachy_bpm: float = 100.0
    af_threshold: float = 0.55
    af_min_s: float = 30.0
    detect_no_organized: bool = False  # FV experimental (ver MODEL_CARD)
    pacing_mode: str = "auto"         # auto | on | off
    crt_mode: str = "auto"            # auto | on | off
    force_lead: int = -1              # derivación de análisis (-1 = automática)
    max_beats: int = 400_000
    reject_artifacts: bool = True     # máscara multicanal de ventanas ilegibles
    qtc_formula: str = "bazett"       # bazett | fridericia | framingham | hodges
    # --- ST (exploratorio) ---------------------------------------------
    st_j_ms: int = 60                 # punto de medición: J+60 o J+80 ms
    st_depression_uv: float = 100.0   # umbral de depresión (criterios)
    st_elevation_uv: float = 200.0    # umbral de elevación (criterios)
    st_min_s: float = 60.0            # duración mínima de un episodio
    calibration: bool = True          # temperatura + prevalencia + abstención Q
    morphology: bool = True           # onda P, PR y morfología del QRS por minuto

    def st_criteria(self) -> ST.StCriteria:
        """Criterio ST a partir de las opciones del análisis."""
        return ST.StCriteria(j_ms=self.st_j_ms, depression_uv=self.st_depression_uv,
                             elevation_uv=self.st_elevation_uv, min_s=self.st_min_s)

    av_mode: str = "auto"             # auto | off: PR y bloqueos AV (T24)
    av_windows: int = 6               # tramos analizados en un Holter largo
    av_window_s: float = 600.0        # largo de cada tramo (s)


Progress = "callable | None"


def _p(progress, msg: str, frac: float):
    if progress:
        try:
            progress(msg, max(0.0, min(1.0, frac)))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 1. Detección de latidos por bloques
# ---------------------------------------------------------------------------
def choose_analysis_lead(rec: Recording, probe_s: float = 180.0,
                         max_beats: int = 250) -> int:
    """Elige la derivación donde el modelo se siente más seguro.

    Un Holter puede traer canales con morfología atípica; clasificar en la
    derivación donde el modelo tiene mayor confianza evita errores masivos.
    """
    if rec.n_leads <= 1:
        return 0
    try:
        net = BeatNet.load(_model_path("beat_cnn.npz"))
    except Exception as exc:
        log.warning("sin modelo de latidos para elegir derivación: %s", exc)
        return 0
    n = min(rec.n_samples, int(probe_s * rec.fs))
    best, best_key = 0, (-1.0, -1)
    for j in range(rec.n_leads):
        seg = np.asarray(rec.samples[:n, j], dtype=np.float64)
        try:
            peaks = detect_qrs(seg, rec.fs)
        except Exception:
            continue
        if peaks.size < 10:
            continue
        rr = np.diff(peaks) / rec.fs
        valid = float(np.mean((rr > 0.30) & (rr < 2.0))) if rr.size else 0.0
        rate = peaks.size / max(1.0, n / rec.fs) * 60.0
        # Se prefiere un canal con RR plausibles y suficiente señal; el modelo
        # de clasificación no debe decidir la derivación (puede confiar en ruido).
        try:
            quality = multilead.lead_quality(preprocess(seg, rec.fs, "display"), rec.fs)
        except Exception:
            quality = 0.0
        if 25.0 <= rate <= 240.0:
            key = (valid, quality, int(peaks.size))
        else:
            key = (valid * 0.5, quality, int(peaks.size))
        if key > best_key:
            best, best_key = j, key
    log.info("derivación elegida: %d (latidos %d, RR válidos %.2f)", best, best_key[0], best_key[1])
    return best


def _lead_loses_beats(rec: Recording, lead: int, probe_s: float = 300.0,
                      min_recall: float = 0.60, margin: float = 0.10,
                      min_valid: float = 0.80) -> int | None:
    """Devuelve otra derivación si la elegida pierde latidos, o None.

    Dos señales de alarma, medidas contra la fusión de los tres canales en la
    sonda inicial: (1) la derivación elegida recupera menos del 60 % de lo que
    ve el consenso; (2) otra derivación con RR igual de plausibles y parecido
    acuerdo con el consenso detecta al menos un 10 % más de latidos.
    """
    if rec.n_leads < 2:
        return None
    probe = np.asarray(rec.samples[: int(min(rec.n_samples, probe_s * rec.fs))],
                       dtype=np.float64)
    try:
        ranking = rank_leads(probe, rec.fs, probe_s=probe_s)
    except Exception as exc:
        log.warning("no se pudo comparar derivaciones: %s", exc)
        return None
    own = next((r for r in ranking if int(r["lead"]) == int(lead)), None)
    if own is None:
        return None
    alt = [r for r in ranking
           if int(r["lead"]) != int(lead) and r["rate_ok"]
           and float(r["valid"]) >= min_valid
           and float(r["precision"]) >= float(own["precision"]) - 0.05]
    if own["recall"] < min_recall:
        alt_pick = max(alt, key=lambda r: r["recall"], default=None)
        if alt_pick is not None and float(alt_pick["recall"]) > float(own["recall"]):
            log.info("derivación %d pierde latidos (recuerdo %.2f); se usa %d (%.2f)",
                     int(lead), float(own["recall"]), int(alt_pick["lead"]),
                     float(alt_pick["recall"]))
            return int(alt_pick["lead"])
        return None
    alt_pick = max(alt, key=lambda r: r["n"], default=None)
    if alt_pick is not None and float(alt_pick["n"]) >= (1.0 + margin) * float(own["n"]):
        log.info("derivación %d detecta %d latidos y %d detecta %d; se usa %d",
                 int(lead), int(own["n"]), int(alt_pick["lead"]), int(alt_pick["n"]),
                 int(alt_pick["lead"]))
        return int(alt_pick["lead"])
    return None


def detect_beats(rec: Recording, opts: Options, progress=None, lead: int | None = None,
                 xf: np.ndarray | None = None):
    """Devuelve (idx globales, windows float16, qrs_anchos, amplitudes).

    Los bloques se procesan en paralelo con hilos: el filtrado FFT y las
    operaciones numpy liberan el GIL, así que se aprovechan varios núcleos.
    """
    fs = rec.fs
    n = rec.n_samples
    chunk = int(opts.chunk_s * fs)
    ov = int(opts.overlap_s * fs)

    # derivación de análisis (detección + clasificación)
    if lead is None:
        probe = rec.samples[: min(n, int(60 * fs))]
        try:
            scores = [lead_score(probe[:, j], fs) for j in range(rec.n_leads)]
            lead = int(np.argmax(scores))
        except Exception:
            lead = 0

    want = _thread_count()
    n_chunks = max(1, int(np.ceil(n / chunk)))

    def work(k: int):
        a = k * chunk
        b = min(n, a + chunk)
        lo = max(0, a - ov)
        hi = min(n, b + ov)
        seg = np.asarray(rec.samples[lo:hi, lead], dtype=np.float64)
        try:
            peaks = detect_qrs(seg, fs)
        except Exception:
            peaks = np.array([], dtype=int)
        peaks = peaks + lo
        peaks = peaks[(peaks >= a) & (peaks < b)]
        if peaks.size == 0:
            return k, None
        xf_local = xf[lo:hi] if xf is not None else preprocess(seg, fs, "display")
        local = (peaks - lo).astype(np.int64)
        wins = FT.extract_windows_batch(xf_local, fs, local)
        widths = np.array([FT.qrs_width(xf_local, fs, int(p)) for p in local])
        amp = np.array([float(np.max(np.abs(xf_local[max(0, p - int(0.1 * fs)):p + int(0.1 * fs)])))
                        for p in local])
        noises = FT.noise_batch(xf_local, fs, local)
        return k, (peaks, wins.astype(np.float16), widths, amp, noises)

    results: dict[int, tuple] = {}
    done = 0
    if want > 1 and n_chunks > 2:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=want) as ex:
            for k, res in ex.map(work, range(n_chunks)):
                done += 1
                if res is not None:
                    results[k] = res
                _p(progress, "Detectando latidos", 0.05 + 0.45 * done / n_chunks)
    else:
        for k in range(n_chunks):
            _, res = work(k)
            if res is not None:
                results[k] = res
            _p(progress, "Detectando latidos", 0.05 + 0.45 * (k + 1) / n_chunks)

    idx_all, win_all, width_all, amp_all, noise_all = [], [], [], [], []
    for k in sorted(results):
        peaks, wins, widths, amps, noises = results[k]
        idx_all.append(peaks)
        win_all.append(wins)
        width_all.append(widths)
        amp_all.append(amps)
        noise_all.append(noises)
    if not idx_all:
        return (np.array([], dtype=int), np.zeros((0, FT.WIN_POINTS), np.float16),
                np.zeros(0), np.zeros(0), np.zeros(0))
    return (np.concatenate(idx_all), np.concatenate(win_all),
            np.concatenate(width_all), np.concatenate(amp_all),
            np.concatenate(noise_all))


def _thread_count() -> int:
    import os
    try:
        return max(1, min(8, (os.cpu_count() or 4)))
    except Exception:
        return 1


# ---------------------------------------------------------------------------
# 2. Clasificación de latidos
# ---------------------------------------------------------------------------
def _local_rr_median_vec(rr: np.ndarray, half: int = 5) -> np.ndarray:
    """Mediana local de RR, idéntica a `ml.dataset_mitdb._local_rr_median`.

    Ventana de ±`half` latidos; se descartan los RR no finitos y los que caen
    fuera de 0,15–3 s, y si la ventana se queda sin valores válidos devuelve 0
    (igual que el entrenamiento). Es la referencia con la que se entrenó
    `beat_cnn`; el percentil 75 solo se usa para el modelo PAC.
    """
    rr = np.asarray(rr, dtype=np.float64)
    n = rr.size
    if n == 0:
        return rr
    ok = np.isfinite(rr) & (rr > 0.15) & (rr < 3.0)
    # Los valores no válidos van a +inf para que queden al final al ordenar y
    # `cnt` reproduzca la ventana truncada de los extremos.
    vals = np.where(ok, rr, np.inf)
    pad = np.pad(vals, half, mode="constant", constant_values=np.inf)
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)
    srt = np.sort(win, axis=1)
    cnt = np.isfinite(win).sum(axis=1)
    lo = np.maximum(cnt - 1, 0) // 2
    hi = np.maximum(cnt, 0) // 2
    rows = np.arange(n)
    med = 0.5 * (srt[rows, lo] + srt[rows, hi])
    med[cnt == 0] = 0.0
    return med


def _feature_matrix(rr_prev, rr_next, rr_med, widths, amps) -> np.ndarray:
    f = np.zeros((len(rr_prev), 5), dtype=np.float32)
    f[:, 0] = np.clip(rr_prev, 0, 4)
    f[:, 1] = np.clip(rr_next, 0, 4)
    # Misma razón que `signal.features.beat_features`: sin mediana válida, 1.0.
    rp = np.asarray(rr_prev, dtype=np.float64)
    med = np.asarray(rr_med, dtype=np.float64)
    ratio = np.divide(rp, med, out=np.ones_like(med), where=med > 0)
    f[:, 2] = np.clip(ratio, 0.3, 3.0)
    f[:, 3] = np.clip(widths, 0.03, 0.25)
    f[:, 4] = np.clip(amps, 0, 5)
    return f


def fuse_multilead(rec: Recording, idx: np.ndarray, widths: np.ndarray,
                   amps: np.ndarray, noises: np.ndarray, primary: int,
                   rr_prev: np.ndarray, probs_primary: np.ndarray,
                   xf_primary: np.ndarray, progress=None):
    """Fusiona las probabilidades de varios canales y aplica el veto cruzado.

    Devuelve (clases, confianza, veto, probabilidades_fusionadas) o None si no
    hay canales adicionales utilizables.
    """
    n = len(idx)
    if n == 0 or rec.n_leads < 2:
        return None
    fs = rec.fs
    probe = min(rec.n_samples, int(60 * fs))
    cand: list[tuple[int, float]] = []
    for l in range(rec.n_leads):
        if l == primary:
            continue
        xp = np.asarray(rec.samples[:probe, l], dtype=np.float64)
        q = multilead.lead_quality(preprocess(xp, fs, "display"), fs)
        if q > 0.2:
            cand.append((l, q))
    cand.sort(key=lambda t: -t[1])
    cand = cand[:2]
    if not cand:
        log.info("multicanal: no hay otros canales utilizables")
        return None

    try:
        net = BeatNet.load(_model_path("beat_cnn.npz"))
    except Exception as exc:
        log.warning("multicanal: sin modelo de latidos: %s", exc)
        return None

    rr = np.diff(idx).astype(np.float64) / fs
    rr_next = np.concatenate([rr, [0.0]])
    rr_med = _local_rr_median_vec(rr_prev)   # igual que el entrenamiento de beat_cnn

    leads = [(primary, 1.0)] + cand
    psum = np.zeros((n, 5), dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)
    per_lead = []
    presences = []

    for l, q in leads:
        if l == primary:
            probs_l = probs_primary.astype(np.float64)
            pres = multilead.qrs_presence(xf_primary, fs, idx)
        else:
            probs_l = np.zeros((n, 5), dtype=np.float64)
            pres = np.zeros(n, dtype=bool)
            step = 20000
            for s in range(0, n, step):
                e = min(n, s + step)
                t0 = max(0.0, idx[s] / fs - 0.5)
                t1 = min(rec.duration_s, idx[e - 1] / fs + 0.5)
                a = int(t0 * fs)
                b = int(t1 * fs)
                seg = np.asarray(rec.samples[a:b, l], dtype=np.float64)
                if seg.size < 16:
                    continue
                xf_b = preprocess(seg, fs, "display")
                local = (idx[s:e] - a).astype(np.int64)
                np.clip(local, 0, xf_b.size - 1, out=local)
                wins_b = FT.extract_windows_batch(xf_b, fs, local)
                w_b = np.array([FT.qrs_width(xf_b, fs, int(p)) for p in local])
                amp_b = np.array([float(np.max(np.abs(
                    xf_b[max(0, p - int(0.1 * fs)):p + int(0.1 * fs)])))
                    for p in local])
                feats_b = _feature_matrix(rr_prev[s:e], rr_next[s:e],
                                          rr_med[s:e], w_b, amp_b)
                probs_l[s:e] = net.predict_proba(wins_b, feats_b, batch=512)
                pres[s:e] = multilead.qrs_presence(xf_b, fs, local)
            _p(progress, "Fusionando canales", 0.56)

        weight = np.full(n, float(q))
        psum += weight[:, None] * probs_l
        wsum += weight
        per_lead.append(probs_l.argmax(axis=1))
        presences.append(pres)

    probs = psum / np.maximum(wsum, 1e-9)[:, None]
    cls = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    veto = multilead.veto_mask(presences, rec.n_leads)
    agreement = np.ones(n, dtype=np.float32)
    if len(per_lead) > 1:
        stack = np.stack(per_lead)
        agreement = np.mean(stack == stack[0], axis=0).astype(np.float32)
    log.info("multicanal: %d canales, %d latidos vetados (%.1f %%)",
             len(leads), int(veto.sum()), 100.0 * veto.mean())
    return cls, conf, veto, probs, agreement


def classify_beats(rec: Recording, idx: np.ndarray, wins: np.ndarray,
                   widths: np.ndarray, amps: np.ndarray,
                   progress=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (clases int, confianza, rr_prev)."""
    fs = rec.fs
    if len(idx) == 0:
        return np.zeros(0, int), np.zeros(0), np.zeros(0)
    rr = np.diff(idx).astype(np.float64) / fs
    rr_prev = np.concatenate([[0.0], rr])
    rr_next = np.concatenate([rr, [0.0]])
    # mediana local de ±5 latidos, la misma que usa `ml.dataset_mitdb` para
    # entrenar beat_cnn (el entrenamiento y el uso deben coincidir)
    rr_med = _local_rr_median_vec(rr_prev)

    feats = _feature_matrix(rr_prev, rr_next, rr_med, widths, amps)

    probs = np.zeros((len(idx), 5), dtype=np.float32)
    try:
        net = BeatNet.load(_model_path("beat_cnn.npz"))
        probs = net.predict_proba(np.asarray(wins, dtype=np.float32), feats, batch=512)
        cls = probs.argmax(axis=1)
        conf = probs.max(axis=1)
    except Exception as exc:
        log.warning("clasificador de latidos no disponible: %s", exc)
        cls = np.zeros(len(idx), dtype=int)
        conf = np.zeros(len(idx))
    _p(progress, "Clasificando latidos", 0.55)
    return cls, conf, rr_prev, probs


# ---------------------------------------------------------------------------
# 3. Supraventriculares (S): puntaje clínico + modelo
# ---------------------------------------------------------------------------
def get_svpb_config() -> svpb.SvpbConfig:
    """Umbrales de la decisión S (`HOLTERIA_SVPB` puede apuntar a un JSON)."""
    global _SVPB_CONFIG
    if _SVPB_CONFIG is None:
        import os
        path = os.environ.get("HOLTERIA_SVPB") or None
        _SVPB_CONFIG = svpb.SvpbConfig.load(path)
    return _SVPB_CONFIG


def refine_supraventricular(rec: Recording, idx: np.ndarray, cls: np.ndarray,
                            conf: np.ndarray, widths: np.ndarray, noises: np.ndarray,
                            lead: int, progress=None, xf: np.ndarray | None = None,
                            probs: np.ndarray | None = None,
                            cfg: svpb.SvpbConfig | None = None):
    """Reajusta latidos S con la capa clínica de `signal.svpb` (T22).

    Combina prematuridad (RR previo contra la mediana local), forma del QRS
    (correlación con la plantilla sinusal y ancho no aumentado), evidencia de
    onda P (presente/ausente, morfología, PR y P-P) y pausa no compensadora, y
    lo mezcla con la probabilidad calibrada del clasificador (T06) y la del
    modelo específico de PAC. Los umbrales salen de
    :class:`signal.svpb.SvpbConfig` (elegidos en validación).

    Devuelve ``(clases, confianza, p_pac, corr_qrs, decision)``; ``decision``
    es el dict de :func:`signal.svpb.decide` o ``None`` si no se pudo evaluar.
    """
    n = len(idx)
    pac_prob = np.zeros(n, dtype=np.float32)
    corr = np.zeros(n, dtype=np.float32)
    if n < 20:
        return cls, conf, pac_prob, corr, None
    cfg = cfg or get_svpb_config()
    try:
        from holterai.legacy.app.ml.pac_model import PACModel
        from holterai.legacy.app.signal import pac as PAC
    except Exception as exc:                       # pragma: no cover - import roto
        log.warning("sin análisis de onda P: %s", exc)
        return cls, conf, pac_prob, corr, None

    fs = rec.fs
    if xf is None:
        x = np.asarray(rec.samples[:, lead], dtype=np.float64)
        xf = preprocess(x, fs, "display")
    _p(progress, "Analizando ondas P", 0.60)

    tpl = PAC.build_templates(xf, fs, idx)
    if not tpl.get("ok"):
        return cls, conf, pac_prob, corr, None

    rr = np.diff(idx).astype(np.float64) / fs
    rr_prev = np.concatenate([[0.0], rr])
    rr_next = np.concatenate([rr, [0.0]])
    rr_prev2 = np.concatenate([[0.0, 0.0], rr[:-1]]) if rr.size > 1 else np.zeros(n)
    # percentil 75 para las características del modelo PAC (así se entrenó,
    # ver ml/dataset_pac.py) y ciclo sinusal local para la prematuridad: la
    # mediana de los RR se contamina en bigeminia, el promedio de pares no
    rr_med = rolling_percentile(rr_prev, half=5, q=75.0)
    rr_local = _local_rr_median_vec(rr_prev)
    rr_sinus = svpb.sinus_rr_reference(rr_prev, rr_next, rr_local)

    try:                                          # el modelo de PAC es opcional
        model = PACModel.load(_model_path("pac_model.npz"))
    except Exception as exc:
        log.warning("modelo PAC no disponible: se decide con las reglas: %s", exc)
        model = None

    step = 20000
    for s in range(0, n, step):
        e = min(n, s + step)
        feats = PAC.beat_features_batch(
            xf, fs, idx[s:e], tpl, rr_prev[s:e], rr_next[s:e], rr_med[s:e],
            widths[s:e], noises[s:e], np.zeros(e - s), rr_prev2[s:e])
        if model is not None:
            pac_prob[s:e] = model.predict(feats).astype(np.float32)
        corr[s:e] = feats[:, PAC.IDX["qrs_corr"]]
        _p(progress, "Analizando ondas P", 0.60 + 0.05 * e / n)

    # probabilidad de S: la del clasificador calibrado (T06) es poblacional; la
    # del modelo de PAC está inflada hacia S por el balanceo del entrenamiento,
    # así que entra atenuada por su peso
    if probs is not None and np.asarray(probs).shape[0] == n:
        p_beat = np.asarray(probs, dtype=np.float64)[:, CLASSES.index("S")]
    else:
        p_beat = np.zeros(n)
    model_prob = 1.0 - (1.0 - np.clip(p_beat, 0.0, 1.0)) * \
        (1.0 - cfg.pac_model_weight * np.clip(pac_prob.astype(np.float64), 0.0, 1.0))

    decision = svpb.decide_from_pwave(
        xf, fs, idx, widths, rr_prev, rr_next, rr_sinus, corr, model_prob,
        noise=noises, cfg=cfg, base_cls=cls)

    # N -> S y V -> S (aberrante con QRS sinusal); F y Q no se tocan. La clase S
    # queda decidida por la capa: una S del clasificador que no cumple las
    # compuertas clínicas vuelve a N (antes sólo se podía añadir S).
    changed = svpb.apply(cls, decision["is_s"], corr, cfg,
                         support=decision["is_s"])
    # un V con el QRS idéntico al sinusal, sin evidencia de S y con el modelo en
    # contra es un latido normal mal clasificado
    look = (svpb.normal_lookalike_mask(cls, corr, cfg) & ~decision["is_s"]
            & (pac_prob < 0.5))
    if look.any():
        cls[look] = 0
        changed += int(look.sum())
    known = float(np.mean(decision["p_feat"]["p_state"] != svpb.P_UNKNOWN))
    # componentes de la probabilidad del modelo, para re-pesar sin recargar nada
    decision["p_beat"] = np.asarray(p_beat, dtype=np.float64)
    decision["pac_prob"] = np.asarray(pac_prob, dtype=np.float64)
    log.info("S: %d latidos reclasificados (umbral %.2f, P evaluable %.0f %%)",
             changed, cfg.threshold, 100.0 * known)
    return cls, conf, pac_prob, corr, decision


def refine_pac(rec: Recording, idx: np.ndarray, cls: np.ndarray, conf: np.ndarray,
               widths: np.ndarray, noises: np.ndarray, lead: int,
               progress=None, xf: np.ndarray | None = None):
    """Compatibilidad con la firma histórica: (clases, conf, p_pac, corr)."""
    cls, conf, pac_prob, corr, _ = refine_supraventricular(
        rec, idx, cls, conf, widths, noises, lead, progress=progress, xf=xf)
    return cls, conf, pac_prob, corr


_FOUNDER = None


def _get_founder():
    """Instancia única de ECGFounder (cargar 124 MB por análisis sería un desperdicio)."""
    global _FOUNDER
    if _FOUNDER is None:
        from holterai.legacy.app.ml.ecgfounder import ECGFounder
        m = ECGFounder(task="all71")
        m.load()
        _FOUNDER = m
    return _FOUNDER


def _v1_negative(rec: Recording, idx: np.ndarray, fs: float) -> bool:
    """¿El QRS de V1 es predominantemente negativo? (orienta biventricular)."""
    names = [str(n).upper().replace(" ", "") for n in rec.leads]
    for j, n in enumerate(names):
        if n.endswith("V1") or n == "V1":
            if j >= rec.n_leads or idx.size == 0:
                return False
            x = rec.samples[:, j]
            span = int(0.09 * fs)
            neg = pos = 0
            for p in idx[:2000]:
                a, b = max(0, int(p) - span), min(len(x), int(p) + span)
                if b > a:
                    seg = np.asarray(x[a:b], dtype=float)
                    if abs(seg.min()) > abs(seg.max()):
                        neg += 1
                    else:
                        pos += 1
            return neg > pos
    return False


def ecgfounder_layer(rec: Recording, opts: Options, progress=None) -> dict:
    """Segunda opinión con el modelo fundacional (71 diagnósticos PTB-XL)."""
    model = _get_founder()
    _p(progress, "Modelo fundacional", 0.72)
    res = model.predict_recording(rec.samples, rec.fs, rec.leads,
                                  stride_s=opts.ecgfounder_stride_s)
    probs = res["probs"]
    labels = res["labels"]
    idx = {name: i for i, name in enumerate(labels)}
    out: dict = {"provider": model.accel.get("label", "CPU"),
                 "n_windows": int(len(res["times"])), "labels": {}}
    for key in ("AFIB", "AFLT", "PVC", "PAC", "SR", "STACH", "SBRAD", "BIGU", "TRIGU"):
        if key in idx:
            v = probs[:, idx[key]]
            out["labels"][key] = {
                "mean": round(float(v.mean()), 3),
                "max": round(float(v.max()), 3),
                "frac_high": round(float((v > 0.5).mean()), 3),
            }
    return out


_T158 = None


def _get_t158():
    """Instancia única del ECGFounder ajustado de T158 (GPU con respaldo a CPU)."""
    global _T158
    if _T158 is None:
        from holterai.legacy.app.ml.ecgfounder_t158 import T158Model
        _T158 = T158Model().load()
    return _T158


def t158_layer(rec: Recording, idx: np.ndarray, cls: np.ndarray, opts: Options,
               progress=None) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """T173: FA/PVC/PAC con el ECGFounder ajustado de T158 (3 canales).

    Ventanas de 10 s cada ``t158_stride_s``. Cada latido recibe el máximo (PVC,
    PAC) y la media (FA) de las ventanas que lo contienen. Un V (o S) cuyas
    ventanas no muestran PVC (o PAC) pasa a N; la p(FA) por latido se devuelve
    para promediarla con la del modelo de RR. Devuelve (cls, p_fa, resumen).
    """
    from holterai.legacy.app.ml.ecgfounder_t158 import TARGETS, beat_probs
    _p(progress, "ECGFounder T158", 0.5)
    model = _get_t158()
    res = model.predict_recording(np.asarray(rec.samples), rec.fs, list(rec.leads),
                                  stride_s=opts.t158_stride_s)
    t = np.asarray(idx, dtype=np.float64) / rec.fs
    pmax = beat_probs(t, res["starts_s"], res["probs"], "max")
    pmean = beat_probs(t, res["starts_s"], res["probs"], "mean")
    cls = np.array(cls, dtype=int)
    iv, i_s, i_n = CLASSES.index("V"), CLASSES.index("S"), CLASSES.index("N")
    veto_v = (cls == iv) & (pmax[:, TARGETS.index("PVC")] < opts.t158_v_veto)
    veto_s = (cls == i_s) & (pmax[:, TARGETS.index("PAC")] < opts.t158_s_veto)
    cls[veto_v | veto_s] = i_n
    summ = {"device": model.device, "n_windows": int(len(res["starts_s"])),
            "v_to_n": int(veto_v.sum()), "s_to_n": int(veto_s.sum()),
            "v_veto": opts.t158_v_veto, "s_veto": opts.t158_s_veto,
            "af_weight": opts.t158_af_weight,
            "frac_fa_alta": round(float((res["probs"][:, 0] > 0.5).mean()), 4)
            if len(res["probs"]) else 0.0}
    return cls, pmean[:, TARGETS.index("FA")], summ


def rhythm_analysis(beats: list[Beat], opts: Options, progress=None,
                    unreadable: np.ndarray | None = None,
                    ext_af: np.ndarray | None = None):
    """Probabilidad de FA por latido y segmentos de ritmo.

    ``unreadable`` (booleano por latido) anula las probabilidades de los tramos
    que caen en ventanas ilegibles: el RR de un latido Q es ruido del detector,
    no fibrilación del paciente.
    """
    n = len(beats)
    if n < 12:
        return np.zeros(n), []
    # la FA se lee sobre la secuencia de ritmo: sin los Q que son onda T (T164),
    # que dejarían RR de 0,28 s seguidos de 0,77 s
    seq_pos = np.flatnonzero(ritmo_mask(beats))
    rr = np.array([b.rr_prev for b in ritmo_beats(beats)], dtype=float)
    win, stride = 30, 5
    try:
        model = RhythmModel.load(_model_path("rhythm_af.npz"))
    except Exception as exc:
        log.warning("modelo de ritmo no disponible: %s", exc)
        return np.zeros(n), []

    # cada latido toma la probabilidad de la última posición de la secuencia que le precede
    probs = af_probabilities(rr, model, win, stride)[
        np.searchsorted(seq_pos, np.arange(n), side="right") - 1]
    # T173: p(FA) por latido de ECGFounder T158, promediada con la del RR
    w_ext = float(getattr(opts, "t158_af_weight", 0.0))
    if ext_af is not None and len(ext_af) == n and w_ext > 0.0:
        probs = (1.0 - w_ext) * probs + w_ext * np.asarray(ext_af, dtype=float)
    # La ectopia (V o S) también produce RR irregular: si explica buena parte
    # del tramo, no se atribuye a FA. Se limita (no se anula) para no ocultar
    # una FA real con ectopia.
    for end in range(win, n + 1, stride):
        seg_beats = beats[end - win:end]
        v_frac = np.mean([b.cls == "V" for b in seg_beats])
        ect = np.mean([b.cls in ("V", "S") for b in seg_beats])
        if v_frac > 0.15:
            # bigeminismo ventricular: RR irregular sin ser FA
            probs[end - win:end] = np.minimum(probs[end - win:end], 0.30)
        elif ect > 0.40:
            probs[end - win:end] = np.minimum(probs[end - win:end], 0.40)
    if unreadable is not None and len(unreadable) == n and np.any(unreadable):
        unread = np.asarray(unreadable, dtype=bool)
        probs[unread] = 0.0
        for end in range(win, n + 1, stride):
            if np.mean(unread[end - win:end]) > 0.10:
                probs[end - win:end] = 0.0
    _p(progress, "Analizando ritmo", 0.65)

    # segmentos AF con histéresis y duración mínima
    segs: list[RhythmSegment] = []
    t = np.array([b.t for b in beats])
    state = False
    start = 0.0
    for i in range(n):
        on = probs[i] >= opts.af_threshold
        off = probs[i] < opts.af_threshold * 0.7
        if on and not state:
            state, start = True, t[i]
        elif off and state:
            state = False
            if t[i] - start >= opts.af_min_s:
                segs.append(RhythmSegment(start, t[i], "AF", float(np.mean(probs[i - 1:i + 1]))))
    if state and n and t[-1] - start >= opts.af_min_s:
        segs.append(RhythmSegment(start, t[-1], "AF", 0.5))
    return probs, segs


def _inside_spans(event: dict, spans, min_frac: float = 0.5) -> bool:
    """¿El evento cae mayormente dentro de alguno de los tramos dados?"""
    a, b = float(event["start_s"]), float(event["end_s"])
    dur = max(1e-6, b - a)
    for s in spans:
        lo = max(a, float(s["start_s"]))
        hi = min(b, float(s["end_s"]))
        if hi - lo >= min_frac * dur:
            return True
    return False


def _p_wave_times(decision: dict | None, fs: float) -> np.ndarray:
    """Tiempos (s) de las P que T10 encontró en sus latidos; vacío si no hay."""
    res = (decision or {}).get("p_res")
    if not res or fs <= 0:
        return np.zeros(0)
    present = np.asarray(res.get("present", []), dtype=bool)
    peak = np.asarray(res.get("peak", []), dtype=np.int64)
    if present.size == 0 or present.size != peak.size:
        return np.zeros(0)
    return peak[present & (peak >= 0)] / float(fs)


def build_events(beats: list[Beat], af_segs: list[RhythmSegment],
                 opts: Options, corr=None, duration_s: float = 0.0,
                 xf=None, fs: float = 250.0,
                 no_signal: list[dict] | None = None,
                 samples=None, p_waves=None, unreadable=None) -> list[Event]:
    raw = []
    # las pausas se revisan con la señal (QRS perdidos, bajo voltaje), las P de
    # T10 y los tramos de FA; sin ellas la causa salía solo del PP
    raw += pause_events(beats, opts.pause_s, samples=samples, fs=fs,
                        quality=unreadable, p_waves=p_waves, rhythm=af_segs)
    raw += rate_events(beats, opts.brady_bpm, opts.tachy_bpm)
    raw += ectopic_runs(beats)
    raw += bigeminy_events(beats)
    if corr is not None and opts.detect_no_organized:
        # EXPERIMENTAL: detecta FV en SDDB con ~7 s de retardo, pero también
        # marca bigeminismo y FA (14 falsos positivos/hora en MIT-BIH 119).
        # Desactivado por defecto hasta tener un detector espectral propio.
        cand = no_organized_rhythm_events(beats, corr)
        raw += cand
    if duration_s > 0:
        raw += gap_events(beats, duration_s)

    # durante un ritmo no organizado el detector pierde latidos: esos huecos
    # no son asistolia, no deben reportarse como tal
    no_org = [e for e in raw if e["kind"] == "no_organized"]

    def _inside_no_org(e) -> bool:
        return any(not (e["end_s"] < n["start_s"] - 2.0 or e["start_s"] > n["end_s"] + 2.0)
                   for n in no_org)

    raw = [e for e in raw if e["kind"] != "asystole" or not _inside_no_org(e)]
    # sin señal no hay pausa ni asistolia: el hueco lo explica la desconexión
    if no_signal:
        raw = [e for e in raw
               if e["kind"] not in ("pause", "asystole") or not _inside_spans(e, no_signal)]
        raw.extend(no_signal)
    for s in af_segs:
        raw.append({
            "kind": "afib", "start_s": s.start_s, "end_s": s.end_s,
            "severity": "warn",
            "detail": f"FA ({s.end_s - s.start_s:.0f} s)",
            "beats": [],
        })
    out = []
    for e in raw:
        out.append(Event(
            kind=e["kind"], start_s=float(e["start_s"]), end_s=float(e["end_s"]),
            severity=e.get("severity", "info"), detail=e.get("detail", ""),
            beats=[int(x) for x in e.get("beats", [])][:200],
        ))
    out.sort(key=lambda e: e.start_s)
    return out


# ---------------------------------------------------------------------------
# 4. Resumen y calidad
# ---------------------------------------------------------------------------
def summarize(beats: list[Beat], events: list[Event], af_segs: list[RhythmSegment],
              duration_s: float, quality: dict) -> dict:
    n = len(beats)
    counts = {"N": 0, "S": 0, "V": 0, "F": 0, "Q": 0}
    for b in beats:
        counts[b.cls] = counts.get(b.cls, 0) + 1
    rr_raw = np.array([b.rr_prev for b in beats[1:]], dtype=float)
    rr = rr_raw[(rr_raw > 0.2) & (rr_raw < 5.0)]
    rr_plausible = (
        round(float(100.0 * np.mean((rr_raw > 0.30) & (rr_raw < 2.0))), 1)
        if rr_raw.size else 0.0
    )
    hr = 60.0 / rr if rr.size else np.array([])
    af_dur = sum(s.end_s - s.start_s for s in af_segs)
    hrv = hrv_mod.compute(beats, duration_s)
    dur_s = max(1.0, float(duration_s))
    summary = {
        "duration_h": round(duration_s / 3600.0, 2),
        "n_beats": n,
        "counts": counts,
        "hr_mean": round(float(np.mean(hr)), 1) if hr.size else 0.0,
        "hr_min": round(float(np.percentile(hr, 1)), 1) if hr.size else 0.0,
        "hr_max": round(float(np.percentile(hr, 99)), 1) if hr.size else 0.0,
        "pvc_burden_pct": round(100.0 * counts.get("V", 0) / max(1, n), 2),
        "pac_burden_pct": round(100.0 * counts.get("S", 0) / max(1, n), 2),
        "af_burden_pct": round(100.0 * af_dur / max(1.0, duration_s), 2),
        "n_pauses": sum(1 for e in events if e.kind == "pause"),
        "n_vetoed": sum(1 for b in beats if b.features.get("vetoed")),
        "leads_fused": int(quality.get("n_leads", 1)),
        "n_no_organized": sum(1 for e in events if e.kind == "no_organized"),
        "n_asystole": sum(1 for e in events if e.kind == "asystole"),
        "n_no_signal": sum(1 for e in events if e.kind == "no_signal"),
        "n_af_episodes": len(af_segs),
        "hrv": hrv,
        "rr_plausible_pct": rr_plausible,
        "qrs_quality": "ok" if rr_plausible >= 70.0 else "revisar",
        "qrs_rate_mean": round(float(60.0 / np.median(rr)), 1) if rr.size else 0.0,
        "artifact_pct": round(100.0 * float(hrv.get("artifact_s", 0.0)) / 1000.0 / dur_s, 1),
        "ectopy_excluded_pct": round(100.0 * float(hrv.get("ectopy_s", 0.0)) / 1000.0 / dur_s, 1),
        "analyzed_pct": float(hrv.get("pct_used", 0.0)),
        "unreadable_s": round(float(quality.get("unreadable_s", 0.0)), 1),
        "unreadable_pct": round(float(quality.get("unreadable_pct", 0.0)), 2),
        "quality": quality,
    }
    return summary


def hrv_metrics(rr: np.ndarray) -> dict:
    rr = np.asarray(rr, dtype=float)
    if rr.size < 3:
        return {}
    drr = np.diff(rr)
    sdnn = float(np.std(rr, ddof=1)) * 1000.0
    rmssd = float(np.sqrt(np.mean(drr ** 2))) * 1000.0
    pnn50 = float(np.mean(np.abs(drr) > 0.05)) * 100.0
    # índice triangular (histograma de 7.8 ms)
    bins = np.arange(rr.min(), rr.max() + 0.0078, 0.0078)
    if bins.size > 2:
        hist, _ = np.histogram(rr, bins=bins)
        tri = float(rr.size / max(1, hist.max()))
    else:
        tri = 0.0
    return {
        "sdnn_ms": round(sdnn, 1),
        "rmssd_ms": round(rmssd, 1),
        "pnn50_pct": round(pnn50, 1),
        "triangular_index": round(tri, 1),
    }


def _model_path(name: str):
    from holterai.legacy.app.core.config import MODELS_DIR
    return MODELS_DIR / "pretrained" / name


_CALIBRATION: Calibration | None = None
_SVPB_CONFIG: svpb.SvpbConfig | None = None


def get_calibration() -> Calibration:
    """Calibración del clasificador publicada en `models/pretrained/`.

    Si el JSON no está (o está roto) se devuelve la calibración identidad: el
    pipeline sigue funcionando con las probabilidades crudas del modelo.
    """
    global _CALIBRATION
    if _CALIBRATION is None:
        try:
            _CALIBRATION = Calibration.load(_model_path("beat_calibration.json"))
            log.info("calibración de latidos: T=%.3f, prevalencia objetivo %s",
                     _CALIBRATION.temperature,
                     [round(float(v), 4) for v in _CALIBRATION.target_prior])
        except Exception as exc:
            log.warning("sin calibración de latidos: %s", exc)
            _CALIBRATION = Calibration(enabled=False)
    return _CALIBRATION


# ---------------------------------------------------------------------------
# 5. Orquestador
# ---------------------------------------------------------------------------

def _suppress_normal_lookalikes(wins: np.ndarray, cls: np.ndarray,
                                rr_prev: np.ndarray, fs: float,
                                keep: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """No llamar S a un latido que es morfológicamente el normal del paciente.

    Es una regla conservadora: solo actúa sobre S, exige correlación muy alta
    con la plantilla dominante y que el latido no sea prematuro. No toca V.
    ``keep`` protege los latidos que la capa clínica (``signal.svpb``) marcó
    como S con todas sus compuertas: ahí la prematuridad ya está medida contra
    el RR local, así que esta regla global no debe deshacerla.
    """
    n = len(cls)
    normal = np.flatnonzero(cls == 0)
    if n < 40 or normal.size < max(20, int(0.20 * n)):
        return cls, 0
    tpl = np.median(wins[normal], axis=0).astype(np.float64)
    tn = float(np.linalg.norm(tpl))
    if tn <= 1e-9:
        return cls, 0
    rr = np.asarray(rr_prev, dtype=np.float64)
    valid = np.isfinite(rr) & (rr > 0.25) & (rr < 3.0)
    rr_med = float(np.median(rr[valid])) if valid.any() else 0.8
    if rr_med <= 0:
        return cls, 0
    changed = 0
    for i in np.flatnonzero(cls == 1):
        if keep is not None and bool(keep[i]):
            continue
        w = wins[i].astype(np.float64)
        wn = float(np.linalg.norm(w))
        if wn <= 1e-9:
            continue
        corr = float(np.dot(w, tpl) / (wn * tn))
        if corr >= 0.985 and rr[i] > 0.80 * rr_med:
            cls[i] = 0
            changed += 1
    return cls, changed


def _mark_artifact_beats(noises: np.ndarray, agreement: np.ndarray,
                         cls: np.ndarray) -> np.ndarray:
    """Marca como Q los ectópicos propuestos en tramos muy ruidosos.

    No borra el latido: queda como Q y se informa aparte. Solo actúa sobre S/V/F.
    """
    artifact = np.zeros(len(cls), dtype=bool)
    if len(cls) == 0:
        return artifact
    bad = (np.asarray(noises) > 0.15) & (np.asarray(agreement) < 0.75)
    ect = np.isin(cls, [1, 2, 3])
    artifact[:] = bad & ect
    cls[artifact] = 4
    return artifact


def _extend_qrs_by_template(xf: np.ndarray, idx: np.ndarray, fs: float,
                            max_extra_frac: float = 0.35) -> np.ndarray:
    """Busca QRS de bajo voltaje por correlación con la plantilla dominante."""
    try:
        from scipy.signal import fftconvolve, find_peaks
    except Exception:
        return np.zeros(0, dtype=int)
    if idx.size < 20:
        return np.zeros(0, dtype=int)
    pre = int(0.12 * fs)
    post = int(0.12 * fs)
    windows = []
    for p in idx:
        p = int(p)
        if p - pre >= 0 and p + post < xf.size:
            windows.append(xf[p - pre:p + post])
    if len(windows) < 10:
        return np.zeros(0, dtype=int)
    tpl = np.median(np.asarray(windows), axis=0).astype(np.float64)
    ts = float(np.std(tpl))
    if ts <= 1e-9:
        return np.zeros(0, dtype=int)
    tpl = (tpl - float(np.mean(tpl))) / ts
    y = np.asarray(xf, dtype=np.float64)
    ys = float(np.std(y))
    if ys <= 1e-9:
        return np.zeros(0, dtype=int)
    y = (y - float(np.mean(y))) / ys
    corr = fftconvolve(y, tpl[::-1], mode="same") / max(1, tpl.size)
    peaks, _ = find_peaks(corr, height=0.45, distance=max(1, int(0.28 * fs)))
    if peaks.size == 0:
        return np.zeros(0, dtype=int)
    keep = []
    limit = int(max_extra_frac * idx.size)
    for p in peaks:
        if np.any(np.abs(idx - p) < int(0.20 * fs)):
            continue
        keep.append(int(p))
        if len(keep) >= limit:
            break
    return np.asarray(keep, dtype=int)



def _rebuild_beat_arrays(xf: np.ndarray, fs: float, idx: np.ndarray):
    wins = FT.extract_windows_batch(xf, fs, idx)
    widths = np.array([FT.qrs_width(xf, fs, int(p)) for p in idx])
    amps = np.array([
        float(np.max(np.abs(xf[max(0, p - int(0.1 * fs)):p + int(0.1 * fs)])))
        for p in idx
    ])
    noises = FT.noise_batch(xf, fs, idx)
    return idx, wins, widths, amps, noises


def _extend_qrs_multilead(rec: Recording, xf: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Añade QRS que el canal primario perdió y que ≥2 canales confirman.

    La unión sin control de las detecciones de todos los canales mete ruido de
    los canales malos; `detect_qrs_multilead` solo admite candidatos de baja
    prominencia cuando otra derivación los respalda.
    """
    if rec.n_leads < 2:
        return np.zeros(0, dtype=int)
    try:
        peaks, _confirmed = detect_qrs_multilead(
            np.asarray(rec.samples, dtype=np.float32), rec.fs)
    except Exception as exc:
        log.warning("fusión multicanal falló: %s", exc)
        return np.zeros(0, dtype=int)
    if peaks.size == 0:
        return np.zeros(0, dtype=int)
    return np.array([int(p) for p in peaks
                     if not np.any(np.abs(idx - p) < int(0.20 * rec.fs))],
                    dtype=int)


def _artifact_mask(rec: Recording):
    """Máscara de ventanas ilegibles por artefactos (``signal/artifact.py``)."""
    try:
        return artifact_mod.quality_mask(rec.samples, rec.fs)
    except Exception as exc:  # una máscara rota no debe tumbar el análisis
        log.warning("no se pudo evaluar la calidad de la señal: %s", exc)
        return None


def _lead_quality_note(lead: int, mask, rec: Recording, quality: dict) -> None:
    """Anota en ``quality`` si la derivación analizada es la peor del registro.

    La máscara **no** cambia de derivación: en Exam_1424 pasar de la 1 a la 2
    bajaba la sensibilidad de QRS de 0,9997 a 0,9565 (medido con
    ``scripts/eval_artifact.py``), y el criterio de aceptación de T40 exige que
    la detección no se mueva. La elección de derivación sigue siendo la de T32;
    aquí solo se informa para que el revisor sepa que hay un canal mejor.
    """
    if mask is None or mask.n_leads < 2 or lead >= mask.n_leads:
        return
    pct = np.asarray(mask.bad_pct, dtype=float)
    best = int(np.argmin(pct))
    quality["lead_bad_pct"] = round(float(pct[lead]), 1)
    if best != lead and pct[lead] > 30.0 and pct[best] < 10.0:
        quality["better_lead"] = best
        quality["better_lead_bad_pct"] = round(float(pct[best]), 1)
        log.info("artefactos: la derivación %d está mala en %.0f %% del registro "
                 "y la %d en %.0f %% (no se cambia: movería la sensibilidad de QRS)",
                 lead, pct[lead], best, pct[best])


def _unreadable_times(mask, idx: np.ndarray, fs: float) -> np.ndarray:
    """Máscara por latido: ¿cae su ventana de 2 s en un tramo ilegible?"""
    if mask is None or not len(idx):
        return np.zeros(len(idx), dtype=bool)
    return np.asarray(mask.at(np.asarray(idx, dtype=float) / float(fs)), dtype=bool)


def _drop_in_spans(events: list[Event], spans: list[dict]) -> list[Event]:
    """Descarta los eventos que caen mayormente dentro de un tramo ilegible.

    ``no_signal`` se conserva: no es una afirmación sobre el ritmo sino la
    explicación del hueco (electrodo desconectado); borrarlo dejaría el registro
    sin la única advertencia útil de ese tramo.
    """
    if not spans:
        return events
    return [e for e in events
            if e.kind == "no_signal"
            or not artifact_mod.inside_spans(e.start_s, e.end_s, spans)]


def _av_windows(duration_s: float, opts: Options) -> list[tuple[float, float]]:
    """Tramos (a, b) del registro donde se mide la conducción AV.

    El análisis de P y bloqueos recorre el registro completo buscando el patrón
    de P en cada muestra: en un Holter de 48 h eso cuesta más que todo el resto
    del pipeline. Se analizan ``av_windows`` tramos de ``av_window_s`` repartidos
    a lo largo del registro: alcanza para ver la tendencia del PR y para buscar
    bloqueos, que son eventos de minutos.
    """
    if duration_s <= 0:
        return []
    w = min(float(opts.av_window_s), duration_s)
    n = int(max(1, min(opts.av_windows, int(duration_s // max(w, 1.0)))))
    if n == 1:
        return [(0.0, min(duration_s, w))]
    span = duration_s - w
    return [(span * k / (n - 1), span * k / (n - 1) + w) for k in range(n)]


KIND_ISO = "av_isorritmica"   # disociación AV isorrítmica (T164): no es BAV


def _isorrhythmic_window(res, fs: float) -> list[dict]:
    """Tramos de disociación isorrítmica de un tramo AV (tiempos locales al tramo)."""
    from holterai.legacy.app.signal.rhythm import isorrhythmic_runs

    tr = res.train
    if tr is None or len(tr) == 0:
        return []
    cond = np.asarray(tr.conducted, dtype=bool)
    qon = np.asarray(res.qrs_on, dtype=np.float64)
    bi = np.asarray(tr.beat, dtype=int)
    qrs = np.full(len(tr), np.nan)
    ok = cond & (bi >= 0) & (bi < qon.size)
    ok[ok] &= qon[bi[ok]] >= 0
    qrs[ok] = qon[bi[ok]] / float(fs)
    pr = np.where(cond, np.asarray(tr.pr_ms, dtype=np.float64), np.nan)
    return isorrhythmic_runs(np.asarray(tr.peak, dtype=np.float64) / float(fs), cond, qrs, pr)


def _dentro_de_tramos(e: dict, tramos: list[dict], margen_s: float = 1.0) -> bool:
    """¿El evento de bloqueo cae dentro de un tramo isorrítmico (con margen)?"""
    return any(e["start_s"] >= t["start_s"] - margen_s and e["end_s"] <= t["end_s"] + margen_s
               for t in tramos)


def av_conduction_analysis(rec: Recording, beats: list[Beat], idx: np.ndarray,
                           xf: np.ndarray, lead: int, opts: Options,
                           af_segs: list | None = None) -> tuple[list[Event], dict]:
    """Conducción AV (T24): PR por latido y por minuto, tendencia y bloqueos.

    Todo evento sale **a revisión** y con la tira de 10 s que lo muestra
    (``strip_start_s``/``strip_end_s`` en el detalle). No se mide PR en FA ni en
    latidos ventriculares: la relación P-QRS no significa nada ahí.
    """
    from holterai.legacy.app.signal import av_conduction as av

    fs = rec.fs
    idx = np.asarray(idx, dtype=int)
    if idx.size < 12:
        return [], {}
    af = list(af_segs or [])
    in_af = np.zeros(idx.size, dtype=bool)
    for s in af:
        in_af |= (idx / fs >= s.start_s) & (idx / fs <= s.end_s)
    ok_cls = np.zeros(idx.size, dtype=bool)
    cls_ok = np.array([b.cls in ("N", "S") for b in beats], dtype=bool)
    k = min(idx.size, cls_ok.size)
    ok_cls[:k] = cls_ok[:k]
    beat_ok = ok_cls & ~in_af
    other = [j for j in range(rec.n_leads) if j != lead][:2]
    events: list[Event] = []
    serie: list[list] = []
    pr_vals: list[np.ndarray] = []
    n_blocks: dict[str, int] = {}
    n_iso = 0
    n_ok_win = 0
    windows = _av_windows(rec.duration_s, opts)
    for a_s, b_s in windows:
        a, b = int(a_s * fs), int(min(b_s * fs, rec.n_samples))
        sel = np.flatnonzero((idx >= a) & (idx < b))
        if sel.size < 12:
            continue
        try:
            xw = np.asarray(xf[a:b], dtype=np.float64)
            leads = [preprocess(np.asarray(rec.samples[a:b, j], dtype=np.float64),
                                fs, "display").astype(np.float64) for j in other]
            local = idx[sel] - a
            res = av.av_analysis(xw, fs, local, leads=leads or None,
                                 beat_ok=beat_ok[sel],
                                 duration_s=(b - a) / fs)
        except Exception as exc:
            log.warning("conducción AV no disponible en %.0f-%.0f s: %s", a_s, b_s, exc)
            continue
        # disociación AV isorrítmica (T164): tramo con P conducidas, A≈V y PR variable.
        # No es un BAV: los bloqueos que caigan dentro no se cuentan como tales.
        iso = _isorrhythmic_window(res, fs)
        for r in iso:
            events.append(Event(
                kind=KIND_ISO, start_s=float(r["start_s"]) + a_s,
                end_s=float(r["end_s"]) + a_s, severity="info",
                detail=(f"Disociación AV isorrítmica: A {r['a_lpm']:.0f} lpm, V {r['v_lpm']:.0f} lpm "
                        f"durante {r['n']} latidos, PR {r['pr_min_ms']:.0f}-{r['pr_max_ms']:.0f} ms "
                        f"(no es BAV) · tira {r['start_s'] + a_s:.0f}-{r['end_s'] + a_s:.0f} s"),
                beats=[]))
            n_iso += 1
        for e in av.block_events(res):
            if _dentro_de_tramos(e, iso):
                continue
            detail = e.get("detail", "")
            strip = (e.get("strip_start_s", e["start_s"]), e.get("strip_end_s", e["end_s"]))
            events.append(Event(
                kind=str(e["kind"]), start_s=float(e["start_s"]) + a_s,
                end_s=float(e["end_s"]) + a_s, severity="warn",
                detail=f"{detail} · tira {strip[0] + a_s:.0f}-{strip[1] + a_s:.0f} s",
                beats=[int(sel[k]) for k in e.get("beats", []) if 0 <= k < sel.size]))
            n_blocks[str(e["kind"])] = n_blocks.get(str(e["kind"]), 0) + 1
        for t0, val, cnt in res.minute_pr:
            serie.append([float(t0) + a_s, float(val), int(cnt)])
        vals = np.asarray(res.pr_ms, dtype=np.float64)
        keep = np.asarray(res.pr_valid, dtype=bool) & np.isfinite(vals)
        if keep.any():
            pr_vals.append(vals[keep])
        n_ok_win += int(beat_ok[sel].sum())
    if not serie and not pr_vals:
        return events, {}
    allpr = np.concatenate(pr_vals) if pr_vals else np.zeros(0)
    trend = av.pr_trend(serie)
    info = {
        "review": True,
        "av_windows": len(windows),
        "pr_measured": int(allpr.size),
        "pr_measured_pct": round(100.0 * allpr.size / max(1, n_ok_win), 1),
        "pr_median_ms": round(float(np.median(allpr)), 1) if allpr.size else None,
        "pr_min_ms": round(float(np.percentile(allpr, 5)), 1) if allpr.size else None,
        "pr_max_ms": round(float(np.percentile(allpr, 95)), 1) if allpr.size else None,
        "first_degree_pct": round(100.0 * float(np.mean(allpr > av.PR_FIRST_DEGREE_MS)), 1)
        if allpr.size else None,
        "blocks": {k: v for k, v in n_blocks.items() if v},
        "isorrhythmic": n_iso,
        "note": "bloqueos AV y PR a revisión; se miden en los tramos analizados",
    }
    info.update({k: v for k, v in trend.items() if k not in info})
    return events, info


def _af_response(beats: list[Beat], af_segs: list) -> dict:
    """FC mediana y rango durante episodios de FA."""
    if not af_segs:
        return {}
    rates = [60.0 / b.rr_prev for b in beats
             if b.rr_prev > 0.2
             and any(a.start_s <= b.t <= a.end_s for a in af_segs)]
    if not rates:
        return {}
    return {
        "af_response_bpm": round(float(np.median(rates)), 1),
        "af_hr_min": round(float(np.min(rates)), 1),
        "af_hr_max": round(float(np.max(rates)), 1),
    }


def st_events(beats: list[Beat], crit: "ST.StCriteria", lead: int = 0,
              lead_name: str = "") -> tuple[list[Event], dict]:
    """Episodios ST (exploratorio) como eventos + bloque de resumen.

    Solo lee `beat.features["st_uv"]`, así que cambiar umbrales o punto J no
    requiere volver a detectar latidos.
    """
    eps = ST.episodes(beats, crit, lead=lead, lead_name=lead_name)
    events = [
        Event(kind=e["kind"], start_s=float(e["start_s"]), end_s=float(e["end_s"]),
              severity=e["severity"], detail=e["detail"])
        for e in ST.event_dicts(eps, crit)
    ]
    return events, ST.summary_block(beats, crit, lead=lead, lead_name=lead_name)


def apply_st(analysis: Analysis, crit: "ST.StCriteria", lead: int | None = None,
             lead_name: str | None = None) -> dict:
    """Recalcula los episodios ST de un análisis ya hecho (sin redetectar)."""
    lead_i = int(analysis.quality.get("lead", 0) if lead is None else lead)
    if lead_name is None:
        rec = analysis.recording
        lead_name = rec.leads[lead_i] if lead_i < len(rec.leads) else f"L{lead_i + 1}"
    events, block = st_events(analysis.beats, crit, lead=lead_i, lead_name=lead_name)
    analysis.events = [e for e in analysis.events if e.kind not in ("st_depression", "st_elevation")]
    analysis.events.extend(events)
    analysis.events.sort(key=lambda e: e.start_s)
    analysis.summary["st"] = block
    analysis.summary["st_depressions"] = block.get("n_depressions", 0)
    analysis.summary["st_elevations"] = block.get("n_elevations", 0)
    if block.get("median_uv") is not None:
        analysis.summary["st_median_uv"] = block["median_uv"]
    analysis.summary["n_st_episodes"] = block.get("n_episodes", 0)
    return block


# ---------------------------------------------------------------------------
# 5. Onda P, PR y morfología del QRS (T33)
#
# `signal/pwave.py` (T10) y `signal/qrs_morph.py` (T25) existían sin conectar.
# Aquí se pasan minuto a minuto sobre el registro completo:
#
#   * onda P por latido con la plantilla del minuto (pwave.detect_p_waves),
#   * PR = inicio de la P (por latido) al inicio del QRS del latido promediado
#     del minuto (`features.qrs_bounds_channel` sobre el canal de análisis),
#   * duración y patrón del QRS sobre el latido promediado del minuto
#     (`features.qrs_duration_multichannel` + `qrs_morph.classify_beat`), con
#     `qrs_morph.summarise_episode` para decidir persistente / intermitente /
#     dependiente de FC.
#
# Un minuto con mala calidad (plano, saturado, ruidoso o con la mayoría de sus
# latidos marcados como ruidosos) o con menos latidos de los necesarios queda
# **no evaluable** y no entra en los denominadores.
# ---------------------------------------------------------------------------
MINUTE_S = 60.0
MORPH_MIN_BEATS = 5          # latidos por minuto para promediar el QRS
PWAVE_MIN_EVAL = 3           # latidos con P evaluable por minuto
PR_VALID_MS = (60.0, 400.0)  # PR fisiológicamente posible; fuera se descarta
MORPH_PRE_S = 0.30           # ventana del latido promediado (igual que median_beat)
MORPH_POST_S = 0.45
MORPH_MAX_CHANNELS = 3       # canal de análisis + hasta 2 (el H3+ exporta 3)
MORPH_MAX_BASELINE_BEATS = 240
KNOWN_LEADS = ("I", "II", "III", "AVR", "AVL", "AVF",
               "V1", "V2", "V3", "V4", "V5", "V6")
NOISE_REL_MAX = 0.9          # ruido relativo (std de la derivada / std) tolerable
NOISE_REL_FACTOR = 1.5       # ... y veces la mediana del propio registro
NOISY_BEAT = 0.08            # umbral de `features.noise` del detector


def clock_label(start_time, t_s: float) -> str:
    """Etiqueta de hora para un instante del registro.

    Con `Recording.start_time` da la hora local; sin él (WFDB exportado sin
    fecha) da el tiempo transcurrido desde el inicio, que es lo único cierto.
    """
    t = max(0.0, float(t_s))
    base = start_time
    if isinstance(base, str):
        try:
            base = datetime.fromisoformat(base)
        except ValueError:
            base = None
    if isinstance(base, datetime):
        return (base + timedelta(seconds=t)).strftime("%H:%M")
    hours, minutes = int(t // 3600), int((t % 3600) // 60)
    return f"{hours:02d}:{minutes:02d}"


def minute_quality(xf: np.ndarray, fs: float, idx: np.ndarray,
                   noises: np.ndarray | None = None,
                   minute_s: float = MINUTE_S) -> dict:
    """Minutos con señal utilizable en el canal de análisis.

    Reproduce en el pipeline los criterios del mapa de calidad de la sesión
    (`server.Session.quality_map`: plano, saturado y ruido por minuto), que no
    se puede usar aquí porque `analyze` corre sin sesión y `server` importa a
    `pipeline`. Dos diferencias deliberadas: el ruido se mide sobre el canal ya
    filtrado y su umbral es relativo al propio registro, para no descartar un
    Holter ruidoso entero. Un minuto es **no evaluable** si:

    * la señal está plana o saturada (> 4,5 mV), o
    * su ruido relativo (std de la derivada / std) supera a la vez
      `NOISE_REL_MAX` y `NOISE_REL_FACTOR` veces la mediana del propio registro,
      que es lo que distingue un tramo ruidoso de un registro ruidoso entero, o
    * más de la mitad de sus latidos los marcó el detector como ruidosos
      (`features.noise > 0.08`, el mismo umbral que informa `quality`).
    """
    x = np.asarray(xf, dtype=np.float64)
    idx = np.asarray(idx, dtype=np.int64)
    step = max(1, int(minute_s * fs))
    n_min = int(max(1, np.ceil(x.size / step)))
    noises = (np.zeros(idx.size) if noises is None
              else np.asarray(noises, dtype=np.float64).ravel())
    beat_minute = (idx / step).astype(int) if idx.size else np.zeros(0, int)
    ok = np.zeros(n_min, dtype=bool)
    noise_rel = np.full(n_min, np.nan, dtype=np.float64)
    for m in range(n_min):
        seg = x[m * step:(m + 1) * step]
        if seg.size < 8:
            continue
        std = float(np.std(seg))
        noise_rel[m] = float(np.std(np.diff(seg))) / (abs(std) + 1e-9)
        flat = std < 1e-5
        saturated = float(np.max(np.abs(seg))) > 4.5
        ok[m] = (not flat) and (not saturated)
    finite = noise_rel[np.isfinite(noise_rel)]
    med = float(np.median(finite)) if finite.size else 0.0
    limit = max(NOISE_REL_MAX, NOISE_REL_FACTOR * med)
    ok &= np.isfinite(noise_rel) & (noise_rel <= limit)
    for m in np.flatnonzero(ok):
        if noises.size != idx.size:
            break
        sel = beat_minute == m
        if bool(sel.any()) and float(np.mean(noises[sel] > NOISY_BEAT)) > 0.5:
            ok[m] = False
    return {
        "bin_s": float(minute_s),
        "ok": ok,
        "noise_rel": np.round(noise_rel, 4),
        "noise_rel_median": round(med, 4),
        "noise_rel_limit": round(limit, 4),
        "n_ok": int(ok.sum()),
        "n_minutes": int(n_min),
    }


def _morph_channels(rec: Recording, lead: int) -> list[int]:
    """Canal de análisis primero y hasta 3 canales con señal para promediar."""
    out = [int(lead)]
    if rec.n_leads < 2:
        return out
    fs = rec.fs
    probe = min(rec.n_samples, int(60 * fs))
    cand: list[tuple[int, float]] = []
    for ch in range(rec.n_leads):
        if ch == lead:
            continue
        try:
            seg = np.asarray(rec.samples[:probe, ch], dtype=np.float64)
            q = multilead.lead_quality(preprocess(seg, fs, "display"), fs)
        except Exception:                                   # canal ilegible
            q = 0.0
        if q > 0.2:
            cand.append((ch, q))
    cand.sort(key=lambda t: -t[1])
    out.extend(ch for ch, _q in cand[:MORPH_MAX_CHANNELS - 1])
    return out


def _roles_for(rec: Recording, channels: list[int]) -> dict[int, str]:
    """Nombre reconocible de cada canal (`Lead1`, ... no se adivina)."""
    roles = {}
    for k, ch in enumerate(channels):
        name = str(rec.leads[ch] if ch < len(rec.leads) else "").upper().replace(" ", "")
        if name in KNOWN_LEADS:
            roles[k] = name
    return roles


def _minute_medians(sig: np.ndarray, idx: np.ndarray, groups: dict[int, np.ndarray],
                    pre: int, post: int) -> dict[int, np.ndarray]:
    """Mediana punto a punto de las ventanas de cada minuto, para un canal.

    Es el mismo cálculo que `features.median_beat`, pero extrayendo sólo la
    ventana de cada latido y agrupando los minutos de a tandas: así el canal
    filtrado puede quedarse en float32 (62 MB por 24 h) y la mediana se hace en
    float64, que es mucho más rápida.
    """
    offsets = np.arange(-pre, post)
    out: dict[int, np.ndarray] = {}
    minutes = list(groups)
    block = 120                                     # minutos por tanda
    for start in range(0, len(minutes), block):
        chunk = minutes[start:start + block]
        counts = [int(groups[m].size) for m in chunk]
        sel = np.concatenate([groups[m] for m in chunk])
        win = np.asarray(sig[idx[sel][:, None] + offsets[None, :]], dtype=np.float64)
        pos = 0
        for m, count in zip(chunk, counts, strict=True):
            out[int(m)] = np.median(win[pos:pos + count], axis=0)
            pos += count
    return out


def _minute_beats(rec: Recording, idx: np.ndarray, channels: list[int], q: dict,
                  xf: np.ndarray, minute_s: float,
                  progress=None) -> tuple[list[dict], dict[int, str]]:
    """Latido promediado (mediana) de cada minuto evaluable.

    El canal de análisis ya está filtrado (`xf`); los demás se filtran una vez
    por registro y se descartan tras promediar, así el pico de memoria es de un
    canal (62 MB por 24 h a 180 Hz) y no de tres.

    Devuelve los minutos medidos y el motivo de los que quedaron fuera.
    """
    fs = float(rec.fs)
    idx = np.asarray(idx, dtype=np.int64)
    step = max(1, int(minute_s * fs))
    n_min = int(q["ok"].size)
    drops: dict[int, str] = {}
    if idx.size == 0:
        return [], {m: "sin_latidos" for m in range(n_min)}
    pre = int(round(MORPH_PRE_S * fs))
    post = int(round(MORPH_POST_S * fs))
    beat_minute = (idx / step).astype(int)
    order = np.argsort(beat_minute, kind="stable")
    sorted_min = beat_minute[order]
    bounds = np.searchsorted(sorted_min, np.arange(n_min + 1))
    groups: dict[int, np.ndarray] = {}
    for m in range(n_min):
        if not bool(q["ok"][m]):
            drops[m] = "calidad"
            continue
        sel = order[bounds[m]:bounds[m + 1]]
        if sel.size < MORPH_MIN_BEATS:
            drops[m] = "pocos_latidos"
            continue
        if idx[sel[0]] < pre or idx[sel[-1]] + post > rec.n_samples:
            drops[m] = "borde"
            continue
        groups[int(m)] = np.sort(sel)
    if not groups:
        return [], drops
    columns: list[dict[int, np.ndarray]] = []
    for k, ch in enumerate(channels):
        if k == 0:
            sig = np.asarray(xf, dtype=np.float32)        # ya filtrado
        else:
            sig = preprocess(np.asarray(rec.samples[:, ch], dtype=np.float64),
                             fs, "display")
        columns.append(_minute_medians(sig, idx, groups, pre, post))
        del sig
        _p(progress, "Promediando latidos por minuto", 0.80)
    out: list[dict] = []
    for m, sel in groups.items():
        beat = np.column_stack([c[m] for c in columns])
        rr = float(np.median(np.diff(idx[sel])) / fs) if sel.size > 1 else 0.0
        out.append({"minute": int(m), "ids": sel, "n": int(sel.size),
                    "beat": beat, "center": pre, "rr_s": rr,
                    "hr_bpm": (60.0 / rr if rr > 0 else 0.0)})
    return out, drops


def _baseline_beat(minutes: list[dict]) -> np.ndarray | None:
    """Latido de referencia del ritmo de base (mediana de los minutos)."""
    beats = [m["beat"] for m in minutes]
    if not beats:
        return None
    if len(beats) > MORPH_MAX_BASELINE_BEATS:
        picks = np.linspace(0, len(beats) - 1, MORPH_MAX_BASELINE_BEATS).astype(int)
        beats = [beats[i] for i in picks]
    return np.median(np.stack(beats), axis=0)


def _beat_payload(beat: np.ndarray, center: int, fs: float, names: list[str],
                  markers: dict) -> dict:
    """Latido promediado en JSON para dibujarlo con la P y el PR encima."""
    n_ch = min(2, beat.shape[1])
    t_ms = (np.arange(beat.shape[0]) - center) * 1000.0 / fs
    return {
        "t_ms": [round(float(v), 1) for v in t_ms],
        "channels": [[round(float(v), 3) for v in beat[:, k]] for k in range(n_ch)],
        "names": names[:n_ch],
        "markers": {k: (None if v is None else round(float(v), 1))
                    for k, v in markers.items()},
    }


def p_and_qrs_morphology(rec: Recording, xf: np.ndarray, idx: np.ndarray,
                         noises: np.ndarray | None = None,
                         lead: int = 0, minute_s: float = MINUTE_S,
                         progress=None) -> dict:
    """Onda P, PR y morfología del QRS minuto a minuto (bloque `summary`).

    Devuelve ``{"pwave", "qrs_morph", "minutes", "beats"}``: los dos bloques de
    resumen, la tabla por minuto y los arreglos por latido (P presente, amplitud
    de la P, PR y QRS ancho) alineados con ``idx``.
    """
    fs = float(rec.fs)
    idx = np.asarray(idx, dtype=np.int64)
    n = idx.size
    empty_beats = {
        "p_present": np.zeros(n), "p_amp_uv": np.full(n, np.nan),
        "pr_ms": np.full(n, np.nan), "qrs_wide": np.full(n, np.nan),
    }
    out = {"pwave": {}, "qrs_morph": {}, "minutes": [], "beats": empty_beats}
    if n < 3:
        out["pwave"] = {"n_minutes": 0, "n_evaluable": 0, "note": "registro sin latidos"}
        out["qrs_morph"] = {"n_minutes": 0, "kind": "sin_datos"}
        return out

    step = max(1, int(minute_s * fs))
    n_min = int(max(1, np.ceil(rec.n_samples / step)))
    q = minute_quality(xf, fs, idx, noises, minute_s)
    if q["ok"].size < n_min:                    # canal más corto que el registro
        pad = np.zeros(n_min - q["ok"].size, dtype=bool)
        q["ok"] = np.concatenate([q["ok"], pad])
        q["noise_rel"] = np.concatenate([q["noise_rel"], np.ones(pad.size)])

    # --- onda P: una pasada, plantilla por minuto y control par/impar --------
    p_res = pwave_mod.detect_p_waves(np.asarray(xf, dtype=np.float64), fs, idx)
    p_eval = np.asarray(p_res["evaluable"], dtype=bool)
    p_present = p_eval & np.asarray(p_res["present"], dtype=bool)
    out["beats"]["p_present"] = p_present.astype(float)
    out["beats"]["p_amp_uv"] = np.where(p_present, np.asarray(p_res["amp_uv"]), np.nan)

    # --- QRS: latido promediado por minuto ----------------------------------
    channels = _morph_channels(rec, int(lead))
    roles = _roles_for(rec, channels)
    minutes, drops = _minute_beats(rec, idx, channels, q, xf, minute_s, progress)
    baseline = _baseline_beat(minutes)
    # La posición de las derivaciones se infiere una sola vez con el latido de
    # base (`_resolve_roles` de qrs_morph): repetirla en cada minuto costaba el
    # triple. Si el llamador la declara (II, V1...) se usa tal cual.
    assumed = False
    assumed_notes: list[str] = []
    if baseline is not None and minutes:
        center0 = int(minutes[0]["center"])
        probe = qm.classify_beat(baseline, fs, center0, roles=roles or None,
                                 baseline=baseline)
        if not roles:
            roles = {int(k): str(v) for k, v in probe.roles.items()}
            assumed = bool(probe.assumed)
            assumed_notes = list(probe.notes)

    rows: list[dict] = []
    morphs: list[qm.MinuteQrs] = []
    aha_morphs: list[qm.MinuteQrs] = []     # criterios AHA de T128 (ramas, T166)
    aha_rules = _aha_rules()
    for item in minutes:
        m = item["minute"]
        beat, center = item["beat"], item["center"]
        on, off, _amp = FT.qrs_bounds_channel(np.asarray(beat[:, 0], dtype=np.float64),
                                              fs, center)
        sel = item["ids"]
        # PR por latido: inicio de la P del latido → inicio del QRS del minuto
        pr_vals = np.full(sel.size, np.nan)
        ok_p = p_present[sel]
        if ok_p.any():
            on_abs = idx[sel] + (on - center)
            pr = (on_abs - np.asarray(p_res["onset"])[sel]) * 1000.0 / fs
            valid = ok_p & np.isfinite(pr) & (pr >= PR_VALID_MS[0]) & (pr <= PR_VALID_MS[1])
            pr_vals[valid] = pr[valid]
        n_pe = int((p_eval[sel]).sum())
        n_pp = int((p_present[sel]).sum())
        pr_ok = pr_vals[np.isfinite(pr_vals)]
        pr_med = float(np.median(pr_ok)) if pr_ok.size else None
        morph = qm.classify_beat(np.asarray(beat, dtype=np.float64), fs, center,
                                 roles=roles or None, baseline=None,
                                 pr_ms=pr_med)
        if not morph.measurable:
            rows.append({"m": int(m), "eval": False, "why": "qrs_no_medible",
                         "n": item["n"], "p": None, "pr": None})
            continue
        p_minute = None
        if n_pe >= PWAVE_MIN_EVAL:
            p_minute = bool(n_pp >= 0.5 * n_pe)
        width = float(morph.width_ms)
        morphs.append(qm.MinuteQrs(
            minute=int(m), n_beats=int(item["n"]), rr_s=float(item["rr_s"]),
            hr_bpm=float(item["hr_bpm"]), width_ms=width, pattern=morph.pattern,
            confidence=float(morph.confidence), detail=morph.to_dict()))
        aha = qm.classify_beat(np.asarray(beat, dtype=np.float64), fs, center,
                               roles=roles or None, baseline=None, pr_ms=pr_med,
                               rules=aha_rules)
        if aha.measurable:
            aha_morphs.append(qm.MinuteQrs(
                minute=int(m), n_beats=int(item["n"]), rr_s=float(item["rr_s"]),
                hr_bpm=float(item["hr_bpm"]), width_ms=float(aha.width_ms),
                pattern=aha.pattern, confidence=float(aha.confidence),
                detail=aha.to_dict()))
        rows.append({
            "m": int(m),
            "t_s": round(m * minute_s, 1),
            "at": clock_label(rec.start_time, m * minute_s),
            "eval": True, "why": "",
            "n": int(item["n"]),
            "p": p_minute,
            "p_frac": (round(n_pp / n_pe, 3) if n_pe else None),
            "n_pe": n_pe, "n_pp": n_pp,
            "pr": (round(pr_med, 1) if pr_med is not None else None),
            "pr_lo": (round(float(pr_ok.min()), 1) if pr_ok.size else None),
            "pr_hi": (round(float(pr_ok.max()), 1) if pr_ok.size else None),
            "qrs": round(width, 1),
            "wide": bool(morph.wide),
            "pat": morph.pattern,
            "conf": round(float(morph.confidence), 2),
            "hr": round(float(item["hr_bpm"]), 1),
            "on_ms": round((on - center) * 1000.0 / fs, 1),
            "off_ms": round((off - center) * 1000.0 / fs, 1),
        })
        out["beats"]["pr_ms"][sel] = pr_vals
        out["beats"]["qrs_wide"][sel] = float(bool(morph.wide))

    # los minutos que no se midieron se informan con su motivo (no se omiten)
    measured = {r["m"] for r in rows}
    for m in sorted(m for m in drops if m not in measured):
        rows.append({"m": int(m), "t_s": round(m * minute_s, 1),
                     "at": clock_label(rec.start_time, m * minute_s),
                     "eval": False, "why": drops[m]})
    rows.sort(key=lambda r: r["m"])
    out["minutes"] = rows
    out["aha_minutes"] = aha_morphs
    out["pwave"] = _pwave_block(rows, p_res, n_min, rec)
    out["qrs_morph"] = _morph_block(rows, morphs, n_min, rec, assumed, assumed_notes)
    out["beat"] = _representative_beat(rows, minutes, rec, fs, p_res, idx,
                                       channels)
    return out


def _pwave_block(rows: list[dict], p_res: dict, n_min: int,
                 rec: Recording) -> dict:
    """Resumen de onda P y PR con la hora del mínimo, el mediano y el máximo."""
    ev = [r for r in rows if r.get("eval")]
    with_p = [r for r in ev if r.get("p") is not None]
    present = [r for r in with_p if r["p"]]
    pr_rows = [r for r in ev if r.get("pr") is not None]
    block: dict = {
        "n_minutes": int(n_min),
        "n_evaluable": len(ev),
        "n_no_evaluable": int(n_min - len(ev)),
        "evaluable_pct": round(100.0 * len(ev) / max(1, n_min), 1),
        "n_minutes_with_p": len(with_p),
        "n_present": len(present),
        "n_absent": len(with_p) - len(present),
        "p_present_pct": round(100.0 * len(present) / max(1, len(with_p)), 1),
        "n_minutes_pr": len(pr_rows),
        "clock": ("hora local" if isinstance(rec.start_time, datetime)
                  else "tiempo transcurrido"),
        "method": ("P por plantilla del minuto con control par/impar (pwave); "
                   "PR = inicio de la P al inicio del QRS del latido promediado"),
    }
    if pr_rows:
        prs = np.array([r["pr"] for r in pr_rows], dtype=np.float64)
        i_med = int(np.argmin(np.abs(prs - np.median(prs))))
        i_min = int(np.argmin(prs))
        i_max = int(np.argmax(prs))
        block.update({
            "pr_median_ms": round(float(np.median(prs)), 1),
            "pr_min_ms": round(float(prs[i_min]), 1),
            "pr_max_ms": round(float(prs[i_max]), 1),
            "pr_p5_ms": round(float(np.percentile(prs, 5)), 1),
            "pr_p95_ms": round(float(np.percentile(prs, 95)), 1),
            "pr_median_hour": pr_rows[i_med]["at"],
            "pr_min_hour": pr_rows[i_min]["at"],
            "pr_max_hour": pr_rows[i_max]["at"],
            "pr_median_t_s": pr_rows[i_med]["t_s"],
            "pr_min_t_s": pr_rows[i_min]["t_s"],
            "pr_max_t_s": pr_rows[i_max]["t_s"],
            "pr_minute_median": int(pr_rows[i_med]["m"]),
        })
    reasons: dict[str, int] = {}
    for r in rows:
        if not r.get("eval"):
            reasons[str(r.get("why") or "no_evaluable")] = \
                reasons.get(str(r.get("why") or "no_evaluable"), 0) + 1
    if reasons:
        block["no_evaluable_reasons"] = reasons
    block["p_summary"] = pwave_mod.p_summary(p_res)
    return block


def _morph_block(rows: list[dict], morphs: list, n_min: int,
                 rec: Recording, assumed: bool = False,
                 assumed_notes: list[str] | None = None) -> dict:
    """Resumen de duración y patrón del QRS y de su comportamiento en el tiempo."""
    ev = [r for r in rows if r.get("eval")]
    ep = qm.summarise_episode(morphs)
    wide = [r for r in ev if r.get("wide")]
    incomplete = [r for r in ev if r.get("qrs") is not None
                  and FT.QRS_INCOMPLETE_MS <= r["qrs"] < FT.QRS_WIDE_MS]
    widths = np.array([r["qrs"] for r in ev if r.get("qrs") is not None], dtype=float)
    pat_counts: dict[str, int] = {}
    for r in wide:
        pat_counts[r["pat"]] = pat_counts.get(r["pat"], 0) + 1
    missing: list[str] = []
    roles: dict[str, str] = {}
    for mq in morphs:
        det = mq.detail or {}
        for txt in det.get("missing") or []:
            if txt not in missing:
                missing.append(txt)
        if not roles:
            roles = dict(det.get("roles") or {})
    if assumed:
        missing.insert(0, "posición de derivaciones inferida por polaridad "
                          "(no declarada por el equipo)")
    missing.extend(n for n in (assumed_notes or []) if n not in missing)
    block = {
        "n_minutes": int(n_min),
        "n_evaluable": len(ev),
        "n_wide_minutes": len(wide),
        "n_incomplete_minutes": len(incomplete),
        "wide_pct_time": round(100.0 * len(wide) / max(1, len(ev)), 1),
        "incomplete_pct_time": round(100.0 * len(incomplete) / max(1, len(ev)), 1),
        "median_width_ms": (round(float(np.median(widths)), 1) if widths.size else None),
        "p95_width_ms": (round(float(np.percentile(widths, 95)), 1) if widths.size else None),
        "wide_threshold_ms": FT.QRS_WIDE_MS,
        "pattern": ep.get("pattern", "estrecho"),
        "pattern_name": ep.get("pattern_name", qm.PATTERN_NAMES["estrecho"]),
        "kind": ep.get("kind", "sin_datos"),
        "wide_fraction": ep.get("wide_fraction", 0.0),
        "consistency": ep.get("consistency", 0.0),
        "rate_dependent": bool(ep.get("rate_dependent", False)),
        "wide_hr": ep.get("wide_hr"),
        "narrow_hr": ep.get("narrow_hr"),
        "delta_bpm": ep.get("delta_bpm"),
        "pattern_share": {k: round(v / max(1, len(wide)), 3)
                          for k, v in sorted(pat_counts.items(), key=lambda t: -t[1])},
        "assumed_lead_position": assumed,
        "roles": roles,
        "missing": missing[:4],
        "method": ("duración por unión multicanal del latido promediado del minuto; "
                   "patrón con qrs_morph.classify_beat"),
    }
    block["persistent"] = block["kind"] == "br_persistente"
    block["intermittent"] = block["kind"] in ("br_intermitente",
                                              "br_frecuencia_dependiente",
                                              "morfologia_variable")
    block["leads"] = [str(rec.leads[c]) for c in range(rec.n_leads)]
    return block


def _representative_beat(rows: list[dict], minutes: list[dict], rec: Recording,
                         fs: float, p_res: dict, idx: np.ndarray,
                         channels: list[int]) -> dict:
    """Latido promediado del minuto más cercano al PR mediano (para la interfaz)."""
    by_minute = {m["minute"]: m for m in minutes}
    ev = [r for r in rows if r.get("eval")]
    if not ev or not by_minute:
        return {}
    target = None
    pr_rows = [r for r in ev if r.get("pr") is not None]
    if pr_rows:
        med = float(np.median([r["pr"] for r in pr_rows]))
        target = min(pr_rows, key=lambda r: abs(r["pr"] - med))
    else:
        widths = sorted(ev, key=lambda r: (r.get("wide"), r.get("qrs") or 0.0))
        target = widths[0]
    item = by_minute.get(target["m"])
    if item is None:
        return {}
    beat, center = item["beat"], item["center"]
    sel = item["ids"]
    p_peak = np.asarray(p_res["peak"])[sel]
    p_on = np.asarray(p_res["onset"])[sel]
    have = (p_peak >= 0) & (p_on >= 0)
    t0 = float(idx[sel[0]]) if sel.size else 0.0
    marker = {"p_onset_ms": None, "p_peak_ms": None,
              "qrs_onset_ms": target.get("on_ms"),
              "qrs_offset_ms": target.get("off_ms"),
              "pr_ms": target.get("pr")}
    if bool(have.any()):
        base = idx[sel][have].astype(np.float64)
        marker["p_onset_ms"] = float(np.median((p_on[have] - base) * 1000.0 / fs))
        marker["p_peak_ms"] = float(np.median((p_peak[have] - base) * 1000.0 / fs))
    names = [str(rec.leads[c]) for c in channels]
    payload = _beat_payload(np.asarray(beat, dtype=np.float64), int(center), fs,
                            names, marker)
    payload["minute"] = int(target["m"])
    payload["at"] = target.get("at")
    payload["n_beats"] = int(item["n"])
    payload["pr_ms"] = target.get("pr")
    payload["pattern"] = target.get("pat")
    payload["width_ms"] = target.get("qrs")
    payload["lead"] = names[0] if names else ""
    payload["t0_s"] = round(t0 / fs, 1)
    return payload


def apply_p_and_morphology(analysis: Analysis, lead: int | None = None) -> dict:
    """Recalcula P/PR/morfología de un análisis ya hecho (sin redetectar latidos)."""
    rec = analysis.recording
    lead_i = int(analysis.quality.get("lead", 0) if lead is None else lead)
    xf = preprocess(np.asarray(rec.samples[:, lead_i], dtype=np.float64), rec.fs,
                    "display")
    idx = np.array([b.idx for b in analysis.beats], dtype=np.int64)
    noises = np.array([b.features.get("noise", 0.0) or 0.0 for b in analysis.beats],
                      dtype=np.float64)
    block = p_and_qrs_morphology(rec, xf, idx, noises, lead=lead_i)
    analysis.summary["pwave"] = block["pwave"]
    analysis.summary["qrs_morph"] = block["qrs_morph"]
    analysis.summary["minute_p_qrs"] = block["minutes"]
    if block.get("beat"):
        analysis.summary["pwave"]["beat"] = block["beat"]
    for i, b in enumerate(analysis.beats):
        for key in ("p_present", "pr_ms", "p_amp_uv", "qrs_wide"):
            values = block["beats"][key]
            value = values[i] if i < values.size else np.nan
            b.features[key] = None if not np.isfinite(value) else float(value)
    return block


def _brd_por_latido(rec: Recording, xf: np.ndarray, lead: int, idx: np.ndarray,
                    cls: np.ndarray, widths_ms: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Forma de BRD de cada latido V ancho legible (regla 6 de T162).

    Solo se mide en los candidatos (V, QRS > 110 ms, legible): ``classify_beat`` sobre
    la ventana multicanal del latido. La posición de las derivaciones sale del nombre
    (V1, V5…) o, si no hay nombre, del latido estrecho mediano del registro (como en
    ``p_and_qrs_morphology``). Sin morfología determinada el latido queda sin BRD.
    """
    n = idx.size
    out = np.zeros(n, dtype=bool)
    cand = np.flatnonzero((cls == CLASSES.index("V")) & (widths_ms > fa_reglas.UMBRALES["qrs_ancho_ms"])
                          & ~bad)
    if cand.size == 0:
        return out
    fs = float(rec.fs)
    pre = int(round(MORPH_PRE_S * fs))
    post = int(round(MORPH_POST_S * fs))
    channels = _morph_channels(rec, int(lead))
    sigs = [np.asarray(xf, dtype=np.float64)]
    for ch in channels[1:]:
        sigs.append(preprocess(np.asarray(rec.samples[:, ch], dtype=np.float64), fs, "display"))
    inside = (idx >= pre) & (idx + post <= rec.n_samples)

    def ventana(j: int) -> np.ndarray:
        return np.column_stack([sg[idx[j] - pre: idx[j] + post] for sg in sigs])

    roles = _roles_for(rec, channels)
    narrow = np.flatnonzero((cls == CLASSES.index("N")) & (widths_ms <= fa_reglas.UMBRALES["qrs_ancho_ms"])
                            & ~bad & inside)
    base = None
    if narrow.size:
        pick = narrow[np.linspace(0, narrow.size - 1, min(narrow.size, 200)).astype(int)]
        base = np.median(np.stack([ventana(j) for j in pick]), axis=0)
    if not roles and base is not None:
        probe = qm.classify_beat(base, fs, pre, roles=None, baseline=base)
        roles = {int(k): str(v) for k, v in probe.roles.items()}
    for j in cand:
        if not inside[j]:
            continue
        morph = qm.classify_beat(ventana(int(j)), fs, pre, roles=roles or None, baseline=None)
        out[j] = morph.pattern in ("brd", "brd_incompleto")
    return out


def _fa_tratados_layer(rec: Recording, beats: list[Beat], idx: np.ndarray, widths: np.ndarray,
                       lead: int, xf: np.ndarray, af_segs: list, bad: np.ndarray,
                       amask) -> tuple[list[Event], dict]:
    """Reglas de FA de los tratados integradas en el análisis (T162).

    Aberrancia de Ashman (regla 6), FA preexcitada (regla 7) sobre los episodios que
    informa la rama, y flutter a revisión (T11) fuera de FA. Los eventos dentro de
    un tramo ilegible no se informan.
    """
    t = np.array([b.t for b in beats], dtype=float)
    cls_str = np.array([b.cls for b in beats], dtype="<U1")
    cls_idx = np.array([CLASSES.index(c) if c in CLASSES else -1 for c in cls_str], dtype=int)
    widths_ms = np.asarray(widths, dtype=float) * 1000.0
    brd = _brd_por_latido(rec, xf, lead, idx, cls_idx, widths_ms, bad)
    for b, v in zip(beats, brd):
        b.features["brd"] = float(v)
    fa_eps = [{"start_s": float(s.start_s), "end_s": float(s.end_s)} for s in af_segs]
    res = fa_reglas.analizar_fa(t, cls_str, rec.duration_s, unreadable=bad,
                                widths_ms=widths_ms, brd=brd, fa_episodes=fa_eps)
    at = atrial.analyze(np.asarray(rec.samples[:, lead], dtype=np.float64), float(rec.fs),
                        np.asarray(idx, dtype=np.int64))
    q_ok = at["summary"]["quality"] == "ok"
    events: list[Event] = []
    for al in res["alerts"]:
        if al["tipo"] in ("aberrancia_ashman", "fa_preexcitada"):
            events.append(Event(kind=al["tipo"], start_s=float(al["start_s"]),
                                end_s=float(al["end_s"]), severity="warn",
                                detail=al["texto"]))
    for fl in fa_reglas.flutter_eventos(at["windows"], fa_eps, quality_ok=q_ok):
        events.append(Event(kind="flutter", start_s=float(fl["start_s"]),
                            end_s=float(fl["end_s"]), severity="info",
                            detail=fa_reglas.TEXTOS["flutter_a_revision"]))
    if amask is not None:
        events = _drop_in_spans(events, amask.spans)
    summ = {
        "ashman": res["ashman"],
        "n_aberrancia_ashman": sum(e.kind == "aberrancia_ashman" for e in events),
        "n_fa_preexcitada": sum(e.kind == "fa_preexcitada" for e in events),
        "n_flutter_a_revision": sum(e.kind == "flutter" for e in events),
        "flutter_quality": at["summary"]["quality"],
        "n_brd_latidos": int(brd.sum()),
    }
    return events, summ


KIND_BASE = "ritmo_"   # prefijo de los eventos de ritmo de base (T165): ritmo_union, …


def _base_rhythm_layer(rec: Recording, beats: list[Beat], idx: np.ndarray,
                       widths: np.ndarray, xf: np.ndarray, af_segs: list,
                       bad: np.ndarray) -> tuple[list[Event], dict]:
    """Ritmos de la unión e idioventricular como tramos con inicio, fin y FC (T165).

    Reglas de ``base_rhythm_analysis`` (T123; CRITERIOS-TRATADOS §D): ventanas de 10 s
    sobre los latidos, con la P de T10 (presencia, evaluabilidad, PR y correlación).
    Solo se emiten los tramos de unión (40–60 y 60–100 lpm) e idioventricular
    (20–40 lpm); la sinusal ya la informa la rama de ritmo. Las ventanas con latidos
    ilegibles o artefacto (≥2 de 3 canales malos) quedan "no informado", y la FA
    (episodios de la rama) queda "sin regla".
    """
    fs = float(rec.fs)
    if len(beats) < 3:
        return [], {"nota": "pocos latidos para ritmos de base"}
    # secuencia de ritmo (T164): sin los Q que son onda T tomada por QRS
    keep = ritmo_mask(beats)
    seq = ritmo_beats(beats)
    idx = np.asarray(idx, dtype=np.int64)[keep]
    bad = np.asarray(bad, dtype=bool)[keep]
    t = np.array([b.t for b in seq], dtype=float)
    rr = np.array([b.rr_prev for b in seq], dtype=float)
    normal = np.array([b.cls == "N" for b in seq], dtype=bool)
    qrs_s = np.asarray(widths, dtype=float)[keep]
    pw = pwave_mod.detect_p_waves(np.asarray(xf, dtype=np.float64), fs, idx)
    present = np.asarray(pw["present"], dtype=bool)
    evaluable = np.asarray(pw["evaluable"], dtype=bool)
    onset = np.asarray(pw["onset"], dtype=float)
    # PR = inicio de P → pico R, menos medio QRS (mismo cálculo que scripts/eval_ritmos.py)
    pr_s = np.where(present & (onset >= 0), (idx - onset) / fs - qrs_s / 2.0, np.nan)
    corr = np.where(present, np.asarray(pw["corr"], dtype=float), np.nan)
    af_iv = [(float(s.start_s), float(s.end_s)) for s in af_segs]
    out = base_rhythm_analysis(t, rr, normal, qrs_s, present, evaluable, pr_s, corr,
                               duration_s=float(rec.duration_s), bad=bad, af_episodes=af_iv)
    events: list[Event] = []
    tramos: list[dict] = []
    for ev in out["events"]:
        if ev["kind"] not in BASE_EVENT_KINDS:
            continue
        a_s, b_s, fc = float(ev["start_s"]), float(ev["end_s"]), float(ev["hr_med"])
        events.append(Event(
            kind=KIND_BASE + ev["kind"], start_s=a_s, end_s=b_s, severity="info",
            detail=(f"{BASE_RHYTHM_NAMES[ev['kind']]}: FC mediana {fc:.0f} lpm "
                    f"· tira {a_s:.0f}-{b_s:.0f} s")))
        tramos.append({"tipo": ev["kind"], "start_s": round(a_s, 1), "end_s": round(b_s, 1),
                       "fc_mediana": round(fc, 1)})
    labels = Counter(w["label"] for w in out["windows"])
    summ = {
        "criterio": "CRITERIOS-TRATADOS §D (T123)",
        "tramos": tramos,
        "n_tramos": {k: sum(1 for x in tramos if x["tipo"] == k) for k in BASE_EVENT_KINDS},
        "ventanas": {k: int(v) for k, v in labels.items()},
        # NaN → None: un NaN no es igual a sí mismo y rompe la comparación de resúmenes
        "fc_hora": [{k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                     for k, v in h.items()} for h in out["hourly"]],
    }
    return events, summ


@functools.cache
def _criterios_carga():
    return cargar_criterios()


def _carga_layer(beats: list[Beat], duration_s: float, amask,
                 no_signal: list[dict]) -> dict:
    """Carga de ESV y EV por examen (T163; reglas de T120 en ``carga_ectopia``).

    Cuentan los latidos válidos: los Q (ilegibles, artefacto o abstenidos) no entran
    ni en el numerador ni en el denominador. Las horas cuentan solo tiempo analizable:
    se restan las ventanas ilegibles (≥2 de 3 canales malos) y los tramos sin señal.
    """
    no_an: list[tuple[float, float]] = []
    if amask is not None:
        no_an += [(float(s["start_s"]), float(s["end_s"])) for s in amask.spans]
    no_an += [(float(e["start_s"]), float(e["end_s"])) for e in no_signal]
    latidos = [(b.t, b.cls if b.cls in ("S", "V") else "N")
               for b in beats if b.cls != "Q"]
    crit = _criterios_carga()
    out = resumen_carga(latidos, duration_s, no_an, crit).as_dict()
    out["fuente"] = crit.source
    return out


# --- T166: criterios AHA de bloqueo de rama y preexcitación (T128; §H) -------
KIND_RAMA_PATRON = {"brd": "rama_brd", "brd_incompleto": "rama_brd_incompleto",
                    "bri": "rama_bri", "bri_incompleto": "rama_bri_incompleto",
                    "ivcd": "rama_tciv", "preexcitacion": "rama_preexcitacion"}
KIND_RAMA_FC = "rama_frecuencia_dependiente"
RAMA_NOMBRES = {
    "rama_brd": "bloqueo de rama derecha (AHA)",
    "rama_brd_incompleto": "bloqueo de rama derecha incompleto (QRS 110-119 ms)",
    "rama_bri": "bloqueo de rama izquierda (AHA)",
    "rama_bri_incompleto": "bloqueo de rama izquierda incompleto (QRS 110-119 ms)",
    "rama_tciv": "trastorno inespecífico de la conducción intraventricular "
                 "(QRS > 110 ms sin criterios de rama)",
    "rama_preexcitacion": "preexcitación probable (PR corto, onda delta, QRS > 120 ms)",
}


@functools.cache
def _aha_rules():
    return qm.load_aha_rules()


def _rachas_patron(minutos: list) -> list[list]:
    """Racha de minutos consecutivos con el mismo patrón: [patrón, 1.º, último, FC]."""
    out: list[list] = []
    for m in minutos:
        if out and out[-1][0] == m.pattern and m.minute == out[-1][2] + 1:
            out[-1][2] = m.minute
            out[-1][3].append(m.hr_bpm)
        else:
            out.append([m.pattern, m.minute, m.minute, [m.hr_bpm]])
    return out


def _patrones_v_layer(beats: list[Beat], wins, spans: list[dict]) -> tuple[list[Event], dict]:
    """R sobre T (C7), EV multiforme (C11) y parasistolia probable (C13) en analyze() (T167).

    Umbrales en ``rhythm`` (CRITERIOS-TRATADOS §C, T122). Un evento que toca un tramo
    con ≥2 de 3 canales malos (``spans``) no se informa: el R sobre T y la parasistolia
    se suprimen por ``ventricular_patterns``/``parasistolia_analyze`` y las V de un
    tramo ilegible quedan fuera de la plantilla multiforme. La fusión (F) sale del
    clasificador; hoy no lo emite (T131 C10), así que la parasistolia no se informa.
    """
    unread = [(float(s["start_s"]), float(s["end_s"])) for s in spans]
    events: list[Event] = []
    vp = rhythm.ventricular_patterns(beats, rhythm.VentricularCriteria(), unreadable=unread)
    n_rst = 0
    for e in vp["events"]:
        if e["kind"] != "r_sobre_t":
            continue
        n_rst += 1
        qt = float(beats[e["i0"] - 1].features.get("qt_ms") or 0.0)
        cpl = float(e["coupling_ms"])
        events.append(Event(kind="r_sobre_t", start_s=e["start_s"], end_s=e["end_s"],
                            severity="warn",
                            detail=f"R sobre T: acoplamiento {cpl:.0f} ms, QT previo "
                                   f"{qt:.0f} ms (índice {cpl / qt:.2f} < 1,0)"))
    summary: dict = {"r_sobre_t": n_rst, "no_evaluable": vp["not_evaluable"],
                     "suprimidos": vp["suppressed"]}
    t = np.array([b.t for b in beats], dtype=float)
    cls = np.array([b.cls for b in beats], dtype=object)
    if len(wins) != len(beats):
        summary["multiforme"] = {"nota": "ventanas y latidos no alineados"}
        summary["parasistolia"] = {"nota": "ventanas y latidos no alineados"}
        return events, summary
    v_mask = np.array([c == "V" and not rhythm.span_hit(tt, tt, unread)
                       for c, tt in zip(cls, t, strict=True)], dtype=bool)
    w = np.asarray(wins, dtype=float)
    pts_s = w.shape[1] / (FT.WIN_PRE + FT.WIN_POST)      # puntos por segundo de la ventana
    c0 = FT.WIN_PRE * pts_s
    a_col = int(round(c0 + rhythm.MULTIFORME_WIN_S[0] * pts_s))
    b_col = int(round(c0 + rhythm.MULTIFORME_WIN_S[1] * pts_s))
    tpl = rhythm.multiforme_templates(w[:, a_col:b_col], v_mask)
    summary["multiforme"] = {"plantillas": len(tpl), "latidos": [len(g) for g in tpl]}
    if len(tpl) >= 2:
        members = [i for g in tpl for i in g]
        events.append(Event(kind="multiforme", start_s=float(t[min(members)]),
                            end_s=float(t[max(members)]), severity="warn",
                            detail=f"EV multiforme: {len(tpl)} plantillas "
                                   f"({', '.join(str(len(g)) for g in tpl)} latidos), "
                                   f"correlación < {rhythm.MULTIFORME_CORR_MAX:.2f} entre ellas"))
    labels = np.full(len(beats), -1, dtype=int)
    v_idx = np.flatnonzero(cls == "V")
    if v_idx.size >= 3:
        labels[v_idx] = rhythm.cluster_windows(np.asarray(wins)[v_idx])
    rr_p = np.array([b.rr_prev for b in beats], dtype=float)
    para = rhythm.parasistolia_analyze(t, cls, rr_p, labels, unread)
    probables = [r for r in para["templates"] if r.get("verdict") == rhythm.PARA_PROBABLE]
    for r in probables:
        a, b = r["span_s"]
        events.append(Event(kind="parasistolia", start_s=float(a), end_s=float(b),
                            severity="warn",
                            detail=f"parasistolia probable: {r['n_V']} V, DE del acoplamiento "
                                   f"{r['de_coupling_ms']:.0f} ms, ciclo común "
                                   f"{r['period_ms']:.0f} ms, {r['n_fusion']} fusiones"))
    summary["parasistolia"] = {
        "plantillas_evaluadas": len(para["templates"]),
        "probable": len(probables),
        "candidatas": sum(1 for r in para["templates"]
                          if r.get("verdict") == rhythm.PARA_CANDIDATA),
        "suprimidas": para["suppressed"],
    }
    return events, summary


PACER_MIN_CHANNELS = 2   # espiga de marcapasos vista en ≥2 canales (concordancia multicanal)


def _pacer_layer(rec: Recording, beats: list[Beat], xf: np.ndarray,
                 amask, lead: int = 0) -> tuple[list[Event], dict]:
    """Latidos estimulados con fusión (J2) y fallos de sensado (J4) de CRITERIOS §J (T168).

    Reglas de ``signal/pacer.py`` (T130). Una espiga cuenta solo si la ven ≥2 canales;
    la ventana de sensado llega al fin de la T (pico + QT del latido, si hay QT); un
    evento en tramo con ≥2 de 3 canales malos no se informa. Detección limitada a 180 Hz.
    """
    from holterai.signal import pacer as pacer_mod  # import local: el bloque de imports ya es E402

    fs = float(rec.fs)
    if fs < 120.0 or rec.n_leads < PACER_MIN_CHANNELS or not beats:
        return [], {"nota": "frecuencia, canales o latidos insuficientes para marcapasos"}
    x = np.asarray(rec.samples, dtype=np.float64).T              # (canales, n)
    spikes_all, support = pacer_mod.detect_spikes(x, fs)
    keep = support >= min(PACER_MIN_CHANNELS, rec.n_leads)
    spikes = spikes_all[keep]
    # los bordes del QRS se miden sin espigas: la espiga, si no, entra en el inicio del QRS
    if spikes.size:
        xr = np.asarray(rec.samples[:, lead], dtype=np.float64).copy()
        for i in spikes:
            a_i, b_i = max(0, int(i) - 3), min(xr.size, int(i) + 4)
            xr[a_i:b_i] = np.linspace(xr[a_i], xr[b_i - 1], b_i - a_i)
        xf = preprocess(xr, fs, "display")
    # una sola conversión: qrs_bounds_channel copiaría el registro completo en cada latido
    xf = np.ascontiguousarray(xf, dtype=np.float64)
    peak = np.array([b.idx for b in beats], dtype=int)
    on = np.zeros(peak.size, dtype=int)
    off = np.zeros(peak.size, dtype=int)
    tend = np.zeros(peak.size, dtype=int)
    sens_t = int(round(pacer_mod.DEFAULT_RULES.sensing_t_ms / 1000.0 * fs))
    n_qt = 0
    for k, b in enumerate(beats):
        a, f, _ = FT.qrs_bounds_channel(xf, fs, int(b.idx))
        on[k], off[k] = a, f
        qt = float(b.features.get("qt_ms") or 0.0)
        n_qt += qt > 0
        tend[k] = int(b.idx + round(qt / 1000.0 * fs)) if qt > 0 else f + sens_t
    tend = np.maximum(tend, off)
    bad = None
    if amask is not None and getattr(amask, "bad", None) is not None and amask.bad.size:
        w = max(1, int(round(amask.win_s * fs)))
        bad = np.repeat(np.asarray(amask.bad, dtype=bool).T, w, axis=1)[:, :rec.n_samples]
        if bad.shape[1] < rec.n_samples:
            bad = np.pad(bad, ((0, 0), (0, rec.n_samples - bad.shape[1])))
    res = pacer_mod.analyze(peak, on, off, spikes, fs, bad=bad, t_end=tend)
    events: list[Event] = []
    for e in res.events:
        t = e.spike / fs
        if e.label == pacer_mod.LABEL_FUSION:
            events.append(Event(kind="pacing_fusion", start_s=t, end_s=t, severity="info",
                                detail=f"latido estimulado con QRS de {e.width_ms:.0f} ms "
                                       f"(fusión por ancho), espiga {e.dt_ms:.0f} ms antes"))
        elif e.label in (pacer_mod.LABEL_SENSADO, pacer_mod.LABEL_PSEUDOFUSION):
            what = ("pseudofusión: espiga sobre QRS estrecho propio"
                    if e.label == pacer_mod.LABEL_PSEUDOFUSION
                    else "espiga dentro del QRS o la T propios")
            events.append(Event(kind="pacing_sensing", start_s=t, end_s=t, severity="warn",
                                detail=f"fallo de sensado: {what}"))
    if amask is not None and events:
        events = _drop_in_spans(events, amask.spans)
    summ = pacer_mod.pacing_report(res)
    summ.update({
        "espigas_2_canales": int(spikes.size),
        "espigas_1_canal": int(spikes_all.size - spikes.size),
        "fusion": sum(1 for e in events if e.kind == "pacing_fusion"),
        "fallo_sensado": sum(1 for e in events if e.kind == "pacing_sensing"),
        "ventana_T": f"pico + QT en {n_qt} de {len(beats)} latidos (resto: fin QRS + 80 ms)",
    })
    return events, summ


def _rama_layer(aha_minutes: list, duration_s: float,
                spans: list[dict]) -> tuple[list[Event], dict]:
    """Bloqueos de rama, TCIV, preexcitación y BR dependiente de FC (T128 en analyze).

    Cada minuto medido (latido promediado de 3 canales) ya viene clasificado con las
    reglas AHA (`qm.classify_beat(rules=)`). Un minuto que toca un tramo con ≥2 de 3
    canales malos (`spans`) no cuenta: ni para los eventos ni para la estadística. Un
    evento es una racha de minutos consecutivos con el mismo patrón. La dependencia de
    la FC (fase 3 o 4) sale de `qm.rate_dependence`, con la FC de aparición. Hemibloqueos
    y eje no se evalúan con 3 canales.
    """
    def _toca(m) -> bool:
        a, b = m.minute * MINUTE_S, (m.minute + 1) * MINUTE_S
        return any(min(b, float(s["end_s"])) - max(a, float(s["start_s"])) > 0 for s in spans)

    minutos = sorted((m for m in aha_minutes if not _toca(m)), key=lambda m: m.minute)
    ep = qm.summarise_episode(minutos)
    rate = qm.rate_dependence(minutos)
    events: list[Event] = []
    tramos: list[dict] = []
    for pat, m0, m1, hrs in _rachas_patron(minutos):
        kind = KIND_RAMA_PATRON.get(pat)
        if kind is None:
            continue
        a_s = float(m0 * MINUTE_S)
        b_s = float(min(duration_s, (m1 + 1) * MINUTE_S))
        fc = float(np.median(hrs))
        n_min = m1 - m0 + 1
        events.append(Event(kind=kind, start_s=a_s, end_s=b_s, severity="warn",
                            detail=f"{RAMA_NOMBRES[kind]}: {n_min} min, FC mediana "
                                   f"{fc:.0f} lpm · tira {a_s:.0f}-{b_s:.0f} s"))
        tramos.append({"tipo": kind, "start_s": round(a_s, 1), "end_s": round(b_s, 1),
                       "minutos": n_min, "fc_mediana": round(fc, 1)})
    if ep.get("kind") == "br_frecuencia_dependiente":
        wide = [m for m in minutos if m.width_ms >= FT.QRS_WIDE_MS]
        a_s = float(wide[0].minute * MINUTE_S)
        b_s = float(min(duration_s, (wide[-1].minute + 1) * MINUTE_S))
        bpm = ep.get("appearance_bpm")
        if ep.get("phase") == "fase4":
            cuando = "aparece por debajo"
        else:
            cuando = "aparece por encima"
        cifra = f"de {bpm:.0f} lpm" if bpm is not None else "(corte no concluyente)"
        events.append(Event(kind=KIND_RAMA_FC, start_s=a_s, end_s=b_s, severity="warn",
                            detail=f"BR dependiente de la frecuencia (fase "
                                   f"{ep.get('phase', 'fase3')[-1]}): ancho {cuando} {cifra}"))
        tramos.append({"tipo": KIND_RAMA_FC, "start_s": round(a_s, 1),
                       "end_s": round(b_s, 1), "fase": ep.get("phase"),
                       "appearance_bpm": bpm})
    block = {
        "criterio": "CRITERIOS-TRATADOS §H (T128, reglas AHA 2009)",
        "umbrales": "configs/criteria/aha_ramas.yaml",
        "n_minutos": len(minutos),
        "kind": ep.get("kind", "sin_datos") if minutos else "sin_datos",
        "pattern": ep.get("pattern"),
        "pattern_name": ep.get("pattern_name"),
        "wide_fraction": ep.get("wide_fraction", 0.0),
        "consistency": ep.get("consistency", 0.0),
        "persistent": ep.get("kind") == "br_persistente",
        "rate_dependent": bool(rate.get("dependent", False)),
        "phase": ep.get("phase"),
        "appearance_bpm": ep.get("appearance_bpm"),
        "preexcitation_intermittent": bool(ep.get("preexcitation_intermittent", False)),
        "rate_dependence": rate,
        "n_tramos": {k: sum(1 for t in tramos if t["tipo"] == k)
                     for k in (*KIND_RAMA_PATRON.values(), KIND_RAMA_FC)},
        "tramos": tramos,
        "no_evaluado": "hemibloqueos y eje (necesitan 12 derivaciones)",
    }
    return events, block


def analyze(rec: Recording, opts: Options | None = None, progress=None) -> Analysis:
    opts = opts or Options()
    t0 = time.time()
    _p(progress, "Preparando", 0.01)

    amask = _artifact_mask(rec) if opts.reject_artifacts else None
    if amask is not None and amask.unreadable_s > 0:
        log.info("calidad: %.1f %% del registro ilegible por artefactos (%d tramos)",
                 100.0 * float(amask.unreadable.mean()), len(amask.spans))

    if 0 <= int(opts.force_lead) < rec.n_leads:
        lead = int(opts.force_lead)
        log.info("derivación forzada por el usuario: %d", lead)
    else:
        lead = choose_analysis_lead(rec)
        alt = _lead_loses_beats(rec, lead)
        if alt is not None:
            lead = alt
    # un único filtrado del canal de análisis, compartido por detección y PAC
    xf_full = preprocess(np.asarray(rec.samples[:, lead], dtype=np.float64),
                         rec.fs, "display")
    idx, wins, widths, amps, noises = detect_beats(rec, opts, progress, lead=lead,
                                                   xf=xf_full)
    # En tramos ruidosos el detector principal pierde QRS de bajo voltaje.
    # Una segunda pasada por correlación con la plantilla recupera parte.
    noisy_frac = float(np.mean(noises > 0.08)) if len(noises) else 0.0
    rate = idx.size / max(1.0, rec.duration_s / 60.0)
    extra = np.zeros(0, dtype=int)
    if rate < 45.0:
        extra = _extend_qrs_multilead(rec, xf_full, idx)
        if extra.size < 5:
            extra = _extend_qrs_by_template(xf_full, idx, rec.fs)
    elif noisy_frac > 0.15:
        extra = _extend_qrs_by_template(xf_full, idx, rec.fs)
    if extra.size:
        merged = np.sort(np.concatenate([idx, extra]).astype(np.int64))
        idx, wins, widths, amps, noises = _rebuild_beat_arrays(xf_full, rec.fs, merged)
    if idx.size > opts.max_beats:
        sel = np.linspace(0, idx.size - 1, opts.max_beats).astype(int)
        idx, wins, widths, amps, noises = idx[sel], wins[sel], widths[sel], amps[sel], noises[sel]

    cls, conf, rr_prev, probs_primary = classify_beats(rec, idx, wins, widths, amps, progress)

    # fusión multicanal + veto cruzado de artefactos
    veto = np.zeros(len(idx), dtype=bool)
    agreement = np.ones(len(idx), dtype=np.float32)
    fused = fuse_multilead(rec, idx, widths, amps, noises, lead, rr_prev,
                           probs_primary, xf_full, progress)
    if fused is not None:
        cls, conf, veto, probs, agreement = fused
    else:
        probs = probs_primary
    # Calibración (sección 5 del protocolo): temperatura ajustada por NLL en
    # MIT-BIH DS1 y corrección de prevalencia por logits. Va antes de la PAC y
    # de la abstención, que deben ver confianzas calibradas.
    calib = get_calibration() if opts.calibration else Calibration(enabled=False)
    if len(idx) and calib.enabled:
        probs = calib.apply(probs).astype(np.float32)
        cls = probs.argmax(axis=1)
        conf = probs.max(axis=1)
    else:
        probs = np.asarray(probs, dtype=np.float32)
    cls[veto] = CLASSES.index("Q")
    conf[veto] = 0.0

    cls, conf, pac_prob, qrs_corr, svpb_dec = refine_supraventricular(
        rec, idx, cls, conf, widths, noises, lead, progress, xf=xf_full,
        probs=probs)
    svpb_cfg = get_svpb_config()
    sv_keep = np.asarray(svpb_dec["is_s"], dtype=bool) if svpb_dec is not None else None
    cls, n_suppressed = _suppress_normal_lookalikes(wins, cls, rr_prev, rec.fs,
                                                    keep=sv_keep)
    # Abstención: una S/V/F dudosa, con margen pequeño o en ruido va a Q en vez
    # de contarse. La regla histórica de confianza < 0,32 se mantiene siempre.
    abstain = (conf < 0.32) & np.isin(cls, [1, 2])
    abstain_by_reason: dict[str, int] = {}
    if calib.enabled:
        trig = calib.abstain_triggers(probs, cls=cls, noise=noises,
                                      widths=widths, corr=qrs_corr)
        for name, mask in trig.items():
            abstain_by_reason[name] = int(mask.sum())
            abstain |= mask
    cls[abstain] = 4
    artifact = _mark_artifact_beats(noises, agreement, cls)

    # --- artefactos: una ventana ilegible no describe al paciente -----------
    unreadable = _unreadable_times(amask, idx, rec.fs)
    if unreadable.any():
        cls = np.array(cls, dtype=int)
        conf = np.array(conf, dtype=float)
        cls[unreadable] = CLASSES.index("Q")
        conf[unreadable] = 0.0
        log.info("artefactos: %d latidos (%.1f %%) pasan a Q por ventana ilegible",
                 int(unreadable.sum()), 100.0 * float(unreadable.mean()))

    # --- T173: ECGFounder ajustado (T158) para PVC/PAC/FA --------------------
    t158_af = None
    t158_summ: dict | None = None
    if str(opts.ecgf_model).lower() == "t158" and len(idx):
        try:
            cls, t158_af, t158_summ = t158_layer(rec, idx, cls, opts, progress)
            if unreadable.any():
                cls[unreadable] = CLASSES.index("Q")
        except Exception as exc:  # sin modelo o sin torch: sigue el detector actual
            log.warning("ECGFounder T158 no disponible: %s", exc)
            t158_summ = {"error": str(exc)}

    fs = rec.fs
    st_crit = opts.st_criteria()
    normal = np.array([CLASSES[int(c)] == "N" for c in cls], dtype=bool) if len(cls) else np.zeros(0, bool)
    st_values = np.full(len(idx), np.nan, dtype=np.float64)
    if bool(normal.any()):
        st_values[normal] = ST.st_level_batch(xf_full, fs, idx[normal], st_crit.j_ms)
    n_beats = len(idx)
    sv_score = (svpb_dec["score"] if svpb_dec is not None
                else np.zeros(n_beats, dtype=np.float64))
    sv_ratio = (svpb_dec["features"][:, svpb.IDX["rr_ratio"]] if svpb_dec is not None
                else np.ones(n_beats, dtype=np.float64))
    p_state = (svpb_dec["p_feat"]["p_state"] if svpb_dec is not None
               else np.full(n_beats, svpb.P_UNKNOWN, dtype=np.int8))
    beats: list[Beat] = []
    for i in range(len(idx)):
        beats.append(Beat(
            idx=int(idx[i]), t=float(idx[i] / fs),
            cls=CLASSES[int(cls[i])], conf=float(conf[i]),
            rr_prev=float(rr_prev[i]),
            rr_next=float(rr_prev[i + 1]) if i + 1 < len(rr_prev) else 0.0,
            features={"pac_prob": float(pac_prob[i]),
                      "svpb_score": round(float(sv_score[i]), 3),
                      "rr_ratio_local": round(float(sv_ratio[i]), 3),
                      "p_state": int(p_state[i]),
                      "width_ms": round(float(widths[i]) * 1000.0, 1),
                      "agreement": float(agreement[i]),
                      "noise": float(noises[i]),
                      "st_uv": float(st_values[i]) if bool(normal[i]) else None,
                      "qt_ms": float(FT.qt_interval_ms(xf_full, fs, int(idx[i]),
                                                       float(rr_prev[i])))
                      if CLASSES[int(cls[i])] == "N" else None,
                      "st_slope_uv_s": float(FT.st_slope_uv_s(xf_full, fs, int(idx[i]),
                                                              st_crit.j_ms))
                      if CLASSES[int(cls[i])] == "N" else None,
                      "vetoed": bool(veto[i]),
                      "unreadable": bool(unreadable[i]),
                      "artifact": bool(artifact[i])},
        ))

    af_probs, af_segs = rhythm_analysis(beats, opts, progress, unreadable=unreadable,
                                        ext_af=t158_af)
    for i, b in enumerate(beats):
        b.features["af_prob"] = float(af_probs[i])

    # aviso de polaridad por canal: se informa, nunca se corrige solo
    try:
        polarity = qrs_polarity(rec.samples, rec.fs, idx, rec.leads)
    except Exception as exc:
        log.warning("no se pudo medir la polaridad del QRS: %s", exc)
        polarity = []
    try:
        no_signal = no_signal_events(np.asarray(rec.samples), rec.fs)
    except Exception as exc:
        log.warning("no se pudo evaluar la pérdida de señal: %s", exc)
        no_signal = []

    quality = {
        "lead": int(lead),
        "lead_name": rec.leads[lead] if lead < len(rec.leads) else "",
        "noise_mean": round(float(np.mean(noises)) if len(noises) else 0.0, 4),
        "noisy_beats_pct": round(float(100.0 * np.mean(noises > 0.08)) if len(noises) else 0.0, 1),
        "n_leads": rec.n_leads,
        "fs": rec.fs,
        "duration_h": round(rec.duration_s / 3600.0, 2),
        "polarity": polarity,
        "polarity_warnings": [p["name"] for p in polarity if p["invert_suggested"]],
        "no_signal_s": round(sum(e["end_s"] - e["start_s"] for e in no_signal), 1),
        "n_unreadable_beats": int(unreadable.sum()),
        "unreadable_beats_pct": round(100.0 * float(unreadable.mean())
                                      if len(unreadable) else 0.0, 1),
    }
    if amask is not None:
        quality.update(amask.as_quality())
        _lead_quality_note(lead, amask, rec, quality)

    events = build_events(beats, af_segs, opts, qrs_corr, rec.duration_s,
                          xf_full, rec.fs, no_signal=no_signal,
                          samples=rec.samples,
                          p_waves=_p_wave_times(svpb_dec, rec.fs),
                          unreadable=amask.spans if amask is not None else None)
    st_ev, st_block = st_events(beats, st_crit, lead=lead,
                                lead_name=quality["lead_name"])
    events.extend(st_ev)
    if amask is not None:
        # ni pausas, ni TV, ni FA, ni bloqueos en medio del ruido
        kept = _drop_in_spans(events, amask.spans)
        if len(kept) != len(events):
            log.info("artefactos: %d eventos descartados por caer en tramos ilegibles",
                     len(events) - len(kept))
        events = kept

    # --- onda P, PR y morfología del QRS (T33) --------------------------
    morph_block: dict = {}
    if opts.morphology:
        try:
            _p(progress, "Onda P y morfología del QRS", 0.79)
            morph_block = p_and_qrs_morphology(rec, xf_full, idx, noises, lead=lead,
                                               progress=progress)
            for i, b in enumerate(beats):
                for key in ("p_present", "pr_ms", "p_amp_uv", "qrs_wide"):
                    values = morph_block["beats"][key]
                    value = values[i] if i < values.size else np.nan
                    b.features[key] = None if not np.isfinite(value) else float(value)
        except Exception as exc:                       # no debe tumbar el análisis
            log.warning("onda P y morfología del QRS no disponibles: %s", exc)
            morph_block = {"pwave": {"error": str(exc)}, "qrs_morph": {"error": str(exc)},
                           "minutes": [], "beats": {}}
    summary = summarize(beats, events, af_segs, rec.duration_s, quality)
    if morph_block:
        summary["pwave"] = morph_block.get("pwave") or {}
        summary["qrs_morph"] = morph_block.get("qrs_morph") or {}
        if morph_block.get("beat"):
            summary["pwave"]["beat"] = morph_block["beat"]
        summary["minute_p_qrs"] = morph_block.get("minutes") or []
    summary["st"] = st_block
    summary["n_st_episodes"] = st_block.get("n_episodes", 0)
    summary["st_depressions"] = st_block.get("n_depressions", 0)
    summary["st_elevations"] = st_block.get("n_elevations", 0)
    if st_block.get("median_uv") is not None:
        summary["st_median_uv"] = st_block["median_uv"]
    st_slopes = [float(b.features["st_slope_uv_s"]) for b in beats
                 if b.features.get("st_slope_uv_s") is not None]
    if st_slopes:
        summary["st_slope_median_uv_s"] = round(float(np.median(st_slopes)), 1)
    qt_vals = [float(b.features["qt_ms"]) for b in beats
               if b.features.get("qt_ms")]
    if qt_vals:
        summary["qt_median_ms"] = round(float(np.median(qt_vals)), 1)
        summary["qtc_median_ms"] = round(float(np.median([
            b.features["qt_ms"] / np.sqrt(max(b.rr_prev, 0.2))
            for b in beats if b.features.get("qt_ms")
        ])), 1)
        summary["qtc_fridericia_ms"] = round(float(np.median([
            b.features["qt_ms"] / (max(b.rr_prev, 0.2) ** (1.0 / 3.0))
            for b in beats if b.features.get("qt_ms")
        ])), 1)
    # --- QTc por fórmula (exploratorio) --------------------------------
    af_windows = [(float(s.start_s), float(s.end_s)) for s in af_segs]
    qt_marks = qt_mod.af_mask([b.t for b in beats], af_windows)
    qt_samples = [
        qt_mod.QtcSample(t=float(b.t), qt_ms=float(b.features["qt_ms"]),
                         rr_s=float(b.rr_prev), af=bool(qt_marks[i]))
        for i, b in enumerate(beats) if b.features.get("qt_ms")
    ]
    by_formula = qt_mod.qtc_all_formulas(qt_samples)
    selected = qt_mod.normalize_formula(opts.qtc_formula)
    summary["qtc_formula"] = selected
    summary["qtc_label"] = qt_mod.formula_label(selected)
    summary["qtc"] = by_formula[selected]
    summary["qtc_by_formula"] = by_formula
    try:
        # Ventanas día/noche configurables (criterios): el reparto por hora
        # local vive en hrv.split_day_night.
        dn = hrv_mod.day_night_hrv(beats, rec.start_time)
        summary["hrv_windows"] = {k: v for k, v in dn.items()
                                  if k not in ("day", "night")}
        summary["hrv_day"] = dn["day"]
        summary["hrv_night"] = dn["night"]
    except Exception as exc:
        log.debug("HRV día/noche no disponible: %s", exc)
    summary.update(_af_response(beats, af_segs))
    summary["bbb_count"] = int(np.sum(widths >= 0.12))
    summary["bbb_pct"] = round(100.0 * summary["bbb_count"] / max(1, len(beats)), 2)
    summary["sv_isolated"] = sum(
        1 for i, b in enumerate(beats) if b.cls == "S"
        and (i == 0 or beats[i - 1].cls != "S")
        and (i + 1 == len(beats) or beats[i + 1].cls != "S")
    )
    summary["n_artifact"] = int(artifact.sum())
    summary["n_suppressed_s"] = int(n_suppressed)
    summary["n_abstained"] = int(abstain.sum())
    if abstain_by_reason:
        summary["abstained_by_reason"] = abstain_by_reason
    if calib.enabled:
        summary["calibration"] = {
            "temperature": round(float(calib.temperature), 3),
            "target_prior": [round(float(v), 4) for v in calib.target_prior],
            "note": "temperatura (NLL en MIT-BIH DS1) + prevalencia por logits",
        }
    if svpb_dec is not None:
        # auditoría de la decisión S: cuántos latidos pasaron cada compuerta y
        # con qué puntaje, para revisar el umbral sin volver a correr todo
        summary["svpb"] = svpb.summary(svpb_dec, [b.cls for b in beats],
                                       svpb_dec.get("p_feat"), svpb_cfg)
        summary["svpb"]["s_per_hour"] = round(
            summary["counts"].get("S", 0) / max(1e-6, rec.duration_s / 3600.0), 1)

    # --- ectopia ventricular detallada y resumen horario ---------------
    counts = summary.get("counts", {})
    dur_h = max(1e-6, rec.duration_s / 3600.0)
    v_idx = np.flatnonzero(cls == CLASSES.index("V")) if len(cls) else np.zeros(0, int)
    try:
        morph = count_pvc_morphologies(wins, v_idx)
    except Exception as exc:
        log.warning("no se pudieron agrupar morfologías de PVC: %s", exc)
        morph = 1 if v_idx.size else 0
    vev = ventricular_events(beats)
    pvc = int(counts.get("V", 0))
    summary["pvc_morphologies"] = int(morph)
    summary["pvc_per_hour"] = round(pvc / dur_h, 1)
    summary["ventricular"] = vev
    summary["supraventricular"] = supraventricular_events(beats)
    sv_run: list[Beat] = []
    for b in beats + [None]:  # type: ignore[list-item]
        if b is not None and b.cls == "S":
            sv_run.append(b)
        elif len(sv_run) >= 3:
            if amask is None or not artifact_mod.inside_spans(
                    sv_run[0].t, sv_run[-1].t, amask.spans):
                events.append(Event(kind="svt", start_s=sv_run[0].t, end_s=sv_run[-1].t,
                                    severity="warn", detail=f"{len(sv_run)} latidos SV"))
            sv_run = []
        else:
            sv_run = []
    # --- T161: salvas S por criterio temporal; TA/TAM/errante (T124) en eventos
    try:
        p_feat = (svpb_dec or {}).get("p_feat") or {}
        p_res = (svpb_dec or {}).get("p_res") or {}
        sal = s_salva_layer(
            idx / fs, [CLASSES[int(c)] for c in cls], width_s=widths,
            p_state=p_feat.get("p_state"), p_corr=p_feat.get("p_corr"),
            pr_ms=p_feat.get("pr_ms"), p_peak=p_res.get("peak"), fs=fs, x=xf_full,
            pr_sinus_ms=p_feat.get("pr_ref_ms"),
            bad=np.asarray(unreadable, dtype=bool) | np.asarray(artifact, dtype=bool))
        summary["salvas"] = sal["summary"]
        for a_s, b_s, kind, nb, rate in sal["events"]:
            events.append(Event(kind="svt", start_s=a_s, end_s=b_s, severity="info",
                                detail=f"salva {kind}: {nb} latidos, {rate:.0f} lpm"))
    except Exception as exc:  # la capa de salvas no debe tumbar el análisis
        log.warning("salvas S no disponibles: %s", exc)
        summary["salvas"] = {"error": str(exc)}
    # --- T162: reglas de FA de los tratados (Ashman, FA preexcitada, flutter a revisión)
    try:
        fa_ev, fa_summ = _fa_tratados_layer(rec, beats, idx, widths, lead, xf_full, af_segs,
                                            np.asarray(unreadable, dtype=bool)
                                            | np.asarray(artifact, dtype=bool), amask)
        events.extend(fa_ev)
        summary["fa_tratados"] = fa_summ
    except Exception as exc:  # las reglas de FA no deben tumbar el análisis
        log.warning("reglas de FA de los tratados no disponibles: %s", exc)
        summary["fa_tratados"] = {"error": str(exc)}
    # --- T163: carga de ESV y EV (CRITERIOS-TRATADOS §B) -------------------
    try:
        summary["carga"] = _carga_layer(beats, rec.duration_s, amask, no_signal)
    except Exception as exc:  # la carga no debe tumbar el análisis
        log.warning("carga de ESV y EV no disponible: %s", exc)
        summary["carga"] = {"error": str(exc)}
    # --- T165: ritmos de la unión e idioventricular (CRITERIOS-TRATADOS §D) ---
    try:
        base_ev, summary["ritmo_base"] = _base_rhythm_layer(
            rec, beats, idx, widths, xf_full, af_segs,
            np.asarray(unreadable, dtype=bool) | np.asarray(artifact, dtype=bool))
        if amask is not None:
            base_ev = _drop_in_spans(base_ev, amask.spans)
        events.extend(base_ev)
    except Exception as exc:  # los ritmos de base no deben tumbar el análisis
        log.warning("ritmos de base no disponibles: %s", exc)
        summary["ritmo_base"] = {"error": str(exc)}
    # --- T166: bloqueos de rama, TCIV, preexcitación y BR dependiente de FC (§H) ---
    try:
        rama_ev, summary["ramas"] = _rama_layer(
            morph_block.get("aha_minutes") or [], rec.duration_s,
            amask.spans if amask is not None else [])
        events.extend(rama_ev)
    except Exception as exc:  # la capa de ramas no debe tumbar el análisis
        log.warning("bloqueos de rama (AHA) no disponibles: %s", exc)
        summary["ramas"] = {"error": str(exc)}
    # --- T167: R sobre T (C7), EV multiforme (C11) y parasistolia probable (C13) ---
    try:
        pv_ev, summary["patrones_v"] = _patrones_v_layer(
            beats, wins, amask.spans if amask is not None else [])
        if amask is not None:
            pv_ev = _drop_in_spans(pv_ev, amask.spans)
        events.extend(pv_ev)
    except Exception as exc:  # los patrones V no deben tumbar el análisis
        log.warning("patrones ventriculares (T167) no disponibles: %s", exc)
        summary["patrones_v"] = {"error": str(exc)}
    # --- T168: marcapasos, fusión (J2) y fallo de sensado (J4) de CRITERIOS §J ---
    if opts.pacing_mode != "off":
        try:
            pac_ev, summary["marcapasos_reglas"] = _pacer_layer(rec, beats, xf_full, amask,
                                                                  lead)
            events.extend(pac_ev)
        except Exception as exc:  # el marcapasos por reglas no debe tumbar el análisis
            log.warning("marcapasos por reglas (T168) no disponible: %s", exc)
            summary["marcapasos_reglas"] = {"error": str(exc)}
    events.sort(key=lambda e: e.start_s)
    summary["lown"] = lown_grade(pvc, pvc / dur_h, morph, vev["couplets"],
                                 vev["triplets"], vev["r_on_t"])
    summary["hourly"] = hourly_summary(beats, af_probs, rec.duration_s, opts.pause_s)

    if t158_summ is not None:
        summary["ecgf_t158"] = t158_summ
    if opts.use_ecgfounder:
        try:
            summary["ecgfounder"] = ecgfounder_layer(rec, opts, progress)
        except Exception as exc:
            log.warning("ECGFounder no disponible: %s", exc)
            summary["ecgfounder"] = {"error": str(exc)}

    # --- marcapasos / resincronización --------------------------------
    if opts.pacing_mode != "off":
        try:
            probes = []
            for l in range(min(3, rec.n_leads)):
                probes.append(pacing.detect_spikes(
                    np.asarray(rec.samples[:, l], dtype=np.float64), fs))
            s_idx, s_amp = pacing.merge_spikes(probes, fs)
            sp = {"idx": s_idx, "amps": s_amp, "found": s_idx.size > 0,
                  "noise": max((p.get("noise", 0.0) for p in probes), default=0.0)}
            wide_frac = float(np.mean(widths > 0.12)) if len(widths) else 0.0
            v1n = _v1_negative(rec, idx, fs)
            info = pacing.summarize(sp, beats, fs, wide_frac, v1n,
                                    qrs_amp=float(np.median(amps)) if len(amps) else 0.0)
            info["mode"] = opts.pacing_mode
            info["crt_mode"] = opts.crt_mode
            info["crt_v1"] = bool(v1n)
            info["qrs_median_ms"] = round(float(np.median(widths)) * 1000, 0) if len(widths) else None
            info["qrs_wide_frac"] = round(wide_frac, 2)
            if opts.pacing_mode == "on" and not info.get("spikes_detected"):
                info["paced_rhythm_suspected"] = bool(wide_frac > 0.4)
                info["note"] = ("modo marcapasos activado: espigas no visibles; "
                                "se analiza la morfología del ritmo")
            if opts.crt_mode == "on":
                info["crt_suspected"] = True
                qms = info.get("qrs_median_ms")
                info["note"] = ("modo resincronizador activado: QRS "
                                + (f"{qms:.0f} ms" if qms else "—")
                                + f", V1 {'negativo' if v1n else 'no negativo'}")
            summary["pacing"] = info
            if info.get("capture_failures") and (
                    amask is None or not artifact_mod.inside_spans(
                        info["capture_failure_samples"][0] / fs,
                        info["capture_failure_samples"][0] / fs, amask.spans)):
                first = info["capture_failure_samples"][0] / fs
                events.append(Event(
                    kind="pacing_failure", start_s=float(first), end_s=float(first),
                    severity="critical",
                    detail=f"{info['capture_failures']} espigas sin captura"))
                events.sort(key=lambda e: e.start_s)
            _p(progress, "Marcapasos", 0.78)
        except Exception as exc:
            log.warning("módulo de marcapasos no disponible: %s", exc)
            summary["pacing"] = {"error": str(exc)}

    # --- conducción AV: PR y bloqueos (T24) ---------------------------
    if opts.av_mode != "off":
        try:
            av_events, av_info = av_conduction_analysis(
                rec, beats, idx, xf_full, lead, opts, af_segs)
            if av_info:
                summary["av_conduction"] = av_info
            if av_events:
                events.extend(av_events)
                events.sort(key=lambda e: e.start_s)
            _p(progress, "Conducción AV", 0.85)
        except Exception as exc:
            log.warning("módulo de conducción AV no disponible: %s", exc)
            summary["av_conduction"] = {"error": str(exc)}
    model_info = {
        "beat_model": "CNN 1D (MIT-BIH, 104k latidos)",
        "beat_calibration": (f"temperatura T={calib.temperature:.2f} + prevalencia "
                             "N 0,985 / S 0,008 / V 0,007" if calib.enabled
                             else "desactivada"),
        "rhythm_model": "LogReg RR (AFDB, AUC 0.96)",
        "pac_model": "LogReg PAC 14 features (MIT-BIH + SVDB)",
    }
    _p(progress, "Listo", 1.0)
    return Analysis(
        recording=rec, beats=beats, events=events, rhythm=af_segs,
        summary=summary, quality=quality, model_info=model_info,
        elapsed_s=time.time() - t0,
    )
