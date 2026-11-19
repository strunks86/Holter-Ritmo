"""Análisis de ritmo: características de la serie RR y detección de eventos.

Incluye el detector de fibrilación auricular por episodios (T23): ventanas
deslizantes de 30-60 latidos con irregularidad del RR (entropía muestral,
COSEn, AFEvidence, RMSSD normalizado y dispersión de Poincaré), apoyo de la
ausencia de onda P (``signal/pwave.py``, T10) y de la actividad f del residuo
del QRST (``signal/atrial.py``, T11), exclusión de ectopia frecuente y de los
tramos ilegibles (``signal/artifact.py``, T21) y agregado en episodios de
>= 30 s con carga (%) y episodio más largo.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np

from holterai.legacy.app.signal.leads import map_leads

from .features import noise_batch
from .filters import fir_bandpass, filtfilt, moving_average
from .qrs import detect_qrs, qrs_energy

AF_FEATURES = [
    "rr_mean", "rmssd_rel", "sdnn_rel", "pnn50", "range_rel",
    "frac_dev", "mean_abs_drr_rel", "sampen", "rr_median", "rr_skew",
]


def rr_features(rr: np.ndarray) -> np.ndarray:
    """Vector de 10 características de una ventana de intervalos RR.

    Pensado para discriminar fibrilación auricular de ritmo sinusal,
    incluso cuando hay ectopia (que también produce irregularidad).
    """
    rr = np.asarray(rr, dtype=np.float64)
    rr = rr[np.isfinite(rr) & (rr > 0.15) & (rr < 4.0)]
    if rr.size < 6:
        return np.zeros(len(AF_FEATURES), dtype=np.float32)
    mean = float(rr.mean())
    med = float(np.median(rr))
    sdnn = float(rr.std(ddof=1)) if rr.size > 1 else 0.0
    drr = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(drr ** 2))) if drr.size else 0.0
    pnn50 = float(np.mean(np.abs(drr) > 0.05)) if drr.size else 0.0
    rng = float(rr.max() - rr.min())
    dev = float(np.mean(np.abs(rr - med) > 0.10))
    mad = float(np.mean(np.abs(drr)) / mean) if drr.size else 0.0
    skew = float(np.mean(((rr - mean) / (sdnn + 1e-9)) ** 3)) if sdnn > 0 else 0.0
    se = _sample_entropy(rr, m=2, r=0.10 * mean)
    return np.array(
        [mean, rmssd / (mean + 1e-9), sdnn / (mean + 1e-9), pnn50, rng / (med + 1e-9),
         dev, mad, se, med, skew],
        dtype=np.float32,
    )


def _sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.05) -> float:
    """Entropía muestral aproximada (vectorizada, n pequeño)."""
    n = len(x)
    if n <= m + 1 or r <= 0:
        return 0.0

    def count(mm: int) -> int:
        if n - mm + 1 < 2:
            return 0
        win = np.lib.stride_tricks.sliding_window_view(x, mm)  # (n-mm+1, mm)
        d = np.abs(win[:, None, :] - win[None, :, :]).max(axis=2)
        return int((d <= r).sum() - (n - mm + 1)) // 2

    b = count(m)
    a = count(m + 1)
    if b <= 0 or a <= 0:
        return 0.0
    return float(-np.log(a / b))


# ---------------------------------------------------------------------------
# Eventos clínicos
# ---------------------------------------------------------------------------
# Pausas significativas y su causa
# ---------------------------------------------------------------------------
# Umbrales clínicos admitidos en la sección de pausas del informe.
PAUSE_THRESHOLDS_S = (2.0, 2.5, 3.0)

# Causas informables de una pausa.
PAUSE_CAUSES = {
    "bloqueo_sa": "bloqueo sinoauricular",
    "paro_sinusal": "paro sinusal",
    "extrasistole_auricular_bloqueada": "extrasístole auricular bloqueada",
    "bloqueo_av": "bloqueo auriculoventricular",
    "pausa_post_fa": "pausa post-fibrilación auricular",
    "pausa_post_extrasistolica": "pausa postextrasistólica",
    "pausa_en_fa": "pausa en fibrilación auricular",
    "indeterminada": "causa no determinada",
}
# Causas que NO se informan como pausa (falsos positivos del detector).
PAUSE_REJECTS = {
    "artefacto": "pérdida de señal o ruido en ≥2 canales",
    "qrs_perdido": "QRS de bajo voltaje no detectado",
    "bajo_voltaje": "tramo de bajo voltaje (QRS atenuados)",
}

# Un latido Q (vetado o ilegible) a menos de este intervalo del último latido
# conservado no es un latido del paciente: es la onda T, o ruido, que el detector
# tomó por QRS. Ningún latido real llega tan pronto tras otro.
TWAVE_REFRACTORY_S = 0.45


def ritmo_mask(beats) -> np.ndarray:
    """Qué latidos forman la secuencia de ritmo (huecos, PP/RR de la FA).

    Quita los Q que llegan a menos de ``TWAVE_REFRACTORY_S`` del último latido
    conservado. Un Q así deja un RR de ~0,28 s seguido de otro de ~0,77 s: eso se
    lee como irregularidad (FA) o como un BSA tipo I falso. Los demás se conservan.
    """
    keep = np.ones(len(beats), dtype=bool)
    last = None
    for k, b in enumerate(beats):
        if (last is not None and b.cls == "Q"
                and float(b.t) - float(beats[last].t) < TWAVE_REFRACTORY_S):
            keep[k] = False
            continue
        last = k
    return keep


def ritmo_beats(beats) -> list:
    """Latidos de la secuencia de ritmo; ``rr_prev`` se recalcula donde cambia el previo.

    Si no se descarta ningún latido devuelve los mismos valores que la entrada.
    """
    keep = ritmo_mask(beats)
    if keep.all():
        return list(beats)
    out, prev = [], None
    for k in np.flatnonzero(keep):
        k = int(k)
        b = beats[k]
        if prev is not None and prev != k - 1:
            b = replace(b, rr_prev=float(b.t) - float(out[-1].t))
        out.append(b)
        prev = k
    return out


@dataclass
class PauseCriteria:
    """Criterios de la sección de pausas.

    `threshold_s` es el umbral de la configuración (2,0 / 2,5 / 3,0 s); el
    resto son tolerancias de la revisión y de la clasificación de la causa.
    """

    threshold_s: float = 2.0
    severe_s: float = 3.0          # pausa "crítica" a partir de este valor
    split_min_s: float = 3.0       # sub-pausa que se informa dentro de un hueco con latidos
    max_pause_s: float = 12.0      # huecos mayores: señal perdida o fallo del detector
    pp_tolerance: float = 0.10     # bloqueo SA: ±10 % del múltiplo del PP basal
    p_tolerance: float = 0.20      # ventana para la P esperada (±20 % del PP)
    prematurity: float = 0.85      # P prematura si llega antes del 85 % del PP
    pp_min_beats: int = 3
    wenck_step_s: float = 0.04     # BSA tipo I: acortamiento mínimo de cada PP (tolerancia de ruido)
    wenck_max_ratio: float = 2.0   # BSA tipo I: la pausa es menor que 2 × PP previo
    miss_min_leads: int = 1       # canales que deben coincidir para dar QRS perdido
    artifact_frac: float = 0.35    # fracción del hueco ilegible en ≥2 canales
    flat_uv: float = 20.0          # señal plana: pico a pico por bloque de 0,5 s
    sat_uv: float = 25000.0        # saturación del convertidor
    max_p2p_uv: float = 3000.0     # salto de la línea de base o golpe (por bloque)
    margin_s: float = 0.80         # contexto leído a cada lado del hueco
    use_beat_flags: bool = True    # usar las banderas de ruido del pipeline
    noise_flag: float = 0.10       # ruido por latido que cuenta como tramo ilegible
    lowvolt_sigma: float = 3.0     # QRS de bajo voltaje: umbral en σ combinada del ruido (10-30 Hz)
    lowvolt_floor_mv: float = 0.015  # ... y suelo absoluto (15 µV)
    lowvolt_ratio: float = 0.5     # QRS al < 50 % de la amplitud normal: tramo de bajo voltaje
    qrs_hf_lf: float = 0.15        # energía 10-30 Hz / 2-10 Hz mínima de un QRS (una P no llega)
    af_cover_frac: float = 0.5     # fracción del hueco cubierta por FA: pausa en FA
    terminal_p_s: float = 0.40     # la P del latido que cierra el hueco no cuenta como P dentro
    premature_ratio: float = 0.80  # latido prematuro: RR < 80 % del RR normal previo


def pause_threshold(value: float | str | None) -> float:
    """Normaliza el umbral de pausa a 2,0 / 2,5 / 3,0 s (admite otros ≥1 s)."""
    try:
        v = float(value) if value is not None else 2.0
    except (TypeError, ValueError):
        return 2.0
    if not np.isfinite(v) or v < 1.0:
        return 2.0
    for cand in PAUSE_THRESHOLDS_S:
        if abs(v - cand) < 1e-6:
            return cand
    return float(v)


def _mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _peaks_above(x: np.ndarray, thr: float, min_dist: int) -> np.ndarray:
    """Máximos locales por encima de `thr`, separados al menos `min_dist`."""
    if x.size < 3 or not np.isfinite(thr) or thr <= 0:
        return np.array([], dtype=int)
    c = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] > thr))[0] + 1
    if c.size == 0:
        return c
    order = c[np.argsort(x[c])[::-1]]
    taken = np.zeros(x.size, dtype=bool)
    keep: list[int] = []
    for i in order:
        if taken[i]:
            continue
        keep.append(int(i))
        taken[max(0, i - min_dist): i + min_dist + 1] = True
    return np.sort(np.asarray(keep, dtype=int))


def _coincident_peaks(envs, fs: float, thrs, min_leads: int,
                      min_dist_s: float) -> list[float]:
    """Picos que aparecen en ≥`min_leads` canales dentro de 60 ms."""
    if not envs:
        return []
    min_dist = max(1, int(round(min_dist_s * fs)))
    tol = max(1, int(round(0.06 * fs)))
    groups: list[list[int]] = []
    for env, thr in zip(envs, thrs):
        peaks = _peaks_above(np.asarray(env, dtype=float), float(thr), min_dist)
        if not peaks.size:
            continue
        if not groups:
            groups = [[int(p)] for p in peaks]
            continue
        used = np.zeros(peaks.size, dtype=bool)
        for g in groups:
            cen = g[0] if len(g) == 1 else int(round(float(np.median(g))))
            j = int(np.searchsorted(peaks, cen))
            best = None
            for k in (j - 1, j):
                if 0 <= k < peaks.size and not used[k] and abs(int(peaks[k]) - cen) <= tol:
                    if best is None or abs(int(peaks[k]) - cen) < abs(int(peaks[best]) - cen):
                        best = int(k)
            if best is not None:
                used[best] = True
                g.append(int(peaks[best]))
    need = max(1, min(int(min_leads), len(envs)))
    return sorted(float(np.median(g)) / fs for g in groups if len(g) >= need)


def _ref_level(e: np.ndarray, a: int, b: int) -> float:
    """Nivel de energía típico fuera del hueco (los márgenes tienen QRS)."""
    outside = np.concatenate([e[: max(0, a)], e[b:]])
    if outside.size < 3:
        outside = e
    return float(np.percentile(outside, 95.0)) if outside.size else 0.0


def find_missed_qrs(seg, fs: float, *, gap: tuple[float, float] = (0.0, 0.0),
                    min_leads: int = 2, min_dist_s: float = 0.24,
                    guard_s: float = 0.45) -> list[float]:
    """Busca QRS de bajo voltaje dentro de un hueco antes de declarar pausa.

    `seg` es la señal (muestras × derivaciones, en mV) que cubre el hueco más
    un margen a cada lado; `gap` son los límites del hueco en segundos desde
    el inicio de `seg`. Se corre el mismo detector de QRS del pipeline
    (`detect_qrs`, umbral adaptativo) en cada derivación y se quedan los
    picos que aparecen en al menos `min_leads` canales dentro de 60 ms. Se
    usa un detector independiente del que produjo los latidos: si coincide,
    el hueco no es una pausa sino un QRS que el detector principal perdió.

    Los primeros y últimos `guard_s` segundos del hueco no se revisan: ahí
    caen el QRS y la T de los latidos que lo delimitan.

    Devuelve los tiempos (segundos desde el inicio de `seg`) de los QRS
    encontrados, para que el hueco no se informe como pausa.
    """
    x = np.asarray(seg, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n, nl = x.shape
    if nl == 0 or n < int(0.5 * fs):
        return []
    a = int(round(max(0.0, float(gap[0])) * fs))
    b = int(round(min(float(n - 1), float(gap[1]) * fs)))
    guard = int(round(max(0.0, float(guard_s)) * fs))
    if b - a <= 2 * guard + int(0.15 * fs):
        return []
    lo, hi = a + guard, b - guard
    peaks: list[tuple[int, int]] = []
    for j in range(nl):
        found = detect_qrs(x[:, j], fs)
        peaks.extend((int(p), j) for p in found if lo <= p <= hi)
    if not peaks:
        return []
    tol = max(1, int(round(0.06 * fs)))
    groups: list[list[tuple[int, int]]] = []
    for p, j in sorted(peaks):
        if groups and p - groups[-1][-1][0] <= tol:
            groups[-1].append((p, j))
        else:
            groups.append([(p, j)])
    need = max(1, min(int(min_leads), nl))
    hits = [float(np.median([g[0] for g in gr])) / fs for gr in groups
            if len({g[1] for g in gr}) >= need]
    # separación mínima entre QRS (el detector ya la impone por canal)
    out: list[float] = []
    for t in hits:
        if not out or t - out[-1] >= min_dist_s:
            out.append(t)
    return out


def missing_qrs_in_gap(samples, fs: float, t0: float, t1: float,
                       criteria: PauseCriteria | None = None) -> list[float]:
    """QRS presentes en el intervalo [t0, t1] (tiempos absolutos).

    Es la revisión que se hace antes de informar una pausa: si hay QRS en el
    hueco, el hueco no es una pausa (el detector los perdió). Devuelve los
    tiempos de esos QRS; lista vacía = no hay QRS en la señal.
    """
    crit = criteria or PauseCriteria()
    seg, off = _gap_signal(samples, fs, t0 - crit.margin_s, t1 + crit.margin_s)
    if seg is None:
        return []
    hits = find_missed_qrs(seg, fs, gap=(max(0.0, t0 - off), max(0.0, t1 - off)),
                           min_leads=crit.miss_min_leads)
    return [float(h) + off for h in hits]


def detect_p_waves(seg, fs: float, *, min_leads: int = 2,
                   min_dist_s: float = 0.20) -> list[float]:
    """Ondas P candidatas en un segmento corto (coincidencia en ≥2 canales).

    Red de seguridad para clasificar una pausa cuando el detector de P de
    T10 no está disponible: se queda con los picos de la banda 2–16 Hz cuya
    amplitud es propia de una P (ni ruido ni QRS). Devuelve tiempos en
    segundos desde el inicio de `seg`.
    """
    x = np.asarray(seg, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n, nl = x.shape
    if nl == 0 or n < int(0.5 * fs):
        return []
    h = fir_bandpass(fs, 2.0, min(16.0, 0.40 * fs), 201)
    win = max(3, int(round(0.05 * fs)))
    ys: list[np.ndarray] = []
    envs: list[np.ndarray] = []
    thrs: list[float] = []
    refs: list[float] = []
    for j in range(nl):
        y = filtfilt(x[:, j], h)
        env = moving_average(np.abs(y), win)
        ref = float(np.percentile(np.abs(y), 99.5))
        refs.append(ref)
        ys.append(y)
        envs.append(env)
        thrs.append(max(3.0 * _mad_sigma(env), 0.04 * ref))
    out: list[float] = []
    half = max(1, int(round(0.03 * fs)))
    need = max(1, min(int(min_leads), nl))
    for t in _coincident_peaks(envs, fs, thrs, need, min_dist_s):
        i = int(round(t * fs))
        ok = 0
        for y, ref in zip(ys, refs):
            if ref <= 0:
                continue
            amp = float(np.max(np.abs(y[max(0, i - half): i + half + 1])))
            if 0.03 * ref <= amp <= 0.75 * ref:
                ok += 1
        if ok >= need:
            out.append(float(t))
    return out


def _quality_intervals(quality) -> list[tuple[float, float]]:
    """Tramos ilegibles de T21 como [(t0, t1)] (acepta dict o lista)."""
    if quality is None:
        return []
    raw = quality
    if isinstance(quality, dict):
        raw = (quality.get("unreadable_intervals") or quality.get("unreadable")
               or quality.get("intervals") or [])
    out: list[tuple[float, float]] = []
    for item in raw or []:
        try:
            if isinstance(item, dict):
                a, b = float(item["start_s"]), float(item["end_s"])
            else:
                a, b = float(item[0]), float(item[1])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if b > a:
            out.append((a, b))
    return out


def _overlap_frac(t0: float, t1: float, intervals) -> float:
    dur = max(1e-6, t1 - t0)
    total = 0.0
    for a, b in intervals or []:
        lo, hi = max(t0, a), min(t1, b)
        if hi > lo:
            total += hi - lo
    return float(min(1.0, total / dur))


def _af_intervals(rhythm) -> list[tuple[float, float]]:
    """Tramos de fibrilación auricular (RhythmSegment o dicts)."""
    out: list[tuple[float, float]] = []
    for seg in rhythm or []:
        label = getattr(seg, "label", None)
        if label is None and isinstance(seg, dict):
            label = seg.get("label")
        if str(label).upper() not in ("AF", "AFIB"):
            continue
        a = getattr(seg, "start_s", None)
        b = getattr(seg, "end_s", None)
        if a is None and isinstance(seg, dict):
            a, b = seg.get("start_s"), seg.get("end_s")
        try:
            out.append((float(a), float(b)))
        except (TypeError, ValueError):
            continue
    return out


def _gap_signal(samples, fs: float, t0: float, t1: float):
    """Recorta la señal entre dos tiempos; devuelve (señal, tiempo de inicio).

    `samples` puede ser un ndarray, un memmap o cualquier objeto con `shape`
    y `__getitem__` (así un Holter de 24 h no se copia entero).
    """
    shape = getattr(samples, "shape", None)
    if shape is None:
        shape = np.shape(samples)
    if len(shape) not in (1, 2):
        return None, 0.0
    n = int(shape[0])
    a = max(0, int(t0 * fs))
    b = min(n, int(t1 * fs) + 1)
    if b - a < 4:
        return None, 0.0
    seg = np.array(samples[a:b], dtype=np.float64, copy=True)
    if seg.ndim == 1:
        seg = seg[:, None]
    return seg, a / float(fs)


def _unreadable_fraction(seg: np.ndarray, fs: float, crit: PauseCriteria,
                         span: tuple[float, float] | None = None) -> float:
    """Fracción del hueco con pérdida de señal en ≥2 canales.

    Bloquea de 0,5 s con la señal plana (ruido o electrodo caído) o
    saturada. Con una sola derivación basta con que esa derivación falle.
    """
    x = np.asarray(seg, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if span is not None:
        a0 = max(0, int(round(max(0.0, span[0]) * fs)))
        b0 = min(x.shape[0], int(round(max(0.0, span[1]) * fs)))
        if b0 > a0:
            x = x[a0:b0]
    n, nl = x.shape
    win = max(8, int(round(0.5 * fs)))
    n_blk = n // win
    if nl == 0 or n_blk == 0:
        return 0.0
    blocks = x[: n_blk * win].reshape(n_blk, win, nl)
    p2p = blocks.max(axis=1) - blocks.min(axis=1)
    peak = np.abs(blocks).max(axis=1)
    bad = ((p2p < crit.flat_uv / 1000.0) | (peak >= crit.sat_uv / 1000.0)
           | (p2p > crit.max_p2p_uv / 1000.0))
    centers = (np.arange(n_blk) * win + win // 2).astype(np.int64)
    for j in range(nl):
        nz = noise_batch(x[:, j], fs, centers, win_s=win / fs)
        bad[:, j] |= np.asarray(nz, dtype=float) > crit.noise_flag
    need = 1 if nl == 1 else 2
    return float(np.mean(bad.sum(axis=1) >= need))


def _beat_flags_artifact(beats, i: int, crit: PauseCriteria) -> bool:
    """¿Los latidos que rodean el hueco están marcados como ruidosos?"""
    if not crit.use_beat_flags:
        return False
    near = list(beats[max(0, i - 2): min(len(beats), i + 2)])
    flags = [bool(b.features.get("artifact")) for b in near
             if getattr(b, "features", None) and "artifact" in b.features]
    if flags and sum(flags) >= max(2, len(flags) - 1):
        return True
    noises = [float(b.features.get("noise", 0.0)) for b in near
              if getattr(b, "features", None) and b.features.get("noise") is not None]
    return bool(noises) and float(np.median(noises)) > crit.noise_flag


def _pp_series(beats, i: int, crit: PauseCriteria, p_times, t0: float) -> tuple[list[float], str]:
    """Intervalos PP (en orden) justo antes de la pausa, y su fuente.

    Fuente "P" si hay ondas P medidas (el PP real). Si no, "RR": el RR de los
    latidos no ventriculares es un estimador del PP porque el PR es constante, y
    el informe lo avisa. No se descartan los S: el clasificador marca como
    prematuros los PP acortados de un Wenckebach, y son justo los que hay que ver.
    """
    if p_times is not None:
        prev = np.asarray([float(x) for x in p_times if float(x) < t0 - 0.05], dtype=float)
        if prev.size >= crit.pp_min_beats:
            d = np.diff(prev[-(crit.pp_min_beats + 1):])
            d = d[(d > 0.25) & (d < 2.0)]
            if d.size >= 2:
                return [float(x) for x in d], "P"
    vals = [float(beats[k].rr_prev) for k in range(max(1, i - 6), i)
            if beats[k].cls != "V" and 0.25 < float(beats[k].rr_prev) < 1.8]
    return (vals, "RR") if len(vals) >= 2 else ([], "RR")


def _basal_pp(beats, i: int, crit: PauseCriteria, p_times, t0: float) -> float:
    """PP basal antes de la pausa (de las P si hay, si no del RR normal)."""
    pps, _ = _pp_series(beats, i, crit, p_times, t0)
    return float(np.median(pps)) if pps else 0.0


def _wenckebach_ratio(dur: float, pps: list[float], crit: PauseCriteria) -> float | None:
    """BSA tipo I (Wenckebach): `dur / PP previo`, o None si no lo es.

    Exige (a) acortamiento progresivo de ≥2 ciclos antes de la pausa: tres PP
    seguidos, cada uno más corto que el anterior en más de `wenck_step_s`; y
    (b) pausa mayor que el último PP pero menor que `wenck_max_ratio` veces ese PP.
    Sin (a) un PP irregular cualquiera calzaría; sin (b) es un 2:1 o un paro.
    """
    if len(pps) < 3:
        return None
    d1, d2, d3 = pps[-3:]
    if not (d1 - d2 > crit.wenck_step_s and d2 - d3 > crit.wenck_step_s):
        return None
    if not (d3 < dur < crit.wenck_max_ratio * d3):
        return None
    return round(dur / d3, 2)


LOWAMP_BAND_HZ = (10.0, 30.0)   # banda del QRS: sin las ondas f de la FA (5-9 Hz)


def _qrs_band(x: np.ndarray, fs: float) -> np.ndarray:
    """Señal en la banda del QRS (10-30 Hz), muestras × derivaciones."""
    h = fir_bandpass(fs, LOWAMP_BAND_HZ[0], min(LOWAMP_BAND_HZ[1], 0.40 * fs), 201)
    return np.column_stack([filtfilt(x[:, j], h) for j in range(x.shape[1])])


def _low_band(x: np.ndarray, fs: float) -> np.ndarray:
    """Señal en 2-10 Hz (donde está la onda P), muestras × derivaciones."""
    h = fir_bandpass(fs, 2.0, min(10.0, 0.40 * fs), 201)
    return np.column_stack([filtfilt(x[:, j], h) for j in range(x.shape[1])])


def _qrs_amp(y: np.ndarray, fs: float, t: float) -> float:
    """Amplitud de un QRS en la banda del QRS: suma cuadrática de los picos por derivación."""
    c = int(round(t * fs))
    w = max(1, int(round(0.08 * fs)))
    a, b = max(0, c - w), min(y.shape[0], c + w + 1)
    if b <= a:
        return 0.0
    pk = np.abs(y[a:b]).max(axis=0)
    return float(np.sqrt(np.sum(pk * pk)))


def _lowamp_qrs(y: np.ndarray, lf: np.ndarray, fs: float, gap: tuple[float, float],
                crit: PauseCriteria) -> list[float]:
    """QRS de amplitud baja dentro del hueco (segundos desde el inicio de `y`).

    Se combinan las derivaciones en una envolvente (raíz de la suma de
    cuadrados) y se busca en ella con un umbral de `lowvolt_sigma` veces la
    σ combinada del ruido del propio hueco (con suelo `lowvolt_floor_mv`).
    Cada pico debe tener energía de QRS (10-30 Hz frente a 2-10 Hz): una onda P
    de bajo voltaje pasa el umbral pero no esa prueba. Como en `find_missed_qrs`,
    no se revisan los 0,45 s de borde: ahí caen el QRS y la T de los vecinos.
    """
    n, nl = y.shape
    a = int(round(max(0.0, gap[0]) * fs))
    b = int(round(min(float(n - 1), gap[1]) * fs))
    guard = int(round(0.45 * fs))
    if nl == 0 or b - a <= 2 * guard + int(0.15 * fs):
        return []
    sig2 = 0.0
    z2 = np.zeros(n)
    for j in range(nl):
        sig2 += _mad_sigma(y[a:b, j]) ** 2
        z2 += y[:, j] ** 2
    thr = max(crit.lowvolt_sigma * float(np.sqrt(sig2)), crit.lowvolt_floor_mv)
    peaks = _peaks_above(np.sqrt(z2), thr, int(round(0.25 * fs)))
    lo, hi = (a + guard) / fs, (b - guard) / fs
    w = int(round(0.08 * fs))
    out: list[float] = []
    for pk in peaks:
        t = float(pk) / fs
        if not lo <= t <= hi:
            continue
        i0, i1 = max(0, int(pk) - w), min(n, int(pk) + w + 1)
        e_hf = float(np.sum(y[i0:i1] ** 2))
        e_lf = float(np.sum(lf[i0:i1] ** 2))
        if e_lf > 0 and e_hf >= crit.qrs_hf_lf * e_lf:
            out.append(t)
    return out


def _ref_qrs_amp(beats, i: int, t0: float, t1: float, y: np.ndarray, fs: float,
                 off: float, crit: PauseCriteria) -> float | None:
    """Amplitud típica de los QRS normales de los latidos vecinos al hueco."""
    amps = []
    for k in range(max(0, i - 2), min(len(beats), i + 2)):
        tk = float(beats[k].t)
        if beats[k].cls != "N" or not (t0 - crit.margin_s <= tk <= t1 + crit.margin_s):
            continue
        amps.append(_qrs_amp(y, fs, tk - off))
    amps = [v for v in amps if v > 0]
    return float(np.median(amps)) if amps else None


def _ectopic_before(beats, k: int, crit: PauseCriteria) -> bool:
    """¿El latido k es prematuro (extrasístole)? RR < `premature_ratio` del RR normal previo.

    La clase del latido no basta: el clasificador marca S latidos sinusales
    de un ritmo con bloqueo (Wenckebach) que no llegan antes de tiempo.
    """
    b = beats[k]
    rr = float(b.rr_prev)
    ref = [float(beats[j].rr_prev) for j in range(max(1, k - 8), k)
           if beats[j].cls == "N" and 0.25 < float(beats[j].rr_prev) < 1.8]
    if rr <= 0:
        return False
    if len(ref) < 3:
        return b.cls in ("V", "S")
    return rr < crit.premature_ratio * float(np.median(ref))


def _pause_cause(beats, i: int, t0: float, t1: float, crit: PauseCriteria,
                 samples, fs: float, q_int, p_times, af_int,
                 search_missed: bool = True) -> tuple[str, dict, list[float]]:
    """Revisa el hueco y decide la causa (o por qué se descarta).

    Devuelve `(causa, evidencia, qrs_en_el_hueco)`. Los QRS encontrados por el
    buscador de bajo umbral se devuelven en tiempo absoluto: sirven para
    partir el hueco y no perder una pausa real dentro de un tramo con latidos.
    """
    ev: dict = {"dur_s": round(t1 - t0, 3)}
    found: list[float] = []
    # un silencio de minutos no es una pausa: es pérdida de señal o un fallo
    # del detector (la asistolia se informa en su propia sección)
    if crit.max_pause_s > 0 and (t1 - t0) > crit.max_pause_s:
        ev["sin_senal"] = 1.0
        return "artefacto", ev, found
    seg, off = (None, 0.0)
    if samples is not None:
        seg, off = _gap_signal(samples, fs, t0 - crit.margin_s, t1 + crit.margin_s)

    # 1) artefacto: máscara de calidad, pérdida de señal o ruido multicanal
    art = _overlap_frac(t0, t1, q_int)
    if seg is not None:
        art = max(art, _unreadable_fraction(seg, fs, crit,
                                            span=(t0 - off, t1 - off)))
    ev["artifact_frac"] = round(float(art), 3)
    flags = _beat_flags_artifact(beats, i, crit)
    ev["beat_flags"] = float(flags)
    if art >= crit.artifact_frac or flags:
        return "artefacto", ev, found

    # 2) QRS perdido: se busca en todos los canales. Si los QRS que aparecen
    #    son de amplitud baja frente a los latidos vecinos, es un tramo de bajo
    #    voltaje (no se informa); si son de amplitud normal, es un QRS perdido.
    if search_missed:
        if seg is not None:
            found = [float(h) + off for h in
                     find_missed_qrs(seg, fs, gap=(max(0.0, t0 - off), max(0.0, t1 - off)),
                                     min_leads=crit.miss_min_leads)]
        elif samples is not None:
            found = missing_qrs_in_gap(samples, fs, t0, t1, crit)
        if found:
            ev["missed_qrs"] = len(found)
            return "qrs_perdido", ev, found
        if seg is not None:
            y = _qrs_band(seg, fs)
            low = _lowamp_qrs(y, _low_band(seg, fs), fs,
                              (max(0.0, t0 - off), max(0.0, t1 - off)), crit)
            if low:
                ref = _ref_qrs_amp(beats, i, t0, t1, y, fs, off, crit)
                amp = float(np.median([_qrs_amp(y, fs, h) for h in low]))
                ratio = (amp / ref) if ref else None
                found = [float(h) + off for h in low]
                ev["missed_qrs"] = len(found)
                if ratio is not None:
                    ev["lowvolt_ratio"] = round(float(ratio), 3)
                    if ratio < crit.lowvolt_ratio:
                        return "bajo_voltaje", ev, found
                return "qrs_perdido", ev, found

    # 3) pausa en fibrilación auricular: sin ondas P justo antes y la FA cubre
    #    el hueco o lo rodea (sigue antes de la pausa y se reanuda después)
    af_cov = _overlap_frac(t0, t1, af_int)
    prev_af = any(b >= t0 - 30.0 and a <= t0 + 0.5 for a, b in af_int)
    resumes = any(t1 - 1.0 <= a <= t1 + 30.0 for a, _ in af_int)
    no_p = p_times is None or not any(t0 - 8.0 <= float(x) <= t0 for x in p_times)
    if no_p and (af_cov >= crit.af_cover_frac or (prev_af and resumes)):
        ev["af_frac"] = round(af_cov, 3)
        return "pausa_en_fa", ev, found

    # 4) pausa post-fibrilación auricular: la FA termina en la pausa
    if any(t0 - 10.0 <= b <= t1 + 0.5 for _, b in af_int):
        return "pausa_post_fa", ev, found

    # 5) pausa postextrasistólica: el hueco empieza en una extrasístole
    if abs(float(beats[i - 1].t) - t0) < 0.05 and _ectopic_before(beats, i - 1, crit):
        return "pausa_post_extrasistolica", ev, found

    # 6) causa auricular: P bloqueada, P regular sin QRS, o múltiplo del PP.
    #    La P del latido que cierra el hueco (su PR) no cuenta como P dentro.
    pps, pp_src = _pp_series(beats, i, crit, p_times, t0)
    pp = float(np.median(pps)) if pps else 0.0
    ev["pp_s"] = round(pp, 3)
    ev["pp_fuente"] = pp_src
    p_rel: list[float] = []
    if p_times is not None:
        p_rel = [float(x) - t0 for x in p_times if t0 - 0.8 <= float(x) <= t1 + 0.2]
    elif seg is not None:
        p_rel = [float(x) + off - t0 for x in detect_p_waves(seg, fs)]
    dur = t1 - t0
    p_rel = sorted(d for d in p_rel if d <= dur - crit.terminal_p_s)
    ev["p_found"] = len(p_rel)
    if pp > 0:
        inside = [d for d in p_rel if d >= 0.10]
        at_pp = [d for d in p_rel if abs(d - pp) <= crit.p_tolerance * pp]
        early = [d for d in p_rel if 0.12 <= d <= crit.prematurity * pp]
        k = int(round(dur / pp))
        if k >= 1:
            ev["pp_multiple"] = float(k)
            ev["pp_err"] = round(abs(dur - k * pp) / (k * pp), 3)
        if early and not at_pp:
            return "extrasistole_auricular_bloqueada", ev, found
        if inside and (at_pp or len(inside) >= 2):
            return "bloqueo_av", ev, found
        # BSA tipo I (Wenckebach) antes que el múltiplo: una pausa de Wenckebach
        # (< 2 × PP) también puede calzar con 2 × PP mediano
        wb = _wenckebach_ratio(dur, pps, crit)
        if wb is not None and not inside:
            ev["bsa_tipo"] = "I"
            ev["pp_ratio"] = wb
            return "bloqueo_sa", ev, found
        if k >= 2 and ev.get("pp_err", 1.0) <= crit.pp_tolerance and not inside:
            ev["bsa_tipo"] = "II"
            return "bloqueo_sa", ev, found
        if inside:
            return "indeterminada", ev, found
    return "paro_sinusal", ev, found


def _split_gap(t0: float, t1: float, found, crit: PauseCriteria) -> list[tuple[float, float]]:
    """Parte un hueco por los QRS encontrados y devuelve los sub-huecos largos.

    Un hueco con latidos de bajo voltaje no es una pausa, pero puede contener
    una: entre dos latidos seguidos el intervalo puede seguir siendo ≥ umbral.
    Solo se informan los sub-huecos ≥ `split_min_s` (3 s): los cortos no
    cambian el informe y sí añaden ruido.
    """
    min_s = max(float(crit.threshold_s), float(crit.split_min_s))
    inside = sorted({round(float(x), 3) for x in found if t0 + 0.20 < float(x) < t1 - 0.20})
    bounds = [t0, *inside, t1]
    return [(a, b) for a, b in zip(bounds, bounds[1:], strict=False) if b - a >= min_s]


def _pause_detail(cause: str, dur: float, ev: dict) -> str:
    if cause in PAUSE_REJECTS:
        base = f"Hueco de {dur:.1f} s descartado: {PAUSE_REJECTS[cause]}"
    else:
        base = f"Pausa de {dur:.1f} s: {PAUSE_CAUSES.get(cause, cause)}"
    extra = []
    if cause == "bloqueo_sa" and ev.get("bsa_tipo") == "I":
        extra.append(f"tipo I, Wenckebach: {ev.get('pp_ratio', 0.0):.2f}× PP previo")
    elif cause == "bloqueo_sa" and ev.get("pp_multiple"):
        extra.append(f"tipo II, {int(ev['pp_multiple'])}× PP de {ev.get('pp_s', 0.0):.2f} s")
    elif cause in ("bloqueo_av", "extrasistole_auricular_bloqueada") and ev.get("pp_s"):
        extra.append(f"PP basal {ev['pp_s']:.2f} s")
    if (cause in ("bloqueo_sa", "bloqueo_av", "extrasistole_auricular_bloqueada")
            and ev.get("pp_fuente") == "RR"):
        extra.append("PP por RR: sin ondas P fiables")
    elif cause == "qrs_perdido" and ev.get("missed_qrs"):
        extra.append(f"{int(ev['missed_qrs'])} en los canales con umbral bajo")
    elif cause == "artefacto" and ev.get("artifact_frac"):
        extra.append(f"{100.0 * ev['artifact_frac']:.0f} % del hueco ilegible")
    elif cause == "bajo_voltaje" and ev.get("lowvolt_ratio") is not None:
        extra.append(f"QRS al {100.0 * ev['lowvolt_ratio']:.0f} % de la amplitud normal")
    return base + (f" ({'; '.join(extra)})" if extra else "")


def pause_events(
    beats,
    threshold_s: float = 2.0,
    *,
    samples=None,
    fs: float = 250.0,
    quality=None,
    p_waves=None,
    rhythm=None,
    criteria: PauseCriteria | None = None,
    include_rejected: bool = False,
) -> list[dict]:
    """Pausas significativas (RR ≥ umbral) con revisión y causa.

    Antes de informar una pausa se descartan dos causas de falso positivo:

    * **QRS perdido por bajo voltaje**: se busca un QRS dentro del hueco en
      todos los canales con umbral bajo (`find_missed_qrs`).
    * **Artefacto**: pérdida de señal o ruido en ≥2 canales, medido con la
      máscara de calidad de T21, con la propia señal o con las banderas de
      ruido por latido del pipeline.

    Cada pausa informada queda clasificada en `pause_class`: paro sinusal,
    bloqueo sinoauricular (la pausa es un múltiplo entero del PP basal
    ±10 %), extrasístole auricular bloqueada (P prematura en la T previa),
    bloqueo AV (P sin QRS, de T24), pausa en FA, pausa post-FA, pausa
    postextrasistólica o indeterminada. Un hueco con QRS de amplitud baja
    (< 50 % de la normal) se descarta como `bajo_voltaje`.

    Si el hueco tiene latidos de bajo voltaje, no se informa como pausa; pero
    se revisa si entre esos latidos queda un silencio ≥ `split_min_s` (3 s) y
    ese silencio sí se informa.

    Parámetros opcionales
    ---------------------
    samples : matriz (n_muestras, n_derivaciones) en mV o None.
    quality : tramos ilegibles de T21 (`[(t0, t1)]` o dict con la lista).
    p_waves : tiempos de P en segundos, en el mismo reloj que `beats`.
    rhythm  : tramos de ritmo (`RhythmSegment` o dicts con `label`).
    include_rejected : devuelve también los huecos descartados (auditoría).
    """
    crit = criteria or PauseCriteria(threshold_s=pause_threshold(threshold_s))
    thr = crit.threshold_s
    # la onda T tomada por QRS no es un latido: no parte el hueco ni falsea el PP/RR
    beats = ritmo_beats(beats)
    n = len(beats)
    if n < 2:
        return []
    t = np.array([b.t for b in beats], dtype=float)
    q_int = _quality_intervals(quality)
    af_int = _af_intervals(rhythm)
    p_times = None
    if p_waves is not None:
        p_times = np.asarray([float(x) for x in p_waves], dtype=float)
        p_times = np.sort(p_times[np.isfinite(p_times)])
    out: list[dict] = []
    for i in range(1, n):
        t0, t1 = float(t[i - 1]), float(t[i])
        dur = t1 - t0
        if dur < thr or dur <= 0:
            continue
        cause, evidence, found = _pause_cause(beats, i, t0, t1, crit, samples, fs,
                                              q_int, p_times, af_int)
        if cause in ("qrs_perdido", "bajo_voltaje") and found:
            # el hueco tiene latidos, pero entre ellos puede quedar una pausa
            for a, b in _split_gap(t0, t1, found, crit):
                sub_cause, sub_ev, _ = _pause_cause(beats, i, a, b, crit, samples, fs,
                                                    q_int, p_times, af_int,
                                                    search_missed=False)
                if sub_cause in PAUSE_REJECTS and not include_rejected:
                    continue
                out.append(_pause_event(beats, i, a, b, sub_cause, sub_ev, crit))
        if cause in PAUSE_REJECTS and not include_rejected:
            continue
        out.append(_pause_event(beats, i, t0, t1, cause, evidence, crit))
    out.sort(key=lambda e: (e["start_s"], e["end_s"]))
    return out


def _pause_event(beats, i: int, t0: float, t1: float, cause: str, ev: dict,
                 crit: PauseCriteria) -> dict:
    """Evento `pause` del informe a partir del hueco y su causa."""
    dur = t1 - t0
    return {
        "kind": "pause",
        "start_s": t0,
        "end_s": t1,
        "duration_s": round(dur, 3),
        "severity": "critical" if dur >= crit.severe_s else "warn",
        "detail": _pause_detail(cause, dur, ev),
        "beats": [int(beats[i].idx)],
        "pause_class": cause,
        "reported": cause not in PAUSE_REJECTS,
        "evidence": ev,
    }


def rate_events(beats, brady: float = 50.0, tachy: float = 100.0,
                min_beats: int = 5) -> list[dict]:
    """Episodios de bradicardia/taquicardia sostenidos."""
    out = []
    # una onda T tomada por QRS (Q) daría RR de 0,28 s y una taquicardia falsa (T164)
    beats = ritmo_beats(beats)
    if len(beats) < min_beats + 1:
        return out
    rr = np.array([b.rr_prev for b in beats], dtype=float)
    hr = np.where(rr > 0, 60.0 / np.maximum(rr, 1e-6), np.nan)
    for kind, cond, thr, sev in (
        ("brady", hr < brady, brady, "warn"),
        ("tachy", hr > tachy, tachy, "warn"),
    ):
        run_start = None
        for i in range(len(hr)):
            ok = bool(cond[i]) and np.isfinite(hr[i])
            if ok and run_start is None:
                run_start = i
            elif not ok and run_start is not None:
                if i - run_start >= min_beats:
                    out.append({
                        "kind": kind,
                        "start_s": beats[run_start].t,
                        "end_s": beats[i].t,
                        "severity": sev,
                        "detail": f"{kind} ({i - run_start} latidos, {np.nanmean(hr[run_start:i]):.0f} lpm)",
                        "beats": [b.idx for b in beats[run_start:i]],
                    })
                run_start = None
        if run_start is not None and len(hr) - run_start >= min_beats:
            out.append({
                "kind": kind,
                "start_s": beats[run_start].t,
                "end_s": beats[-1].t,
                "severity": sev,
                "detail": f"{kind} ({len(hr) - run_start} latidos)",
                "beats": [b.idx for b in beats[run_start:]],
            })
    return out


def ectopic_runs(beats, min_v: int = 3) -> list[dict]:
    """Rachas de ectopia ventricular (≥3 PVC seguidos = TV no sostenida)."""
    out = []
    run: list = []
    for b in beats:
        if b.cls == "V":
            run.append(b)
        else:
            if len(run) >= min_v:
                out.append({
                    "kind": "vt_run",
                    "start_s": run[0].t,
                    "end_s": run[-1].t,
                    "severity": "critical",
                    "detail": f"Racha de {len(run)} latidos ventriculares",
                    "beats": [x.idx for x in run],
                })
            run = []
    if len(run) >= min_v:
        out.append({
            "kind": "vt_run",
            "start_s": run[0].t,
            "end_s": run[-1].t,
            "severity": "critical",
            "detail": f"Racha de {len(run)} latidos ventriculares",
            "beats": [x.idx for x in run],
        })
    return out


def bigeminy_events(beats) -> list[dict]:
    """Detecta bigeminismo (alternancia N-V) en tramos de ≥6 latidos."""
    out = []
    pat = [b.cls for b in beats]
    i = 0
    n = len(pat)
    while i < n - 5:
        k = 0
        while (i + 2 * k + 1 < n and pat[i + 2 * k] == "N" and pat[i + 2 * k + 1] == "V"):
            k += 1
        if k >= 3:
            out.append({
                "kind": "bigeminy",
                "start_s": beats[i].t,
                "end_s": beats[i + 2 * k].t,
                "severity": "info",
                "detail": f"Bigeminismo ({k} ciclos)",
                "beats": [b.idx for b in beats[i:i + 2 * k + 1]],
            })
            i += 2 * k
        else:
            i += 1
    return out


def af_probabilities(rr: np.ndarray, model, win: int = 30,
                     stride: int = 5) -> np.ndarray:
    """Probabilidad de FA por latido (ventanas deslizantes de 30 RR)."""
    rr = np.asarray(rr, dtype=float)
    n = rr.size
    probs = np.zeros(n)
    count = np.zeros(n)
    if n < win + 1:
        return probs
    for end in range(win, n + 1, stride):
        seg = rr[end - win:end]
        if np.any((seg < 0.2) | (seg > 3.0)):
            continue
        p = float(model.predict(rr_features(seg)[None])[0])
        probs[end - win:end] += p
        count[end - win:end] += 1
    return np.divide(probs, np.maximum(count, 1))


def rolling_percentile(x: np.ndarray, half: int = 5, q: float = 75.0) -> np.ndarray:
    """Percentil móvil (ventana 2*half+1) sin bucle Python."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    xc = np.clip(x, 0.15, 3.0)
    pad = np.pad(xc, half, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)
    return np.percentile(win, q, axis=1)


def baseline_fraction(xf: np.ndarray, fs: float, t0: float, t1: float,
                      thr: float = 0.02) -> float:
    """Fracción de tiempo con señal cerca de la línea de base.

    En ritmo sinusal hay diástole isoeléctrica (fracción alta); en
    fibrilación ventricular la señal oscila de forma continua (fracción baja).
    Es el criterio más específico para distinguir FV de bigeminismo o FA.
    """
    a = max(0, int(t0 * fs))
    b = min(len(xf), int(t1 * fs))
    if b - a < 10:
        return 0.0
    seg = np.asarray(xf[a:b], dtype=float)
    med = float(np.median(seg))
    return float(np.mean(np.abs(seg - med) < thr))


def no_organized_rhythm_events(beats, corr, win: int = 10, corr_thr: float = 0.30,
                               frac_thr: float = 0.35, cv_thr: float = 0.30,
                               min_dur: float = 6.0) -> list[dict]:
    """Detecta tramos sin ritmo organizado (FV / TV polimórfica).

    En fibrilación ventricular el detector de QRS sigue encontrando "picos",
    pero su morfología es incoherente y el RR, caótico. La distribución de
    correlaciones es bimodal, así que se usa la **fracción** de latidos con
    correlación baja en la ventana (no la mediana) junto al CV del RR.
    """
    n = len(beats)
    if n < win or corr is None:
        return []
    corr = np.asarray(corr, dtype=float)
    t = np.array([b.t for b in beats], dtype=float)
    rr = np.array([b.rr_prev for b in beats], dtype=float)
    flags = np.zeros(n, dtype=bool)
    half = win // 2
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        r = rr[a:b]
        r = r[(r > 0.10) & (r < 2.0)]
        if r.size < 3:
            continue
        cv = float(np.std(r) / max(np.mean(r), 1e-6))
        frac = float(np.mean(corr[a:b] < corr_thr))
        flags[i] = (frac > frac_thr) and (cv > cv_thr)
    out = []
    start = None
    for i in range(n):
        if flags[i] and start is None:
            start = i
        elif not flags[i] and start is not None:
            t0, t1 = t[start], t[i]
            if t1 - t0 >= min_dur:
                out.append({"kind": "no_organized", "start_s": float(t0),
                            "end_s": float(t1), "severity": "critical",
                            "detail": f"Ritmo no organizado ({t1-t0:.0f} s, "
                                      f"corr {np.median(corr[start:i]):.2f})",
                            "beats": []})
            start = None
    if start is not None and t[-1] - t[start] >= min_dur:
        out.append({"kind": "no_organized", "start_s": float(t[start]),
                    "end_s": float(t[-1]), "severity": "critical",
                    "detail": f"Ritmo no organizado ({t[-1]-t[start]:.0f} s)",
                    "beats": []})
    return out


def gap_events(beats, duration_s: float, min_gap: float = 3.0) -> list[dict]:
    """Ausencia de QRS detectado (posible asistolia o corte de señal)."""
    out = []
    t = [0.0] + [b.t for b in beats] + [max(0.0, duration_s)]
    for a, b in zip(t[:-1], t[1:]):
        gap = b - a
        if gap < min_gap:
            continue
        if a == 0.0 and gap < 5.0:
            continue
        out.append({"kind": "asystole", "start_s": float(a), "end_s": float(b),
                    "severity": "critical" if gap >= 5.0 else "warn",
                    "detail": f"Sin QRS durante {gap:.1f} s", "beats": []})
    return out


def pause_counts_by_window(events, start_time, day=None, night=None) -> dict | None:
    """Pausas informadas por umbral (2,0 / 2,5 / 3,0 s), de día y de noche por separado.

    La hora de una pausa es la de su inicio, en hora local del examen
    (`start_time`). Las ventanas salen de los criterios si no se pasan. Las pausas
    de las horas entre ventanas (p. ej. 20–23 h) no cuentan en ninguna: van en
    `entre`. Sin `start_time` no hay reparto y devuelve None.
    """
    from .hrv import beat_clock, hour_in_window, windows_from_criteria

    if start_time is None:
        return None
    if day is None or night is None:
        crit_day, crit_night = windows_from_criteria()
        day = crit_day if day is None else day
        night = crit_night if night is None else night
    out: dict = {"dia": {f"{t:.1f}": 0 for t in PAUSE_THRESHOLDS_S},
                 "noche": {f"{t:.1f}": 0 for t in PAUSE_THRESHOLDS_S}, "entre": 0}
    for ev in events:
        if not ev.get("reported", True):
            continue
        hour = beat_clock(start_time, float(ev["start_s"])).hour
        if hour_in_window(hour, *day):
            key = "dia"
        elif hour_in_window(hour, *night):
            key = "noche"
        else:
            out["entre"] += 1
            continue
        for t in PAUSE_THRESHOLDS_S:
            if float(ev["duration_s"]) >= t:
                out[key][f"{t:.1f}"] += 1
    return out


def hourly_summary(beats, af_probs=None, duration_s: float = 0.0,
                   pause_threshold: float = 2.0) -> list[dict]:
    """Tabla por hora: FC, PVC, PAC, FA y pausas.

    Es una de las secciones que todo informe Holter comercial incluye y que
    permite ver el patrón diurno/nocturno.
    """
    if not beats:
        return []
    t = np.array([b.t for b in beats], dtype=float)
    cls = np.array([b.cls for b in beats])
    rr = np.array([b.rr_prev for b in beats], dtype=float)
    hr = np.where(rr > 0.2, 60.0 / np.maximum(rr, 1e-6), np.nan)
    ventana = float(max(duration_s, float(t[-1]) + 1.0))
    n_hour = max(1, int(np.ceil(ventana / 3600.0)))
    rows = []
    valid = np.zeros(len(beats), dtype=bool)
    valid[1:] = (cls[1:] == "N") & (cls[:-1] == "N")
    for h in range(n_hour):
        a, b = h * 3600.0, (h + 1) * 3600.0
        m = (t >= a) & (t < b)
        n = int(m.sum())
        if n == 0:
            rows.append({"hour": h, "n": 0, "hr_mean": None, "hr_min": None,
                         "hr_max": None, "pvc": 0, "pac": 0, "af_pct": 0.0,
                         "pauses": 0, "artifact_pct": 0.0})
            continue
        hh = hr[m]
        hh = hh[np.isfinite(hh)]
        af = float(np.mean(af_probs[m])) if af_probs is not None else 0.0
        tot = float(np.nansum(rr[m]))
        art = float(np.nansum(rr[m & ~valid]))
        rows.append({
            "hour": h, "n": n,
            "hr_mean": round(float(np.mean(hh)), 1) if hh.size else None,
            "hr_min": round(float(np.percentile(hh, 1)), 1) if hh.size else None,
            "hr_max": round(float(np.percentile(hh, 99)), 1) if hh.size else None,
            "pvc": int((cls[m] == "V").sum()),
            "pac": int((cls[m] == "S").sum()),
            "af_pct": round(100.0 * af, 1),
            "pauses": int(np.sum(rr[m] >= pause_threshold)),
            "artifact_pct": round(100.0 * art / tot, 1) if tot > 0 else 0.0,
        })
    return rows


def ventricular_events(beats) -> dict:
    """Parejas, tríos y R-on-T de origen ventricular."""
    couplets = triplets = 0
    runs = []
    rates: list[float] = []
    run: list = []
    def _close_run(items: list) -> None:
        if len(items) >= 2:
            runs.append(len(items))
            rrs = [b.rr_prev for b in items[1:] if b.rr_prev > 0.2]
            if rrs:
                rates.append(60.0 / (sum(rrs) / len(rrs)))
    for b in beats:
        if b.cls == "V":
            run.append(b)
        else:
            _close_run(run)
            run = []
    _close_run(run)
    for r in runs:
        if r == 2:
            couplets += 1
        elif r == 3:
            triplets += 1
    r_on_t = 0
    coupling: list[float] = []
    for i, b in enumerate(beats):
        if b.cls != "V" or i == 0:
            continue
        prev = beats[i - 1]
        # R-on-T: el QRS ventricular cae sobre la T precedente. El intervalo
        # de acoplamiento tiene que ser menor que el QT (≈0,40 s); en el
        # bigeminismo habitual el acoplamiento es mayor y no cuenta.
        if prev.cls != "V" and 0.2 < b.rr_prev < 2.5:
            coupling.append(b.rr_prev * 1000.0)
        if 0 < b.rr_prev < 0.40 and prev.cls != "V":
            r_on_t += 1
    coupling.sort()
    normals = [b.rr_prev for b in beats if b.cls != "V" and 0.2 < b.rr_prev < 2.5]
    local_rr = sorted(normals)[len(normals) // 2] if normals else 0.8
    interpolated = sum(
        1 for i, b in enumerate(beats)
        if b.cls == "V" and 0 < b.rr_prev < 2.5 and 0 < b.rr_next < 2.5
        and (b.rr_prev + b.rr_next) < 1.25 * 2 * local_rr
    )
    trigeminy = sum(
        1 for i in range(2, len(beats))
        if beats[i].cls == "V" and beats[i - 1].cls != "V" and beats[i - 2].cls != "V"
    )
    isolated = sum(1 for i, b in enumerate(beats) if b.cls == "V"
                   and (i == 0 or beats[i - 1].cls != "V")
                   and (i + 1 == len(beats) or beats[i + 1].cls != "V"))
    return {"couplets": couplets, "triplets": triplets, "runs": runs,
            "r_on_t": r_on_t, "isolated": isolated, "trigeminy": trigeminy,
            "interpolated": interpolated,
            "fastest": round(max(rates), 1) if rates else None,
            "slowest": round(min(rates), 1) if rates else None,
            "coupling_median_ms": round(coupling[len(coupling) // 2], 1) if coupling else None,
            "coupling_min_ms": round(coupling[0], 1) if coupling else None,
            "coupling_max_ms": round(coupling[-1], 1) if coupling else None}


def supraventricular_events(beats) -> dict:
    """Parejas y rachas de latidos supraventriculares consecutivos."""
    runs: list[int] = []
    rates: list[float] = []
    run = 0
    run_rrs: list[float] = []
    def _close_sv() -> None:
        if run >= 2:
            runs.append(run)
            if run_rrs:
                rates.append(60.0 / (sum(run_rrs) / len(run_rrs)))
    for b in beats:
        if b.cls == "S":
            run += 1
            if b.rr_prev > 0.2:
                run_rrs.append(b.rr_prev)
        elif run:
            _close_sv()
            run = 0
            run_rrs = []
    _close_sv()
    couplets = sum(1 for r in runs if r == 2)
    triplets = sum(1 for r in runs if r == 3)
    return {"couplets": couplets, "triplets": triplets, "runs": runs,
            "longest": max(runs) if runs else 0,
            "fastest": round(max(rates), 1) if rates else None,
            "slowest": round(min(rates), 1) if rates else None}


def lown_grade(pvc_count: int, pvc_per_hour: float, multifocal: int,
               couplets: int, triplets: int, r_on_t: int) -> dict:
    """Gradación de Lown (0-V) del ectopismo ventricular."""
    if pvc_count == 0:
        return {"grade": 0, "label": "0 — sin ectopia ventricular"}
    g = 1
    if pvc_per_hour >= 30:
        g = 2
    if multifocal >= 2:
        g = 3
    if couplets > 0:
        g = 4
    if triplets > 0:
        g = 4
    if r_on_t > 0:
        g = 5
    labels = {
        0: "0 — sin ectopia ventricular",
        1: "I — PVC aislados < 30/h",
        2: "II — PVC ≥ 30/h",
        3: "III — PVC multifocal",
        4: "IV — parejas o tríos",
        5: "V — R-on-T",
    }
    detail = []
    if pvc_per_hour >= 30:
        detail.append(f"{pvc_per_hour:.0f} PVC/h")
    if multifocal >= 2:
        detail.append(f"{multifocal} morfologías")
    if couplets:
        detail.append(f"{couplets} parejas")
    if triplets:
        detail.append(f"{triplets} tríos")
    if r_on_t:
        detail.append(f"{r_on_t} R-on-T")
    return {"grade": g, "label": labels.get(g, str(g)),
            "detail": ", ".join(detail)}


def cluster_windows(wins: np.ndarray, thr: float = 0.85,
                    min_frac: float = 0.05) -> np.ndarray:
    """Agrupa latidos por morfología (voraz por correlación).

    Devuelve etiquetas: 0..k-1 para grupos con al menos min_frac de los
    latidos, y -1 para los sueltos.
    """
    n = len(wins)
    if n == 0:
        return np.zeros(0, dtype=int)
    w = np.asarray(wins, dtype=np.float64)
    w = w - w.mean(axis=1, keepdims=True)
    w = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-9)
    sim = w @ w.T
    labels = -np.ones(n, dtype=int)
    k = 0
    min_count = max(3, int(min_frac * n))
    for i in range(n):
        if labels[i] != -1:
            continue
        members = (sim[i] >= thr) & (labels == -1)
        if members.sum() >= min_count:
            labels[members] = k
            k += 1
    return labels


def count_pvc_morphologies(wins: np.ndarray, v_idx: np.ndarray,
                           thr: float = 0.85, min_frac: float = 0.10) -> int:
    """Número de morfologías ventriculares distintas (unifocal vs multifocal).

    Agrupamiento voraz por correlación sobre las ventanas de latido.
    """
    if len(v_idx) < 8:
        return int(len(v_idx) > 0)
    w = np.asarray(wins, dtype=np.float32)[v_idx].astype(np.float64)
    w = w - w.mean(axis=1, keepdims=True)
    n = np.linalg.norm(w, axis=1, keepdims=True)
    w = w / np.maximum(n, 1e-9)
    sim = w @ w.T  # matriz de correlación
    assigned = np.zeros(len(w), dtype=bool)
    clusters = 0
    min_count = max(3, int(min_frac * len(w)))
    for i in range(len(w)):
        if assigned[i]:
            continue
        members = (~assigned) & (sim[i] >= thr)
        if members.sum() >= min_count:
            clusters += 1
            assigned |= members
        else:
            assigned[i] = True
    return max(1, clusters)


# ---------------------------------------------------------------------------
# Polaridad del QRS y tramos sin señal
# ---------------------------------------------------------------------------
# Polaridad esperada del QRS en el adulto normal: +1 positiva, -1 negativa,
# 0 sin expectativa (eje variable o zona de transición). Solo sirve para
# avisar: nunca se invierte un canal de forma automática. V2/V3/V4 quedan
# fuera a propósito porque la transición precordial varía entre pacientes.
LEAD_POLARITY = {
    "I": 1, "II": 1, "III": 0, "AVR": -1, "AVL": 0, "AVF": 0,
    "V1": -1, "V2": 0, "V3": 0, "V4": 0, "V5": 1, "V6": 1,
}
POLARITY_WARN_FRAC = 0.80


def qrs_polarity(samples, fs: float, idx, leads=None, max_beats: int = 400,
                 half_ms: float = 60.0, min_uv: float = 50.0,
                 warn_frac: float = POLARITY_WARN_FRAC) -> list[dict]:
    """Polaridad dominante del QRS en cada canal frente a la esperada.

    Para cada latido se compara el pico positivo con el negativo de la
    ventana de QRS respecto de la mediana local (los canales de Holter
    pueden traer línea de base desplazada). Si la polaridad dominante es la
    opuesta a la esperada para esa derivación, se marca `invert_suggested`:
    es solo un aviso, la inversión la decide el revisor.

    `samples` va en mV y `idx` son las muestras R detectadas. No se convierte
    el registro completo a float64: solo se copian las ventanas de QRS.
    """
    x = np.asarray(samples)
    if x.ndim == 1:
        x = x[:, None]
    n, n_leads = x.shape
    idx = np.asarray(idx, dtype=np.int64).ravel()
    idx = idx[(idx >= 0) & (idx < n)]
    if idx.size == 0 or fs <= 0:
        return []
    if idx.size > max_beats:
        idx = idx[np.linspace(0, idx.size - 1, int(max_beats)).astype(int)]
    half = max(1, int(round(half_ms * 1e-3 * fs)))
    raw_names = [str(n) for n in leads] if leads else [""] * n_leads
    canon = map_leads(raw_names) if leads else [""] * n_leads
    pos = np.zeros(n_leads)
    tot = np.zeros(n_leads)
    for p in idx:
        a, b = max(0, int(p) - half), min(n, int(p) + half)
        if b - a < 3:
            continue
        seg = np.asarray(x[a:b, :], dtype=np.float64)
        base = np.median(seg, axis=0)
        up = (seg - base).max(axis=0)
        down = (base - seg).max(axis=0)
        ok = np.maximum(up, down) >= min_uv / 1000.0
        pos[ok & (up > down)] += 1
        tot[ok] += 1

    out: list[dict] = []
    min_meas = max(5, int(0.25 * idx.size))
    for j in range(n_leads):
        canonical = canon[j] if j < len(canon) else ""
        shown = raw_names[j] if j < len(raw_names) and raw_names[j] else (
            canonical or f"L{j + 1}")
        if tot[j] < min_meas:
            continue
        frac_pos = float(pos[j] / tot[j])
        dominant = max(frac_pos, 1.0 - frac_pos)
        observed = "+" if frac_pos >= 0.5 else "-"
        expected = LEAD_POLARITY.get(canonical, 0)
        need = "+" if expected > 0 else ("-" if expected < 0 else "")
        inverted = bool(need and observed != need and dominant >= warn_frac)
        label = "positivo" if observed == "+" else "negativo"
        if not need:
            detail = f"{shown}: QRS {label} ({100 * dominant:.0f} %), sin polaridad esperada"
        elif inverted:
            detail = (f"{shown}: QRS {label} en {100 * dominant:.0f} % de {int(tot[j])} latidos"
                      f" (se espera {'positivo' if need == '+' else 'negativo'}):"
                      " posible canal invertido")
        else:
            detail = f"{shown}: QRS {label}, acorde a lo esperado"
        out.append({
            "lead": int(j), "name": shown, "lead_id": canonical,
            "n": int(tot[j]), "frac_positive": round(frac_pos, 3),
            "observed": observed, "expected": need,
            "invert_suggested": inverted, "detail": detail,
        })
    return out


def no_signal_spans(samples, fs: float, win_s: float = 1.0, flat_uv: float = 15.0,
                    rail_uv: float = 4500.0, min_dur: float = 2.0,
                    gap_s: float = 0.5) -> list[tuple[float, float]]:
    """Intervalos en los que ningún canal tiene señal utilizable.

    Un canal cuenta como "sin señal" si está plano (pico a pico < `flat_uv`)
    o desconectado/saturado (|x| ≥ `rail_uv`). El tramo se informa solo si
    **todos** los canales están así: con un canal sano el ritmo es analizable.
    Los bloques se recorren por tandas para no copiar el registro entero.
    """
    x = np.asarray(samples)
    if x.ndim == 1:
        x = x[:, None]
    n, n_leads = x.shape
    if fs <= 0 or n <= 0:
        return []
    k = max(1, int(round(win_s * fs)))
    m = n // k
    if m < 1:
        return []
    blk = k / float(fs)
    all_dead = np.zeros(m, dtype=bool)
    step = max(1, int(4_000_000 / max(1, k * n_leads)))  # ~32 MB por tanda
    for s0 in range(0, m, step):
        s1 = min(m, s0 + step)
        blocks = np.asarray(x[s0 * k: s1 * k], dtype=np.float32).reshape(
            s1 - s0, k, n_leads)
        ptp = blocks.max(axis=1) - blocks.min(axis=1)
        rail = np.abs(blocks).max(axis=1)
        dead = (ptp < flat_uv / 1000.0) | (rail >= rail_uv / 1000.0)
        all_dead[s0:s1] = np.all(dead, axis=1)
    if not all_dead.any():
        return []
    max_gap = max(1, int(round(gap_s / blk)))
    spans: list[tuple[float, float]] = []
    i = 0
    while i < m:
        if not all_dead[i]:
            i += 1
            continue
        j = i
        while j + 1 < m:
            if all_dead[j + 1]:
                j += 1
                continue
            nxt = np.flatnonzero(all_dead[j + 1: j + 1 + max_gap])
            if nxt.size == 0:
                break
            j = j + 1 + int(nxt[-1])
        a, b = i * blk, min((j + 1) * blk, n / float(fs))
        if b - a >= min_dur:
            spans.append((float(a), float(b)))
        i = j + 1
    return spans


def no_signal_events(samples, fs: float, **kw) -> list[dict]:
    """Evento `no_signal` por cada tramo sin señal en todos los canales."""
    out = []
    for a, b in no_signal_spans(samples, fs, **kw):
        dur = b - a
        out.append({
            "kind": "no_signal", "start_s": a, "end_s": b,
            "severity": "critical" if dur >= 10.0 else "warn",
            "detail": f"Sin señal en todos los canales ({dur:.1f} s)",
            "beats": [],
        })
    return out


def hourly_rr_stats(beats, duration_s: float, bin_s: float = 3600.0):
    """Media de FC por intervalo (para la gráfica de tendencia)."""
    if not beats:
        return np.array([]), np.array([])
    t = np.array([b.t for b in beats], dtype=float)
    rr = np.array([b.rr_prev for b in beats], dtype=float)
    hr = np.where(rr > 0, 60.0 / np.maximum(rr, 1e-6), np.nan)
    edges = np.arange(0, max(duration_s, t.max()) + bin_s, bin_s)
    centers, means = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t >= a) & (t < b)
        vals = hr[m]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            centers.append((a + b) / 2)
            means.append(float(np.median(vals)))
    return np.array(centers), np.array(means)


# ---------------------------------------------------------------------------
# Fibrilación auricular por episodios (T23)
# ---------------------------------------------------------------------------
# Ventana deslizante en latidos (el enunciado pide 30-60). Con 30 latidos a
# 60-100 lpm la ventana dura 18-30 s: es el mínimo para que la entropía
# muestral sea estable y para no marcar episodios de < 30 s.
AF_WINDOW_BEATS = 30
AF_SLIDE_BEATS = 3
AF_MIN_BEATS = 20
# Umbrales con histéresis sobre la puntuación de 0 a 1.
AF_ON = 0.55
AF_OFF = 0.15
# Suavizado (mediana móvil en latidos) de la puntuación antes de la histéresis.
AF_SMOOTH_BEATS = 21
# Un episodio clínico de FA dura >= 30 s y se une a otro si el hueco es < 10 s.
AF_MIN_EPISODE_S = 30.0
AF_MERGE_GAP_S = 10.0
# Intervalos RR que no describen el ritmo (pausa, extrasístole, ruido).
AF_MIN_RR_S = 0.25
AF_MAX_RR_S = 2.5
# Ectopia: con bigeminia (V 50 %) o S muy frecuentes el RR no es interpretable.
AF_V_MAX_FRAC = 0.15
AF_ECTOPIC_MAX_FRAC = 0.40
AF_BAD_MAX_FRAC = 0.30
# Tolerancia de la entropía muestral (fracción de la mediana del RR).
AF_SAMPEN_R_REL = 0.10
# COSEn: tolerancia fija de 50 ms como en Lake y Moorman (2011).
AF_COSEN_R_S = 0.05
# Apoyo de la onda P y de la actividad f (no deciden solos).
AF_P_ABSENT_MAX = 0.20
AF_P_PRESENT_MIN = 0.60
AF_P_MIN_EVALUABLE = 0.30
# La actividad f refuerza la puntuación hasta un 30 % (ver ``_apply_supports``).
AF_FWAVE_GAIN = 0.30
# Veto por onda P (T10): un episodio con P presente en más de la mitad de los
# latidos evaluables no es FA (ver ``_p_veto``).
AF_P_VETO_FRAC = 0.50
AF_P_VETO_MIN_EVAL = 0.30


def _ramp(x: float, lo: float, hi: float, down: bool = False) -> float:
    """Lleva ``x`` a 0..1 entre ``lo`` y ``hi`` (``down``: alto en ``lo``)."""
    if not np.isfinite(x):
        return 0.0
    v = (float(x) - lo) / (hi - lo) if hi != lo else float(x >= lo)
    v = float(np.clip(v, 0.0, 1.0))
    return 1.0 - v if down else v


def af_features(rr: np.ndarray) -> dict:
    """Características de irregularidad de una ventana de RR.

    ``sampen``
        entropía muestral (m=2, r=0,10·mediana): sube con el desorden del RR.
    ``cosen``
        coeficiente de entropía muestral (Lake y Moorman): SampEn + ln(2r) −
        ln(RMSSD) con r = 50 ms. En este uso sobre RR **baja** cuando el ritmo
        es irregular (el término 1/RMSSD domina), así que es evidencia de FA
        cuando es bajo.
    ``af_evidence``
        AFEvidence del diagrama de Poincaré de las ΔRR: fracción de puntos
        fuera de la caja central (±max(40 ms, 15 % de la mediana)) menos el
        doble de la fracción de puntos en la caja. > 0 → irregular.
    ``rmssd_rel`` y ``sd_ratio``
        RMSSD/mediana y SD1/SD2 de Poincaré (en FA SD1 tiende a SD2).
    ``pnn50``
        fracción de |ΔRR| > 50 ms.
    """
    rr = np.asarray(rr, dtype=np.float64).ravel()
    rr = rr[np.isfinite(rr) & (rr > AF_MIN_RR_S) & (rr < AF_MAX_RR_S)]
    out = {"n": int(rr.size), "ok": False, "rr_median": 0.0, "rmssd_rel": 0.0,
           "sampen": 0.0, "cosen": 0.0, "af_evidence": -2.0, "pnn50": 0.0,
           "sd1": 0.0, "sd2": 0.0, "sd_ratio": 0.0, "poincare_area": 0.0,
           "mad_drr_rel": 0.0, "ectopic_frac": 0.0}
    if rr.size < AF_MIN_BEATS:
        return out
    rr, ectopic_frac = _drop_ectopic_pairs(rr)
    out["ectopic_frac"] = float(ectopic_frac)
    if rr.size < AF_MIN_BEATS:
        return out
    med = float(np.median(rr))
    drr = np.diff(rr)
    if drr.size < AF_MIN_BEATS - 1:
        return out
    rmssd = float(np.sqrt(np.mean(drr ** 2)))
    sdnn = float(rr.std(ddof=1))
    sd1 = float(np.sqrt(0.5) * drr.std(ddof=1)) if drr.size > 1 else 0.0
    sd2 = float(np.sqrt(max(0.0, 2.0 * sdnn ** 2 - 0.5 * drr.std(ddof=1) ** 2)))
    sampen = _sample_entropy(rr, m=2, r=AF_SAMPEN_R_REL * med)
    cosen = sampen + float(np.log(2.0 * AF_COSEN_R_S)) - float(np.log(max(rmssd, 1e-3)))
    box = max(0.04, 0.15 * med)
    a = np.abs(drr[:-1])
    b = np.abs(drr[1:])
    if a.size:
        inside = int(np.sum((a <= box) & (b <= box)))
        outside = int(a.size - inside)
        af_ev = (outside - 2.0 * inside) / float(a.size)
    else:
        af_ev = 0.0
    out.update({
        "ok": True, "rr_median": med, "rmssd_rel": rmssd / max(med, 1e-6),
        "sampen": float(sampen), "cosen": float(cosen), "af_evidence": float(af_ev),
        "pnn50": float(np.mean(np.abs(drr) > 0.050)),
        "sd1": sd1, "sd2": sd2,
        "sd_ratio": sd1 / sd2 if sd2 > 1e-6 else 0.0,
        "poincare_area": float(np.pi * sd1 * sd2),
        "mad_drr_rel": float(np.median(np.abs(drr)) / max(med, 1e-6)),
    })
    return out


def _drop_ectopic_pairs(rr: np.ndarray) -> tuple[np.ndarray, float]:
    """Quita los pares corto-largo típicos de extrasístole con pausa compensatoria.

    Una extrasístole adelanta un latido y atrasa el siguiente: el par
    ``RR < 0,85·mediana`` seguido de ``RR > 1,10·mediana`` explica la
    irregularidad sin que haya fibrilación. Se descartan los dos intervalos
    (una sola pasada; en FA el patrón es aleatorio y sobrevive la mayoría).
    """
    if rr.size < 4:
        return rr, 0.0
    med = float(np.median(rr))
    corto = rr[:-1] < 0.85 * med
    largo = rr[1:] > 1.10 * med
    idx = np.nonzero(corto & largo)[0]
    if idx.size == 0:
        return rr, 0.0
    keep = np.ones(rr.size, dtype=bool)
    keep[idx] = False
    keep[np.minimum(idx + 1, rr.size - 1)] = False
    frac = float(1.0 - keep.mean())
    return rr[keep], frac


#: Peso de cada voto en la puntuación de FA. Los dos últimos (Poincaré SD1/SD2
#: y COSEn) separan mal en AFDB (mediana 0,99 frente a 0,92 y 0,73 frente a
#: 1,58) y solo cuentan como desempate.
AF_VOTE_WEIGHTS = {
    "rmssd_rel": 1.0, "sampen": 1.0, "af_evidence": 1.0, "mad_drr_rel": 1.0,
    "pnn50": 0.5, "sd_ratio": 0.25, "cosen": 0.25,
}


def af_window_score(feat: dict) -> float:
    """Puntuación de 0 a 1 de una ventana a partir de sus características.

    Es la media ponderada de votos con umbrales fijados *a priori* a partir de
    los valores publicados para FA y para ritmo sinusal (no se ajustaron sobre
    el conjunto de validación):

    * RMSSD/mediana: 0,08 (sinusal) → 0,16 (FA).
    * Entropía muestral: 0,80 → 1,20.
    * pNN50: 0,40 → 0,70.
    * SD1/SD2 de Poincaré: 0,60 → 0,90.
    * AFEvidence: 0,00 → 0,40.
    * COSEn (baja en FA): 1,60 → 1,00.
    * MAD de ΔRR / mediana (robusta a latidos aislados): 0,04 → 0,10.
    """
    if not feat.get("ok"):
        return float("nan")
    votes = {
        "rmssd_rel": _ramp(feat["rmssd_rel"], 0.08, 0.16),
        "sampen": _ramp(feat["sampen"], 0.80, 1.20),
        "pnn50": _ramp(feat["pnn50"], 0.40, 0.70),
        "sd_ratio": _ramp(feat["sd_ratio"], 0.60, 0.90),
        "af_evidence": _ramp(feat["af_evidence"], 0.00, 0.40),
        "cosen": _ramp(feat["cosen"], 1.60, 1.00, down=True),
        "mad_drr_rel": _ramp(feat["mad_drr_rel"], 0.04, 0.10),
    }
    total = sum(AF_VOTE_WEIGHTS.values())
    return float(sum(AF_VOTE_WEIGHTS[k] * v for k, v in votes.items()) / total)


def _as_class_array(n: int, classes) -> np.ndarray:
    if classes is None:
        return np.full(n, "N", dtype="<U1")
    cls = np.asarray(list(classes) if not isinstance(classes, np.ndarray) else classes)
    if cls.size != n:
        raise ValueError(f"clases ({cls.size}) y tiempos ({n}) no coinciden")
    return np.char.upper(cls.astype(str))


def _fwave_per_beat(n: int, fwave) -> np.ndarray | None:
    """Convierte el apoyo auricular de T11 en un valor 0/1 por latido."""
    if fwave is None:
        return None
    if isinstance(fwave, dict):
        conc = float(fwave.get("conc_med", 0.0) or 0.0)
        dom = float(fwave.get("dom_hz_med", float("nan")))
        ok = str(fwave.get("quality", "sin datos")) == "ok"
        support = bool(ok and conc >= 0.40 and 3.5 <= dom <= 6.5)
        return np.full(n, 1.0 if support else 0.0)
    arr = np.asarray(fwave, dtype=float).ravel()
    if arr.size != n:
        raise ValueError(f"apoyo f ({arr.size}) y tiempos ({n}) no coinciden")
    return arr


def af_scores(times, classes=None, unreadable=None, *, p_present=None,
              p_evaluable=None, fwave=None, window: int = AF_WINDOW_BEATS,
              step: int = AF_SLIDE_BEATS, min_beats: int = AF_MIN_BEATS,
              smooth: int = AF_SMOOTH_BEATS) -> np.ndarray:
    """Puntuación de FA por latido (0..1) con ventanas deslizantes de ``window``.

    La puntuación de cada ventana se reparte entre sus latidos y se suaviza con
    una mediana de ``smooth`` latidos, para que un tramo de RR regularizado (o
    ruidoso) dentro de una FA no parta el episodio en dos.

    Devuelve ``NaN`` en los latidos sin ninguna ventana interpretable (tramo
    ilegible, bigeminia, S muy frecuentes o demasiados RR fuera de rango).
    """
    t = np.asarray(times, dtype=np.float64).ravel()
    n = t.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n < max(min_beats, 4):
        return out
    cls = _as_class_array(n, classes)
    skip = np.isin(cls, ["V", "Q", "F"])
    if unreadable is not None:
        skip |= np.asarray(unreadable, dtype=bool).ravel()
    if p_present is not None:
        p_present = np.asarray(p_present, dtype=float).ravel()
        if p_evaluable is None:
            p_evaluable = np.isfinite(p_present)
        else:
            p_evaluable = np.asarray(p_evaluable, dtype=bool).ravel()
    fw = _fwave_per_beat(n, fwave)

    rr = np.full(n, np.nan)
    rr[1:] = np.diff(t)
    rr_ok = (np.isfinite(rr) & (rr > AF_MIN_RR_S) & (rr < AF_MAX_RR_S))
    rr_ok[1:] &= ~skip[1:] & ~skip[:-1]
    is_v = cls == "V"
    is_ect = np.isin(cls, ["V", "S"])

    acc = np.zeros(n)
    cnt = np.zeros(n)
    for end in range(min_beats, n + 1, max(1, step)):
        a = max(0, end - window)
        frac_bad = float(np.mean(skip[a:end]))
        if frac_bad > AF_BAD_MAX_FRAC:
            continue
        if float(np.mean(is_v[a:end])) > AF_V_MAX_FRAC:
            continue
        if float(np.mean(is_ect[a:end])) > AF_ECTOPIC_MAX_FRAC:
            continue
        seg = rr[a:end]
        feat = af_features(seg[rr_ok[a:end]])
        score = af_window_score(feat)
        if not np.isfinite(score):
            continue
        acc[a:end] += float(np.clip(score, 0.0, 1.0))
        cnt[a:end] += 1.0
    seen = cnt > 0
    out[seen] = acc[seen] / cnt[seen]
    if smooth > 1:
        out = _nan_median_filter(out, int(smooth))
    return _apply_supports(out, fw, int(max(1, smooth)))


def _apply_supports(score: np.ndarray, fw, smooth: int) -> np.ndarray:
    """Refuerza la puntuación con la actividad f del residuo (T11).

    Solo la actividad auricular suma (hasta ``AF_FWAVE_GAIN``), y de forma
    continua sobre la puntuación ya suavizada. La onda P **no** se resta aquí:
    se usa como veto por episodio (``_p_veto``), porque atenuar la puntuación
    dentro de una FA real la parte en trozos y multiplica los falsos episodios.
    """
    if fw is None:
        return score
    fs_ = _nan_median_filter(np.asarray(fw, dtype=float).ravel(), smooth)
    gain = np.nan_to_num(fs_, nan=0.0)
    return np.where(np.isfinite(score),
                    np.clip(score * (1.0 + AF_FWAVE_GAIN * gain), 0.0, 1.0), score)


def _p_veto(segments: list[dict], times: np.ndarray, p_present, p_evaluable,
            min_eval: float = AF_P_VETO_MIN_EVAL,
            frac: float = AF_P_VETO_FRAC) -> tuple[list[dict], int]:
    """Descarta episodios con onda P presente (no son FA).

    Es un veto por episodio, no un descuento por latido: si dentro del episodio
    hay P en más de ``frac`` de los latidos evaluables (y se evaluó al menos
    ``min_eval`` de ellos), el episodio se descarta. Así el apoyo de T10 solo
    puede quitar falsos positivos, nunca partir un episodio verdadero.
    """
    if p_present is None or p_evaluable is None or not segments:
        return segments, 0
    pres = np.asarray(p_present, dtype=float).ravel()
    ev = np.asarray(p_evaluable, dtype=bool).ravel()
    keep = []
    dropped = 0
    for seg in segments:
        m = (times >= seg["start_s"]) & (times <= seg["end_s"])
        usable = ev[m] & np.isfinite(pres[m])
        if usable.size and float(np.mean(ev[m])) >= min_eval:
            p_frac = float(np.mean(pres[m][usable] > 0.5))
            seg = dict(seg, p_frac=round(p_frac, 3))
            if p_frac >= frac:
                dropped += 1
                continue
        keep.append(seg)
    return keep, dropped


def _nan_median_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Mediana móvil de ``k`` muestras que ignora los ``NaN``."""
    if k <= 1 or x.size == 0:
        return x
    pad = k // 2
    v = np.pad(np.asarray(x, dtype=float), pad, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(v, k)
    finite = np.isfinite(win)
    n_ok = finite.sum(axis=1)
    # los no finitos van al final (+inf) para que el índice medio caiga siempre
    # sobre un valor real
    srt = np.sort(np.where(finite, win, np.inf), axis=1)
    pick = np.clip((n_ok - 1) // 2, 0, k - 1)
    med = np.take_along_axis(srt, pick[:, None], axis=1)[:, 0]
    return np.where(n_ok > 0, med, np.nan)


def _support_delta(p_present, p_evaluable, fw, a: int, end: int) -> float:
    """Ajuste (±) de la puntuación por la onda P (T10) y la actividad f (T11)."""
    delta = 0.0
    if p_present is not None and p_evaluable is not None:
        usable = p_evaluable[a:end]
        if usable.size and float(np.mean(usable)) >= AF_P_MIN_EVALUABLE:
            frac = float(np.mean(p_present[a:end][usable] > 0.5))
            if frac <= AF_P_ABSENT_MAX:
                delta += AF_P_BONUS
            elif frac >= AF_P_PRESENT_MIN:
                delta -= AF_P_PENALTY
    if fw is not None and float(np.mean(fw[a:end])) >= 0.5:
        delta += AF_FWAVE_BONUS
    return delta


def af_segments(times, scores, *, on: float = AF_ON, off: float = AF_OFF,
                min_s: float = AF_MIN_EPISODE_S,
                merge_gap_s: float = AF_MERGE_GAP_S) -> list[dict]:
    """Convierte la puntuación por latido en episodios de FA.

    Histéresis ``on``/``off`` para no cortar un episodio por un latido suelto,
    unión de huecos menores que ``merge_gap_s`` y descarte de los tramos de
    menos de ``min_s`` segundos.
    """
    t = np.asarray(times, dtype=np.float64).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if t.size != s.size:
        raise ValueError(f"puntuaciones ({s.size}) y tiempos ({t.size}) no coinciden")
    if t.size == 0:
        return []
    raw: list[list[float]] = []
    state = False
    start = 0.0
    last_high = 0.0
    for i in range(t.size):
        v = s[i]
        if np.isfinite(v) and v >= on:
            if not state:
                state, start = True, float(t[i])
            last_high = float(t[i])
        elif state and (not np.isfinite(v) or v < off):
            state = False
            if last_high - start >= min_s:
                raw.append([start, last_high])
    if state and last_high - start >= min_s:
        raw.append([start, last_high])
    merged: list[list[float]] = []
    for a, b in raw:
        if merged and a - merged[-1][1] < merge_gap_s:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        if b - a >= min_s:
            mask = (t >= a) & (t <= b) & np.isfinite(s)
            out.append({
                "start_s": round(a, 2), "end_s": round(b, 2),
                "duration_s": round(b - a, 2), "label": "AF",
                "score": round(float(np.mean(s[mask])) if mask.any() else 0.0, 3),
            })
    return out


def af_burden(segments: list[dict], duration_s: float) -> dict:
    """Carga de FA (%) y episodio más largo a partir de los episodios."""
    total = float(sum(max(0.0, seg["end_s"] - seg["start_s"]) for seg in segments))
    longest = float(max((seg["end_s"] - seg["start_s"] for seg in segments), default=0.0))
    dur = float(duration_s) if duration_s and duration_s > 0 else 0.0
    return {
        "n_episodes": len(segments),
        "total_s": round(total, 1),
        "longest_s": round(longest, 1),
        "analyzed_s": round(dur, 1),
        "burden_pct": round(100.0 * total / dur, 2) if dur > 0 else 0.0,
    }


def af_analysis(times, classes=None, duration_s: float = 0.0, *, unreadable=None,
                p_present=None, p_evaluable=None, fwave=None,
                window: int = AF_WINDOW_BEATS, step: int = AF_SLIDE_BEATS,
                on: float = AF_ON, off: float = AF_OFF,
                min_s: float = AF_MIN_EPISODE_S,
                merge_gap_s: float = AF_MERGE_GAP_S) -> dict:
    """Detector de FA completo: puntuación por latido, episodios y carga."""
    t = np.asarray(times, dtype=np.float64).ravel()
    scores = af_scores(t, classes, unreadable, p_present=p_present,
                       p_evaluable=p_evaluable, fwave=fwave,
                       window=window, step=step)
    segs = af_segments(t, scores, on=on, off=off, min_s=min_s,
                       merge_gap_s=merge_gap_s)
    segs, n_veto = _p_veto(segs, t, p_present, p_evaluable)
    dur = float(duration_s)
    if dur <= 0 and t.size:
        dur = float(t[-1] - t[0])
    res = af_burden(segs, dur)
    res["n_p_veto"] = n_veto
    res["episodes"] = segs
    res["scores"] = scores
    usable = np.isfinite(scores)
    res["analyzed_pct"] = round(100.0 * float(usable.mean()), 1) if t.size else 0.0
    res["score_median"] = round(float(np.nanmedian(scores)), 3) if usable.any() else 0.0
    return res


def af_analysis_beats(beats, duration_s: float = 0.0, **kwargs) -> dict:
    """``af_analysis`` sobre la lista de latidos del pipeline (``Beat``)."""
    if not beats:
        return af_burden([], duration_s) | {"episodes": [], "scores": np.zeros(0)}
    t = [float(b.t) for b in beats]
    cls = [str(b.cls) for b in beats]
    unread = [bool(getattr(b, "features", {}).get("unreadable", False)) for b in beats]
    kwargs.setdefault("unreadable", unread if any(unread) else None)
    return af_analysis(t, cls, duration_s, **kwargs)


# --- apoyos de T10 (onda P) y T11 (actividad auricular del residuo) ---------
def af_pwave_support(x: np.ndarray, fs: float, peaks) -> dict:
    """Apoyo de la onda P: ``present``/``evaluable`` por latido (T10)."""
    from .pwave import detect_p_waves

    res = detect_p_waves(np.asarray(x, dtype=np.float64), float(fs),
                        np.asarray(peaks, dtype=int).ravel())
    present = np.asarray(res["present"], dtype=bool).astype(float)
    evaluable = np.asarray(res["evaluable"], dtype=bool)
    present[~evaluable] = np.nan
    return {"p_present": present, "p_evaluable": evaluable,
            "p_frac": round(float(np.nanmean(present)), 3) if evaluable.any() else None,
            "status": res.get("status")}


def af_atrial_support(x: np.ndarray, fs: float, peaks, win_s: float = 5.0,
                      step_s: float = 5.0) -> dict:
    """Apoyo de la actividad f del residuo del QRST (T11), agregado."""
    from .atrial import analyze as atrial_analyze

    res = atrial_analyze(np.asarray(x, dtype=np.float64), float(fs),
                         np.asarray(peaks, dtype=int).ravel(),
                         win_s=win_s, step_s=step_s)
    summary = dict(res["summary"])
    summary["residual"] = None  # la señal completa no se devuelve: es grande
    return summary


def af_fwave_support(x: np.ndarray, fs: float, peaks, duration_s: float = 0.0, *,
                     win_s: float = 10.0, step_s: float = 10.0,
                     conc_min: float = 0.40, band: tuple[float, float] = (3.5, 6.5),
                     half_s: float = 30.0) -> np.ndarray:
    """Actividad f **por latido** (0..1) a partir de las ventanas de T11.

    Una ventana de ``win_s`` cuenta como actividad f si su espectro tiene un
    pico concentrado (``conc`` >= ``conc_min``) en la banda auricular. Para cada
    latido se mide la fracción de los ``2·half_s`` segundos a su alrededor que
    cae en ventanas con actividad f. Si la cancelación del QRST fue mala
    (``quality != "ok"``) devuelve ceros: el residuo no es interpretable.
    """
    from .atrial import analyze as atrial_analyze

    x = np.asarray(x, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=np.int64).ravel()
    res = atrial_analyze(x, fs, peaks, win_s=win_s, step_s=step_s)
    out = np.zeros(peaks.size, dtype=float)
    if str(res["summary"].get("quality", "")) != "ok":
        return out
    dur = float(duration_s) if duration_s > 0 else x.size / float(fs)
    n = max(1, int(np.ceil(dur)))
    flag = np.zeros(n + 1, dtype=float)
    for w in res["windows"]:
        if w["conc"] < conc_min or not (band[0] <= w["dom_hz"] <= band[1]):
            continue
        a = int(np.clip(np.floor(w["t0"]), 0, n))
        b = int(np.clip(np.ceil(w["t1"]), 0, n))
        flag[a:b] = 1.0
    csum = np.concatenate([[0.0], np.cumsum(flag)])
    tb = peaks.astype(float) / float(fs)
    lo = np.clip(np.round(tb - half_s).astype(int), 0, n)
    hi = np.clip(np.round(tb + half_s).astype(int), 0, n)
    span = np.maximum(hi - lo, 1)
    return (csum[hi] - csum[lo]) / span


def af_analysis_from_signal(x: np.ndarray, fs: float, peaks, classes=None,
                            duration_s: float = 0.0, *, use_pwave: bool = True,
                            use_atrial: bool = True, **kwargs) -> dict:
    """Detector de FA sobre una señal: añade los apoyos de T10 y T11."""
    peaks = np.asarray(peaks, dtype=np.int64).ravel()
    times = peaks.astype(np.float64) / float(fs)
    res: dict = {}
    if use_pwave:
        res["pwave"] = af_pwave_support(x, fs, peaks)
        kwargs.setdefault("p_present", res["pwave"]["p_present"])
        kwargs.setdefault("p_evaluable", res["pwave"]["p_evaluable"])
    if use_atrial:
        res["atrial"] = af_atrial_support(x, fs, peaks)
        kwargs.setdefault("fwave", res["atrial"])
    out = af_analysis(times, classes, duration_s, **kwargs)
    out["support"] = res
    return out


# ---------------------------------------------------------------------------
# Patrones ventriculares (T121, sección C de CRITERIOS-TRATADOS)
# ---------------------------------------------------------------------------
# Reglas sobre la secuencia de latidos ya clasificada (N, S, V, F, Q). Sin
# modelos: solo umbrales, que se leen del YAML de criterios
# (``configs/criteria/ventricular.yaml``).
VENTRICULAR_KINDS = (
    "pareja", "tvns", "tv_sostenida", "riva", "bigeminismo", "trigeminismo",
    "r_sobre_t", "interpolada", "pausa_compensadora", "fusion", "multiforme",
)

_RR_MIN_S = 0.2
_RR_MAX_S = 2.5


@dataclass
class VentricularCriteria:
    """Umbrales de los patrones ventriculares.

    ``run_min_beats`` y ``cycles_min`` son el mínimo de latidos seguidos (salva,
    bigeminismo, trigeminismo). ``tvns_min_bpm`` separa TVNS y TV (>100 lpm);
    ``sustained_s`` separa TVNS (<30 s) de TV sostenida (≥30 s). La RIVA usa la
    banda 50–110 lpm, pero la regla de TVNS (>100 lpm) se aplica antes, así que
    en la práctica RIVA cubre 50–100 lpm. ``basal_tol`` es la tolerancia
    (±10 %) para interpolada, compensadora y fusión; ``basal_window`` cuenta
    los latidos normales a cada lado para el RR basal. ``risk_coupling_ms``
    (≥500 ms) se informa como factor de riesgo del acoplamiento.
    """

    run_min_beats: int = 3
    tvns_min_bpm: float = 100.0
    sustained_s: float = 30.0
    riva_min_bpm: float = 50.0
    riva_max_bpm: float = 110.0
    cycles_min: int = 3
    basal_tol: float = 0.10
    basal_window: int = 15
    risk_coupling_ms: float = 500.0

    @classmethod
    def from_mapping(cls, data) -> VentricularCriteria:
        """Umbrales desde un dict; las claves ausentes o no numéricas usan el default."""
        base = cls()
        if not isinstance(data, dict):
            return base
        vals = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            kind = type(getattr(base, f.name))
            try:
                vals[f.name] = kind(data[f.name])
            except (TypeError, ValueError):
                continue
        return replace(base, **vals)

    @classmethod
    def load(cls, path: str | Path | None) -> VentricularCriteria:
        """Lee la sección ``ventricular:`` del YAML (o el YAML plano si no la tiene)."""
        if path is None:
            return cls()
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            return cls()
        return cls.from_mapping(data.get("ventricular", data))


def _v_runs(cls_seq: list[str]) -> list[tuple[int, int]]:
    """Rachas de V consecutivos como pares (primer índice, último índice)."""
    runs: list[tuple[int, int]] = []
    i, n = 0, len(cls_seq)
    while i < n:
        if cls_seq[i] != "V":
            i += 1
            continue
        j = i
        while j + 1 < n and cls_seq[j + 1] == "V":
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _nv_cycles(cls_seq: list[str], start: int, period: int) -> int:
    """Ciclos completos (period-1 × N seguidos de V) desde ``start``."""
    n, k = len(cls_seq), 0
    while start + (k + 1) * period <= n:
        base = start + k * period
        if cls_seq[base + period - 1] != "V":
            break
        if any(cls_seq[base + q] != "N" for q in range(period - 1)):
            break
        k += 1
    return k


def _nv_segments(cls_seq: list[str], period: int, min_cycles: int) -> list[tuple[int, int, int]]:
    """Tramos alternantes (inicio, último índice, ciclos) de al menos ``min_cycles``."""
    out: list[tuple[int, int, int]] = []
    i, n = 0, len(cls_seq)
    while i < n:
        if cls_seq[i] != "N":
            i += 1
            continue
        k = _nv_cycles(cls_seq, i, period)
        if k >= min_cycles:
            out.append((i, i + k * period - 1, k))
            i += k * period
        else:
            i += 1
    return out


def ventricular_patterns(
    beats,
    criteria: VentricularCriteria | None = None,
    *,
    unreadable: list[tuple[float, float]] | None = None,
    vt_labels=None,
    fusion_mid=None,
) -> dict:
    """Patrones ventriculares de una secuencia de latidos (T121).

    ``beats`` necesita ``cls``, ``t``, ``rr_prev`` y ``rr_next``; ``features["qt_ms"]``
    (opcional) habilita R sobre T. ``unreadable`` son tramos ``(t0, t1)`` en
    segundos: cualquier evento que los toque no se informa (artefacto en ≥2
    canales). ``vt_labels`` es una etiqueta de morfología por latido (-1 = sin
    grupo) para la EV multiforme; ``fusion_mid`` marca latidos de morfología
    intermedia N-V. Sin esas entradas, multiforme y fusión por morfología no se
    evalúan. Devuelve eventos, conteos por tipo, estadística del acoplamiento,
    eventos suprimidos y casos no evaluables.
    """
    crit = criteria or VentricularCriteria()
    n = len(beats)
    cls_seq = [str(b.cls) for b in beats]
    t = np.array([float(b.t) for b in beats], dtype=float)
    rr_p = np.array([float(getattr(b, "rr_prev", 0.0) or 0.0) for b in beats], dtype=float)
    rr_n = np.array([float(getattr(b, "rr_next", 0.0) or 0.0) for b in beats], dtype=float)
    qt = np.array(
        [
            float((getattr(b, "features", None) or {}).get("qt_ms") or np.nan)
            for b in beats
        ],
        dtype=float,
    )
    gates = [(float(a), float(b)) for a, b in (unreadable or []) if b > a]

    def hidden(a: float, b: float) -> bool:
        return any(a <= t1 and b >= t0 for t0, t1 in gates)

    counts = {k: 0 for k in VENTRICULAR_KINDS}
    events: list[dict] = []
    not_eval = {"r_sobre_t_sin_qt": 0, "sin_rr_basal": 0, "sin_frecuencia": 0}
    suppressed = 0

    def add(kind: str, i0: int, i1: int, **extra) -> None:
        nonlocal suppressed
        start, end = float(t[i0]), float(t[i1])
        if hidden(start, end):
            suppressed += 1
            return
        counts[kind] += 1
        events.append({"kind": kind, "start_s": round(start, 3), "end_s": round(end, 3),
                       "i0": int(i0), "i1": int(i1), **extra})

    # Rachas de V: pareja (2), salva/TVNS, TV sostenida y RIVA (≥3).
    for a, b in _v_runs(cls_seq):
        length = b - a + 1
        if length == 2:
            add("pareja", a, b, n_beats=2)
            continue
        if length < crit.run_min_beats:
            continue
        rrs = rr_p[a + 1: b + 1]
        rrs = rrs[(rrs > _RR_MIN_S) & (rrs < _RR_MAX_S)]
        if rrs.size == 0:
            not_eval["sin_frecuencia"] += 1
            continue
        rate = 60.0 / float(rrs.mean())
        dur = float(t[b] - t[a])
        if rate > crit.tvns_min_bpm:
            kind = "tv_sostenida" if dur >= crit.sustained_s else "tvns"
            add(kind, a, b, n_beats=length, rate_bpm=round(rate, 1), duration_s=round(dur, 2))
        elif crit.riva_min_bpm <= rate <= crit.riva_max_bpm:
            add("riva", a, b, n_beats=length, rate_bpm=round(rate, 1), duration_s=round(dur, 2))

    # Bigeminismo (N-V) y trigeminismo (N-N-V) de al menos cycles_min ciclos.
    for start, end, k in _nv_segments(cls_seq, 2, crit.cycles_min):
        add("bigeminismo", start, end, n_beats=2 * k, cycles=k)
    for start, end, k in _nv_segments(cls_seq, 3, crit.cycles_min):
        add("trigeminismo", start, end, n_beats=3 * k, cycles=k)

    # RR basal: mediana de los RR normales (N precedido de N) alrededor del latido.
    normal_rr = [rr_p[k] for k in range(1, n)
                 if cls_seq[k] == "N" and cls_seq[k - 1] == "N"
                 and _RR_MIN_S < rr_p[k] < _RR_MAX_S]
    global_basal = float(np.median(normal_rr)) if len(normal_rr) >= 3 else None
    basal_cache: dict[int, float | None] = {}

    def basal(j: int) -> float | None:
        if j in basal_cache:
            return basal_cache[j]
        lo, hi = max(1, j - crit.basal_window), min(n, j + crit.basal_window + 1)
        local = [rr_p[k] for k in range(lo, hi)
                 if k != j and cls_seq[k] == "N" and cls_seq[k - 1] == "N"
                 and _RR_MIN_S < rr_p[k] < _RR_MAX_S]
        val = float(np.median(local)) if len(local) >= 3 else global_basal
        basal_cache[j] = val
        return val

    fusion_flag = None if fusion_mid is None else np.asarray(fusion_mid, dtype=bool)
    for j in range(n):
        c = cls_seq[j]
        if c == "V" and j > 0 and rr_p[j] > 0:
            # R sobre T: el acoplamiento es menor que el QT del latido previo.
            if np.isfinite(qt[j - 1]) and qt[j - 1] > 0:
                if rr_p[j] * 1000.0 < RSOBRET_INDEX_MAX * qt[j - 1]:
                    add("r_sobre_t", j, j, coupling_ms=round(rr_p[j] * 1000.0, 1))
            else:
                not_eval["r_sobre_t_sin_qt"] += 1
        if c == "V" and 0 < j < n - 1 and cls_seq[j - 1] != "V" and cls_seq[j + 1] != "V":
            if rr_p[j] > 0 and rr_n[j] > 0:
                b_rr = basal(j)
                if b_rr is None:
                    not_eval["sin_rr_basal"] += 1
                else:
                    s = rr_p[j] + rr_n[j]
                    if abs(s - b_rr) <= crit.basal_tol * b_rr:
                        add("interpolada", j, j, pre_s=round(rr_p[j], 3), post_s=round(rr_n[j], 3))
                    elif abs(s - 2.0 * b_rr) <= crit.basal_tol * 2.0 * b_rr:
                        add("pausa_compensadora", j, j, pre_s=round(rr_p[j], 3),
                            post_s=round(rr_n[j], 3))
        is_fusion_cand = c != "N" and (c == "F" or bool(fusion_flag is not None and fusion_flag[j]))
        if is_fusion_cand and rr_p[j] > 0:
            b_rr = basal(j)
            if b_rr is None:
                not_eval["sin_rr_basal"] += 1
            elif abs(rr_p[j] - b_rr) <= crit.basal_tol * b_rr:
                add("fusion", j, j, coupling_ms=round(rr_p[j] * 1000.0, 1))

    # EV multiforme: ≥2 grupos de morfología entre los V no ilegibles.
    if vt_labels is not None:
        labels = np.asarray(vt_labels, dtype=int)
        v_idx = [j for j in range(n) if cls_seq[j] == "V" and not hidden(t[j], t[j])]
        groups = {int(labels[j]) for j in v_idx if labels[j] >= 0}
        if len(groups) >= 2:
            add("multiforme", v_idx[0], v_idx[-1], templates=len(groups), n_beats=len(v_idx))

    # Acoplamiento de las EV aisladas tras latido normal (sin ilegibles).
    cps = [rr_p[j] * 1000.0 for j in range(1, n)
           if cls_seq[j] == "V" and cls_seq[j - 1] == "N"
           and _RR_MIN_S < rr_p[j] < _RR_MAX_S and not hidden(t[j], t[j])]
    coupling = {
        "n": len(cps),
        "mean_ms": round(float(np.mean(cps)), 1) if cps else None,
        "sd_ms": round(float(np.std(cps, ddof=1)), 1) if len(cps) > 1 else None,
        "n_ge_risk": int(sum(1 for x in cps if x >= crit.risk_coupling_ms)),
    }
    return {"events": events, "counts": counts, "coupling": coupling,
            "suppressed": suppressed, "not_evaluable": not_eval}


# ---------------------------------------------------------------------------
# Integración en analyze() (T167): R sobre T (C7), EV multiforme (C11) y
# parasistolia probable (C13). Umbrales de CRITERIOS-TRATADOS §C, de T122 y de
# la tarea; ninguno nuevo. Sin modelos.
# ---------------------------------------------------------------------------
RSOBRET_INDEX_MAX = 1.0       # C7: acoplamiento de la EV / QT del latido previo < 1,0
MULTIFORME_CORR_MAX = 0.80    # C11: dos plantillas son distintas si correlan < 0,80
MULTIFORME_MIN_BEATS = 3      # C11: cada plantilla con al menos 3 latidos
# C11 compara solo el QRS y el inicio del ST (s desde el pico): la ventana completa de
# 0,6 s trae la P y la T de los vecinos, que cambian con el acoplamiento y dan
# plantillas espurias.
MULTIFORME_WIN_S = (-0.12, 0.20)
PARA_PROBABLE = "parasistolia probable"
PARA_CANDIDATA = "ciclo común sin fusión"


def span_hit(a: float, b: float, spans) -> bool:
    """¿El intervalo [a, b] toca algún tramo (t0, t1)? Tramos ilegibles (≥2 de 3 canales)."""
    return any(a <= t1 and b >= t0 for t0, t1 in spans)


def multiforme_templates(wins, v_mask, *, corr_max: float = MULTIFORME_CORR_MAX,
                         min_beats: int = MULTIFORME_MIN_BEATS) -> list[list[int]]:
    """Plantillas de EV distintas (C11). Devuelve los índices de latido de cada plantilla.

    Agrupamiento voraz por correlación ≥ ``corr_max`` con la forma de onda de cada V
    (ventana centrada en el latido); un grupo cuenta si tiene ≥ ``min_beats`` latidos.
    Dos grupos son morfologías distintas solo si la correlación entre sus medias es
    < ``corr_max``; los grupos se aceptan del más numeroso al menos numeroso.
    """
    v_idx = np.flatnonzero(np.asarray(v_mask, dtype=bool))
    if v_idx.size < min_beats:
        return []
    w = np.asarray(wins, dtype=np.float64)[v_idx]
    w = w - w.mean(axis=1, keepdims=True)
    w = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-9)
    sim = w @ w.T
    assigned = np.zeros(v_idx.size, dtype=bool)
    groups: list[np.ndarray] = []
    for i in range(v_idx.size):
        if assigned[i]:
            continue
        mem = np.flatnonzero((sim[i] >= corr_max) & ~assigned)
        if mem.size >= min_beats:
            assigned[mem] = True
            groups.append(mem)
    groups.sort(key=lambda g: (-g.size, int(g[0])))
    accepted: list[np.ndarray] = []
    out: list[list[int]] = []
    for g in groups:
        c = w[g].mean(axis=0)
        c = c / max(float(np.linalg.norm(c)), 1e-9)
        if any(float(c @ a) >= corr_max for a in accepted):
            continue
        accepted.append(c)
        out.append([int(v_idx[k]) for k in g])
    return out


@dataclass
class ParasistoliaCriteria:
    """Umbrales de la parasistolia probable (T122, sección C de CRITERIOS-TRATADOS).

    Los valores de CRITERIOS-TRATADOS solo dan la forma (acoplamiento variable + ciclo
    común + fusiones); los números salen de T122 y de constantes de ``rhythm``. Se
    pueden sobrescribir con una sección ``parasistolia:`` del YAML de criterios.
    """

    de_min_ms: float = 80.0          # DE del acoplamiento > 80 ms
    template_min_beats: int = 20     # plantilla V con ≥20 latidos (y ≥20 acoplamientos)
    period_tol: float = 0.05         # ±5 % del período P (no del múltiplo k·P)
    fit_min_fraction: float = 1.0    # fracción de intervalos V–V que deben ajustarse
    period_min_s: float = _RR_MIN_S
    period_max_s: float = _RR_MAX_S

    @classmethod
    def from_mapping(cls, data) -> ParasistoliaCriteria:
        base = cls()
        if not isinstance(data, dict):
            return base
        vals = {}
        for f in fields(cls):
            if f.name in data:
                try:
                    vals[f.name] = type(getattr(base, f.name))(data[f.name])
                except (TypeError, ValueError):
                    continue
        return replace(base, **vals)

    @classmethod
    def load(cls, path: str | Path | None) -> ParasistoliaCriteria:
        if path is None or not Path(path).exists():
            return cls()
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_mapping(data.get("parasistolia", {}) if isinstance(data, dict) else {})


def find_common_cycle(d_ms: np.ndarray, crit: ParasistoliaCriteria):
    """Ciclo común de los V–V (ms): (P, fracción, k→nº, mejor fracción).

    Tolerancia: |d − k·P| ≤ period_tol · P (±5 % del período, no del múltiplo: con
    ±5 % de k·P la ventana crece con k y el criterio deja de discriminar). Candidatos:
    d_i/k dentro de [period_min, period_max]. Se devuelve el mayor P que cumple
    ``fit_min_fraction`` (el ciclo fundamental), o ``P=None``. La mejor fracción
    alcanzada por cualquier candidato va en el informe, aunque no cumpla.
    """
    if d_ms.size < 2:
        return None, 0.0, {}, 0.0
    p_lo, p_hi = crit.period_min_s * 1000.0, crit.period_max_s * 1000.0
    cands: set[float] = set()
    for di in d_ms:
        k = 1
        while di / k >= p_lo:
            if di / k <= p_hi:
                cands.add(float(di / k))
            k += 1
    best_frac = 0.0
    found: tuple[float, float, dict] | None = None
    for p in sorted(cands, reverse=True):
        k = np.rint(d_ms / p)
        ok = (k >= 1) & (np.abs(d_ms - k * p) <= crit.period_tol * p)
        frac = float(ok.mean())
        best_frac = max(best_frac, frac)
        if found is None and frac >= crit.fit_min_fraction - 1e-12:
            ks, counts = np.unique(k[ok].astype(int), return_counts=True)
            found = (p, frac, {int(a): int(b) for a, b in zip(ks, counts, strict=True)})
    if found is None:
        return None, 0.0, {}, best_frac
    return found[0], found[1], found[2], best_frac


def parasistolia_analyze(t, cls, rr_p, labels, unread,
                         crit: ParasistoliaCriteria | None = None) -> dict:
    """Parasistolia por plantilla V (T122): acoplamiento DE, ciclo común y fusiones.

    ``t``, ``cls`` (AAMI por latido), ``rr_p`` (RR previo, s), ``labels`` (plantilla por
    latido, -1 = sin plantilla) y ``unread`` (tramos ilegibles (t0, t1) en s). Una
    plantilla cuyo tramo (primer al último V) toca un tramo ilegible se suprime.
    «parasistolia probable» = acoplamiento variable + ciclo V–V común + F dentro del
    tramo de la plantilla; sin F sale «ciclo común sin fusión» (candidata).
    """
    crit = crit or ParasistoliaCriteria()
    t = np.asarray(t, dtype=float)
    cls = np.asarray(cls)
    rr_p = np.asarray(rr_p, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = t.size
    templates: dict[int, list[int]] = {}
    for j in range(n):
        if cls[j] == "V" and labels[j] >= 0:
            templates.setdefault(int(labels[j]), []).append(j)

    rows: list[dict] = []
    suppressed = too_few = 0
    for lab, idx in sorted(templates.items()):
        if len(idx) < crit.template_min_beats:
            too_few += 1
            continue
        t0, t1 = float(t[idx[0]]), float(t[idx[-1]])
        if span_hit(t0, t1, unread):
            suppressed += 1
            continue
        cpl = [rr_p[j] * 1000.0 for j in idx
               if j > 0 and cls[j - 1] == "N" and crit.period_min_s < rr_p[j] < crit.period_max_s]
        if len(cpl) < crit.template_min_beats:
            too_few += 1
            continue
        cpl_arr = np.asarray(cpl)
        de = float(np.std(cpl_arr, ddof=1))
        row = {
            "template": lab, "n_V": len(idx), "n_coupling": len(cpl),
            "mean_coupling_ms": round(float(cpl_arr.mean()), 1), "de_coupling_ms": round(de, 1),
            "de_gate": bool(de > crit.de_min_ms),
        }
        if not row["de_gate"]:
            row.update(period_ms=None, fit_fraction=None, best_fraction=None, n_fusion=None,
                       verdict="acoplamiento fijo (DE ≤ umbral)")
            rows.append(row)
            continue
        dd = np.diff(t[idx]) * 1000.0
        dd = dd[dd > 0]
        period, frac, khist, best = find_common_cycle(dd, crit)
        n_fus = int(np.count_nonzero((cls == "F") & (t >= t0) & (t <= t1)))
        if period is None:
            verdict = "sin ciclo común"
        elif n_fus == 0:
            verdict = PARA_CANDIDATA
        else:
            verdict = PARA_PROBABLE
        row.update(
            period_ms=None if period is None else round(period, 1),
            fit_fraction=None if period is None else round(frac, 3),
            best_fraction=round(best, 3),
            n_vv=int(dd.size), k_hist=khist, n_fusion=n_fus, verdict=verdict,
            span_s=[round(t0, 2), round(t1, 2)],
        )
        rows.append(row)
    return {"templates": rows, "suppressed": suppressed, "too_few": too_few}


# Ritmos sinusales, de la unión y de escape (T123, CRITERIOS-TRATADOS §D)
# ---------------------------------------------------------------------------
# Etiquetas de ventana. Solo las de BASE_RHYTHM_NAMES son informables; el resto
# son el ritmo sinusal normal (ninguna regla se cumple), "sin regla" (fuera de
# esta sección: FA, TSV, bigeminia, P no evaluable) y "no informado" (artefacto).
BASE_RHYTHM_NAMES = {
    "taquicardia_sinusal": "taquicardia sinusal",
    "bradicardia_sinusal": "bradicardia sinusal",
    "arritmia_sinusal": "arritmia sinusal",
    "union": "ritmo de la unión",
    "union_acelerado": "ritmo de la unión acelerado",
    "idioventricular": "ritmo idioventricular de escape",
}
BASE_SINUS = "sinusal"
BASE_NO_RULE = "sin_regla"
BASE_NOT_REPORTED = "no_informado"
#: Ritmos de base que se integran como evento en analyze() (T165, filas D6–D8).
#: Los de la sección D sinusal (taquicardia, bradicardia, arritmia) ya los informa
#: la rama de ritmo existente y no se duplican aquí.
BASE_EVENT_KINDS = ("union", "union_acelerado", "idioventricular")


@dataclass
class BaseRhythmCriteria:
    """Umbrales de la sección D de CRITERIOS-TRATADOS (no se inventan otros).

    Las tolerancias de ventana y de presencia de P reutilizan los umbrales de la
    FA (``AF_P_*``) y el umbral de pausa de la sección F (2,0 s).
    """

    tachy_bpm: float = 100.0            # taquicardia sinusal: > 100 lpm
    brady_bpm: float = 50.0             # bradicardia sinusal: < 50 lpm
    brady_severe_bpm: float = 40.0      # se informa también < 40 (despierto no se distingue)
    pp_range_s: float = 0.120           # arritmia sinusal: PP máx − PP mín > 120 ms
    junc_bpm: tuple = (40.0, 60.0)      # ritmo de la unión: 40–60 lpm
    junc_acc_bpm: tuple = (60.0, 100.0)  # ritmo de la unión acelerado: 60–100 lpm
    ivr_bpm: tuple = (20.0, 40.0)       # idioventricular de escape: 20–40 lpm
    wide_qrs_s: float = 0.120           # QRS ancho (AHA 2009 parte III, §H)
    pr_min_s: float = 0.120             # PR < 120 ms: no es P conducida sinusal
    escape_pause_s: float = 2.0         # latido de escape: tras pausa ≥ 2,0 s (§F)
    rr_min_s: float = AF_MIN_RR_S       # RR fisiológico para la FC (20–240 lpm)
    rr_max_s: float = 3.0               # 3,0 s = 20 lpm, límite inferior de la IVR
    win_s: float = 10.0                 # ventana = tramo mínimo de 10 s
    step_s: float = 5.0                 # paso de la ventana deslizante
    min_beats: int = 3                  # RR válidos por ventana (IVR a 20 lpm: 3–4 latidos)
    p_min_eval: float = AF_P_MIN_EVALUABLE
    p_present_min: float = AF_P_PRESENT_MIN
    p_absent_max: float = AF_P_ABSENT_MAX
    p_corr_min: float = 0.80            # P idéntica: correlación mediana con la plantilla


def _classify_window(hr: float, wide: bool, p_state: str, pp_range: float,
                     p_ident: bool, crit: BaseRhythmCriteria) -> str:
    # La sinusal exige P conducida y no depende de la anchura del QRS (un BRI
    # con P sinusal sigue siendo sinusal).
    if p_state == "sinusal":
        if hr > crit.tachy_bpm:
            return "taquicardia_sinusal"
        if hr < crit.brady_bpm:
            return "bradicardia_sinusal"
        if pp_range > crit.pp_range_s and p_ident:
            return "arritmia_sinusal"
        return BASE_SINUS
    # Ritmo de escape ventricular: QRS ancho a 20–40 lpm.
    if wide:
        lo, hi = crit.ivr_bpm
        return "idioventricular" if lo <= hr <= hi else BASE_NO_RULE
    # Unión: QRS estrecho sin P conducida (ausente, o PR < 120 ms).
    if p_state in ("ausente", "corta"):
        if crit.junc_bpm[0] <= hr < crit.junc_bpm[1]:
            return "union"
        if crit.junc_acc_bpm[0] <= hr <= crit.junc_acc_bpm[1]:
            return "union_acelerado"
    # P mixta o no evaluable: no se decide entre sinusal y unión (ni taq/bradi,
    # que exigen P sinusal confirmada).
    return BASE_NO_RULE


def _window_rhythm(rr, rr_ok, normal, qrs_s, present, evaluable, pr_s, corr,
                   crit: BaseRhythmCriteria) -> tuple[str, dict]:
    n_beats = rr.size
    nrr = int(rr_ok.sum())
    if nrr < crit.min_beats:
        return BASE_NO_RULE, {"hr": float("nan"), "n_rr": nrr}
    hr = 60.0 / float(np.median(rr[rr_ok]))
    qrs = qrs_s[np.isfinite(qrs_s)]
    wide = bool(qrs.size) and float(np.median(qrs)) >= crit.wide_qrs_s

    n_ev = int(evaluable.sum())
    if n_ev and n_ev >= crit.p_min_eval * n_beats:
        pres_ev = present & evaluable
        pfrac = float(pres_ev.sum()) / n_ev
        pr_vals = pr_s[pres_ev]
        pr_vals = pr_vals[np.isfinite(pr_vals)]
        if pfrac >= crit.p_present_min:
            pr_med = float(np.median(pr_vals)) if pr_vals.size else float("nan")
            p_state = "sinusal" if pr_vals.size and pr_med >= crit.pr_min_s else "corta"
        elif pfrac <= crit.p_absent_max:
            p_state = "ausente"
        else:
            p_state = "mixta"
    else:
        p_state = "desconocido"

    # PP: intervalos entre latidos normales consecutivos, sin pausas.
    ok_pp = normal[1:] & normal[:-1] & rr_ok[1:] & (rr[1:] <= crit.escape_pause_s)
    pp = rr[1:][ok_pp]
    pp_range = float(pp.max() - pp.min()) if pp.size >= 2 else 0.0
    cv = corr[present]
    cv = cv[np.isfinite(cv)]
    p_ident = bool(cv.size >= 2 and float(np.median(cv)) >= crit.p_corr_min)

    label = _classify_window(hr, wide, p_state, pp_range, p_ident, crit)
    return label, {"hr": hr, "wide": wide, "p_state": p_state,
                   "pp_range_s": pp_range, "p_ident": p_ident, "n_rr": nrr}


def base_rhythm_analysis(t, rr, normal, qrs_s, present, evaluable, pr_s, corr, *,
                         duration_s: float, bad=None, af_episodes=None,
                         crit: BaseRhythmCriteria | None = None) -> dict:
    """Etiqueta ritmos de base por ventanas de 10 s, tramos, latidos de escape y FC/hora.

    Entradas por latido (arreglos alineados, ``t`` ascendente en segundos):
      ``rr``        intervalo con el latido anterior (s); NaN en el primero.
      ``normal``    latido no ectópico (para la variación PP).
      ``qrs_s``     anchura del QRS (s).
      ``present`` / ``evaluable`` / ``pr_s`` / ``corr``  salida de la P (T10):
                    P presente, P evaluable, intervalo PR (s) y correlación.
      ``bad``       ≥2 de 3 canales malos: la ventana no se informa (opcional).
      ``af_episodes``  episodios de FA (T23) como (inicio_s, fin_s): la ventana que
                    los solapa queda "sin regla" (la unión es regular; la FA no).

    Una ventana sin P evaluable en ≥``p_min_eval`` de sus latidos no decide entre
    sinusal y unión: queda "sin regla".
    """
    crit = BaseRhythmCriteria() if crit is None else crit
    t = np.asarray(t, dtype=float)
    n = t.size
    if n and np.any(np.diff(t) < 0):
        raise ValueError("t debe ser ascendente")
    arr = {
        "rr": np.asarray(rr, dtype=float), "normal": np.asarray(normal, dtype=bool),
        "qrs_s": np.asarray(qrs_s, dtype=float), "present": np.asarray(present, dtype=bool),
        "evaluable": np.asarray(evaluable, dtype=bool), "pr_s": np.asarray(pr_s, dtype=float),
        "corr": np.asarray(corr, dtype=float),
    }
    bad_arr = np.zeros(n, dtype=bool) if bad is None else np.asarray(bad, dtype=bool)
    rr_ok = np.isfinite(arr["rr"]) & (arr["rr"] >= crit.rr_min_s) & (arr["rr"] <= crit.rr_max_s)
    conducted = arr["present"] & arr["evaluable"] & np.isfinite(arr["pr_s"]) \
        & (arr["pr_s"] >= crit.pr_min_s)
    # Escape = pausa aislada: RR > 2,0 s precedido de RR ≤ 2,0 s. Un ritmo
    # idioventricular lento (RR > 2 s en cada latido) no genera un escape por latido.
    pause = np.isfinite(arr["rr"]) & (arr["rr"] > crit.escape_pause_s)
    # Sin RR previo (primer latido) no hay pausa previa: no es escape.
    prev_short = np.r_[False, np.nan_to_num(arr["rr"][:-1], nan=np.inf) <= crit.escape_pause_s]
    af_iv = np.asarray(list(af_episodes or []), dtype=float).reshape(-1, 2)
    in_af_beat = np.zeros(n, dtype=bool)
    for a0, a1 in af_iv:
        in_af_beat |= (t >= a0) & (t < a1)
    escape = pause & prev_short & ~conducted & ~bad_arr & ~in_af_beat

    last = max(float(duration_s), float(t[-1]) if n else 0.0)
    starts = np.arange(0.0, max(last - crit.win_s, 0.0) + 1e-9, crit.step_s) if last > 0 else []
    windows: list[dict] = []
    for a in starts:
        b = a + crit.win_s
        i0 = int(np.searchsorted(t, a, side="left"))
        i1 = int(np.searchsorted(t, b, side="left"))
        in_af = bool(af_iv.size) and bool(np.any((af_iv[:, 0] < b) & (af_iv[:, 1] > a)))
        if bad_arr[i0:i1].any():
            label, info = BASE_NOT_REPORTED, {}
        elif in_af:
            label, info = BASE_NO_RULE, {"af": True}
        else:
            sl = slice(i0, i1)
            label, info = _window_rhythm(
                arr["rr"][sl], rr_ok[sl], arr["normal"][sl], arr["qrs_s"][sl],
                arr["present"][sl], arr["evaluable"][sl], arr["pr_s"][sl], arr["corr"][sl],
                crit) if i1 > i0 else (BASE_NO_RULE, {})
        windows.append({"start_s": float(a), "end_s": float(b), "label": label, **info})

    segments: list[dict] = []
    for w in windows:
        if segments and segments[-1]["label"] == w["label"]:
            seg = segments[-1]
            seg["end_s"] = w["end_s"]
            seg["hr_values"].append(w.get("hr", float("nan")))
            seg["n_windows"] += 1
        else:
            segments.append({"label": w["label"], "start_s": w["start_s"],
                             "end_s": w["end_s"], "n_windows": 1,
                             "hr_values": [w.get("hr", float("nan"))]})
    events: list[dict] = []
    for seg in segments:
        hrs = np.asarray(seg.pop("hr_values"), dtype=float)
        hrs = hrs[np.isfinite(hrs)]
        seg["hr_med"] = float(np.median(hrs)) if hrs.size else float("nan")
        seg["duration_s"] = seg["end_s"] - seg["start_s"]
        if seg["label"] in BASE_RHYTHM_NAMES and seg["duration_s"] >= crit.win_s:
            ev = {"kind": seg["label"], "start_s": seg["start_s"], "end_s": seg["end_s"],
                  "hr_med": round(seg["hr_med"], 1), "detail": BASE_RHYTHM_NAMES[seg["label"]]}
            if seg["label"] == "bradicardia_sinusal" and seg["hr_med"] < crit.brady_severe_bpm:
                ev["detail"] += f" (<{crit.brady_severe_bpm:.0f} lpm)"
            events.append(ev)

    for i in np.nonzero(escape)[0]:
        events.append({"kind": "escape", "start_s": float(t[i] - arr["rr"][i]),
                       "end_s": float(t[i]), "beat": int(i),
                       "subtype": "ventricular" if arr["qrs_s"][i] >= crit.wide_qrs_s else "union",
                       "detail": "latido de escape tras pausa"})

    return {"windows": windows, "segments": segments, "events": events,
            "hourly": base_rate_hourly(t, arr["rr"], duration_s, bad_arr, crit)}


def base_rate_hourly(t, rr, duration_s: float, bad=None,
                     crit: BaseRhythmCriteria | None = None) -> list[dict]:
    """FC mín/máx/media por hora, excluyendo RR no fisiológicos y artefacto."""
    crit = BaseRhythmCriteria() if crit is None else crit
    t = np.asarray(t, dtype=float)
    rr = np.asarray(rr, dtype=float)
    valid = np.isfinite(rr) & (rr >= crit.rr_min_s) & (rr <= crit.rr_max_s)
    if bad is not None:
        valid &= ~np.asarray(bad, dtype=bool)
    hr = np.where(valid, 60.0 / np.maximum(rr, 1e-6), np.nan)
    n_hours = int(np.ceil(max(float(duration_s), float(t[-1]) if t.size else 0.0) / 3600.0))
    rows = []
    for h in range(n_hours):
        m = (t >= h * 3600.0) & (t < (h + 1) * 3600.0) & np.isfinite(hr)
        vals = hr[m]
        rows.append({"hour": h, "n_beats": int(vals.size),
                     "hr_min": float(vals.min()) if vals.size else float("nan"),
                     "hr_max": float(vals.max()) if vals.size else float("nan"),
                     "hr_mean": float(vals.mean()) if vals.size else float("nan")})
    return rows

# ---------------------------------------------------------------------------
# T124 · Salvas de QRS estrecho >100 lpm: sinusal, TSV, TA, TAM y marcapasos
# errante. Sólo reglas y umbrales (sin entrenar nada). Umbrales de
# tratados/CRITERIOS-TRATADOS.md, sección D; se sobrescriben pasando un dict
# (p. ej. leído de un YAML de criterios) a ``classify_salva``.
# ---------------------------------------------------------------------------
SALVA_CRITERIA: dict[str, float] = {
    "min_beats": 3,               # salva = >= 3 S seguidas
    "rate_min_bpm": 100.0,        # por encima: sinusal, TSV, TA, TAM
    "jump_ratio": 0.25,           # cambio de RR > 25 % en un latido = borde súbito
    "regular_ratio": 1.15,        # RR máximo / RR mínimo dentro de la salva
    "p_sinus_corr": 0.80,         # correlación con la P sinusal: "P igual"
    "morph_corr": 0.90,           # misma morfología de P entre latidos
    "pr_conducted_max_ms": 250.0,  # PR conducido (TA)
    "rp_trin_ms": 90.0,           # RP < 90 ms: TRIN típico probable
    "pp_cv_irregular": 0.10,      # P-P irregular (TAM y marcapasos errante)
    "min_p_n": 3,                 # P necesarias para TA/TAM/errante
    "unreadable_frac": 0.0,       # cualquier solape con tramo ilegible: no se informa
    "p_sinus_frac_min": 0.80,     # fracción de P sinusal para llamarla sinusal
}

#: Grupo que usa la matriz TSV/TA/sinusal para cada etiqueta fina.
SALVA_GROUP = {
    "sinusal": "sinusal", "TSV": "TSV", "TA": "TA", "TAM": "TA",
    "marcapasos_errante": "TA", "indeterminada": "indeterminada",
    "irregular": "indeterminada", "no_informar": "no_informar",
}


def salva_criteria(criteria: dict | None = None) -> dict[str, float]:
    """Umbrales efectivos: los por defecto con las claves pasadas encima."""
    out = dict(SALVA_CRITERIA)
    if criteria:
        out.update({k: float(v) for k, v in criteria.items() if k in SALVA_CRITERIA})
    return out


def p_windows(x: np.ndarray, fs: float, p_peak_idx, half_ms: float = 60.0) -> np.ndarray:
    """Ventanas de ±``half_ms`` alrededor del pico de P de cada latido (NaN si no hay)."""
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(p_peak_idx, dtype=float).ravel()
    h = int(round(half_ms * fs / 1000.0))
    out = np.full((p.size, 2 * h + 1), np.nan)
    for i, c in enumerate(p):
        if not np.isfinite(c):
            continue
        c = int(c)
        if c - h < 0 or c + h >= x.size:
            continue
        out[i] = x[c - h:c + h + 1]
    return out


def p_morph_clusters(windows: np.ndarray, corr_min: float = 0.90) -> np.ndarray:
    """Agrupa ventanas de P (una fila por latido) por morfología (greedy).

    Cada ventana entra en el primer grupo cuya plantilla (la primera ventana
    del grupo) correlaciona ≥ ``corr_min``; si no, abre grupo nuevo. Las filas
    con NaN o planas quedan en -1 (P no medible).
    """
    w = np.asarray(windows, dtype=np.float64)
    n = w.shape[0] if w.ndim == 2 else 0
    labels = np.full(n, -1, dtype=int)
    reps: list[np.ndarray] = []
    for i in range(n):
        v = w[i]
        if not np.all(np.isfinite(v)) or float(np.std(v)) <= 1e-12:
            continue
        v = (v - v.mean()) / v.std()
        for g, r in enumerate(reps):
            if float(np.mean(v * r)) >= corr_min:
                labels[i] = g
                break
        else:
            reps.append(v)
            labels[i] = len(reps) - 1
    return labels


@dataclass(frozen=True)
class SalvaInput:
    """Datos de una salva (≥3 latidos seguidos) para :func:`classify_salva`.

    ``rr``: intervalos entre latidos de la salva (s), largo L-1.
    ``n_beats``: latidos de la salva (L).
    ``rr_pre`` / ``rr_post``: RR justo antes y después de la salva (s) o None.
    ``p_state``: por latido, -1 desconocida, 0 ausente, 1 presente (``svpb``).
    ``p_corr``: correlación de la P con la plantilla sinusal (NaN si no hay).
    ``pr_ms``: PR de cada latido (NaN si no hay P conducida).
    ``rp_ms``: RP (QRS→P retrógrada) por latido (NaN si no se mide).
    ``pp_ms``: P-P (ms) que termina en cada latido (NaN si no se mide).
    ``p_morph``: grupo de morfología de P (:func:`p_morph_clusters`, -1 = n/a).
    ``pr_sinus_ms``: PR sinusal del registro (ms) o None.
    ``unreadable``: fracción de la salva dentro de tramos ilegibles (0-1).
    """

    rr: np.ndarray
    n_beats: int
    rr_pre: float | None = None
    rr_post: float | None = None
    p_state: np.ndarray | None = None
    p_corr: np.ndarray | None = None
    pr_ms: np.ndarray | None = None
    rp_ms: np.ndarray | None = None
    pp_ms: np.ndarray | None = None
    p_morph: np.ndarray | None = None
    pr_sinus_ms: float | None = None
    unreadable: float = 0.0


def _vec(v, n: int, fill: float) -> np.ndarray:
    if v is None:
        return np.full(n, fill, dtype=np.float64)
    a = np.asarray(v, dtype=np.float64).ravel()
    return a if a.size == n else np.full(n, fill, dtype=np.float64)


def classify_salva(sv: SalvaInput, criteria: dict | None = None) -> dict:
    """Clasifica una salva de QRS estrecho en sinusal, TSV, TA, TAM o errante.

    Orden de las reglas (tratados, sección D):

    1. Solape con tramo ilegible (≥2 de 3 canales malos) → ``no_informar``.
    2. ≥3 morfologías de P con P-P irregular → ``TAM`` si >100 lpm, o
       ``marcapasos_errante`` si ≤100 lpm. (Son irregularmente irregulares, así
       que no se exige RR regular.)
    3. Bordes graduales (RR cambia ≤25 % por latido en inicio y fin) y P sinusal
       (≥80 % de los latidos con P evaluable) → ``sinusal``. Sin P sinusal
       confirmada no se llama sinusal.
    4. ≥3 P distintas de la sinusal con PR conducido y una sola morfología, en
       >100 lpm → ``TA``.
    5. RR irregular dentro de la salva (máx/mín > 1,15) → ``irregular``.
    6. >100 lpm con borde súbito (RR cambia >25 % en un latido) → ``TSV``, con
       subtipo ``TRIN`` si la P no es visible o RP < 90 ms, ``TRAV`` si
       90 ms ≤ RP < PR sinusal, y sin subtipo si RP no se puede medir.
    7. Cualquier otro caso → ``indeterminada``.

    Devuelve ``kind`` (etiqueta fina), ``group`` (sinusal/TSV/TA/…) y los
    indicadores que usaron las reglas, para informes y pruebas.
    """
    c = salva_criteria(criteria)
    n = int(sv.n_beats)
    rr = np.asarray(sv.rr, dtype=np.float64).ravel()
    rr = rr[np.isfinite(rr) & (rr > 0)]
    rate = 60.0 / float(np.mean(rr)) if rr.size else 0.0
    out: dict = {"length": n, "rate_bpm": round(rate, 1), "subtype": None,
                 "abrupt_onset": None, "abrupt_offset": None, "regular": None,
                 "p_known": 0, "p_sinus_frac": None, "p_distinct": 0,
                 "p_morph_n": 0, "pp_cv": None, "rp_median_ms": None}

    def done(kind: str) -> dict:
        out["kind"] = kind
        out["group"] = SALVA_GROUP[kind]
        return out

    if n < int(c["min_beats"]) or rr.size < 2:
        return done("indeterminada")
    if sv.unreadable > c["unreadable_frac"]:
        return done("no_informar")

    out["regular"] = bool(float(rr.max() / rr.min()) <= c["regular_ratio"])
    j = c["jump_ratio"]
    if sv.rr_pre is not None and sv.rr_pre > 0:
        out["abrupt_onset"] = bool(abs(rr[0] - sv.rr_pre) / sv.rr_pre > j)
    if sv.rr_post is not None and sv.rr_post > 0:
        out["abrupt_offset"] = bool(abs(sv.rr_post - rr[-1]) / rr[-1] > j)
    abrupt = out["abrupt_onset"] is True or out["abrupt_offset"] is True
    gradual = out["abrupt_onset"] is False and out["abrupt_offset"] is False

    p_state = _vec(sv.p_state, n, -1.0)
    p_corr = _vec(sv.p_corr, n, np.nan)
    pr = _vec(sv.pr_ms, n, np.nan)
    rp = _vec(sv.rp_ms, n, np.nan)
    pp = _vec(sv.pp_ms, n, np.nan)
    morph = (np.asarray(sv.p_morph).ravel().astype(int)
             if sv.p_morph is not None and np.asarray(sv.p_morph).size == n
             else np.full(n, -1, dtype=int))
    pr_sinus = float(sv.pr_sinus_ms) if sv.pr_sinus_ms is not None \
        and np.isfinite(sv.pr_sinus_ms) else None

    known = p_state >= 0
    present = p_state == 1
    sinus_p = present & np.isfinite(p_corr) & (p_corr >= c["p_sinus_corr"])
    distinct = present & np.isfinite(p_corr) & (p_corr < c["p_sinus_corr"])
    out["p_known"] = int(known.sum())
    out["p_distinct"] = int(distinct.sum())
    p_evaluable = known.sum() >= max(c["min_p_n"], 0.5 * n)
    if known.sum():
        out["p_sinus_frac"] = round(float(sinus_p.sum() / known.sum()), 3)
    p_sinus_like = bool(p_evaluable and out["p_sinus_frac"] >= c["p_sinus_frac_min"])
    n_morph = len({int(m) for m in morph[distinct] if m >= 0})
    out["p_morph_n"] = n_morph
    pp_ok = pp[np.isfinite(pp) & (pp > 0)]
    if pp_ok.size >= 2:
        out["pp_cv"] = round(float(np.std(pp_ok) / np.mean(pp_ok)), 3)
    pp_irregular = out["pp_cv"] is not None and out["pp_cv"] > c["pp_cv_irregular"]
    rp_ok = rp[np.isfinite(rp) & (rp > 0)]
    rp_med = float(np.median(rp_ok)) if rp_ok.size else None
    out["rp_median_ms"] = None if rp_med is None else round(rp_med, 1)

    # 2. multifocal: ≥3 morfologías de P con P-P irregular
    if n_morph >= 3 and pp_irregular:
        return done("TAM" if rate > c["rate_min_bpm"] else "marcapasos_errante")
    # 3. sinusal: bordes graduales y P igual a la sinusal
    if gradual and p_sinus_like:
        return done("sinusal")
    # 4. taquicardia auricular: P distinta de la sinusal, misma morfología, PR conducido
    conducted = np.isfinite(pr) & (pr <= c["pr_conducted_max_ms"])
    if rate > c["rate_min_bpm"] and (distinct & conducted).sum() >= c["min_p_n"] \
            and n_morph == 1:
        return done("TA")
    # 5. ritmo irregular que no es ninguna de las anteriores
    if not out["regular"]:
        return done("irregular")
    if rate <= c["rate_min_bpm"]:
        return done("indeterminada")
    # 6. TSV: borde súbito
    if abrupt:
        if rp_med is None:
            out["subtype"] = "TRIN" if p_evaluable else None
        elif rp_med < c["rp_trin_ms"]:
            out["subtype"] = "TRIN"
        elif pr_sinus is not None and rp_med < pr_sinus:
            out["subtype"] = "TRAV"
        return done("TSV")
    return done("indeterminada")


# ---------------------------------------------------------------------------
# T161 · Salvas S por criterio temporal, salva más larga y más rápida, y
# clasificación de salvas (T124) dentro de analyze(). Sin entrenar nada.
# Una salva S = ≥3 latidos seguidos, cada uno con prematuridad ≥15 % sobre la
# mediana de los RR sinusales previos O con P distinta de la sinusal (las dos
# condiciones no se exigen a la vez). No depende de la etiqueta S.
# ---------------------------------------------------------------------------
S_SALVA: dict[str, float] = {
    "min_beats": 3,               # salva = >= 3 latidos seguidos
    "premature_frac": 0.15,       # prematuridad >= 15 % sobre la mediana sinusal
    "n_sinus_rr": 8,              # RR sinusales previos para la mediana
    "min_sinus_rr": 3,            # mínimo de RR sinusales para juzgar prematuridad
    "narrow_qrs_s": 0.110,        # QRS estrecho ≤110 ms (tratados H: QRS anormal >110 ms)
    "long_beats": 20,             # salva larga: >= 20 latidos
}

#: Grupos de salva que se informan como evento en analyze() (filas D10–D12).
SALVA_EVENT_GROUPS = ("TA", "TAM", "marcapasos_errante")


def _true_runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Tramos [a, b) de True consecutivos de largo ≥ ``min_len``."""
    out: list[tuple[int, int]] = []
    m = np.asarray(mask, dtype=bool).ravel()
    i, n = 0, m.size
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def s_salva_eligible(times, labels, *, width_s=None, p_state=None, p_corr=None,
                     bad=None, criteria: dict | None = None) -> np.ndarray:
    """Latidos candidatos a salva S (máscara).

    ``labels``: etiqueta por latido ("N", "S", "V", "F", "Q"). Cuentan los N y S,
    y los V con QRS estrecho (≤ ``narrow_qrs_s``): un V ancho no es supraventricular.
    ``bad``: latidos en tramo ilegible o marcados artefacto: nunca cuentan.
    Un latido cuenta si es prematuro (RR previo ≤ (1 - ``premature_frac``) × la
    mediana de los últimos ``n_sinus_rr`` RR entre dos N) o si su P es distinta
    de la sinusal (presente y correlación < ``salva_criteria()['p_sinus_corr']``).
    """
    c = dict(S_SALVA)
    c.update({k: float(v) for k, v in (criteria or {}).items() if k in S_SALVA})
    t = np.asarray(times, dtype=np.float64).ravel()
    n = t.size
    lab = np.asarray(labels).astype(str).ravel()
    w = _vec(width_s, n, np.nan)
    ps = _vec(p_state, n, -1.0)
    pc = _vec(p_corr, n, np.nan)
    badm = np.zeros(n, dtype=bool) if bad is None else np.asarray(bad, dtype=bool).ravel()
    narrow = np.isfinite(w) & (w <= c["narrow_qrs_s"])
    shape_ok = (np.isin(lab, ["N", "S"]) | ((lab == "V") & narrow)) & ~badm
    prem = np.zeros(n, dtype=bool)
    sinus: list[float] = []
    for i in range(1, n):
        rr = t[i] - t[i - 1]
        if len(sinus) >= c["min_sinus_rr"]:
            med = float(np.median(sinus[-int(c["n_sinus_rr"]):]))
            prem[i] = rr <= (1.0 - c["premature_frac"]) * med
        if lab[i] == "N" and lab[i - 1] == "N" and not badm[i] and not badm[i - 1]:
            sinus.append(rr)
    p_distinct = (ps == 1) & np.isfinite(pc) & (pc < salva_criteria()["p_sinus_corr"])
    return shape_ok & (prem | p_distinct)


def s_salva_runs(times, labels, criteria: dict | None = None, **kw) -> list[tuple[int, int]]:
    """Salvas S como tramos [a, b) de latidos (≥ ``min_beats`` seguidos)."""
    c = dict(S_SALVA)
    c.update({k: float(v) for k, v in (criteria or {}).items() if k in S_SALVA})
    elig = s_salva_eligible(times, labels, criteria=criteria, **kw)
    return _true_runs(elig, int(c["min_beats"]))


def s_salva_summary(times, runs: list[tuple[int, int]],
                    criteria: dict | None = None) -> dict:
    """Número de salvas, la más larga y la más rápida, y las ≥ ``long_beats``."""
    c = dict(S_SALVA)
    c.update({k: float(v) for k, v in (criteria or {}).items() if k in S_SALVA})
    t = np.asarray(times, dtype=np.float64).ravel()
    items = []
    for a, b in runs:
        seg = t[a:b]
        mean_rr = float(np.mean(np.diff(seg))) if b - a >= 2 else 0.0
        items.append({"n": int(b - a), "start_s": round(float(seg[0]), 1),
                      "end_s": round(float(seg[-1]), 1),
                      "rate_bpm": round(60.0 / mean_rr, 1) if mean_rr > 0 else 0.0})
    longest = max(items, key=lambda s: s["n"]) if items else None
    fastest = max(items, key=lambda s: s["rate_bpm"]) if items else None
    return {"n_salvas": len(items),
            "n_long": int(sum(s["n"] >= c["long_beats"] for s in items)),
            "longest": longest, "fastest": fastest, "items": items}


def salva_morphology(x: np.ndarray | None, fs: float, p_peak, present) -> np.ndarray:
    """Grupo de morfología de P por latido (-1 si no hay P o no hay señal)."""
    n = len(present)
    if x is None or p_peak is None:
        return np.full(n, -1, dtype=int)
    pk = np.where(np.asarray(present, dtype=bool) & (np.asarray(p_peak) >= 0),
                  np.asarray(p_peak, dtype=float), np.nan)
    return p_morph_clusters(p_windows(x, fs, pk), corr_min=salva_criteria()["morph_corr"])


def s_salva_layer(times, labels, *, width_s=None, p_state=None, p_corr=None,
                  pr_ms=None, p_peak=None, fs: float = 0.0, x=None, pr_sinus_ms=None,
                  bad=None, criteria: dict | None = None) -> dict:
    """Salvas S de un registro: resumen, clasificación (T124) y eventos.

    ``x``: señal filtrada del canal de análisis (para la morfología de P);
    ``p_peak``: índice de pico de P por latido (muestras, -1 si no hay).
    Devuelve ``summary`` (para el informe), ``groups`` (conteo por grupo) y
    ``events`` (tuplas ``(a, b, kind, n, rate_bpm)`` de TA, TAM y marcapasos
    errante; la S no genera evento, solo el resumen).
    """
    t = np.asarray(times, dtype=np.float64).ravel()
    n = t.size
    labs = np.asarray(labels).astype(str).ravel()
    ps = _vec(p_state, n, -1.0)
    pc = _vec(p_corr, n, np.nan)
    prm = _vec(pr_ms, n, np.nan)
    runs = s_salva_runs(t, labs, criteria=criteria, width_s=width_s, p_state=ps,
                        p_corr=pc, bad=bad)
    morph = salva_morphology(x, fs, p_peak, ps == 1) if runs else np.full(n, -1, int)
    pp_ms = np.full(n, np.nan)
    if p_peak is not None and fs > 0:
        pk = _vec(p_peak, n, -1.0)
        last = -1
        for i in range(n):
            if ps[i] == 1 and pk[i] >= 0:
                if last >= 0:
                    pp_ms[i] = (pk[i] - pk[last]) * 1000.0 / fs
                last = i
    rp_ms = np.full(n, np.nan)
    rr_all = np.diff(t, append=np.nan) * 1000.0
    for i in range(n - 1):
        if ps[i + 1] == 1 and np.isfinite(prm[i + 1]) and np.isfinite(rr_all[i]):
            v = rr_all[i] - prm[i + 1]
            if v > 0:
                rp_ms[i] = v
    summ = s_salva_summary(t, runs, criteria)
    groups: Counter = Counter()
    events = []
    for k, (a, b) in enumerate(runs):
        sv = SalvaInput(
            rr=np.diff(t[a:b]), n_beats=b - a,
            rr_pre=float(t[a] - t[a - 1]) if a >= 1 else None,
            rr_post=float(t[b] - t[b - 1]) if b < n else None,
            p_state=ps[a:b], p_corr=pc[a:b], pr_ms=prm[a:b], rp_ms=rp_ms[a:b],
            pp_ms=pp_ms[a:b], p_morph=morph[a:b], pr_sinus_ms=pr_sinus_ms)
        res = classify_salva(sv, criteria)
        groups[res["kind"]] += 1
        summ["items"][k]["kind"] = res["kind"]  # mismo dict que longest/fastest
        if res["group"] in SALVA_EVENT_GROUPS:
            events.append((float(t[a]), float(t[b - 1]), res["kind"], b - a,
                           float(res["rate_bpm"])))
    summ.pop("items")  # longest/fastest comparten el dict con kind ya asignado
    summ["groups"] = dict(groups)
    return {"summary": summ, "groups": dict(groups), "events": events}


# ---------------------------------------------------------------------------
# Disociación AV isorrítmica (T164; fila G7 de CRITERIOS-TRATADOS)
# ---------------------------------------------------------------------------
ISO_MIN_BEATS = 10         # latidos conducidos seguidos
ISO_RATE_TOL_BPM = 5.0     # frecuencia auricular y ventricular dentro de ±5 lpm
ISO_PR_RANGE_MS = 40.0     # el PR varía más de 40 ms dentro del tramo


def isorrhythmic_runs(p_s, conducted, qrs_s, pr_ms, *, min_beats: int = ISO_MIN_BEATS,
                      rate_tol_bpm: float = ISO_RATE_TOL_BPM,
                      pr_range_ms: float = ISO_PR_RANGE_MS) -> list[dict]:
    """Tramos de disociación AV isorrítmica.

    Entradas, una por P y en orden: tiempo del pico de P (s), si conduce, tiempo
    del inicio del QRS que sigue (s; NaN si no conduce) y PR (ms; NaN si no conduce).

    Un tramo es una racha de P **consecutivas que conducen** (una P bloqueada la
    corta: por eso en el tramo no hay P bloqueadas) con al menos ``min_beats``
    latidos conducidos, en la que:

    * la frecuencia auricular (60 / media del PP) y la ventricular (60 / media del
      RR de los QRS) difieren como mucho ``rate_tol_bpm``. Se usa la media, no la
      mediana: con PR variable los RR de los QRS alternan y la mediana depende de la
      paridad;
    * el PR varía más de ``pr_range_ms`` (máximo menos mínimo).

    Esto no es un bloqueo AV: el tramo se informa aparte y no cuenta como BAV.
    Devuelve dicts con ``start_s``, ``end_s``, ``n``, ``a_lpm``, ``v_lpm``,
    ``pr_min_ms`` y ``pr_max_ms``.
    """
    p = np.asarray(p_s, dtype=np.float64).ravel()
    cond = np.asarray(conducted, dtype=bool).ravel()
    q = np.asarray(qrs_s, dtype=np.float64).ravel()
    pr = np.asarray(pr_ms, dtype=np.float64).ravel()
    out: list[dict] = []
    i, n = 0, p.size
    while i < n:
        if not cond[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and cond[j + 1]:
            j += 1
        ids = np.arange(i, j + 1)
        if ids.size >= min_beats:
            pp = np.diff(p[ids])
            qv = q[ids]
            qv = qv[np.isfinite(qv)]
            rr = np.diff(qv)
            rr = rr[(rr > 0.20) & (rr < 3.0)]
            prv = pr[ids][np.isfinite(pr[ids])]
            if pp.size and rr.size and prv.size and np.mean(pp) > 0 and np.mean(rr) > 0:
                a_lpm = 60.0 / float(np.mean(pp))
                v_lpm = 60.0 / float(np.mean(rr))
                rng = float(prv.max() - prv.min())
                if abs(a_lpm - v_lpm) <= rate_tol_bpm and rng > pr_range_ms:
                    out.append({
                        "start_s": float(p[i]), "end_s": float(p[j]), "n": int(ids.size),
                        "a_lpm": round(a_lpm, 1), "v_lpm": round(v_lpm, 1),
                        "pr_min_ms": round(float(prv.min()), 1),
                        "pr_max_ms": round(float(prv.max()), 1),
                    })
        i = j + 1
    return out
