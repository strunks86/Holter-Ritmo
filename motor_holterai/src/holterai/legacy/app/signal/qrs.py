"""Detección de complejos QRS.

Variante de Pan-Tompkins adaptada a Holter: banda 5-18 Hz, derivada al
cuadrado, integración en ventana de 120 ms y umbral adaptativo con
búsqueda retrospectiva para latidos perdidos.

El umbral es doble: una referencia **local** (mediana móvil de la energía de
5 s) y una referencia **global robusta** (`energy_scale`, mediana del máximo de
cada bloque de 5 s). Un artefacto aislado (movimiento, electrodo que se
despega) no cambia esa mediana, mientras que un percentil global como p99,5 se
dispara y borra todos los QRS del tramo: con el registro completo el artefacto
se diluye y con una exportación de 600 s no.
"""
from __future__ import annotations

import numpy as np

from .filters import filtfilt, lead_params, moving_average

#: energía mínima (fracción de `energy_scale`) para admitir candidatos
FLOOR_FRAC = 0.05
#: cuántos RR medianos debe cubrir un hueco para buscar un latido perdido
SEARCHBACK_RR = 1.6
#: prominencia mínima (fracción del umbral) del pico recuperado
SEARCHBACK_FRAC = 0.35


def energy_from_signal(y: np.ndarray, fs: float) -> np.ndarray:
    """Envoltura de energía a partir de la señal ya filtrada (5-18 Hz)."""
    d = np.diff(np.asarray(y, dtype=np.float64), prepend=np.asarray(y[:1], dtype=np.float64))
    win = max(3, int(round(0.12 * fs)))
    return moving_average(d * d, win)


def qrs_energy(x: np.ndarray, fs: float) -> np.ndarray:
    """Envoltura de energía usada para localizar QRS."""
    y = filtfilt(x, lead_params(fs)["detect"])
    return energy_from_signal(y, fs)


def energy_scale(e: np.ndarray, fs: float, block_s: float = 5.0) -> float:
    """Escala robusta de energía: mediana del máximo de cada bloque de 5 s.

    Equivale a la energía de un QRS típico. A diferencia de un percentil
    global (p99,5), no la alteran los artefactos de pocos bloques.
    """
    maxima = _block_maxima(np.asarray(e, dtype=np.float64).ravel(), fs, block_s)
    if maxima.size == 0:
        return float(np.max(e)) if np.size(e) else 0.0
    return float(np.median(maxima))


def _block_maxima(e: np.ndarray, fs: float, block_s: float = 5.0) -> np.ndarray:
    bs = max(16, int(round(block_s * fs)))
    n = e.size // bs
    if n >= 1:
        return e[: n * bs].reshape(n, bs).max(axis=1)
    return np.asarray([e.max()]) if e.size else np.asarray([])


def local_energy_scale(e: np.ndarray, fs: float, block_s: float = 5.0,
                       half_blocks: int = 4) -> np.ndarray:
    """Escala de energía local: mediana móvil del máximo de cada bloque.

    Una racha de artefactos sube el suelo solo mientras dura (unos minutos),
    en vez de anular la detección de todo el bloque de análisis.
    """
    e = np.asarray(e, dtype=np.float64).ravel()
    n = e.size
    if n == 0:
        return e
    maxima = _block_maxima(e, fs, block_s)
    if maxima.size < 3:
        return np.full(n, float(maxima.max()) if maxima.size else 0.0)
    k = max(1, int(half_blocks))
    w = 2 * k + 1
    if maxima.size >= w:
        pad = np.pad(maxima, k, mode="edge")
        vals = np.median(np.lib.stride_tricks.sliding_window_view(pad, w), axis=1)
    else:
        vals = np.full(maxima.size, float(np.median(maxima)))
    centers = (np.arange(maxima.size) + 0.5) * max(16, int(round(block_s * fs)))
    return np.interp(np.arange(n), centers, vals)


def lead_score(x: np.ndarray, fs: float) -> float:
    """Puntuación de calidad de una derivación para detectar QRS.

    Alta cuando el QRS domina sobre el ruido de fondo (ratio p99/p50) y
    la distribución de energía es picuda (kurtosis).
    """
    e = qrs_energy(x, fs)
    if e.size < 50:
        return 0.0
    p50 = float(np.percentile(e, 50))
    p99 = float(np.percentile(e, 99))
    if p50 <= 0:
        p50 = 1e-12
    ratio = p99 / p50
    sd = float(np.std(e)) + 1e-12
    kurt = float(np.mean(((e - np.mean(e)) / sd) ** 4))
    return float(np.log1p(ratio) * np.log1p(max(0.0, kurt)))


def yield_score(peaks: np.ndarray, fs: float, duration_s: float) -> float:
    """Plausibilidad de la detección de una derivación (0-1).

    Penaliza tanto los canales sin detecciones como los que entregan una
    taquicardia inverosímil (ruido tomado por QRS).
    """
    if duration_s <= 0 or np.size(peaks) < 4:
        return 0.0
    rr = np.diff(np.asarray(peaks, dtype=np.float64)) / float(fs)
    valid = float(np.mean((rr > 0.30) & (rr < 2.0)))
    rate = np.size(peaks) / (duration_s / 60.0)
    if not 25.0 <= rate <= 240.0:
        return 0.35 * valid
    return valid


def _local_maxima(x: np.ndarray, thr: float, min_dist: int) -> np.ndarray:
    if x.size < 3:
        return np.array([], dtype=int)
    c = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] > thr))[0] + 1
    if c.size == 0:
        return c
    # selección voraz por altura respetando la distancia mínima
    order = c[np.argsort(x[c])[::-1]]
    taken = np.zeros(x.size, dtype=bool)
    keep: list[int] = []
    for i in order:
        if taken[i]:
            continue
        keep.append(int(i))
        lo = max(0, i - min_dist)
        hi = min(x.size, i + min_dist + 1)
        taken[lo:hi] = True
    return np.sort(np.asarray(keep, dtype=int))


def _refine(y: np.ndarray, idx: np.ndarray, fs: float, win_s: float = 0.10) -> np.ndarray:
    """Ajusta cada pico al máximo local de la señal filtrada."""
    w = max(1, int(round(win_s * fs)))
    out = idx.copy()
    for k, i in enumerate(idx):
        a = max(0, i - w)
        b = min(y.size, i + w + 1)
        if b > a:
            out[k] = a + int(np.argmax(np.abs(y[a:b])))
    return out


def _running_median(x: np.ndarray, fs: float, block_s: float = 5.0) -> np.ndarray:
    """Mediana local por bloques (rápida) interpolada a la longitud original."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    bs = max(16, int(block_s * fs))
    if n == 0:
        return x
    n_full = n // bs
    vals: list[float] = []
    centers: list[float] = []
    if n_full:
        med = np.median(x[: n_full * bs].reshape(n_full, bs), axis=1)
        vals = list(med)
        centers = list((np.arange(n_full) + 0.5) * bs)
    rest = x[n_full * bs:]
    if rest.size:
        vals.append(float(np.median(rest)))
        centers.append(n_full * bs + rest.size / 2.0)
    if len(vals) == 1:
        return np.full(n, float(vals[0]))
    return np.interp(np.arange(n), np.asarray(centers), np.asarray(vals))


def relative_energy(e: np.ndarray, fs: float) -> np.ndarray:
    """Energía relativa a la mediana local, con suelo robusto.

    El suelo (`FLOOR_FRAC` × escala local) evita picos en tramos casi planos,
    donde la mediana local es minúscula y el cociente se vuelve enorme.
    """
    e = np.asarray(e, dtype=np.float64)
    base = _running_median(e, fs, 5.0)
    rel = e / np.maximum(base, 1e-12)
    ok = e >= FLOOR_FRAC * local_energy_scale(e, fs)
    return np.where(ok, rel, 0.0)


def _peaks(signal_rel: np.ndarray, y: np.ndarray, fs: float,
           sens: float) -> np.ndarray:
    min_dist = max(1, int(round(0.20 * fs)))
    cand = _local_maxima(signal_rel, sens, min_dist)
    if cand.size == 0:
        return cand
    peaks = _refine(y, cand, fs)
    # de-duplicación por proximidad conservando el de mayor amplitud
    keep = [int(peaks[0])]
    for p in peaks[1:]:
        if p - keep[-1] < int(0.24 * fs):
            if abs(y[p]) > abs(y[keep[-1]]):
                keep[-1] = int(p)
        else:
            keep.append(int(p))
    return np.asarray(keep, dtype=int)


def _searchback(signal_rel: np.ndarray, peaks: np.ndarray, fs: float,
                thr: float) -> np.ndarray:
    """Busca latidos perdidos en intervalos RR anormalmente largos."""
    rr = np.diff(peaks)
    if rr.size < 3:
        return peaks
    med = float(np.median(rr))
    if med <= 0:
        return peaks
    extra: list[int] = []
    for i, r in enumerate(rr):
        if r <= SEARCHBACK_RR * med:
            continue
        a, b = int(peaks[i]), int(peaks[i + 1])
        seg = signal_rel[a:b]
        if seg.size < 4:
            continue
        j = int(np.argmax(seg))
        if (seg[j] > SEARCHBACK_FRAC * thr and (a + j) - a > 0.22 * fs
                and b - (a + j) > 0.22 * fs):
            extra.append(a + j)
    if not extra:
        return peaks
    return np.sort(np.concatenate([peaks, np.asarray(extra, dtype=int)]))


def detect_qrs(x: np.ndarray, fs: float, searchback: bool = True,
               sens: float = 2.6) -> np.ndarray:
    """Devuelve los índices de muestra de los picos R detectados.

    El umbral es relativo a la energía de fondo local, de modo que cambios
    bruscos de amplitud (PVC, artefactos) no oculten los latidos normales.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < int(0.5 * fs):
        return np.array([], dtype=int)

    y = filtfilt(x, lead_params(fs)["detect"])
    e = energy_from_signal(y, fs)
    scale = energy_scale(e, fs)
    if scale <= 0 or scale <= float(np.percentile(e, 50)) * 1.05:  # señal plana
        return np.array([], dtype=int)

    signal_rel = relative_energy(e, fs)
    peaks = _peaks(signal_rel, y, fs, sens)
    if searchback and peaks.size >= 4:
        peaks = _searchback(signal_rel, peaks, fs, sens)
    return peaks


def _lead_peaks(x: np.ndarray, fs: float, sens: float):
    """Picos de una derivación con su prominencia relativa (para fusionar)."""
    y = filtfilt(x, lead_params(fs)["detect"])
    e = energy_from_signal(y, fs)
    scale = energy_scale(e, fs)
    if scale <= 0 or scale <= float(np.percentile(e, 50)) * 1.05:
        return np.array([], dtype=int), np.array([], dtype=float), e
    signal_rel = relative_energy(e, fs)
    peaks = _peaks(signal_rel, y, fs, sens)
    if peaks.size >= 4:
        peaks = _searchback(signal_rel, peaks, fs, sens)
    w = max(1, int(round(0.05 * fs)))
    vals = np.asarray([float(signal_rel[max(0, p - w):p + w + 1].max()) for p in peaks],
                      dtype=float)
    return peaks, vals, e


def rank_leads(signals: np.ndarray, fs: float, probe_s: float = 300.0,
               min_rate: float = 25.0, max_rate: float = 240.0) -> list[dict]:
    """Ordena las derivaciones por calidad de *detección* en una sonda.

    Usa la fusión multicanal como consenso: la mejor derivación es la que más
    se parece al consenso (recupera lo que las otras ven y no inventa picos
    propios). Devuelve una lista de dicts con `lead`, `rate`, `rate_ok`,
    `valid`, `recall`, `precision` y `score`.
    """
    signals = np.asarray(signals)
    if signals.ndim == 1:
        signals = signals[:, None]
    n = int(min(signals.shape[0], max(1, probe_s * fs)))
    x = np.asarray(signals[:n], dtype=np.float64)
    duration = n / float(fs)
    fused, _conf = detect_qrs_multilead(x, fs)
    win = max(1, int(round(0.10 * fs)))
    out: list[dict] = []
    for j in range(x.shape[1]):
        peaks = detect_qrs(x[:, j], fs)
        rr = np.diff(peaks) / float(fs)
        valid = float(np.mean((rr > 0.30) & (rr < 2.0))) if rr.size else 0.0
        rate = peaks.size / max(1e-9, duration / 60.0)
        rate_ok = bool(min_rate <= rate <= max_rate)
        recall = _near_fraction(fused, peaks, win) if fused.size and peaks.size else 0.0
        precision = _near_fraction(peaks, fused, win) if fused.size and peaks.size else 0.0
        agree = float(min(recall, precision))
        out.append({"lead": j, "rate": round(rate, 2), "rate_ok": rate_ok,
                    "valid": round(valid, 3), "recall": round(recall, 3),
                    "precision": round(precision, 3), "n": int(peaks.size),
                    "score": round(agree, 4)})
    return out


def _near_fraction(target: np.ndarray, other: np.ndarray, win: int) -> float:
    """Fracción de `target` que tiene un pico de `other` a ±win muestras."""
    if target.size == 0 or other.size == 0:
        return 0.0
    k = np.clip(np.searchsorted(other, target), 0, other.size - 1)
    left = np.clip(k - 1, 0, other.size - 1)
    d = np.minimum(np.abs(target - other[k]), np.abs(target - other[left]))
    return float(np.mean(d <= win))


def fuse_leads(signals: np.ndarray, fs: float, scores: np.ndarray | None = None) -> int:
    """Elige la derivación con mejor puntuación para la detección."""
    if signals.ndim == 1:
        return 0
    if scores is None:
        scores = np.array([lead_score(signals[:, i], fs) for i in range(signals.shape[1])])
    return int(np.argmax(scores))


def detect_qrs_multilead(signals: np.ndarray, fs: float, sens: float = 2.6,
                         sens_low: float = 1.4, merge_s: float = 0.10,
                         min_support: int = 2):
    """Detecta QRS en varias derivaciones y fusiona con regla de apoyo.

    Un candidato de baja prominencia (`sens_low`) se acepta si al menos
    `min_support` derivaciones lo confirman dentro de `merge_s`. Así un canal
    con ruido no inventa latidos (una espiga de un solo canal no se acepta,
    por alta que sea) y un canal con bajo voltaje no los pierde.

    Devuelve ``(picos, fracción_confirmada_por_derivación)``. Con una sola
    derivación equivale a `detect_qrs`.
    """
    signals = np.asarray(signals)
    if signals.ndim == 1:
        signals = signals[:, None]
    n_leads = int(signals.shape[1])
    if n_leads == 1:
        return detect_qrs(signals[:, 0], fs, sens=sens), np.asarray([1.0])

    groups = [_lead_peaks(np.asarray(signals[:, j], dtype=np.float64), fs, sens_low)
              for j in range(n_leads)]
    samples = [g[0] for g in groups if g[0].size]
    vals = [g[1] for g in groups if g[0].size]
    leadof = [np.full(g[0].size, j, dtype=np.int64)
              for j, g in enumerate(groups) if g[0].size]
    if not samples:
        return np.array([], dtype=int), np.zeros(n_leads)

    all_p = np.concatenate(samples)
    all_v = np.concatenate(vals)
    all_l = np.concatenate(leadof)
    order = np.argsort(all_p, kind="stable")
    all_p, all_v, all_l = all_p[order], all_v[order], all_l[order]

    win = max(1, int(round(merge_s * fs)))
    accepted: list[tuple[int, float]] = []
    i, n = 0, all_p.size
    while i < n:
        j = i
        while j + 1 < n and all_p[j + 1] - all_p[i] <= win:
            j += 1
        grp = slice(i, j + 1)
        support = len(set(all_l[grp].tolist()))
        k = i + int(np.argmax(all_v[grp]))
        strength = float(all_v[k])
        if support >= min_support:
            accepted.append((int(all_p[k]), strength))
        i = j + 1

    if not accepted:
        return np.array([], dtype=int), np.zeros(n_leads)
    accepted.sort()
    merged: list[tuple[int, float]] = []
    for p, v in accepted:
        if merged and p - merged[-1][0] < int(0.20 * fs):
            if v > merged[-1][1]:
                merged[-1] = (p, v)
            continue
        merged.append((p, v))
    peaks = np.asarray([p for p, _ in merged], dtype=np.int64)

    # fracción de los picos de cada derivación que la fusión confirmó
    confirmed = np.zeros(n_leads)
    for j, (p, _v, _e) in enumerate(groups):
        if p.size == 0:
            continue
        k = np.clip(np.searchsorted(peaks, p), 0, max(0, peaks.size - 1))
        left = np.clip(k - 1, 0, max(0, peaks.size - 1))
        d = np.minimum(np.abs(p - peaks[k]), np.abs(p - peaks[left]))
        confirmed[j] = float(np.mean(d <= win))
    return peaks, confirmed
