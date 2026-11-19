"""Marcapasos y resincronización (CRT).

Detección de espigas de estimulación y de los fallos clásicos:

  * **captura**: espiga seguida de QRS en < 200 ms;
  * **sensado (inhibición)**: pausa mayor que el intervalo programado;
  * **biventricular**: QRS ancho estimulado con V1 negativo (orientativo).

Limitación real: la espiga dura 0,5-2 ms. Si el Holter muestrea a 250 Hz y
aplica un filtro paso-bajo (lo habitual), la espiga **no está en la señal** y
no se puede detectar; el programa lo informa en vez de inventar latidos
estimulados.
"""
from __future__ import annotations

import numpy as np

from holterai.legacy.app.signal.filters import filtfilt, fir_highpass

SPIKE_HP = 40.0        # Hz: por encima del QRS
MAX_WIDTH_MS = 20.0
MIN_RATIO = 8.0        # veces el ruido de fondo


def detect_spikes(x: np.ndarray, fs: float) -> dict:
    """Busca espigas de estimulación en la señal cruda de un canal."""
    x = np.asarray(x, dtype=np.float64)
    out = {"found": False, "idx": np.zeros(0, dtype=int), "noise": 0.0,
           "threshold": 0.0, "fs": fs}
    if x.size < 4 * fs or fs < 120:
        return out
    hp = filtfilt(x, fir_highpass(fs, min(SPIKE_HP, 0.42 * fs), 101))
    env = np.abs(hp)
    noise = float(np.median(env)) * 1.4826 + 1e-9
    thr = MIN_RATIO * noise
    out["noise"] = noise
    out["threshold"] = thr
    if env.max() < thr:
        return out
    cand = np.flatnonzero((env[1:-1] > thr) & (env[1:-1] >= env[:-2]) & (env[1:-1] >= env[2:])) + 1
    if cand.size == 0:
        return out
    keep = []
    max_w = max(1, int(round(MAX_WIDTH_MS / 1000.0 * fs)))
    for i in cand:
        half = env[i] / 2.0
        lo = i
        while lo > 0 and env[lo - 1] > half and i - lo < max_w:
            lo -= 1
        hi = i
        while hi < env.size - 1 and env[hi + 1] > half and hi - i < max_w:
            hi += 1
        if hi - lo <= max_w:
            keep.append(int(i))
    out["idx"] = np.asarray(keep, dtype=int)
    out["amps"] = env[out["idx"]] if out["idx"].size else np.zeros(0)
    out["found"] = len(keep) > 0
    return out


def paced_flags(beat_idx: np.ndarray, spike_idx: np.ndarray, fs: float,
                pre_ms: float = 80.0, post_ms: float = 5.0) -> np.ndarray:
    """Marca los latidos inmediatamente precedidos por una espiga."""
    beat_idx = np.asarray(beat_idx, dtype=np.int64)
    flags = np.zeros(beat_idx.size, dtype=bool)
    if spike_idx.size == 0 or beat_idx.size == 0:
        return flags
    pre = int(round(pre_ms / 1000.0 * fs))
    post = int(round(post_ms / 1000.0 * fs))
    sp = np.sort(spike_idx)
    for k, b in enumerate(beat_idx):
        a = b - pre
        c = b + post
        j0 = np.searchsorted(sp, a)
        j1 = np.searchsorted(sp, c)
        flags[k] = j1 > j0
    return flags


def capture_failures(spike_idx: np.ndarray, beat_idx: np.ndarray, fs: float,
                     window_ms: float = 220.0) -> list[int]:
    """Espigas sin QRS posterior (fallo de captura)."""
    if spike_idx.size == 0:
        return []
    w = int(round(window_ms / 1000.0 * fs))
    beats = np.sort(np.asarray(beat_idx, dtype=np.int64))
    out = []
    for s in spike_idx:
        j = np.searchsorted(beats, s)
        nxt = beats[j] if j < beats.size else None
        prev = beats[j - 1] if j > 0 else None
        near = (nxt is not None and nxt - s <= w) or (prev is not None and s - prev <= w)
        if not near:
            out.append(int(s))
    return out


def drop_qrs_spikes(spike_idx: np.ndarray, beat_idx: np.ndarray, fs: float,
                    pre_ms: float = 8.0, post_ms: float = 60.0) -> np.ndarray:
    """Quita los picos que en realidad son el propio QRS.

    La espiga de estimulación va ~20-40 ms ANTES del inicio del QRS, así que
    se elimina lo que cae dentro del complejo.
    """
    if spike_idx.size == 0:
        return spike_idx
    beats = np.sort(np.asarray(beat_idx, dtype=np.int64))
    if beats.size == 0:
        return spike_idx
    pre = int(round(pre_ms / 1000.0 * fs))
    post = int(round(post_ms / 1000.0 * fs))
    keep = []
    for s in spike_idx:
        j = np.searchsorted(beats, s)
        bad = False
        for k in (j - 1, j):
            if 0 <= k < beats.size:
                b = beats[k]
                if b - pre <= s <= b + post:
                    bad = True
                    break
        if not bad:
            keep.append(int(s))
    return np.asarray(keep, dtype=int)


def merge_spikes(results: list[dict], fs: float, tol_ms: float = 6.0):
    """Une las espigas de varios canales, quitando duplicados."""
    pairs: list[tuple[int, float]] = []
    for r in results:
        idx = r.get("idx", np.zeros(0, dtype=int))
        amps = r.get("amps", np.zeros(len(idx)))
        if len(amps) != len(idx):
            amps = np.zeros(len(idx))
        pairs.extend((int(i), float(a)) for i, a in zip(idx, amps))
    if not pairs:
        return np.zeros(0, dtype=int), np.zeros(0)
    pairs.sort()
    tol = tol_ms / 1000.0 * fs
    merged: list[tuple[int, float]] = []
    for i, a in pairs:
        if merged and i - merged[-1][0] <= tol:
            if a > merged[-1][1]:
                merged[-1] = (i, a)
        else:
            merged.append((i, a))
    return (np.array([m[0] for m in merged], dtype=int),
            np.array([m[1] for m in merged], dtype=float))


def summarize(spikes: dict, beats, fs: float, wide_frac: float,
              v1_negative: bool, qrs_amp: float = 0.0) -> dict:
    """Resumen del módulo de marcapasos/CRT.

    qrs_amp: amplitud mediana del QRS en el canal (mV). Una espiga real tiene
    un transitorio apreciable respecto al QRS; si no lo supera, el registro no
    las conserva y se informa en vez de inventar estimulación.
    """
    beat_idx = np.array([b.idx for b in beats], dtype=np.int64)
    idx = spikes.get("idx", np.zeros(0, dtype=int))
    amps = spikes.get("amps", np.zeros(len(idx)))
    if idx.size and amps.size == idx.size:
        # espiga real: transitorio > 5 % del QRS (mínimo absoluto 0,01 mV)
        need = max(0.05 * float(qrs_amp), 0.010)
        keep = amps >= need
        idx = idx[keep]
        amps = amps[keep]
    spike_idx = drop_qrs_spikes(idx, beat_idx, fs)
    spikes = {**spikes, "idx": spike_idx, "found": spike_idx.size > 0}
    flags = paced_flags(beat_idx, spike_idx, fs)
    n = max(1, len(beats))
    res = {
        "spikes_detected": bool(spikes.get("found")),
        "n_spikes": int(spike_idx.size),
        "n_paced": int(flags.sum()),
        "pct_paced": round(100.0 * float(flags.sum()) / n, 1),
        "noise_mv": round(float(spikes.get("noise", 0.0)), 4),
    }
    if fs < 200:
        res["spikes_detected"] = False
        res["n_paced"] = 0
        res["pct_paced"] = 0.0
        res["note"] = (f"frecuencia insuficiente ({fs:.0f} Hz): las espigas de "
                       "marcapasos no son resolubles")
        res["capture_failures"] = 0
        res["capture_failure_samples"] = []
        return res
    if spike_idx.size >= 4:
        d = np.diff(np.sort(spike_idx)) / fs
        d = d[(d > 0.2) & (d < 3.0)]
        res["pacing_rate"] = round(float(60.0 / np.median(d)), 1) if d.size else None
        res["pacing_cv"] = round(float(np.std(d) / max(np.mean(d), 1e-6)), 3) if d.size else None
    # un tren de estimulación es regular; picos dispersos son ruido
    if spike_idx.size >= 5 and (res.get("pacing_cv") or 1.0) > 0.15:
        res["spikes_detected"] = False
        res["n_paced"] = 0
        res["pct_paced"] = 0.0
        res["note"] = "picos irregulares compatibles con ruido, no con estimulación"
        res["capture_failures"] = 0
        res["capture_failure_samples"] = []
        return res
    fails = capture_failures(spike_idx, beat_idx, fs)
    res["capture_failures"] = len(fails)
    res["capture_failure_samples"] = [int(x) for x in fails[:20]]
    if spike_idx.size == 0:
        res["note"] = ("sin espigas visibles: el registro no las conserva "
                       "(250 Hz o filtro del grabador)")
    if res["n_paced"] and wide_frac > 0.5 and v1_negative:
        res["crt_suspected"] = True
        res["note"] = "estimulación con QRS ancho y V1 negativo: biventricular probable"
    return res

