"""Fusión multicanal y veto cruzado de artefactos.

Con 3 derivaciones simultáneas se puede hacer algo que un analizador de un
solo canal no puede: comprobar si un latido está presente en los demás
canales. Un pico que solo existe en el canal analizado es un artefacto de
electrodo, no una extrasístole.

    presencia: energía del QRS en ±25 ms por encima del 25 % del p99 del canal
    veto:     si aparece en menos de 2 de 3 canales, se marca como artefacto
"""
from __future__ import annotations

import numpy as np


def lead_quality(xf: np.ndarray, fs: float) -> float:
    """Calidad aproximada de un canal (0 = muerto/plano, 1 = limpio)."""
    xf = np.asarray(xf, dtype=float)
    if xf.size < 4 * fs:
        return 0.0
    sd = float(np.std(xf))
    if sd < 5e-4:          # señal plana o desconectada
        return 0.0
    d = np.abs(np.diff(xf))
    flat = float(np.mean(d < 1e-7))
    if flat > 0.6:         # saturada o plana la mayor parte del tiempo
        return 0.0
    return float(np.clip(1.0 - flat, 0.05, 1.0))


def energy_envelope(xf: np.ndarray, fs: float, smooth_ms: float = 30.0) -> np.ndarray:
    d = np.abs(np.diff(np.asarray(xf, dtype=float), prepend=xf[:1]))
    w = max(1, int(smooth_ms / 1000.0 * fs))
    if w > 1:
        d = np.convolve(d, np.ones(w) / w, mode="same")
    return d


def qrs_presence(xf: np.ndarray, fs: float, idx: np.ndarray,
                 half_ms: float = 25.0, rel: float = 0.25) -> np.ndarray:
    """Para cada latido, si hay energía de QRS en ese canal."""
    idx = np.asarray(idx, dtype=int)
    out = np.zeros(idx.size, dtype=bool)
    if idx.size == 0 or xf.size < 10:
        return out
    env = energy_envelope(xf, fs)
    p99 = float(np.percentile(env, 99))
    if p99 <= 0:
        return out
    thr = rel * p99
    half = max(1, int(round(half_ms / 1000.0 * fs)))
    for k, p in enumerate(idx):
        a = max(0, p - half)
        b = min(env.size, p + half + 1)
        if b > a:
            out[k] = bool(env[a:b].max() > thr)
    return out


def veto_mask(presences: list[np.ndarray], n_leads: int) -> np.ndarray:
    """Veto: el latido debe aparecer en al menos 2 canales.

    Con menos de 3 canales no se aplica (no hay testigo suficiente).
    """
    if n_leads < 3 or not presences:
        return np.zeros(presences[0].size if presences else 0, dtype=bool)
    count = np.sum(np.stack(presences, axis=0), axis=0)
    return count < 2

