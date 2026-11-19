"""Generador sintético de ECG tipo Holter multi-derivación.

Se usa para (a) entrenar el clasificador de latidos y el detector de FA,
(b) disponer de casos de prueba con verdad conocida y (c) validar el
pipeline completo sin depender de datos externos.

El modelo es una suma de ondas gaussianas (P, Q, R, S, T) con morfologías
distintas por clase y por derivación, más ruido fisiológico y de equipo.
"""
from __future__ import annotations

import numpy as np

CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}

# ---------------------------------------------------------------------------
# Morfologías
# ---------------------------------------------------------------------------
# Cada componente: (mu_s, sigma_s, amp_base)
# mu y sigma en segundos respecto del pico R.
_MORPH_N = {
    "P": (-0.16, 0.026, 0.11),
    "Q": (-0.022, 0.007, -0.10),
    "R": (0.0, 0.0095, 1.00),
    "S": (0.024, 0.009, -0.22),
    "T": (0.27, 0.055, 0.29),
}
_MORPH_S = {  # ectópico supraventricular: parecido a N, P precoz/anómala
    "P": (-0.19, 0.024, 0.07),
    "Q": (-0.020, 0.007, -0.08),
    "R": (0.0, 0.0105, 0.92),
    "S": (0.026, 0.010, -0.20),
    "T": (0.26, 0.055, 0.25),
}
_MORPH_V = {  # ectópico ventricular: complejo ancho, sin P, T discordante
    "P": None,
    "Q": (-0.050, 0.022, -0.30),
    "R": (0.0, 0.048, 1.15),
    "S": (0.085, 0.040, -0.90),
    "T": (0.34, 0.095, -0.50),
}
_MORPH_F = {  # fusión: mezcla N/V
    "P": (-0.15, 0.026, 0.05),
    "Q": (-0.026, 0.011, -0.22),
    "R": (0.0, 0.020, 1.20),
    "S": (0.038, 0.016, -0.55),
    "T": (0.28, 0.065, -0.10),
}
_MORPH_Q = {  # artefacto / no clasificable
    "P": None,
    "Q": (-0.030, 0.020, -0.60),
    "R": (0.0, 0.045, 0.90),
    "S": (0.060, 0.035, -0.70),
    "T": (0.25, 0.10, 0.20),
}

_MORPHS = {"N": _MORPH_N, "S": _MORPH_S, "V": _MORPH_V, "F": _MORPH_F, "Q": _MORPH_Q}


def _lead_gain(comp: str, lead: int, kind: str, rng: np.random.Generator) -> float:
    """Ganancia de cada componente por derivación (aprox. II / V1 / V5)."""
    base = {
        # (P, Q, R, S, T)
        0: (1.00, 1.00, 1.00, 1.00, 1.00),
        1: (0.60, 0.25, 0.32, 2.10, 0.42),
        2: (0.90, 0.55, 1.18, 0.70, 1.15),
    }.get(lead % 3, (1.0, 1.0, 1.0, 1.0, 1.0))
    idx = {"P": 0, "Q": 1, "R": 2, "S": 3, "T": 4}[comp]
    g = base[idx]
    if kind == "V":  # en V1 el PVC suele ser negativo y en V5 positivo
        g *= {0: 1.0, 1: -0.9, 2: 1.05}.get(lead % 3, 1.0)
    # variabilidad latido a latido
    return g * (1.0 + rng.normal(0.0, 0.06))


def beat_waveform(t_rel: np.ndarray, kind: str, lead: int,
                  rng: np.random.Generator, stretch: float = 1.0) -> np.ndarray:
    """Forma de onda de un latido en una derivación, en mV."""
    if kind not in _MORPHS:
        kind = "N"
    out = np.zeros_like(t_rel, dtype=np.float64)
    for comp, spec in _MORPHS[kind].items():
        if spec is None:
            continue
        mu, sig, amp = spec
        g = _lead_gain(comp, lead, kind, rng)
        # pequeñas variaciones de repolarización
        amp_l = amp * g * (1.0 + rng.normal(0.0, 0.05))
        if comp == "T":
            amp_l *= 1.0 + rng.normal(0.0, 0.08)
        out += amp_l * np.exp(-0.5 * ((t_rel - mu * stretch) / (sig * stretch)) ** 2)
    return out


# ---------------------------------------------------------------------------
# Ritmos
# ---------------------------------------------------------------------------
def rr_sequence(mode: str, n: int, hr_bpm: float, rng: np.random.Generator) -> np.ndarray:
    """Serie de intervalos RR (s) para un modo de ritmo."""
    base = 60.0 / max(30.0, hr_bpm)
    if mode == "af":
        rr = base * (0.75 + 1.15 * rng.gamma(2.0, 0.5, n))
        rr = np.clip(rr, 0.32, 1.65)
        return rr
    if mode == "tachy":
        base = 60.0 / 125.0
    elif mode == "brady":
        base = 60.0 / 44.0
    # sinusal con arritmia respiratoria
    phase = rng.uniform(0, 2 * np.pi)
    t = np.cumsum(np.full(n, base))
    rr = base * (1.0 + 0.055 * np.sin(2 * np.pi * 0.23 * t + phase))
    rr = rr * (1.0 + rng.normal(0.0, 0.015, n))
    return np.clip(rr, 0.30, 2.0)


def generate_rr_with_ectopy(mode: str, n: int, hr_bpm: float,
                            pvc_burden: float, pac_burden: float,
                            rng: np.random.Generator):
    """Devuelve (rr, labels) con la misma longitud: labels[i] es la clase del
    latido i (el RR i es el intervalo previo al latido i)."""
    rr = rr_sequence(mode, n, hr_bpm, rng)
    if pvc_burden <= 0 and pac_burden <= 0:
        return rr, ["N"] * n

    new_rr: list[float] = []
    labels: list[str] = []
    i = 0
    while i < len(rr):
        r = float(rr[i])
        u = rng.random()
        if u < pvc_burden and i < len(rr) - 2:
            prem = r * rng.uniform(0.52, 0.70)
            pause = max(2.0 * r - prem, 0.85 * r)
            new_rr.extend([prem, pause])
            labels.extend(["V", "N"])
            i += 1
        elif u < pvc_burden + pac_burden and i < len(rr) - 2:
            prem = r * rng.uniform(0.48, 0.65)
            pause = max(1.05 * r, 0.55 * r)
            new_rr.extend([prem, pause])
            labels.extend(["S", "N"])
            i += 1
        else:
            new_rr.append(r)
            labels.append("N")
            i += 1
    return np.asarray(new_rr[:n], dtype=float), labels[:n]


# ---------------------------------------------------------------------------
# Segmento completo
# ---------------------------------------------------------------------------
def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    k = np.hanning(win)
    k /= k.sum()
    return np.convolve(x, k, mode="same")


def add_noise(sig: np.ndarray, fs: float, rng: np.random.Generator,
              level: float = 1.0, motion: bool = True) -> np.ndarray:
    """Añade ruido realista (línea base, red eléctrica, EMG, movimiento)."""
    n, L = sig.shape
    t = np.arange(n) / fs
    out = sig.astype(np.float64, copy=True)

    # 1) deriva de línea base: senoidales lentas + paseo aleatorio
    wander = np.zeros(n)
    for _ in range(rng.integers(2, 4)):
        f = rng.uniform(0.05, 0.35)
        wander += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) * rng.uniform(0.02, 0.09)
    wander += _smooth(rng.normal(0, 1, n), max(3, int(fs * 1.5))) * 0.35
    out += (wander * level)[:, None]

    # 2) red eléctrica 50 Hz (y armónico)
    for f0 in (50.0, 100.0):
        if f0 < fs / 2:
            amp = rng.uniform(0.01, 0.06) * level
            out += (amp * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi)))[:, None]

    # 3) EMG: ruido blanco filtrado en banda 20-100 Hz
    emg = rng.normal(0, 1, n)
    k = max(3, int(fs * 0.02))
    emg = emg - _smooth(emg, k)  # paso alto suave -> alta frecuencia
    emg *= rng.uniform(0.004, 0.012) * level
    out += emg[:, None]

    # 4) artefactos de movimiento puntuales
    if motion:
        n_mot = rng.poisson(1.2)
        for _ in range(n_mot):
            c = rng.integers(0, n)
            w = int(rng.uniform(0.15, 0.8) * fs)
            a = rng.uniform(0.15, 0.9) * level * rng.choice([-1, 1])
            lo, hi = max(0, c - w), min(n, c + w)
            bump = np.exp(-0.5 * ((np.arange(hi - lo) - w) / (w / 2.5)) ** 2)
            out[lo:hi, :] += (a * bump)[:, None]
    return out.astype(np.float32)


def generate_segment(fs: float = 250.0, duration: float = 30.0, mode: str = "nsr",
                     hr_bpm: float = 72.0, pvc_burden: float = 0.0,
                     pac_burden: float = 0.0, n_leads: int = 3,
                     noise: float = 1.0, seed: int | None = None,
                     start_offset: float = 0.0):
    """Genera un segmento de ECG sintético.

    Devuelve (samples (n, n_leads) float32 mV, r_peaks idx, labels, fs).
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration * fs))
    n_beats = max(2, int(np.ceil(duration * hr_bpm / 60.0)) + 2)

    rr, labels = generate_rr_with_ectopy(mode, n_beats, hr_bpm, pvc_burden, pac_burden, rng)
    lead_in = 0.6  # evita latidos partidos en el borde inicial
    t_r = start_offset + lead_in + np.cumsum(np.concatenate([[0.0], rr[:-1]]))
    # descartamos latidos fuera del segmento
    keep = (t_r >= 0) & (t_r < duration)
    t_r = t_r[keep]
    labels = [l for l, k in zip(labels, keep) if k]

    sig = np.zeros((n, n_leads), dtype=np.float64)
    for ri, lab in zip(t_r, labels):
        idx0 = int(round(ri * fs))
        half_pre = int(0.30 * fs)
        half_post = int(0.55 * fs)
        a = max(0, idx0 - half_pre)
        b = min(n, idx0 + half_post)
        t_rel = (np.arange(a, b) - idx0) / fs
        stretch = 1.0 + rng.normal(0.0, 0.02)
        for L in range(n_leads):
            sig[a:b, L] += beat_waveform(t_rel, lab, L, rng, stretch)

    sig = add_noise(sig.astype(np.float32), fs, rng, level=noise)
    peaks = np.round(t_r * fs).astype(int)
    peaks = peaks[(peaks >= 0) & (peaks < n)]
    return sig, peaks, labels, fs

