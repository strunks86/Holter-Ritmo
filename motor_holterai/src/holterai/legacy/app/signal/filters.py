"""Filtrado FIR y utilidades de señal (solo numpy).

Diseño de filtros por ventana (sinc enventanado) y convolución FFT por
bloques para que un Holter de 48 h no consuma gigabytes de RAM.
"""
from __future__ import annotations

import numpy as np

_FIR_CACHE: dict[tuple, np.ndarray] = {}


# ---------------------------------------------------------------------------
# Diseño de filtros
# ---------------------------------------------------------------------------
def fir_lowpass(fs: float, fc: float, numtaps: int = 301) -> np.ndarray:
    """FIR paso bajo de fase lineal (ventana de Hamming)."""
    if numtaps % 2 == 0:
        numtaps += 1
    fc = min(fc, 0.49 * fs)
    m = numtaps // 2
    n = np.arange(-m, m + 1)
    h = 2 * fc / fs * np.sinc(2 * fc / fs * n)
    h *= np.hamming(numtaps)
    h /= h.sum()
    return h


def fir_highpass(fs: float, fc: float, numtaps: int = 301) -> np.ndarray:
    h = fir_lowpass(fs, fc, numtaps)
    delta = np.zeros_like(h)
    delta[len(h) // 2] = 1.0
    return delta - h


def fir_bandpass(fs: float, lo: float, hi: float, numtaps: int = 301) -> np.ndarray:
    key = ("bp", round(fs, 3), round(lo, 3), round(hi, 3), numtaps)
    if key in _FIR_CACHE:
        return _FIR_CACHE[key]
    h = fir_lowpass(fs, hi, numtaps) - fir_lowpass(fs, lo, numtaps)
    key2 = ("bp2", key)
    _FIR_CACHE[key] = h
    _FIR_CACHE[key2] = h
    return h


def remove_powerline(x: np.ndarray, fs: float, f0: float = 50.0,
                     block_s: float = 10.0) -> np.ndarray:
    """Elimina la componente de red (y armónicos) por proyección local.

    Más estable que un notch FIR truncado y se adapta a deriva de frecuencia.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0 or f0 >= fs / 2:
        return x.astype(np.float32)
    out = x.copy()
    bs = max(64, int(block_s * fs))
    for start in range(0, n, bs):
        end = min(n, start + bs)
        seg = x[start:end]
        if seg.size < 16:
            continue
        t = np.arange(seg.size) / fs
        cols = []
        for k in (1, 2, 3):
            if k * f0 < fs / 2:
                cols.append(np.sin(2 * np.pi * k * f0 * t))
                cols.append(np.cos(2 * np.pi * k * f0 * t))
        if not cols:
            continue
        A = np.stack(cols, axis=1)
        try:
            coef, *_ = np.linalg.lstsq(A, seg, rcond=None)
            out[start:end] = seg - A @ coef
        except np.linalg.LinAlgError:
            continue
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Convolución
# ---------------------------------------------------------------------------
def _next_fast(target: int) -> int:
    """Menor número 5-suave >= target (FFT rápida), equivalente a scipy."""
    if target <= 6:
        return target
    if not (target & (target - 1)):
        return target
    best = 1 << (target - 1).bit_length()
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            quotient = -(-target // p35)
            p2 = 1 << (quotient - 1).bit_length()
            n = p2 * p35
            if n == target:
                return n
            if n < best:
                best = n
            p35 *= 3
        if p35 < best:
            best = p35
        p5 *= 5
    return best


def fft_conv(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolución 'full' de x con h, con memoria acotada."""
    x = np.asarray(x, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    n, m = x.size, h.size
    if n == 0 or m == 0:
        return np.zeros(max(0, n + m - 1))
    if n * m <= 8_000_000:
        return np.convolve(x, h, mode="full")
    block = 1 << 18
    L = _next_fast(block + m - 1)
    H = np.fft.rfft(h, L)
    out = np.zeros(n + m - 1, dtype=np.float64)
    for start in range(0, n, block):
        seg = x[start:start + block]
        if seg.size < 8:
            out[start:start + seg.size + m - 1] += np.convolve(seg, h, "full")
            continue
        X = np.fft.rfft(seg, L)
        y = np.fft.irfft(X * H, L)[: seg.size + m - 1]
        out[start:start + y.size] += y
    return out


def filtfilt(x: np.ndarray, h: np.ndarray, pad: int | None = None) -> np.ndarray:
    """Filtrado de fase cero (adelante + atrás) con relleno por reflexión."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    if pad is None:
        pad = min(h.size, x.size - 1) if x.size > 1 else 0
    if pad > 0:
        head = x[1:pad + 1][::-1]
        tail = x[-pad - 1:-1][::-1]
        xp = np.concatenate([head, x, tail])
    else:
        xp = x
    y = fft_conv(xp, h)
    y = y[: xp.size]  # 'same'
    y = y[::-1]
    y2 = fft_conv(y, h)[: xp.size]
    y2 = y2[::-1]
    if pad > 0:
        return y2[pad:pad + x.size]
    return y2


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win))
    if win == 1:
        return np.asarray(x, dtype=np.float64)
    k = np.ones(win) / win
    return np.convolve(x, k, mode="same")


# ---------------------------------------------------------------------------
# Preprocesado estándar de ECG
# ---------------------------------------------------------------------------
def lead_params(fs: float):
    """Filtros estándar de Holter a la frecuencia de muestreo dada."""
    return {
        "display": fir_bandpass(fs, 0.5, min(40.0, 0.45 * fs), 801),
        "detect": fir_bandpass(fs, 5.0, min(18.0, 0.40 * fs), 301),
        "baseline": fir_highpass(fs, 0.5, 801),
    }


def preprocess(x: np.ndarray, fs: float, band: str = "display") -> np.ndarray:
    h = lead_params(fs)[band]
    y = filtfilt(x, h)
    if band == "display":
        y = remove_powerline(y, fs, 50.0)
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Filtros SOLO de pantalla
# ---------------------------------------------------------------------------
# Se aplican a la copia de la señal que se dibuja. Nunca entran al detector,
# al clasificador ni a las medidas de ST/QT, de modo que cambiar el selector
# no altera el análisis ni el informe.
DISPLAY_NOTCH_HZ: tuple[float, ...] = (0.0, 50.0, 60.0)
DISPLAY_LOWPASS_HZ: tuple[float, ...] = (0.0, 25.0, 30.0, 35.0)


def _display_choice(value, choices: tuple[float, ...]) -> float:
    """Normaliza un valor del selector; lo no reconocido queda apagado (0)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(v):
        return 0.0
    for c in choices:
        if abs(v - c) < 1e-6:
            return float(c)
    return 0.0


def display_filters(notch_hz=0.0, lowpass_hz=0.0) -> tuple[float, float]:
    """Valida el par (filtro de red, paso bajo) que llega del selector."""
    return (_display_choice(notch_hz, DISPLAY_NOTCH_HZ),
            _display_choice(lowpass_hz, DISPLAY_LOWPASS_HZ))


def _display_lowpass(fs: float, fc: float, numtaps: int = 301) -> np.ndarray:
    key = ("display_lp", round(fs, 3), round(fc, 3), numtaps)
    h = _FIR_CACHE.get(key)
    if h is None:
        h = fir_lowpass(fs, fc, numtaps)
        _FIR_CACHE[key] = h
    return h


def display_filter(x: np.ndarray, fs: float, notch_hz=0.0, lowpass_hz=0.0,
                   numtaps: int = 301) -> np.ndarray:
    """Filtro de visualización: red 50/60 Hz y paso bajo 25–35 Hz opcionales.

    Devuelve siempre una copia float32 del mismo tamaño que ``x`` (1-D o
    2-D muestras x canales), sin tocar el arreglo de entrada. ``0`` = apagado.
    """
    notch_hz, lowpass_hz = display_filters(notch_hz, lowpass_hz)
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0 or (notch_hz == 0.0 and lowpass_hz == 0.0):
        return arr.copy()
    fs = float(fs)
    if fs <= 0:
        return arr.copy()
    mono = arr.ndim == 1
    y = arr.reshape(-1, 1) if mono else arr
    h = _display_lowpass(fs, lowpass_hz, numtaps) if lowpass_hz > 0 else None
    # Ajuste local de la red: bloques de 10 s para seguir la deriva de fase,
    # acotados a ~600 ajustes para que la vista de todo el registro sea barata.
    n = y.shape[0]
    block_s = max(10.0, n / fs / 600.0)
    out = np.empty_like(y, dtype=np.float32)
    for j in range(y.shape[1]):
        col = np.asarray(y[:, j], dtype=np.float64)
        if notch_hz > 0:
            col = remove_powerline(col, fs, notch_hz, block_s=block_s)
        if h is not None:
            col = filtfilt(col, h)
        out[:, j] = col.astype(np.float32)
    return out.reshape(arr.shape)


def resample(x: np.ndarray, n_out: int) -> np.ndarray:
    """Remuestreo lineal a un número fijo de puntos."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if x.size == 0:
        return np.zeros(n_out, dtype=np.float32)
    if x.size == n_out:
        return x.copy()
    src = np.linspace(0.0, 1.0, x.size)
    dst = np.linspace(0.0, 1.0, n_out)
    return np.interp(dst, src, x).astype(np.float32)


def normalize_beat(w: np.ndarray) -> np.ndarray:
    """Centra y normaliza amplitud de una ventana de latido."""
    w = np.asarray(w, dtype=np.float32).ravel()
    if w.size == 0:
        return w
    w = w - np.median(w[: max(1, w.size // 10)])
    scale = np.percentile(np.abs(w), 99.0) + 1e-6
    w = w / scale
    return np.clip(w, -3.0, 3.0).astype(np.float32)


