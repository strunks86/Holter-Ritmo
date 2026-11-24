"""Extracción de ventanas de latido y características para el modelo."""
from __future__ import annotations

import numpy as np

from . import st
from .filters import filtfilt, fir_bandpass, moving_average, normalize_beat, resample

WIN_PRE = 0.25
WIN_POST = 0.35
WIN_POINTS = 256


_INTERP_CACHE: dict[int, np.ndarray] = {}


def _interp_weights(n_in: int, n_out: int = WIN_POINTS) -> np.ndarray:
    """Matriz de remuestreo lineal (n_out, n_in), calculada una vez por tamaño."""
    if n_in in _INTERP_CACHE:
        return _INTERP_CACHE[n_in]
    src = np.linspace(0.0, 1.0, n_in)
    dst = np.linspace(0.0, 1.0, n_out)
    i = np.clip(np.searchsorted(src, dst, side="right") - 1, 0, n_in - 2)
    w = (dst - src[i]) / (src[i + 1] - src[i])
    m = np.zeros((n_out, n_in), dtype=np.float32)
    rows = np.arange(n_out)
    m[rows, i] = 1.0 - w
    m[rows, i + 1] += w
    _INTERP_CACHE[n_in] = m
    return m


def extract_windows_batch(x: np.ndarray, fs: float, peaks: np.ndarray,
                          pre: float = WIN_PRE, post: float = WIN_POST,
                          n_out: int = WIN_POINTS) -> np.ndarray:
    """Ventanas normalizadas para muchos latidos a la vez (sin bucle Python).

    Mantiene exactamente la normalización de `normalize_beat` (mediana del
    primer 10 % y percentil 99), para no alterar la entrada del modelo.
    """
    peaks = np.asarray(peaks, dtype=np.int64)
    if peaks.size == 0:
        return np.zeros((0, n_out), dtype=np.float32)
    a = int(round(pre * fs))
    b = int(round(post * fs))
    w = b + a
    x = np.asarray(x, dtype=np.float32)
    base = peaks - a
    idx = base[:, None] + np.arange(w)[None, :]
    np.clip(idx, 0, x.size - 1, out=idx)
    win = x[idx]                       # (N, w)
    m = _interp_weights(w, n_out)
    out = win @ m.T                    # (N, n_out)
    k = max(1, n_out // 10)
    out = out - np.median(out[:, :k], axis=1, keepdims=True)
    scale = np.percentile(np.abs(out), 99, axis=1, keepdims=True) + 1e-6
    out = out / scale
    return np.clip(out, -3.0, 3.0).astype(np.float32)


def extract_window(x: np.ndarray, fs: float, idx: int, pre: float = WIN_PRE,
                   post: float = WIN_POST, n_out: int = WIN_POINTS,
                   jitter: int = 0) -> np.ndarray:
    """Ventana de un solo latido (compatibilidad)."""
    return extract_windows_batch(x, fs, np.array([int(idx) + int(jitter)]),
                                 pre, post, n_out)[0]


def noise_batch(x: np.ndarray, fs: float, peaks: np.ndarray,
                win_s: float = 0.12) -> np.ndarray:
    """Ruido de alta frecuencia por latido (vectorizado, normalizado a 250 Hz)."""
    peaks = np.asarray(peaks, dtype=np.int64)
    if peaks.size == 0:
        return np.zeros(0, dtype=np.float32)
    half = max(2, int(round(win_s * fs)))
    idx = peaks[:, None] + np.arange(-half, half + 1)[None, :]
    np.clip(idx, 0, x.size - 1, out=idx)
    seg = np.asarray(x, dtype=np.float64)[idx]
    d2 = np.diff(seg, n=2, axis=1)
    med = np.median(d2, axis=1, keepdims=True)
    mad = np.median(np.abs(d2 - med), axis=1) * 1.4826
    scale = (250.0 / max(1.0, fs)) ** 2
    return (mad * 0.5 * scale).astype(np.float32)


def robust_sd(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return 1e-6
    med = np.median(v)
    return float(np.median(np.abs(v - med)) * 1.4826 + 1e-6)


def qrs_width(xf: np.ndarray, fs: float, idx: int) -> float:
    """Anchura del QRS (s) a partir de la envolvente de la derivada."""
    a = max(0, int(idx - 0.18 * fs))
    b = min(xf.size, int(idx + 0.18 * fs))
    if b - a < 5:
        return 0.08
    seg = np.asarray(xf[a:b], dtype=np.float64)
    d = np.abs(np.diff(seg, prepend=seg[:1]))
    d = np.convolve(d, np.ones(3) / 3.0, mode="same")
    mx = float(d.max())
    if mx <= 0:
        return 0.08
    on = d > 0.25 * mx
    # región contigua que contiene el máximo
    j = int(np.argmax(d))
    lo = j
    while lo > 0 and on[lo - 1]:
        lo -= 1
    hi = j
    while hi < on.size - 1 and on[hi + 1]:
        hi += 1
    return float((hi - lo + 1) / fs)


# ---------------------------------------------------------------------------
# Duración del QRS multicanal (T25)
# ---------------------------------------------------------------------------
QRS_WIDE_MS = 120.0
QRS_INCOMPLETE_MS = 110.0
QRS_SEARCH_MS = 200.0
QRS_BAND_LO = 5.0
QRS_BAND_HI = 30.0
QRS_SMOOTH_MS = 8.0
QRS_THR_FRAC = 0.14
QRS_MERGE_MS = 16.0
QRS_EDGE_CORR_MS = 2.0
QRS_LOWFS_EDGE_CORR = 0.25
QRS_INTERP_FS = 400.0
QRS_QUALITY_WINDOW_MS = 350.0
QRS_QUALITY_GUARD_MS = 110.0
QRS_QUALITY_FLOOR = 0.02


def qrs_envelope(y: np.ndarray, fs: float, smooth_ms: float = QRS_SMOOTH_MS) -> np.ndarray:
    """Envoltura de |derivada| suavizada (actividad de alta frecuencia)."""
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.size == 0:
        return y
    d = np.abs(np.diff(y, prepend=y[:1]))
    w = max(1, int(round(smooth_ms / 1000.0 * fs)))
    if w > 1:
        d = np.convolve(d, np.ones(w) / w, mode="same")
    return d


def qrs_band_envelope(y: np.ndarray, fs: float, lo: float = QRS_BAND_LO,
                      hi: float = QRS_BAND_HI,
                      smooth_ms: float = QRS_SMOOTH_MS) -> np.ndarray:
    """Envoltura del QRS: |señal pasabanda 5-30 Hz| suavizada.

    La banda se obtiene con dos medias móviles (quitapendiente de ~200 ms y
    suavizado de ~30 ms) en vez de un FIR largo: la ventana de un latido es
    corta y un FIR de 81 coeficientes no cabe sin artefactos de borde a
    100-200 Hz. La banda conserva la energía del complejo y descarta la onda
    P, la onda T y la pendiente del ST, que son las que alargaban el QRS
    cuando el umbral se aplicaba sobre la derivada en banda ancha.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.size < 16:
        return np.abs(y)
    taps = 81 if fs >= 250 else max(11, int(round(0.30 * y.size)) // 2 * 2 + 1)
    h = fir_bandpass(fs, lo, min(hi, 0.45 * fs), taps)
    bp = np.abs(filtfilt(y, h))
    w = max(1, int(round(smooth_ms / 1000.0 * fs)))
    if w > 1:
        bp = np.convolve(bp, np.ones(w) / w, mode="same")
    return bp


def _fill_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Rellena huecos cortos (muescas) dentro de una máscara booleana."""
    out = mask.copy()
    n = out.size
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and not out[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap:
            out[i:j] = True
        i = j
    return out


def qrs_bounds_channel(x: np.ndarray, fs: float, center: int, *,
                       search_ms: float = QRS_SEARCH_MS,
                       band_lo: float = QRS_BAND_LO,
                       band_hi: float = QRS_BAND_HI,
                       smooth_ms: float = QRS_SMOOTH_MS,
                       thr_frac: float = QRS_THR_FRAC,
                       merge_ms: float = QRS_MERGE_MS,
                       edge_corr_ms: float = QRS_EDGE_CORR_MS) -> tuple[int, int, float]:
    """Inicio y fin del QRS en un canal, en muestras.

    Se toma la envolvente de la señal filtrada 5-30 Hz, se busca su máximo
    dentro de ±`search_ms` del centro y se crece desde ahí con un umbral
    `thr_frac` del pico, rellenando muescas de hasta `merge_ms` (patrones
    rsR' y QRS fragmentados quedan enteros). El suavizado de la envolvente
    adelanta el inicio y retrasa el fin ~`smooth_ms/2`; se corrige con
    `edge_corr_ms`.

    Devuelve `(inicio, fin, amplitud_p2p_mV)`; `fin` es excluyente.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    half = max(3, int(round(search_ms / 1000.0 * fs)))
    c = int(center)
    a = max(0, c - half)
    b = min(x.size, c + half)
    if b - a < 16:
        return c, c, 0.0
    seg = x[a:b]
    edge = max(2, int(round(0.01 * fs)))
    base = float(np.median(np.concatenate([seg[:edge], seg[-edge:]])))
    seg = seg - base
    p2p = float(np.max(seg) - np.min(seg))
    env = qrs_band_envelope(seg, fs, band_lo, band_hi, smooth_ms)
    peak = float(env.max()) if env.size else 0.0
    if peak <= 1e-9:
        return c, c, p2p
    j = int(np.argmax(env))
    filled = _fill_gaps(env >= thr_frac * peak, max(1, int(round(merge_ms / 1000.0 * fs))))
    lo = hi = j
    while lo > 0 and filled[lo - 1]:
        lo -= 1
    while hi < filled.size - 1 and filled[hi + 1]:
        hi += 1
    corr = int(round(edge_corr_ms / 1000.0 * fs))
    lo = min(lo + corr, j)
    hi = max(hi - corr, j)
    onset = a + lo
    offset = a + hi + 1
    region = seg[lo:hi + 1]
    amp = float(np.max(region) - np.min(region)) if region.size else 0.0
    return int(onset), int(min(offset, x.size)), amp


def qrs_channel_quality(x: np.ndarray, fs: float, center: int, *,
                        window_ms: float = QRS_QUALITY_WINDOW_MS,
                        guard_ms: float = QRS_QUALITY_GUARD_MS,
                        band_lo: float = QRS_BAND_LO,
                        band_hi: float = QRS_BAND_HI,
                        smooth_ms: float = QRS_SMOOTH_MS) -> float:
    """Pico de la envolvente sobre el ruido del resto del latido.

    Un canal limpio mantiene un pico bien destacado sobre el tramo sin QRS
    (≈15 o más), aunque el complejo sea ancho o de poca amplitud; uno dominado
    por ruido queda en 2-4. Se toma como ruido la mediana de la envolvente de
    una ventana de ±`window_ms` **fuera** de ±`guard_ms` del pico, no la
    mediana de toda la ventana de búsqueda: en un QRS ancho (±200 ms) el
    complejo ocupa casi toda la ventana y la mediana global se confunde con el
    pico, con lo que se descartaba la precordial del bloqueo de rama (en
    MIT-BIH 214 la V1 del BRI daba 8,1 por mediana global y 18,8 por ruido del
    resto del latido, justo en el umbral de descarte).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    half = max(8, int(round(window_ms / 1000.0 * fs)))
    a = max(0, int(center) - half)
    b = min(x.size, int(center) + half)
    if b - a < 32:
        return 0.0
    env = qrs_band_envelope(x[a:b], fs, band_lo, band_hi, smooth_ms)
    if env.size == 0:
        return 0.0
    j = int(np.argmax(env))
    peak = float(env[j])
    if peak <= 0:
        return 0.0
    g = max(4, int(round(guard_ms / 1000.0 * fs)))
    rest = np.concatenate([env[:max(0, j - g)], env[min(env.size, j + g):]])
    if rest.size < 8:
        return 0.0
    # suelo del 2 % del pico: un latido sintético sin ruido no da infinito
    floor = QRS_QUALITY_FLOOR * peak
    return peak / max(float(np.median(rest)), floor, 1e-9)


def qrs_duration_multichannel(beat: np.ndarray, fs: float, center: int, *,
                              min_amp_ratio: float = 0.08,
                              min_quality: float = 10.0,
                              max_width_ms: float = 250.0,
                              **kwargs) -> dict:
    """Duración del QRS con inicio más temprano y fin más tardío entre canales.

    `beat` es un latido promediado `(muestras, canales)` o `(muestras,)`.
    Los canales con amplitud despreciable frente al mejor canal no cuentan
    (un electrodo caído no debe alargar ni acortar el QRS).

    Devuelve un dict con `onset`, `offset`, `width_ms`, `wide`, `incomplete`,
    `per_channel` (lista de dicts) y `amplitude_mv`.
    """
    y = np.asarray(beat, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    n = y.shape[0]
    fs_eff = float(fs)
    c_eff = int(center)
    y_eff = y
    factor = 1
    if fs < 200.0:
        # a 100 Hz el QRS mide 8-15 muestras: se interpola para no medir de 10 en
        # 10 ms, y hasta 400 Hz para que el recorte de borde no quede cuantizado
        factor = int(np.ceil(QRS_INTERP_FS / max(fs, 1.0)))
        n_eff = (n - 1) * factor + 1
        t_src = np.arange(n, dtype=np.float64)
        t_dst = np.linspace(0.0, n - 1.0, n_eff)
        y_eff = np.stack([np.interp(t_dst, t_src, y[:, ch]) for ch in range(y.shape[1])],
                         axis=1)
        fs_eff = fs * factor
        c_eff = int(round(center * factor))
    if factor > 1:
        # al interpolar desde 100 Hz el umbral cae hasta una muestra original
        # (10 ms) más afuera en cada borde: se recorta proporcional al intervalo
        corr = kwargs.pop("edge_corr_ms", QRS_EDGE_CORR_MS)
        kwargs["edge_corr_ms"] = corr + QRS_LOWFS_EDGE_CORR * (1000.0 / fs)
    per = []
    for ch in range(y_eff.shape[1]):
        on, off, amp = qrs_bounds_channel(y_eff[:, ch], fs_eff, c_eff, **kwargs)
        per.append({"channel": ch, "onset": on, "offset": off, "amplitude_mv": amp,
                    "width_ms": round((off - on) / fs_eff * 1000.0, 1)})
    if not per:
        return {"onset": int(center), "offset": int(center), "width_ms": 0.0,
                "wide": False, "incomplete": False, "measurable": False,
                "per_channel": [], "amplitude_mv": 0.0, "n_channels_used": 0}
    best = max(p["amplitude_mv"] for p in per)
    thr = min_amp_ratio * best
    used = [p for p in per if p["amplitude_mv"] > thr and p["offset"] > p["onset"]]
    if not used:
        used = per
    # canales dominados por ruido o medidas absurdas no deciden la duración
    quality = {p["channel"]: qrs_channel_quality(y[:, p["channel"]], fs, center)
               for p in per}
    for p in per:
        p["quality"] = round(float(quality[p["channel"]]), 2)
    clean = [p for p in used
             if p["quality"] >= min_quality and (p["offset"] - p["onset"]) / fs_eff * 1000.0
             <= max_width_ms]
    measurable = bool(clean)
    if clean:
        used = clean
    onset_eff = min(p["onset"] for p in used)
    offset_eff = max(p["offset"] for p in used)
    width_ms = float((offset_eff - onset_eff) / fs_eff * 1000.0) if fs_eff > 0 else 0.0
    if not measurable:
        width_ms = 0.0
    onset = int(round(onset_eff / factor))
    offset = int(round(offset_eff / factor))
    return {
        "onset": onset,
        "offset": offset,
        "width_ms": round(width_ms, 1),
        "wide": bool(measurable and width_ms >= QRS_WIDE_MS),
        "incomplete": bool(measurable and QRS_INCOMPLETE_MS <= width_ms < QRS_WIDE_MS),
        "measurable": measurable,
        "per_channel": per,
        "amplitude_mv": round(float(best), 3),
        "n_channels_used": len(used) if measurable else 0,
        "interpolated_x": factor,
        "span_ok": bool(0 <= onset < offset <= n),
    }


# ---------------------------------------------------------------------------
# Forma del inicio del QRS y eje frontal (T44)
# ---------------------------------------------------------------------------
QRS_HEAD_MS = 40.0          # ventana del "inicio" del QRS
QRS_DELTA_UPSTROKE_MS = 45.0  # ascenso lento típico de una onda delta
QRS_DELTA_HEAD_RATIO = 0.45   # pendiente del inicio / pendiente máxima
QRS_DELTA_SLOPE_FRAC = 0.45   # el máximo de |dQRS| cae tarde en el complejo


def qrs_onset_shape(y: np.ndarray, fs: float, onset: int, offset: int) -> dict:
    """Forma del comienzo del QRS: onda delta, tiempo al pico y centro de pendiente.

    La onda delta de una preexcitación es un ascenso inicial lento: la
    pendiente máxima del complejo llega tarde (``slope_peak_frac`` alta), el
    ascenso hasta la mitad del pico dura más de 40 ms (``upstroke_ms``) y la
    pendiente del primer tramo es una fracción pequeña de la máxima
    (``head_ratio``). En un QRS normal el ascenso es rápido y la pendiente
    máxima está al principio (``head_ratio`` ≈ 1, ``upstroke_ms`` < 30 ms).

    ``initial_frac`` es la fracción del área del complejo que cae en el primer
    40 ms: en la onda delta el inicio aporta poco a la amplitud total.

    Devuelve ``{}`` si el segmento es demasiado corto o no tiene amplitud.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    onset = max(0, int(onset))
    offset = min(y.size, int(offset))
    if offset - onset < 4:
        return {}
    guard = max(2, int(round(0.03 * fs)))
    pre = y[max(0, onset - guard):onset]
    post = y[offset:offset + guard]
    base = float(np.median(np.concatenate([pre, post]))) if pre.size or post.size else 0.0
    seg = y[onset:offset] - base
    amp = float(np.max(np.abs(seg)))
    if amp <= 1e-6:
        return {}
    # polaridad de la deflexión dominante: todo se mide sobre la señal "positiva"
    sign = 1.0 if abs(float(np.max(seg))) >= abs(float(np.min(seg))) else -1.0
    signed = seg * sign
    peak = float(signed.max())
    out: dict[str, float] = {"amp_mv": amp, "polarity": sign,
                             "head_ratio": 1.0, "upstroke_ms": 0.0,
                             "slope_peak_ms": 0.0, "slope_peak_frac": 0.0,
                             "initial_frac": 0.0}
    if peak <= 1e-6:
        return out
    over = np.flatnonzero(signed >= 0.5 * peak)
    if over.size:
        out["upstroke_ms"] = round(float(over[0]) / fs * 1000.0, 1)
    d = np.abs(np.diff(seg, prepend=seg[:1]))
    dmax = float(d.max())
    n = seg.size
    if dmax <= 0:
        return out
    head = max(2, int(round(QRS_HEAD_MS / 1000.0 * fs)))
    out["head_ratio"] = round(float(d[:head].max() / dmax), 3)
    k = int(np.argmax(d))
    out["slope_peak_ms"] = round(float(k) / fs * 1000.0, 1)
    out["slope_peak_frac"] = round(float(k) / n, 3)
    head_area = float(np.abs(seg[:head]).sum())
    out["initial_frac"] = round(head_area / max(float(np.abs(seg).sum()), 1e-12), 3)
    return out


def frontal_axis_deg(net_i: float, net_ii: float) -> float:
    """Eje frontal aproximado (grados) desde las fuerzas netas de I y II.

    Con Einthoven ``aVF = II - I/2``; el eje es ``atan2(aVF, I)``. 0° es la
    derivación I, +90° aVF (eje normal entre -30° y +90°). Se usa solo como
    criterio de apoyo: sin las derivaciones I y II no se calcula.
    """
    avf = float(net_ii) - 0.5 * float(net_i)
    return float(np.degrees(np.arctan2(avf, float(net_i))))


def median_beat(x: np.ndarray, fs: float, peaks, *, pre: float = 0.30,
                post: float = 0.45, jitter_ms: float = 8.0) -> tuple[np.ndarray, int]:
    """Latido promedio (mediana punto a punto) alrededor de los picos.

    Devuelve `(beat, centro)` con `beat` de forma `(muestras, canales)`. El
    promedio por mediana es robusto a latidos ectópicos y a artefactos
    aislados, que es lo que se quiere para medir la morfología dominante.
    """
    sig = np.asarray(x, dtype=np.float64)
    if sig.ndim == 1:
        sig = sig[:, None]
    peaks = np.asarray(peaks, dtype=np.int64)
    a = int(round(pre * fs))
    b = int(round(post * fs))
    n_out = a + b
    if peaks.size == 0 or n_out < 4:
        return np.zeros((max(n_out, 0), sig.shape[1])), a
    idx = peaks[:, None] + np.arange(-a, b)[None, :]
    inside = (idx[:, 0] >= 0) & (idx[:, -1] < sig.shape[0])
    idx = idx[inside]
    if idx.size == 0:
        return np.zeros((n_out, sig.shape[1])), a
    win = sig[idx]                       # (n_latidos, n_out, n_canales)
    beat = np.median(win, axis=0)
    return beat, a


def p_wave_score(xf: np.ndarray, fs: float, idx: int) -> float:
    """Presencia de onda P antes del QRS, normalizada por el ruido local."""
    a = max(0, int(idx - 0.32 * fs))
    b = max(a + 4, int(idx - 0.05 * fs))
    if b - a < 6:
        return 0.0
    seg = np.asarray(xf[a:b], dtype=np.float64)
    t = np.linspace(0, 1, seg.size)
    # elimina tendencia lineal
    coef = np.polyfit(t, seg, 1)
    seg = seg - np.polyval(coef, t)
    sd = robust_sd(seg)
    peak = float(np.max(np.abs(seg)))
    return float(np.clip(peak / (4.0 * sd), 0.0, 2.0))


def t_wave_score(xf: np.ndarray, fs: float, idx: int) -> float:
    a = int(idx + 0.12 * fs)
    b = int(idx + 0.40 * fs)
    if b <= a or b > xf.size:
        return 0.0
    seg = np.asarray(xf[a:b], dtype=np.float64)
    sd = robust_sd(seg)
    return float(np.clip(np.max(np.abs(seg)) / (4.0 * sd), 0.0, 2.0))


def noise_level(x: np.ndarray, fs: float, idx: int) -> float:
    """Ruido de alta frecuencia alrededor del latido (mV, normalizado a 250 Hz).

    La segunda diferencia crece con la frecuencia de muestreo; se corrige
    para que la característica sea comparable entre 128, 250 y 360 Hz.
    """
    a = max(0, int(idx - 0.12 * fs))
    b = min(x.size, int(idx + 0.12 * fs))
    if b - a < 6:
        return 0.0
    seg = np.asarray(x[a:b], dtype=np.float64)
    d2 = np.diff(seg, n=2)
    scale = (250.0 / max(1.0, fs)) ** 2
    return float(robust_sd(d2) * 0.5 * scale)


def st_level_uv(xf: np.ndarray, fs: float, idx: int, j_ms: int = 60) -> float:
    """ST aproximado en µV respecto del segmento PQ (ver `signal.st`).

    `j_ms` es el punto de medición tras el pico R: 60 (por defecto) u 80 ms.
    """
    return st.st_level_uv(xf, fs, idx, j_ms)


def st_slope_uv_s(xf: np.ndarray, fs: float, idx: int, j_ms: int = 80) -> float:
    """Pendiente ST aproximada entre J+20 y J+`j_ms` (µV/s)."""
    return st.st_slope_uv_s(xf, fs, idx, j_ms)


def qt_interval_ms(xf: np.ndarray, fs: float, idx: int, rr_s: float) -> float:
    """QT aproximado por retorno de la onda T a la línea de base.

    Es una medición experimental; no sustituye la revisión del latido medio.
    """
    if rr_s <= 0.2:
        return 0.0
    a = max(0, int(idx + 0.12 * fs))
    b = min(xf.size, int(idx + min(0.60, 0.55 * float(np.sqrt(rr_s))) * fs))
    if b - a < int(0.12 * fs):
        return 0.0
    seg = np.asarray(xf[a:b], dtype=np.float64)
    base = float(np.median(seg[:max(3, int(0.04 * fs))]))
    dev = np.abs(seg - base)
    peak = int(np.argmax(dev))
    amp = float(dev[peak])
    if amp <= 1e-6:
        return 0.0
    thresh = 0.08 * amp
    end = peak
    quiet = 0
    need = max(2, int(0.02 * fs))
    for k in range(peak, len(dev)):
        if dev[k] <= thresh:
            quiet += 1
            if quiet >= need:
                end = k - need + 1
                break
        else:
            quiet = 0
            end = k
    qt = (a + end - idx) / fs * 1000.0
    return round(float(np.clip(qt, 180.0, 650.0)), 1)


def beat_features(xf: np.ndarray, fs: float, idx: int, rr_prev: float,
                  rr_next: float, rr_med: float, x_raw: np.ndarray | None = None):
    """Devuelve (vector de 5 características, dict descriptivo)."""
    rr_ratio = float(rr_prev / rr_med) if rr_med > 0 else 1.0
    w = qrs_width(xf, fs, idx)
    a = max(0, int(idx - 0.10 * fs))
    b = min(xf.size, int(idx + 0.10 * fs))
    amp = float(np.max(np.abs(xf[a:b]))) if b > a else 0.0
    feat = np.array(
        [
            float(np.clip(rr_prev, 0, 4)),
            float(np.clip(rr_next, 0, 4)),
            float(np.clip(rr_ratio, 0.3, 3.0)),
            float(np.clip(w, 0.03, 0.25)),
            float(np.clip(amp, 0, 5)),
        ],
        dtype=np.float32,
    )
    desc = {
        "qrs_width": w,
        "amplitude": amp,
        "rr_prev": rr_prev,
        "rr_next": rr_next,
        "rr_ratio": rr_ratio,
        "p_score": p_wave_score(xf, fs, idx),
        "t_score": t_wave_score(xf, fs, idx),
        "noise": noise_level(x_raw if x_raw is not None else xf, fs, idx),
    }
    return feat, desc


FEATURE_NAMES = ["rr_prev", "rr_next", "rr_ratio", "qrs_width", "amplitude"]


def extract_batch(xf: np.ndarray, fs: float, peaks: np.ndarray,
                  rr: np.ndarray, rr_med_local: np.ndarray,
                  jitter: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Ventanas (N,256) y características (N,5) para un conjunto de picos."""
    n = len(peaks)
    wins = np.zeros((n, WIN_POINTS), dtype=np.float32)
    feats = np.zeros((n, 5), dtype=np.float32)
    for i, p in enumerate(peaks):
        wins[i] = extract_window(xf, fs, int(p), jitter=jitter)
        rp = float(rr[i - 1]) if i > 0 else 0.0
        rn = float(rr[i]) if i < len(rr) else 0.0
        feats[i], _ = beat_features(xf, fs, int(p), rp, rn, float(rr_med_local[i]))
    return wins, feats


