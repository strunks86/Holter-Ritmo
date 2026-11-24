"""Actividad auricular: cancelación del QRST y espectro del residuo (3-12 Hz).

En fibrilación auricular (FA) y en flutter el ECG de superficie muestra ondas
auriculares continuas de 3-12 Hz que el QRS y la onda T tapan. El método es el
clásico de sustracción de latido promedio ("average beat subtraction"):

1. Se construye una plantilla QRST **por minuto** con la mediana de los latidos
   de ese minuto (robusta a ectopia y a cambios de morfología).
2. Se resta la plantilla en cada latido, con una ventana de Tukey en los bordes
   para no introducir escalones en el residuo.
3. Del residuo se calcula el espectro en ventanas de 5 s y se miden frecuencia
   dominante, concentración espectral, ancho de banda del pico y relación con su
   armónico: un pico estrecho de 4-6 Hz (hasta 6.5 Hz en el flutter rápido) con
   armónico es el patrón del flutter.

`flutter_review` solo indica "a revisión" (señal de apoyo para el revisor);
no es un diagnóstico y no modifica el informe ni la interfaz.
"""
from __future__ import annotations

import numpy as np

from .filters import filtfilt, fir_bandpass

# Banda de la actividad auricular y del flutter (Hz).
BAND_HZ = (3.0, 12.0)
# 4-6 Hz (240-360 lpm) es el rango clásico del flutter. Se admite hasta 6.5 Hz
# porque el flutter rápido real llega a ~5.9 Hz (p. ej. MIT-BIH 222).
FLUTTER_HZ = (4.0, 6.5)
CONC_HALF_HZ = 0.5
# Ventanas de análisis y de la plantilla.
WIN_S = 5.0
STEP_S = 5.0
SEG_S = 60.0
TEMPLATE_PRE_S = 0.25
TEMPLATE_POST_S = 0.55
MIN_BEATS_PER_SEGMENT = 4
MIN_WINDOW_FRACTION = 0.6
# Criterios de flutter (conservadores: solo "a revisión").
FLUTTER_MIN_CONC = 0.40
FLUTTER_MAX_BW_HZ = 0.6
FLUTTER_MIN_HARMONIC = 0.08
FLUTTER_MIN_FRACTION = 0.40
HARMONIC_MAX_HZ = 14.0
# Por debajo de esta reducción del QRST el espectro del residuo no es fiable.
MIN_QUALITY_REDUCTION = 0.10

WINDOW_KEYS = (
    "t0", "t1", "dom_hz", "conc", "bw_hz", "harmonic", "prominence",
    "rms_band", "rr_hz", "flutter",
)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def prepare(x: np.ndarray, fs: float) -> np.ndarray:
    """Paso alto 0.5 Hz y paso bajo 40 Hz: quita deriva y EMG antes del QRST."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float32)
    hi = min(40.0, 0.45 * fs)
    h = fir_bandpass(fs, 0.5, hi, 401)
    return filtfilt(x, h).astype(np.float32)


def tukey(n: int, alpha: float = 0.2) -> np.ndarray:
    """Ventana de Tukey (numpy) para desvanecer la plantilla en los bordes."""
    n = int(max(1, n))
    if alpha <= 0 or n == 1:
        return np.ones(n)
    t = np.linspace(0.0, 1.0, n)
    edge = alpha / 2.0
    w = np.ones(n)
    lo = t < edge
    w[lo] = 0.5 * (1.0 + np.cos(np.pi * (t[lo] / edge - 1.0)))
    hi = t > 1.0 - edge
    w[hi] = 0.5 * (1.0 + np.cos(np.pi * ((t[hi] - 1.0) / edge + 1.0)))
    return w


def beat_matrix(x: np.ndarray, fs: float, peaks: np.ndarray, pre: float,
                post: float) -> tuple[np.ndarray, np.ndarray]:
    """Matriz (n, L) con la ventana de cada latido; descarta los del borde."""
    x = np.asarray(x, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=np.int64).ravel()
    a, b = int(round(pre * fs)), int(round(post * fs))
    length = a + b
    if peaks.size == 0 or length < 4 or x.size < length:
        return np.zeros((0, max(length, 0))), np.zeros(0, dtype=bool)
    ok = (peaks - a >= 0) & (peaks + b <= x.size)
    sel = peaks[ok]
    if sel.size == 0:
        return np.zeros((0, length)), ok
    idx = sel[:, None] + np.arange(-a, b, dtype=np.int64)[None, :]
    return x[idx], ok


# ---------------------------------------------------------------------------
# Cancelación del QRST
# ---------------------------------------------------------------------------
def qrst_templates(x: np.ndarray, fs: float, peaks: np.ndarray,
                   seg_s: float = SEG_S, pre: float = TEMPLATE_PRE_S,
                   post: float = TEMPLATE_POST_S,
                   min_beats: int = MIN_BEATS_PER_SEGMENT) -> dict:
    """Plantilla QRST por minuto (mediana de los latidos del minuto).

    Devuelve un diccionario con `template` (n_segmentos x L), la ventana de
    Tukey, el número de latidos usados por segmento y la plantilla de respaldo
    (mediana de las plantillas válidas) para los minutos con pocos latidos.
    """
    x = np.asarray(x, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=np.int64).ravel()
    pre_n, post_n = int(round(pre * fs)), int(round(post * fs))
    length = pre_n + post_n
    out = {
        "ok": False, "template": np.zeros((0, length)), "taper": tukey(length),
        "fallback": np.zeros(length), "n_beats": np.zeros(0, dtype=int),
        "used": np.zeros(0, dtype=bool), "pre_n": pre_n, "post_n": post_n,
        "length": length, "seg_s": float(seg_s), "n_segments": 0,
        "n_applied": 0, "n_peaks": int(peaks.size),
    }
    if length < 4 or x.size < length:
        return out
    ok = (peaks - pre_n >= 0) & (peaks + post_n <= x.size)
    p = peaks[ok]
    if p.size == 0:
        return out
    seg_samples = max(1, int(round(seg_s * fs)))
    seg = p // seg_samples
    n_seg = int(seg.max()) + 1
    template = np.zeros((n_seg, length), dtype=np.float64)
    n_beats = np.zeros(n_seg, dtype=int)
    offsets = np.arange(-pre_n, post_n, dtype=np.int64)
    for s in range(n_seg):
        sel = p[seg == s]
        n_beats[s] = int(sel.size)
        if n_beats[s] < max(1, min_beats):
            continue
        tpl = np.median(x[sel[:, None] + offsets[None, :]], axis=0)
        template[s] = tpl - np.median(tpl)  # sin offset local
    used = n_beats >= max(1, min_beats)
    if not used.any():
        return out
    fallback = np.median(template[used], axis=0)
    template[~used] = fallback
    out.update(ok=True, template=template, fallback=fallback, n_beats=n_beats,
               used=used, n_segments=n_seg, n_applied=int(used[seg].sum()))
    return out


def cancel_qrst(x: np.ndarray, fs: float, peaks: np.ndarray,
                seg_s: float = SEG_S, pre: float = TEMPLATE_PRE_S,
                post: float = TEMPLATE_POST_S,
                min_beats: int = MIN_BEATS_PER_SEGMENT,
                templates: dict | None = None) -> tuple[np.ndarray, dict]:
    """Resta la plantilla QRST por minuto y devuelve (residuo, calidad).

    `reduction` es 1 - RMS(residuo)/RMS(señal) dentro de las ventanas de la
    plantilla: cuánta energía del QRST explicó la plantilla (0 a 1).
    """
    x = np.asarray(x, dtype=np.float64)
    tpl = templates if templates is not None else qrst_templates(
        x, fs, peaks, seg_s=seg_s, pre=pre, post=post, min_beats=min_beats)
    residual = x.copy()
    info = {"ok": bool(tpl["ok"]), "reduction": 0.0, "n_applied": 0,
            "n_peaks": int(tpl["n_peaks"]), "n_segments": int(tpl["n_segments"])}
    if not tpl["ok"] or x.size == 0:
        return residual.astype(np.float32), info

    peaks = np.asarray(peaks, dtype=np.int64).ravel()
    pre_n, post_n = int(tpl["pre_n"]), int(tpl["post_n"])
    length = pre_n + post_n
    ok = (peaks - pre_n >= 0) & (peaks + post_n <= x.size)
    p = peaks[ok]
    if p.size == 0:
        return residual.astype(np.float32), info
    seg_samples = max(1, int(round(float(tpl["seg_s"]) * fs)))
    seg = p // seg_samples
    taper = tpl["taper"]
    templates_all = tpl["template"]

    num = 0.0
    den = 0.0
    for k in range(p.size):
        a = int(p[k]) - pre_n
        t = templates_all[seg[k]] * taper
        residual[a:a + length] -= t
        den += float(np.dot(x[a:a + length], x[a:a + length]))
        num += float(np.dot(residual[a:a + length], residual[a:a + length]))
    info["n_applied"] = int(p.size)
    if den > 0:
        info["reduction"] = float(1.0 - np.sqrt(num / den))
    return residual.astype(np.float32), info


# ---------------------------------------------------------------------------
# Espectro del residuo
# ---------------------------------------------------------------------------
def residual_spectrum(seg: np.ndarray, fs: float,
                      zero_pad: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Densidad espectral de potencia (un lado) del residuo, en mV²/Hz."""
    seg = np.asarray(seg, dtype=np.float64).ravel()
    seg = seg[np.isfinite(seg)]
    n = seg.size
    if n < 8:
        return np.zeros(0), np.zeros(0)
    seg = seg - float(seg.mean())
    w = np.hanning(n)
    nfft = 1 << int(np.ceil(np.log2(max(64, n * max(1, zero_pad)))))
    spec = np.abs(np.fft.rfft(seg * w, nfft)) ** 2
    df = fs / nfft
    psd = 2.0 * spec / (fs * float(np.sum(w ** 2)))
    f = np.arange(psd.size) * df
    return f, psd


def _refine_peak(f: np.ndarray, p: np.ndarray, k: int) -> float:
    """Interpolación parabólica (en log-potencia) del pico k."""
    if k <= 0 or k + 1 >= p.size:
        return float(f[k])
    y0 = np.log(max(float(p[k - 1]), 1e-30))
    y1 = np.log(max(float(p[k]), 1e-30))
    y2 = np.log(max(float(p[k + 1]), 1e-30))
    den = y0 - 2.0 * y1 + y2
    if abs(den) < 1e-12:
        return float(f[k])
    delta = 0.5 * (y0 - y2) / den
    return float(f[k] + np.clip(delta, -0.5, 0.5) * (f[1] - f[0]))


def _half_width(f: np.ndarray, p: np.ndarray, k: int) -> float:
    """Ancho a media altura del pico k (satura en el ancho de la banda)."""
    half = 0.5 * float(p[k])
    lo = hi = k
    while lo > 0 and p[lo - 1] >= half:
        lo -= 1
    while hi + 1 < p.size and p[hi + 1] >= half:
        hi += 1
    return float(f[hi] - f[lo] + (f[1] - f[0]))


def _harmonic_ratio(f: np.ndarray, p: np.ndarray, f0: float,
                    band: tuple[float, float] = BAND_HZ,
                    half: float = CONC_HALF_HZ) -> float:
    """Amplitud en 2*f0 relativa a la del pico (NaN si el armónico no cabe).

    Se mide en amplitud (raíz de la potencia) porque es más interpretable:
    0.1 = el armónico tiene el 10 % de la amplitud del pico. El armónico puede
    caer algo por encima de 12 Hz (flutter de 6 Hz -> 12 Hz), así que solo se
    exige que quepa en el espectro calculado.
    """
    f1 = 2.0 * f0
    top = min(HARMONIC_MAX_HZ, float(f[-1]) - half)
    if f1 < band[0] or f1 > top:
        return float("nan")
    base = float(p[np.abs(f - f0) <= half].sum())
    harm = float(p[np.abs(f - f1) <= half].sum())
    if base <= 0 or harm <= 0:
        return float("nan")
    return float(np.sqrt(harm / base))


def is_flutter(feat: dict) -> bool:
    """¿La ventana cumple el patrón de flutter (pico estrecho 4-6 Hz + armónico)?"""
    f0 = float(feat.get("dom_hz", float("nan")))
    harm = float(feat.get("harmonic", float("nan")))
    if not np.isfinite(f0) or not np.isfinite(harm):
        return False
    if not (FLUTTER_HZ[0] <= f0 <= FLUTTER_HZ[1]):
        return False
    return bool(
        float(feat.get("conc", 0.0)) >= FLUTTER_MIN_CONC
        and float(feat.get("bw_hz", np.inf)) <= FLUTTER_MAX_BW_HZ
        and harm >= FLUTTER_MIN_HARMONIC
    )


def spectral_features(seg: np.ndarray, fs: float, band: tuple[float, float] = BAND_HZ,
                      zero_pad: int = 4, conc_half: float = CONC_HALF_HZ,
                      rr_hz: float = float("nan")) -> dict:
    """Frecuencia dominante, concentración, ancho y armónico de una ventana.

    `rr_hz` (frecuencia ventricular) solo se copia al resultado como contexto
    para el revisor: `ratio` = dom_hz / rr_hz (p. ej. 3 = conducción 3:1).
    """
    out = {"dom_hz": float("nan"), "conc": 0.0, "bw_hz": float("nan"),
           "harmonic": float("nan"), "prominence": 0.0, "rms_band": 0.0,
           "rr_hz": float(rr_hz), "ratio": float("nan"), "flutter": False}
    f, p = residual_spectrum(seg, fs, zero_pad=zero_pad)
    if f.size == 0:
        return out
    m = (f >= band[0]) & (f <= band[1])
    if not m.any():
        return out
    ff, pf = f[m], p[m]
    df = float(f[1] - f[0])
    total = float(pf.sum())
    out["rms_band"] = float(np.sqrt(max(total * df, 0.0)))
    if total <= 0:
        return out
    k = int(np.argmax(pf))
    f0 = _refine_peak(ff, pf, k)
    out["dom_hz"] = f0
    out["conc"] = float(pf[np.abs(ff - f0) <= conc_half].sum() / total)
    out["bw_hz"] = _half_width(ff, pf, k)
    out["prominence"] = float(pf[k] / max(float(np.median(pf)), 1e-30))
    out["harmonic"] = _harmonic_ratio(f, p, f0, band, conc_half)
    if np.isfinite(rr_hz) and rr_hz > 0:
        out["ratio"] = float(f0 / rr_hz)
    out["flutter"] = is_flutter(out)
    return out


def _window_rr_hz(peaks: np.ndarray, a: int, b: int, fs: float) -> float:
    """Frecuencia ventricular (Hz) de los latidos que caen en [a, b)."""
    if peaks.size < 3:
        return float("nan")
    lo = int(np.searchsorted(peaks, a))
    hi = int(np.searchsorted(peaks, b))
    rr = np.diff(peaks[lo:hi]).astype(np.float64) / fs
    rr = rr[(rr > 0.2) & (rr < 3.0)]
    if rr.size == 0:
        return float("nan")
    return 1.0 / float(np.median(rr))


def _summarize(windows: list[dict], cancel: dict | None = None) -> dict:
    out = {
        "n_windows": len(windows), "dom_hz_med": float("nan"),
        "dom_hz_p25": float("nan"), "dom_hz_p75": float("nan"),
        "conc_med": 0.0, "bw_med": float("nan"), "harmonic_med": float("nan"),
        "rms_band_med": 0.0, "rr_hz_med": float("nan"), "ratio_med": float("nan"),
        "flutter_fraction": 0.0, "flutter_review": False,
        "pattern": "sin ventanas", "review": "",
        "cancel_reduction": float((cancel or {}).get("reduction", 0.0)),
        "quality": "sin datos",
    }
    if not windows:
        return out
    dom = np.array([w["dom_hz"] for w in windows], dtype=float)
    fin = np.isfinite(dom)
    if fin.any():
        out["dom_hz_med"] = float(np.median(dom[fin]))
        out["dom_hz_p25"] = float(np.percentile(dom[fin], 25))
        out["dom_hz_p75"] = float(np.percentile(dom[fin], 75))
    out["conc_med"] = float(np.median([w["conc"] for w in windows]))
    for key, name in (("bw_hz", "bw_med"), ("harmonic", "harmonic_med"),
                      ("rms_band", "rms_band_med"), ("rr_hz", "rr_hz_med"),
                      ("ratio", "ratio_med")):
        vals = np.array([w[key] for w in windows], dtype=float)
        fin = np.isfinite(vals)
        if fin.any():
            out[name] = float(np.median(vals[fin]))
    out["quality"] = ("baja" if out["cancel_reduction"] < MIN_QUALITY_REDUCTION
                      else "ok")
    out["flutter_fraction"] = float(np.mean([bool(w["flutter"]) for w in windows]))
    out["flutter_review"] = bool(
        out["flutter_fraction"] >= FLUTTER_MIN_FRACTION
        and out["conc_med"] >= FLUTTER_MIN_CONC
        and out["quality"] == "ok"
    )
    if out["flutter_review"]:
        out["review"] = "pico 4-6 Hz con armónico: a revisión"
        out["pattern"] = "pico estrecho"
    elif out["conc_med"] >= FLUTTER_MIN_CONC and np.isfinite(out["bw_med"]) \
            and out["bw_med"] <= FLUTTER_MAX_BW_HZ:
        out["pattern"] = "pico estrecho"
    elif out["conc_med"] >= 0.20:
        out["pattern"] = "pico moderado"
    else:
        out["pattern"] = "espectro ancho"
    return out


def analyze(x: np.ndarray, fs: float, peaks: np.ndarray, win_s: float = WIN_S,
            step_s: float = STEP_S, seg_s: float = SEG_S,
            band: tuple[float, float] = BAND_HZ, prepared: bool = False) -> dict:
    """Cancela el QRST y mide el espectro auricular ventana a ventana.

    Devuelve `residual` (señal sin QRST), `cancel` (calidad de la resta),
    `windows` (una entrada por ventana de 5 s) y `summary` (agregado del
    episodio). `flutter_review` es solo una marca para revisión manual.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        empty = {"windows": [], "summary": _summarize([]),
                 "cancel": {"ok": False, "reduction": 0.0, "n_applied": 0,
                            "n_peaks": int(np.asarray(peaks).size)},
                 "residual": np.zeros(0, dtype=np.float32)}
        return empty
    xf = x if prepared else prepare(x, fs)
    residual, cancel = cancel_qrst(xf, fs, peaks, seg_s=seg_s)
    pks = np.sort(np.asarray(peaks, dtype=np.int64).ravel())
    win = max(8, int(round(win_s * fs)))
    step = max(1, int(round(step_s * fs)))
    min_len = max(8, int(round(MIN_WINDOW_FRACTION * win)))
    windows: list[dict] = []
    pos = 0
    while pos + min_len <= residual.size:
        seg = residual[pos:pos + win]
        if seg.size < min_len:
            break
        feats = spectral_features(seg, fs, band=band,
                                  rr_hz=_window_rr_hz(pks, pos, pos + seg.size, fs))
        if np.isfinite(feats["dom_hz"]):
            feats["t0"] = pos / fs
            feats["t1"] = (pos + seg.size) / fs
            windows.append(feats)
        pos += step
    return {"windows": windows, "summary": _summarize(windows, cancel),
            "cancel": cancel, "residual": residual}
