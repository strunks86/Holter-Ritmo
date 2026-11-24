"""HRV completa: serie NN, dominio temporal, frecuencial y no lineal.

Sigue las recomendaciones del Task Force (1996) y el enfoque de Kubios /
PhysioNet HRV Toolkit: antes de calcular se construye la serie **NN**
excluyendo ectopia y artefactos, y se informa qué proporción se descartó.

Cuatro decisiones de esta versión:

* **Ventana día/noche configurable** (por defecto día 08–20 h y noche 23–07 h,
  editables en los criterios del examen).  Un latido fuera de ambas ventanas
  queda como "transición" y no entra en ninguna de las dos, en vez de contarse
  como noche por descarte.
* **Bandas ULF/VLF/LF/HF con duración mínima**: si el tramo es más corto que
  el mínimo del Task Force la banda se informa como "no calculable" en vez de
  entregar un número que en realidad mide ruido de tendencia.  Lo mismo para
  SampEn cuando hay menos de ``MIN_SAMPEN_NN`` intervalos.
* **Welch y tacograma calcados a NeuroKit2** (spline cuadrática sobre rejilla a
  4 Hz, ventana Hann periódica de la mitad del registro, ``nfft = 2·nperseg``,
  densidad unilateral y trapecio sobre las mismas frecuencias), de modo que
  ``compare_neurokit`` verifica la coincidencia sobre la misma serie RR.
* **Nada de scipy obligatorio**: si scipy no está instalado el tacograma se
  interpola linealmente (queda anotado en ``interp="linear"``, que subestima
  HF); con scipy se usa la misma spline cuadrática que NeuroKit2.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# Bandas del Task Force (Hz).
BANDS = {
    "ulf": (0.0, 0.0033),
    "vlf": (0.0033, 0.04),
    "lf": (0.04, 0.15),
    "hf": (0.15, 0.40),
}
BAND_LABELS = {"ulf": "ULF", "vlf": "VLF", "lf": "LF", "hf": "HF"}

# Duración mínima (s) para que la potencia de una banda sea creíble.  La ULF
# exige 2 h: su límite superior son 0,0033 Hz (un ciclo cada 5 min) y con menos
# de eso la "potencia" mide la tendencia del tramo, no una oscilación.  El resto
# sigue el mínimo del Task Force para registros cortos (5 min de VLF, 2 min de
# LF y 1 min de HF).
MIN_BAND_S = {"ulf": 7200.0, "vlf": 300.0, "lf": 120.0, "hf": 60.0}

# Muestras NN mínimas para SampEn(m=2): con menos, la mitad de las plantillas
# no tiene con quién compararse.
MIN_SAMPEN_NN = 100

# Ventana día/noche por defecto (hora local, extremo final excluido).
DAY_DEFAULT = (8, 20)
NIGHT_DEFAULT = (23, 7)

# Frecuencia de interpolación del tacograma (Hz): la misma que NeuroKit2/Kubios.
INTERP_RATE = 4.0


# ---------------------------------------------------------------------------
# Serie NN
# ---------------------------------------------------------------------------
def nn_series(beats) -> dict:
    """Serie NN (ms) excluyendo ectopia/artefacto, con auditoría de descartes."""
    n = len(beats)
    if n < 3:
        return {"nn": np.zeros(0), "rr_total": 0.0, "rr_used": 0.0,
                "artifact_s": 0.0, "ectopy_s": 0.0, "rec": 0.0, "pct_used": 0.0,
                "n_nn": 0, "n_excluded": n}
    rr_ms = np.array([b.rr_prev for b in beats], dtype=float) * 1000.0
    cls = np.array([b.cls for b in beats])
    rr_total = float(np.nansum(rr_ms[1:]))
    med = float(np.median(rr_ms[(rr_ms > 300) & (rr_ms < 2000)])) if np.any(
        (rr_ms > 300) & (rr_ms < 2000)) else 800.0
    ok = np.zeros(n, dtype=bool)
    ectopic_s = 0.0
    artifact_s = 0.0
    for i in range(1, n):
        rr = rr_ms[i]
        if not np.isfinite(rr):
            continue
        if cls[i] != "N" or cls[i - 1] != "N":
            ectopic_s += rr          # ectopia: no es artefacto, es arritmia
            continue
        if 0.30 * med <= rr <= 2.0 * med:
            ok[i] = True
        else:
            artifact_s += rr         # latido perdido, doble detección, ruido
    nn = rr_ms[ok]
    rr_used = float(nn.sum())
    return {
        "nn": nn,
        "rr_total": rr_total,
        "rr_used": rr_used,
        "ectopy_s": ectopic_s,
        "artifact_s": artifact_s,
        "rec": float(rr_used / rr_total) if rr_total > 0 else 0.0,
        "pct_used": 100.0 * (rr_used / rr_total) if rr_total > 0 else 0.0,
        "n_nn": int(nn.size),
        "n_excluded": int(n - nn.size),
    }


# ---------------------------------------------------------------------------
# Ventana día/noche configurable
# ---------------------------------------------------------------------------
def hour_in_window(hour: int, start_h: int, end_h: int) -> bool:
    """¿La hora (0–23) cae en la ventana [start_h, end_h), que puede cruzar medianoche?"""
    start_h %= 24
    end_h %= 24
    if start_h == end_h:
        return True                                   # ventana de 24 h
    if start_h < end_h:
        return start_h <= hour < end_h
    return hour >= start_h or hour < end_h


def window_label(window: tuple[int, int]) -> str:
    """'23:00–07:00' para el informe y el visor."""
    return f"{int(window[0]) % 24:02d}:00–{int(window[1]) % 24:02d}:00"


def beat_clock(start_time: datetime, t_s: float) -> datetime:
    """Hora local del latido: inicio del examen más su tiempo transcurrido."""
    return start_time + timedelta(seconds=float(t_s))


def criteria_dict() -> dict:
    """Criterios vigentes: YAML por defecto más los ajustes guardados por el visor."""
    out: dict = {}
    try:
        from holterai.config import load_criteria
        out.update(load_criteria().model_dump())
    except Exception:
        pass
    try:
        from holterai.legacy.app.core import config as app_config
        path = Path(app_config.DATA_DIR) / "criteria.json"
        out.update(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        pass
    return out


def windows_from_criteria(criteria: dict | None = None
                          ) -> tuple[tuple[int, int], tuple[int, int]]:
    """Ventanas ``(día, noche)`` en horas a partir de los criterios del examen.

    Acepta el resultado de :func:`criteria_dict` o un ``CriteriaConfig``; las
    claves ausentes o fuera de rango caen a ``DAY_DEFAULT``/``NIGHT_DEFAULT``.
    """
    if criteria is None:
        data = criteria_dict()
    else:
        data = dict(getattr(criteria, "model_dump", lambda: criteria)())

    def hour(key: str, default: int) -> int:
        try:
            return int(data.get(key, default)) % 24
        except (TypeError, ValueError):
            return default

    day = (hour("day_window_start_h", DAY_DEFAULT[0]),
           hour("day_window_end_h", DAY_DEFAULT[1]))
    night = (hour("night_window_start_h", NIGHT_DEFAULT[0]),
             hour("night_window_end_h", NIGHT_DEFAULT[1]))
    return day, night


def split_day_night(beats, start_time: datetime | None,
                    day: tuple[int, int] | None = None,
                    night: tuple[int, int] | None = None):
    """Separa los latidos en día y noche según la hora local del examen.

    Devuelve ``(dia, noche, info)``.  Las ventanas salen de los criterios si no
    se pasan explícitas.  Los latidos de las horas que no caen en ninguna de las
    dos (por ejemplo las 21 h con día 08–20 y noche 23–07) no se asignan a
    ninguna: mezclarlos con la noche era justo lo que hacía el cálculo anterior
    por descarte.
    """
    if day is None or night is None:
        crit_day, crit_night = windows_from_criteria()
        day = crit_day if day is None else day
        night = crit_night if night is None else night
    day_beats, night_beats, between = [], [], 0
    if start_time is None:
        return day_beats, night_beats, {
            "day_window": window_label(day), "night_window": window_label(night),
            "n_day": 0, "n_night": 0, "n_between": 0, "start_time": None,
            "day_s": 0.0, "night_s": 0.0,
        }
    for b in beats:
        hour = beat_clock(start_time, b.t).hour
        if hour_in_window(hour, *day):
            day_beats.append(b)
        elif hour_in_window(hour, *night):
            night_beats.append(b)
        else:
            between += 1
    info = {
        "day_window": window_label(day),
        "night_window": window_label(night),
        "day_h": [int(day[0]) % 24, int(day[1]) % 24],
        "night_h": [int(night[0]) % 24, int(night[1]) % 24],
        "n_day": len(day_beats),
        "n_night": len(night_beats),
        "n_between": between,
        "day_s": round(_covered_s([b.rr_prev for b in day_beats])),
        "night_s": round(_covered_s([b.rr_prev for b in night_beats])),
        "start_time": start_time.isoformat(sep=" ", timespec="minutes"),
    }
    return day_beats, night_beats, info


# ---------------------------------------------------------------------------
# Dominio temporal
# ---------------------------------------------------------------------------
def time_domain(nn_ms: np.ndarray) -> dict:
    nn = np.asarray(nn_ms, dtype=float)
    if nn.size < 5:
        return {}
    drr = np.diff(nn)
    out = {
        "mean_nn_ms": round(float(nn.mean()), 1),
        "sdnn_ms": round(float(nn.std(ddof=1)), 1),
        "rmssd_ms": round(float(np.sqrt(np.mean(drr ** 2))), 1),
        # Task Force: NN50 se divide por el número de intervalos NN (igual que
        # NeuroKit2), no por el número de diferencias.  Dos decimales para que
        # el contraste con NeuroKit2 no se coma el error de redondeo.
        "pnn50_pct": round(float(np.sum(np.abs(drr) > 50.0)) / nn.size * 100.0, 2),
        "pnn20_pct": round(float(np.sum(np.abs(drr) > 20.0)) / nn.size * 100.0, 2),
        "hr_mean": round(60000.0 / float(nn.mean()), 1),
    }
    # índice triangular
    bins = np.arange(300, 2200, 7.8125)
    hist, _ = np.histogram(nn, bins=bins)
    out["triangular_index"] = round(float(nn.size / max(1, hist.max())), 1)
    # TINN por interpolación triangular
    try:
        idx = np.flatnonzero(hist > 0)
        if idx.size >= 3:
            x = bins[idx[:-1]] + 3.9
            y = hist[idx[:-1]].astype(float)
            h = y.max()
            area = np.trapezoid(y, x)
            out["tinn_ms"] = round(float(2.0 * area / max(h, 1e-9)), 1)
    except Exception:
        pass
    # SDANN y SDNN index sobre tramos de 5 min
    blocks, _edges = _blocks_5min(nn)
    if len(blocks) >= 2:
        means = [b.mean() for b in blocks]
        sds = [b.std(ddof=1) for b in blocks if b.size > 2]
        out["sdann_ms"] = round(float(np.std(means, ddof=1)), 1)
        out["sdnn_index_ms"] = round(float(np.mean(sds)), 1) if sds else 0.0
        out["n_5min_blocks"] = len(blocks)
    if nn.size >= 2:
        out["sd1_ms"] = round(float(np.sqrt(0.5) * np.std(drr, ddof=1)), 1)
        out["sd2_ms"] = round(float(np.sqrt(max(0.0, 2 * nn.var(ddof=1)
                                                - 0.5 * drr.var(ddof=1)))), 1)
        out["sd1_sd2"] = round(out["sd1_ms"] / max(1e-6, out["sd2_ms"]), 3)
    return out


def _blocks_5min(nn_ms: np.ndarray, block_ms: float = 300_000.0):
    t = np.cumsum(nn_ms)
    total = t[-1] if t.size else 0.0
    blocks = []
    edges = np.arange(0.0, total + block_ms, block_ms)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t >= a) & (t < b)
        if m.sum() >= 3:
            blocks.append(nn_ms[m])
    return blocks, edges


def _covered_s(rr_s) -> float:
    """Segundos cubiertos por la serie NN (lo que realmente ve la PSD)."""
    arr = np.asarray([x for x in rr_s if np.isfinite(x)], dtype=float)
    return float(np.nansum(arr))


# ---------------------------------------------------------------------------
# Dominio frecuencial (tacograma y Welch equivalentes a NeuroKit2)
# ---------------------------------------------------------------------------
def tachogram(nn_ms: np.ndarray, fs: float = INTERP_RATE):
    """Rejilla temporal uniforme y tacograma interpolado como NeuroKit2.

    NeuroKit2 interpola la serie NN con una spline cuadrática sobre una rejilla
    que empieza en el primer intervalo, termina un paso después del último y
    rellena el extremo con el último valor; se reproduce igual para que el
    contraste con NeuroKit2 no dependa de la interpolación.  Devuelve
    ``(grid_s, señal, método)`` o ``None`` si la serie es demasiado corta.
    """
    nn = np.asarray(nn_ms, dtype=float)
    if nn.size < 4:
        return None
    t = np.cumsum(nn) / 1000.0
    grid = np.arange(t[0], t[-1] + 1.0 / fs, 1.0 / fs)
    if grid.size < 16:
        return None
    try:
        from scipy.interpolate import interp1d
    except Exception:
        return grid, np.interp(grid, t, nn), "linear"
    spline = interp1d(t, nn, kind="quadratic", bounds_error=False,
                      fill_value=(nn[0], nn[-1]))
    return grid, np.asarray(spline(grid), dtype=float), "quadratic"


def welch_psd(nn_ms: np.ndarray, fs: float = INTERP_RATE) -> dict:
    """PSD de Welch de la serie NN interpolada a ``fs`` Hz, como NeuroKit2.

    Reproduce ``scipy.signal.welch(scaling="density", detrend=False,
    window="hann" periódica, noverlap=nperseg//2, nfft=2·nperseg,
    average="mean")`` sobre la señal sin media con ``nperseg = n//2`` (dos
    ciclos de la frecuencia mínima resoluble) y descarta las frecuencias bajo
    ``4·fs/n`` como ``signal_psd(min_frequency="default")``.
    Se implementa en numpy para no exigir scipy al visor.
    """
    tac = tachogram(nn_ms, fs)
    if tac is None:
        return {}
    _grid, x, interp = tac
    x = x - x.mean()
    n = x.size
    nperseg = max(4, int(n // 2))
    if nperseg > n:
        nperseg = n
    nfft = 2 * nperseg
    win = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(nperseg) / nperseg)
    step = nperseg - nperseg // 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    scale = 1.0 / (fs * float((win ** 2).sum()))
    acc = np.zeros(freqs.size)
    nseg = 0
    for start in range(0, n - nperseg + 1, step):
        seg = x[start:start + nperseg] * win
        p = np.abs(np.fft.rfft(seg, n=nfft)) ** 2 * scale
        p[1:] *= 2.0                       # densidad unilateral
        if nfft % 2 == 0:
            p[-1] *= 0.5                   # el bin de Nyquist no se dobla
        acc += p
        nseg += 1
    if nseg == 0:
        return {}
    return {
        "freqs": freqs,
        "psd": acc / nseg,
        "seg_s": round(nperseg / fs, 1),
        "df_hz": round(fs / nfft, 6),
        "min_f_hz": 4.0 * fs / n,
        "n_segments": nseg,
        "interp": interp,
    }


def band_powers(nn_ms: np.ndarray, fs: float = INTERP_RATE) -> dict:
    """Potencia (ms²) por banda integrando la PSD con trapecio.

    Devuelve las cuatro bandas sin aplicar mínimos de duración (para contraste
    numérico); el filtro clínico lo aplica :func:`frequency_domain`.
    """
    return _powers_from_psd(welch_psd(nn_ms, fs))


def _powers_from_psd(psd_info: dict) -> dict:
    """Potencia (ms²) por banda a partir de una PSD ya calculada."""
    if not psd_info:
        return {k: 0.0 for k in BANDS}
    freqs = psd_info["freqs"]
    psd = psd_info["psd"]
    f_min = psd_info["min_f_hz"]
    out = {}
    for key, (lo, hi) in BANDS.items():
        m = (freqs >= max(lo, f_min)) & (freqs < hi)
        out[key] = float(np.trapezoid(psd[m], freqs[m])) if int(m.sum()) >= 2 else 0.0
    return out


def _band_rows(raw: dict, dur_s: float) -> list[dict]:
    total = float(sum(raw.values()))
    rows = []
    for key in BANDS:
        enough = dur_s >= MIN_BAND_S[key]
        rows.append({
            "key": key,
            "band": BAND_LABELS[key],
            "lo_hz": BANDS[key][0],
            "hi_hz": BANDS[key][1],
            "ms2": round(raw[key], 1) if enough else None,
            "pct": round(100.0 * raw[key] / total, 1) if enough and total > 0 else None,
            "calculable": bool(enough),
            "need_s": MIN_BAND_S[key],
            "have_s": round(float(dur_s), 1),
        })
    return rows


def frequency_domain(nn_ms: np.ndarray, fs: float = INTERP_RATE,
                     duration_s: float | None = None) -> dict:
    """Densidad espectral por Welch sobre la serie NN interpolada a 4 Hz.

    La duración que manda es la que cubren los intervalos NN (es la que ve la
    PSD); ``duration_s`` solo se usa si esa serie quedó vacía.  Las bandas que
    el tramo no permite resolver se devuelven en ``None`` y quedan marcadas en
    ``bands`` con ``calculable=False`` ("no calculable"), en vez de un número
    engañoso.
    """
    nn = np.asarray(nn_ms, dtype=float)
    dur_s = _covered_s(nn / 1000.0)
    if duration_s is not None and dur_s <= 0:
        dur_s = float(duration_s)
    vacio = {f"{k}_ms2": None for k in BANDS}
    vacio.update({f"{k}_pct": None for k in BANDS})
    if nn.size < 60 or dur_s < MIN_BAND_S["hf"]:
        out: dict = dict(vacio)
        out.update({
            "total_power_ms2": None, "lf_hf": None, "lf_nu": None, "hf_nu": None,
            "bands": _band_rows({k: 0.0 for k in BANDS}, dur_s),
            "no_calculables": [BAND_LABELS[k] for k in BANDS],
            "note": f"tramo insuficiente para el dominio frecuencial ({dur_s:.0f} s)",
        })
        return out

    psd_info = welch_psd(nn, fs=fs)
    if not psd_info:
        out = dict(vacio)
        out.update({"total_power_ms2": None, "lf_hf": None, "lf_nu": None, "hf_nu": None,
                    "bands": _band_rows({k: 0.0 for k in BANDS}, 0.0),
                    "no_calculables": [BAND_LABELS[k] for k in BANDS],
                    "note": "no se pudo estimar la densidad espectral"})
        return out

    raw = _powers_from_psd(psd_info)
    rows = _band_rows(raw, dur_s)
    total = float(sum(raw.values()))
    out = {}
    for row in rows:
        key = row["key"]
        out[f"{key}_ms2"] = row["ms2"]
        out[f"{key}_pct"] = row["pct"]
    out["total_power_ms2"] = round(total, 1) if total > 0 else None
    calc = {r["key"]: r["calculable"] for r in rows}
    lf, hf = raw["lf"], raw["hf"]
    ok_lf_hf = calc["lf"] and calc["hf"] and hf > 0
    out["lf_hf"] = round(lf / hf, 3) if ok_lf_hf else None
    denom = lf + hf if ok_lf_hf else 0.0
    out["lf_nu"] = round(100.0 * lf / denom, 1) if denom > 0 else None
    out["hf_nu"] = round(100.0 * hf / denom, 1) if denom > 0 else None
    out["bands"] = rows
    out["n_segments"] = psd_info["n_segments"]
    out["seg_s"] = psd_info["seg_s"]
    out["df_hz"] = psd_info["df_hz"]
    out["interp"] = psd_info["interp"]
    out["duration_s"] = round(dur_s, 1)
    out["no_calculables"] = [r["band"] for r in rows if not r["calculable"]]
    if out["no_calculables"]:
        falta = [f"{r['band']} ({_dur(dur_s)} < {_dur(r['need_s'])})"
                 for r in rows if not r["calculable"]]
        out["note"] = "bandas no calculables: " + ", ".join(falta)
    return out


# ---------------------------------------------------------------------------
# Dominio no lineal
# ---------------------------------------------------------------------------
def dfa_alpha(nn_ms: np.ndarray) -> dict:
    nn = np.asarray(nn_ms, dtype=float)
    if nn.size < 100:
        return {}
    y = np.cumsum(nn - nn.mean())
    scales = np.unique(np.round(np.logspace(np.log10(4), np.log10(64), 12)).astype(int))
    fs_vals, ns = [], []
    for s in scales:
        if s >= y.size // 2:
            continue
        nwin = y.size // s
        if nwin < 2:
            continue
        seg = y[:nwin * s].reshape(nwin, s)
        t = np.arange(s)
        tm = t - t.mean()
        denom = (tm ** 2).sum()
        slopes = (seg * tm).sum(axis=1) / max(denom, 1e-9)
        fit = seg.mean(axis=1, keepdims=True) + slopes[:, None] * tm
        rms = float(np.sqrt(((seg - fit) ** 2).mean()))
        if rms <= 0.0:                      # serie constante: no hay escala
            continue
        fs_vals.append(rms)
        ns.append(s)
    if len(fs_vals) < 3:
        return {}
    alpha = float(np.polyfit(np.log(ns), np.log(fs_vals), 1)[0])
    out = {"dfa_alpha": round(alpha, 3)}
    a1 = [(n, f) for n, f in zip(ns, fs_vals) if 4 <= n <= 16]
    a2 = [(n, f) for n, f in zip(ns, fs_vals) if 16 <= n <= 64]
    for name, pts in (("dfa_alpha1", a1), ("dfa_alpha2", a2)):
        if len(pts) >= 3:
            lx = np.log([p[0] for p in pts])
            ly = np.log([p[1] for p in pts])
            out[name] = round(float(np.polyfit(lx, ly, 1)[0]), 3)
    return out


def sample_entropy(nn_ms: np.ndarray, m: int = 2, max_points: int = 2000) -> float | None:
    """SampEn(m, r = 0,2·DE) o ``None`` si no hay intervalos suficientes."""
    from holterai.legacy.app.signal.rhythm import _sample_entropy
    x = np.asarray(nn_ms, dtype=float)
    if x.size < MIN_SAMPEN_NN:
        return None
    x = x[:max_points]
    r = 0.2 * float(x.std(ddof=1))
    if r <= 0:
        return None
    return round(float(_sample_entropy(x, m=m, r=r)), 3)


# ---------------------------------------------------------------------------
# Bloque completo para el informe
# ---------------------------------------------------------------------------
def compute(beats, duration_s: float | None = None) -> dict:
    """Bloque HRV completo, listo para el informe.

    Siempre devuelve las claves ``bands`` (ULF, VLF, LF, HF) y ``sampen``; las
    que el tramo no permite calcular quedan en ``None`` con ``calculable=False``
    para que el visor y el informe muestren "no calculable".
    """
    series = nn_series(beats)
    nn = series["nn"]
    dur_s = _covered_s(nn / 1000.0)
    if dur_s <= 0 and duration_s:
        dur_s = float(duration_s)
    out = {
        "rec": round(series["rec"], 3),
        "pct_used": round(series["pct_used"], 1),
        "artifact_s": round(series["artifact_s"], 1),
        "ectopy_s": round(series["ectopy_s"], 1),
        "n_nn": series["n_nn"],
    }
    if nn.size < 10:
        fd = frequency_domain(nn, duration_s=dur_s)
        out.update(fd)
        out["note"] = "intervalos NN insuficientes (ectopia o artefacto)"
        out["sampen"] = None
        out["sampen_note"] = f"no calculable (faltan {MIN_SAMPEN_NN} intervalos NN)"
        return out
    out.update(time_domain(nn))
    fd = frequency_domain(nn, duration_s=dur_s)
    out.update(fd)
    out.update(dfa_alpha(nn))
    se = sample_entropy(nn)
    out["sampen"] = se
    if se is not None:
        out["sampen_note"] = None
    elif float(np.std(nn, ddof=1)) <= 0.0:
        out["sampen_note"] = "no calculable (serie sin variabilidad)"
    else:
        out["sampen_note"] = f"no calculable (faltan {MIN_SAMPEN_NN} intervalos NN)"
    return out


def day_night_hrv(beats, start_time, day: tuple[int, int] | None = None,
                  night: tuple[int, int] | None = None) -> dict:
    """Bloques HRV día/noche con las ventanas vigentes.

    Devuelve ``{"day": {...}, "night": {...}, ...info de ventanas}``; si el
    examen no tiene hora de inicio las dos listas quedan vacías y las ventanas
    se informan igual para que el visor muestre la configuración aplicada.
    """
    day_beats, night_beats, info = split_day_night(beats, start_time,
                                                   day=day, night=night)
    out = dict(info)
    out["day"] = compute(day_beats, info.get("day_s") or None)
    out["night"] = compute(night_beats, info.get("night_s") or None)
    return out


# ---------------------------------------------------------------------------
# Texto para el informe
# ---------------------------------------------------------------------------
def _txt(value, nd: int = 1, unit: str = "") -> str:
    if value is None:
        return "no calculable"
    return f"{float(value):.{nd}f}{unit}"


def _dur(seconds: float) -> str:
    """'2.0 h' o '90 min' según la magnitud."""
    seg = float(seconds)
    return f"{seg / 3600.0:.1f} h" if seg >= 3600.0 else f"{seg / 60.0:.0f} min"


def band_texts(hrv: dict) -> list[tuple[str, str]]:
    """``(banda, texto)`` de cada banda: 'ms² (%)' o 'no calculable (faltan …)'."""
    out = []
    for row in hrv.get("bands") or []:
        if row.get("calculable"):
            pct = row.get("pct")
            txt = f"{row['ms2']:.1f} ms²" + (f" ({pct:.1f} %)" if pct is not None else "")
        else:
            txt = (f"no calculable ({_dur(row['have_s'])} "
                   f"< {_dur(row['need_s'])})")
        out.append((row["band"], txt))
    return out


def _metric(label: str, value, nd: int = 1, unit: str = "") -> str:
    """'SDNN 132.0 ms' o 'SDNN no calculable' (la unidad no se cuelga del texto)."""
    if value is None:
        return f"{label} no calculable"
    return f"{label} {float(value):.{nd}f}{unit}"


def summary_lines(hrv: dict) -> list[str]:
    """Líneas del bloque HRV para el informe en Word/HTML.

    Siempre incluye las cuatro bandas y SampEn: cuando el tramo no alcanza la
    duración mínima el texto dice "no calculable" con las horas que faltan.
    """
    if not hrv:
        return []
    lines = []
    if hrv.get("sdnn_ms") is not None:
        lines.append("Dominio temporal: " + " · ".join([
            _metric("NN usados", hrv.get("pct_used"), 1, " %"),
            _metric("FC media", hrv.get("hr_mean"), 1, " lpm"),
            _metric("SDNN", hrv.get("sdnn_ms"), 1, " ms"),
            _metric("RMSSD", hrv.get("rmssd_ms"), 1, " ms"),
            _metric("SDANN", hrv.get("sdann_ms"), 1, " ms"),
            _metric("pNN50", hrv.get("pnn50_pct"), 1, " %"),
            _metric("SD1/SD2", hrv.get("sd1_sd2"), 2),
        ]))
    bands = band_texts(hrv)
    if bands:
        detalle = " · ".join(f"{name} {txt}" for name, txt in bands)
        extra = []
        if hrv.get("lf_hf") is not None:
            extra.append(_metric("LF/HF", hrv.get("lf_hf"), 2))
        if hrv.get("lf_nu") is not None:
            extra.append(f"LF {_txt(hrv.get('lf_nu'))} n.u. / HF {_txt(hrv.get('hf_nu'))} n.u.")
        lines.append("Dominio frecuencial: " + detalle +
                     ("; " + " · ".join(extra) if extra else ""))
    nonlinear = []
    if hrv.get("sampen") is not None:
        nonlinear.append(f"SampEn {_txt(hrv.get('sampen'), 3)} (m=2, r=0,2·DE)")
    else:
        nota = str(hrv.get("sampen_note") or "")
        if nota.startswith("no calculable"):
            nota = nota[len("no calculable"):].strip(" ()")
        nonlinear.append("SampEn no calculable" + (f" ({nota})" if nota else ""))
    if hrv.get("dfa_alpha1") is not None:
        nonlinear.append(f"DFA α1 {_txt(hrv.get('dfa_alpha1'), 2)}")
    lines.append("Dominio no lineal: " + " · ".join(nonlinear))
    nota = str(hrv.get("note") or "")
    # El detalle banda por banda ya dice qué falta; repetirlo sobra en el informe.
    if nota and not nota.startswith("bandas no calculables"):
        lines.append(nota)
    return lines


def day_night_lines(day: dict | None, night: dict | None,
                    windows: dict | None = None) -> list[str]:
    """Una línea por ventana con SDNN, RMSSD y bandas calculables."""
    windows = windows or {}
    if "start_time" in windows and not windows["start_time"]:
        return ["Ventanas día/noche no calculables: el examen no tiene hora de inicio."]
    out = []
    for label, block, key in (("día", day or {}, "day_window"),
                              ("noche", night or {}, "night_window")):
        window = windows.get(key)
        head = f"Ventana {label}" + (f" {window}" if window else "") + ":"
        if not block or block.get("n_nn", 0) < 10:
            out.append(f"{head} sin intervalos NN suficientes.")
            continue
        parts = [_metric("SDNN", block.get("sdnn_ms"), 1, " ms"),
                 _metric("RMSSD", block.get("rmssd_ms"), 1, " ms")]
        for row in block.get("bands") or []:
            parts.append(_metric(row["band"], row.get("ms2"), 1, " ms²")
                         if row.get("calculable") else f"{row['band']} no calculable")
        parts.append(_metric("SampEn", block.get("sampen"), 3))
        out.append(f"{head} " + " · ".join(parts))
    return out


# ---------------------------------------------------------------------------
# Contraste con NeuroKit2 (si está instalado)
# ---------------------------------------------------------------------------
def compare_neurokit(nn_ms: np.ndarray, fs: float = INTERP_RATE,
                     tol_pct: float = 5.0) -> dict | None:
    """Compara nuestros índices con NeuroKit2 sobre la misma serie NN.

    Devuelve ``{"ours": {...}, "nk": {...}, "diff_pct": {...}, "max_diff_pct":
    float, "over_tol": [...]}`` o ``None`` si NeuroKit2 no está instalado.

    Se entregan a NeuroKit2 los mismos RR en milisegundos enteros (como los ve
    al leer picos) y se lee su ``hrv_frequency``/``hrv_time`` estándar, de modo
    que la comparación cubre tacograma, PSD, integración por bandas y dominio
    temporal.  SampEn se compara sobre la misma porción de 2000 intervalos y el
    mismo radio ``r = 0,2·DE``.  ``over_tol`` lista las métricas que se separan
    más de ``tol_pct`` por ciento.
    """
    try:
        import neurokit2 as nk
        from neurokit2.complexity import entropy_sample
    except Exception:
        return None
    nn = np.asarray(nn_ms, dtype=float)
    nn = nn[np.isfinite(nn)]
    if nn.size < MIN_SAMPEN_NN:
        return None
    peaks = np.concatenate(([0], np.cumsum(np.round(nn)))).astype(np.int64)
    nn_int = np.diff(peaks).astype(float)          # RR tal como los lee NeuroKit2

    ours_time = time_domain(nn_int)
    ours = {
        "mean_nn_ms": ours_time.get("mean_nn_ms"),
        "sdnn_ms": ours_time.get("sdnn_ms"),
        "rmssd_ms": ours_time.get("rmssd_ms"),
        "pnn50_pct": ours_time.get("pnn50_pct"),
        "sampen": sample_entropy(nn_int),
    }
    raw = band_powers(nn_int, fs=fs)
    ours.update({f"{k}_ms2": round(v, 3) for k, v in raw.items()})
    ours["lf_hf"] = round(raw["lf"] / raw["hf"], 4) if raw["hf"] > 0 else None

    try:
        t_nk = nk.hrv_time(peaks, sampling_rate=1000).iloc[0]
        f_nk = nk.hrv_frequency(peaks, sampling_rate=1000,
                                interpolation_rate=int(round(fs)),
                                normalize=False).iloc[0]
    except Exception:
        return None
    x = nn_int[:2000]
    r = 0.2 * float(x.std(ddof=1))
    try:
        se_nk = float(entropy_sample(x, delay=1, dimension=2, tolerance=r)[0])
    except Exception:
        se_nk = None
    nk_vals = {
        "mean_nn_ms": float(t_nk.get("HRV_MeanNN", np.nan)),
        "sdnn_ms": float(t_nk.get("HRV_SDNN", np.nan)),
        "rmssd_ms": float(t_nk.get("HRV_RMSSD", np.nan)),
        "pnn50_pct": float(t_nk.get("HRV_pNN50", np.nan)),
        "ulf_ms2": float(f_nk.get("HRV_ULF", np.nan)),
        "vlf_ms2": float(f_nk.get("HRV_VLF", np.nan)),
        "lf_ms2": float(f_nk.get("HRV_LF", np.nan)),
        "hf_ms2": float(f_nk.get("HRV_HF", np.nan)),
        "lf_hf": float(f_nk.get("HRV_LFHF", np.nan)),
        "sampen": se_nk,
    }
    diff: dict = {}
    for key, ours_v in ours.items():
        ref = nk_vals.get(key)
        if ours_v is None or ref is None:
            diff[key] = None
            continue
        if not np.isfinite(ref) or ref == 0.0:
            diff[key] = None
            continue
        diff[key] = round(100.0 * abs(float(ours_v) - float(ref)) / abs(float(ref)), 3)
    validos = {k: v for k, v in diff.items() if v is not None}
    return {
        "ours": ours,
        "nk": nk_vals,
        "diff_pct": diff,
        "max_diff_pct": max(validos.values()) if validos else None,
        "over_tol": sorted(k for k, v in validos.items() if v >= tol_pct),
        "tol_pct": float(tol_pct),
        "n_nn": int(nn.size),
        "fs": float(fs),
        "interp": (tachogram(nn_int, fs) or (None, None, None))[2],
        "n_comparados": len(validos),
        "nk_version": getattr(nk, "__version__", None),
    }
