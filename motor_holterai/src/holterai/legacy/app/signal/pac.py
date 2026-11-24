"""Análisis de extrasístoles supraventriculares (PAC).

La morfología del QRS de una PAC es prácticamente igual a la sinusal; lo
que la delata es el contexto:

  * prematuridad (RR corto),
  * onda P prematura y de morfología distinta (o ausente, si es juncional),
  * pausa no compensadora (RR previo + siguiente < 2 x RR basal),
  * QRS estrecho e idéntico al sinusal.

Se construyen plantillas del propio paciente (QRS, P y T de los latidos
sinusales) y se mide cuánto se aparta cada latido de ellas. El cálculo está
vectorizado por registro para que un Holter de 24 h no tarde minutos.
"""
from __future__ import annotations

import numpy as np

from holterai.legacy.app.signal.filters import resample

QRS_PRE, QRS_POST, QRS_N = 0.10, 0.10, 48
P_PRE, P_POST, P_N = 0.33, 0.04, 64
T_PRE, T_POST, T_N = 0.12, 0.40, 48
P_MS_PER_POINT = (P_PRE + P_POST) * 1000.0 / P_N

PAC_FEATURES = [
    "rr_prev", "rr_next", "rr_ratio", "pause_ratio", "qrs_width",
    "qrs_corr", "p_corr", "p_shift", "p_amp", "p_present", "noise", "t_corr",
    "rr_slope", "af_prob",
]
IDX = {name: i for i, name in enumerate(PAC_FEATURES)}


# ---------------------------------------------------------------------------
# Ventanas
# ---------------------------------------------------------------------------
def windows(x: np.ndarray, fs: float, peaks: np.ndarray, pre: float,
            post: float, n: int) -> np.ndarray:
    peaks = np.asarray(peaks, dtype=int)
    out = np.zeros((len(peaks), n), dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    for k, p in enumerate(peaks):
        a = int(p - round(pre * fs))
        b = int(p + round(post * fs))
        seg = np.zeros(b - a, dtype=np.float64)
        lo, hi = max(0, a), min(x.size, b)
        if hi > lo:
            seg[lo - a:hi - a] = x[lo:hi]
        out[k] = resample(seg, n)
    return out


def _norm_rows(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    w = w - w.mean(axis=1, keepdims=True)
    n = np.linalg.norm(w, axis=1, keepdims=True)
    return w / np.maximum(n, 1e-9)


def ncc_batch(w: np.ndarray, template: np.ndarray,
              max_shift: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Correlación cruzada normalizada de cada fila con la plantilla."""
    if template is None or len(w) == 0:
        return np.zeros(len(w)), np.zeros(len(w), dtype=int)
    wn = _norm_rows(w)
    t = np.asarray(template, dtype=np.float64)
    if max_shift <= 0:
        tn = _norm_rows(t[None])[0]
        return wn @ tn, np.zeros(len(w), dtype=int)

    best = np.full(len(w), -1.0)
    best_s = np.zeros(len(w), dtype=int)
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            a = wn[:, s:]
            b = t[: t.size - s]
        else:
            a = wn[:, : wn.shape[1] + s]
            b = t[-s:]
        if a.shape[1] < 8:
            continue
        a = _norm_rows(a)
        bb = b - b.mean()
        nb = np.linalg.norm(bb)
        if nb < 1e-9:
            continue
        c = a @ (bb / nb)
        upd = c > best
        best[upd] = c[upd]
        best_s[upd] = s
    return np.clip(best, -1, 1), best_s


# ---------------------------------------------------------------------------
# Plantillas
# ---------------------------------------------------------------------------
def build_templates(x: np.ndarray, fs: float, peaks: np.ndarray,
                    n_iter: int = 2) -> dict:
    """Plantillas de QRS, P y T a partir de la mediana de los latidos."""
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size < 5:
        return {"qrs": None, "p": None, "t": None, "ok": False}
    qw = windows(x, fs, peaks, QRS_PRE, QRS_POST, QRS_N)
    pw = windows(x, fs, peaks, P_PRE, P_POST, P_N)
    tw = windows(x, fs, peaks, T_PRE, T_POST, T_N)

    q_t, p_t, t_t = np.median(qw, axis=0), np.median(pw, axis=0), np.median(tw, axis=0)
    for _ in range(max(1, n_iter)):
        c, _ = ncc_batch(qw, q_t)
        sel = c > np.percentile(c, 40)
        if sel.sum() < max(8, len(peaks) // 5):
            break
        q_t = np.median(qw[sel], axis=0)
        p_t = np.median(pw[sel], axis=0)
        t_t = np.median(tw[sel], axis=0)

    p_scale = float(np.percentile(np.abs(p_t - p_t.mean()), 99)) or 1e-3
    return {"qrs": q_t, "p": p_t, "t": t_t, "p_scale": p_scale, "ok": True}


# ---------------------------------------------------------------------------
# Características
# ---------------------------------------------------------------------------
def beat_features_batch(x: np.ndarray, fs: float, peaks: np.ndarray,
                        templates: dict, rr_prev: np.ndarray, rr_next: np.ndarray,
                        rr_med: np.ndarray, widths: np.ndarray,
                        noises: np.ndarray, af_probs: np.ndarray | None = None,
                        rr_prev2: np.ndarray | None = None) -> np.ndarray:
    """Matriz (N, 14) de características orientadas a PAC."""
    n = len(peaks)
    feats = np.zeros((n, len(PAC_FEATURES)), dtype=np.float32)
    if n == 0:
        return feats
    rr_prev = np.asarray(rr_prev, dtype=np.float64)
    rr_next = np.asarray(rr_next, dtype=np.float64)
    rr_med = np.asarray(rr_med, dtype=np.float64)
    if af_probs is None:
        af_probs = np.zeros(n)
    if rr_prev2 is None:
        rr_prev2 = np.concatenate([[0.0], rr_prev[:-1]]) if n > 1 else np.zeros(n)

    qw = windows(x, fs, peaks, QRS_PRE, QRS_POST, QRS_N)
    pw = windows(x, fs, peaks, P_PRE, P_POST, P_N)
    tw = windows(x, fs, peaks, T_PRE, T_POST, T_N)

    if templates.get("ok"):
        q_corr, _ = ncc_batch(qw, templates["qrs"])
        p_corr, p_shift = ncc_batch(pw, templates["p"], max_shift=8)
        t_corr, _ = ncc_batch(tw, templates["t"], max_shift=4)
        p_scale = max(1e-6, float(templates.get("p_scale", 1.0)))
    else:
        q_corr = np.zeros(n)
        p_corr = np.zeros(n)
        p_shift = np.zeros(n, dtype=int)
        t_corr = np.zeros(n)
        p_scale = 1.0

    p_centered = pw - pw.mean(axis=1, keepdims=True)
    p_amp = np.max(np.abs(p_centered), axis=1) / p_scale
    p_present = np.max(np.abs(p_centered), axis=1) / (4.0 * np.maximum(noises, 1e-4))

    rr_ratio = rr_prev / np.maximum(rr_med, 1e-6)
    pause = (rr_prev + rr_next) / (2.0 * np.maximum(rr_med, 1e-6))
    rr_slope = rr_prev / np.maximum(rr_prev2, 1e-6)

    feats[:, 0] = np.clip(rr_prev, 0, 4)
    feats[:, 1] = np.clip(rr_next, 0, 4)
    feats[:, 2] = np.clip(rr_ratio, 0.3, 3.0)
    feats[:, 3] = np.clip(pause, 0.5, 3.0)
    feats[:, 4] = np.clip(widths, 0.03, 0.25)
    feats[:, 5] = np.clip(q_corr, -1, 1)
    feats[:, 6] = np.clip(p_corr, -1, 1)
    feats[:, 7] = np.clip(p_shift * P_MS_PER_POINT / 100.0, -1, 1)
    feats[:, 8] = np.clip(p_amp, 0, 4)
    feats[:, 9] = np.clip(p_present, 0, 5)
    feats[:, 10] = np.clip(noises, 0, 1)
    feats[:, 11] = np.clip(t_corr, -1, 1)
    feats[:, 12] = np.clip(rr_slope, 0.3, 3.0)
    feats[:, 13] = np.clip(af_probs, 0, 1)
    return feats


def beat_features(x: np.ndarray, fs: float, idx: int, templates: dict,
                  rr_prev: float, rr_next: float, rr_med: float,
                  qrs_width: float, noise: float,
                  rr_prev2: float = 0.0, af_prob: float = 0.0) -> np.ndarray:
    """Versión para un solo latido (pruebas y utilidades)."""
    arr = np.array([idx], dtype=int)
    out = beat_features_batch(
        x, fs, arr, templates,
        np.array([rr_prev]), np.array([rr_next]), np.array([rr_med]),
        np.array([qrs_width]), np.array([noise]),
        np.array([af_prob]), np.array([rr_prev2]),
    )
    return out[0]


# ---------------------------------------------------------------------------
# Reglas clínicas
# ---------------------------------------------------------------------------
def pac_rules(feat: np.ndarray, qrs_width_s: float | None = None) -> dict:
    """Interpretación clínica de las características de un latido."""
    if feat is None or len(feat) < len(PAC_FEATURES):
        return {"score": 0.0, "is_pac": False, "reasons": []}
    f = [float(v) for v in feat]
    rr_ratio, pause_ratio = f[IDX["rr_ratio"]], f[IDX["pause_ratio"]]
    width, q_corr = f[IDX["qrs_width"]], f[IDX["qrs_corr"]]
    p_corr, p_shift = f[IDX["p_corr"]], f[IDX["p_shift"]]
    p_present, noise, af_prob = f[IDX["p_present"]], f[IDX["noise"]], f[IDX["af_prob"]]

    premature = rr_ratio < 0.88
    narrow = width < 0.115
    sinus_like = q_corr > 0.80
    non_compensatory = pause_ratio < 1.85
    p_different = (p_corr < 0.72) or (abs(p_shift) > 0.25)
    p_absent = p_present < 0.9
    irregular = af_prob > 0.5

    score = 0.0
    reasons = []
    if premature:
        score += 0.32
        reasons.append(f"prematuro (RR {rr_ratio:.2f}x)")
    if sinus_like:
        score += 0.20
        reasons.append(f"QRS igual al sinusal (r={q_corr:.2f})")
    if narrow:
        score += 0.14
        reasons.append(f"QRS estrecho ({width*1000:.0f} ms)")
    if non_compensatory:
        score += 0.14
        reasons.append(f"pausa no compensadora ({pause_ratio:.2f}x)")
    else:
        score -= 0.10
    if p_different:
        score += 0.20
        reasons.append(f"P distinta (r={p_corr:.2f})")
    if p_absent:
        score += 0.08
        reasons.append("P no visible")
    if irregular:
        score -= 0.45
        reasons.append("ritmo irregular tipo FA")
    if noise > 0.25:
        score -= 0.15
        reasons.append("ruido alto")
    score = float(np.clip(score, 0.0, 1.0))
    return {"score": score,
            "is_pac": bool(score >= 0.62 and premature and not irregular),
            "reasons": reasons}

