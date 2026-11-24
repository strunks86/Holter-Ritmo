"""Detección de onda P por plantilla local (prototipo T10).

Diseño (RESPUESTAS-SECCION19 §2):

  * la plantilla de P se arma **por minuto** con la mediana de las ventanas
    del propio registro (no hay una plantilla universal: la P cambia con la
    derivación y con la posición del paciente),
  * dentro de cada minuto la plantilla sale de una de las dos mitades de
    latidos (pares o impares) y se evalúa sobre la otra mitad; qué mitad hace
    de plantilla alterna con el minuto (**control de mitades par/impar**), de
    modo que ningún latido se mida contra una plantilla construida con él,
  * antes de buscar la P se **resta la onda T del latido anterior**, escalada
    en tiempo con sqrt(RR / RR medio) (Bazett), para que la cola de la T no se
    confunda con una P y para no perder la P cuando el RR es corto,
  * la P de una plantilla se **acepta o se descarta antes de medir latidos**:
    amplitud pico-pico mínima, tres veces el ruido de alta frecuencia de la
    propia plantilla y correlación >= 0,80 entre las plantillas de las dos
    mitades (el ruido no se repite, una P sí). Si el minuto no pasa el control,
    sus latidos quedan evaluados como "sin P" y no se inventan detecciones,
  * la P de cada latido se localiza por **correlación normalizada con la
    plantilla en ±40 ms** alrededor de su posición esperada, y se exige una
    amplitud comparable a la de la plantilla (no un piso absoluto: una P de
    bajo voltaje se pierde si se compara con el ruido del latido),
  * con **FC > 120 lpm** (RR < 500 ms) la P cae sobre la T previa: el latido
    queda como *no evaluable* en vez de fabricar una detección.

El módulo es numpy puro y trabaja sobre un registro ya filtrado (0.5-40 Hz,
p. ej. ``filters.preprocess(x, fs, "display")``) y en las mismas unidades que
la entrada (el informe de ``amp_uv`` supone mV). No filtra ni remuestrea.
Todas las salidas son arreglos por latido, alineados con ``peaks``.
"""
from __future__ import annotations

import numpy as np

# --- geometría de la ventana de P (segundos, relativa al pico R) ------------
P_PRE = 0.30          # la ventana de P empieza 300 ms antes del R ...
P_POST = -0.10        # ... y termina 100 ms antes del R (QRS excluido)
T_ON = 0.10           # plantilla de T del latido previo: R + 0.10 s ...
T_OFF = 0.45          # ... hasta R + 0.45 s
SHIFT_MS = 40.0       # ±40 ms de búsqueda por correlación
SMOOTH_MS = 12.0      # suavizado de la ventana antes de correlacionar

# --- reglas de decisión ----------------------------------------------------
MAX_HR = 120.0                                  # FC máxima evaluable
RR_MIN = 60.0 / MAX_HR                          # 0.5 s
SEGMENT_S = 60.0                                # plantilla por minuto
MIN_TEMPLATE_BEATS = 3                          # mínimo para armar plantilla
CORR_MIN = 0.60                                 # correlación mínima de la P
AMP_MIN_UV = 12.0                               # amplitud pico-pico mínima
AMP_REL_MIN = 0.50                              # ... y fracción de la plantilla
TPL_AMP_MIN_UV = 8.0                            # P mínima en la plantilla
TPL_NOISE_FACTOR = 3.0                          # (informativo) P vs ruido HF
HALVES_CORR_MIN = 0.80                          # consistencia par/impar (diseño)
ONSET_FRAC = 0.20                               # inicio al 20 % de la amplitud
ONSET_MAX_MS = 120.0                            # duración máxima de la P

STATUS_OK = "ok"
STATUS_EDGE = "borde"
STATUS_FAST = "fc>120"
STATUS_NO_TEMPLATE = "sin_plantilla"
STATUS_TEMPLATE_HALF = "mitad_plantilla"
STATUS_NOT_EVALUABLE = (STATUS_EDGE, STATUS_FAST, STATUS_NO_TEMPLATE,
                        STATUS_TEMPLATE_HALF)


# ---------------------------------------------------------------------------
# Utilidades de ventanas
# ---------------------------------------------------------------------------
def _geom(fs: float) -> tuple[float, float, int, int, int]:
    """(pre, post, n_ventana, corrimiento máximo, largo de plantilla)."""
    pre = P_PRE + SHIFT_MS / 1000.0
    post = P_POST + SHIFT_MS / 1000.0
    n_wide = max(9, int(round((pre + post) * fs)))
    shift = max(1, int(round(SHIFT_MS / 1000.0 * fs)))
    m = n_wide - 2 * shift
    if m < 9:                     # fs muy baja: recorta el corrimiento
        shift = max(1, (n_wide - 9) // 2)
        m = n_wide - 2 * shift
    return pre, post, n_wide, shift, m


def beat_windows(x: np.ndarray, fs: float, peaks: np.ndarray,
                 pre: float, post: float) -> tuple[np.ndarray, np.ndarray]:
    """Ventanas crudas alrededor de cada pico; ``pre`` puede ser negativo."""
    x = np.asarray(x, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=int)
    n = max(2, int(round((pre + post) * fs)))
    if peaks.size == 0:
        return np.zeros((0, n), dtype=np.float64), np.zeros(0, dtype=bool)
    starts = peaks - int(round(pre * fs))
    idx = starts[:, None] + np.arange(n)[None, :]
    inside = (idx >= 0) & (idx < x.size)
    out = x[np.clip(idx, 0, max(0, x.size - 1))]
    out = np.where(inside, out, 0.0)
    return out, inside.all(axis=1)


def _smooth_rows(w: np.ndarray, win: int) -> np.ndarray:
    """Media móvil por fila (ventana impar, borde replicado)."""
    win = max(1, int(win))
    if win % 2 == 0:
        win += 1
    if win <= 1 or w.size == 0:
        return w
    pad = win // 2
    w2 = np.pad(w, ((0, 0), (pad, pad)), mode="edge")
    c = np.cumsum(w2, axis=1)
    c = np.concatenate([np.zeros((w.shape[0], 1)), c], axis=1)
    return (c[:, win:] - c[:, :-win]) / win


def _detrend_rows(w: np.ndarray) -> np.ndarray:
    """Quita la recta que une las medianas de los extremos de la ventana."""
    if w.size == 0:
        return w
    k = max(2, w.shape[1] // 5)
    left = np.median(w[:, :k], axis=1, keepdims=True)
    right = np.median(w[:, -k:], axis=1, keepdims=True)
    ramp = np.linspace(0.0, 1.0, w.shape[1])[None, :]
    return w - (left + (right - left) * ramp)


def _robust_sd(v: np.ndarray) -> np.ndarray | float:
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return 0.0
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def _hf_noise(v: np.ndarray) -> float:
    """Ruido de alta frecuencia de una forma de onda (por su 2ª diferencia).

    Para ruido blanco de desvío ``s`` la 2ª diferencia tiene desvío ``s*sqrt(6)``;
    sirve para informar cuánto de la plantilla es ruido y no onda.
    """
    v = np.asarray(v, dtype=np.float64)
    if v.size < 5:
        return 0.0
    d2 = np.diff(v, 2)
    return float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6.0))


def prepare_windows(w: np.ndarray, fs: float) -> np.ndarray:
    """Suaviza y quita tendencia: deja la P como desviación respecto de 0."""
    return _detrend_rows(_smooth_rows(np.asarray(w, dtype=np.float64),
                                      int(round(SMOOTH_MS / 1000.0 * fs))))


# ---------------------------------------------------------------------------
# Correlación con corrimiento
# ---------------------------------------------------------------------------
def ncc_shift(wide: np.ndarray, template: np.ndarray,
              max_shift: int) -> tuple[np.ndarray, np.ndarray]:
    """Correlación normalizada de cada fila con la plantilla, en ±max_shift.

    ``wide`` tiene ``largo_plantilla + 2 * max_shift`` columnas. El corrimiento
    ``s`` devuelto significa que la plantilla encaja en ``wide[s + S : ...]``,
    es decir que la P del latido está ``s`` muestras más a la derecha de lo
    esperado.
    """
    m = template.size
    wide = np.asarray(wide, dtype=np.float64)
    if wide.shape[0] == 0:
        return np.zeros(0), np.zeros(0, dtype=int)
    offs = np.arange(2 * max_shift + 1)[:, None] + np.arange(m)[None, :]
    seg = wide[:, offs]                                   # (N, 2S+1, m)
    seg = seg - seg.mean(axis=2, keepdims=True)
    seg = seg / np.maximum(np.linalg.norm(seg, axis=2, keepdims=True), 1e-9)
    t = np.asarray(template, dtype=np.float64)
    t = t - t.mean()
    norm = float(np.linalg.norm(t))
    if norm < 1e-9:
        return np.zeros(wide.shape[0]), np.zeros(wide.shape[0], dtype=int)
    corr = seg @ (t / norm)                               # (N, 2S+1)
    best = np.argmax(corr, axis=1)
    rows = np.arange(wide.shape[0])
    return np.clip(corr[rows, best], -1.0, 1.0), best - max_shift


def align_windows(wide: np.ndarray, shifts: np.ndarray, max_shift: int,
                  m: int) -> np.ndarray:
    """Recorta de ``wide`` la porción alineada con la plantilla."""
    offs = max_shift + np.asarray(shifts, dtype=int)
    idx = offs[:, None] + np.arange(m)[None, :]
    return np.take_along_axis(wide, idx, axis=1)


# ---------------------------------------------------------------------------
# Resta de la onda T previa
# ---------------------------------------------------------------------------
def t_template(x: np.ndarray, fs: float, peaks: np.ndarray,
               min_beats: int = MIN_TEMPLATE_BEATS,
               t_end: float = T_OFF) -> tuple[np.ndarray, float] | None:
    """Mediana de la onda T entre R+0.10 s y R+``t_end``.

    ``t_end`` se recorta para no tragarse la P del latido siguiente en ritmos
    rápidos; la cola de la plantilla se enmudece con un coseno para que la
    resta no deje un escalón.
    """
    t_end = float(np.clip(t_end, T_ON + 0.10, T_OFF))
    if len(peaks) < min_beats:
        return None
    w, ok = beat_windows(x, fs, peaks, -T_ON, t_end)
    w = w[ok]
    if w.shape[0] < min_beats:
        return None
    tmpl = np.median(_detrend_rows(w), axis=0).astype(np.float64)
    taper = min(tmpl.size, int(round(0.06 * fs)))
    if taper > 1:
        tmpl[-taper:] *= 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, taper)))
    return tmpl, t_end


def subtract_previous_t(seg_wide: np.ndarray, fs: float, peaks: np.ndarray,
                        ids: np.ndarray, t_tmpl: tuple[np.ndarray, float] | None,
                        rr_ref: float) -> np.ndarray:
    """Resta la T del latido anterior en las ventanas de ``ids``.

    ``seg_wide`` son las ventanas (una fila por latido de ``ids``). La T se
    escala en tiempo con sqrt(RR/RR medio) y se interpola sobre la grilla de la
    ventana; fuera del rango de la T el aporte es cero.
    """
    out = np.array(seg_wide, dtype=np.float64, copy=True)
    if t_tmpl is None or out.shape[0] == 0:
        return out
    tmpl, t_end = t_tmpl
    grid = T_ON + np.arange(tmpl.size) * (t_end - T_ON) / max(1, tmpl.size - 1)
    pre_wide, _, n_wide, _, _ = _geom(fs)
    lag = np.arange(n_wide) / fs
    for r, i in enumerate(ids):
        if i == 0:
            continue
        prev = int(peaks[i - 1])
        rr = (peaks[i] - prev) / fs
        k = float(np.clip(np.sqrt(max(rr, 1e-3) / max(rr_ref, 1e-3)),
                          0.80, 1.25))
        start = peaks[i] - int(round(pre_wide * fs))     # inicio de la ventana
        tau = (start - prev) / fs + lag                  # tiempo desde el R previo
        out[r] -= np.interp(tau / k, grid, tmpl, left=0.0, right=0.0)
    return out


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
def _template_shape(tmpl: np.ndarray, fs: float) -> tuple[int, int]:
    """(pico, inicio relativo) de la P dentro de la plantilla."""
    pico = int(np.argmax(np.abs(tmpl)))
    amp = float(abs(tmpl[pico]))
    max_lead = min(pico, max(1, int(round(ONSET_MAX_MS / 1000.0 * fs))))
    lead = max_lead
    if amp > 0:
        thr = ONSET_FRAC * amp
        for i in range(pico - 1, pico - max_lead - 1, -1):
            if abs(tmpl[i]) < thr:
                lead = pico - i
                break
    return pico, max(1, lead)


def _halves_consistency(work: np.ndarray, shift: int, m: int,
                        min_beats: int) -> float | None:
    """Correlación entre las plantillas de las dos mitades (o ``None``).

    Se calcula sólo con los latidos que arman la plantilla (nunca con los que
    se evalúan) y es la defensa contra el ruido: lo que no se repite en las dos
    mitades no es una P. Con menos de ``min_beats`` latidos por mitad no hay
    con qué comparar y se devuelve ``None``.
    """
    n = work.shape[0]
    if n < 2 * min_beats:
        return None
    a = np.median(align_windows(work[0::2], np.zeros(n - n // 2, int), shift, m),
                  axis=0)
    b = np.median(align_windows(work[1::2], np.zeros(n // 2, int), shift, m),
                  axis=0)
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den < 1e-12:
        return None
    return float(np.dot(a, b) / den)


def _empty(n: int) -> dict:
    return {
        "present": np.zeros(n, dtype=bool),
        "evaluable": np.zeros(n, dtype=bool),
        "corr": np.zeros(n, dtype=np.float64),
        "shift_ms": np.zeros(n, dtype=np.float64),
        "amp_uv": np.zeros(n, dtype=np.float64),
        "noise_uv": np.zeros(n, dtype=np.float64),
        "template_amp_uv": np.zeros(n, dtype=np.float64),
        "template_noise_uv": np.zeros(n, dtype=np.float64),
        "template_p": np.zeros(n, dtype=bool),
        "halves_corr": np.full(n, np.nan, dtype=np.float64),
        "peak": np.full(n, -1, dtype=int),
        "onset": np.full(n, -1, dtype=int),
        "status": np.full(n, STATUS_NO_TEMPLATE, dtype=object),
        "template_peak": np.full(n, -1, dtype=int),
    }


def detect_p_waves(x: np.ndarray, fs: float, peaks: np.ndarray, *,
                   segment_s: float = SEGMENT_S, remove_t: bool = True,
                   halves: bool = True, max_hr: float = MAX_HR,
                   min_template_beats: int = MIN_TEMPLATE_BEATS) -> dict:
    """Detecta la P de cada latido con plantilla por minuto y control de mitades.

    Devuelve un dict de arreglos del largo de ``peaks``:

    ``present``
        la P se consideró presente (correlación y amplitud comparables a la de
        la plantilla, y plantilla con P).
    ``evaluable``
        el latido entró en la mitad de evaluación de su minuto y tiene plantilla
        (``status == "ok"``); sólo éstos sirven para medir Se/PPV.
    ``corr`` / ``shift_ms``
        correlación con la plantilla y corrimiento encontrado (±40 ms).
    ``amp_uv`` / ``noise_uv``
        amplitud pico-pico de la P del latido y ruido local (MAD) en µV.
    ``template_amp_uv`` / ``template_noise_uv`` / ``halves_corr``
        amplitud de la P en la plantilla del minuto, su ruido de alta frecuencia
        y la correlación entre las plantillas de las dos mitades (NaN si no
        alcanzan los latidos para compararlas).
    ``template_p``
        la plantilla del minuto pasó el control de P (si no, ``present`` es
        False en todos sus latidos aunque la correlación sea alta).
    ``peak`` / ``onset``
        muestra absoluta del pico y del inicio de la P (-1 si no hay P).
    ``status``
        ``ok``, ``borde``, ``fc>120``, ``sin_plantilla`` o ``mitad_plantilla``.

    ``halves=False`` arma la plantilla con todos los latidos del minuto y los
    evalúa a todos: sirve sólo para medir cuánto infla el resultado la fuga de
    la plantilla, no para informar métricas.
    """
    x = np.asarray(x, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=int).ravel()
    n = peaks.size
    out = _empty(n)
    if n < 3 or x.size < 16 or fs <= 0:
        return out

    pre_wide, post_wide, n_wide, shift, m = _geom(fs)
    rr_min = 60.0 / max(1.0, max_hr)

    wide, ok = beat_windows(x, fs, peaks, pre_wide, post_wide)
    out["status"][:] = STATUS_OK
    out["status"][~ok] = STATUS_EDGE

    rr_prev = np.full(n, np.nan)
    if n > 1:
        rr_prev[1:] = (peaks[1:] - peaks[:-1]) / fs
    fast = np.isfinite(rr_prev) & (rr_prev < rr_min)
    out["status"][fast & ok] = STATUS_FAST

    seg = np.floor(peaks / max(1.0, segment_s * fs)).astype(int)
    ready = np.flatnonzero(ok & ~fast)
    if ready.size < min_template_beats:
        return out

    for s in np.unique(seg[ready]):
        ids = ready[seg[ready] == s]
        if ids.size < 2 * min_template_beats:
            out["status"][ids] = STATUS_NO_TEMPLATE
            continue
        half_a, half_b = ids[0::2], ids[1::2]
        if halves:
            tmpl_ids, ev_ids = (half_a, half_b) if s % 2 == 0 else (half_b, half_a)
            out["status"][np.setdiff1d(ids, ev_ids)] = STATUS_TEMPLATE_HALF
        else:                       # sin control: mide el sesgo por fuga
            tmpl_ids = ev_ids = ids
        if tmpl_ids.size < min_template_beats:
            out["status"][ev_ids] = STATUS_NO_TEMPLATE
            continue

        rr_ref = float(np.median(rr_prev[ids][np.isfinite(rr_prev[ids])])) \
            if np.isfinite(rr_prev[ids]).any() else 0.8
        t_end = float(np.clip(rr_ref - P_PRE - 0.02, T_ON + 0.10, T_OFF))
        tt = t_template(x, fs, peaks[tmpl_ids], min_template_beats, t_end) \
            if remove_t else None
        work = prepare_windows(subtract_previous_t(wide[ids], fs, peaks, ids,
                                                   tt, rr_ref), fs)
        pos = {int(i): k for k, i in enumerate(ids)}
        rows_t = np.array([pos[int(i)] for i in tmpl_ids], dtype=int)
        rows_e = np.array([pos[int(i)] for i in ev_ids], dtype=int)

        # plantilla: mediana de la mitad par/impar + una refinación alineada
        tmpl = np.median(align_windows(work[rows_t], np.zeros(rows_t.size, int),
                                       shift, m), axis=0)
        corr_t, sh_t = ncc_shift(work[rows_t], tmpl, shift)
        keep = corr_t >= np.percentile(corr_t, 30) if corr_t.size >= 4 else \
            np.ones(corr_t.size, dtype=bool)
        if keep.sum() >= min_template_beats:
            tmpl = np.median(align_windows(work[rows_t][keep], sh_t[keep],
                                           shift, m), axis=0)

        pico_t, lead_t = _template_shape(tmpl, fs)
        w_pico = int(round(0.03 * fs))
        lo = np.clip(pico_t - w_pico, 0, m - 1)
        hi = np.clip(pico_t + w_pico + 1, 1, m)

        # ¿hay una P en la plantilla? amplitud mínima y, sobre todo,
        # consistencia entre las dos mitades: el ruido no se repite en la
        # mitad par y en la impar, una P sí (diseño §2.2 y §2.6)
        tpl_amp = 1000.0 * float(tmpl[lo:hi].max() - tmpl[lo:hi].min())
        tpl_noise = 1000.0 * _hf_noise(tmpl)
        halves_corr = _halves_consistency(work[rows_t], shift, m,
                                          min_template_beats)
        template_p = (tpl_amp >= max(TPL_AMP_MIN_UV, TPL_NOISE_FACTOR * tpl_noise)
                      and (halves_corr is None or halves_corr >= HALVES_CORR_MIN))

        corr, sh = ncc_shift(work[rows_e], tmpl, shift)
        aligned = align_windows(work[rows_e], sh, shift, m)
        amp_uv = 1000.0 * (aligned[:, lo:hi].max(axis=1)
                           - aligned[:, lo:hi].min(axis=1))
        cols = np.arange(m)[None, :]
        away = np.abs(cols - pico_t) > int(round(0.09 * fs))
        masked = np.where(away, aligned, np.nan)
        noise_uv = 1000.0 * np.nanmedian(
            np.abs(masked - np.nanmedian(masked, axis=1, keepdims=True)),
            axis=1) * 1.4826

        # La P de un latido es la de la plantilla: se exige correlación y una
        # amplitud comparable (la plantilla ya pasó el control de ruido, así
        # que no se vuelve a exigir amplitud absoluta contra el ruido del
        # latido: en P de bajo voltaje eso borraba la detección).
        present = (corr >= CORR_MIN) & (amp_uv >= AMP_MIN_UV) & \
                  (amp_uv >= AMP_REL_MIN * tpl_amp)
        present &= template_p

        peak_pos = np.clip(pico_t, 0, m - 1)
        lead_lo = max(1, lead_t - int(round(0.02 * fs)))
        lead_hi = min(peak_pos, lead_t + int(round(0.03 * fs)))
        onset = np.full(ev_ids.size, -1, dtype=int)
        for r in range(ev_ids.size):
            if not present[r]:
                continue
            thr = ONSET_FRAC * abs(aligned[r, peak_pos])
            lead = lead_t
            for i in range(peak_pos - lead_lo, peak_pos - lead_hi - 1, -1):
                if abs(aligned[r, i]) < thr:
                    lead = peak_pos - i
                    break
            onset[r] = peak_pos - lead

        base = peaks[ev_ids] - int(round(pre_wide * fs)) + shift
        out["corr"][ev_ids] = corr
        out["shift_ms"][ev_ids] = sh * 1000.0 / fs
        out["amp_uv"][ev_ids] = amp_uv
        out["noise_uv"][ev_ids] = noise_uv
        out["template_amp_uv"][ev_ids] = tpl_amp
        out["template_noise_uv"][ev_ids] = tpl_noise
        out["template_p"][ev_ids] = template_p
        out["halves_corr"][ev_ids] = np.nan if halves_corr is None else halves_corr
        out["present"][ev_ids] = present
        out["peak"][ev_ids] = np.where(present, base + sh + pico_t, -1)
        out["onset"][ev_ids] = np.where(present, base + sh + onset, -1)
        out["template_peak"][ev_ids] = base + pico_t
        out["status"][ev_ids] = STATUS_OK

    out["evaluable"] = out["status"] == STATUS_OK
    return out


def p_summary(res: dict) -> dict:
    """Resumen agregado de una corrida (para informes y pruebas)."""
    ev = np.asarray(res["evaluable"], dtype=bool)
    pres = np.asarray(res["present"], dtype=bool)
    tpl = np.asarray(res["template_p"], dtype=bool)
    n_ev = int(ev.sum())
    return {
        "beats": int(len(ev)),
        "evaluables": n_ev,
        "frac_evaluable": float(n_ev / max(1, len(ev))),
        "present": int((pres & ev).sum()),
        "frac_present": float((pres & ev).sum() / max(1, n_ev)),
        "frac_template_p": float((tpl & ev).sum() / max(1, n_ev)),
        "status": {k: int(np.sum(np.asarray(res["status"]) == k))
                   for k in STATUS_NOT_EVALUABLE + (STATUS_OK,)},
    }
