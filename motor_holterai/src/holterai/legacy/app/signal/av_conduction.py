"""Conducción auriculoventricular (T24): intervalo PR y bloqueos AV.

Qué mide
--------
* **PR por latido**: inicio de la P (detector de T10, ``signal/pwave.py``, si
  está; si no, el detector por plantilla de este módulo) al inicio del QRS;
  mediana por minuto y tendencia del registro (ms/h).
* **Bloqueos AV**: primer grado (PR > 200 ms sostenido >= 1 min), Mobitz I
  (PR que se alarga antes de una P bloqueada), Mobitz II (P bloqueada con PR
  previo estable), 2:1, avanzado (>= 2 P bloqueadas seguidas) y completo
  (disociación AV: P regulares que no conducen, escape ventricular regular).

Cómo se mide el inicio de la P
------------------------------
El pico de la P se localiza por correlación con la plantilla del minuto y el
**inicio** se toma de la plantilla: donde la pendiente de la P cae al 20 % de su
máxima (criterio de la tangente). La plantilla es la mediana de decenas de
latidos alineados, así que su pendiente está bien condicionada; medir el inicio
latido a latido sobre 180 Hz con ruido agrega ±20 ms que entran enteros en el
PR. En LUDB (185 registros con delineación manual) esto deja el error del inicio
de P en 5,6 ms de mediana (71 % dentro de 10 ms) contra 11 ms del umbral sobre
la amplitud. El inicio se traslada a cada latido con el pico correlacionado y el
desfase pico→inicio de la plantilla (``PInfo.peak_to_onset``), igual para todas
las P del tramo.

Reglas de evidencia
-------------------
Una **P bloqueada** es una P que no conduce ningún QRS. Para aceptarla se exige:

1. P detectada en **>= 2 canales** (si el registro trae más de uno) y **fuera
   de un tramo ilegible** (máscara de ``signal/artifact.py``, T21),
2. que llegue **a tiempo**: el intervalo con la P anterior dentro de la
   tolerancia del PP local (si no, es ruido o una P ectópica),
3. que haya **ritmo sinusal establecido**: >= 3 P conducidas antes, con PR
   estable y PP regular (se admite que falte una P: puede estar escondida en la
   T o en el QRS, así que cada intervalo debe ser un múltiplo del PP),
4. que ningún QRS la siga en la ventana de conducción (PR de 60-500 ms),
5. que deje **hueco ventricular**: entre el QRS de antes y el de después tiene
   que haber un latido de menos (2 PP en el 2:1 y en el Mobitz; 3-4 PP en el
   avanzado). Sin eso, una P que sí condujo (con el PR mal medido) o una pausa
   larga se leerían como bloqueo: es el falso positivo que importa.

En un registro de una sola derivación no hay segundo canal que confirme la P:
ahí el barrido exige más correlación (``P_SCAN_CORR_ONE_LEAD``) y el bloqueo
queda igualmente "a revisión".

El **primer grado** avisa con 200 ms + ``PR_FIRST_DEGREE_TOL_MS`` (20 ms, la DE
del error de medida) sobre la mediana móvil de varios minutos: el PR de un
registro sano se mueve ±30 ms por minuto y con el umbral pelado cruzaría los
200 ms sin ser un bloqueo.

Todo evento sale marcado "a revisión" y con la ventana de la tira de 10 s que lo
muestra (``strip_start_s`` / ``strip_end_s``).

Entradas
--------
Señal **ya filtrada** (0.5-40 Hz, ``filters.preprocess(x, fs, "display")``), en
las mismas unidades que la entrada (el informe de amplitudes supone mV), la
frecuencia de muestreo y las muestras del pico R. Opcionalmente los otros
canales filtrados, los tramos ilegibles y ``beat_ok`` (latidos donde el ritmo
permite hablar de conducción: sin FA, sin rachas ventriculares).

Módulo numpy puro: no filtra ni remuestrea.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- geometría de las ventanas (segundos) ---------------------------------
QRS_PRE = 0.20          # búsqueda del inicio de QRS: 200 ms antes del R
QRS_POST = 0.10
QRS_MAX_LEAD = 0.13     # el inicio de QRS no puede estar más atrás
QRS_D_FRAC = 0.08       # umbral de la derivada (fracción del máximo local)
QRS_A_FRAC = 0.03       # umbral de amplitud (fracción de la del QRS)
QRS_A_MIN = 0.005       # ... y su piso absoluto (mV)
QRS_SMOOTH_S = 0.012    # suavizado de la derivada antes del umbral
QRS_A_MAX_BACK = 0.04   # la regla de amplitud no adelanta más de 40 ms
P_PRE = 0.34            # la ventana de P empieza 340 ms antes del R ...
P_END = 0.06            # ... y termina 60 ms antes del R (QRS excluido)
P_SHIFT_S = 0.04        # búsqueda por correlación ±40 ms
P_SEGMENT_S = 60.0      # plantilla de P por minuto
P_MIN_BEATS = 3
P_ONSET_MAX_S = 0.12    # duración máxima aceptada de la P
P_ONSET_FRAC = 0.10     # inicio de la P al 10 % de su amplitud (respaldo)
P_ONSET_SLOPE_FRAC = 0.20   # inicio de la P al 20 % de su pendiente máxima
P_ONSET_SMOOTH_S = 0.010    # suavizado de la pendiente
P_BASE_PRE_S = 0.10     # línea de base de la P: 100 ms antes de su pico ...
P_BASE_POST_S = 0.04    # ... hasta 40 ms antes (segmento TP-PQ)
P_PEAK_SNAP_S = 0.020   # el pico de la P no se aleja más de esto de la mediana
P_PEAK_SNAP_MARGIN = 0.06   # ... salvo que lejos correlacione claramente mejor
P_ONSET_MODE = "template"   # "template" (pico + plantilla), "amp" (por latido) o "mix"
SCAN_BLOCK_S = 300.0    # el barrido de P se hace por bloques (memoria)

# --- umbrales del PR ------------------------------------------------------
PR_MIN_MS = 60.0        # fuera de este rango no es un PR: es otra cosa
PR_MAX_MS = 500.0
PR_FIRST_DEGREE_MS = 200.0   # criterio clínico
PR_FIRST_DEGREE_TOL_MS = 20.0   # DE del error de PR: se avisa a partir de 220
PR_FIRST_DEGREE_SMOOTH = 2      # mediana móvil de minutos (±2) para "sostenido"
PR_FIRST_DEGREE_MIN_S = 60.0
PR_MIN_BEATS_PER_MINUTE = 5
QRS_PRE_GATE_MS = 20.0     # el R→inicio de QRS es constante para una morfología
P_OFF_GATE_MS = 30.0       # ... y la duración pico→inicio de la P también
P_PR_CORR_MIN = 0.70       # P usada para PR: correlación con la plantilla
PR_GATE_SEGMENT_S = 60.0   # ventana de la mediana local del desfase del QRS
PR_CONSIST_MS = 40.0       # el PR de un latido no se separa esto de su minuto

# --- umbrales de la P -----------------------------------------------------
P_CORR_MIN = 0.60       # correlación con la plantilla
P_AMP_MIN_UV = 12.0     # amplitud pico-pico mínima de la P
P_AMP_REL = 0.50        # ... y fracción de la amplitud de la plantilla
P_TPL_MIN_UV = 8.0      # P mínima en la plantilla
P_HALVES_MIN = 0.70     # consistencia entre las dos mitades de la plantilla
P_SCAN_CORR_MIN = 0.60  # barrido: correlación mínima
P_SCAN_CORR_ONE_LEAD = 0.72   # con un solo canal se exige más
P_SCAN_AMP_REL = 0.60
P_BLOCK_HI_AMP = 3.50   # una P no puede ser 3,5 veces la plantilla (sería T)
LEAD_BUMP_MIN_UV = 15.0
LEAD_BUMP_NOISE = 3.0

# --- reglas de los bloqueos ----------------------------------------------
BLOCK_AFTER_QRS_S = 0.42     # nada de P dentro del QRS ni de la onda T
PP_TOL_FRAC = 0.18           # tolerancia de "P a tiempo" sobre el PP local
PP_TOL_MS = 60.0
PP_REGULAR_FRAC = 0.16       # PP regular: MAD/mediana máximo
MIN_CONDUCTED_BEFORE = 3
MOBITZ1_GROWTH_MS = 30.0     # alargamiento mínimo del PR antes de la P caída
MOBITZ2_STABLE_MS = 25.0     # PR previo estable (máx-mín)
TWO_TO_ONE_MIN_BLOCKED = 3
COMPLETE_MIN_S = 8.0
COMPLETE_WINDOW_S = 12.0
COMPLETE_BLOCKED_FRAC = 0.45
COMPLETE_PR_SD_MS = 40.0
COMPLETE_RR_REGULAR_FRAC = 0.15
COMPLETE_MIN_P_RATIO = 1.40
STRIP_S = 10.0

KIND_1 = "av_block_1"
KIND_M1 = "av_block_mobitz1"
KIND_M2 = "av_block_mobitz2"
KIND_21 = "av_block_2to1"
KIND_ADV = "av_block_advanced"
KIND_COMPLETE = "av_block_complete"
SECOND_DEGREE = (KIND_M1, KIND_M2, KIND_21, KIND_ADV, KIND_COMPLETE)

KIND_NAMES = {
    KIND_1: "Bloqueo AV de primer grado",
    KIND_M1: "Bloqueo AV de segundo grado Mobitz I",
    KIND_M2: "Bloqueo AV de segundo grado Mobitz II",
    KIND_21: "Bloqueo AV 2:1",
    KIND_ADV: "Bloqueo AV avanzado",
    KIND_COMPLETE: "Bloqueo AV completo",
}


# ---------------------------------------------------------------------------
# Utilidades numéricas
# ---------------------------------------------------------------------------
def _robust_sd(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    m = float(np.median(v))
    return float(np.median(np.abs(v - m)) * 1.4826)


def _smooth_rows(w: np.ndarray, win: int) -> np.ndarray:
    """Media móvil por fila (ventana impar, borde replicado)."""
    win = max(1, int(win))
    if win % 2 == 0:
        win += 2
    if win <= 1 or w.size == 0:
        return w
    pad = win // 2
    w2 = np.pad(w, ((0, 0), (pad, pad)), mode="edge")
    c = np.cumsum(w2, axis=1)
    c = np.concatenate([np.zeros((w.shape[0], 1)), c], axis=1)
    return (c[:, win:] - c[:, :-win]) / win


def _detrend_rows(w: np.ndarray) -> np.ndarray:
    """Quita la recta que une las medianas de los extremos de cada fila."""
    if w.size == 0:
        return w
    k = max(2, w.shape[1] // 6)
    left = np.median(w[:, :k], axis=1, keepdims=True)
    right = np.median(w[:, -k:], axis=1, keepdims=True)
    ramp = np.linspace(0.0, 1.0, w.shape[1])[None, :]
    return w - (left + (right - left) * ramp)


def _run_start(above: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Primer índice del tramo contiguo con ``above`` que contiene ``ref``.

    ``above`` es (n, w) booleano y ``ref`` (n,) la columna de referencia; se
    busca hacia la izquierda. Devuelve -1 si no hay ningún hueco a la izquierda.
    """
    n, w = above.shape
    cols = np.arange(w)[None, :]
    last_below = np.maximum.accumulate(np.where(~above, cols, -1), axis=1)
    rows = np.arange(n)
    ref = np.clip(np.asarray(ref, dtype=int), 0, w - 1)
    prev = np.where(ref > 0, last_below[rows, np.maximum(ref - 1, 0)], -1)
    return prev + 1


def _windows(x: np.ndarray, peaks: np.ndarray, pre: int, post: int):
    """Ventanas (n, pre+post) y filas completas dentro de la señal."""
    peaks = np.asarray(peaks, dtype=int)
    w = int(pre) + int(post)
    if peaks.size == 0 or w < 3:
        return np.zeros((0, max(3, w)), dtype=np.float64), np.zeros(0, dtype=bool)
    start = peaks - int(pre)
    idx = start[:, None] + np.arange(w)[None, :]
    inside = (idx >= 0) & (idx < x.size)
    return x[np.clip(idx, 0, max(0, x.size - 1))], inside.all(axis=1)


# ---------------------------------------------------------------------------
# Inicio del QRS
# ---------------------------------------------------------------------------
def qrs_onsets(x: np.ndarray, fs: float, peaks: np.ndarray, *,
               pre: float = QRS_PRE, post: float = QRS_POST,
               max_lead: float = QRS_MAX_LEAD) -> np.ndarray:
    """Muestra del **inicio del QRS** de cada latido (-1 si no se pudo).

    Dos reglas y se toma la más temprana (la que no llega tarde, que es el
    error caro para el PR):

    * derivada: primer punto, hacia la izquierda del pico de la derivada, donde
      la envolvente cae al 12 % del máximo local,
    * amplitud: primer punto donde la señal vuelve a la línea de base del
      segmento PQ (6 % de la amplitud del QRS).
    """
    x = np.asarray(x, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=int)
    out = np.full(peaks.size, -1, dtype=int)
    a = int(round(pre * fs))
    b = int(round(post * fs))
    if peaks.size == 0 or x.size < a + b + 3:
        return out
    seg, ok = _windows(x, peaks, a, b)
    w = seg.shape[1]
    seg = _detrend_rows(seg)
    d = np.abs(np.diff(seg, axis=1, prepend=seg[:, :1]))
    d = _smooth_rows(d, max(3, int(round(QRS_SMOOTH_S * fs))))
    # zona del pico del QRS: [-100, +60] ms respecto del fiducial
    lo = max(1, a - int(round(0.10 * fs)))
    hi = min(w - 1, a + int(round(0.06 * fs)) + 1)
    core = d[:, lo:hi]
    jmax = lo + np.argmax(core, axis=1)
    rows = np.arange(seg.shape[0])

    above_d = d > QRS_D_FRAC * d[rows, jmax][:, None]
    onset_d = _run_start(above_d, jmax)

    # Regla de amplitud: refina el inicio hacia atrás, con la línea de base
    # tomada justo en el codo del QRS (no en una ventana fija, que en un PR
    # corto cae dentro de la P) y con un límite de 40 ms para no caminar hasta
    # la P del latido.
    col = np.clip(onset_d - int(round(0.02 * fs)), 1, w - 2)
    idx3 = (col[:, None] + np.arange(-1, 2)[None, :]).clip(0, w - 1)
    base = np.median(np.take_along_axis(seg, idx3, axis=1), axis=1)
    dev = np.abs(seg - base[:, None])
    amp = np.max(dev[:, lo:hi], axis=1)
    jpeak = lo + np.argmax(dev[:, lo:hi], axis=1)
    above_a = dev > np.maximum(QRS_A_FRAC * amp, QRS_A_MIN)[:, None]
    onset_a = _run_start(above_a, jpeak)

    onset = np.minimum(onset_d, onset_a)
    onset = np.maximum(onset, onset_d - int(round(QRS_A_MAX_BACK * fs)))
    floor = a - int(round(max_lead * fs))
    onset = np.maximum(onset, floor)
    onset = np.minimum(onset, a)                 # nunca después del fiducial
    onset = np.where(ok & (onset >= 0), onset, -1)
    out[:] = (peaks - a + onset).astype(int)
    out[~ok] = -1
    return out


# ---------------------------------------------------------------------------
# Onda P: por latido (T10) y barrido del registro completo
# ---------------------------------------------------------------------------
@dataclass
class PInfo:
    """P de cada latido (arreglos alineados con ``peaks``)."""

    onset: np.ndarray
    peak: np.ndarray
    present: np.ndarray
    corr: np.ndarray
    amp_uv: np.ndarray
    evaluable: np.ndarray
    template_amp_uv: float = 0.0
    peak_to_onset: int = 0          # muestras del pico al inicio de la P (plantilla)

    def __len__(self) -> int:
        return int(self.present.size)


def _empty_pinfo(n: int) -> PInfo:
    return PInfo(
        onset=np.full(n, -1, dtype=int),
        peak=np.full(n, -1, dtype=int),
        present=np.zeros(n, dtype=bool),
        corr=np.zeros(n, dtype=np.float64),
        amp_uv=np.zeros(n, dtype=np.float64),
        evaluable=np.zeros(n, dtype=bool),
    )


def _p_geometry(fs: float) -> tuple[int, int, int, int]:
    """(muestras antes del R, largo de la ventana ancha, corrimiento, útil).

    La ventana útil va de R−``P_PRE`` a R−``P_END`` y la ancha se extiende
    ±``P_SHIFT_S`` para poder correlacionar con corrimiento.
    """
    shift = max(1, int(round(P_SHIFT_S * fs)))
    m = max(9, int(round((P_PRE - P_END) * fs)))
    n_w = m + 2 * shift
    pre_w = int(round(P_PRE * fs)) + shift
    return pre_w, n_w, shift, m


def _detect_p_internal(x: np.ndarray, fs: float, peaks: np.ndarray,
                       segment_s: float = P_SEGMENT_S) -> PInfo:
    """Detector de P por plantilla local (respaldo si T10 no está integrado).

    Misma idea que ``signal/pwave.py``: plantilla por minuto, control de
    mitades par/impar, correlación ±40 ms, inicio por la tangente de la
    plantilla (``_template_onset``) y latidos con RR < 500 ms no evaluables (la
    P cae sobre la T previa).
    """
    peaks = np.asarray(peaks, dtype=int)
    n = peaks.size
    info = _empty_pinfo(n)
    if n == 0:
        return info
    pre, n_w, shift, m = _p_geometry(fs)
    wide, ok = _windows(x, peaks, pre, n_w - pre)
    rr_prev = np.zeros(n, dtype=np.float64)
    if n > 1:
        rr_prev[1:] = np.diff(peaks) / fs
    fast = (rr_prev > 0) & (rr_prev < 0.5)
    ready = np.flatnonzero(ok & ~fast)
    info.evaluable[ready] = True
    if ready.size == 0:
        return info

    sh = np.arange(-shift, shift + 1)
    seg_id = np.floor(peaks / max(1.0, segment_s * fs)).astype(int)
    tmpl_amps: list[float] = []
    offsets: list[int] = []
    for s in np.unique(seg_id[ready]):
        ids = ready[seg_id[ready] == s]
        if ids.size < P_MIN_BEATS:
            info.evaluable[ids] = False
            continue
        work = _detrend_rows(wide[ids])
        aligned = np.stack([work[:, shift + k: shift + k + m] for k in sh])
        tmpl = np.median(aligned[shift], axis=0)
        pico = int(np.argmax(np.abs(tmpl)))
        near = slice(max(0, pico - int(round(0.03 * fs))),
                     min(m, pico + int(round(0.03 * fs)) + 1))
        amp_tpl = 1000.0 * float(np.ptp(tmpl[near]))
        halves = None
        if ids.size >= 2 * P_MIN_BEATS:
            a_h = np.median(aligned[shift][0::2], axis=0)
            b_h = np.median(aligned[shift][1::2], axis=0)
            na, nb = float(np.linalg.norm(a_h - a_h.mean())), float(np.linalg.norm(b_h - b_h.mean()))
            if na > 1e-12 and nb > 1e-12:
                halves = float(np.dot(a_h - a_h.mean(), b_h - b_h.mean()) / (na * nb))
        noise = 1000.0 * float(np.median(np.abs(np.diff(tmpl, n=2)))) * 0.5
        tpl_ok = amp_tpl >= max(P_TPL_MIN_UV, 3.0 * noise) and \
            (halves is None or halves >= P_HALVES_MIN)
        if not tpl_ok:
            info.evaluable[ids] = False
            continue
        tmpl_amps.append(amp_tpl)
        onset_tpl = _template_onset(tmpl, pico, fs)
        offsets.append(int(pico) - int(onset_tpl))

        # correlación con corrimiento para cada latido del minuto
        tm = tmpl - tmpl.mean()
        tn = float(np.linalg.norm(tm))
        best_corr = np.full(ids.size, -2.0)
        best_k = np.zeros(ids.size, dtype=int)
        for k in sh:
            sub = aligned[k + shift]
            sub_c = sub - sub.mean(axis=1, keepdims=True)
            den = np.linalg.norm(sub_c, axis=1) * tn
            c = np.divide(sub_c @ tm, np.maximum(den, 1e-12))
            better = c > best_corr
            best_corr = np.where(better, c, best_corr)
            best_k = np.where(better, k, best_k)

        # El pico de la P se busca por correlación en ±40 ms y con señal de
        # 180-360 Hz una P pequeña puede hacer saltar el máximo una decena de
        # muestras: eso entra entero en el PR de ese latido. El PR de un paciente
        # cambia despacio (salvo el Wenckebach, que crece por ciclo), así que el
        # pico se ata a la mediana del minuto y solo se acepta la solución lejana
        # si la correlación ahí es claramente mejor.
        if P_PEAK_SNAP_S > 0 and ids.size >= P_MIN_BEATS:
            lim = int(round(P_PEAK_SNAP_S * fs))
            pk0 = pico + best_k
            med_k = int(round(float(np.median(pk0)))) - pico
            far = np.flatnonzero(np.abs(pk0 - (pico + med_k)) > lim)
            for r in far:
                lo_k = max(-shift, med_k - lim)
                hi_k = min(shift, med_k + lim)
                best_c, best_kk = -2.0, best_k[r]
                for k in range(lo_k, hi_k + 1):
                    sub = aligned[k + shift][r]
                    sub_c = sub - sub.mean()
                    den = float(np.linalg.norm(sub_c)) * tn
                    c = float(sub_c @ tm) / max(den, 1e-12)
                    if c > best_c:
                        best_c, best_kk = c, k
                if best_c >= best_corr[r] - P_PEAK_SNAP_MARGIN:
                    best_corr[r] = max(best_c, best_corr[r])
                    best_k[r] = best_kk
        pk = pico + best_k
        rows = np.arange(ids.size)
        base = np.median(aligned[shift][:, : max(2, int(round(0.03 * fs)))], axis=1)
        v = np.abs(aligned[shift][rows, np.clip(pk, 0, m - 1)] - base)
        near_m = slice(max(0, pico - int(round(0.03 * fs))),
                       min(m, pico + int(round(0.03 * fs)) + 1))
        amp_uv = 1000.0 * np.ptp(aligned[shift][:, near_m], axis=1)
        present = (best_corr >= P_CORR_MIN) & \
                  (amp_uv >= np.maximum(P_AMP_MIN_UV, P_AMP_REL * amp_tpl))
        if P_ONSET_MODE == "template":
            onset = _p_onset_from_template(aligned[shift], pico, onset_tpl,
                                           np.clip(pk, 0, m - 1))
        elif P_ONSET_MODE == "mix":
            o_t = _p_onset_from_template(aligned[shift], pico, onset_tpl,
                                         np.clip(pk, 0, m - 1))
            o_a = _p_onset(aligned[shift], base, np.clip(pk, 0, m - 1), v, fs)
            onset = ((o_t + o_a) // 2).astype(int)
        else:
            onset = _p_onset(aligned[shift], base, np.clip(pk, 0, m - 1), v, fs)
        start = peaks[ids] - pre + shift
        info.corr[ids] = np.clip(best_corr, -1.0, 1.0)
        info.amp_uv[ids] = amp_uv
        info.present[ids] = present
        info.peak[ids] = np.where(present, start + pk, -1)
        info.onset[ids] = np.where(present, start + onset, -1)
    info.template_amp_uv = float(np.median(tmpl_amps)) if tmpl_amps else 0.0
    info.peak_to_onset = int(round(float(np.median(offsets)))) if offsets else 0
    return info


def smooth_qrs_onsets(peaks: np.ndarray, qrs_on: np.ndarray, fs: float,
                      segment_s: float = 60.0,
                      min_beats: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Inicio de QRS con el desfase R→inicio de la **mediana local**.

    El desfase entre el pico R y el inicio del QRS es una propiedad de la
    morfología, no del latido: cambia poco dentro de un minuto. Medirlo en cada
    latido sobre señal de 180 Hz agrega ±15 ms de ruido que después aparece
    entero en el PR; tomar la mediana del minuto deja el PR limpio y conserva lo
    que sí varía latido a latido (la posición de la P). Devuelve
    (inicios suavizados, mediana local del desfase en muestras).
    """
    peaks = np.asarray(peaks, dtype=int)
    qrs_on = np.asarray(qrs_on, dtype=int)
    pre = (peaks - qrs_on).astype(np.float64)
    valid = (qrs_on >= 0) & (pre >= 0)
    out = np.full(peaks.size, -1, dtype=int)
    if not valid.any():
        return out, np.zeros(0)
    if not peaks.size:
        return out, np.zeros(0)
    gid = np.floor(peaks / max(1.0, segment_s * fs)).astype(int)
    glob = float(np.median(pre[valid]))
    med = np.full(peaks.size, glob)
    for g in np.unique(gid):
        k = valid & (gid == g)
        if k.sum() >= min_beats:
            med[gid == g] = float(np.median(pre[k]))
    out[valid] = np.round(peaks[valid] - med[valid]).astype(int)
    return out, med


def _p_onset(aligned: np.ndarray, base: np.ndarray, pk: np.ndarray,
             v: np.ndarray, fs: float) -> np.ndarray:
    """Inicio de la P por el criterio del 10 % de su amplitud, desde su pico."""
    above = np.abs(aligned - base[:, None]) > (P_ONSET_FRAC * v)[:, None]
    onset = _run_start(above, pk)
    onset = np.minimum(onset, pk)
    return np.maximum(onset, np.clip(pk - int(round(P_ONSET_MAX_S * fs)), 0, None))


def _p_onset_from_template(aligned: np.ndarray, pico: int, onset_tpl: int,
                           pk: np.ndarray) -> np.ndarray:
    """Inicio de la P de cada latido trasladando el de la plantilla.

    El pico de cada P se localiza por correlación (bien condicionado) y la
    distancia pico→inicio se toma de la plantilla del minuto, que es la mediana
    de decenas de latidos: cambia poco de un latido a otro, pero medirla latido
    a latido sobre señal de 180 Hz con ruido mete ±20 ms de error.
    """
    onset = np.asarray(pk, dtype=int) - (int(pico) - int(onset_tpl))
    return np.clip(onset, 0, aligned.shape[1] - 1)


def _amplitude_onset(tmpl: np.ndarray, pico: int, fs: float,
                     frac: float = P_ONSET_FRAC) -> int:
    """Inicio por umbral de amplitud (respaldo si la P no tiene pendiente clara)."""
    base = float(np.median(tmpl[: max(2, int(round(0.03 * fs)))]))
    v = abs(float(tmpl[pico]) - base)
    if v <= 0:
        return int(pico)
    above = np.abs(tmpl - base) > frac * v
    onset = int(_run_start(above[None, :], np.array([pico]))[0])
    return max(onset, int(pico) - int(round(P_ONSET_MAX_S * fs)))


def _template_onset(tmpl: np.ndarray, pico: int, fs: float,
                    frac: float = P_ONSET_SLOPE_FRAC) -> int:
    """Inicio de la P sobre la **plantilla del minuto**: 20 % de la pendiente máxima.

    La plantilla es la mediana de decenas de latidos alineados, así que su
    pendiente está bien condicionada y no depende de cuánto mide la P ni del
    ruido de la línea de base. Es el criterio de la tangente, el que siguen los
    delineadores manuales: en LUDB el error típico del inicio de P baja de 11 a
    6 ms (y la DE entre registros de 18 a 13 ms) respecto del umbral sobre la
    amplitud. Si la deflexión no tiene pendiente aprovechable se vuelve al
    umbral de amplitud.
    """
    t = np.asarray(tmpl, dtype=np.float64)
    if t.size < 5 or not (0 <= int(pico) < t.size):
        return max(0, int(pico))
    pico = int(pico)
    k = max(3, int(round(P_ONSET_SMOOTH_S * fs)))
    sm = np.convolve(t, np.ones(k) / k, mode="same") if 1 < k < t.size else t
    a = max(0, pico - int(round(P_BASE_PRE_S * fs)))
    b = pico - int(round(P_BASE_POST_S * fs))
    base = float(np.median(sm[a:b])) if b > a + 2 else float(sm[max(0, pico - 8)])
    sgn = 1.0 if t[pico] >= base else -1.0
    lo = max(0, pico - int(round(P_ONSET_MAX_S * fs)))
    if pico - lo < 2:
        return lo
    d = sgn * np.diff(sm[lo: pico + 1])
    j = lo + int(np.argmax(d))
    dmax = float(np.max(d))
    if dmax <= 0:
        return _amplitude_onset(t, pico, fs)
    while j > lo and sgn * (sm[j] - sm[j - 1]) > frac * dmax:
        j -= 1
    return int(max(lo, j))


def _p_from_result(res: dict, n: int, fs: float = 0.0) -> PInfo:
    """Adapta el diccionario de ``pwave.detect_p_waves`` (T10) a ``PInfo``."""
    info = _empty_pinfo(n)
    if not res:
        return info
    k = min(n, int(np.size(res.get("present", []))))
    if k == 0:
        return info
    present = np.asarray(res["present"], dtype=bool)[:k]
    onset = np.asarray(res.get("onset", np.full(k, -1)), dtype=int)[:k]
    peak = np.asarray(res.get("peak", np.full(k, -1)), dtype=int)[:k]
    ev = np.asarray(res.get("evaluable", np.ones(k)), dtype=bool)[:k]
    info.present[:k] = present & (onset >= 0)
    info.onset[:k] = np.where(info.present[:k], onset, -1)
    info.peak[:k] = np.where(info.present[:k], peak, -1)
    info.evaluable[:k] = ev
    corr = np.asarray(res.get("corr", np.zeros(k)), dtype=np.float64)[:k]
    info.corr[:k] = corr
    amp = np.asarray(res.get("amp_uv", np.zeros(k)), dtype=np.float64)[:k]
    info.amp_uv[:k] = amp
    tpl = res.get("template_amp_uv", np.zeros(k))
    vals = np.asarray(tpl, dtype=np.float64)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    info.template_amp_uv = float(np.median(vals)) if vals.size else 0.0
    info.peak_to_onset = int(res.get("peak_to_onset", 0) or 0)
    if info.peak_to_onset <= 0:
        # desfase pico→inicio de la plantilla a partir de las P aceptadas: es el
        # que usa el barrido para dar el inicio de una P bloqueada
        d = (info.peak - info.onset)[info.present & (info.onset >= 0)]
        d = d[d > 0]
        if fs > 0:
            d = d[d < int(round(P_ONSET_MAX_S * fs))]
        if d.size >= P_MIN_BEATS:
            info.peak_to_onset = int(round(float(np.median(d))))
    return info


def detect_p(x: np.ndarray, fs: float, peaks: np.ndarray,
             p_result: dict | None = None) -> PInfo:
    """P de cada latido: usa T10 (``signal/pwave.py``) si está disponible."""
    peaks = np.asarray(peaks, dtype=int)
    if p_result is not None:
        return _p_from_result(p_result, peaks.size, fs)
    try:
        from holterai.legacy.app.signal.pwave import detect_p_waves
    except Exception:                       # T10 todavía no integrado
        return _detect_p_internal(x, fs, peaks)
    try:
        res = detect_p_waves(np.asarray(x, dtype=np.float64), float(fs), peaks)
    except Exception:                       # una P rota no debe tumbar el AV
        return _detect_p_internal(x, fs, peaks)
    # T10 todavía no entrega el inicio de P por la tangente (``peak_to_onset``):
    # su ``onset`` por umbral de amplitud adelanta la P ~40 ms y sesga el PR
    # (medido con los sintéticos de T24 sobre esta rama). Se usa el detector
    # interno, que sí mide el inicio en la plantilla, hasta que T10 lo exporte.
    if "peak_to_onset" not in res:
        return _detect_p_internal(x, fs, peaks)
    return _p_from_result(res, peaks.size, fs)


def _p_subtemplate(x: np.ndarray, fs: float, peaks: np.ndarray,
                   p_info: PInfo | None) -> tuple[np.ndarray | None, float]:
    """Patrón de P centrado en su pico (para el barrido) y su amplitud (µV).

    Se arma con las P conducidas que el detector ya aceptó (mediana, robusta) y,
    si no hay suficientes, con la mediana de las ventanas pre-QRS.
    """
    peaks = np.asarray(peaks, dtype=int)
    pre, n_w, shift, m = _p_geometry(fs)
    half = max(4, int(round(0.075 * fs)))
    if peaks.size < P_MIN_BEATS:
        return None, 0.0
    wide, ok = _windows(x, peaks, pre, n_w - pre)
    rows = np.flatnonzero(ok)
    if rows.size < P_MIN_BEATS:
        return None, 0.0
    work = _detrend_rows(wide[rows])
    row_of = {int(r): k for k, r in enumerate(rows)}
    if p_info is not None and p_info.present.any():
        sel = np.flatnonzero(p_info.present & ok)
        if sel.size >= P_MIN_BEATS:
            cols = np.asarray([int(p_info.peak[i]) - int(peaks[i] - pre + shift)
                               for i in sel])
            keep = (cols >= half) & (cols < m - half)
            segs = [work[row_of[int(i)], int(c) - half: int(c) + half + 1]
                    for i, c, k in zip(sel, cols, keep) if k]
            if len(segs) >= P_MIN_BEATS:
                sub = np.median(np.stack(segs), axis=0)
                amp = 1000.0 * float(np.ptp(sub))
                return (sub - sub.mean()), amp
    # sin P aceptadas: mediana de la ventana pre-QRS y pico del patrón
    tmpl = np.median(work, axis=0)
    pico = int(np.argmax(np.abs(tmpl)))
    if pico < half or pico >= m - half:
        return None, 0.0
    sub = tmpl[pico - half: pico + half + 1]
    return (sub - sub.mean()), 1000.0 * float(np.ptp(sub))


def _bump_uv(x: np.ndarray, fs: float, center: int, half_s: float = 0.05) -> float:
    """Amplitud pico-pico (µV) de una deflexión centrada en ``center``."""
    h = max(1, int(round(half_s * fs)))
    a, b = max(0, center - h), min(x.size, center + h + 1)
    if b - a < 3:
        return 0.0
    return 1000.0 * float(np.ptp(x[a:b]))


def _lead_noise_uv(x: np.ndarray, fs: float, center: int, half_s: float = 1.0) -> float:
    h = max(4, int(round(half_s * fs)))
    a, b = max(0, center - h), min(x.size, center + h + 1)
    if b - a < 8:
        return 1e-6
    d = np.diff(np.asarray(x[a:b], dtype=np.float64))
    return max(1e-6, _robust_sd(d) * 0.5 * 1000.0)


def _scan_correlation(x: np.ndarray, sub: np.ndarray, block_s: float, fs: float):
    """Correlación normalizada del patrón ``sub`` en todo el registro.

    Devuelve (centros, corr): el índice del centro del patrón. Se calcula por
    bloques para no llenar la memoria en un Holter de 48 h.
    """
    m = sub.size
    half = m // 2
    ones = np.ones(m, dtype=np.float64)
    tn = float(np.linalg.norm(sub))
    centers: list[np.ndarray] = []
    corrs: list[np.ndarray] = []
    n = x.size
    step = max(m * 4, int(round(block_s * fs)))
    pad = int(round(8 * fs))
    for a in range(0, n, step):
        b = min(n, a + step)
        lo = max(0, a - pad)
        hi = min(n, b + pad)
        ys = np.asarray(x[lo:hi], dtype=np.float64)
        if ys.size < m + 2:
            continue
        e = a - lo
        num = _conv(ys, sub[::-1])[m - 1:m - 1 + ys.size]
        s1 = _conv(ys, ones)[m - 1:m - 1 + ys.size]
        s2 = _conv(ys * ys, ones)[m - 1:m - 1 + ys.size]
        var = np.maximum(s2 - s1 * s1 / m, 1e-18)
        corr = num / (np.sqrt(var) * max(tn, 1e-12))
        # ventanas cuyo centro cae en [a, b)
        i0 = max(0, e - half)
        i1 = min(corr.size, e + (b - a) + half)
        centers.append(np.arange(i0, i1) + lo + half)
        corrs.append(corr[i0:i1])
    if not centers:
        return np.zeros(0, dtype=int), np.zeros(0)
    return np.concatenate(centers).astype(int), np.concatenate(corrs)


def _conv(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolución 'full' con la ruta rápida de ``filters.fft_conv``."""
    from holterai.legacy.app.signal.filters import fft_conv
    return fft_conv(x, h)


def _peak_indices(corr: np.ndarray, centers: np.ndarray, thr: float,
                  fs: float) -> np.ndarray:
    """Máximos locales de la correlación, separados >= 160 ms."""
    if corr.size == 0:
        return np.zeros(0, dtype=int)
    cand = np.flatnonzero(corr >= thr)
    if cand.size == 0:
        return np.zeros(0, dtype=int)
    # máximos locales
    keep = [int(i) for i in cand
            if corr[i] >= corr[i - 1] and (i + 1 >= corr.size or corr[i] >= corr[i + 1])]
    if not keep:
        return np.zeros(0, dtype=int)
    keep.sort(key=lambda i: -corr[i])
    taken = np.zeros(corr.size, dtype=bool)
    min_dist = max(1, int(round(0.16 * fs)))
    out: list[int] = []
    for i in keep:
        if taken[i]:
            continue
        out.append(i)
        taken[max(0, i - min_dist):i + min_dist + 1] = True
    return np.sort(np.asarray(out, dtype=int))


# ---------------------------------------------------------------------------
# Tren de P (incluye las bloqueadas) y conducción
# ---------------------------------------------------------------------------
@dataclass
class PTrain:
    """P detectadas en el registro, con o sin QRS que las siga."""

    peak: np.ndarray               # muestras del pico de P
    onset: np.ndarray              # muestras del inicio de P
    corr: np.ndarray
    amp_uv: np.ndarray
    n_leads: np.ndarray            # canales donde se vio la P
    conducted: np.ndarray          # ¿conduce?
    beat: np.ndarray               # índice de latido conducido (-1 si bloqueada)
    pr_ms: np.ndarray              # PR de la P (NaN si bloqueada)
    t_end: float = 0.0             # duración del registro (s)

    def __len__(self) -> int:
        return int(self.peak.size)


def _merge_events(peak: list[int], onset: list[int], corr: list[float],
                  amp: list[float], leads: list[int], fs: float) -> tuple:
    """Une candidatos cercanos (< 80 ms) quedándose con el de mayor correlación."""
    if not peak:
        return (np.zeros(0, int), np.zeros(0, int), np.zeros(0),
                np.zeros(0), np.zeros(0, int))
    order = np.argsort(np.asarray(peak))
    pks = np.asarray(peak)[order]
    ons = np.asarray(onset)[order]
    cor = np.asarray(corr, dtype=np.float64)[order]
    ampv = np.asarray(amp, dtype=np.float64)[order]
    nl = np.asarray(leads, dtype=int)[order]
    keep_p, keep_o, keep_c, keep_a, keep_l = [], [], [], [], []
    for i in range(pks.size):
        if keep_p and abs(int(pks[i]) - keep_p[-1]) < int(round(0.08 * fs)):
            if cor[i] > keep_c[-1]:
                keep_p[-1], keep_o[-1], keep_c[-1], keep_a[-1], keep_l[-1] = \
                    int(pks[i]), int(ons[i]), float(cor[i]), float(ampv[i]), int(nl[i])
            else:
                keep_l[-1] = max(keep_l[-1], int(nl[i]))
            continue
        keep_p.append(int(pks[i]))
        keep_o.append(int(ons[i]))
        keep_c.append(float(cor[i]))
        keep_a.append(float(ampv[i]))
        keep_l.append(int(nl[i]))
    return (np.asarray(keep_p, dtype=int), np.asarray(keep_o, dtype=int),
            np.asarray(keep_c), np.asarray(keep_a), np.asarray(keep_l, dtype=int))


def _blind_p_subtemplate(x: np.ndarray, fs: float, qrs_on: np.ndarray,
                         gap_s: float) -> tuple[np.ndarray | None, float]:
    """Patrón de P **sin ancla en el QRS** (disociación AV).

    Cuando las P no conducen no hay "ventana pre-QRS" de donde sacar la
    plantilla. Se busca entonces un tren *regular* de deflexiones pequeñas en la
    diástole (fuera del QRS y de la T): si los intervalos son regulares
    (MAD/mediana <= 12 %) y la mediana de esas ventanas se repite (mitades
    correlacionadas), ese es el patrón de P. En un registro normal esto no se
    usa nunca: la plantilla anclada al QRS siempre existe.
    """
    from holterai.legacy.app.signal.filters import filtfilt, fir_bandpass
    y = filtfilt(np.asarray(x, dtype=np.float64), fir_bandpass(fs, 2.0, 12.0))
    n = y.size
    allowed = np.ones(n, dtype=bool)
    q = np.asarray(qrs_on, dtype=int)
    q = q[q >= 0]
    for s in q:
        lo = max(0, s - int(round(0.05 * fs)))
        hi = min(n, s + int(round(gap_s * fs)))
        allowed[lo:hi] = False
    if allowed.sum() < int(2 * fs):
        return None, 0.0
    half = max(4, int(round(0.075 * fs)))
    dist = max(1, int(round(0.20 * fs)))
    cand: list[int] = []
    for i in range(half, n - half):
        if not allowed[i]:
            continue
        v = abs(y[i])
        seg = np.abs(y[max(0, i - dist): i + dist + 1])
        if seg.size and v >= seg.max() and (not cand or i - cand[-1] >= dist):
            cand.append(i)
    if len(cand) < 6:
        return None, 0.0
    idx = np.asarray(cand, dtype=int)
    amps = np.abs(y[idx])
    keep = amps >= max(0.5 * float(np.median(amps)), 1e-6)
    idx = idx[keep]
    if idx.size < 6:
        return None, 0.0
    d = np.diff(idx) / fs
    d = d[(d > 0.2) & (d < 2.0)]
    if d.size < 5:
        return None, 0.0
    med = float(np.median(d))
    if med <= 0 or float(np.median(np.abs(d - med))) / med > 0.12:
        return None, 0.0
    segs = np.stack([y[i - half: i + half + 1] for i in idx])
    segs = segs - np.median(segs, axis=1, keepdims=True)
    tmpl = np.median(segs, axis=0)
    a_h = np.median(segs[0::2], axis=0)
    b_h = np.median(segs[1::2], axis=0)
    na, nb = float(np.linalg.norm(a_h - a_h.mean())), float(np.linalg.norm(b_h - b_h.mean()))
    if na <= 1e-12 or nb <= 1e-12:
        return None, 0.0
    halves = float(np.dot(a_h - a_h.mean(), b_h - b_h.mean()) / (na * nb))
    amp = 1000.0 * float(np.ptp(tmpl))
    if amp < P_TPL_MIN_UV or halves < P_HALVES_MIN:
        return None, 0.0
    sub = tmpl - tmpl.mean()
    return sub, amp


def p_train(x: np.ndarray, fs: float, qrs_on: np.ndarray, peaks: np.ndarray,
            p_info: PInfo | None, leads: list[np.ndarray] | None = None,
            unreadable=None, beat_ok: np.ndarray | None = None) -> PTrain:
    """Tren de P del registro, incluidas las que **no** conducen.

    Combina el barrido por correlación (que ve las P sin QRS) con la P por
    latido (que es más exacta en su inicio) y aplica las tres exigencias de la
    P bloqueada: >= 2 canales, fuera de tramo ilegible y a tiempo (esto último
    en ``find_blocks``).
    """
    peaks = np.asarray(peaks, dtype=int)
    qrs_on = np.asarray(qrs_on, dtype=int)
    valid_qrs = qrs_on[qrs_on >= 0]
    sub, sub_amp = _p_subtemplate(x, fs, peaks, p_info)
    n_leads = 1 + (len(leads) if leads else 1)
    corr_thr = P_SCAN_CORR_MIN if len(leads or []) >= 1 else P_SCAN_CORR_ONE_LEAD
    # zona en la que no puede haber P: QRS y onda T del latido previo. Se
    # escala con el RR para no comerse la P bloqueada en ritmos rápidos.
    rr_med = float(np.median(np.diff(peaks))) / fs if peaks.size > 1 else 1.0
    gap_s = min(BLOCK_AFTER_QRS_S, 0.62 * max(rr_med, 0.25))
    if sub is None or (p_info is not None and p_info.present.mean() < 0.5):
        # pocas P ancladas al QRS: puede ser disociación AV. Se intenta el
        # patrón "ciego" (tren regular de deflexiones en la diástole) y, si
        # aparece, se usa ese; si no, queda el anclado (o nada).
        blind, blind_amp = _blind_p_subtemplate(x, fs, qrs_on, gap_s)
        if blind is not None:
            sub, sub_amp = blind, blind_amp
    cand_p: list[int] = []
    cand_o: list[int] = []
    cand_c: list[float] = []
    cand_a: list[float] = []
    cand_l: list[int] = []
    half = max(4, int(round(0.075 * fs)))

    if sub is not None and sub.size >= 9:
        centers, corr = _scan_correlation(x, sub, SCAN_BLOCK_S, fs)
        idx = _peak_indices(corr, centers, corr_thr, fs)
        for i in idx:
            c = int(centers[i])
            if c < half or c >= x.size - half:
                continue
            if valid_qrs.size:
                after = valid_qrs[valid_qrs <= c]
                before = valid_qrs[valid_qrs > c]
                if after.size and (c - int(after[-1])) / fs < gap_s:
                    continue
                if before.size and (int(before[0]) - c) / fs < 0.01:
                    continue
            amp = _bump_uv(x, fs, c)
            if amp < max(P_AMP_MIN_UV, P_SCAN_AMP_REL * sub_amp):
                continue
            if sub_amp > 0 and amp > P_BLOCK_HI_AMP * sub_amp:
                continue
            nl = 0
            for lx in (leads or []):
                bump = _bump_uv(lx, fs, c)
                if bump >= max(LEAD_BUMP_MIN_UV, LEAD_BUMP_NOISE * _lead_noise_uv(lx, fs, c)):
                    nl += 1
            cand_p.append(c)
            cand_o.append(_scan_onset(x, fs, c, p_info))
            cand_c.append(float(corr[i]))
            cand_a.append(amp)
            cand_l.append(1 + nl)

    # P por latido (T10): más exacta en el inicio; añade las que el barrido no ve
    if p_info is not None:
        for i in np.flatnonzero(p_info.present & (p_info.onset >= 0)):
            pk = int(p_info.peak[i]) if p_info.peak[i] >= 0 else int(p_info.onset[i]) + half
            bump = _bump_uv(x, fs, pk)
            if sub_amp > 0 and bump > P_BLOCK_HI_AMP * sub_amp:
                continue
            cand_p.append(pk)
            cand_o.append(int(p_info.onset[i]))
            cand_c.append(float(p_info.corr[i]))
            cand_a.append(bump)
            cand_l.append(1)

    pk, on, co, am, nl = _merge_events(cand_p, cand_o, cand_c, cand_a, cand_l, fs)

    # la P de un latido manda sobre el inicio del barrido (T10 es más exacto)
    if p_info is not None and pk.size:
        for i in np.flatnonzero(p_info.present & (p_info.onset >= 0)):
            if pk.size == 0:
                break
            d = np.abs(pk - int(p_info.peak[i]))
            j = int(np.argmin(d))
            if d[j] <= int(round(0.08 * fs)):
                on[j] = int(p_info.onset[i])
                nl[j] = max(nl[j], 1)

    # sin P en tramo ilegible
    if unreadable is not None and pk.size:
        keep = ~np.asarray(unreadable(pk / fs), dtype=bool)
        pk, on, co, am, nl = pk[keep], on[keep], co[keep], am[keep], nl[keep]

    train = PTrain(peak=pk, onset=on, corr=co, amp_uv=am, n_leads=nl,
                   conducted=np.zeros(pk.size, dtype=bool),
                   beat=np.full(pk.size, -1, dtype=int),
                   pr_ms=np.full(pk.size, np.nan, dtype=np.float64))
    _assign_conduction(train, qrs_on, peaks, fs, beat_ok)
    return train


def _scan_onset(x: np.ndarray, fs: float, center: int,
                p_info: PInfo | None) -> int:
    """Inicio de una P del barrido: pico menos el desfase de la plantilla.

    El barrido localiza la P por correlación (el pico es fiable), pero medir su
    inicio con un umbral sobre la señal cruda agrega ±20 ms de ruido que entran
    enteros en el PR. La distancia pico→inicio de la plantilla del minuto
    (``PInfo.peak_to_onset``) es la misma para todas las P del tramo y mucho más
    estable; solo si no hay plantilla se mide en el latido.
    """
    if p_info is not None and p_info.peak_to_onset > 0:
        onset = int(center) - int(p_info.peak_to_onset)
        if onset >= 0:
            return onset
    return _onset_from_peak(x, fs, center)


def _onset_from_peak(x: np.ndarray, fs: float, center: int) -> int:
    """Inicio de la P por el criterio del 10 % de la amplitud, desde su pico."""
    h = max(1, int(round(0.10 * fs)))
    a = max(0, center - int(round(P_ONSET_MAX_S * fs)))
    b = min(x.size, center + h)
    if b - a < 4:
        return int(center)
    seg = np.asarray(x[a:b], dtype=np.float64)
    base = float(np.median(seg[: max(2, int(round(0.03 * fs)))]))
    peak = float(np.max(np.abs(seg - base)))
    if peak <= 1e-9:
        return int(center)
    above = np.abs(seg - base) > P_ONSET_FRAC * peak
    ref = int(np.argmax(np.abs(seg - base)))
    start = _run_start(above[None, :], np.array([ref]))[0]
    return int(a + max(0, start))


def _assign_conduction(train: PTrain, qrs_on: np.ndarray, peaks: np.ndarray,
                       fs: float, beat_ok: np.ndarray | None) -> None:
    """Marca qué P conduce y con qué PR; el resto queda bloqueada."""
    if len(train) == 0:
        return
    valid = np.flatnonzero(qrs_on >= 0)
    if valid.size == 0:
        return
    q_on = qrs_on[valid]
    for i in range(len(train)):
        p_on = int(train.onset[i])
        if p_on < 0:
            continue
        j = int(np.searchsorted(q_on, p_on, side="left"))
        # un QRS antes del inicio de su P no conduce esa P; el candidato es el
        # primer QRS cuyo inicio caiga en la ventana de conducción
        best = -1
        best_pr = np.nan
        for k in (j - 1, j):
            if k < 0 or k >= q_on.size:
                continue
            pr = (int(q_on[k]) - p_on) * 1000.0 / fs
            if PR_MIN_MS <= pr <= PR_MAX_MS:
                if best < 0 or pr < best_pr:
                    best, best_pr = int(k), float(pr)
        if best < 0:
            continue
        bi = int(valid[best])
        if beat_ok is not None and bi < beat_ok.size and not bool(beat_ok[bi]):
            continue
        train.conducted[i] = True
        train.beat[i] = bi
        train.pr_ms[i] = best_pr


def _beat_pr(qrs_on: np.ndarray, train: PTrain) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PR por latido (ms), su validez y la P elegida (índice del tren).

    Si dos P conducidas reclaman el mismo QRS se queda la **inmediatamente
    anterior** al QRS: es la que lo despolarizó. Con la primera de la lista, un
    candidato espurio (una T tomada por P) se adelantaba y metía un PR falso de
    400 ms en la serie.
    """
    n = qrs_on.size
    pr = np.full(n, np.nan)
    best = np.full(n, -1, dtype=int)
    sel = np.full(n, -1, dtype=int)
    ok = np.zeros(n, dtype=bool)
    for i in np.flatnonzero(train.conducted):
        b = int(train.beat[i])
        if not (0 <= b < n):
            continue
        if not ok[b] or int(train.onset[i]) > best[b]:
            pr[b] = train.pr_ms[i]
            best[b] = int(train.onset[i])
            sel[b] = int(i)
            ok[b] = True
    return pr, ok, sel


def _mad_gate(values: np.ndarray, sel: np.ndarray, min_ms: float) -> np.ndarray:
    """Máscara de valores dentro de ``min_ms`` o 3 MAD de la mediana."""
    out = np.zeros(values.shape, dtype=bool)
    if sel.sum() < 3:
        out[sel] = True
        return out
    v = values[sel]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    tol = max(min_ms, 3.0 * mad)
    out = np.abs(values - med) <= tol
    return out & sel


def pr_gates(peaks: np.ndarray, qrs_raw: np.ndarray, qrs_on: np.ndarray,
             train: PTrain, pr: np.ndarray, ok: np.ndarray, sel: np.ndarray,
             fs: float) -> np.ndarray:
    """Control de calidad del PR: morfología de QRS y de P consistentes.

    El PR es una diferencia entre dos marcas; si una de las dos se midió mal el
    número sale mal sin avisar. Tres filtros, todos internos al registro (no
    usan anotación): (a) el intervalo R→inicio de QRS de cada latido no puede
    separarse del de la mediana local, (b) la duración pico→inicio de la P
    tampoco, (c) la P elegida tiene que correlacionar con la plantilla del
    minuto.
    """
    keep = np.array(ok, dtype=bool)
    if not keep.any():
        return keep
    pre_raw = (peaks - qrs_raw).astype(np.float64)
    pre_used = (peaks - qrs_on).astype(np.float64)
    mad = np.zeros(pre_used.shape)
    gid = np.floor(peaks / max(1.0, PR_GATE_SEGMENT_S * fs)).astype(int)
    for g in np.unique(gid[keep]):
        k = keep & (gid == g)
        if k.sum() >= 3:
            mad[k] = float(np.median(np.abs(pre_raw[k] - pre_used[k]))) * 1.4826
    tol = np.maximum(QRS_PRE_GATE_MS, 3.0 * mad)
    keep = keep & (np.abs(pre_raw - pre_used) <= tol)
    idx = np.flatnonzero(keep)
    if idx.size and len(train):
        p_off = (train.peak - train.onset) * 1000.0 / fs
        mask = np.zeros(len(train), dtype=bool)
        mask[sel[idx]] = True
        good_p = _mad_gate(p_off, mask, P_OFF_GATE_MS)
        for i in idx:
            j = int(sel[i])
            if not good_p[j] or train.corr[j] < P_PR_CORR_MIN:
                keep[i] = False
    # (d) coherencia temporal: el PR de un paciente cambia despacio. Un latido
    # cuyo PR se separa de la mediana del minuto más que la tolerancia no es un
    # PR: es una P mal localizada o una onda T tomada por P. Se informa aparte
    # (cobertura del PR) en vez de ensuciar la serie, la tendencia y el primer
    # grado.
    keep = pr_consistency_gate(pr, keep, peaks, fs)
    return keep


def pr_consistency_gate(pr_ms: np.ndarray, keep: np.ndarray, peaks: np.ndarray,
                        fs: float, tol_ms: float | None = None,
                        segment_s: float = PR_GATE_SEGMENT_S) -> np.ndarray:
    """Descarta los PR que se alejan de la mediana local del minuto.

    Dos pasadas: la primera quita los valores disparatados, la segunda vuelve a
    estimar la mediana con los que quedan (así una racha de valores malos no
    arrastra el centro). La tolerancia es ``max(tol_ms, 3 MAD)``, de modo que un
    PR que de verdad varía (Wenckebach, PR que crece por ciclos) no se recorta.
    """
    keep = np.asarray(keep, dtype=bool).copy()
    if tol_ms is None:
        tol_ms = PR_CONSIST_MS
    if tol_ms <= 0 or not keep.any():
        return keep
    gid = np.floor(np.asarray(peaks) / max(1.0, segment_s * fs)).astype(int)
    v = np.asarray(pr_ms, dtype=np.float64)
    for _ in range(2):
        base = keep.copy()
        for g in np.unique(gid[base]):
            ids = np.flatnonzero(base & (gid == g))
            if ids.size < 3:
                continue
            med = float(np.median(v[ids]))
            mad = float(np.median(np.abs(v[ids] - med))) * 1.4826
            tol = max(float(tol_ms), 3.0 * mad)
            keep[ids] = np.abs(v[ids] - med) <= tol
    return keep


# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------
@dataclass
class AVResult:
    """Resultado de la conducción AV de un registro."""

    pr_ms: np.ndarray
    pr_valid: np.ndarray
    p_on: np.ndarray
    qrs_on: np.ndarray
    qrs_on_raw: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    minute_pr: list = field(default_factory=list)
    trend: dict = field(default_factory=dict)
    blocks: list = field(default_factory=list)
    train: PTrain | None = None
    summary: dict = field(default_factory=dict)


def minute_series(t: np.ndarray, pr_ms: np.ndarray, ok: np.ndarray,
                  minute_s: float = 60.0,
                  min_beats: int = PR_MIN_BEATS_PER_MINUTE) -> list:
    """Mediana de PR por minuto: ``[[t0_s, mediana_ms, n], ...]``."""
    if t.size == 0:
        return []
    sel = np.asarray(ok, dtype=bool) & np.isfinite(pr_ms)
    if not sel.any():
        return []
    minute = np.floor(t[sel] / minute_s).astype(int)
    out = []
    for m in np.unique(minute):
        vals = np.asarray(pr_ms[sel])[minute == m]
        if vals.size < min_beats:
            continue
        out.append([float(m * minute_s), round(float(np.median(vals)), 1), int(vals.size)])
    return out


def pr_trend(minute_pr: list, hour_s: float = 3600.0) -> dict:
    """Tendencia del PR: pendiente (ms/h) y primer/último tramo del registro."""
    if not minute_pr:
        return {}
    t = np.array([x[0] for x in minute_pr], dtype=np.float64)
    v = np.array([x[1] for x in minute_pr], dtype=np.float64)
    n = np.array([x[2] for x in minute_pr], dtype=np.float64)
    out = {"pr_minutes": int(t.size), "pr_beats": int(n.sum()),
           "pr_median_ms": round(float(np.average(v, weights=n)), 1)}
    if t.size >= 3 and np.ptp(t) > 0:
        tm = t - t.mean()
        den = float(np.sum(tm ** 2))
        if den > 0:
            slope_ms_per_s = float(np.sum(tm * (v - np.average(v, weights=n)))) / den
            out["slope_ms_per_h"] = round(slope_ms_per_s * hour_s, 2)
    first = t <= t.min() + hour_s
    last = t >= max(t.min(), t.max() - hour_s)
    out["pr_first_hour_ms"] = round(float(np.average(v[first], weights=n[first])), 1)
    out["pr_last_hour_ms"] = round(float(np.average(v[last], weights=n[last])), 1)
    out["duration_h"] = round(float(np.ptp(t) + 60.0) / hour_s, 2)
    return out


# ---------------------------------------------------------------------------
# Detección de bloqueos
# ---------------------------------------------------------------------------
def _pp_estimate(times: np.ndarray, half_frac: float = 0.15) -> float:
    """PP (s) a partir de los intervalos: el **corto**, no la mediana.

    Cuando dos P caen muy juntas en el tiempo (una P sobre la onda T, una P
    dentro del QRS) el detector ve una sola y el intervalo se duplica. Con la
    mediana, media docena de P perdidas bastan para que el "PP" salga al doble y
    ninguna P bloqueada parezca estar a tiempo; el percentil 25 es el PP de las
    P que sí se ven, y se afina con la mediana de los intervalos que lo rodean.
    """
    d = np.diff(np.sort(np.asarray(times, dtype=np.float64)))
    d = d[(d > 0.20) & (d < 2.0)]
    if d.size < 3:
        return 0.0
    pp = float(np.percentile(d, 25))
    if pp <= 0:
        return 0.0
    near = d[np.abs(d - pp) <= half_frac * pp]
    if near.size >= 2:
        pp = float(np.median(near))
    return pp if 0.30 <= pp <= 1.70 else 0.0


def _local_pp(train: PTrain, i: int, fs: float, half: int = 6) -> float:
    """PP local (s) alrededor de la P ``i``.

    Se usan **todas** las P, no solo las que conducen: el PP es la frecuencia
    auricular y las P bloqueadas forman parte de ella (con solo las conducidas,
    en un 2:1 el "PP" sale al doble y ninguna P bloqueada parece estar a tiempo).
    """
    lo = max(0, i - half)
    hi = min(len(train), i + half + 1)
    if len(train) == 0:
        return 0.0
    return _pp_estimate(np.sort(train.peak[lo:hi] / fs))


def _pp_regular(train: PTrain, i: int, fs: float, half: int = 6) -> bool:
    """¿El tren de P es regular alrededor de ``i``? (ritmo auricular organizado)

    Regular = cada intervalo visible es un múltiplo entero del PP (1, 2, 3 o 4
    veces: una P puede quedar escondida en la T o en el QRS). En fibrilación
    auricular los intervalos son aleatorios y no caen en múltiplos.
    """
    lo = max(0, i - half)
    hi = min(len(train), i + half + 1)
    if len(train) == 0 or hi - lo < 4:
        return False
    times = np.sort(np.asarray(train.peak[lo:hi] / fs))
    pp = _pp_estimate(times)
    if pp <= 0:
        return False
    d = np.diff(times)
    d = d[(d > 0.20) & (d < 2.0)]
    if d.size < 3:
        return False
    tol = max(PP_TOL_MS / 1000.0, PP_TOL_FRAC * pp)
    k = np.round(d / pp)
    ok = (k >= 1) & (k <= 4) & (np.abs(d - k * pp) <= tol)
    return bool(ok.mean() >= 0.80)


def _on_time(train: PTrain, i: int, pp: float, fs: float) -> bool:
    """¿La P ``i`` llega cuando corresponde según el PP local?

    Se admite que falte la P anterior (el hueco vale 2 o 3 PP): la P bloqueada
    puede ser la primera de una pareja que el detector ve como una sola.
    """
    if pp <= 0 or i == 0:
        return False
    gap = (int(train.peak[i]) - int(train.peak[i - 1])) / fs
    tol = max(PP_TOL_MS / 1000.0, PP_TOL_FRAC * pp)
    k = max(1.0, round(gap / pp))
    if abs(gap - k * pp) > tol:
        return False
    if i + 1 < len(train):
        gap2 = (int(train.peak[i + 1]) - int(train.peak[i])) / fs
        if gap2 < 0.85 * pp:            # dos P demasiado juntas: no es sinusal
            return False
    return True


def _conducted_run(train: PTrain, i: int, need: int = MIN_CONDUCTED_BEFORE) -> list:
    """Índices de las P conducidas inmediatamente anteriores a ``i``."""
    out = []
    k = i - 1
    while k >= 0 and len(out) < need:
        if train.conducted[k]:
            out.append(k)
        else:
            break
        k -= 1
    return out[::-1]


def _blocked_ok(train: PTrain, i: int, fs: float, n_leads_req: int,
                qrs_on: np.ndarray | None = None) -> tuple[bool, str]:
    """Filtros comunes de una P bloqueada: canales, ritmo, puntualidad y hueco."""
    if train.conducted[i]:
        return False, "conduce"
    if train.n_leads[i] < n_leads_req:
        return False, "un_canal"
    if not _pp_regular(train, i, fs):
        return False, "pp_irregular"
    prev = _conducted_run(train, i)
    if len(prev) < MIN_CONDUCTED_BEFORE:
        return False, "sin_ritmo_establecido"
    pp = _local_pp(train, i, fs)
    if not _on_time(train, i, pp, fs):
        return False, "fuera_de_tiempo"
    prs = [train.pr_ms[k] for k in prev if np.isfinite(train.pr_ms[k])]
    if len(prs) < 2:
        return False, "sin_pr_previo"
    if qrs_on is not None:
        ok, why = _ventricular_evidence(train, i, qrs_on, fs, pp)
        if not ok:
            return False, why
    return True, ""


def _qrs_around(qrs_on: np.ndarray, p_on: int, p_peak: int) -> tuple[int, int]:
    """QRS inmediatamente antes del inicio de P y después de su pico (−1 si no)."""
    q = np.asarray(qrs_on, dtype=int)
    q = q[q >= 0]
    if q.size == 0:
        return -1, -1
    before = q[q <= p_on]
    after = q[q > p_peak]
    return (int(before[-1]) if before.size else -1,
            int(after[0]) if after.size else -1)


def _ventricular_evidence(train: PTrain, i: int, qrs_on: np.ndarray, fs: float,
                          pp: float) -> tuple[bool, str]:
    """Una P caída deja **un latido ventricular de menos**, y eso se ve en el RR.

    Entre el QRS de antes y el de después de la P bloqueada tiene que haber un
    intervalo PP de más (2:1 y Mobitz: el hueco vale 2 PP; en el bloqueo
    avanzado, 3 o 4). Sin esta condición, una P que sí condujo (con un PR fuera
    del rango aceptado) o el final de una racha de ruido se leerían como
    bloqueo, que es el falso positivo caro: en un registro normal sin bloqueos
    el hueco ventricular nunca aparece.
    """
    if pp <= 0:
        return False, "sin_pp"
    q = np.asarray(qrs_on, dtype=int)
    q = q[q >= 0]
    if q.size and float(np.min(np.abs(q - int(train.peak[i])))) / fs < 0.10:
        return False, "p_en_qrs"          # la "P" está dentro de un QRS
    before, after = _qrs_around(qrs_on, int(train.onset[i]), int(train.peak[i]))
    if before < 0 or after < 0:
        return False, "sin_qrs_despues"
    span = (after - before) / fs
    if span < 1.55 * pp:
        return False, "rr_corto"          # no se perdió ningún latido
    if span > 4.6 * pp:
        return False, "rr_largo"          # pausa larga: no es un bloqueo 2:1
    return True, ""


def _strip(t: float, duration_s: float, center_s: float) -> tuple[float, float]:
    """Ventana de 10 s que muestra el evento (dentro del registro)."""
    a = max(0.0, center_s - STRIP_S / 2.0)
    b = min(max(duration_s, STRIP_S), a + STRIP_S)
    return (round(a, 1), round(max(a + 1.0, b), 1))


def _event(kind: str, start_s: float, end_s: float, detail: str,
           beats: list[int], duration_s: float) -> dict:
    center = 0.5 * (start_s + end_s)
    a, b = _strip(start_s, duration_s, center)
    return {
        "kind": kind,
        "start_s": round(float(start_s), 2),
        "end_s": round(float(max(end_s, start_s + 0.2)), 2),
        "severity": "warn",
        "detail": f"{KIND_NAMES.get(kind, kind)}: {detail} (a revisión)",
        "beats": [int(x) for x in beats][:200],
        "strip_start_s": a,
        "strip_end_s": b,
    }


def _smooth_minute_series(serie: list, half: int = PR_FIRST_DEGREE_SMOOTH,
                          min_pts: int = 3) -> list:
    """Mediana móvil de la serie de PR por minuto (robusta a un minuto raro).

    La mediana de un minuto todavía se mueve ±30 ms por cambios de morfología
    (respiración, línea de base): en un registro normal eso alcanza para cruzar
    los 200 ms un par de minutos. El criterio clínico es "sostenido", así que la
    decisión se toma sobre la mediana móvil de varios minutos, no sobre un
    minuto aislado.
    """
    if half <= 0 or len(serie) < min_pts:
        return list(serie)
    t = np.asarray([r[0] for r in serie], dtype=np.float64)
    v = np.asarray([r[1] for r in serie], dtype=np.float64)
    out = []
    for i in range(len(serie)):
        lo, hi = max(0, i - half), min(len(serie), i + half + 1)
        if hi - lo >= min_pts:
            out.append([float(t[i]), round(float(np.median(v[lo:hi])), 1),
                        int(sum(serie[k][2] for k in range(lo, hi)))])
        else:
            out.append([float(t[i]), float(v[i]), int(serie[i][2])])
    return out


def first_degree_blocks(t: np.ndarray, pr_ms: np.ndarray, ok: np.ndarray,
                        minute_s: float = 60.0,
                        min_s: float = PR_FIRST_DEGREE_MIN_S,
                        duration_s: float = 0.0) -> list:
    """Bloqueo de primer grado: PR > 200 ms sostenido >= 1 min.

    El umbral de aviso suma ``PR_FIRST_DEGREE_TOL_MS``: el PR medido tiene un
    error de ±10 ms (ver la validación en LUDB) y con el criterio pelado un
    registro normal con PR de 195 ms entra y sale del bloqueo cada pocos
    minutos. La serie se suaviza con mediana móvil para no confundir el
    vaivén normal del PR con un bloqueo.
    """
    serie = _smooth_minute_series(minute_series(t, pr_ms, ok, minute_s))
    if not serie:
        return []
    thr = PR_FIRST_DEGREE_MS + PR_FIRST_DEGREE_TOL_MS
    out = []
    run: list[list] = []
    for row in serie:
        if row[1] > thr:
            run.append(row)
        elif run:
            out.extend(_emit_first_degree(run, min_s, duration_s, thr))
            run = []
    if run:
        out.extend(_emit_first_degree(run, min_s, duration_s, thr))
    return out


def _emit_first_degree(run: list, min_s: float, duration_s: float,
                       thr: float = PR_FIRST_DEGREE_MS) -> list:
    span = run[-1][0] - run[0][0] + 60.0
    if span < min_s:
        return []
    vals = [r[1] for r in run]
    detail = (f"PR mediana {float(np.median(vals)):.0f} ms "
              f"(umbral {thr:.0f} ms, máx {max(vals):.0f} ms) "
              f"durante {span / 60.0:.0f} min")
    return [_event(KIND_1, run[0][0], run[-1][0] + 60.0, detail, [], duration_s)]


def _two_to_one(train: PTrain, fs: float, duration_s: float,
                qrs_on: np.ndarray | None = None,
                n_leads_req: int = 1) -> list:
    """2:1: una P conduce y la siguiente se cae, repetido y con PR estable."""
    out = []
    n = len(train)
    if n < 2 * TWO_TO_ONE_MIN_BLOCKED:
        return out
    i = 0
    while i < n - 1:
        start = None
        count = 0
        j = i
        pr_ref = None
        while j + 1 < n:
            c, b = train.conducted[j], train.conducted[j + 1]
            if not (c and not b):
                break
            if train.n_leads[j + 1] < n_leads_req:
                break
            if start is None:
                start = j
                pr_ref = train.pr_ms[j]
            elif not (np.isfinite(train.pr_ms[j]) and abs(train.pr_ms[j] - pr_ref) <= MOBITZ2_STABLE_MS):
                break
            pp = _local_pp(train, j + 1, fs)
            if pp <= 0 or not _on_time(train, j + 1, pp, fs):
                break
            if qrs_on is not None:
                okv, _why = _ventricular_evidence(train, j + 1, qrs_on, fs, pp)
                if not okv:
                    break
                # en un 2:1 el hueco ventricular vale exactamente 2 PP (la
                # frecuencia ventricular es la mitad de la auricular). Si el PP
                # salió de P espurias (T tomadas por P), el cociente no cierra.
                before, after = _qrs_around(qrs_on, int(train.onset[j + 1]),
                                            int(train.peak[j + 1]))
                span = (after - before) / fs
                if not (1.70 * pp <= span <= 2.35 * pp):
                    break
            count += 1
            j += 2
        if count >= TWO_TO_ONE_MIN_BLOCKED:
            t0 = train.peak[start] / fs
            t1 = train.peak[j - 1] / fs if j - 1 < n else t0 + 1.0
            beats = [int(train.beat[k]) for k in range(start, min(j, n))
                     if train.conducted[k] and train.beat[k] >= 0]
            ratio = count
            out.append(_event(KIND_21, t0, t1,
                              f"{ratio} ciclos con 2 P por QRS (PR {pr_ref:.0f} ms)", beats,
                              duration_s))
            i = j
        else:
            i += 1
    return out


def _second_degree(train: PTrain, fs: float, duration_s: float,
                   n_leads_req: int, qrs_on: np.ndarray | None = None) -> list:
    """Mobitz I / II y bloqueo avanzado a partir de las P bloqueadas."""
    out: list[dict] = []
    n = len(train)
    used = np.zeros(n, dtype=bool)
    for i in range(n):
        ok, _why = _blocked_ok(train, i, fs, n_leads_req, qrs_on)
        if not ok:
            continue
        prev = _conducted_run(train, i)
        prs = [float(train.pr_ms[k]) for k in prev if np.isfinite(train.pr_ms[k])]
        t0 = train.peak[i] / fs
        beats = [int(train.beat[k]) for k in prev if train.beat[k] >= 0]
        # bloqueo avanzado: >= 2 P bloqueadas seguidas
        adv = 1
        j = i + 1
        while j < n and not train.conducted[j] and train.n_leads[j] >= n_leads_req:
            okj, _ = _blocked_ok(train, j, fs, n_leads_req, qrs_on)
            if not okj:
                break
            adv += 1
            j += 1
        if adv >= 2:
            t1 = train.peak[j - 1] / fs
            out.append(_event(KIND_ADV, t0, t1, f"{adv} P bloqueadas seguidas",
                              beats, duration_s))
            used[i:j] = True
            continue
        if used[i]:
            continue
        spread = max(prs) - min(prs) if prs else 0.0
        growth = prs[-1] - prs[0] if len(prs) >= 3 else 0.0
        rising = len(prs) >= 3 and growth >= MOBITZ1_GROWTH_MS and prs[-1] >= max(prs) - 10.0
        if rising:
            out.append(_event(
                KIND_M1, t0, t0,
                f"PR crece de {prs[0]:.0f} a {prs[-1]:.0f} ms antes de la P caída",
                beats, duration_s))
        elif len(prs) >= 3 and spread <= MOBITZ2_STABLE_MS:
            out.append(_event(
                KIND_M2, t0, t0,
                f"PR estable ({float(np.median(prs)):.0f} ms) y una P caída",
                beats, duration_s))
        # si el PR previo no es ni creciente ni estable no se clasifica: una P
        # caída sin patrón puede ser una P ectópica bloqueada, no un Mobitz
    return out


def _complete_blocks(train: PTrain, fs: float, duration_s: float,
                     n_leads_req: int, qrs_on: np.ndarray | None = None) -> list:
    """Bloqueo AV completo: disociación AV (P regulares que no conducen).

    Se recorre el tren con una ventana de ``COMPLETE_WINDOW_S`` y se exige, a la
    vez: PP regular (ritmo auricular organizado), RR regular (escape),
    frecuencia auricular >= 1,4 veces la ventricular, más de un 45 % de P sin
    conducir y PR variable (las P "caminan" a través del QRS). Un 2:1 también
    tiene PP/RR regulares y el 50 % de P sin conducir; lo que lo separa es que
    ahí el PR de las conducidas es constante, así que la variabilidad del PR es
    la condición que no puede faltar.
    """
    out: list[dict] = []
    n = len(train)
    if n < 6:
        return out
    t = train.peak / fs
    blocked = ~train.conducted & (train.n_leads >= n_leads_req)
    i = 0
    while i < n:
        if not blocked[i]:
            i += 1
            continue
        if not _pp_regular(train, i, fs, half=8):
            i += 1
            continue
        j = i
        while j < n and t[j] - t[i] <= COMPLETE_WINDOW_S:
            j += 1
        ids = np.arange(i, j)
        if ids.size < 6:
            i = j
            continue
        tb = t[ids]
        frac = float(blocked[ids].mean())
        cond = [int(k) for k in ids if train.conducted[k]]
        prs = np.asarray([train.pr_ms[k] for k in cond if np.isfinite(train.pr_ms[k])])
        # el RR de la escape se mide sobre las muestras del QRS (``train.beat``
        # es el índice del latido, no su tiempo)
        beats = np.asarray(sorted({int(train.beat[k]) for k in cond if train.beat[k] >= 0}))
        if qrs_on is not None and beats.size >= 2:
            qt = np.asarray(qrs_on, dtype=int)
            qt = qt[np.clip(beats, 0, max(qt.size - 1, 0))]
            rr = np.diff(qt[(qt >= 0)]) / fs
        else:
            rr = np.zeros(0)
        rr = rr[(rr > 0.30) & (rr < 3.0)]
        pp = _pp_estimate(tb)
        p_rate = 60.0 / pp if pp > 0 else 0.0
        v_rate = 60.0 / float(np.median(rr)) if rr.size else 0.0
        rr_reg = (rr.size >= 2 and
                  float(np.median(np.abs(rr - np.median(rr)))) / max(float(np.median(rr)), 1e-6)
                  <= COMPLETE_RR_REGULAR_FRAC)
        diss = prs.size >= 2 and float(np.std(prs)) >= COMPLETE_PR_SD_MS
        span = float(tb[-1] - tb[0])
        if (span >= COMPLETE_MIN_S and frac >= COMPLETE_BLOCKED_FRAC and rr_reg
                and diss and p_rate >= COMPLETE_MIN_P_RATIO * max(v_rate, 1e-6)):
            out.append(_event(
                KIND_COMPLETE, float(tb[0]), float(tb[-1]),
                f"P a {p_rate:.0f} lpm sin conducir, escape a {v_rate:.0f} lpm, "
                f"PR variable (DE {float(np.std(prs)):.0f} ms)",
                [int(b) for b in beats], duration_s))
            i = j
            continue
        i += 1
    return out


def find_blocks(train: PTrain, qrs_on: np.ndarray, pr_ms: np.ndarray,
                pr_ok: np.ndarray, fs: float, duration_s: float,
                n_leads_req: int = 1) -> list:
    """Todos los eventos de bloqueo AV del registro (a revisión)."""
    t = qrs_on / fs if qrs_on.size else np.zeros(0)
    out: list[dict] = []
    out.extend(first_degree_blocks(t, pr_ms, pr_ok, duration_s=duration_s))
    out.extend(_complete_blocks(train, fs, duration_s, n_leads_req, qrs_on))
    complete = [(e["start_s"], e["end_s"]) for e in out if e["kind"] == KIND_COMPLETE]
    second = _two_to_one(train, fs, duration_s, qrs_on, n_leads_req)
    second.extend(_second_degree(train, fs, duration_s, n_leads_req, qrs_on))
    keep = []
    for e in second:
        inside = any(e["start_s"] >= a - 1.0 and e["end_s"] <= b + 1.0
                     for a, b in complete)
        if not inside:
            keep.append(e)
    out.extend(keep)
    out.sort(key=lambda e: e["start_s"])
    return _merge_blocks(out)


def _merge_blocks(events: list) -> list:
    """Une eventos del mismo tipo que se solapan o quedan a < 30 s."""
    out: list[dict] = []
    for e in sorted(events, key=lambda x: (x["kind"], x["start_s"])):
        if out and out[-1]["kind"] == e["kind"] and e["start_s"] <= out[-1]["end_s"] + 30.0:
            last = out[-1]
            last["end_s"] = max(last["end_s"], e["end_s"])
            last["beats"] = sorted(set(last["beats"]) | set(e["beats"]))[:200]
            continue
        out.append(dict(e))
    out.sort(key=lambda x: x["start_s"])
    return out


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------
def _unreadable_fn(unreadable):
    """Normaliza la máscara de tramos ilegibles a una función tiempo -> bool."""
    if unreadable is None:
        return None
    if callable(unreadable):
        return unreadable
    spans = np.asarray(unreadable, dtype=np.float64)
    if spans.size == 0:
        return None
    if spans.ndim == 1:
        spans = spans[None, :]
    if spans.shape[1] == 2:
        def inside(times: np.ndarray) -> np.ndarray:
            times = np.asarray(times, dtype=np.float64)
            out = np.zeros(times.shape, dtype=bool)
            for a, b in spans:
                out |= (times >= a) & (times <= b)
            return out
        return inside
    raise ValueError("unreadable debe ser una función, o una lista de tramos (a, b) en segundos")


def av_analysis(x: np.ndarray, fs: float, peaks: np.ndarray, *,
                leads: list[np.ndarray] | None = None,
                unreadable=None,
                beat_ok: np.ndarray | None = None,
                p_result: dict | None = None,
                duration_s: float | None = None,
                minute_s: float = 60.0) -> AVResult:
    """Conducción AV completa: PR por latido, minutos, tendencia y bloqueos.

    ``x`` es el canal de análisis ya filtrado; ``leads`` los otros canales
    filtrados (solo se usan para confirmar que una P bloqueada se ve en >= 2
    canales). ``unreadable`` es la máscara de T21 (función tiempo->bool o lista
    de tramos en segundos) y ``beat_ok`` los latidos donde el ritmo permite
    hablar de conducción (ni FA ni rachas ventriculares).
    """
    x = np.asarray(x, dtype=np.float64)
    peaks = np.asarray(peaks, dtype=int)
    fs = float(fs)
    dur = float(duration_s) if duration_s is not None else x.size / max(fs, 1e-9)
    qrs_raw = qrs_onsets(x, fs, peaks)
    qrs_on, pre_med = smooth_qrs_onsets(peaks, qrs_raw, fs)
    p_info = detect_p(x, fs, peaks, p_result)
    n_leads_req = 2 if leads else 1
    train = p_train(x, fs, qrs_on, peaks, p_info, leads=leads,
                    unreadable=_unreadable_fn(unreadable), beat_ok=beat_ok)
    pr_ms, pr_ok, sel = _beat_pr(qrs_on, train)
    pr_ok = pr_gates(peaks, qrs_raw, qrs_on, train, pr_ms, pr_ok, sel, fs)
    t = qrs_on / fs
    minute_pr = minute_series(t, pr_ms, pr_ok, minute_s)
    trend = pr_trend(minute_pr)
    blocks = find_blocks(train, qrs_on, pr_ms, pr_ok, fs, dur,
                         n_leads_req=n_leads_req)
    out = AVResult(pr_ms=pr_ms, pr_valid=pr_ok,
                   p_on=np.asarray([int(p_info.onset[i]) for i in range(len(p_info))], dtype=int),
                   qrs_on=qrs_on, qrs_on_raw=qrs_raw, minute_pr=minute_pr,
                   trend=trend, blocks=blocks, train=train)
    out.summary = av_summary(out, p_info, train, peaks.size)
    return out


def av_summary(res: AVResult, p_info: PInfo | None = None,
               train: PTrain | None = None, n_beats: int = 0) -> dict:
    """Resumen numérico del AV para el informe."""
    pr = np.asarray(res.pr_ms, dtype=np.float64)
    ok = np.asarray(res.pr_valid, dtype=bool) & np.isfinite(pr)
    vals = pr[ok]
    out = {
        "n_beats": int(n_beats),
        "pr_measured": int(vals.size),
        "pr_measured_pct": round(100.0 * vals.size / max(1, n_beats), 1),
        "pr_median_ms": round(float(np.median(vals)), 1) if vals.size else None,
        "pr_iqr_ms": [round(float(np.percentile(vals, 25)), 1),
                      round(float(np.percentile(vals, 75)), 1)] if vals.size else None,
        "pr_p_min_ms": round(float(np.percentile(vals, 5)), 1) if vals.size else None,
        "pr_max_ms": round(float(np.max(vals)), 1) if vals.size else None,
        "first_degree_pct": round(100.0 * float(np.mean(vals > PR_FIRST_DEGREE_MS)), 1)
        if vals.size else None,
        "n_blocks": len(res.blocks),
        "blocks": {k: sum(1 for e in res.blocks if e["kind"] == k) for k in KIND_NAMES},
        "review": True,
    }
    out.update(res.trend)
    if train is not None:
        out["p_train"] = int(len(train))
        out["p_conducted"] = int(train.conducted.sum())
        out["p_blocked"] = int((~train.conducted).sum())
    return out


def block_events(res: AVResult) -> list:
    """Eventos de bloqueo listos para el pipeline (dicts, a revisión)."""
    return [dict(e) for e in res.blocks]
