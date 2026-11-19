"""Índice de calidad multicanal y máscara de ventanas ilegibles.

Un Holter con tres derivaciones permite algo que un analizador de un canal no
puede: decidir si una ventana de señal es *legible* mirando los tres canales a
la vez. La mayoría de los artefactos (movimiento del paciente, electrodo suelto,
ruido muscular) aparecen en los tres canales al mismo tiempo; entonces no es un
trastorno del ritmo, es ruido, y los modelos deben callarse en vez de inventar
extrasístoles, taquicardias, pausas o fibrilación auricular.

El módulo trabaja en ventanas de ``WIN_S`` segundos (2 s por defecto) y, para
cada canal, calcula un conjunto de índices acotados:

===============  =====================================================
``hf``           ruido de alta frecuencia: RMS de la banda > 40 Hz en µV
                 (ruido muscular, interferencia, espigas) y, además, su
                 relación con el piso del propio canal (``hf_rel``). Se mide
                 sobre la señal **cruda**: el resto del pipeline filtra a
                 0,5-40 Hz y borraría la prueba.
``flat``         desviación típica < ``flat_mv`` o más de ``plateau_frac`` de
                 muestras consecutivas idénticas: canal muerto, electrodo
                 desconectado o señal digitalmente plana.
``sat``          fracción de muestras pegadas al riel del canal
                 (``|x| >= 0,98 * max|x|`` con ``max|x| >= sat_mv``): saturación
                 del amplificador.
``step``         mayor salto de la mediana entre sub-bloques de 0,5 s, o
                 excursión pico a pico de la ventana mayor que ``p2p_mv`` y que
                 ``p2p_rel`` veces la excursión típica del canal (movimiento del
                 electrodo en los dos casos).
``kurt``         curtosis de la ventana muy alta (espigas aisladas gigantes).
``sqi``          bSQI: concordancia F1 entre dos detectores de QRS distintos
                 (sumada sobre 6 s para que 2 s de señal no den una F1 ruidosa).
``rate``         latidos detectados en la ventana muy por encima de la mediana
                 del bloque (espigas que el detector confunde con QRS).
``corr``         correlación con la plantilla del bloque (mediana de los latidos
                 de esos 5 minutos): si ningún latido se parece a la plantilla
                 del paciente, la ventana no es señal cardíaca.
``disc``         desconexión o saturación **sostenida** del canal: la ventana es
                 plana, de meseta digital o pegada al riel durante al menos
                 ``disc_min_s`` segundos. Una extrasístole o un QRS de bajo
                 voltaje no son desconexión; una caída del electrodo sí.
``motion``       artefacto de movimiento: ráfaga de alta frecuencia
                 **sincrónica** en dos o más canales (ruido muscular y de
                 tracción del cable tienen la misma envolvente en todas las
                 derivaciones) que dura al menos ``motion_min_s`` segundos.
===============  =====================================================

Regla multicanal: una ventana es **ilegible** si ``min(2, n_canales)`` canales
o más están malos (con 3 derivaciones: 2 de 3; con 2: las dos; con 1: la única).
Si solo un canal está malo, la ventana se considera legible y ``ArtifactMask.bad``
dice qué canales sirven para analizarla. Además el tramo ilegible debe durar un
mínimo: una extrasístole aislada ensucia su propia ventana (curtosis, bSQI) y no
debe borrarse. El mínimo depende del motivo:

* ``MIN_RUN_S`` (20 s) para el ruido difuso: dos ventanas sucias no cambian la
  lectura del ritmo.
* ``MIN_RUN_HARD_S`` (4 s) para desconexión o saturación: cuatro segundos sin
  señal ya inventan una pausa.
* ``MIN_RUN_MOTION_S`` (8 s) para la ráfaga de movimiento: es el artefacto
  típico de 5-15 s que el detector confunde con una racha de taquicardia.

Los latidos que caen en ventanas ilegibles se etiquetan ``Q`` y no generan
eventos: ni extrasístoles (V/S), ni rachas, ni TV, ni pausas, ni FA, ni
bloqueos. El criterio es el mismo que usa ``signal/multilead.veto_mask`` para
el veto cruzado: con menos de dos canales testigo no se puede afirmar nada.

Los umbrales están en milivoltios (la unidad de ``Recording.samples``) y se
calibraron con MIT-BIH DS2, NSTDB (``em``/``ma`` a 24…0 dB), BUT QDB y los 20
exámenes completos de ``D:\\HolterIA\\full``; ver ``scripts/eval_artifact.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .filters import fir_bandpass, filtfilt, moving_average
from .qrs import detect_qrs

WIN_S = 2.0
BLOCK_S = 300.0
MATCH_S = 0.150          # tolerancia del bSQI entre detectores (s)
MIN_RUN_S = 20.0         # ruido difuso: un artefacto más corto no cambia la lectura
MIN_RUN_HARD_S = 4.0     # desconexión/saturación: 4 s ya inventan una pausa
MIN_RUN_MOTION_S = 8.0   # ráfaga de movimiento
MAX_SPANS = 200          # tramos que se guardan en el resumen

CRITERIA = ("hf", "flat", "sat", "step", "kurt", "rate", "sqi", "corr", "disc", "motion")
METRICS = ("hf", "hf_uv", "hf_rel", "flat", "plateau", "sat", "step", "p2p", "kurt",
           "rate", "sqi", "corr")

_FLAT_BIT = 1 << CRITERIA.index("flat")
_SAT_BIT = 1 << CRITERIA.index("sat")
_DISC_BIT = 1 << CRITERIA.index("disc")
_MOTION_BIT = 1 << CRITERIA.index("motion")


@dataclass(frozen=True)
class Limits:
    """Umbrales del índice de calidad (mV, salvo los adimensionales)."""

    hf_uv: float = 60.0       # RMS de la banda >40 Hz, en µV (ruido muscular)
    hf_rel: float = 8.0       # ... / p20 del propio canal (ventanas más limpias)
    flat_mv: float = 0.015    # 15 µV de desviación típica
    plateau_frac: float = 0.50
    sat_mv: float = 2.0       # por debajo de esto no hay riel que saturar
    sat_frac: float = 0.02
    step_mv: float = 0.80     # salto de la mediana entre sub-bloques de 0,5 s
    p2p_mv: float = 4.00      # excursión pico a pico de la ventana (movimiento)
    p2p_rel: float = 4.00     # ... y veces la excursión mediana del canal
    kurt_max: float = 200.0   # espigas gigantes; en ECG normal no se llega
    sqi_min: float = 0.40     # se informa, no decide (el 2.º detector es frágil)
    rate_max: float = 2.50    # × mediana del bloque
    corr_min: float = 0.25
    disc_min_s: float = 4.0   # desconexión sostenida
    motion_hf_uv: float = 40.0  # piso absoluto de la ráfaga (>40 Hz)
    motion_z: float = 2.50    # ... y veces la mediana del propio canal
    motion_min_s: float = 8.0
    #: ruido de fondo del **mejor** canal por encima del cual el registro entero
    #: se considera degradado (el ruido estacionario no se distingue ventana a
    #: ventana de la línea de base; ver ``quality_mask``)
    global_hf_uv: float = 90.0


# ---------------------------------------------------------------------------
# Detectores y auxiliares
# ---------------------------------------------------------------------------
def _block_median(x: np.ndarray, fs: float, block_s: float = 5.0) -> np.ndarray:
    """Mediana por bloques, replicada a la longitud original."""
    bs = max(4, int(round(block_s * fs)))
    nblk = max(1, int(np.ceil(x.size / bs)))
    pad = nblk * bs - x.size
    xp = np.concatenate([x, np.repeat(x[-1:], pad)]) if pad else x
    med = np.median(xp.reshape(nblk, bs), axis=1)
    return np.repeat(med, bs)[: x.size]


def detect_qrs_alt(x: np.ndarray, fs: float, sens: float = 3.0,
                   p99_frac: float = 0.5, block_s: float = 0.25) -> np.ndarray:
    """Segundo detector de QRS, distinto del de ``qrs.py``.

    Banda 12-30 Hz (la onda T casi no tiene energía ahí), derivada al cuadrado,
    integración de 60 ms y umbral max(sens * mediana local, p99_frac * p99 del
    bloque) evaluado en máximos de bloque de 0,25 s. Devuelve índices de muestra.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < int(0.5 * fs):
        return np.array([], dtype=int)
    y = filtfilt(x, fir_bandpass(fs, 12.0, min(30.0, 0.45 * fs), 101))
    d = np.diff(y, prepend=y[:1])
    e = moving_average(d * d, max(3, int(0.06 * fs)))
    p99 = float(np.percentile(e, 99.0)) if e.size else 0.0
    thr = np.maximum(sens * _block_median(e, fs), p99_frac * p99)
    blk = max(2, int(round(block_s * fs)))
    nblk = n // blk
    if nblk == 0:
        return np.array([], dtype=int)
    m = e[: nblk * blk].reshape(nblk, blk)
    peak = m.argmax(axis=1)
    val = m[np.arange(nblk), peak]
    tmax = thr[: nblk * blk].reshape(nblk, blk).max(axis=1)
    keep = val > tmax
    return (np.flatnonzero(keep) * blk + peak[keep]).astype(np.int64)


def _match_pairs(a: np.ndarray, b: np.ndarray, tol_s: float, fs: float) -> np.ndarray:
    """Empareja dos listas ordenadas de índices (vecino más cercano voraz)."""
    tol = tol_s * fs
    ia = ib = 0
    out: list[int] = []
    na, nb = a.size, b.size
    while ia < na and ib < nb:
        diff = float(a[ia] - b[ib])
        if abs(diff) <= tol:
            out.append(int(a[ia]))
            ia += 1
            ib += 1
        elif diff < 0:
            ia += 1
        else:
            ib += 1
    return np.asarray(out, dtype=np.int64)


def _window_counts(t, n_win: int, win: int) -> np.ndarray:
    if len(t) == 0:
        return np.zeros(n_win, dtype=float)
    idx = np.clip(np.asarray(t, dtype=np.int64) // win, 0, n_win - 1)
    return np.bincount(idx, minlength=n_win).astype(float)


def _pool3(v: np.ndarray) -> np.ndarray:
    """Suma de la ventana y sus vecinas (±1 ventana): 6 s de contexto."""
    if v.size < 3:
        return v.copy()
    pad = np.concatenate([[v[0]], v, [v[-1]]])
    return np.convolve(pad, np.ones(3), mode="valid")


def _sliding_template_corr(xb: np.ndarray, tpl: np.ndarray, anchor: int) -> np.ndarray:
    """Correlación normalizada de ``tpl`` en cada muestra de ``xb``.

    El valor en la posición ``p`` compara la plantilla (centrada en su muestra
    ``anchor``) con la señal alrededor de ``p``. Se calcula con una convolución
    FFT y una suma acumulada de energías.
    """
    L = tpl.size
    out = np.zeros(xb.size, dtype=np.float32)
    if L == 0 or xb.size < L + 2:
        return out
    num = np.convolve(xb, tpl[::-1])          # num[k] = plantilla terminando en k
    csum = np.concatenate([[0.0], np.cumsum(xb * xb)])
    shift = L - 1 - anchor
    p = np.arange(xb.size)                    # posición del latido
    k = p + shift                             # índice del final de la plantilla
    ok = (p >= anchor) & (k <= xb.size - 1)
    if not ok.any():
        return out
    pp = p[ok]
    kk = k[ok]
    e = csum[kk + 1] - csum[kk + 1 - L]
    c = num[kk] / np.sqrt(np.maximum(e, 1e-12))
    out[pp] = np.clip(np.nan_to_num(c), -1.0, 1.0)
    return out


def _window_stats(m: np.ndarray, fs: float, limits: Limits,
                  rail: float, iso: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Índices por ventana a partir de la matriz (n_ventanas, muestras).

    ``iso`` marca las muestras que **no** son complejo QRS. El ruido de alta
    frecuencia se mide preferentemente ahí (segmento isoelectrico): el QRS tiene
    energía >40 Hz propia y sin esta exclusión una derivación con QRS picudos
    parecería ruidosa. Si la ventana tiene menos de la mitad de muestras
    isoelectricas se usa la ventana completa.
    """
    k, win = m.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = m.mean(axis=1)
        sd = m.std(axis=1)
        taper = np.hanning(win)
        x = (m - mu[:, None]) * taper
        sp = np.abs(np.fft.rfft(x, axis=1)) ** 2
        freqs = np.fft.rfftfreq(win, 1.0 / fs)
        total = sp[:, freqs >= 0.5].sum(axis=1)
        hf_band = freqs > 40.0
        hf = np.where(total > 0, sp[:, hf_band].sum(axis=1) / np.maximum(total, 1e-20), 0.0)
        scale = 2.0 / (win * win * float(np.mean(taper ** 2))) * 1e6   # µV²
        hf_uv = np.sqrt(sp[:, hf_band].sum(axis=1) * scale)
        if iso is not None:
            mi = iso.reshape(k, win)
            n_iso = mi.sum(axis=1)
            enough = n_iso >= win // 2
            if enough.any():
                xi = ((m[enough] - mu[enough, None]) * mi[enough]) * taper
                spi = np.abs(np.fft.rfft(xi, axis=1)) ** 2
                hf_uv[enough] = np.sqrt(spi[:, hf_band].sum(axis=1) * scale)
        plateau = (np.diff(m, axis=1) == 0).mean(axis=1)
        p2p = (m.max(axis=1) - m.min(axis=1)).astype(np.float32)
        sat = ((np.abs(m) >= 0.98 * rail).mean(axis=1) if rail >= limits.sat_mv
               else np.zeros(k))
        sd_safe = np.maximum(sd, 1e-9)
        kurt = ((m - mu[:, None]) ** 4).mean(axis=1) / sd_safe ** 4
        kurt = np.where(sd > limits.flat_mv, kurt, 0.0)
    return {
        "hf": hf.astype(np.float32),
        "hf_uv": hf_uv.astype(np.float32),
        "flat": sd.astype(np.float32),
        "plateau": plateau.astype(np.float32),
        "p2p": p2p,
        "sat": sat.astype(np.float32),
        "kurt": kurt.astype(np.float32),
    }


def _make_template(xb: np.ndarray, fs: float,
                   det: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Plantilla mediana del canal a partir de sus primeros latidos."""
    a = int(round(0.12 * fs))
    b = int(round(0.20 * fs))
    if det.size < 5 or xb.size < a + b + 4:
        return None, a
    starts = det[(det - a >= 0) & (det + b <= xb.size)]
    if starts.size < 5:
        return None, a
    if starts.size > 400:
        starts = starts[np.linspace(0, starts.size - 1, 400).astype(int)]
    stack = np.stack([xb[s - a: s + b] for s in starts])
    tpl = np.median(stack, axis=0).astype(np.float64)
    tpl -= tpl.mean()
    nrm = float(np.linalg.norm(tpl))
    if not np.isfinite(nrm) or nrm <= 1e-9:
        return None, a
    return (tpl / nrm).astype(np.float32), a


def _block_report(body: np.ndarray, fs: float, win: int,
                  limits: Limits) -> dict:
    """Índices y criterios de un bloque alineado a ventanas completas."""
    n_full = body.size // win
    body = body[: n_full * win]
    m = body.reshape(n_full, win)
    rail = float(np.max(np.abs(body))) if body.size else 0.0
    det_a = detect_qrs(body, fs)
    det_b = detect_qrs_alt(body, fs)
    stats = _window_stats(m, fs, limits, rail)
    # saltos de línea base: mediana de sub-bloques de 0,5 s a lo largo de todo
    # el bloque; así también se ven los saltos que caen en el borde de ventana
    sub = max(1, int(round(0.5 * fs)))
    nsub = body.size // sub
    stats["step"] = np.zeros(n_full, dtype=np.float32)
    if nsub >= 2:
        med = np.median(body[: nsub * sub].reshape(nsub, sub), axis=1)
        jump = np.abs(np.diff(med))
        where = np.minimum(n_full - 1, (np.arange(1, nsub) * sub) // win)
        np.maximum.at(stats["step"], where, jump.astype(np.float32))
    flags = {
        "hf": stats["hf_uv"] > limits.hf_uv,
        "flat": (stats["flat"] < limits.flat_mv) | (stats["plateau"] > limits.plateau_frac),
        "sat": stats["sat"] > limits.sat_frac,
        "step": stats["step"] > limits.step_mv,
        "kurt": stats["kurt"] > limits.kurt_max,
    }
    xb = filtfilt(body, fir_bandpass(fs, 5.0, min(18.0, 0.40 * fs), 121))
    na = _window_counts(det_a, n_full, win)
    nb = _window_counts(det_b, n_full, win)
    nm = _window_counts(_match_pairs(det_a, det_b, MATCH_S, fs), n_full, win)
    pooled = _pool3(na) + _pool3(nb)
    sqi = np.where(pooled > 0, 2.0 * _pool3(nm) / np.maximum(pooled, 1e-9), 1.0)
    # sin latidos del detector principal en el canal no se puede juzgar la
    # concordancia (una derivación de bajo voltaje no es una ventana ilegible);
    # el bSQI se informa pero no decide: el segundo detector es frágil en
    # derivaciones de bajo voltaje y marcaría como ruido medio Holter
    flags["sqi"] = np.zeros(n_full, dtype=bool)
    expected = float(np.median(na)) if na.size else 0.0
    # >180 lpm en una ventana de 2 s (6 latidos) no es taquicardia: es espurio
    flags["rate"] = na > max(6.0, limits.rate_max * expected)
    stats["rate"] = na.astype(np.float32)
    stats["sqi"] = sqi.astype(np.float32)
    # plantilla del propio bloque: la morfología cambia a lo largo de un Holter
    tpl, anchor = _make_template(xb, fs, det_a)
    corr = np.ones(n_full, dtype=np.float32)
    flags["corr"] = np.zeros(n_full, dtype=bool)
    if tpl is not None:
        c = _sliding_template_corr(xb, tpl, anchor)
        bounds = np.searchsorted(det_a, np.arange(n_full + 1) * win)
        for w in range(n_full):
            a, b = int(bounds[w]), int(bounds[w + 1])
            if b > a:
                pos = det_a[a:b]
                pos = pos[(pos >= 0) & (pos < c.size)]
                if pos.size:
                    corr[w] = float(np.max(c[pos]))
        flags["corr"] = corr < limits.corr_min
    stats["corr"] = corr
    return {**stats, "__flags__": flags}


# ---------------------------------------------------------------------------
# Máscara
# ---------------------------------------------------------------------------
@dataclass
class ArtifactMask:
    """Resultado de :func:`quality_mask`."""

    fs: float
    win_s: float
    n_windows: int
    n_leads: int
    bad: np.ndarray                      # (n_ventanas, n_canales) bool
    unreadable: np.ndarray               # (n_ventanas,) bool
    reasons: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.uint16))
    spans: list[dict] = field(default_factory=list)
    unreadable_s: float = 0.0
    metrics: dict[str, np.ndarray] = field(default_factory=dict)
    #: segundos ilegibles por motivo (``disc``/``motion``/``noise``)
    by_reason_s: dict = field(default_factory=dict)

    # -- consultas -----------------------------------------------------------
    @property
    def bad_pct(self) -> np.ndarray:
        if self.n_windows == 0:
            return np.zeros(self.n_leads)
        return 100.0 * self.bad.mean(axis=0)

    def usable_leads(self) -> list[int]:
        """Canales malos en menos de la mitad del registro."""
        return [int(i) for i in np.flatnonzero(self.bad_pct < 50.0)]

    def at(self, times_s) -> np.ndarray:
        """¿Cae el instante (o instantes) en una ventana ilegible?"""
        t = np.asarray(times_s, dtype=float).ravel()
        if self.n_windows == 0 or t.size == 0:
            return np.zeros(t.size, dtype=bool)
        idx = np.clip(np.floor(t / self.win_s).astype(int), 0, self.n_windows - 1)
        return self.unreadable[idx]

    def readable_leads(self, t: float) -> list[int]:
        """Canales utilizables en el instante ``t``."""
        if self.n_windows == 0:
            return list(range(self.n_leads))
        i = int(min(self.n_windows - 1, max(0, int(t / self.win_s))))
        return [int(c) for c in np.flatnonzero(~self.bad[i])]

    def as_quality(self, max_spans: int = MAX_SPANS) -> dict:
        """Bloque para ``quality``/``summary`` (sin datos de paciente)."""
        shown = self.spans[:max_spans]
        return {
            "unreadable_s": round(float(self.unreadable_s), 1),
            "unreadable_pct": round(
                100.0 * float(self.unreadable.mean()) if self.n_windows else 0.0, 2),
            "n_unreadable_spans": len(self.spans),
            "unreadable_spans": [{"start_s": s["start_s"], "end_s": s["end_s"]} for s in shown],
            "unreadable_spans_truncated": len(self.spans) > len(shown),
            "per_lead_bad_pct": [round(float(v), 1) for v in self.bad_pct],
            "unreadable_by_reason_s": {k: round(float(v), 1)
                                       for k, v in self.by_reason_s.items()},
            "usable_leads": self.usable_leads(),
            "win_s": self.win_s,
        }


def keep_long_runs(mask: np.ndarray, min_run: int, max_gap: int = 0) -> np.ndarray:
    """Deja solo los tramos ``True`` de al menos ``min_run`` ventanas.

    ``max_gap`` une huecos de hasta esa cantidad de ventanas dentro de un mismo
    tramo: un artefacto real alterna ventanas peores y mejores, y sin tolerancia
    un ruido continuo nunca formaría un tramo lo bastante largo.

    Un artefacto que dura menos que el mínimo no cambia la interpretación del
    ritmo y, en cambio, una extrasístole aislada ensucia su propia ventana
    (curtosis, bSQI): sin esta poda se perderían V reales por ruido de 2 s.
    """
    m = np.asarray(mask, dtype=bool)
    if min_run <= 1 or not m.any():
        return m
    out = np.zeros_like(m)
    i, n = 0, m.size
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i
        filled = 0
        k = i
        while k + 1 < n:
            if m[k + 1]:
                j = k + 1
                filled = 0
            elif filled < max_gap:
                filled += 1
            else:
                break
            k += 1
        if j - i + 1 >= min_run:
            out[i: j + 1] = True
        i = j + 1
    return out


def spans_of(mask: np.ndarray, win_s: float) -> list[dict]:
    """Convierte una máscara por ventana en tramos continuos ``start_s/end_s``."""
    m = np.asarray(mask, dtype=bool)
    spans: list[dict] = []
    i, n = 0, m.size
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and m[j + 1]:
            j += 1
        spans.append({"start_s": round(i * win_s, 2), "end_s": round((j + 1) * win_s, 2)})
        i = j + 1
    return spans


def inside_spans(t0: float, t1: float, spans: list[dict], min_frac: float = 0.5) -> bool:
    """¿El intervalo ``[t0, t1]`` cae mayormente dentro de algún tramo?"""
    dur = max(1e-6, float(t1) - float(t0))
    for s in spans:
        lo = max(float(t0), float(s["start_s"]))
        hi = min(float(t1), float(s["end_s"]))
        if hi - lo >= min_frac * dur:
            return True
    return False


def channel_metrics(x: np.ndarray, fs: float, win_s: float = WIN_S,
                    limits: Limits | None = None) -> dict[str, np.ndarray]:
    """Índices de calidad de un canal, uno por ventana de ``win_s`` segundos.

    Devuelve ``METRICS`` (valor por ventana), ``bad`` (bool por ventana) y
    ``flags`` (bits de ``CRITERIA`` que dispararon).
    """
    limits = limits or Limits()
    x = np.asarray(x, dtype=np.float64).ravel()
    win = max(8, int(round(win_s * fs)))
    n_win = max(1, int(np.ceil(x.size / win))) if x.size else 0
    out: dict[str, np.ndarray] = {k: np.zeros(n_win, np.float32) for k in METRICS}
    out["sqi"] = np.ones(n_win, np.float32)
    out["corr"] = np.ones(n_win, np.float32)
    flags = np.zeros(n_win, dtype=np.uint16)
    if n_win == 0:
        return {**out, "flags": flags, "bad": np.zeros(0, dtype=bool)}
    finite = np.isfinite(x)
    nonfinite_frac = np.zeros(n_win, dtype=np.float32)
    if not finite.all():
        cnt = np.bincount(np.arange(x.size) // win, weights=(~finite).astype(float),
                          minlength=n_win)[:n_win]
        nonfinite_frac = (cnt / win).astype(np.float32)
        x = np.where(finite, x, 0.0)
    blk = max(win, int(round(BLOCK_S * fs / win)) * win)
    for start in range(0, x.size, blk):
        body = x[start: start + blk]
        if body.size < win:
            break
        report = _block_report(body, fs, win, limits)
        crit = report.pop("__flags__")
        k0 = start // win
        n_here = len(next(iter(crit.values())))
        sl = slice(k0, k0 + n_here)
        for key in METRICS:
            if key in report:
                out[key][sl] = np.asarray(report[key], dtype=np.float32)
        for name, val in crit.items():
            flags[sl] |= np.asarray(val, dtype=np.uint16) << CRITERIA.index(name)
    if nonfinite_frac.any():
        flags[nonfinite_frac > 0.10] |= np.uint16(_FLAT_BIT)
    # el ruido de alta frecuencia se mide en µV absolutos (banda >40 Hz) y,
    # además, contra el piso del propio canal. La parte relativa exige al menos
    # la mitad del umbral absoluto: en un canal con piso ínfimo, «8 veces el
    # piso» sigue siendo señal limpia.
    hf_bit = np.uint16(1 << CRITERIA.index("hf"))
    base = float(np.percentile(out["hf"], 20)) if n_win else 0.0
    hf_rel = out["hf"] / max(base, 1e-9)
    out["hf_rel"] = hf_rel.astype(np.float32)
    flags &= ~hf_bit
    flags[(out["hf_uv"] > limits.hf_uv)
          | ((hf_rel > limits.hf_rel) & (out["hf_uv"] > 0.5 * limits.hf_uv))] |= hf_bit
    # excursión de gran amplitud (movimiento del electrodo): umbral absoluto y,
    # además, relativo a la excursión típica del canal, para no marcar como
    # artefacto un registro con QRS de gran voltaje
    step_bit = np.uint16(1 << CRITERIA.index("step"))
    swing_thr = max(limits.p2p_mv, limits.p2p_rel * float(np.percentile(out["p2p"], 50)))
    flags[out["p2p"] > swing_thr] |= step_bit
    # desconexión/saturación sostenida: el canal está muerto o pegado al riel
    dead = ((out["flat"] < limits.flat_mv)
            | (out["plateau"] > limits.plateau_frac)
            | (out["sat"] > limits.sat_frac))
    run_disc = max(1, int(round(float(limits.disc_min_s) / win_s)))
    disc = keep_long_runs(dead, run_disc)
    flags[disc] |= np.uint16(_DISC_BIT)
    return {**out, "flags": flags, "bad": flags != 0}


def motion_windows(metrics: dict[str, np.ndarray], limits: Limits,
                   min_channels: int) -> np.ndarray:
    """Ventanas con ráfaga de alta frecuencia sincrónica en varios canales.

    ``metrics["hf_uv"]`` es (n_ventanas, n_canales). Una ráfaga es un valor muy
    por encima del piso del propio canal (``motion_z``) y del piso absoluto
    (``motion_hf_uv``). Se exige que ocurra en ``min_channels`` canales a la vez:
    el ruido muscular y la tracción del cable se ven igual en todas las
    derivaciones, una extrasístole morfológicamente rara no.
    """
    hf = np.asarray(metrics.get("hf_uv"))
    if hf.ndim != 2 or hf.size == 0:
        return np.zeros(0, dtype=bool)
    base = np.median(hf, axis=0)
    thr = np.maximum(float(limits.motion_hf_uv),
                     float(limits.motion_z) * np.maximum(base, 1e-6))
    burst = hf > thr
    return (burst.sum(axis=1) >= max(1, int(min_channels))).astype(bool)


def quality_mask(samples, fs: float, win_s: float = WIN_S,
                 limits: Limits | None = None,
                 min_bad_channels: int | None = None,
                 min_run_s: float = MIN_RUN_S) -> ArtifactMask:
    """Máscara de artefactos de un registro multicanal.

    ``samples`` es (n_muestras, n_canales) en mV. Una ventana es ilegible si
    ``min_bad_channels`` canales (por defecto ``min(2, n_canales)``) están malos.
    Los tramos ilegibles duran al menos ``min_run_s`` (ruido difuso),
    ``MIN_RUN_HARD_S`` (desconexión/saturación) o ``limits.motion_min_s``
    (movimiento), según el motivo.
    """
    limits = limits or Limits()
    x = np.asarray(samples)
    if x.ndim == 1:
        x = x[:, None]
    n_leads = int(x.shape[1]) if x.ndim == 2 else 0
    win = max(8, int(round(win_s * fs)))
    n_win = max(1, int(np.ceil(x.shape[0] / win))) if x.shape[0] else 0
    if n_leads == 0 or n_win == 0:
        return ArtifactMask(fs=fs, win_s=win_s, n_windows=0, n_leads=0,
                            bad=np.zeros((0, 0), bool), unreadable=np.zeros(0, bool))
    bad = np.zeros((n_win, n_leads), dtype=bool)
    reasons = np.zeros((n_win, n_leads), dtype=np.uint16)
    metrics: dict[str, list[np.ndarray]] = {k: [] for k in METRICS}
    for ch in range(n_leads):
        met = channel_metrics(x[:, ch], fs, win_s, limits)
        bad[:, ch] = met["bad"][:n_win]
        reasons[:, ch] = met["flags"][:n_win]
        for k in METRICS:
            metrics[k].append(np.asarray(met[k][:n_win], dtype=np.float32))
    stacked = {k: np.stack(v, axis=1) if v else np.zeros((n_win, 0), np.float32)
               for k, v in metrics.items()}
    return mask_from_metrics(stacked, bad, reasons, fs=fs, win_s=win_s,
                             limits=limits, min_bad_channels=min_bad_channels,
                             min_run_s=min_run_s)


def mask_from_metrics(stacked: dict[str, np.ndarray], bad: np.ndarray,
                      reasons: np.ndarray, fs: float, win_s: float = WIN_S,
                      limits: Limits | None = None,
                      min_bad_channels: int | None = None,
                      min_run_s: float = MIN_RUN_S) -> ArtifactMask:
    """Máscara a partir de los índices por canal ya calculados.

    Separado de :func:`quality_mask` para poder recalibrar umbrales sin volver
    a leer la señal (``scripts/eval_artifact.py``): ``bad``/``reasons`` son
    (n_ventanas, n_canales) y ``stacked`` los ``METRICS`` apilados igual.
    """
    limits = limits or Limits()
    n_win, n_leads = int(bad.shape[0]), int(bad.shape[1])
    bad = np.array(bad, dtype=bool, copy=True)
    reasons = np.array(reasons, dtype=np.uint16, copy=True)
    need = int(min_bad_channels) if min_bad_channels else min(2, n_leads)
    # --- ruido estacionario: el registro entero está degradado ---------------
    # Un ruido que dura las 24 h no se distingue ventana a ventana de la línea
    # de base del propio canal (no hay "ventana limpia" de referencia). Si el
    # **mejor** canal tiene un piso de alta frecuencia por encima de
    # ``global_hf_uv`` (un piso muy alto, solo para registros inservibles), el
    # registro no permite leer el ritmo y se descarta entero: es preferible
    # "no interpretable" a inventar extrasístoles.
    best_hf = float(np.min(np.median(stacked["hf_uv"], axis=0))) if n_win else 0.0
    degraded = best_hf > float(limits.global_hf_uv)
    # --- artefacto de movimiento: ráfaga sincrónica en varios canales -------
    motion = motion_windows(stacked, limits, need)
    run_motion = max(1, int(round(float(limits.motion_min_s) / win_s)))
    motion = keep_long_runs(motion, run_motion)
    if motion.any():
        burst = stacked["hf_uv"] > np.maximum(
            float(limits.motion_hf_uv),
            float(limits.motion_z) * np.maximum(np.median(stacked["hf_uv"], axis=0), 1e-6))
        for ch in range(n_leads):
            sel = motion & burst[:, ch]
            bad[sel, ch] = True
            reasons[sel, ch] |= np.uint16(_MOTION_BIT)
    # --- tres motivos con su propia duración mínima ------------------------
    count_bad = bad.sum(axis=1)
    count_hard = (bad & ((reasons & np.uint16(_DISC_BIT | _SAT_BIT)) != 0)).sum(axis=1)
    count_motion = (bad & ((reasons & np.uint16(_MOTION_BIT)) != 0)).sum(axis=1)
    hard = count_hard >= need
    mot = count_motion >= need
    other = (count_bad >= need) & ~hard & ~mot
    run_other = max(1, int(np.ceil(float(min_run_s) / win_s))) if min_run_s > 0 else 1
    run_hard = max(1, int(np.ceil(float(MIN_RUN_HARD_S) / win_s)))
    max_gap = 2
    hard_mask = keep_long_runs(hard, run_hard, max_gap)
    mot_mask = keep_long_runs(mot, run_motion, max_gap)
    noise_mask = keep_long_runs(other, run_other, max_gap)
    unreadable = hard_mask | mot_mask | noise_mask
    by_reason = {
        "desc": float(hard_mask.sum()) * win_s,
        "motion": float(mot_mask.sum()) * win_s,
        "noise": float(noise_mask.sum()) * win_s,
    }
    if degraded:
        # todo el registro es ilegible: no hay forma de distinguir el ruido
        unreadable = np.ones(n_win, dtype=bool)
        bad = np.ones_like(bad)
        by_reason = {"global": float(n_win) * win_s}
    return ArtifactMask(
        fs=fs, win_s=win_s, n_windows=n_win, n_leads=n_leads,
        bad=bad, unreadable=unreadable, reasons=reasons,
        spans=spans_of(unreadable, win_s),
        unreadable_s=float(unreadable.sum()) * win_s,
        metrics=stacked, by_reason_s=by_reason,
    )


def bad_from_metrics(stacked: dict[str, np.ndarray], limits: Limits,
                     win_s: float = WIN_S) -> tuple[np.ndarray, np.ndarray]:
    """Recalcula ``bad``/``reasons`` con otros umbrales (calibración).

    Reproduce la regla de :func:`channel_metrics` a partir de los índices ya
    guardados, sin volver a filtrar la señal.
    """
    n_win, n_leads = np.asarray(stacked["hf_uv"]).shape
    flags = np.zeros((n_win, n_leads), dtype=np.uint16)
    hf_bit = np.uint16(1 << CRITERIA.index("hf"))
    flags[(stacked["hf_uv"] > limits.hf_uv)
          | ((stacked["hf_rel"] > limits.hf_rel)
             & (stacked["hf_uv"] > 0.5 * limits.hf_uv))] |= hf_bit
    flat_bit = np.uint16(1 << CRITERIA.index("flat"))
    flags[(stacked["flat"] < limits.flat_mv)
          | (stacked["plateau"] > limits.plateau_frac)] |= flat_bit
    sat_bit = np.uint16(1 << CRITERIA.index("sat"))
    flags[stacked["sat"] > limits.sat_frac] |= sat_bit
    step_bit = np.uint16(1 << CRITERIA.index("step"))
    thr = np.maximum(limits.p2p_mv, limits.p2p_rel * np.median(stacked["p2p"], axis=0))
    flags[(stacked["step"] > limits.step_mv) | (stacked["p2p"] > thr)] |= step_bit
    kurt_bit = np.uint16(1 << CRITERIA.index("kurt"))
    flags[stacked["kurt"] > limits.kurt_max] |= kurt_bit
    rate_bit = np.uint16(1 << CRITERIA.index("rate"))
    rmax = np.maximum(6.0, limits.rate_max * np.median(stacked["rate"], axis=0))
    flags[stacked["rate"] > rmax] |= rate_bit
    corr_bit = np.uint16(1 << CRITERIA.index("corr"))
    flags[stacked["corr"] < limits.corr_min] |= corr_bit
    dead = (((flags & flat_bit) != 0) | ((flags & sat_bit) != 0))
    run_disc = max(1, int(round(float(limits.disc_min_s) / win_s)))
    for ch in range(n_leads):
        flags[keep_long_runs(dead[:, ch], run_disc), ch] |= np.uint16(_DISC_BIT)
    return flags != 0, flags


def flag_names(flags: int) -> list[str]:
    """Nombres de los criterios activos en un byte de ``reasons``."""
    return [name for i, name in enumerate(CRITERIA) if int(flags) & (1 << i)]


__all__ = [
    "ArtifactMask", "CRITERIA", "Limits", "METRICS", "MIN_RUN_HARD_S",
    "MIN_RUN_MOTION_S", "MIN_RUN_S", "WIN_S", "bad_from_metrics",
    "channel_metrics", "detect_qrs_alt", "flag_names", "inside_spans",
    "keep_long_runs", "mask_from_metrics", "motion_windows", "quality_mask",
    "spans_of",
]
