"""Detector QRS multicanal ponderado por calidad (port de la estación 0.1).

Port a numpy de ``prepChannel``, ``localScale``, ``channelQuality`` y
``detectQrs`` de ``referencia/engine.js`` (HolterLocal 0.1), más la fusión por
calidad por segundo de ``E.pass1`` y el bSQI entre dos detectores.

Idea: cada canal se lleva a una señal de análisis de 0,5-40 Hz y a una función
de detección de 5-20 Hz; la función de detección se normaliza por su escala
local y se promedian los canales con un peso que depende de la calidad de cada
segundo (1 buena, 0,4 aceptable, 0,12 inservible, 0 plano/saturado/sin señal).
Sobre la función fusionada corre un detector con umbral adaptativo, refractario
de 200 ms, rechazo de onda T y búsqueda hacia atrás.

El módulo es numpy puro (el paquete legacy no depende de scipy): el filtro
bicuadrado de fase cero del motor JS se reproduce convolucionando con su
respuesta al impulso truncada, arrancando del estado estacionario (lo mismo que
hace ``runBiquad`` al inicializar con ``y0 = G·x0``).

Todavía no se usa desde el pipeline: se compara con ``signal.qrs.detect_qrs``
en ``tests/acceptance/test_t12_qrs_fused.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .filters import fft_conv

TH = 0.30  # umbral relativo por defecto (engine.js: P.th)
T_RATIO = 0.50  # rechazo de onda T: pico < esta fracción del anterior (engine.js: 0.5)
_IR_EPS = 1e-10  # cola relativa de la respuesta al impulso
_IR_RUN = 32  # muestras seguidas por debajo de la cola para cortar
_IR_CAP = 200_000
_SQRT1_2 = math.sqrt(0.5)


# ---------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------
def _rint(x: float) -> int:
    """Redondeo con el criterio de Math.round (mitad hacia arriba)."""
    return int(math.floor(x + 0.5))


def _median(a: np.ndarray | list[float]) -> float:
    arr = np.asarray(a, dtype=np.float64).ravel()
    return float(np.median(arr)) if arr.size else 0.0


def analysis_rate(fs: float) -> tuple[int, float]:
    """Decimación y frecuencia de análisis (engine.js: E.analysisRate)."""
    dec = max(1, _rint(fs / 200.0)) if fs >= 300 else 1
    return dec, fs / dec


def biquad(kind: str, fc: float, fs: float, q: float = _SQRT1_2) -> np.ndarray:
    """Coeficientes [b0, b1, b2, a1, a2] normalizados por a0 (engine.js: biquad)."""
    w0 = 2.0 * math.pi * fc / fs
    c = math.cos(w0)
    s = math.sin(w0)
    al = s / (2.0 * q)
    if kind == "lp":
        b0, b1, b2 = (1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0
    elif kind == "hp":
        b0, b1, b2 = (1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0
    else:  # notch
        b0, b1, b2 = 1.0, -2.0 * c, 1.0
    a0 = 1.0 + al
    return np.array([b0 / a0, b1 / a0, b2 / a0, -2.0 * c / a0, (1.0 - al) / a0])


@lru_cache(maxsize=128)
def _impulse(kind: str, fc: float, fs: float) -> tuple:
    """Respuesta al impulso truncada del bicuadrado, como tupla (cacheable)."""
    b0, b1, b2, a1, a2 = biquad(kind, fc, fs)
    h = np.zeros(_IR_CAP)
    h[0] = b0
    peak = abs(b0)  # se actualiza con el máximo recorrido de la respuesta
    last = _IR_CAP
    h[1] = b1 - a1 * h[0]
    if _IR_CAP > 2:
        h[2] = b2 - a1 * h[1] - a2 * h[0]
    peak = max(peak, abs(h[1]), abs(h[2]))
    run = 0
    for n in range(3, _IR_CAP):
        v = -a1 * h[n - 1] - a2 * h[n - 2]
        h[n] = v
        av = abs(v)
        if av > peak:
            peak = av
        if av < _IR_EPS * peak:
            run += 1
            if run >= _IR_RUN:
                last = n + 1
                break
        else:
            run = 0
    return tuple(np.round(h[:last], 12))


def _run_biquad(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Un pasada del bicuadrado arrancando del estado estacionario.

    Equivale a ``runBiquad`` sin invertir: se antepone una constante igual a la
    primera muestra, que es el estado estacionario que el motor JS impone.
    """
    if x.size == 0:
        return x
    k = h.size
    if k > 1:
        xp = np.concatenate((np.full(k - 1, float(x[0])), x))
    else:
        xp = x
    y = fft_conv(xp, h)
    return y[k - 1: k - 1 + x.size]


def biquad_filtfilt(x: np.ndarray, kind: str, fc: float, fs: float) -> np.ndarray:
    """Filtrado de fase cero, port de ``filtfilt(x, biquad(kind, fc, fs))``."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr
    h = np.asarray(_impulse(kind, round(fc, 6), round(fs, 6)), dtype=np.float64)
    y = _run_biquad(arr, h)
    y = _run_biquad(y[::-1].copy(), h)[::-1]
    return y


def _mov_avg(x: np.ndarray, w: int) -> np.ndarray:
    """Media móvil centrada (engine.js: movAvg), conservando la cola en cero."""
    n = x.size
    w = max(1, int(w))
    h = w >> 1
    out = np.zeros(n)
    if n == 0:
        return out
    cs = np.concatenate(([0.0], np.cumsum(x)))
    i = np.arange(n) + h
    ok = i < n
    lo = np.maximum(0, i - w + 1)
    hi = np.minimum(i + 1, n)
    out[ok] = (cs[hi[ok]] - cs[lo[ok]]) / w
    return out


# ---------------------------------------------------------------------------
# preparación de un canal (engine.js: prepChannel)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ChannelNoise:
    """Métricas de ruido por segundo de un canal."""

    flat: np.ndarray  # línea plana
    sat: np.ndarray  # saturación del conversor
    hf: np.ndarray  # RMS del contenido sobre 40 Hz (µV)
    bw: np.ndarray  # amplitud de la deriva de línea base (µV)
    amp: np.ndarray  # máximo |x| del segundo (µV)


@dataclass(slots=True)
class ChannelPrep:
    """Señal de análisis y función de detección de un canal."""

    ana: np.ndarray  # 0,5-40 Hz a fs_a (µV)
    feat: np.ndarray  # 5-20 Hz, derivada², integración 100 ms, raíz
    noise: ChannelNoise
    fs_a: float
    decim: int
    n: int


def prep_channel(
    raw: np.ndarray,
    fs: float,
    gain: float = 1000.0,
    baseline: float = 0.0,
    decim: int | None = None,
    rails: tuple[float, float] | None = None,
) -> ChannelPrep:
    """Prepara un canal: filtra, calcula la función de detección y la calidad.

    ``raw`` son unidades ADC y ``gain`` son unidades ADC por mV (``gain=1000``
    significa que ``raw`` ya viene en µV). ``rails`` son los extremos del
    conversor para detectar saturación; por defecto se usan los de la señal,
    como hace el motor JS.
    """
    arr = np.asarray(raw, dtype=np.float64).ravel()
    n0 = arr.size
    if decim is None:
        decim, fs_a = analysis_rate(fs)
    else:
        decim = max(1, int(decim))
        fs_a = fs / decim
    n = n0 // decim
    k = 1000.0 / gain
    if rails is None:
        sample = arr[::7] if arr.size else arr
        rmin = float(sample.min()) if sample.size else 0.0
        rmax = float(sample.max()) if sample.size else 0.0
    else:
        rmin, rmax = float(rails[0]), float(rails[1])
    if n == 0:
        empty32 = np.zeros(0, dtype=np.float32)
        zeros = np.zeros(0, dtype=np.float64)
        return ChannelPrep(empty32, zeros, ChannelNoise(
            np.zeros(0, bool), np.zeros(0, bool), zeros, zeros, zeros), fs_a, decim, 0)

    if decim == 1:
        x = (arr[:n] - baseline) * k
    else:
        x = (arr[: n * decim].reshape(n, decim).mean(axis=1) - baseline) * k
    nsec = int(math.ceil(n / fs_a))
    spans = _window_edges(n, fs_a, nsec)

    flat = np.zeros(nsec, dtype=bool)
    sat = np.zeros(nsec, dtype=bool)
    lo_rail = (rmin - baseline) * k + 1e-3
    hi_rail = (rmax - baseline) * k - 1e-3
    for s, (a, b) in enumerate(spans):
        if b <= a:
            continue
        seg = x[a:b]
        if float(seg.max()) - float(seg.min()) < 25.0:
            flat[s] = True
        ns = int(np.count_nonzero((seg <= lo_rail) | (seg >= hi_rail)))
        if ns > 0.04 * (b - a) and (rmax - rmin) > 500:
            sat[s] = True

    base = x.copy()
    y = biquad_filtfilt(x, "hp", 0.5, fs_a)
    bw = np.zeros(nsec)
    for s, (a, b) in enumerate(spans):
        if b > a:
            d = base[a:b] - y[a:b]
            bw[s] = float(d.max()) - float(d.min())

    hpin = y.copy()
    sig = biquad_filtfilt(y, "lp", min(40.0, 0.4 * fs_a), fs_a)
    hf = np.zeros(nsec)
    amp = np.zeros(nsec)
    for s, (a, b) in enumerate(spans):
        if b > a:
            d = hpin[a:b] - sig[a:b]
            hf[s] = math.sqrt(float(np.mean(d * d)))
            amp[s] = float(np.max(np.abs(sig[a:b])))

    feat = biquad_filtfilt(sig, "hp", 5.0, fs_a)
    feat = biquad_filtfilt(feat, "lp", min(20.0, 0.35 * fs_a), fs_a)
    d = np.diff(feat, prepend=feat[:1])
    feat = np.sqrt(_mov_avg(d * d, max(3, _rint(0.1 * fs_a))))

    noise = ChannelNoise(flat=flat, sat=sat, hf=hf, bw=bw, amp=amp)
    return ChannelPrep(ana=sig.astype(np.float32), feat=feat, noise=noise,
                       fs_a=fs_a, decim=decim, n=n)


def _window_edges(n: int, fs_a: float, nsec: int) -> list[tuple[int, int]]:
    """Índices [a, b) por segundo, con el redondeo del motor JS."""
    out = []
    for s in range(nsec):
        a = min(n, max(0, _rint(s * fs_a)))
        b = min(n, _rint((s + 1) * fs_a))
        out.append((a, max(a, b)))
    return out


# ---------------------------------------------------------------------------
# escala local (engine.js: localScale)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class LocalScale:
    sc: np.ndarray  # mediana de los máximos por bloque de 2 s
    blk: int
    mx: np.ndarray  # máximo de la función por bloque


def local_scale(feat: np.ndarray, fs_a: float) -> LocalScale:
    """Mediana móvil (±8 bloques) de los máximos de la función por bloque de 2 s."""
    f = np.asarray(feat, dtype=np.float64).ravel()
    blk = max(1, _rint(2 * fs_a))
    nb = max(1, int(math.ceil(f.size / blk)))
    mx = np.zeros(nb)
    for b in range(nb):
        seg = f[b * blk: min(f.size, (b + 1) * blk)]
        if seg.size:
            mx[b] = float(seg.max())
    sc = np.empty(nb)
    for b in range(nb):
        sc[b] = _median(mx[max(0, b - 8): min(nb - 1, b + 8) + 1])
    return LocalScale(sc=sc, blk=blk, mx=mx)


def channel_scale(feat: np.ndarray, ls: LocalScale) -> np.ndarray:
    """Escala por muestra con el mínimo de 0,25·mediana (no amplificar ruido puro)."""
    sc_all = _median(ls.sc) or 1.0
    idx = np.minimum(np.arange(feat.size) // max(1, ls.blk), ls.sc.size - 1)
    return np.maximum(ls.sc[idx], 0.25 * sc_all)


# ---------------------------------------------------------------------------
# calidad por segundo (engine.js: channelQuality)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ChannelQuality:
    """Calidad por segundo: 0 buena, 1 aceptable (sirve para ritmo), 2 inservible."""

    cls: np.ndarray
    a: float  # amplitud típica del QRS (µV)
    flat: np.ndarray
    sat: np.ndarray
    low: np.ndarray  # señal ausente


def channel_quality(
    noise: ChannelNoise, feat: np.ndarray, ls: LocalScale, fs_a: float
) -> ChannelQuality:
    """Combina plano, saturación, ruido de alta frecuencia, deriva, amplitud y
    ruido dentro de la banda del QRS (cruces y ocupación de la función)."""
    f = np.asarray(feat, dtype=np.float64).ravel()
    nsec = noise.hf.size
    n = f.size
    amps = [float(noise.amp[s]) for s in range(0, nsec, 3) if not noise.flat[s]]
    a = max(150.0, _median(amps))
    scale = channel_scale(f, ls) if n else np.zeros(0)
    v = f / scale if n else np.zeros(0)

    cross = np.zeros(nsec, dtype=np.int64)
    minor = np.zeros(nsec, dtype=np.int64)
    busy = np.zeros(nsec)
    if n:
        # solo los valores > 0.2 pueden cambiar el estado del detector de cruces
        cand = np.flatnonzero(v > 0.2)
        up = False
        pk = 0.0
        for i in cand:
            vi = float(v[i])
            if not up and vi > 0.3:
                up = True
                pk = vi
            elif up:
                if vi > pk:
                    pk = vi
                if vi < 0.2:
                    up = False
                    s = min(nsec - 1, max(0, int(i) // max(1, int(fs_a))))
                    cross[s] += 1
                    if pk < 0.6:
                        minor[s] += 1
        above = np.cumsum(v > 0.3)
        edges = np.concatenate(([0], above))
        for s, (lo, hi) in enumerate(_window_edges(n, fs_a, nsec)):
            if hi > lo:
                busy[s] = (edges[hi] - edges[lo]) / float(hi - lo)

    low = np.zeros(nsec, dtype=bool)
    for s in range(nsec):
        lo = max(0, s - 3)
        hi = min(nsec - 1, s + 3)
        if float(noise.amp[lo: hi + 1].max()) < 0.15 * a:
            low[s] = True

    cls = np.zeros(nsec, dtype=np.uint8)
    for s in range(nsec):
        if noise.flat[s] or noise.sat[s] or low[s]:
            cls[s] = 2
            continue
        hf = float(noise.hf[s]) / a
        bwr = float(noise.bw[s]) / a
        am = float(noise.amp[s]) / a
        nxt = s + 1 if s + 1 < nsec else s
        cr = int(cross[s]) + int(cross[nxt])
        mi = int(minor[s]) + int(minor[nxt])
        if hf > 0.20 or bwr > 6 or am > 6 or cr >= 13 or busy[s] > 0.62:
            cls[s] = 2
        elif hf > 0.08 or bwr > 2.5 or am > 3.5 or cr >= 10 or mi >= 3 or busy[s] > 0.5:
            cls[s] = 1
    low = low | noise.flat
    return ChannelQuality(cls=cls, a=a, flat=noise.flat, sat=noise.sat, low=low)


# ---------------------------------------------------------------------------
# detección sobre la función fusionada (engine.js: detectQrs)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Detection:
    idx: np.ndarray  # muestras (rejilla de análisis)
    h: np.ndarray  # altura de la función en cada latido

    def __len__(self) -> int:
        return int(self.idx.size)


def detect_qrs(
    feat: np.ndarray,
    fs_a: float,
    th: float = TH,
    seg_cls: np.ndarray | None = None,
    t_ratio: float = T_RATIO,
) -> Detection:
    """Detector de QRS con umbral adaptativo sobre la función de detección.

    Refractario de 200 ms, rechazo de onda T (pico < ``t_ratio`` del anterior
    antes de 360 ms) y búsqueda hacia atrás de candidatos bajo umbral cuando un
    intervalo supera 1,5 veces el RR mediano. ``seg_cls`` eleva el umbral por
    segundo (0 buena, 1 aceptable, 2 inservible).
    """
    f = np.asarray(feat, dtype=np.float64).ravel()
    n = f.size
    if n < 3:
        return Detection(np.zeros(0, dtype=np.int64), np.zeros(0))
    refr = max(1, _rint(0.20 * fs_a))
    tw = max(1, _rint(0.36 * fs_a))
    min_dist = max(1, _rint(0.12 * fs_a))

    peaks = np.flatnonzero((f[1:-1] > f[:-2]) & (f[1:-1] >= f[2:]) & (f[1:-1] > 0.08)) + 1
    cand: list[int] = []
    cand_h: list[float] = []
    last_i = -10**9
    last_h = 0.0
    for i in peaks:
        v = float(f[i])
        if i - last_i < min_dist:
            if v > last_h:
                cand[-1] = int(i)
                cand_h[-1] = v
                last_i, last_h = int(i), v
        else:
            cand.append(int(i))
            cand_h.append(v)
            last_i, last_h = int(i), v

    def push8(buf: list[float], val: float) -> None:
        buf.append(val)
        if len(buf) > 8:
            buf.pop(0)

    init = []
    for s in range(8):
        a = min(n, max(0, _rint(s * fs_a)))
        b = min(n, max(a, _rint((s + 1) * fs_a)))
        init.append(float(f[a:b].max()) if b > a else 0.0)
    m0 = _median(init) or 1.0
    q_buf = [m0] * 8
    nz_buf = [m0 * 0.1] * 8
    qm = _median(q_buf)
    nm = _median(nz_buf)
    rr_buf: list[float] = []
    rrm = float(fs_a)
    last_q = -10**9
    last_qh = 0.0
    qrs: list[int] = []
    qh: list[float] = []
    pend: list[tuple[int, float]] = []

    def accept(i: int, h: float) -> None:
        nonlocal last_q, last_qh, qm, rrm, pend
        if last_q > -10**8:
            rr = i - last_q
            if rr < 2.5 * fs_a:
                push8(rr_buf, float(rr))
                rrm = _median(rr_buf)
        qrs.append(i)
        qh.append(h)
        last_q, last_qh = i, h
        push8(q_buf, min(h, 3 * qm))
        qm = max(0.25, _median(q_buf))
        pend = []

    nseg = seg_cls.size if seg_cls is not None else 0
    for i, h in zip(cand, cand_h, strict=True):
        if last_q > -10**8 and i - last_q > 1.5 * rrm and pend:
            thr2 = nm + 0.5 * th * (qm - nm)
            best = -1
            bh = 0.0
            for k, (pi, ph) in enumerate(pend):
                if pi - last_q > tw and ph > thr2 and ph > bh:
                    bh, best = ph, k
            if best >= 0:
                pi = pend[best][0]
                rest = pend[best + 1:]
                accept(pi, bh)
                pend = [r for r in rest if r[0] - pi > refr]
        if i - last_q < refr:
            if h > last_qh * 1.5 and h > qm:
                qrs[-1] = i
                qh[-1] = h
                last_q, last_qh = i, h
            continue
        sec = 0 if not nseg else int(min(nseg - 1, max(0, int(i) // max(1, int(fs_a)))))
        k_thr = 0.62 if sec == 2 else 0.45 if sec == 1 else th
        if h > nm + k_thr * (qm - nm):
            if i - last_q < tw and h < t_ratio * last_qh:
                push8(nz_buf, h)
                nm = _median(nz_buf)
                pend.append((i, h))
                continue
            accept(i, h)
        else:
            push8(nz_buf, h)
            nm = _median(nz_buf)
            pend.append((i, h))
            if len(pend) > 40:
                pend.pop(0)
    return Detection(np.asarray(qrs, dtype=np.int64), np.asarray(qh, dtype=np.float64))


# ---------------------------------------------------------------------------
# fusión multicanal por calidad (engine.js: E.pass1)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class FusedDetection:
    """Resultado de la detección fusionada sobre varios canales."""

    idx: np.ndarray  # latidos sobre la función fusionada (rejilla de análisis)
    fid: np.ndarray  # igual, refinado al máximo de |señal| en ±70 ms
    h: np.ndarray
    fs_a: float
    decim: int
    feat: np.ndarray  # función de detección fusionada
    best: np.ndarray  # 0/1/2: calidad del mejor canal en cada segundo
    cls: list[np.ndarray]  # calidad por segundo de cada canal
    amps: np.ndarray  # amplitud típica de cada canal (µV)
    n: int

    def __len__(self) -> int:
        return int(self.idx.size)

    def fid_samples(self) -> np.ndarray:
        """Fiduciales llevados a índices de la señal de entrada."""
        return self.fid * self.decim if self.decim > 1 else self.fid.copy()

    def bsqi(self, other_idx: np.ndarray, fs: float, window_s: float = 10.0) -> BsqiResult:
        """bSQI contra otro conjunto de latidos (misma señal de entrada)."""
        return bsqi(self.fid_samples(), other_idx, fs, n_samples=int(self.n * self.decim),
                    window_s=window_s)


def detect_fused(
    signals: np.ndarray,
    fs: float,
    gains: np.ndarray | float | None = None,
    baselines: np.ndarray | float | None = None,
    decim: int | None = None,
    th: float = TH,
    refine: bool = True,
    t_ratio: float = T_RATIO,
) -> FusedDetection:
    """Detecta latidos fusionando los canales ponderados por su calidad.

    ``signals`` es (n,) o (n, nch) en unidades ADC; ``gains`` son unidades ADC
    por mV (por defecto 1000, o sea µV). La fusión usa un peso por segundo y
    canal: 1 buena, 0,4 aceptable, 0,12 inservible, 0 plano/saturado/sin señal.
    ``t_ratio`` es el rechazo de onda T (pico menor que esa fracción del anterior
    dentro de 360 ms); 0,5 es el valor de la estación 0.1.

    Ejemplo con un ``Recording`` en mV::

        lat = detect_fused(rec.samples * 1000.0, rec.fs)  # µV, ganancia 1000
        idx = lat.fid_samples()                           # índices de rec.samples
    """
    sig = np.asarray(signals, dtype=np.float64)
    if sig.ndim == 1:
        sig = sig[:, None]
    nch = sig.shape[1]
    if nch == 0 or sig.shape[0] == 0:
        empty = np.zeros(0, dtype=np.int64)
        return FusedDetection(empty, empty, np.zeros(0), float(fs), 1,
                              np.zeros(0), np.zeros(0, dtype=np.uint8), [], np.zeros(0), 0)
    g = np.broadcast_to(np.asarray(1000.0 if gains is None else gains, dtype=np.float64), (nch,))
    bl = np.broadcast_to(np.asarray(0.0 if baselines is None else baselines, dtype=np.float64),
                         (nch,))
    preps: list[ChannelPrep] = []
    scales = []
    quals: list[ChannelQuality] = []
    for c in range(nch):
        p = prep_channel(sig[:, c], fs, gain=float(g[c]), baseline=float(bl[c]), decim=decim)
        ls = local_scale(p.feat, p.fs_a)
        quals.append(channel_quality(p.noise, p.feat, ls, p.fs_a))
        preps.append(p)
        scales.append(ls)
    fs_a = preps[0].fs_a
    dec = preps[0].decim
    n = min(p.n for p in preps)
    nsec = min(int(q.cls.size) for q in quals)
    if n == 0 or nsec == 0:
        empty = np.zeros(0, dtype=np.int64)
        return FusedDetection(empty, empty, np.zeros(0), fs_a, dec, np.zeros(0),
                              np.zeros(0, dtype=np.uint8), [q.cls for q in quals],
                              np.asarray([q.a for q in quals]), 0)
    edges = _window_edges(n, fs_a, nsec)
    acc = np.zeros(n)
    wsum = np.zeros(n)
    for p, ls, q in zip(preps, scales, quals, strict=True):
        bad = q.flat | q.sat | q.low
        w_sec = np.where(bad, 0.0, np.where(q.cls == 2, 0.12, np.where(q.cls == 1, 0.4, 1.0)))
        g_feat = p.feat[:n] / channel_scale(p.feat, ls)[:n]
        for s, (a, b) in enumerate(edges):
            if b <= a or w_sec[s] == 0.0:
                continue
            acc[a:b] += w_sec[s] * g_feat[a:b]
            wsum[a:b] += w_sec[s]
    feat = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1.0), 0.0)
    best = np.full(nsec, 2, dtype=np.uint8)
    for q in quals:
        best = np.minimum(best, q.cls[:nsec])
    det = detect_qrs(feat, fs_a, th=th, seg_cls=best, t_ratio=t_ratio)
    amps = np.asarray([q.a for q in quals], dtype=np.float64)
    fid = det.idx.copy()
    if refine and det.idx.size:
        fid = _refine_fiducial(det.idx, preps, quals, amps, fs_a, n, nsec)
    return FusedDetection(idx=det.idx, fid=fid, h=det.h, fs_a=fs_a, decim=dec, feat=feat,
                          best=best, cls=[q.cls for q in quals], amps=amps, n=n)


def _refine_fiducial(
    idx: np.ndarray,
    preps: list[ChannelPrep],
    quals: list[ChannelQuality],
    amps: np.ndarray,
    fs_a: float,
    n: int,
    nsec: int,
) -> np.ndarray:
    """Punto fiducial: máximo de la desviación absoluta combinada en ±70 ms.

    Es la simplificación de ``E.pass1``: se descartan los canales inservibles en
    el segundo del latido y se busca el máximo de Σ|ana - base|/amplitud.
    """
    ana = np.stack([p.ana[:n].astype(np.float64) for p in preps], axis=1)
    valid = np.stack([q.cls[:nsec] < 2 for q in quals], axis=1).astype(np.float64)
    h70 = max(1, _rint(0.07 * fs_a))
    h150 = max(1, _rint(0.15 * fs_a))
    out = idx.copy()
    for b in range(idx.size):
        p = int(idx[b])
        sec = min(nsec - 1, max(0, int(p) // max(1, int(fs_a))))
        mask = valid[min(sec, valid.shape[0] - 1)]
        lo = max(0, p - h70)
        hi = min(n - 1, p + h70)
        w = ana[lo: hi + 1]
        ba = max(0, p - h150)
        bb = min(n - 1, p + h150)
        base = ana[ba: bb + 1: 2].mean(axis=0)
        score = np.abs(w - base) @ (mask / amps)
        out[b] = lo + int(np.argmax(score))
    return out


# ---------------------------------------------------------------------------
# bSQI entre dos detectores
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class BsqiResult:
    """bSQI por ventana (Li, Zheng y Clifford, 2006)."""

    window_s: float
    tolerance_s: float
    values: np.ndarray  # 2·aciertos/(ref+prueba) por ventana; NaN si no hay latidos
    matches: np.ndarray
    n_ref: np.ndarray
    n_test: np.ndarray

    def mean(self) -> float:
        v = self.values[np.isfinite(self.values)]
        return float(v.mean()) if v.size else float("nan")

    def minimum(self) -> float:
        v = self.values[np.isfinite(self.values)]
        return float(v.min()) if v.size else float("nan")

    def pooled(self) -> float:
        tot = float(self.n_ref.sum() + self.n_test.sum())
        return float(2.0 * self.matches.sum() / tot) if tot else float("nan")


def _greedy_match(ref: np.ndarray, test: np.ndarray, tol: int) -> int:
    i = j = m = 0
    nr, nt = ref.size, test.size
    while i < nr and j < nt:
        d = int(ref[i]) - int(test[j])
        if abs(d) <= tol:
            m += 1
            i += 1
            j += 1
        elif d > tol:
            j += 1
        else:
            i += 1
    return m


def bsqi(
    ref: np.ndarray,
    test: np.ndarray,
    fs: float,
    n_samples: int | None = None,
    window_s: float = 10.0,
    tolerance_s: float = 0.075,
) -> BsqiResult:
    """bSQI: acuerdo entre dos detectores por ventana de ``window_s`` segundos.

    Cada ventana vale 2·aciertos/(latidos_ref + latidos_prueba) con
    emparejamiento a ±``tolerance_s``; las ventanas sin latidos de ninguno de los
    dos se excluyen de la media, como en la definición original.
    """
    a = np.sort(np.asarray(ref if ref is not None else [], dtype=np.int64).ravel())
    b = np.sort(np.asarray(test if test is not None else [], dtype=np.int64).ravel())
    a = a[a >= 0]
    b = b[b >= 0]
    if n_samples is None:
        last = max(int(a[-1]) if a.size else 0, int(b[-1]) if b.size else 0)
        n_samples = last + 1
    n = max(1, int(n_samples))
    win = max(1, _rint(window_s * fs))
    tol = max(1, _rint(tolerance_s * fs))
    nwin = max(1, int(math.ceil(n / win)))
    values = np.full(nwin, np.nan)
    matches = np.zeros(nwin, dtype=np.int64)
    n_ref = np.zeros(nwin, dtype=np.int64)
    n_test = np.zeros(nwin, dtype=np.int64)
    for k in range(nwin):
        lo, hi = k * win, min(n, (k + 1) * win)
        ri = a[np.searchsorted(a, lo, "left"): np.searchsorted(a, hi, "left")]
        ti = b[np.searchsorted(b, lo, "left"): np.searchsorted(b, hi, "left")]
        n_ref[k] = ri.size
        n_test[k] = ti.size
        if ri.size + ti.size == 0:
            continue
        matches[k] = _greedy_match(ri, ti, tol)
        values[k] = 2.0 * matches[k] / float(ri.size + ti.size)
    return BsqiResult(float(window_s), float(tolerance_s), values, matches, n_ref, n_test)


__all__ = [
    "TH",
    "T_RATIO",
    "BsqiResult",
    "ChannelNoise",
    "ChannelPrep",
    "ChannelQuality",
    "Detection",
    "FusedDetection",
    "LocalScale",
    "analysis_rate",
    "biquad",
    "biquad_filtfilt",
    "bsqi",
    "channel_quality",
    "channel_scale",
    "detect_fused",
    "detect_qrs",
    "local_scale",
    "prep_channel",
]
