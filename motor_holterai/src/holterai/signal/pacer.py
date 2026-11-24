"""Ritmo de marcapasos en Holter (T130, sección J de CRITERIOS-TRATADOS).

Pipeline sobre latidos ya detectados (picos, inicio y fin de QRS, en muestras) y espigas:

1. `detect_spikes`: espiga = paso de pendiente >umbral que vuelve a la línea base en
   menos de 2 muestras y no viene precedido de una onda ancha (pendientes altas
   en las muestras anteriores). Se busca por canal y se fusionan las detecciones.
2. `analyze`: clasifica cada espiga como latido estimulado (auricular, ventricular,
   biventricular = QRS estimulado estrecho, fusión), fallo de captura (sin QRS ni P
   en la ventana de captura), fallo de sensado (espiga dentro del QRS o la T propios)
   o pseudofusión (espiga sobre un QRS estrecho nativo); marca pausas mayores que el
   intervalo de escape (RR máximo entre latidos estimulados consecutivos) y descarta
   todo evento en un tramo con ≥2 de 3 canales malos.

Limitación: HScribe trabaja a 180 Hz. Una espiga de 0,5–2 ms cabe en una o dos
muestras y su pendiente queda atenuada; la detección es limitada a 180 Hz y el
resultado lo declara `limitacion` en la salida.

Estado (piloto E246, 24 h, sin referencia): la regla de pulso sigue disparando en
flancos de QRS agudos; el 57 % de las detecciones cae dentro de un QRS o de su T según
HScribe. Hasta tener canal de espigas real (`Spikes.dat`) o validación contra
referencia, el detector no debe usarse para informar marcapasos.

Umbrales: los que fija la tarea T130 (ventana de captura 300 ms, espiga de <2
muestras, ≥2 de 3 canales malos) están en `PacerRules` tal cual. Los anchos de QRS
salen de CRITERIOS §H (110 y 120 ms) y la ventana de sensado llega al fin de T (pico +
QT, §I). Lo que sigue «sin fuente» en `PacerRules` (pendiente de espiga, ventana de
onda ancha previa, cierre de canales) no es clínico hasta validarlo.

Integración (T168): `legacy/app/pipeline.py:_pacer_layer` llama a `analyze` con los
bordes del QRS medidos sin espigas, y solo emite fusión (J2) y fallo de sensado (J4)
para espigas vistas en ≥2 canales.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SPIKES_SIGNATURE = b"\x00\x0a\x00\x01"
LIMIT_HZ = 180.0
LABEL_AURICULAR = "auricular"
LABEL_VENTRICULAR = "ventricular"
LABEL_BIVENTRICULAR = "biventricular"
LABEL_FUSION = "fusion"
LABEL_PSEUDOFUSION = "pseudofusion"
LABEL_CAPTURA = "fallo_captura"
LABEL_SENSADO = "fallo_sensado"
STIMULATED_LABELS = (LABEL_AURICULAR, LABEL_VENTRICULAR, LABEL_BIVENTRICULAR, LABEL_FUSION)


@dataclass(frozen=True)
class PacerRules:
    """Reglas configurables. «T130» = fijado por la tarea; el resto «sin fuente»."""

    # sin fuente: 8 × MAD de la primera diferencia (MIN_RATIO de legacy/app/signal/pacing.py)
    slope_mad_mult: float = 8.0
    # sin fuente: pendientes altas que no deben preceder a una espiga (onda ancha)
    quiet_before_samples: int = 6
    # T130: espiga sin QRS/P en 300 ms = fallo de captura; también ventana de captura
    capture_window_ms: float = 300.0
    # CRITERIOS §H: QRS normal ≤110 ms (QRS anormal >110). Estimulado estrecho = biventricular;
    # nativo estrecho con espiga = pseudofusión
    narrow_qrs_max_ms: float = 110.0
    # CRITERIOS §H: ≥120 ms = bloqueo de rama completo. Estimulado ≥120 = ventricular;
    # entre 110 y 120 ms (zona §H «incompleto») = fusión (proxy por ancho, ver módulo)
    wide_qrs_min_ms: float = 120.0
    # sin fuente, solo si no llega el fin de T (QT) del latido: espiga dentro de los 80 ms
    # posteriores al fin del QRS propio = sensado (con QT se usa pico + QT, §I)
    sensing_t_ms: float = 80.0
    # sin fuente: detecciones de canales distintos a ≤2 muestras se consideran la misma espiga
    channel_merge_samples: int = 2
    # T130: ≥2 de 3 canales malos → no se informa el evento
    max_bad_channels: int = 1
    # sin fuente: media ventana para considerar un canal malo en el instante del evento
    bad_half_window_s: float = 0.5


DEFAULT_RULES = PacerRules()


@dataclass(frozen=True)
class PacedEvent:
    label: str
    spike: int  # muestra de la espiga
    beat: int | None  # muestra del pico del QRS asociado, si lo hay
    dt_ms: float | None  # espiga → inicio del QRS
    width_ms: float | None  # ancho del QRS asociado


@dataclass
class PacerResult:
    fs: float
    limitacion: str | None
    spikes: np.ndarray
    events: list[PacedEvent] = field(default_factory=list)
    suprimidos: int = 0
    escape_ms: float | None = None
    pausas: list[tuple[int, int, float]] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.events:
            out[e.label] = out.get(e.label, 0) + 1
        return out


def _spikes_one_channel(x: np.ndarray, rules: PacerRules) -> np.ndarray:
    d = np.diff(x)
    d = np.where(np.isfinite(d), d, 0.0)
    n_d = d.size
    if n_d < 4:
        return np.zeros(0, dtype=int)
    med = np.median(d)
    scale = 1.4826 * np.median(np.abs(d - med)) + 1e-12
    thr = rules.slope_mad_mult * scale
    big = np.abs(d) > thr
    # índice k de d corresponde a la transición x[k] -> x[k+1]; el pico de la espiga es x[i]
    i_arr = np.arange(1, n_d)  # i: muestra pico, subida d[i-1], bajada d[i]
    up, down = d[i_arr - 1], d[i_arr]
    # x[i+2]-x[i-2] = d[i-2]+d[i-1]+d[i]+d[i+1]: la línea base también vuelve a ±2 muestras
    dp = np.concatenate([np.zeros(2), d, np.zeros(2)])  # dp[k+2] = d[k]
    wide = dp[i_arr] + dp[i_arr + 1] + dp[i_arr + 2] + dp[i_arr + 3]
    pulse = (
        big[i_arr - 1]
        & big[i_arr]
        & (np.sign(up) != np.sign(down))
        & (np.abs(up + down) < thr)  # vuelve a la línea base en 1 muestra
        & (np.abs(wide) < thr)  # y sigue plana a ±2 muestras (no es el ápice de un QRS)
    )
    prior = np.zeros(n_d, dtype=bool)
    for lag in range(2, rules.quiet_before_samples + 2):
        prior[lag:] |= big[:-lag]
    quiet = ~prior[i_arr - 1]  # no hay pendiente alta justo antes: no es el QRS
    return i_arr[pulse & quiet].astype(int)


def detect_spikes(
    x: np.ndarray, fs: float, rules: PacerRules = DEFAULT_RULES
) -> tuple[np.ndarray, np.ndarray]:
    """Detecta espigas en `x` (n o (canales, n)). Devuelve (muestras, canales que la ven)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    cands: list[tuple[int, int]] = []
    for c in range(x.shape[0]):
        cands.extend((int(i), c) for i in _spikes_one_channel(x[c], rules))
    if not cands:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    cands.sort()
    idx: list[int] = []
    support: list[int] = []
    group: list[tuple[int, int]] = []
    for i, c in cands:
        if group and i - group[-1][0] > rules.channel_merge_samples:
            idx.append(group[0][0])
            support.append(len({ch for _, ch in group}))
            group = []
        group.append((i, c))
    idx.append(group[0][0])
    support.append(len({ch for _, ch in group}))
    return np.asarray(idx, dtype=int), np.asarray(support, dtype=int)


def read_spikes_dat(path: str | Path, fs: float) -> np.ndarray:
    """Lee `Spikes.dat` de HScribe (firma 00 0a 00 01 + u32 ms) y lo pasa a muestras."""
    raw = Path(path).read_bytes()
    if raw[:4] != SPIKES_SIGNATURE or (len(raw) - 4) % 4:
        raise ValueError(f"Spikes.dat con firma o tamaño no válidos: {path}")
    ms = np.frombuffer(raw[4:], dtype="<u4").astype(float)
    return np.round(ms / 1000.0 * fs).astype(int)


def _bad_channels_at(bad: np.ndarray, s: int, half: int) -> int:
    lo, hi = max(0, s - half), min(bad.shape[1], s + half + 1)
    return int(np.count_nonzero(bad[:, lo:hi].any(axis=1)))


_NEAR_BEATS = 3  # latidos previos a revisar: un QRS dura <1 s, así que basta con unos pocos


def analyze(
    peak: np.ndarray,
    on: np.ndarray,
    off: np.ndarray,
    spikes: np.ndarray,
    fs: float,
    rules: PacerRules = DEFAULT_RULES,
    p_idx: np.ndarray | None = None,
    bad: np.ndarray | None = None,
    t_end: np.ndarray | None = None,
) -> PacerResult:
    """Clasifica espigas y latidos. Índices en muestras; `bad` es (canales, n) booleano.

    `t_end`: fin de la onda T por latido (muestras; p. ej. pico + QT). Si falta, la
    ventana de sensado es el fin del QRS + `rules.sensing_t_ms`.
    """
    spikes = np.sort(np.asarray(spikes, dtype=int))
    peak = np.asarray(peak, dtype=int)
    on = np.asarray(on, dtype=int)
    off = np.asarray(off, dtype=int)
    order = np.argsort(on, kind="stable")
    peak, on, off = peak[order], on[order], off[order]
    t_post = rules.sensing_t_ms / 1000.0 * fs
    if t_end is None:
        tail = off + t_post                   # sin QT: ventana de T «sin fuente»
        t_qrs_t = off                         # sensado en QRS: hasta el fin del QRS
    else:
        tail = np.asarray(t_end, dtype=float)[order]
        t_qrs_t = tail                        # con QT: sensado en QRS y T hasta el fin de T
    p = np.sort(np.asarray(p_idx, dtype=int)) if p_idx is not None else np.zeros(0, dtype=int)
    width = (off - on) * 1000.0 / fs
    win = rules.capture_window_ms / 1000.0 * fs
    narrow, wide = rules.narrow_qrs_max_ms, rules.wide_qrs_min_ms
    half = max(1, int(round(rules.bad_half_window_s * fs)))

    res = PacerResult(
        fs=fs,
        limitacion=(f"detección limitada a {LIMIT_HZ:.0f} Hz" if fs <= LIMIT_HZ else None),
        spikes=spikes,
    )

    def bad_event(s: int) -> bool:
        return bad is not None and _bad_channels_at(bad, s, half) > rules.max_bad_channels

    def _beat_on_qrs(s: int) -> int | None:
        """Latido cuyo QRS (y T, si hay QT) contiene la muestra s."""
        j = int(np.searchsorted(on, s, side="right")) - 1
        for k in range(j, max(-1, j - _NEAR_BEATS - 1), -1):
            if on[k] <= s <= t_qrs_t[k]:
                return k
        return None

    def _beat_before_t(s: int) -> int | None:
        """Latido cuyo fin de QRS + ventana de T sin QT queda justo antes de s."""
        j = int(np.searchsorted(on, s, side="right")) - 1
        for k in range(j, max(-1, j - _NEAR_BEATS - 1), -1):
            if off[k] < s <= tail[k]:
                return k
        return None

    events: list[PacedEvent] = []
    for s in spikes:
        s = int(s)
        if bad_event(s):
            res.suprimidos += 1
            continue
        # 1) ¿la espiga cae sobre el QRS o la T de un latido? (no hay estimulación de por medio)
        b = _beat_on_qrs(s)
        if b is not None:
            k_prev = int(np.searchsorted(spikes, on[b] - win, side="left"))
            prior_spike = k_prev < spikes.size and spikes[k_prev] < on[b]
            if width[b] <= narrow and not prior_spike:
                label = LABEL_PSEUDOFUSION
            else:
                label = LABEL_SENSADO
            events.append(PacedEvent(label, s, int(peak[b]), None, float(width[b])))
            continue
        b = _beat_before_t(s)
        if b is not None:
            events.append(PacedEvent(LABEL_SENSADO, s, int(peak[b]), None, float(width[b])))
            continue
        # 2) espiga libre: ¿qué sigue dentro de la ventana de captura?
        j = int(np.searchsorted(on, s, side="right"))
        has_beat = j < on.size and on[j] - s <= win
        i_p = int(np.searchsorted(p, s, side="right"))
        has_p = i_p < p.size and p[i_p] - s <= win
        if has_beat:
            dt = float((on[j] - s) * 1000.0 / fs)
            if has_p and p[i_p] < on[j]:
                label = LABEL_AURICULAR
            elif width[j] >= wide:
                label = LABEL_VENTRICULAR
            elif width[j] <= narrow:
                label = LABEL_BIVENTRICULAR
            else:
                label = LABEL_FUSION
            events.append(PacedEvent(label, s, int(peak[j]), dt, float(width[j])))
        elif has_p:
            events.append(PacedEvent(LABEL_AURICULAR, s, None, None, None))
        else:
            events.append(PacedEvent(LABEL_CAPTURA, s, None, None, None))

    # 3) pausas: RR máximo entre latidos estimulados adyacentes = intervalo de escape
    stim_beats = {
        int(e.beat) for e in events if e.label in STIMULATED_LABELS and e.beat is not None
    }
    is_stim = np.array([int(pk) in stim_beats for pk in peak], dtype=bool)
    adj = np.flatnonzero(is_stim[:-1] & is_stim[1:])
    if adj.size:
        rr_all = np.diff(on) * 1000.0 / fs
        res.escape_ms = float(rr_all[adj].max())
        for k in np.flatnonzero(rr_all > res.escape_ms):
            if not bad_event(int(on[k])):
                res.pausas.append((int(on[k]), int(on[k + 1]), float(rr_all[k])))
    res.events = events
    return res


def pacing_report(res: PacerResult) -> dict[str, object]:
    """Resumen agregado (sin muestras de paciente): recuentos por etiqueta y pausas."""
    return {
        "fs": res.fs,
        "limitacion": res.limitacion,
        "espigas": int(res.spikes.size),
        "suprimidas_artefacto": res.suprimidos,
        "etiquetas": res.counts(),
        "escape_ms": res.escape_ms,
        "pausas_mayores_escape": len(res.pausas),
    }
