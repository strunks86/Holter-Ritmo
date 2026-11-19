"""Medición de segmento ST exploratoria.

Punto J configurable (J+60 o J+80 ms) y episodios con umbrales tomados de los
criterios del examen (`st_depression_uv` / `st_elevation_uv`). Es una medición
orientativa sobre la derivación de análisis: no sustituye la revisión del
latido medio ni un análisis de ST validado.

El módulo no depende de los modelos de la aplicación (solo numpy) para poder
recalcular episodios a partir de los valores `st_uv` ya medidos por latido,
sin volver a detectar QRS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

# Puntos de medición admitidos (ms después del pico R).
ST_J_CHOICES: tuple[int, ...] = (60, 80)
DEFAULT_J_MS = 60
DEFAULT_DEPRESSION_UV = 100.0
DEFAULT_ELEVATION_UV = 200.0
DEFAULT_MIN_DURATION_S = 60.0
EXPLORATORY_NOTE = "medición exploratoria; no validada clínicamente"

_EPS = 1e-9


def _check_j_ms(j_ms: int) -> int:
    value = int(j_ms)
    if value not in ST_J_CHOICES:
        raise ValueError(
            f"punto J no admitido: {j_ms}; use {ST_J_CHOICES[0]} o {ST_J_CHOICES[1]} ms"
        )
    return value


@dataclass(frozen=True)
class StCriteria:
    """Criterio de ST del examen: punto J, umbrales y duración mínima."""

    j_ms: int = DEFAULT_J_MS
    depression_uv: float = DEFAULT_DEPRESSION_UV
    elevation_uv: float = DEFAULT_ELEVATION_UV
    min_s: float = DEFAULT_MIN_DURATION_S

    def __post_init__(self) -> None:
        object.__setattr__(self, "j_ms", _check_j_ms(self.j_ms))
        object.__setattr__(self, "depression_uv", abs(float(self.depression_uv)))
        object.__setattr__(self, "elevation_uv", abs(float(self.elevation_uv)))
        object.__setattr__(self, "min_s", max(0.0, float(self.min_s)))

    @classmethod
    def from_criteria(cls, raw: Mapping[str, Any] | None = None) -> "StCriteria":
        """Construye el criterio a partir de un dict de criterios del examen."""
        data = dict(raw or {})
        return cls(
            j_ms=int(data.get("st_j_ms", DEFAULT_J_MS)),
            depression_uv=float(data.get("st_depression_uv", DEFAULT_DEPRESSION_UV)),
            elevation_uv=float(data.get("st_elevation_uv", DEFAULT_ELEVATION_UV)),
            min_s=float(data.get("st_min_duration_s", DEFAULT_MIN_DURATION_S)),
        )

    @property
    def j_label(self) -> str:
        return f"J+{self.j_ms} ms"

    def threshold(self, kind: str) -> float:
        return self.elevation_uv if kind == "st_elevation" else self.depression_uv

    def to_dict(self) -> dict[str, Any]:
        return {
            "exploratory": True,
            "note": EXPLORATORY_NOTE,
            "j_ms": int(self.j_ms),
            "j_label": self.j_label,
            "depression_uv": float(self.depression_uv),
            "elevation_uv": float(self.elevation_uv),
            "min_duration_s": float(self.min_s),
        }


# ---------------------------------------------------------------------------
# Medición sobre la señal filtrada
# ---------------------------------------------------------------------------
def _windows(fs: float, j_ms: int) -> tuple[int, int, int, int]:
    """Desplazamientos en muestras: (base_a, base_b, st_j, st_span)."""
    pre = max(1, int(0.20 * fs))
    post = max(1, int(0.08 * fs))
    j = max(1, int(round(j_ms / 1000.0 * fs)))
    span = max(1, int(0.04 * fs))
    return pre, post, j, span


def st_level_uv(xf: np.ndarray, fs: float, idx: int, j_ms: int = DEFAULT_J_MS) -> float:
    """ST en µV respecto del segmento PQ, medido en J+`j_ms` (60 o 80 ms)."""
    pre, post, j_off, span = _windows(fs, _check_j_ms(j_ms))
    a = max(0, int(idx) - pre)
    b = max(a + 1, int(idx) - post)
    j = int(idx) + j_off
    k = min(xf.size, j + span)
    if j >= xf.size or b - a < 3:
        return 0.0
    baseline = float(np.median(xf[a:b]))
    segment = float(np.median(xf[j:k])) if k > j else baseline
    return round((segment - baseline) * 1000.0, 2)


def st_level_batch(xf: np.ndarray, fs: float, idxs: Iterable[int],
                   j_ms: int = DEFAULT_J_MS, chunk: int = 20_000) -> np.ndarray:
    """`st_level_uv` vectorizado; mismo resultado, para muchos latidos."""
    pre, post, j_off, span = _windows(fs, _check_j_ms(j_ms))
    peaks = np.asarray(list(idxs), dtype=np.int64)
    out = np.zeros(peaks.size, dtype=np.float64)
    if peaks.size == 0:
        return out
    n = int(xf.size)
    inner = (peaks - pre >= 0) & (peaks + j_off + span <= n)
    if not bool(inner.all()):
        for i in np.flatnonzero(~inner):
            out[i] = st_level_uv(xf, fs, int(peaks[i]), j_ms)
    pos = np.flatnonzero(inner)
    x64 = np.asarray(xf, dtype=np.float64)
    base_off = np.arange(-pre, -post)
    st_off = np.arange(j_off, j_off + span)
    for start in range(0, pos.size, max(1, int(chunk))):
        sel = pos[start:start + max(1, int(chunk))]
        base = x64[peaks[sel, None] + base_off[None, :]]
        seg = x64[peaks[sel, None] + st_off[None, :]]
        out[sel] = (np.median(seg, axis=1) - np.median(base, axis=1)) * 1000.0
    return np.round(out, 2)


def st_slope_uv_s(xf: np.ndarray, fs: float, idx: int, j_ms: int = 80) -> float:
    """Pendiente ST (µV/s) entre J+20 ms y el punto de medición J+`j_ms`."""
    _check_j_ms(j_ms)
    a = int(idx + 0.02 * fs)
    b = min(xf.size, int(idx + j_ms / 1000.0 * fs))
    if b - a < 4:
        return 0.0
    t = np.arange(b - a, dtype=np.float64) / fs
    y = np.asarray(xf[a:b], dtype=np.float64)
    y = y - float(np.mean(y))
    den = float(np.sum((t - t.mean()) ** 2))
    if den <= _EPS:
        return 0.0
    return round(float(np.sum((t - t.mean()) * y) / den * 1000.0), 1)


# ---------------------------------------------------------------------------
# Episodios a partir de los valores ya medidos por latido
# ---------------------------------------------------------------------------
@dataclass
class StEpisode:
    """Episodio de ST: tipo, canal, duración y desviación máxima."""

    kind: str            # st_depression | st_elevation
    lead: int
    lead_name: str
    start_s: float
    end_s: float
    max_uv: float        # desviación máxima firmada (µV)
    n_beats: int

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.end_s) - float(self.start_s))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "lead": int(self.lead),
            "lead_name": self.lead_name,
            "start_s": round(float(self.start_s), 3),
            "end_s": round(float(self.end_s), 3),
            "duration_s": round(self.duration_s, 3),
            "max_uv": round(float(self.max_uv), 2),
            "max_abs_uv": round(abs(float(self.max_uv)), 2),
            "n_beats": int(self.n_beats),
            "exploratory": True,
        }


def beat_st(beat: Any, lead: int | None = None) -> float | None:
    """Valor ST guardado en un latido (`st_uv` o `st_uv_L<canal>`)."""
    feats = getattr(beat, "features", None) or {}
    key = "st_uv" if lead is None else f"st_uv_L{int(lead)}"
    value = feats.get(key)
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) else None


def classify(value: float, crit: StCriteria) -> str | None:
    """Tipo de desviación de un valor ST según los umbrales del criterio."""
    if value <= -crit.depression_uv:
        return "st_depression"
    if value >= crit.elevation_uv:
        return "st_elevation"
    return None


def episodes(beats: Iterable[Any], crit: StCriteria, lead: int = 0,
             lead_name: str = "") -> list[StEpisode]:
    """Episodios de ST consecutivos que superan los umbrales del criterio."""
    out: list[StEpisode] = []
    current: str | None = None
    start = 0.0
    last_t = 0.0
    extreme = 0.0
    count = 0
    for beat in beats:
        value = beat_st(beat)
        kind = classify(value, crit) if value is not None else None
        t = float(getattr(beat, "t", 0.0))
        if kind != current:
            if current is not None and last_t - start >= crit.min_s:
                out.append(StEpisode(kind=current, lead=int(lead), lead_name=lead_name,
                                     start_s=start, end_s=last_t, max_uv=extreme,
                                     n_beats=count))
            current = kind
            start = t
            extreme = float(value) if kind is not None and value is not None else 0.0
            count = 1 if kind is not None else 0
        elif kind is not None and value is not None:
            extreme = max(extreme, value) if kind == "st_elevation" else min(extreme, value)
            count += 1
        last_t = t
    if current is not None and last_t - start >= crit.min_s:
        out.append(StEpisode(kind=current, lead=int(lead), lead_name=lead_name,
                             start_s=start, end_s=last_t, max_uv=extreme, n_beats=count))
    return out


def event_dicts(eps: Iterable[StEpisode], crit: StCriteria) -> list[dict[str, Any]]:
    """Episodios como eventos de la aplicación (dict), marcados exploratorios."""
    out = []
    for ep in eps:
        name = "Depresión ST" if ep.kind == "st_depression" else "Elevación ST"
        channel = ep.lead_name or f"L{ep.lead + 1}"
        out.append({
            "kind": ep.kind,
            "start_s": float(ep.start_s),
            "end_s": float(ep.end_s),
            "severity": "warn",
            "detail": (f"{name} {ep.max_uv:+.0f} µV · {channel} · "
                       f"{crit.j_label} · {EXPLORATORY_NOTE}"),
            "beats": [],
        })
    return out


def summary_block(beats: Iterable[Any], crit: StCriteria, lead: int = 0,
                  lead_name: str = "", max_table: int = 200) -> dict[str, Any]:
    """Bloque `summary["st"]`: criterio usado, conteos y tabla de episodios."""
    beat_list = list(beats)
    values = [v for v in (beat_st(b) for b in beat_list) if v is not None]
    eps = episodes(beat_list, crit, lead=lead, lead_name=lead_name)
    block = crit.to_dict()
    block.update({
        "lead": int(lead),
        "lead_name": lead_name or f"L{int(lead) + 1}",
        "n_measured": len(values),
        "n_depressions": sum(1 for v in values if classify(v, crit) == "st_depression"),
        "n_elevations": sum(1 for v in values if classify(v, crit) == "st_elevation"),
        "n_episodes": len(eps),
        "episodes": [e.to_dict() for e in eps[:max_table]],
    })
    if values:
        block["median_uv"] = round(float(np.median(values)), 2)
        block["min_uv"] = round(float(np.min(values)), 2)
        block["max_uv"] = round(float(np.max(values)), 2)
    return block


def remeasure(beats: Iterable[Any], xf: np.ndarray, fs: float,
              crit: StCriteria, lead: int = 0, only_normal: bool = True) -> int:
    """Recalcula `st_uv_L<lead>` por latido sobre la señal filtrada (sin redetectar).

    Devuelve cuántos latidos se midieron. Los episodios se reconstruyen después
    con `summary_block`, de modo que cambiar umbrales o punto J no exige volver
    a detectar QRS.
    """
    beat_list = list(beats)
    sel = [b for b in beat_list if (not only_normal) or getattr(b, "cls", "N") == "N"]
    if not sel:
        return 0
    values = st_level_batch(xf, fs, [int(b.idx) for b in sel], crit.j_ms)
    for beat, value in zip(sel, values):
        beat.features[f"st_uv_L{int(lead)}"] = float(value)
        beat.features["st_uv"] = float(value)
    return len(sel)
