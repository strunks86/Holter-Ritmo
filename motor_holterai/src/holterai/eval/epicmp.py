"""epicmp: comparación de episodios (p. ej. fibrilación auricular).

Dos episodios se consideran el mismo si se solapan al menos ``overlap_s``
segundos (por defecto 1 s, criterio habitual para FA). Se informa:

* episodios por episodio: Se y +P, y el detalle de cada uno con su duración;
* duración total y mediana de referencia y detectada, y su cociente;
* estratos por duración (1 min, 5 min, 30 min): cuántos episodios de
  referencia y detectados caen en cada tramo y cuántos emparejaron.

Los episodios se pueden fusionar antes de comparar (``merge_gap_s``) para no
contar dos veces un mismo episodio partido por una pausa corta.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_BINS_S: tuple[tuple[float, float], ...] = (
    (0.0, 60.0),
    (60.0, 300.0),
    (300.0, 1800.0),
    (1800.0, float("inf")),
)


@dataclass(frozen=True)
class Episode:
    start_s: float
    end_s: float
    label: str = "AF"

    def __post_init__(self) -> None:
        if float(self.end_s) < float(self.start_s):
            raise ValueError(
                f"episodio invertido: inicio {self.start_s} > fin {self.end_s}"
            )

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.end_s) - float(self.start_s))

    def overlaps(self, other: Episode, overlap_s: float) -> float:
        return min(self.end_s, other.end_s) - max(self.start_s, other.start_s) >= overlap_s

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_s": round(float(self.start_s), 3),
            "end_s": round(float(self.end_s), 3),
            "duration_s": round(self.duration_s, 3),
            "label": self.label,
        }


def parse_episodes(
    items: Iterable[Episode | Mapping[str, Any] | Sequence[float]],
    default_label: str = "AF",
) -> list[Episode]:
    """Convierte pares/diccionarios en :class:`Episode` ordenados por inicio."""
    out: list[Episode] = []
    for k, item in enumerate(items):
        if isinstance(item, Episode):
            out.append(item)
            continue
        if isinstance(item, Mapping):
            try:
                start = float(item["start_s"])
                end = float(item["end_s"])
            except KeyError as exc:
                raise ValueError(f"episodio {k}: falta {exc.args[0]!r}") from exc
            out.append(Episode(start, end, str(item.get("label", default_label))))
            continue
        seq = list(item)
        if len(seq) < 2:
            raise ValueError(f"episodio {k}: se esperan al menos (inicio, fin)")
        label = str(seq[2]) if len(seq) > 2 else default_label
        out.append(Episode(float(seq[0]), float(seq[1]), label))
    return sorted(out, key=lambda e: (e.start_s, e.end_s))


def merge_episodes(episodes: Iterable[Episode], gap_s: float = 0.0) -> list[Episode]:
    """Fusiona episodios del mismo tipo separados por menos de ``gap_s``.

    Con ``gap_s`` = 0 (por defecto) no se fusiona nada: dos episodios que se
    solapan se dejan tal cual, porque el emparejamiento ya es uno a uno.
    """
    if gap_s < 0:
        raise ValueError("gap_s debe ser >= 0")
    ordered = sorted(episodes, key=lambda e: (e.start_s, e.end_s))
    if gap_s == 0:
        return ordered
    merged: list[Episode] = []
    for ep in ordered:
        if merged and merged[-1].label == ep.label and ep.start_s - merged[-1].end_s <= gap_s:
            last = merged[-1]
            merged[-1] = Episode(last.start_s, max(last.end_s, ep.end_s), last.label)
        else:
            merged.append(ep)
    return merged


def _bin_label(bounds: tuple[float, float]) -> str:
    lo, hi = bounds
    if hi == float("inf"):
        return f">={lo / 60.0:g} min"
    return f"{lo / 60.0:g}-{hi / 60.0:g} min"


def _bin_of(duration_s: float, bins: Sequence[tuple[float, float]]) -> str:
    for bounds in bins:
        if bounds[0] <= duration_s < bounds[1]:
            return _bin_label(bounds)
    return _bin_label(bins[-1])


def _median(values: Iterable[float]) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def compare_episodes(
    ref: Iterable[Episode | Mapping[str, Any] | Sequence[float]],
    det: Iterable[Episode | Mapping[str, Any] | Sequence[float]],
    overlap_s: float = 1.0,
    merge_gap_s: float = 0.0,
    label: str | None = "AF",
    bins: Sequence[tuple[float, float]] = DEFAULT_BINS_S,
) -> dict[str, Any]:
    """Compara episodios por solapamiento temporal y por duración."""
    if overlap_s < 0:
        raise ValueError("overlap_s debe ser >= 0")
    ref_eps = merge_episodes(parse_episodes(ref), merge_gap_s)
    det_eps = merge_episodes(parse_episodes(det), merge_gap_s)
    if label is not None:
        ref_eps = [e for e in ref_eps if e.label == label]
        det_eps = [e for e in det_eps if e.label == label]

    candidates: list[tuple[float, int, int]] = []
    for i, ref_ep in enumerate(ref_eps):
        for j, det_ep in enumerate(det_eps):
            overlap = min(ref_ep.end_s, det_ep.end_s) - max(ref_ep.start_s, det_ep.start_s)
            if overlap >= overlap_s:
                candidates.append((-overlap, i, j))
    candidates.sort()
    used_ref: set[int] = set()
    used_det: set[int] = set()
    pairs: dict[int, int] = {}
    for _, i, j in candidates:
        if i in used_ref or j in used_det:
            continue
        used_ref.add(i)
        used_det.add(j)
        pairs[i] = j

    tp = len(pairs)
    fn = len(ref_eps) - tp
    fp = len(det_eps) - tp
    ref_dur = [e.duration_s for e in ref_eps]
    det_dur = [e.duration_s for e in det_eps]

    by_duration: dict[str, dict[str, int]] = {}
    for bounds in bins:
        key = _bin_label(bounds)
        by_duration[key] = {"n_ref": 0, "n_det": 0, "tp": 0, "fn": 0}
    for i, ep in enumerate(ref_eps):
        key = _bin_of(ep.duration_s, bins)
        by_duration[key]["n_ref"] += 1
        if i in pairs:
            by_duration[key]["tp"] += 1
        else:
            by_duration[key]["fn"] += 1
    for ep in det_eps:
        by_duration[_bin_of(ep.duration_s, bins)]["n_det"] += 1

    detail = []
    for i, ep in enumerate(ref_eps):
        j = pairs.get(i)
        item = {"ref": ep.as_dict(), "matched": j is not None}
        if j is not None:
            item["det"] = det_eps[j].as_dict()
        detail.append(item)
    for j, ep in enumerate(det_eps):
        if j not in used_det:
            detail.append({"ref": None, "det": ep.as_dict(), "matched": False})

    return {
        "label": label,
        "overlap_s": round(float(overlap_s), 3),
        "merge_gap_s": round(float(merge_gap_s), 3),
        "n_ref": len(ref_eps),
        "n_det": len(det_eps),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "se": round(tp / len(ref_eps), 4) if ref_eps else 0.0,
        "ppv": round(tp / len(det_eps), 4) if det_eps else 0.0,
        "ref_duration_s": round(sum(ref_dur), 3),
        "det_duration_s": round(sum(det_dur), 3),
        "duration_ratio": round(sum(det_dur) / sum(ref_dur), 6) if sum(ref_dur) > 0 else 0.0,
        "mean_ref_duration_s": round(sum(ref_dur) / len(ref_dur), 2) if ref_dur else 0.0,
        "median_ref_duration_s": round(_median(ref_dur), 2),
        "mean_det_duration_s": round(sum(det_dur) / len(det_dur), 2) if det_dur else 0.0,
        "median_det_duration_s": round(_median(det_dur), 2),
        "by_duration": by_duration,
        "episodes": detail,
    }
