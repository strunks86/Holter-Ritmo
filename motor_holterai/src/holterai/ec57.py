"""Métricas de detección de latidos al estilo EC57 (sin WFDB).

Solo empareja tiempos de referencia y detectados; no interpreta el ECG.

Este módulo conserva la API histórica (``score_qrs``, ``score_records``,
``run_counts``, ``episode_scores``). La evaluación completa —matriz AAMI con
no detectado/sobrante, rachas 3–5 y ≥6, y episodios por duración— vive en
:mod:`holterai.eval` y se reexporta aquí para no romper a los llamadores:

* ``bxb``/``score_bxb``   → :func:`holterai.eval.bxb.score_bxb`;
* ``rxr``/``score_rxr``   → :func:`holterai.eval.rxr.score_rxr`;
* ``epicmp``/``compare_episodes`` → :func:`holterai.eval.epicmp.compare_episodes`.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from holterai.eval.bxb import score_bxb
from holterai.eval.epicmp import compare_episodes
from holterai.eval.rxr import score_rxr

bxb = score_bxb
rxr = score_rxr
epicmp = compare_episodes


@dataclass(frozen=True)
class QrsScore:
    reference_n: int
    detected_n: int
    tp: int
    fp: int
    fn: int
    sensitivity: float
    ppv: float

    def as_dict(self) -> dict:
        return {
            "reference_n": self.reference_n, "detected_n": self.detected_n,
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "sensitivity": round(self.sensitivity, 4),
            "ppv": round(self.ppv, 4),
        }


def score_qrs(reference: Iterable[float], detected: Iterable[float],
              tolerance_s: float = 0.150) -> QrsScore:
    """Empareja por vecino más cercano dentro de la tolerancia."""
    ref = sorted(float(x) for x in reference)
    det = sorted(float(x) for x in detected)
    used: set[int] = set()
    tp = 0
    for t in ref:
        best = None
        for j in range(max(0, _lower_bound(det, t - tolerance_s) - 1),
                       min(len(det), _lower_bound(det, t + tolerance_s) + 1)):
            if j not in used and abs(det[j] - t) <= tolerance_s:
                if best is None or abs(det[j] - t) < abs(det[best] - t):
                    best = j
        if best is not None:
            used.add(best)
            tp += 1
    fp = len(det) - tp
    fn = len(ref) - tp
    se = tp / len(ref) if ref else 0.0
    ppv = tp / len(det) if det else 0.0
    return QrsScore(len(ref), len(det), tp, fp, fn, se, ppv)


def _lower_bound(arr: Sequence[float], x: float) -> int:
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def score_records(records: dict[str, tuple[Sequence[float], Sequence[float]]],
                  tolerance_s: float = 0.150) -> dict:
    """Promedio bruto por registro y promedio de Se/PPV."""
    rows = {name: score_qrs(ref, det, tolerance_s).as_dict()
            for name, (ref, det) in records.items()}
    if not rows:
        return {"records": {}, "mean_sensitivity": 0.0, "mean_ppv": 0.0}
    return {
        "records": rows,
        "mean_sensitivity": round(sum(r["sensitivity"] for r in rows.values()) / len(rows), 4),
        "mean_ppv": round(sum(r["ppv"] for r in rows.values()) / len(rows), 4),
    }


def run_counts(classes: Iterable[str], min_run: int = 2) -> dict[str, int]:
    """Cuenta rachas de V y S; útil para rxr básico."""
    out = {"V": 0, "S": 0}
    prev = None
    n = 0
    for c in classes:
        c = str(c)
        if c == prev:
            n += 1
        else:
            if prev in out and n >= min_run:
                out[prev] += 1
            prev, n = c, 1
    if prev in out and n >= min_run:
        out[prev] += 1
    return out


def episode_scores(reference: Sequence[tuple[float, float]],
                   detected: Sequence[tuple[float, float]],
                   overlap_s: float = 1.0) -> dict:
    """Episodios por solapamiento temporal (estilo epicmp simplificado)."""
    used: set[int] = set()
    tp = 0
    for a, b in reference:
        for j, (c, d) in enumerate(detected):
            if j in used:
                continue
            if min(b, d) - max(a, c) >= overlap_s:
                used.add(j)
                tp += 1
                break
    return {"reference_n": len(reference), "detected_n": len(detected),
            "tp": tp, "fp": len(detected) - tp, "fn": len(reference) - tp}
