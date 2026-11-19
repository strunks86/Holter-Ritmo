"""rxr: rachas de latidos ectópicos (dupletas, 3–5 y 6 o más).

Cuenta rachas *consecutivas* de una clase AAMI (por defecto V y S) sobre una
secuencia de latidos en orden temporal:

* ``dupletas``  = rachas de exactamente 2 latidos;
* ``rachas_3_5`` = rachas de 3 a 5 latidos;
* ``rachas_6``   = rachas de 6 o más latidos (equivalente a TV en V).

Los conteos se calculan por separado para la referencia y para lo detectado,
y opcionalmente sobre la secuencia *alineada* (clase detectada de cada latido
de referencia, o ``no_detectado`` si no hubo pareja).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from holterai.eval.bxb import NO_DETECTADO, normalize_class

DEFAULT_TARGETS: tuple[str, ...] = ("V", "S")


@dataclass(frozen=True)
class RunSummary:
    """Rachas de una clase en una secuencia."""

    target: str
    runs: tuple[tuple[int, int], ...] = ()  # (índice inicial, longitud)

    @property
    def couplets(self) -> int:
        return sum(1 for _, length in self.runs if length == 2)

    @property
    def runs_3_5(self) -> int:
        return sum(1 for _, length in self.runs if 3 <= length <= 5)

    @property
    def runs_ge6(self) -> int:
        return sum(1 for _, length in self.runs if length >= 6)

    @property
    def beats_in_runs(self) -> int:
        return sum(length for _, length in self.runs)

    @property
    def longest(self) -> int:
        return max((length for _, length in self.runs), default=0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "dupletas": self.couplets,
            "rachas_3_5": self.runs_3_5,
            "rachas_6": self.runs_ge6,
            "latidos_en_rachas": self.beats_in_runs,
            "racha_maxima": self.longest,
            "runs": [list(r) for r in self.runs],
        }


def scan_runs(classes: Iterable[Any], target: str = "V", min_len: int = 2) -> RunSummary:
    """Rachas consecutivas de ``target`` (al menos ``min_len`` latidos)."""
    wanted = normalize_class(target)
    if wanted is None:
        raise ValueError(f"clase no reconocida: {target!r}")
    if min_len < 1:
        raise ValueError("min_len debe ser >= 1")
    runs: list[tuple[int, int]] = []
    start = 0
    length = 0
    for k, label in enumerate(classes):
        if normalize_class(label) == wanted:
            if length == 0:
                start = k
            length += 1
            continue
        if length >= min_len:
            runs.append((start, length))
        length = 0
    if length >= min_len:
        runs.append((start, length))
    return RunSummary(target=wanted, runs=tuple(runs))


def _counts(summary: RunSummary) -> dict[str, int]:
    return {
        "dupletas": summary.couplets,
        "rachas_3_5": summary.runs_3_5,
        "rachas_6": summary.runs_ge6,
        "latidos_en_rachas": summary.beats_in_runs,
    }


def pool_runs(summaries: Iterable[RunSummary]) -> dict[str, int]:
    """Suma los conteos de varias secuencias (no conserva índices)."""
    total = {"dupletas": 0, "rachas_3_5": 0, "rachas_6": 0, "latidos_en_rachas": 0}
    for summary in summaries:
        for key, value in _counts(summary).items():
            total[key] += int(value)
    return total


def aligned_classes(
    ref_classes: Sequence[Any],
    det_classes: Sequence[Any],
    pairs: Iterable[tuple[int, int]],
) -> list[str]:
    """Clase detectada de cada latido de referencia (o ``no_detectado``)."""
    out = [NO_DETECTADO] * len(ref_classes)
    for i, j in pairs:
        det = normalize_class(det_classes[j])
        if det is not None:
            out[i] = det
    return out


def score_rxr(
    ref_classes: Sequence[Any],
    det_classes: Sequence[Any],
    targets: Sequence[str] = DEFAULT_TARGETS,
    pairs: Iterable[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Compara rachas de referencia y detección por clase.

    Si se pasan los pares del emparejamiento bxb, se añade la secuencia
    alineada (cada latido de referencia con la clase que le asignó el detector).
    """
    ref_seq = [normalize_class(x) or "" for x in ref_classes]
    det_seq = [normalize_class(x) or "" for x in det_classes]
    out: dict[str, Any] = {
        "targets": list(targets),
        "ref": {},
        "det": {},
        "delta": {},
        "ref_n": len(ref_seq),
        "det_n": len(det_seq),
    }
    if pairs is not None:
        out["aligned"] = {}
        aligned = aligned_classes(ref_seq, det_seq, pairs)
    for target in targets:
        ref_sum = scan_runs(ref_seq, target)
        det_sum = scan_runs(det_seq, target)
        out["ref"][ref_sum.target] = ref_sum.as_dict()
        out["det"][det_sum.target] = det_sum.as_dict()
        out["delta"][ref_sum.target] = {
            key: _counts(det_sum)[key] - _counts(ref_sum)[key] for key in _counts(ref_sum)
        }
        if pairs is not None:
            out["aligned"][ref_sum.target] = scan_runs(aligned, target).as_dict()
    return out
