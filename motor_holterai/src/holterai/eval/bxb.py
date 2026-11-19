"""bxb: emparejamiento latido a latido con matriz AAMI y estratos de calidad.

La comparación es la de EC57/AAMI llevada a Python sin WFDB:

* ventana de emparejamiento de 150 ms (±0,150 s), uno a uno;
* matriz de confusión con filas = referencia (N/S/V/F/Q, más ``sobrante``
  para detecciones sin latido de referencia) y columnas = detectado
  (N/S/V/F/Q, más ``no_detectado`` para latidos de referencia sin detección);
* por clase: Se, +P y FPR; y a nivel de detección QRS: Se, +P y FPR brutas
  (sumando todos los registros) y promedio por registro.

Definiciones usadas (se documentan porque EC57 no fija una sola):

* ``Se_c = tp_c / n_ref_c``;
* ``+P_c = tp_c / n_det_c``;
* ``FPR_c = fp_c / (n_ref - n_ref_c)``, es decir, qué fracción de los latidos
  que no son de la clase ``c`` se etiquetó como ``c``;
* a nivel QRS no hay verdaderos negativos: ``FPR_qrs = fp / n_ref``
  (detecciones sobrantes por latido de referencia).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

WINDOW_S = 0.150
AAMI_CLASSES: tuple[str, ...] = ("N", "S", "V", "F", "Q")
NO_DETECTADO = "no_detectado"
SOBRANTE = "sobrante"
_EPS = 1e-9

#: Alias de anotación (MIT-BIH / WFDB / HScribe) -> clase AAMI.
_ALIASES: dict[str, str] = {
    "N": "N", "L": "N", "R": "N", "NORMAL": "N",
    "S": "S", "A": "S", "J": "S", "JUNCTIONAL": "S", "PAC": "S", "SVEB": "S",
    "V": "V", "E": "V", "VENTRICULAR": "V", "PVC": "V", "VEB": "V",
    "F": "F", "FUSION": "F",
    "Q": "Q", "?": "Q", "UNKNOWN": "Q", "UNCLASSIFIABLE": "Q", "PACED": "Q",
}

#: Códigos WFDB de una letra donde la caja cambia la clase AAMI
#: (``e`` escape auricular -> N, ``j`` prematuro unional -> S,
#: ``a`` PAC aberrado -> S, ``f`` fusión -> F).
_CASE_SENSITIVE: dict[str, str] = {"e": "N", "j": "S", "a": "S", "f": "F"}


def normalize_class(label: Any) -> str | None:
    """Devuelve la clase AAMI de ``label`` o ``None`` si no se reconoce.

    Acepta letras y alias WFDB (``L``, ``R``, ``A``, ``a``, ``E``, ``/``…),
    enteros 0–4 en el orden N/S/V/F/Q y las palabras usadas en los informes.
    """
    if label is None:
        return None
    if isinstance(label, bool):
        return None
    if isinstance(label, int):
        return AAMI_CLASSES[label] if 0 <= label < len(AAMI_CLASSES) else None
    text = str(label).strip()
    if not text:
        return None
    if text.lstrip("+-").isdigit():
        value = int(text)
        return AAMI_CLASSES[value] if 0 <= value < len(AAMI_CLASSES) else None
    if text in _CASE_SENSITIVE:
        return _CASE_SENSITIVE[text]
    return _ALIASES.get(text.upper())


def _classes(labels: Iterable[Any], what: str) -> list[str]:
    out: list[str] = []
    for k, label in enumerate(labels):
        cls = normalize_class(label)
        if cls is None:
            raise ValueError(f"{what}[{k}]: clase no reconocida {label!r}")
        out.append(cls)
    return out


def _times(values: Iterable[float], what: str) -> list[float]:
    out = [float(v) for v in values]
    if any(t != t for t in out):
        raise ValueError(f"{what}: hay tiempos NaN")
    return out


# ---------------------------------------------------------------------------
# Emparejamiento
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BeatMatches:
    """Pares (referencia, detectado) y los que quedaron sin pareja."""

    pairs: tuple[tuple[int, int], ...]
    unmatched_ref: tuple[int, ...]
    unmatched_det: tuple[int, ...]
    window_s: float

    @property
    def tp(self) -> int:
        return len(self.pairs)

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_s": self.window_s,
            "tp": len(self.pairs),
            "fn": len(self.unmatched_ref),
            "fp": len(self.unmatched_det),
            "pairs": [list(p) for p in self.pairs],
            "unmatched_ref": list(self.unmatched_ref),
            "unmatched_det": list(self.unmatched_det),
        }


def _lower_bound(arr: Sequence[float], x: float) -> int:
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def match_beats(
    ref_times: Iterable[float],
    det_times: Iterable[float],
    window_s: float = WINDOW_S,
) -> BeatMatches:
    """Empareja referencia y detecciones uno a uno dentro de ±``window_s``.

    Se ordenan los candidatos por ``|Δt|`` y se aceptan de menor a mayor, de
    modo que el resultado es determinista y no depende del orden de entrada.
    """
    if window_s <= 0:
        raise ValueError("window_s debe ser > 0")
    ref = _times(ref_times, "ref_times")
    det = _times(det_times, "det_times")
    ref_order = sorted(range(len(ref)), key=lambda i: (ref[i], i))
    det_order = sorted(range(len(det)), key=lambda j: (det[j], j))
    det_sorted = [det[j] for j in det_order]

    candidates: list[tuple[float, int, int]] = []
    for i in ref_order:
        t = ref[i]
        lo = _lower_bound(det_sorted, t - window_s - _EPS)
        hi = _lower_bound(det_sorted, t + window_s + _EPS)
        for k in range(lo, hi):
            j = det_order[k]
            dt = abs(det[j] - t)
            if dt <= window_s + _EPS:
                candidates.append((dt, i, j))
    candidates.sort()

    used_ref: set[int] = set()
    used_det: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, i, j in candidates:
        if i in used_ref or j in used_det:
            continue
        used_ref.add(i)
        used_det.add(j)
        pairs.append((i, j))
    pairs.sort()
    unmatched_ref = tuple(i for i in range(len(ref)) if i not in used_ref)
    unmatched_det = tuple(j for j in range(len(det)) if j not in used_det)
    return BeatMatches(tuple(pairs), unmatched_ref, unmatched_det, float(window_s))


# ---------------------------------------------------------------------------
# Matriz y métricas
# ---------------------------------------------------------------------------
def empty_matrix() -> dict[str, dict[str, int]]:
    rows = AAMI_CLASSES + (SOBRANTE,)
    cols = AAMI_CLASSES + (NO_DETECTADO,)
    return {r: {c: 0 for c in cols} for r in rows}


def confusion_matrix(
    ref_classes: Sequence[str],
    det_classes: Sequence[str],
    matches: BeatMatches,
) -> dict[str, dict[str, int]]:
    """Matriz filas=referencia (incluye ``sobrante``) y columnas=detectado."""
    matrix = empty_matrix()
    for i, j in matches.pairs:
        matrix[ref_classes[i]][det_classes[j]] += 1
    for i in matches.unmatched_ref:
        matrix[ref_classes[i]][NO_DETECTADO] += 1
    for j in matches.unmatched_det:
        matrix[SOBRANTE][det_classes[j]] += 1
    return matrix


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


@dataclass(frozen=True)
class BxbResult:
    """Resultado bxb de un registro (o del conjunto agrupado)."""

    window_s: float
    matrix: dict[str, dict[str, int]]
    label: str = ""

    # -- derivados ----------------------------------------------------
    @property
    def n_ref(self) -> int:
        return sum(sum(row.values()) for cls, row in self.matrix.items() if cls != SOBRANTE)

    @property
    def n_det(self) -> int:
        return sum(row[cls] for row in self.matrix.values() for cls in AAMI_CLASSES)

    @property
    def n_matched(self) -> int:
        return sum(self.matrix[c][c2] for c in AAMI_CLASSES for c2 in AAMI_CLASSES)

    @property
    def n_undetected(self) -> int:
        return sum(self.matrix[c][NO_DETECTADO] for c in AAMI_CLASSES)

    @property
    def n_surplus(self) -> int:
        return sum(self.matrix[SOBRANTE][c] for c in AAMI_CLASSES)

    def class_stats(self, cls: str) -> dict[str, Any]:
        if cls not in AAMI_CLASSES:
            raise ValueError(f"clase AAMI desconocida: {cls!r}")
        row = self.matrix[cls]
        n_ref = sum(row.values())
        n_det = sum(self.matrix[r][cls] for r in self.matrix)
        tp = row[cls]
        fp = n_det - tp
        fn = n_ref - tp
        return {
            "n_ref": n_ref,
            "n_det": n_det,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "se": _safe_div(tp, n_ref),
            "ppv": _safe_div(tp, n_det),
            "fpr": _safe_div(fp, max(0, self.n_ref - n_ref)),
        }

    def detection(self) -> dict[str, Any]:
        """Métricas a nivel de detección QRS (sin mirar la clase)."""
        n_ref = self.n_ref
        n_det = self.n_det
        tp = self.n_matched
        fp = self.n_surplus
        fn = self.n_undetected
        return {
            "n_ref": n_ref,
            "n_det": n_det,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "se": _safe_div(tp, n_ref),
            "ppv": _safe_div(tp, n_det),
            "fpr": _safe_div(fp, n_ref),
        }

    def classes(self) -> dict[str, dict[str, Any]]:
        return {cls: self.class_stats(cls) for cls in AAMI_CLASSES}

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "window_s": round(self.window_s, 4),
            "matrix": {r: dict(cols) for r, cols in self.matrix.items()},
            "detection": {k: _round(v) for k, v in self.detection().items()},
            "classes": {c: {k: _round(v) for k, v in s.items()} for c, s in self.classes().items()},
        }


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    return value


def _from_matrix(
    matrix: dict[str, dict[str, int]], window_s: float, label: str = ""
) -> BxbResult:
    return BxbResult(window_s=float(window_s), matrix=matrix, label=label)


def score_bxb(
    ref_times: Iterable[float],
    ref_classes: Iterable[Any],
    det_times: Iterable[float],
    det_classes: Iterable[Any],
    window_s: float = WINDOW_S,
    label: str = "",
) -> BxbResult:
    """Empareja y construye la matriz AAMI de un registro."""
    ref_t = _times(ref_times, "ref_times")
    det_t = _times(det_times, "det_times")
    ref_c = _classes(ref_classes, "ref_classes")
    det_c = _classes(det_classes, "det_classes")
    if len(ref_t) != len(ref_c):
        raise ValueError("ref_times y ref_classes no coinciden en longitud")
    if len(det_t) != len(det_c):
        raise ValueError("det_times y det_classes no coinciden en longitud")
    matches = match_beats(ref_t, det_t, window_s)
    return _from_matrix(confusion_matrix(ref_c, det_c, matches), window_s, label)


@dataclass(frozen=True)
class RecordBeats:
    """Latidos de referencia y detectados de un registro."""

    ref_times: Sequence[float]
    ref_classes: Sequence[Any]
    det_times: Sequence[float]
    det_classes: Sequence[Any]

    def score(self, window_s: float = WINDOW_S, label: str = "") -> BxbResult:
        return score_bxb(
            self.ref_times,
            self.ref_classes,
            self.det_times,
            self.det_classes,
            window_s=window_s,
            label=label,
        )


def combine_results(results: Iterable[BxbResult], label: str = "") -> BxbResult:
    """Suma las matrices de varios registros (métricas brutas/agrupadas)."""
    items = list(results)
    if not items:
        return _from_matrix(empty_matrix(), WINDOW_S, label)
    acc = empty_matrix()
    for res in items:
        for row, cols in res.matrix.items():
            for col, value in cols.items():
                acc[row][col] += int(value)
    return _from_matrix(acc, items[0].window_s, label)


def _mean_of(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else 0.0


def mean_metrics(results: Iterable[BxbResult]) -> dict[str, Any]:
    """Promedio por registro de Se/+P/FPR (solo registros con denominador > 0)."""
    items = list(results)
    if not items:
        return {"n_records": 0, "detection": {}, "classes": {}}
    det_rows = [r.detection() for r in items]
    classes: dict[str, dict[str, Any]] = {}
    for cls in AAMI_CLASSES:
        stats = [r.class_stats(cls) for r in items]
        classes[cls] = {
            "se": _mean_of(s["se"] for s in stats if s["n_ref"] > 0),
            "ppv": _mean_of(s["ppv"] for s in stats if s["n_det"] > 0),
            "fpr": _mean_of(s["fpr"] for s in stats if s["n_ref"] > 0),
            "n_records": sum(1 for s in stats if s["n_ref"] > 0),
        }
    return {
        "n_records": len(items),
        "detection": {
            "se": _mean_of(d["se"] for d in det_rows if d["n_ref"]),
            "ppv": _mean_of(d["ppv"] for d in det_rows if d["n_det"]),
            "fpr": _mean_of(d["fpr"] for d in det_rows if d["n_ref"]),
        },
        "classes": classes,
    }


def score_bxb_records(
    records: Mapping[str, RecordBeats], window_s: float = WINDOW_S
) -> dict[str, Any]:
    """Evalúa varios registros y devuelve por registro, agrupado y promedio."""
    per_record = {name: rec.score(window_s, label=name) for name, rec in records.items()}
    pooled = combine_results(per_record.values(), label="agrupado")
    means = mean_metrics(per_record.values())
    return {
        "window_s": float(window_s),
        "records": {name: res.as_dict() for name, res in per_record.items()},
        "pooled": pooled.as_dict(),
        "mean": {
            "n_records": means["n_records"],
            "detection": {k: round(v, 4) for k, v in means["detection"].items()},
            "classes": {
                c: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}
                for c, s in means["classes"].items()
            },
        },
    }


# ---------------------------------------------------------------------------
# Estratos por calidad
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QualityStrata:
    """Cortes de ruido por latido (mV) y nombre del estrato de artefacto."""

    cuts: tuple[float, ...] = (0.04, 0.08)
    names: tuple[str, ...] = ("limpio", "medio", "ruidoso")
    artifact_name: str = "artefacto"

    def __post_init__(self) -> None:
        if len(self.names) != len(self.cuts) + 1:
            raise ValueError("names debe tener un elemento más que cuts")
        if list(self.cuts) != sorted(self.cuts):
            raise ValueError("cuts debe ser creciente")

    def name_for(self, noise: float, artifact: bool = False) -> str:
        if artifact:
            return self.artifact_name
        for cut, name in zip(self.cuts, self.names, strict=False):
            if noise < cut:
                return name
        return self.names[-1]

    def all_names(self) -> tuple[str, ...]:
        return (self.artifact_name, *self.names)

    @classmethod
    def from_cuts(
        cls, cuts: Sequence[float], names: Sequence[str] | None = None
    ) -> QualityStrata:
        """Estratos a partir de los cortes en mV, con nombres automáticos.

        Los cortes del producto son ``(0.04, 0.08)``: el pipeline desplegado
        llama "ruidoso" a un latido con ``noise > 0.08`` mV.
        """
        values = tuple(float(c) for c in cuts)
        if not values:
            raise ValueError("hace falta al menos un corte de ruido")
        if names is not None:
            return cls(cuts=values, names=tuple(names))
        if len(values) == 2:
            auto = ("limpio", "medio", "ruidoso")
        else:
            auto = ("limpio", *(f"nivel {k}" for k in range(1, len(values))), "ruidoso")
        return cls(cuts=values, names=auto)


def assign_stratum(
    noise: float,
    artifact: bool = False,
    strata: QualityStrata | None = None,
) -> str:
    """Estrato de calidad de un latido detectado (ruido en mV)."""
    return (strata or QualityStrata()).name_for(float(noise), bool(artifact))


SIN_MEDICION = "sin_medicion"


def score_bxb_by_stratum(
    ref_times: Iterable[float],
    ref_classes: Iterable[Any],
    det_times: Iterable[float],
    det_classes: Iterable[Any],
    det_noise: Sequence[float] | None = None,
    det_artifact: Sequence[bool] | None = None,
    strata: QualityStrata | None = None,
    window_s: float = WINDOW_S,
) -> dict[str, BxbResult]:
    """bxb separado por estrato de calidad del latido detectado.

    El estrato se toma del latido detectado; los latidos de referencia sin
    detección no tienen calidad medible y van a ``sin_medicion``.
    """
    conf = strata or QualityStrata()
    det_t = _times(det_times, "det_times")
    det_c = _classes(det_classes, "det_classes")
    ref_t = _times(ref_times, "ref_times")
    ref_c = _classes(ref_classes, "ref_classes")
    if det_noise is None:
        raise ValueError("det_noise es obligatorio para estratificar")
    noise = [float(v) for v in det_noise]
    if len(noise) != len(det_t):
        raise ValueError("det_noise y det_times no coinciden en longitud")
    artifact = list(det_artifact) if det_artifact is not None else [False] * len(det_t)
    if len(artifact) != len(det_t):
        raise ValueError("det_artifact y det_times no coinciden en longitud")

    matches = match_beats(ref_t, det_t, window_s)
    buckets: dict[str, dict[str, dict[str, int]]] = {
        name: empty_matrix() for name in (*conf.all_names(), SIN_MEDICION)
    }
    for i, j in matches.pairs:
        name = conf.name_for(noise[j], bool(artifact[j]))
        buckets[name][ref_c[i]][det_c[j]] += 1
    for j in matches.unmatched_det:
        name = conf.name_for(noise[j], bool(artifact[j]))
        buckets[name][SOBRANTE][det_c[j]] += 1
    for i in matches.unmatched_ref:
        buckets[SIN_MEDICION][ref_c[i]][NO_DETECTADO] += 1
    return {
        name: _from_matrix(matrix, window_s, label=name)
        for name, matrix in buckets.items()
        if any(sum(cols.values()) for cols in matrix.values())
    }
