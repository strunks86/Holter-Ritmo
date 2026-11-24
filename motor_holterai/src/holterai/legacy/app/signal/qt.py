"""Corrección del intervalo QT por frecuencia cardíaca (QTc) y agregados.

Medición **exploratoria**: el QT se estima por retorno de la onda T a la línea
de base (`features.qt_interval_ms`) y las fórmulas de QTc son aproximaciones
poblacionales. No sustituyen la medición del cardiólogo ni el latido medio.

Fórmulas (QT en ms, RR en s, FC en lpm):

* Bazett:      QTc = QT / sqrt(RR)
* Fridericia:  QTc = QT / RR**(1/3)      (por defecto en el resumen)
* Framingham:  QTc = QT + 154 · (1 − 60/FC)
* Hodges:      QTc = QT + 1,75 · (FC − 60)

Reglas del resumen (T129):

* Solo cuentan latidos sinusales estables: sin FA, FC ≤ 120 lpm, sin artefacto
  y con un RR previo dentro de ±10 % de la mediana de los 10 RR anteriores
  (los 10 primeros latidos del registro no tienen referencia y no cuentan).
* Si el QRS es ≥ 120 ms se informa también JT = QT − QRS. Para ese latido las
  comparaciones con umbrales usan JTc (JT corregido con la misma fórmula), de
  modo que el QRS ancho por sí solo no marca QT largo.
* Umbrales: QTc prolongado > 450 ms (hombre) o > 460 ms (mujer); marcado > 500
  ms; corto < 340 ms. Sin sexo indicado se usa 450 ms. Se cuentan minutos (la
  mediana del minuto supera o no el umbral).
* QTc máximo por hora del registro (hora 0 = primeros 3600 s).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

QTC_FORMULAS: tuple[str, ...] = ("bazett", "fridericia", "framingham", "hodges")
DEFAULT_QTC_FORMULA = "fridericia"
QTC_LABELS: dict[str, str] = {
    "bazett": "Bazett (QT/√RR)",
    "fridericia": "Fridericia (QT/RR^⅓)",
    "framingham": "Framingham (QT + 154·(1 − 60/FC))",
    "hodges": "Hodges (QT + 1,75·(FC − 60))",
}
# Umbrales antiguos por minuto (`minutes_over`), que usan los informes actuales.
QTC_THRESHOLDS_MS: tuple[int, ...] = (450, 480, 500)
QTC_PROLONGED_MS: dict[str, int] = {"hombre": 450, "mujer": 460}
QTC_PROLONGED_DEFAULT_MS = 450  # sexo no indicado: el umbral más sensible
QTC_MARKED_MS = 500
QTC_SHORT_MS = 340
WIDE_QRS_MS = 120.0
RR_HISTORY_BEATS = 10
RR_TOLERANCE = 0.10
MAX_HR_BPM = 120.0
MINUTE_S = 60.0
HOUR_S = 3600.0


def normalize_formula(name: str | None) -> str:
    """Devuelve una fórmula válida; si no se reconoce, la de por defecto."""
    key = str(name or "").strip().lower()
    return key if key in QTC_FORMULAS else DEFAULT_QTC_FORMULA


def formula_label(name: str | None) -> str:
    return QTC_LABELS[normalize_formula(name)]


def normalize_sexo(sexo: str | None) -> str | None:
    """'hombre' / 'mujer' (o 'male' / 'female'); vacío → None. Otro valor, error."""
    key = str(sexo or "").strip().lower()
    if not key:
        return None
    alias = {"hombre": "hombre", "male": "hombre", "mujer": "mujer", "female": "mujer"}
    if key not in alias:
        raise ValueError(f"sexo no reconocido: {sexo!r}")
    return alias[key]


def prolonged_threshold_ms(sexo: str | None) -> int:
    """Umbral de QTc prolongado según sexo (450 hombre, 460 mujer, 450 si no consta)."""
    norm = normalize_sexo(sexo)
    return QTC_PROLONGED_MS[norm] if norm else QTC_PROLONGED_DEFAULT_MS


def beat_hr_bpm(rr_s: float) -> float:
    """Frecuencia instantánea (lpm) a partir del RR previo en segundos."""
    rr = float(rr_s)
    if rr <= 0.0:
        raise ValueError("el RR debe ser > 0")
    return 60.0 / rr


def qtc_ms(qt_ms: float, rr_s: float, formula: str = DEFAULT_QTC_FORMULA) -> float:
    """QT corregido por frecuencia en ms.

    ``rr_s`` es el intervalo RR previo al latido medido (s). Lanza ``ValueError``
    si el RR no es positivo o si la fórmula no existe.
    """
    qt = float(qt_ms)
    rr = float(rr_s)
    if rr <= 0.0:
        raise ValueError("el RR debe ser > 0")
    key = str(formula or "").strip().lower()
    hr = 60.0 / rr
    if key == "bazett":
        return qt / math.sqrt(rr)
    if key == "fridericia":
        return qt / (rr ** (1.0 / 3.0))
    if key == "framingham":
        # 60/FC = RR: QTc = QT + 154 · (1 − RR)
        return qt + 154.0 * (1.0 - rr)
    if key == "hodges":
        return qt + 1.75 * (hr - 60.0)
    raise ValueError(f"fórmula de QTc desconocida: {formula!r}")


def jt_ms(qt_ms: float, qrs_ms: float) -> float:
    """JT = QT − QRS (ms)."""
    return float(qt_ms) - float(qrs_ms)


@dataclass(frozen=True)
class QtcSample:
    """Un latido medido: tiempo (s), QT (ms), RR previo (s) y banderas.

    ``qrs_ms`` es opcional (None si no se midió). ``artifact`` marca el latido en
    un tramo con ≥2 de 3 canales malos: no se informa.
    """

    t: float
    qt_ms: float
    rr_s: float
    af: bool = False
    qrs_ms: float | None = None
    artifact: bool = False


def af_mask(times: Sequence[float], segments: Iterable[Sequence[float]]) -> list[bool]:
    """Marca los tiempos que caen dentro de algún segmento (inicio, fin)."""
    segs = [(float(a), float(b)) for a, b in segments]
    out: list[bool] = []
    for t in times:
        tt = float(t)
        out.append(any(a <= tt <= b for a, b in segs))
    return out


def rr_is_stable(history: Sequence[float], rr_s: float) -> bool:
    """RR dentro de ±10 % de la mediana de los RR anteriores (todos > 0)."""
    vals = [float(x) for x in history]
    if any((not math.isfinite(x)) or x <= 0.0 for x in vals):
        return False
    med = float(np.median(vals))
    return abs(float(rr_s) - med) <= RR_TOLERANCE * med


def _bin_medians(bins: np.ndarray, vals: np.ndarray) -> dict[int, float]:
    """Mediana de ``vals`` por bin entero de ``bins``."""
    order = np.argsort(bins, kind="stable")
    sbins = bins[order]
    sval = vals[order]
    uniq, starts = np.unique(sbins, return_index=True)
    ends = list(starts[1:]) + [len(sval)]
    return {
        int(b): float(np.median(sval[starts[k]:ends[k]]))
        for k, b in enumerate(uniq)
    }


def _empty_stats(formula: str, sexo: str | None, thresholds: Sequence[int]) -> dict:
    return {
        "formula": formula,
        "label": QTC_LABELS.get(formula, formula),
        "exploratory": True,
        "sexo": sexo,
        "prolonged_ms": prolonged_threshold_ms(sexo),
        "marked_ms": QTC_MARKED_MS,
        "short_ms": QTC_SHORT_MS,
        "n_beats": 0,
        "n_used": 0,
        "excluded_af": 0,
        "excluded_hr": 0,
        "excluded_invalid": 0,
        "excluded_artifact": 0,
        "excluded_no_history": 0,
        "excluded_unstable": 0,
        "minutes_analyzed": 0,
        "qtc_min_ms": None,
        "qtc_mean_ms": None,
        "qtc_max_ms": None,
        "qtc_max_t_s": None,
        "qtc_max_minute": None,
        "qtc_max_by_hour": {},
        "n_wide_qrs": 0,
        "jt_mean_ms": None,
        "jt_max_ms": None,
        "thresholds_ms": [int(x) for x in thresholds],
        "minutes_over": {str(int(x)): 0 for x in thresholds},
        "minutes_prolonged": 0,
        "minutes_marked": 0,
        "minutes_short": 0,
    }


def qtc_stats(
    samples: Iterable[QtcSample],
    formula: str = DEFAULT_QTC_FORMULA,
    *,
    sexo: str | None = None,
    thresholds_ms: Sequence[int] = QTC_THRESHOLDS_MS,
    max_hr_bpm: float = MAX_HR_BPM,
    bin_s: float = MINUTE_S,
    hour_s: float = HOUR_S,
) -> dict:
    """Resumen de QTc: mín/medio/máx, máximo por hora, JT y minutos sobre umbral.

    Se excluyen: latidos en FA, FC > ``max_hr_bpm``, artefacto, RR no válido, los
    10 primeros latidos (sin referencia) y los de RR no estable (fuera de ±10 %
    de la mediana de los 10 RR anteriores). Los QRS ≥ 120 ms usan JTc.

    Los minutos se cuentan por bins de ``bin_s``: un minuto suma si la mediana de
    sus latidos supera (o baja de) el umbral. ``minutes_over`` conserva los
    umbrales antiguos (450/480/500) para los informes existentes.
    """
    f = str(formula or "").strip().lower()
    if f not in QTC_FORMULAS:
        raise ValueError(f"fórmula de QTc desconocida: {formula!r}")
    sx = normalize_sexo(sexo)
    prolonged = prolonged_threshold_ms(sx)
    # Orden temporal: la referencia de RR son los latidos anteriores en el tiempo.
    rows = sorted(samples, key=lambda s: float(s.t))
    if not rows:
        return _empty_stats(f, sx, thresholds_ms)

    rr_all = [float(s.rr_s) for s in rows]
    n_af = n_hr = n_bad = n_art = n_nohist = n_unstable = 0
    times: list[float] = []
    values: list[float] = []
    jts: list[float] = []
    for i, s in enumerate(rows):
        rr = rr_all[i]
        qt = float(s.qt_ms)
        if not (math.isfinite(rr) and rr > 0.0 and math.isfinite(qt)):
            n_bad += 1
            continue
        if s.artifact:
            n_art += 1
            continue
        if s.af:
            n_af += 1
            continue
        if beat_hr_bpm(rr) > float(max_hr_bpm):
            n_hr += 1
            continue
        if i < RR_HISTORY_BEATS:
            n_nohist += 1
            continue
        if not rr_is_stable(rr_all[i - RR_HISTORY_BEATS:i], rr):
            n_unstable += 1
            continue
        qrs = s.qrs_ms
        wide = qrs is not None and math.isfinite(float(qrs)) and float(qrs) >= WIDE_QRS_MS
        if wide:
            jt = jt_ms(qt, float(qrs))
            if jt <= 0.0:
                n_bad += 1
                continue
            jts.append(jt)
            values.append(qtc_ms(jt, rr, f))
        else:
            values.append(qtc_ms(qt, rr, f))
        times.append(float(s.t))

    counts = dict(
        excluded_af=n_af,
        excluded_hr=n_hr,
        excluded_invalid=n_bad,
        excluded_artifact=n_art,
        excluded_no_history=n_nohist,
        excluded_unstable=n_unstable,
    )
    if not values:
        out = _empty_stats(f, sx, thresholds_ms)
        out.update(n_beats=len(rows), **counts)
        return out

    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    i_max = int(np.argmax(v))
    minute_medians = _bin_medians((t // float(bin_s)).astype(np.int64), v)
    med = list(minute_medians.values())
    hours = (t // float(hour_s)).astype(np.int64)
    by_hour: dict[str, dict] = {}
    for h in np.unique(hours):
        idx = np.nonzero(hours == h)[0]
        j = int(idx[np.argmax(v[idx])])
        by_hour[str(int(h))] = {
            "qtc_max_ms": round(float(v[j]), 1),
            "qtc_max_t_s": round(float(t[j]), 1),
            "n_used": int(idx.size),
        }
    return {
        "formula": f,
        "label": QTC_LABELS[f],
        "exploratory": True,
        "sexo": sx,
        "prolonged_ms": prolonged,
        "marked_ms": QTC_MARKED_MS,
        "short_ms": QTC_SHORT_MS,
        "n_beats": len(rows),
        "n_used": len(values),
        **counts,
        "minutes_analyzed": len(minute_medians),
        "qtc_min_ms": round(float(v.min()), 1),
        "qtc_mean_ms": round(float(v.mean()), 1),
        "qtc_max_ms": round(float(v.max()), 1),
        "qtc_max_t_s": round(float(t[i_max]), 1),
        "qtc_max_minute": int(t[i_max] // float(bin_s)),
        "qtc_max_by_hour": by_hour,
        "n_wide_qrs": len(jts),
        "jt_mean_ms": round(float(np.mean(jts)), 1) if jts else None,
        "jt_max_ms": round(float(np.max(jts)), 1) if jts else None,
        "thresholds_ms": [int(x) for x in thresholds_ms],
        "minutes_over": {
            str(int(thr)): int(sum(1 for m in med if m > float(thr)))
            for thr in thresholds_ms
        },
        "minutes_prolonged": int(sum(1 for m in med if m > prolonged)),
        "minutes_marked": int(sum(1 for m in med if m > QTC_MARKED_MS)),
        "minutes_short": int(sum(1 for m in med if m < QTC_SHORT_MS)),
    }


def qtc_all_formulas(
    samples: Iterable[QtcSample],
    *,
    sexo: str | None = None,
    thresholds_ms: Sequence[int] = QTC_THRESHOLDS_MS,
    max_hr_bpm: float = MAX_HR_BPM,
    bin_s: float = MINUTE_S,
) -> dict[str, dict]:
    """Estadísticas de las cuatro fórmulas (para el selector de la interfaz)."""
    rows = list(samples)
    return {
        name: qtc_stats(rows, name, sexo=sexo, thresholds_ms=thresholds_ms,
                        max_hr_bpm=max_hr_bpm, bin_s=bin_s)
        for name in QTC_FORMULAS
    }
