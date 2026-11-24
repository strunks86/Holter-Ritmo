"""Lector standalone de `Beats.dat` de HScribe (registros de 132 bytes).

Layout verificado contra `HMngExamAPI.GetBeats` en `Exam_996` y en la muestra:
  offset   4  u32  time_ms
  offset   8  u8   label
  offset  12  u8   abs_type
  offset  96  i16  q_on (ms relativo al R)
  offset  98  i16  q_off
  offset 100  i16  qt
  offset 120  u8   morphology
  offset 124  u16  rr
  offset 128  u16  rr16

No interpreta clínicamente: solo lee lo que HScribe dejó escrito.

Sobre la alineación (`match_reference` / `match_windows`):

`time_ms` de `Beats.dat` y `.t` de HolterIA son milisegundos desde el inicio del
examen (verificado 200/200 en `docs/formatos/hscribe_beats.md`), así que el
desfase esperado es 0, o `start_s` si la exportación comenzó más tarde. Por eso
solo se buscan desfases en ±`OFFSET_MAX_S` alrededor de ese valor: un desfase
grande (la vieja búsqueda hasta `ht[0] - rt[0]`) solo puede producir
emparejamientos falsos por azar.

El desfase real se estima buscando solo en `start_s ± OFFSET_MAX_S` con paso
`OFFSET_STEP_S` (`best_offset`); `start_s_from_header` lee del `.hea` el inicio
de la exportación.

Las métricas de sensibilidad y PPV se calculan **solo** en ventanas de
`WINDOW_S` segundos donde HScribe tiene referencia suficiente (≥ 70 % de los
latidos esperados según su RR mediano local). El resto del tiempo se informa
aparte como `tiempo_sin_referencia_s`: en esos tramos HScribe no anotó latidos
(ruido, electrodos recién colocados) y comparar contra una referencia vacía solo
mide el silencio de HScribe, no los errores del detector.
"""

from __future__ import annotations

import re
import statistics
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

RECORD = 132

# Alineación: desfase esperado 0 (o `start_s`), tolerado ±0,5 s.
OFFSET_MAX_S = 0.5
OFFSET_STEP_S = 0.01
TOL_MS = 150.0
# Validación por ventanas.
WINDOW_S = 60.0
MIN_REF_FRACTION = 0.70
MIN_REF_BEATS = 3
RR_PLAUSIBLE_S = (0.25, 2.5)
_RR_CONTEXT_S = 30.0
_SEARCH_SUBSAMPLE = 20_000


@dataclass(frozen=True)
class HscribeBeat:
    time_ms: int
    label: int
    abs_type: int
    q_on: int
    q_off: int
    qt: int
    morphology: int
    rr: int
    rr16: int


def read_beats(path: str | Path) -> list[HscribeBeat]:
    raw = Path(path).read_bytes()
    n = len(raw) // RECORD
    out: list[HscribeBeat] = []
    for i in range(n):
        o = i * RECORD
        out.append(
            HscribeBeat(
                time_ms=int.from_bytes(raw[o + 4 : o + 8], "little"),
                label=raw[o + 8],
                abs_type=raw[o + 12],
                q_on=int.from_bytes(raw[o + 96 : o + 98], "little", signed=True),
                q_off=int.from_bytes(raw[o + 98 : o + 100], "little", signed=True),
                qt=int.from_bytes(raw[o + 100 : o + 102], "little", signed=True),
                morphology=raw[o + 120],
                rr=int.from_bytes(raw[o + 124 : o + 126], "little"),
                rr16=int.from_bytes(raw[o + 128 : o + 130], "little"),
            )
        )
    return out


def summarize(beats: list[HscribeBeat]) -> dict[str, object]:
    labels: dict[str, int] = {}
    for b in beats:
        key = str(b.label)
        labels[key] = labels.get(key, 0) + 1
    return {
        "n_beats": len(beats),
        "labels": dict(sorted(labels.items(), key=lambda kv: int(kv[0]))),
        "first_ms": beats[0].time_ms if beats else None,
        "last_ms": beats[-1].time_ms if beats else None,
        "source": "Beats.dat (HScribe)",
    }


def crop_reference(
    reference: Sequence[object], t_lo_ms: float = 0.0, t_hi_ms: float = float("inf")
) -> list[object]:
    """Latidos de HScribe dentro de ``[t_lo_ms, t_hi_ms]`` (ms absolutos del examen).

    `Beats.dat` cubre el examen completo, pero la exportación WFDB puede ser más
    corta. Recortar la referencia al tramo exportado evita contar como "no
    detectados" latidos que quedan fuera del registro analizado.
    """
    return [b for b in reference if t_lo_ms <= float(getattr(b, "time_ms", 0)) <= t_hi_ms]


def start_s_from_header(wfdb_path: str | Path) -> float:
    """Inicio de la exportación (s desde el comienzo del examen), leído del `.hea`.

    El exportador escribe un comentario `# start_s: <s>`; si no está, la
    exportación empieza en 0. Se usa para acotar la búsqueda del desfase de
    `match_reference` (que solo explora `start_s ± OFFSET_MAX_S`).
    """
    hea = Path(wfdb_path)
    hea = hea if hea.suffix.lower() == ".hea" else hea.with_suffix(".hea")
    try:
        text = hea.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0.0
    hit = re.search(r"start_s\s*[=:]\s*(-?\d+(?:[.,]\d+)?)", text)
    return float(hit.group(1).replace(",", ".")) if hit else 0.0


def _t_of(beat: object) -> float:
    if isinstance(beat, dict):
        return float(beat.get("t", 0.0))
    return float(getattr(beat, "t", 0.0))


def _match_pairs(
    rt: list[float], ht: list[float], tol_s: float
) -> tuple[list[int], list[int], list[float]]:
    """Empareja 1:1 referencia↔propios por cercanía (±tol_s).

    Devuelve índices de referencia, índices propios y `our - ref` en segundos.
    """
    used = [False] * len(rt)
    ri: list[int] = []
    oi: list[int] = []
    dts: list[float] = []
    for i, t in enumerate(ht):
        j = bisect_left(rt, t)
        best = -1
        best_d = tol_s
        for k in (j - 1, j):
            if 0 <= k < len(rt) and not used[k]:
                d = abs(rt[k] - t)
                if d <= best_d:
                    best = k
                    best_d = d
        if best >= 0:
            used[best] = True
            ri.append(i)
            oi.append(best)
            dts.append(rt[best] - t)
    return ri, oi, dts


def _count_matches(rt: list[float], ht: list[float], cand: float, tol_s: float) -> int:
    """Cuenta aproximada de pares para un desfase candidato (vectorizado)."""
    import numpy as np

    if not rt or not ht:
        return 0
    r = np.asarray(rt, dtype=float)
    if len(ht) > _SEARCH_SUBSAMPLE:
        step = len(ht) / _SEARCH_SUBSAMPLE
        h = np.asarray(ht, dtype=float)[(np.arange(_SEARCH_SUBSAMPLE) * step).astype(int)]
    else:
        h = np.asarray(ht, dtype=float)
    shifted = h - cand
    j = np.searchsorted(r, shifted)
    j0 = np.clip(j - 1, 0, len(r) - 1)
    j1 = np.clip(j, 0, len(r) - 1)
    d = np.minimum(np.abs(r[j0] - shifted), np.abs(r[j1] - shifted))
    return int(np.count_nonzero(d <= tol_s))


def best_offset(
    reference: list[object] | Sequence[object],
    ours: Sequence[object],
    *,
    start_s: float = 0.0,
    tol_ms: float = TOL_MS,
    max_offset_s: float = OFFSET_MAX_S,
    step_s: float = OFFSET_STEP_S,
) -> tuple[float, int, int, bool]:
    """Desfase `ref - ours` que maximiza emparejamientos, buscado en start_s ± max.

    Devuelve `(offset_s, n_pares, n_candidatos, en_el_borde)`. Con RR regulares el
    máximo es una meseta de ancho ±tol, así que se toma su mediana y se refina con
    la mediana de los residuos. `en_el_borde` avisa que el desfase elegido quedó
    pegado al límite de la búsqueda: el desfase real podría estar fuera de
    ±`max_offset_s` y entonces el emparejamiento no es de fiar.
    """
    rt = sorted(_t_of(b) for b in ours)
    ht = [float(getattr(b, "time_ms", 0)) / 1000.0 for b in reference]
    if not rt or not ht:
        return float(start_s), 0, 0, False
    tol_s = tol_ms / 1000.0
    import numpy as np

    cands = np.arange(
        start_s - max_offset_s, start_s + max_offset_s + step_s * 0.5, step_s, dtype=float
    )
    scores = np.asarray([_count_matches(rt, ht, float(c), tol_s) for c in cands])
    best = int(scores.max())
    if best <= 0:
        return round(float(start_s), 3), 0, int(len(cands)), False
    hits = cands[scores == best]
    offset = float(np.median(hits))

    # Refinar: con la meseta elegida, el residuo mediano `our - ref` corrige el
    # sesgo que deja la cuantización del paso.
    aligned = [t - offset for t in ht]
    _ri, _oi, dts = _match_pairs(rt, aligned, tol_s)
    if dts:
        offset = offset - statistics.median(dts)
    offset = min(max(offset, start_s - max_offset_s), start_s + max_offset_s)
    # El desfase quedó en el borde de la búsqueda: el real podría estar fuera.
    clipped = bool(abs(offset - float(cands[0])) <= step_s * 1.5
                   or abs(float(cands[-1]) - offset) <= step_s * 1.5)
    return round(offset, 3), len(dts), int(len(cands)), clipped


def _rr_median_local(ht: object, t0: float, t1: float) -> float | None:
    """RR mediano de los latidos de HScribe en ``[t0-ctx, t1+ctx)``."""
    import numpy as np

    arr = np.asarray(ht, dtype=float)
    lo = int(np.searchsorted(arr, t0 - _RR_CONTEXT_S, "left"))
    hi = int(np.searchsorted(arr, t1 + _RR_CONTEXT_S, "left"))
    sel = arr[lo:hi]
    if len(sel) < 2:
        return None
    diffs = np.diff(sel)
    diffs = diffs[diffs > 0]
    if not len(diffs):
        return None
    return float(np.median(diffs))


def match_windows(
    reference: Sequence[object],
    ours: Sequence[object],
    offset_s: float = 0.0,
    window_s: float = WINDOW_S,
    tol_ms: float = TOL_MS,
    span_s: float | None = None,
) -> dict[str, object]:
    """Cobertura y métricas por ventanas de `window_s` segundos.

    Una ventana tiene referencia (`reference_valid`) si HScribe tiene al menos
    `MIN_REF_BEATS` latidos y ≥ `MIN_REF_FRACTION` de los esperados por su RR
    mediano local. `sensitivity`, `ppv` y `f1` se calculan **solo** sobre las
    ventanas con referencia; el resto se acumula en `tiempo_sin_referencia_s`.

    `span_s` es la duración analizada (s desde el inicio de la exportación). Sin
    él, el registro se supone que llega hasta el último latido (de HScribe o
    propio), lo que subestima el tiempo sin referencia cuando HScribe calla del
    todo en el tramo final.
    """
    import numpy as np

    rt = sorted(_t_of(b) for b in ours)
    ht = sorted(float(getattr(b, "time_ms", 0)) / 1000.0 - float(offset_s) for b in reference)
    t_last = max(ht[-1] if ht else 0.0, rt[-1] if rt else 0.0)
    t_end = float(span_s) if span_s and span_s > 0 else float(t_last)
    empty: dict[str, object] = {
        "windows": [],
        "n_windows": 0,
        "valid": 0,
        "frac_valid": 0.0,
        "median": 0.0,
        "reference_n_valid": 0,
        "ours_n_valid": 0,
        "matched_valid": 0,
        "sensitivity": None,
        "ppv": None,
        "f1": None,
        "median_dt_ms": None,
        "tiempo_con_referencia_s": 0.0,
        "tiempo_sin_referencia_s": round(t_end, 1),
    }
    if not ht or t_end <= 0:
        return empty

    ri, _oi, dts = _match_pairs(rt, ht, tol_ms / 1000.0)
    ht_arr = np.asarray(ht, dtype=float)
    ours_arr = np.asarray(rt, dtype=float)
    n_bins = max(1, int(np.ceil(t_end / window_s)))
    edges = np.arange(n_bins + 1, dtype=float) * window_s
    matched_arr = ht_arr[ri] if ri else np.empty(0, dtype=float)

    def _bins(times: object) -> object:
        idx = np.floor(np.asarray(times, dtype=float) / window_s).astype(int)
        return np.bincount(np.clip(idx, 0, n_bins - 1), minlength=n_bins)

    n_ref_bin = _bins(ht_arr)
    n_match_bin = _bins(matched_arr) if len(matched_arr) else np.zeros(n_bins, dtype=int)
    n_ours_bin = np.diff(np.searchsorted(ours_arr, edges, "left"))

    wins: list[dict[str, object]] = []
    for i in range(n_bins):
        t0 = float(edges[i])
        t1 = float(edges[i + 1])
        dur = min(window_s, max(0.0, t_end - t0))
        n_ref = int(n_ref_bin[i])
        n_ours = int(n_ours_bin[i])
        n_match = int(n_match_bin[i])
        rr = _rr_median_local(ht_arr, t0, t1)
        expected = dur / rr if rr else 0.0
        ok_rr = bool(rr and RR_PLAUSIBLE_S[0] <= rr <= RR_PLAUSIBLE_S[1])
        valid = bool(
            ok_rr
            and n_ref >= MIN_REF_BEATS
            and expected > 0
            and n_ref >= MIN_REF_FRACTION * expected
        )
        wins.append(
            {
                "t0": round(t0, 1),
                "t1": round(t1, 1),
                "dur_s": round(dur, 1),
                "n": int(n_ref),
                "expected": round(expected, 1),
                "rr_med_s": round(float(rr), 3) if rr else None,
                "ours_n": int(n_ours),
                "matched": int(n_match),
                "coverage": round(n_match / max(1, n_ref), 3),
                "reference_valid": valid,
            }
        )

    good = [w for w in wins if w["reference_valid"]]
    valid_bin = np.asarray([bool(w["reference_valid"]) for w in wins], dtype=bool)
    ref_n = int(n_ref_bin[valid_bin].sum())
    ours_n = int(n_ours_bin[valid_bin].sum())
    matched = int(n_match_bin[valid_bin].sum())
    valid_dts: list[float] = []
    if matched and len(matched_arr):
        mbin = np.clip(np.floor(matched_arr / window_s).astype(int), 0, n_bins - 1)
        valid_dts = [float(d) for d, ok in zip(dts, valid_bin[mbin], strict=True) if ok]
    se = matched / ref_n if ref_n else None
    ppv = matched / ours_n if ours_n else None
    if se is not None:
        se = min(1.0, se)
    if ppv is not None:
        ppv = min(1.0, ppv)
    f1 = _f1(se, ppv)
    time_ref = sum(float(w["dur_s"]) for w in good)  # type: ignore[arg-type]
    return {
        "windows": wins,
        "n_windows": len(wins),
        "valid": len(good),
        "frac_valid": round(time_ref / t_end, 3) if t_end > 0 else 0.0,
        "median": round(float(np.median([float(w["coverage"]) for w in wins])), 3)
        if wins
        else 0.0,
        "reference_n_valid": ref_n,
        "ours_n_valid": ours_n,
        "matched_valid": matched,
        "sensitivity": round(se, 4) if se is not None else None,
        "ppv": round(ppv, 4) if ppv is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "median_dt_ms": round(float(np.median(valid_dts)) * 1000.0, 2) if valid_dts else None,
        "tiempo_con_referencia_s": round(time_ref, 1),
        "tiempo_sin_referencia_s": round(max(0.0, t_end - time_ref), 1),
    }


def _f1(se: float | None, ppv: float | None) -> float | None:
    if se is None or ppv is None or (se + ppv) <= 0:
        return None
    return 2 * se * ppv / (se + ppv)


def match_reference(
    reference: Sequence[object],
    ours: Sequence[object],
    tol_ms: float = TOL_MS,
    start_s: float = 0.0,
    window_s: float = WINDOW_S,
    max_offset_s: float = OFFSET_MAX_S,
    duration_s: float | None = None,
) -> dict[str, object]:
    """Empareja latidos HScribe con latidos HolterIA (objetos con `.t`).

    `start_s` es el inicio (en s, desde el comienzo del examen) de la exportación
    WFDB: el desfase esperado entre `Beats.dat` y los latidos propios. Solo se
    exploran desfases en `start_s ± max_offset_s`.

    `duration_s` es la duración analizada (s) y solo sirve para que el tiempo sin
    referencia cubra también el tramo final donde HScribe no anotó nada.

    `sensitivity`, `ppv` y `f1` corresponden solo a las ventanas con referencia;
    `sensitivity_all`/`ppv_all` se incluyen para comparar con las mediciones
    viejas, que contaban todo el registro.
    """
    import numpy as np

    rt = sorted(_t_of(b) for b in ours)
    ht_all = [float(getattr(b, "time_ms", 0)) / 1000.0 for b in reference]
    offset, n_pairs, n_cands, clipped = best_offset(
        reference, ours, start_s=start_s, tol_ms=tol_ms, max_offset_s=max_offset_s
    )
    wins = match_windows(
        reference, ours, offset, window_s=window_s, tol_ms=tol_ms, span_s=duration_s
    )
    ht = [t - offset for t in ht_all]
    _ri, _oi, dts = _match_pairs(rt, ht, tol_ms / 1000.0)
    matched_all = len(dts)
    ref_all = len(ht)
    ours_all = len(rt)
    se_all = matched_all / ref_all if ref_all else None
    ppv_all = matched_all / ours_all if ours_all else None
    frac = float(wins["frac_valid"])  # type: ignore[arg-type]
    time_ref = float(wins["tiempo_con_referencia_s"])  # type: ignore[arg-type]
    return {
        "reference_n": ref_all,
        "ours_n": ours_all,
        "matched": matched_all,
        "coverage": round(matched_all / max(1, min(ref_all, ours_all)), 4),
        "sensitivity_all": round(se_all, 4) if se_all is not None else None,
        "ppv_all": round(ppv_all, 4) if ppv_all is not None else None,
        "median_dt_ms": round(float(np.median(dts)) * 1000.0, 2) if dts else None,
        "offset_s": offset,
        "offset_matches": n_pairs,
        "offset_candidates": n_cands,
        "offset_max_s": max_offset_s,
        "offset_clipped": clipped,
        "start_s": start_s,
        "window_s": window_s,
        "duration_s": duration_s,
        "sensitivity": wins["sensitivity"],
        "ppv": wins["ppv"],
        "f1": wins["f1"],
        "reference_n_valid": wins["reference_n_valid"],
        "ours_n_valid": wins["ours_n_valid"],
        "matched_valid": wins["matched_valid"],
        "windows": wins["windows"],
        "n_windows": wins["n_windows"],
        "valid_windows": wins["valid"],
        "window_median": wins["median"],
        "frac_referencia": frac,
        "tiempo_con_referencia_s": wins["tiempo_con_referencia_s"],
        "tiempo_sin_referencia_s": wins["tiempo_sin_referencia_s"],
        # "Hay referencia utilizable en este examen": al menos un minuto y una
        # cuarta parte del registro con latidos anotados por HScribe.
        "reference_valid": bool(frac >= 0.25 and time_ref >= 60.0),
    }
