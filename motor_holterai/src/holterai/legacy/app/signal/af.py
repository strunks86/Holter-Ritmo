"""Reglas de FA y flutter de los tratados (T125, sección E de CRITERIOS-TRATADOS).

Capa sobre el detector por episodios de ``rhythm.py`` (T23): usa su puntuación
por latido (``rhythm.af_scores``) y sus episodios, sin cambiar sus umbrales. Añade:

1. Episodio de FA solo si dura >= 30 s. Los más cortos (>= ``salva_min_s``) se
   informan como salva S irregular, no como FA.
2. RR regular durante FA (CV < 0,06 en >= 10 RR seguidos) → «posible BAV
   completo o flutter».
3. Pausa >= 3 s al terminar un episodio de FA → pausa post-conversión.
4. Pausa >= 5 s dentro de un episodio de FA → se informa.
5. Durante FA las S no cuentan como ESV (``esv_contables``).
6. QRS ancho aislado, clasificado V, tras un ciclo largo seguido de uno corto, sin
   pausa compensadora y con forma de BRD → «aberrancia de Ashman probable». Ver
   ``_ashman``. El cociente RR largo/RR corto ≥ ``ashman_ratio_min`` queda SIN
   calibrar: con MIT-BIH 201–203 la población ancha no separa (positivos de 201,
   negativos de 203; ``tools/t162_calibra_ashman.py``). Ver el informe de T162.
7. FA con QRS ancho y RR mínimo entre latidos anchos < 250 ms → «posible FA
   preexcitada». Necesita anchos de QRS (``widths_ms``); sin ellos no dispara.
8. Carga de FA en % del tiempo analizable (latidos con puntuación válida).

Flutter (T96, a revisión): ``flutter_eventos`` convierte las ventanas de la
actividad f (``signal/atrial.py``) marcadas como flutter en eventos, fuera de FA.

Los episodios de FA los pone quien llama (``fa_episodes``, el mismo detector que
informa ``analyze()``); si no se pasan, se calculan con ``rhythm.af_scores``.

Todo evento en un tramo con >= ``artefacto_frac`` de latidos ilegibles (≥2 de 3
canales malos) se descarta. No entrena nada: los umbrales están en ``UMBRALES``
y en ``configs/criteria/fa_tratados.yaml`` (``cargar_umbrales``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import rhythm

UMBRALES: dict[str, float] = {
    "fa_min_s": 30.0,            # episodio de FA: duración mínima
    "salva_min_s": 5.0,          # 5–30 s: salva S irregular (se informa, no es FA)
    "regular_cv_max": 0.06,      # CV del RR para ritmo regular
    "regular_min_rr": 10,        # RR consecutivos con CV < umbral
    "pausa_post_s": 3.0,         # pausa post-conversión
    # la pausa empieza a ±10 s del fin del episodio: el fin lo marca la última ventana
    # con puntuación >= on, y la mediana de suavizado (AF_SMOOTH_BEATS) lo retrasa ~10 latidos
    "pausa_post_tol_s": 10.0,
    "pausa_fa_s": 5.0,           # pausa dentro de FA
    "qrs_ancho_ms": 110.0,       # QRS anormal (adulto)
    "preex_rr_min_s": 0.25,      # RR mínimo entre latidos anchos para preexcitación
    "preex_min_anchos": 5,       # latidos anchos mínimos en el episodio
    # Regla 6. Cociente RR previo largo / RR previo corto. Valor de partida 1,4, SIN
    # calibrar: la población de MIT-BIH 201–203 no separa (tools/t162_calibra_ashman.py).
    "ashman_ratio_min": 1.4,
    # Pausa compensadora: RR previo + RR siguiente ≈ 2 × RR mediano ± 10 % (CRITERIOS, C9).
    "pausa_comp_tol": 0.10,
    "flutter_min_s": 30.0,       # flutter sostenido: mismo mínimo que un episodio de FA
    "artefacto_frac": 0.5,       # fracción de latidos ilegibles que anula un evento
}

TEXTOS = {
    "fa_ritmo_regular": "FA con ritmo regular: posible BAV completo o flutter",
    "pausa_post_conversion": "pausa post-conversión (bradi-taqui)",
    "pausa_5s_en_fa": "pausa ≥5 s dentro de FA",
    "fa_preexcitada": "posible FA preexcitada",
    "aberrancia_ashman": "aberrancia de Ashman probable (no V)",
    "flutter_a_revision": "flutter (pico 4–6 Hz con armónico): a revisión",
}


def cargar_umbrales(path: str | Path | None = None) -> dict[str, float]:
    """Umbrales por defecto, o los del YAML ``fa_tratados`` si se pasa ``path``.

    Las claves desconocidas dan error para que una errata no pase en silencio.
    """
    out = dict(UMBRALES)
    if path is None:
        return out
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    sec = data.get("fa_tratados", data)
    unknown = set(sec) - set(UMBRALES)
    if unknown:
        raise ValueError(f"claves de FA desconocidas: {sorted(unknown)}")
    out.update({k: type(UMBRALES[k])(v) for k, v in sec.items()})
    return out


def _ilegible(t: np.ndarray, unr: np.ndarray, a: float, b: float, frac: float) -> bool:
    m = (t >= a) & (t <= b)
    return bool(m.any() and float(np.mean(unr[m])) >= frac)


def _ritmo_regular(t: np.ndarray, normal: np.ndarray, idx: np.ndarray,
                   u: dict[str, Any]) -> list[dict]:
    """Tramos de RR regular (CV < umbral en >= ``regular_min_rr`` RR seguidos)."""
    k = idx[normal[idx]]
    w = int(u["regular_min_rr"])
    if k.size < w + 1:
        return []
    rr = np.diff(t[k])
    # solo RR entre latidos consecutivos y normales (sin ectopia ni ilegibles entre medias)
    ok = (rr > rhythm.AF_MIN_RR_S) & (rr < rhythm.AF_MAX_RR_S) & (np.diff(k) == 1)
    if rr.size < w:
        return []
    from numpy.lib.stride_tricks import sliding_window_view

    win = sliding_window_view(rr, w)
    okw = sliding_window_view(ok, w).all(axis=1)
    mu = win.mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = win.std(axis=1) / mu
    starts = np.flatnonzero(okw & (cv < u["regular_cv_max"]))
    if starts.size == 0:
        return []
    fl = np.zeros(rr.size + 1, dtype=int)
    np.add.at(fl, starts, 1)
    np.add.at(fl, starts + w, -1)
    cover = np.cumsum(fl)[: rr.size] > 0
    edges = np.flatnonzero(np.diff(np.concatenate([[0], cover.astype(int), [0]])))
    out = []
    # intervalos RR [i0, i1) cubiertos por al menos una ventana regular
    for i0, i1 in zip(edges[::2], edges[1::2], strict=True):
        out.append({"tipo": "fa_ritmo_regular",
                    "start_s": round(float(t[k[i0]]), 2),
                    "end_s": round(float(t[k[i1]]), 2)})
    return out


def _preexcitacion(t: np.ndarray, wide: np.ndarray, unr: np.ndarray,
                   idx: np.ndarray, u: dict[str, Any]) -> list[dict]:
    w = idx[wide[idx] & ~unr[idx]]
    if w.size < int(u["preex_min_anchos"]):
        return []
    cons = np.diff(w) == 1
    if not cons.any():
        return []
    rr = np.diff(t[w])[cons]
    rr = rr[rr > 0]
    if rr.size == 0 or float(rr.min()) >= u["preex_rr_min_s"]:
        return []
    return [{"tipo": "fa_preexcitada",
             "start_s": round(float(t[w[0]]), 2), "end_s": round(float(t[w[-1]]), 2),
             "rr_min_ms": round(1000.0 * float(rr.min()), 0)}]


def _ashman(t: np.ndarray, cls: np.ndarray, wide: np.ndarray, unr: np.ndarray,
            idx: np.ndarray, rr_med: float, u: dict[str, Any], stats: dict[str, int],
            brd: np.ndarray | None = None) -> list[dict]:
    """Regla 6. Aberrancia de Ashman probable frente a V.

    Un QRS ancho aislado clasificado V se informa como aberrancia si se cumplen a la
    vez, sobre los latidos del episodio:

    1. Ciclo largo seguido de corto: RR previo largo / RR previo corto ≥
       ``ashman_ratio_min``. (La tarea lo escribe RR(n−1)/RR(n−2) con el ciclo
       largo como denominador de la versión antigua; aquí el numerador es el
       ciclo largo.)
    2. Sin pausa compensadora: RR previo + RR siguiente fuera de 2 × RR mediano ± ``pausa_comp_tol``.
    3. Forma de BRD (``brd[j]``, de ``qrs_morph.classify_beat``). Sin morfología
       determinada no se afirma nada: la alerta no sale.

    Los contadores de ``stats`` muestran cuántos latidos caen en cada condición.
    """
    out = []
    n = t.size
    for j in idx:
        if j < 2 or j >= n - 1 or not wide[j] or unr[j] or cls[j] != "V":
            continue
        if wide[j - 1] or wide[j + 1] or unr[j - 1] or unr[j + 1]:
            continue
        stats["candidatos_v_aislados"] += 1
        rr_prev = float(t[j] - t[j - 1])
        rr_prev2 = float(t[j - 1] - t[j - 2])
        rr_next = float(t[j + 1] - t[j])
        if rr_prev <= 0 or rr_prev2 <= 0 or rr_med <= 0:
            continue
        if rr_prev2 / rr_prev < u["ashman_ratio_min"]:
            continue
        stats["ciclo_largo_corto"] += 1
        comp = abs((rr_prev + rr_next) - 2.0 * rr_med) <= u["pausa_comp_tol"] * 2.0 * rr_med
        if comp:
            continue
        stats["sin_pausa_compensadora"] += 1
        if brd is None or not bool(brd[j]):
            stats["sin_forma_brd"] += 1
            continue
        stats["aberrancia"] += 1
        out.append({"tipo": "aberrancia_ashman", "start_s": round(float(t[j]), 2),
                    "end_s": round(float(t[j]), 2)})
    return out


def flutter_eventos(windows, fa_segs, u: dict[str, Any] | None = None,
                    quality_ok: bool = True) -> list[dict]:
    """Flutter a revisión: tramos seguidos de ventanas marcadas como flutter fuera de FA.

    ``windows`` son las ventanas de ``signal/atrial.analyze`` (``t0``, ``t1``,
    ``flutter``); ``fa_segs`` son los episodios de FA (``start_s``, ``end_s``).
    Solo cuenta si la calidad de la resta del QRST es suficiente (``quality_ok``,
    el criterio de T11). Un tramo debe durar al menos ``flutter_min_s``.
    """
    uu = dict(UMBRALES) if u is None else {**UMBRALES, **u}
    if not quality_ok or not windows:
        return []
    runs: list[list[float]] = []
    cur: list[float] | None = None
    for w in windows:
        if w.get("flutter"):
            if cur is not None and abs(float(w["t0"]) - cur[1]) < 1e-6:
                cur[1] = float(w["t1"])
            else:
                if cur is not None:
                    runs.append(cur)
                cur = [float(w["t0"]), float(w["t1"])]
        else:
            if cur is not None:
                runs.append(cur)
                cur = None
    if cur is not None:
        runs.append(cur)
    out = []
    for a, b in runs:
        if b - a < float(uu["flutter_min_s"]):
            continue
        if any(a <= f["end_s"] and f["start_s"] <= b for f in fa_segs):
            continue
        out.append({"tipo": "flutter_a_revision", "start_s": round(a, 2), "end_s": round(b, 2),
                    "duration_s": round(b - a, 2)})
    return out


def analizar_fa(times, classes=None, duration_s: float = 0.0, *, unreadable=None,
                widths_ms=None, p_present=None, p_evaluable=None, fwave=None,
                brd=None, fa_episodes=None,
                umbrales: dict[str, float] | None = None) -> dict[str, Any]:
    """Detector de FA con las reglas de los tratados (ver el módulo).

    ``times`` y ``classes`` por latido (AAMI: N, S, V, F, Q); ``unreadable`` por
    latido; ``widths_ms`` anchos de QRS por latido (reglas 6 y 7); ``brd`` por
    latido, forma de BRD (regla 6). ``fa_episodes`` (lista de dicts con
    ``start_s``/``end_s``) sustituye al detector de FA interno: así las reglas se
    aplican sobre los mismos episodios que informa ``analyze()``.
    """
    u = dict(UMBRALES) if umbrales is None else {**UMBRALES, **umbrales}
    t = np.asarray(times, dtype=np.float64).ravel()
    n = t.size
    cls = rhythm._as_class_array(n, classes)
    unr = np.zeros(n, dtype=bool) if unreadable is None else np.asarray(unreadable, bool).ravel()
    wide = (np.zeros(n, dtype=bool) if widths_ms is None
            else np.nan_to_num(np.asarray(widths_ms, float).ravel(), nan=0.0) > u["qrs_ancho_ms"])
    frac = float(u["artefacto_frac"])

    scores = None
    if fa_episodes is None:  # con episodios dados no hace falta el detector interno (coste)
        scores = rhythm.af_scores(t, cls, unr if unreadable is not None else None,
                                  p_present=p_present, p_evaluable=p_evaluable, fwave=fwave)
    if fa_episodes is not None:
        # episodios que ya informa analyze(): solo los que duran >= fa_min_s
        fa_pre = [dict(s, duration_s=round(float(s["end_s"]) - float(s["start_s"]), 2))
                  for s in fa_episodes
                  if float(s["end_s"]) - float(s["start_s"]) >= float(u["fa_min_s"])]
        fa, n_veto = fa_pre, 0
        salva = []
    else:
        # Episodios de FA: exactamente los del corte de 30 s de rhythm (la regla 1 no
        # cambia ningún episodio ≥30 s). Las ráfagas de 5–30 s que no solapan ninguno
        # se informan aparte como salva S irregular.
        fa_pre = rhythm.af_segments(t, scores, min_s=float(u["fa_min_s"]))
        fa, n_veto = rhythm._p_veto(fa_pre, t, p_present, p_evaluable)
        cand = rhythm.af_segments(t, scores, min_s=float(u["salva_min_s"]))
        # una ráfaga corta que solapa un episodio (aunque el veto por P lo haya quitado) no es salva
        salva = [dict(s, label="salva_S_irregular") for s in cand
                 if s["duration_s"] < u["fa_min_s"]
                 and not any(s["start_s"] <= f["end_s"] and f["start_s"] <= s["end_s"]
                             for f in fa_pre)]
        salva = [s for s in salva if not _ilegible(t, unr, s["start_s"], s["end_s"], frac)]
    fa = [s for s in fa if not _ilegible(t, unr, s["start_s"], s["end_s"], frac)]

    in_fa = np.zeros(n, dtype=bool)
    for s in fa:
        in_fa |= (t >= s["start_s"]) & (t <= s["end_s"])
    normal = ~np.isin(cls, ["S", "V", "F", "Q"]) & ~unr
    rr_all = np.diff(t)
    normal_rr = normal[:-1] & normal[1:]

    alerts: list[dict] = []
    ash_stats = {"candidatos_v_aislados": 0, "ciclo_largo_corto": 0,
                 "sin_pausa_compensadora": 0, "sin_forma_brd": 0, "aberrancia": 0}
    brd_arr = None if brd is None else np.asarray(brd, dtype=bool).ravel()
    for s in fa:
        idx = np.flatnonzero((t >= s["start_s"]) & (t <= s["end_s"]))
        alerts += _ritmo_regular(t, normal, idx, u)
        alerts += _preexcitacion(t, wide, unr, idx, u)
        m = (t[:-1] >= s["start_s"]) & (t[:-1] <= s["end_s"]) & normal_rr
        rr_med = float(np.median(rr_all[m])) if m.any() else 0.0
        alerts += _ashman(t, cls, wide, unr, idx, rr_med, u, ash_stats, brd=brd_arr)

    # pausas (RR >= 3 s): post-conversión y dentro de FA
    for i in np.flatnonzero(rr_all >= min(u["pausa_post_s"], u["pausa_fa_s"])):
        a, b = float(t[i]), float(t[i + 1])
        if _ilegible(t, unr, a, b, frac):
            continue
        dur = b - a
        for s in fa:
            if dur >= u["pausa_post_s"] and abs(a - s["end_s"]) <= u["pausa_post_tol_s"]:
                alerts.append({"tipo": "pausa_post_conversion", "start_s": round(a, 2),
                               "end_s": round(b, 2), "duration_s": round(dur, 2)})
            if dur >= u["pausa_fa_s"] and a >= s["start_s"] and b <= s["end_s"]:
                alerts.append({"tipo": "pausa_5s_en_fa", "start_s": round(a, 2),
                               "end_s": round(b, 2), "duration_s": round(dur, 2)})

    for al in alerts:
        al["texto"] = TEXTOS[al["tipo"]]
    alerts.sort(key=lambda x: x["start_s"])

    s_all = (cls == "S") & ~unr
    esv_en_fa = int(np.sum(s_all & in_fa))
    esv_total = int(np.sum(s_all))
    dur = float(duration_s) if duration_s and duration_s > 0 else (
        float(t[-1] - t[0]) if n else 0.0)
    fa_total = float(sum(s["duration_s"] for s in fa))
    if scores is not None:
        frac_ok = float(np.mean(np.isfinite(scores))) if n else 0.0
    else:
        frac_ok = float(np.mean(~unr)) if n else 0.0
    analizable = dur * frac_ok

    return {
        "episodes": fa,
        "salvas": salva,
        "alerts": alerts,
        "scores": scores,
        "esv_total": esv_total,
        "esv_en_fa": esv_en_fa,
        "esv_contables": esv_total - esv_en_fa,
        "ashman": ash_stats,
        "burden": {
            "n_episodes": len(fa),
            "total_s": round(fa_total, 1),
            "longest_s": round(max((s["duration_s"] for s in fa), default=0.0), 1),
            "analizable_s": round(analizable, 1),
            "burden_pct_analizable": (round(100.0 * fa_total / analizable, 2)
                                      if analizable > 0 else 0.0),
            "burden_pct_registro": round(100.0 * fa_total / dur, 2) if dur > 0 else 0.0,
        },
        "n_p_veto": n_veto,
        "analyzed_pct": round(100.0 * frac_ok, 1),
    }
