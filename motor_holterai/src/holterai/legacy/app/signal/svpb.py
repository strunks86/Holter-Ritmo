"""Decisión clínica de extrasístoles supraventriculares (S).

La tarea de este módulo es convertir la evidencia dispersa (prematuridad, forma
del QRS, onda P y pausa) más la probabilidad del modelo en **una** decisión por
latido, con umbrales explícitos y calibrables. La clase S del análisis la decide
esta capa (:func:`apply`): el clasificador sólo aporta su probabilidad, y una S
suya que no cumple las compuertas vuelve a N.

Criterios (todos son necesarios para una S "de manual"):

  * **prematuridad**: ``rr_prev / rr_sinus <= prematurity`` (0,88 por defecto)
    contra el **ciclo sinusal local** :func:`sinus_rr_reference` —promedio de
    pares de RR consecutivos—, no contra la mediana de los RR, que en bigeminia
    cae en el intervalo corto y hace desaparecer la prematuridad. Se admiten
    dos ensanchamientos: ``prematurity_relaxed`` (0,94) con evidencia fuerte de
    P anómala, y ``prematurity_pause`` (0,95) con pausa completa y el modelo (o
    la P) a favor, porque en Holter con FC alta hay extrasístoles que sólo
    adelantan el 5-7 % del ciclo,
  * **QRS igual al dominante**: correlación con la plantilla sinusal
    ``>= qrs_corr_min`` (0,90) y ancho no aumentado,
  * **P anómala o adelantada**: ausente, con morfología distinta de la P
    sinusal (correlación baja con su plantilla), con PR distinto del sinusal o
    con intervalo P-P acortado respecto del ciclo sinusal,
  * **pausa no compensadora**: ``rr_next > rr_prev`` y
    ``(rr_prev + rr_next) / (2 * rr_sinus)`` en ``[pause_min, pause_compensatory)``
    (0,88-1,05). Una pausa completamente compensadora apunta a extrasístole
    ventricular.
  * **ritmo de fondo**: si la carga de irregularidad local
    (:func:`irregular_burden`) indica una arritmia sostenida —fibrilación
    auricular— no se diagnostican S salvo con P **visible** pero anómala y RR
    muy prematuro: en la FA el "RR corto" es el ritmo, no una extrasístole.
    Y si el registro entero es irregular y la P casi no se puede medir
    (``af_cv`` + ``p_available_min``), no hay información atrial: sólo se
    admite prematuridad franca con el modelo a favor y no se aceptan rachas.
  * **rachas**: una salva de latidos de QRS angosto, cortos y regulares que
    empieza de golpe (:func:`tachycardia_runs`) es supraventricular en todos sus
    latidos: dupletas, TSV y taquicardia auricular. Se juzgan contra el ciclo
    sinusal del **registro** (:func:`record_long_rr`), no contra el entorno
    inmediato, que dentro de la racha es la racha misma.

El puntaje final es ``w_phys * phys + w_model * model`` multiplicado por las
compuertas duras, y es S si ``score >= threshold``. ``phys`` pesa la evidencia
fisiológica y ``model`` la probabilidad del clasificador ya calibrada (T06) más
la del modelo específico de PAC; así un latido con el modelo en contra puede
seguir siendo S si el cuadro clínico es inequívoco, y al revés.

Los umbrales se eligen en validación (ver ``scripts/eval_svpb.py``), nunca en
el conjunto de prueba; esta capa no reentrena ningún modelo.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

#: Nombres de las columnas de :func:`feature_matrix`.
SVPB_FEATURES = (
    "rr_prev", "rr_next", "rr_ratio", "rr_cv", "pause_ratio", "qrs_corr",
    "width_ratio", "p_present", "p_corr", "pr_ms", "pr_diff_ms", "pp_ratio",
    "prem_score", "p_score", "pause_score", "qrs_score", "phys_score",
    "model_prob", "score",
)
IDX = {name: i for i, name in enumerate(SVPB_FEATURES)}

#: Valores de ``p_state`` por latido (para informes y pruebas).
P_UNKNOWN = -1
P_ABSENT = 0
P_PRESENT = 1


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SvpbConfig:
    """Umbrales de la decisión S. Todos calibrables desde JSON."""

    # compuerta de prematuridad
    prematurity: float = 0.88          # rr_prev / rr sinusal local máximo
    prematurity_relaxed: float = 0.94  # ... si la evidencia de P es fuerte
    p_strong: float = 0.60             # "evidencia fuerte de P" para la relajada
    #: Tercera vía, para las extrasístoles **poco** prematuras (0,93-0,95 del
    #: ciclo, frecuentes en Holter con FC alta): se admiten si hay pausa
    #: completa y algún apoyo del modelo o de la onda P. Sin ese apoyo la vía
    #: no se abre, porque un RR al 95 % del ciclo es lo normal en sinus.
    prematurity_pause: float = 0.95
    p_relaxed: float = 0.20
    model_relaxed: float = 0.50
    # compuerta de forma del QRS
    qrs_corr_min: float = 0.90
    width_abs_max_s: float = 0.12      # ancho absoluto máximo (bloqueo excluido)
    width_ratio_max: float = 1.25      # ... y respecto de la mediana del registro
    # compuerta de pausa
    pause_compensatory: float = 1.05   # (rr_prev + rr_next) / (2 rr sinusal)
    pause_min: float = 0.88            # ... y pausa mínima (una S deja pausa)
    # compuertas de contexto
    #: El ruido de ``features.noise_level`` depende de la morfología del examen
    #: (mediana por registro de 0,01 a 23 mV), así que se compara contra la
    #: mediana del propio registro y no contra un valor absoluto: un umbral
    #: fijo como el 0,30 histórico descartaba el 87 % de las S de SVDB.
    noise_rel_max: float = 4.0
    rr_min_s: float = 0.25
    rr_max_s: float = 2.00
    rr_cv_irregular: float = 0.18      # ritmo irregular: exige evidencia de P
    #: Guarda de registro: si la mediana de :func:`rr_irregularity` supera esto
    #: y además la onda P casi no se puede medir (``p_available_min``), el
    #: registro no tiene ni ciclo sinusal estable ni información auricular —es
    #: el patrón de una fibrilación auricular— y no se diagnostican S salvo
    #: prematuridad franca con apoyo del modelo (``af_model_min``); las rachas
    #: exigen entonces ser una taquicardia sostenida.
    af_cv: float = 0.18
    p_available_min: float = 0.20
    af_prematurity: float = 0.80
    af_model_min: float = 0.60
    #: Carga de irregularidad a partir de la cual el ritmo es una arritmia
    #: sostenida (FA): ahí no se diagnostican S salvo evidencia fuerte.
    af_burden: float = 0.65
    prematurity_strong: float = 0.78   # "muy prematuro" que vence la carga de FA
    # puntaje
    w_prem: float = 0.45
    w_p: float = 0.30
    w_pause: float = 0.15
    w_qrs: float = 0.10
    w_phys: float = 0.60
    w_model: float = 0.50
    #: El modelo PAC está entrenado con S infladas (script de T05), así que
    #: entra atenuado: su probabilidad se suma como "una fuente más".
    pac_model_weight: float = 0.50
    #: Umbral elegido en validación (SVDB pares + INCART pares, ver
    #: ``scripts/eval_svpb.py --sweep``): el F1 agrupado es plano entre 0,20 y
    #: 0,35 (0,551-0,556) y 0,30 es el centro de la meseta; por encima de 0,40
    #: cae la sensibilidad sin ganar precisión.
    threshold: float = 0.30
    # válvulas de seguridad
    max_s_fraction: float = 0.35       # si el puntaje etiqueta más, sube el umbral
    pr_diff_ms: float = 30.0           # |PR - PR sinusal| que ya es "distinto"
    pp_short: float = 0.85             # P-P / RR local que ya es "P adelantada"
    # rachas: dupletas, salvas, TSV y taquicardia auricular (QRS angosto)
    run_min_beats: int = 2             # latidos mínimos de la racha (2 = dupleta)
    #: RR de la racha / ciclo sinusal del registro. Elegido en validación
    #: (SVDB pares + INCART pares): el paso de 0,85 a 0,78 no cambia la
    #: sensibilidad y sube la PPV (SVDB-val 0,661→0,726; DS2 0,673→0,739),
    #: porque una racha sólo 15 % más rápida que el ciclo de base suele ser
    #: arritmia sinusal, no una salva ectópica.
    run_rel: float = 0.78
    #: El ciclo del registro se toma como el percentil 85 de los RR, acotado a
    #: ``run_long_max_ratio`` veces la mediana: en una taquicardia auricular
    #: sostenida la mediana es el ciclo rápido y el percentil 85 sigue siendo el
    #: ciclo de base, pero si el registro entero es rápido el tope evita inventar
    #: un ciclo lento que no existe.
    run_long_max_ratio: float = 1.6
    run_rr_max_s: float = 0.52         # RR máximo de una taquicardia sostenida
    run_long_beats: int = 6            # desde aquí la racha no exige inicio abrupto
    run_regularity: float = 1.15       # RR máximo / RR mínimo dentro de la racha
    run_onset_ratio: float = 1.15      # RR previo a la racha / ciclo del registro
    run_max_fraction: float = 0.30     # si las rachas cubren más, no son rachas
    #: ... pero las rachas de **inicio abrupto** se conservan hasta esta otra
    #: fracción: una bigeminia auricular sostenida anota S en la mitad del
    #: registro y sigue siendo una racha, no el ritmo de base.
    run_hard_fraction: float = 0.85
    #: Fracción máxima de latidos con P visible dentro de una racha sostenida
    #: (no paroxística y rápida) para aceptarla como ritmo auricular ectópico.
    #: Un sinus rápido normalmente deja ver la P en >5 % de sus latidos.
    run_sustained_p_max: float = 0.05

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict) -> SvpbConfig:
        known = cls.__dataclass_fields__
        return cls(**{k: float(v) for k, v in data.items() if k in known})

    @classmethod
    def load(cls, path: str | Path | None) -> SvpbConfig:
        """Lee umbrales de un JSON; si no está o está roto, usa los defaults."""
        if path is None:
            return cls()
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return cls()
        try:
            return cls.from_dict(data)
        except Exception:
            return cls()

    def tuned(self, **kwargs) -> SvpbConfig:
        """Copia con algunos umbrales cambiados (búsqueda en validación)."""
        return replace(self, **kwargs)


DEFAULT = SvpbConfig()


# ---------------------------------------------------------------------------
# Derivadas de RR y QRS
# ---------------------------------------------------------------------------
def _clip01(x: np.ndarray | float) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)


def prematurity_score(rr_ratio: np.ndarray) -> np.ndarray:
    """0 con RR igual al local, 1 con RR <= 0,70 del local."""
    r = np.asarray(rr_ratio, dtype=np.float64)
    return _clip01((0.95 - r) / 0.25)


def sinus_rr_reference(rr_prev: np.ndarray, rr_next: np.ndarray,
                       rr_local: np.ndarray | None = None,
                       half: int = 5) -> np.ndarray:
    """RR sinusal local estimado con **pares** de intervalos consecutivos.

    ``rr_local`` (mediana de los RR previos) se contamina cuando la ectopia es
    frecuente: en una bigeminia auricular los RR alternan 0,55 / 1,25 y la
    mediana local puede caer en el intervalo corto, con lo que la extrasístole
    deja de parecer prematura (ratio ≈ 1,0). El promedio de cada par
    ``(RR(i-1) + RR(i)) / 2`` es el ciclo sinusal tanto en bigeminia como en
    trigeminia, y su mediana en ±``half`` latidos es un estimador estable del
    ciclo de base.
    """
    prev = np.asarray(rr_prev, dtype=np.float64)
    nxt = np.asarray(rr_next, dtype=np.float64)
    pair = 0.5 * (prev + nxt)
    n = pair.size
    fallback = np.asarray(rr_local, dtype=np.float64) if rr_local is not None \
        else np.zeros(n)
    if n == 0:
        return np.zeros(0)
    pad = np.pad(pair, half, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)
    ok = np.isfinite(win) & (win > 0.2) & (win < 3.0)
    filled = np.where(ok, win, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(filled, axis=1)
    out = np.where(np.isfinite(med) & (med > 0), med, fallback)
    return np.where(np.isfinite(out) & (out > 0), out, 0.0)


def rr_irregularity(rr_prev: np.ndarray, rr_local: np.ndarray,
                    half: int = 10) -> np.ndarray:
    """Irregularidad local del RR: mediana de |ΔRR| en la ventana ±``half``.

    Se mide el cambio **latido a latido**, no la dispersión: una extrasístole
    aislada no vuelve irregular al ritmo (sólo cambia dos veces), mientras que
    la fibrilación auricular y el bigeminismo cambian en casi todos los
    intervalos. El valor va normalizado por el RR local:

      * ritmo sinusal (con o sin PAC aisladas) ~0,00-0,05,
      * bigeminismo ~0,3-0,6,
      * fibrilación auricular ~0,15-0,40.
    """
    rr = np.asarray(rr_prev, dtype=np.float64)
    loc = np.asarray(rr_local, dtype=np.float64)
    n = rr.size
    if n < 3:
        return np.zeros(n)
    ok = np.isfinite(rr) & (rr > 0.15) & (rr < 3.0)
    vals = np.where(ok, rr, np.nan)
    d = np.abs(np.diff(vals))
    d = np.concatenate([[np.nan], d])       # Δ respecto del intervalo anterior
    pad = np.pad(d, half, mode="constant", constant_values=np.nan)
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mad = np.nanmedian(win, axis=1)
    cv = np.divide(1.4826 * mad, np.maximum(loc, 1e-3),
                   out=np.zeros(n), where=np.isfinite(mad) & (loc > 0))
    return np.clip(cv, 0.0, 3.0)


def irregular_burden(rr_prev: np.ndarray, rr_sinus: np.ndarray, half: int = 25,
                     rel: float = 0.12) -> np.ndarray:
    """Fracción de latidos, en ±``half``, con RR desviado > ``rel`` del sinusal.

    Distingue la **arritmia sostenida** (fibrilación auricular, ~0,5-0,8: casi
    todos los intervalos se apartan del ciclo de base) del **sinus con
    extrasístoles aisladas** (~0,05-0,15: solo cambian dos intervalos por
    extrasístole). Es la compuerta que evita diagnosticar centenares de S en una
    FA, donde "RR corto + QRS angosto" es el ritmo y no una extrasístole.
    """
    rr = np.asarray(rr_prev, dtype=np.float64)
    loc = np.asarray(rr_sinus, dtype=np.float64)
    n = rr.size
    if n == 0:
        return np.zeros(0)
    dev = (np.abs(rr - loc) / np.maximum(loc, 1e-6) > rel).astype(np.float64)
    cum = np.concatenate([[0.0], np.cumsum(dev)])
    i = np.arange(n)
    lo = np.maximum(0, i - half)
    hi = np.minimum(n, i + half + 1)
    return (cum[hi] - cum[lo]) / (hi - lo)


def pause_score(pause_ratio: np.ndarray, cfg: SvpbConfig = DEFAULT) -> np.ndarray:
    """1 si la pausa es claramente no compensadora, 0 si es compensadora.

    También castiga la pausa demasiado corta (``pause_ratio`` < ``pause_min``):
    sin pausa posterior, un RR corto es solo un RR corto (o una racha), no una
    extrasístole con pausa.
    """
    p = np.asarray(pause_ratio, dtype=np.float64)
    upper = _clip01((cfg.pause_compensatory - 0.15 - p) / 0.50)
    lower = _clip01((p - (cfg.pause_min - 0.30)) / 0.25)
    return np.minimum(upper, lower)


def qrs_score(qrs_corr: np.ndarray, width_ratio: np.ndarray) -> np.ndarray:
    """Similitud del QRS con el dominante: correlación y ancho relativo."""
    c = np.asarray(qrs_corr, dtype=np.float64)
    w = np.asarray(width_ratio, dtype=np.float64)
    return np.minimum(_clip01((c - 0.85) / 0.10), _clip01((1.25 - w) / 0.25))


# ---------------------------------------------------------------------------
# Onda P
# ---------------------------------------------------------------------------
def p_evidence(peaks: np.ndarray, fs: float, res: dict | None,
               cls: np.ndarray | None = None,
               rr_local: np.ndarray | None = None,
               cfg: SvpbConfig = DEFAULT) -> dict:
    """Resume la onda P por latido a partir de ``pwave.detect_p_waves``.

    Devuelve arrays del largo de ``peaks``: ``p_state`` (-1 desconocido, 0
    ausente, 1 presente), ``p_corr``, ``pr_ms``, ``pr_diff_ms``, ``pp_ratio``,
    ``p_score`` y la referencia ``pr_ref_ms`` / ``pp_ref_s`` del registro.
    Un latido sin plantilla de P (minuto ruidoso o FC > 120) queda como
    desconocido y **no** suma evidencia en contra.
    """
    peaks = np.asarray(peaks, dtype=int).ravel()
    n = peaks.size
    out = {
        "p_state": np.full(n, P_UNKNOWN, dtype=np.int8),
        "p_corr": np.zeros(n, dtype=np.float64),
        "pr_ms": np.full(n, np.nan),
        "pr_diff_ms": np.zeros(n),
        "pp_ratio": np.full(n, np.nan),
        "p_score": np.zeros(n),
        "pr_ref_ms": float("nan"),
        "pp_ref_s": float("nan"),
    }
    if res is None or n == 0 or fs <= 0:
        return out
    present = np.asarray(res.get("present", np.zeros(n, bool)), dtype=bool)
    evaluable = np.asarray(res.get("evaluable", np.zeros(n, bool)), dtype=bool)
    tmpl_p = np.asarray(res.get("template_p", np.zeros(n, bool)), dtype=bool)
    corr = np.asarray(res.get("corr", np.zeros(n)), dtype=np.float64)
    if present.size != n:
        return out

    # PR medido desde el inicio de la P (si falta, desde el pico - 40 ms)
    onset = np.asarray(res.get("onset", np.full(n, -1)), dtype=np.int64)
    peak = np.asarray(res.get("peak", np.full(n, -1)), dtype=np.int64)
    ref = np.where(onset >= 0, onset, np.where(peak >= 0, peak, -1))
    pr = np.full(n, np.nan)
    ok = present & (ref >= 0)
    pr[ok] = (peaks[ok] - ref[ok]) * 1000.0 / float(fs)
    pr[(~np.isfinite(pr)) | (pr < 40.0) | (pr > 500.0)] = np.nan

    # P adelantada: intervalo P-P respecto del RR local
    pp_ratio = np.full(n, np.nan)
    pk = np.where(peak >= 0, peak, ref)
    have = present & (pk >= 0)
    prev_pk = np.full(n, -1, dtype=np.int64)
    prev_pk[1:] = np.where(have[:-1], pk[:-1], -1)
    good = have & (prev_pk >= 0) & (rr_local is not None)
    if rr_local is not None and good.any():
        rr_loc = np.asarray(rr_local, dtype=np.float64)
        pp = (pk[good] - prev_pk[good]) / float(fs)
        pp_ratio[good] = pp / np.maximum(rr_loc[good], 1e-3)

    # referencias del registro: sólo latidos sinusales con P medible
    base = evaluable & present & np.isfinite(pr)
    if cls is not None and base.any():
        sinus = base & (np.asarray(cls) == 0)
        if int(sinus.sum()) >= 5:
            base = sinus
    out["pr_ref_ms"] = float(np.median(pr[base])) if base.any() else float("nan")
    pp_ok = np.isfinite(pp_ratio) & base
    out["pp_ref_s"] = (float(np.median(pp_ratio[pp_ok]))
                       if pp_ok.any() else float("nan"))

    # estado por latido: un minuto sin P en la plantilla no es "P ausente" en
    # cada latido (el minuto puede ser ruidoso), así que esos quedan desconocidos
    state = np.full(n, P_UNKNOWN, dtype=np.int8)
    known = evaluable & (tmpl_p | present)
    state[known & present] = P_PRESENT
    state[known & ~present] = P_ABSENT
    out["p_state"] = state
    out["p_corr"] = np.where(state == P_PRESENT, corr, 0.0)
    out["pr_ms"] = pr
    out["pp_ratio"] = pp_ratio

    # evidencia de P anómala por componentes
    corr_z = _clip01((0.75 - corr) / 0.35)
    if np.isfinite(out["pr_ref_ms"]):
        d = pr - out["pr_ref_ms"]
        pr_diff = _clip01((np.abs(d) - cfg.pr_diff_ms) / 60.0)
    else:
        d = np.zeros(n)
        pr_diff = np.zeros(n)
    pp_short = _clip01((cfg.pp_short - pp_ratio) / 0.20)
    absent = np.where(state == P_ABSENT, 0.85, 0.0)
    present_m = state == P_PRESENT
    morph = np.where(present_m, corr_z, 0.0)
    pr_term = np.where(present_m & np.isfinite(pr), pr_diff, 0.0)
    pp_term = np.where(present_m & np.isfinite(pp_ratio), pp_short, 0.0)
    score = np.maximum.reduce([absent, morph, pr_term, pp_term])
    out["pr_diff_ms"] = np.where(np.isfinite(pr), d, 0.0)
    out["p_score"] = _clip01(np.where(state == P_UNKNOWN, 0.0, score))
    return out


# ---------------------------------------------------------------------------
# Puntaje y decisión
# ---------------------------------------------------------------------------
def feature_matrix(rr_prev, rr_next, rr_sinus, widths, width_ref, qrs_corr,
                   p_feat: dict, model_prob, cfg: SvpbConfig = DEFAULT) -> np.ndarray:
    """Matriz (N, len(SVPB_FEATURES)) con el detalle del puntaje.

    ``rr_sinus`` es el ciclo sinusal local (ver :func:`sinus_rr_reference`): la
    prematuridad y la pausa se miden contra él, no contra la mediana de los RR,
    que en bigeminia cae en el intervalo corto.
    """
    rr_prev = np.asarray(rr_prev, dtype=np.float64)
    rr_next = np.asarray(rr_next, dtype=np.float64)
    rr_sinus = np.asarray(rr_sinus, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64)
    n = rr_prev.size
    out = np.zeros((n, len(SVPB_FEATURES)), dtype=np.float32)
    rr_ratio = np.divide(rr_prev, rr_sinus, out=np.zeros(n), where=rr_sinus > 0)
    pause = np.divide(rr_prev + rr_next, 2.0 * rr_sinus,
                      out=np.zeros(n), where=rr_sinus > 0)
    wr = np.divide(widths, np.maximum(width_ref, 1e-6), out=np.ones(n),
                   where=np.asarray(width_ref) > 0)
    prem = prematurity_score(rr_ratio)
    psc = np.asarray(p_feat.get("p_score", np.zeros(n)), dtype=np.float64)
    pau = pause_score(pause, cfg)
    qrs = qrs_score(qrs_corr, wr)
    cv = rr_irregularity(rr_prev, rr_sinus)
    phys = (cfg.w_prem * prem + cfg.w_p * psc + cfg.w_pause * pau
            + cfg.w_qrs * qrs) / max(1e-9, cfg.w_prem + cfg.w_p + cfg.w_pause + cfg.w_qrs)
    model = _clip01(model_prob)
    score = _clip01(cfg.w_phys * phys + cfg.w_model * model)

    out[:, IDX["rr_prev"]] = np.clip(rr_prev, 0, 10)
    out[:, IDX["rr_next"]] = np.clip(rr_next, 0, 10)
    out[:, IDX["rr_ratio"]] = np.clip(rr_ratio, 0, 5)
    out[:, IDX["rr_cv"]] = cv
    out[:, IDX["pause_ratio"]] = np.clip(pause, 0, 6)
    out[:, IDX["qrs_corr"]] = np.clip(qrs_corr, -1, 1)
    out[:, IDX["width_ratio"]] = np.clip(wr, 0, 5)
    out[:, IDX["p_present"]] = np.asarray(p_feat.get("p_state", np.zeros(n)),
                                          dtype=np.float32)
    out[:, IDX["p_corr"]] = np.clip(p_feat.get("p_corr", np.zeros(n)), -1, 1)
    out[:, IDX["pr_ms"]] = np.nan_to_num(np.asarray(p_feat.get("pr_ms", np.zeros(n)),
                                                    dtype=np.float64), nan=0.0)
    out[:, IDX["pr_diff_ms"]] = np.clip(p_feat.get("pr_diff_ms", np.zeros(n)), -500, 500)
    out[:, IDX["pp_ratio"]] = np.nan_to_num(np.asarray(p_feat.get("pp_ratio",
                                                                 np.zeros(n))), nan=0.0)
    out[:, IDX["prem_score"]] = prem
    out[:, IDX["p_score"]] = psc
    out[:, IDX["pause_score"]] = pau
    out[:, IDX["qrs_score"]] = qrs
    out[:, IDX["phys_score"]] = phys
    out[:, IDX["model_prob"]] = model
    out[:, IDX["score"]] = score
    return out


def gates(rr_prev, rr_next, rr_sinus, widths, width_ref, qrs_corr,
          p_score, noise, noise_ref=None, model=None,
          cfg: SvpbConfig = DEFAULT) -> dict[str, np.ndarray]:
    """Compuertas duras: sin todas ellas no hay S, por alto que sea el puntaje.

    ``noise_ref`` es la mediana de ruido del registro: la compuerta es relativa
    porque ``features.noise_level`` depende de la morfología del examen.
    ``model`` es la probabilidad de S del clasificador calibrado y sólo se usa
    para abrir la vía de las extrasístoles poco prematuras.
    """
    rr_prev = np.asarray(rr_prev, dtype=np.float64)
    rr_next = np.asarray(rr_next, dtype=np.float64)
    rr_sinus = np.asarray(rr_sinus, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64)
    width_ref = np.asarray(width_ref, dtype=np.float64)
    n = rr_prev.size
    rr_ratio = np.divide(rr_prev, rr_sinus, out=np.ones(n), where=rr_sinus > 0)
    p_score = np.asarray(p_score, dtype=np.float64)
    model = np.zeros(n) if model is None else np.asarray(model, dtype=np.float64)
    pause = np.divide(rr_prev + rr_next, 2.0 * rr_sinus,
                      out=np.zeros(n), where=rr_sinus > 0)
    noise = np.zeros(n) if noise is None else np.asarray(noise, dtype=np.float64)

    relaxed_ok = (rr_ratio <= cfg.prematurity_relaxed) & (p_score >= cfg.p_strong)
    pause_ok = ((rr_next > rr_prev) & (pause < cfg.pause_compensatory)
                & (pause >= cfg.pause_min))
    # poco prematuro pero con pausa y apoyo del modelo o de la P
    mild_ok = ((rr_ratio <= cfg.prematurity_pause) & pause_ok
               & ((p_score >= cfg.p_relaxed) | (model >= cfg.model_relaxed)))
    premature = (rr_ratio <= cfg.prematurity) | relaxed_ok | mild_ok
    plausible = (rr_sinus >= cfg.rr_min_s) & (rr_sinus <= cfg.rr_max_s) & (rr_prev > 0.15)
    width_ok = widths <= np.maximum(cfg.width_abs_max_s, cfg.width_ratio_max * width_ref)
    qrs_ok = np.asarray(qrs_corr, dtype=np.float64) >= cfg.qrs_corr_min
    # la pausa tiene que existir: rr_next > rr_prev y sin compensación completa.
    # En ritmo irregular (FA, bigeminismo, arritmia sinusal) eso ocurre por azar,
    # así que además se exige que la pausa sea de tamaño normal o evidencia de P.
    cv = rr_irregularity(rr_prev, rr_sinus)
    irregular = cv > cfg.rr_cv_irregular
    chronic = (pause >= cfg.pause_min) | (p_score >= cfg.p_strong)
    pause_ok &= (~irregular) | chronic
    # arritmia sostenida (FA): "RR corto" es el ritmo, no una extrasístole.
    # La excepción se aplica en `decide`, donde está el estado de la onda P: en
    # FA la P falta por definición, así que "P ausente" no puede ser la
    # evidencia que venza la compuerta.
    af_like = irregular_burden(rr_prev, rr_sinus) > cfg.af_burden
    if noise_ref is None:
        clean = np.ones(n, dtype=bool)
    else:
        ref = np.asarray(noise_ref, dtype=np.float64)
        clean = noise <= cfg.noise_rel_max * np.maximum(ref, 0.02)
    return {
        "premature": premature,
        "relaxed": relaxed_ok,
        "mild": mild_ok,
        "plausible": plausible,
        "width_ok": width_ok,
        "qrs_ok": qrs_ok,
        "pause_ok": pause_ok,
        "irregular": irregular,
        "af_like": af_like,
        "clean": clean,
    }


def gate_mask(gates_dict: dict[str, np.ndarray]) -> np.ndarray:
    mask = None
    for key, value in gates_dict.items():
        if key in ("relaxed", "mild", "irregular", "af_like"):   # informativas
            continue
        v = np.asarray(value, dtype=bool)
        mask = v.copy() if mask is None else (mask & v)
    return np.zeros(0, dtype=bool) if mask is None else mask


def record_long_rr(rr_prev, cfg: SvpbConfig = DEFAULT) -> float:
    """Ciclo sinusal de referencia del **registro**, para juzgar rachas.

    La mediana local no sirve cuando la ectopia domina: en una taquicardia
    auricular sostenida (o en una bigeminia auricular de dos latidos) casi todos
    los intervalos son cortos y el ciclo de base desaparece del entorno. El
    percentil 85 de los RR del registro entero sigue apuntando al ciclo sinusal
    —los intervalos largos son justamente los no ectópicos— y se acota a
    ``run_long_max_ratio`` veces la mediana para no inventar un ciclo lento en un
    registro enteramente taquicárdico.
    """
    rr = np.asarray(rr_prev, dtype=np.float64)
    rr = rr[np.isfinite(rr) & (rr > 0.2) & (rr < 3.0)]
    if rr.size < 5:
        return 0.0
    med = float(np.median(rr))
    p85 = float(np.percentile(rr, 85))
    return float(np.clip(p85, med, cfg.run_long_max_ratio * med))


def tachycardia_runs(rr_prev, qrs_ok, width_ok, cfg: SvpbConfig = DEFAULT,
                     rr_long: float | None = None,
                     af_like: np.ndarray | None = None,
                     p_present: np.ndarray | None = None) -> np.ndarray:
    """Rachas supraventriculares: dupletas, salvas, TSV y taquicardia auricular.

    Una extrasístole aislada se decide latido a latido (prematuridad + pausa),
    pero en una **racha** el ritmo ya no es sinusal y esa regla no puede
    aplicarse: la extrasístole siguiente no es prematura respecto de la
    anterior, y la pausa sólo aparece al final. Por eso se marcan como S todos
    los latidos de la racha cuando:

      * el QRS es el del paciente (angosto y correlacionado, ``qrs_ok`` y
        ``width_ok``), así que la racha es supraventricular y no ventricular;
      * el RR de la racha es corto frente al **ciclo del registro**
        (``run_rel`` × :func:`record_long_rr`), no frente al entorno inmediato;
      * el RR es regular (``run_regularity``), lo que descarta un tramo de FA;
      * y la racha **empieza de golpe** (el RR previo supera
        ``run_onset_ratio`` veces el ciclo del registro) o es larga y rápida
        (≥ ``run_long_beats`` latidos con RR ≤ ``run_rr_max_s``). Sin esta
        condición, una taquicardia sinusal sostenida se convertiría en cientos
        de S.

    Las rachas que caen en fibrilación auricular (``af_like``) se descartan si
    no hay onda P visible: ahí el RR corto es el ritmo, no una salva. Y si las
    rachas cubren más de ``run_max_fraction`` del registro, no son rachas sino
    el ritmo de base: se conservan sólo las de inicio abrupto (paroxísticas) y
    las **sostenidas** que son rápidas y no muestran P sinusal
    (``run_sustained_p_max``) —un ritmo auricular ectópico sostenido, que el
    cardiólogo anota como S en cada latido—; si hasta ésas superan
    ``run_hard_fraction``, se devuelve vacío.
    """
    rr = np.asarray(rr_prev, dtype=np.float64)
    n = rr.size
    out = np.zeros(n, dtype=bool)
    parox = np.zeros(n, dtype=bool)
    sustained = np.zeros(n, dtype=bool)
    if n < cfg.run_min_beats:
        return out
    long_rr = float(rr_long) if rr_long else record_long_rr(rr, cfg)
    if long_rr <= 0:
        return out
    fast = (np.asarray(qrs_ok, dtype=bool) & np.asarray(width_ok, dtype=bool)
            & (rr <= cfg.run_rel * long_rr) & (rr > 0.2))
    i = 0
    while i < n:
        if not fast[i]:
            i += 1
            continue
        j = i
        while j < n and fast[j]:
            j += 1
        length = j - i
        if length >= cfg.run_min_beats:
            seg = rr[i:j]
            mean_rr = float(np.mean(seg))
            regular = seg.max() <= cfg.run_regularity * max(seg.min(), 1e-3)
            before = float(rr[i - 1]) if i >= 1 else 0.0
            abrupt = before >= cfg.run_onset_ratio * long_rr
            tachy = length >= cfg.run_long_beats and mean_rr <= cfg.run_rr_max_s
            in_af = False
            if af_like is not None:
                af = np.asarray(af_like, dtype=bool)[i:j]
                if af.size and af.mean() > 0.5:
                    p_ok = True
                    if p_present is not None:
                        p_ok = bool(np.asarray(p_present, dtype=bool)[i:j].any())
                    in_af = not p_ok
            if regular and (abrupt or tachy) and not in_af:
                out[i:j] = True
                if abrupt:
                    parox[i:j] = True
                # Una racha **sostenida** (no paroxística) que ocupa gran parte
                # del registro es un ritmo auricular ectópico, no un sinus: el
                # sinus rápido normalmente deja ver la P, y una FA ya se excluyó
                # por `af_like`. Sólo se conserva si es rápida y la P no se ve.
                no_sinus_p = True
                if p_present is not None:
                    seg_p = np.asarray(p_present, dtype=bool)[i:j]
                    no_sinus_p = (bool(seg_p.size)
                                  and float(seg_p.mean()) <= cfg.run_sustained_p_max)
                if tachy and not abrupt and no_sinus_p:
                    sustained[i:j] = True
        i = j
    if out.any() and out.mean() > cfg.run_max_fraction:
        keep = parox | sustained
        if not keep.any() or float(keep.mean()) > cfg.run_hard_fraction:
            return np.zeros(n, dtype=bool)
        return keep
    return out


def decide(rr_prev, rr_next, rr_sinus, widths, width_ref, qrs_corr, p_feat,
           model_prob, noise=None, noise_ref=None, rr_long: float | None = None,
           cfg: SvpbConfig = DEFAULT) -> dict:
    """Decisión S por latido: puntaje, compuertas y máscara final.

    Devuelve ``is_s`` (bool), ``score`` (puntaje ya enmascarado por las
    compuertas, 0 si no las pasa), el detalle de compuertas y puntajes, y la
    máscara de rachas (``runs``). ``rr_long`` es el ciclo sinusal del registro
    que usan las rachas (:func:`record_long_rr`) si no se pasa.
    """
    feats = feature_matrix(rr_prev, rr_next, rr_sinus, widths, width_ref,
                           qrs_corr, p_feat, model_prob, cfg)
    gts = gates(rr_prev, rr_next, rr_sinus, widths, width_ref, qrs_corr,
                feats[:, IDX["p_score"]], noise, noise_ref=noise_ref,
                model=feats[:, IDX["model_prob"]], cfg=cfg)
    mask = gate_mask(gts)
    # En una arritmia sostenida (FA) solo se admite S con evidencia fuerte: muy
    # prematuro y con P **visible pero anómala**. "P ausente" no vale aquí,
    # porque en la FA la P falta en todos los latidos.
    p_state = np.asarray(p_feat.get("p_state", np.zeros(rr_prev.size)), dtype=np.int8)
    rr_ratio = np.divide(np.asarray(rr_prev, dtype=np.float64),
                         np.maximum(np.asarray(rr_sinus, dtype=np.float64), 1e-6))
    strong = ((rr_ratio <= cfg.prematurity_strong)
              & (feats[:, IDX["p_score"]] >= cfg.p_strong)
              & (p_state == P_PRESENT))
    mask = mask & (~gts["af_like"] | strong)
    score = np.where(mask, feats[:, IDX["score"]], 0.0)
    is_s = score >= cfg.threshold
    # rachas: dupletas, salvas, TSV y taquicardia auricular. Se juzgan contra el
    # ciclo del registro porque dentro de la racha el RR local ES la racha.
    long_rr = float(rr_long) if rr_long else record_long_rr(rr_prev, cfg)
    runs = tachycardia_runs(rr_prev, gts["qrs_ok"], gts["width_ok"], cfg,
                            rr_long=long_rr, af_like=gts["af_like"],
                            p_present=(p_state == P_PRESENT))
    # Guarda de registro: sin ciclo sinusal estable (RR irregular latido a
    # latido) y sin onda P medible no hay forma de distinguir una extrasístole
    # de una fibrilación auricular; ahí sólo se admite prematuridad franca con
    # el modelo a favor.
    cv = rr_irregularity(rr_prev, rr_sinus)
    p_avail = float(np.mean(p_state != P_UNKNOWN)) if p_state.size else 0.0
    af_record = bool(cv.size and float(np.median(cv)) > cfg.af_cv
                     and p_avail < cfg.p_available_min)
    if af_record:
        # En FA no hay "rachas" que valgan: la respuesta ventricular es regular
        # a ratos por puro azar, así que se apagan (queda la prematuridad franca
        # con el modelo a favor).
        strict = ((rr_ratio <= cfg.af_prematurity)
                  & (feats[:, IDX["model_prob"]] >= cfg.af_model_min))
        is_s = is_s & strict
        runs = np.zeros(rr_prev.size, dtype=bool)
    is_s = is_s | runs
    # válvula: si esta corrida etiqueta una fracción imposible, se sube el
    # umbral al percentil correspondiente (nunca se etiqueta media base). Las
    # rachas de taquicardia se mantienen: son un diagnóstico de ritmo, no ruido.
    frac = float(is_s.mean()) if is_s.size else 0.0
    if is_s.size >= 200 and frac > cfg.max_s_fraction:
        cut = float(np.percentile(score[score > 0], 100.0 * (1.0 - cfg.max_s_fraction)))
        is_s = (score >= max(cfg.threshold, cut)) | runs
    return {"is_s": is_s, "score": score, "features": feats, "gates": gts,
            "runs": runs, "rr_long": long_rr, "af_record": af_record,
            "p_available": round(p_avail, 3)}


def decide_from_pwave(x, fs, peaks, widths, rr_prev, rr_next, rr_sinus,
                      qrs_corr, model_prob, noise=None, cfg: SvpbConfig = DEFAULT,
                      p_res: dict | None = None, base_cls: np.ndarray | None = None,
                      ) -> dict:
    """Igual que :func:`decide`, calculando antes las características de P."""
    if p_res is None:
        try:
            from holterai.legacy.app.signal.pwave import detect_p_waves
            p_res = detect_p_waves(x, fs, peaks)
        except Exception:
            p_res = None
    p_feat = p_evidence(peaks, fs, p_res, cls=base_cls, rr_local=rr_sinus, cfg=cfg)
    width_ref = np.full(np.size(rr_prev), _robust_width(widths))
    noise_arr = None if noise is None else np.asarray(noise, dtype=np.float64)
    noise_ref = _robust_noise(noise_arr)
    long_rr = record_long_rr(rr_prev, cfg)
    out = decide(rr_prev, rr_next, rr_sinus, widths, width_ref, qrs_corr, p_feat,
                 model_prob, noise=noise_arr, noise_ref=noise_ref,
                 rr_long=long_rr, cfg=cfg)
    out["p_feat"] = p_feat
    out["width_ref_s"] = float(width_ref[0]) if width_ref.size else 0.0
    out["noise_ref"] = noise_ref
    # se guardan las entradas para poder re-decidir con otro umbral o pesaje sin
    # volver a correr la señal (auditoría y barrido de umbrales en validación)
    out["p_res"] = p_res
    out["inputs"] = {
        "rr_prev": np.asarray(rr_prev, dtype=np.float64),
        "rr_next": np.asarray(rr_next, dtype=np.float64),
        "rr_sinus": np.asarray(rr_sinus, dtype=np.float64),
        "rr_long": long_rr,
        "widths": np.asarray(widths, dtype=np.float64),
        "width_ref": width_ref,
        "qrs_corr": np.asarray(qrs_corr, dtype=np.float64),
        "model_prob": np.asarray(model_prob, dtype=np.float64),
        "noise": noise_arr,
        "noise_ref": noise_ref,
    }
    return out


def _robust_noise(noise) -> float | None:
    """Mediana de ruido del registro (``None`` si no hay medida)."""
    if noise is None:
        return None
    v = np.asarray(noise, dtype=np.float64)
    v = v[np.isfinite(v) & (v >= 0)]
    return float(np.median(v)) if v.size else None


def _robust_width(widths) -> float:
    """Mediana del ancho de QRS del registro (0,09 s si no hay datos)."""
    w = np.asarray(widths, dtype=np.float64)
    w = w[np.isfinite(w) & (w > 0.02) & (w < 0.25)]
    return float(np.median(w)) if w.size else 0.09


# ---------------------------------------------------------------------------
# Del puntaje a la clase
# ---------------------------------------------------------------------------
def apply(cls: np.ndarray, is_s: np.ndarray, qrs_corr: np.ndarray,
          cfg: SvpbConfig = DEFAULT, support: np.ndarray | None = None) -> int:
    """Pasa a S los latidos marcados (N->S y V->S) y **retira** las S sin apoyo.

    Sólo se tocan N y V: F y Q se dejan como están (en FA la prematuridad no
    distingue una S). Un V con el QRS correlacionado con el sinusal es una S
    aberrante mal clasificada; el resto de los V se queda V.

    ``support`` es la máscara de compuertas clínicas del registro: si se pasa,
    una S del clasificador que **no** las cumple y que tampoco fue marcada por
    la capa se devuelve a N. Sin esto la decisión sólo puede añadir S —el
    clasificador calibrado etiqueta el 4 % de los latidos de un Holter de 24 h
    como S y la capa no podía quitarlos, que es la causa de las 25-30× más S
    que HScribe.
    """
    n = min(cls.size, is_s.size)
    changed = 0
    corr = np.asarray(qrs_corr, dtype=np.float64) if qrs_corr is not None else None
    sup = None if support is None else np.asarray(support, dtype=bool)
    for i in range(n):
        if cls[i] == 1 and not is_s[i]:
            if sup is not None and not sup[i] and i < sup.size:
                cls[i] = 0
                changed += 1
            continue
        if not is_s[i]:
            continue
        if cls[i] in (0, 1):
            if cls[i] != 1:
                cls[i] = 1
                changed += 1
        elif cls[i] == 2 and (corr is None or corr[i] >= cfg.qrs_corr_min):
            cls[i] = 1
            changed += 1
    return changed


def normal_lookalike_mask(cls: np.ndarray, corr: np.ndarray,
                          cfg: SvpbConfig = DEFAULT) -> np.ndarray:
    """V con QRS idéntico al sinusal y sin evidencia de S: en realidad es N."""
    cls = np.asarray(cls)
    corr = np.asarray(corr, dtype=np.float64)
    return (cls == 2) & (corr >= 0.94)


# ---------------------------------------------------------------------------
# Rachas: dupletas, TSV y taquicardia auricular
# ---------------------------------------------------------------------------
def run_summary(classes, rr_prev, cfg: SvpbConfig = DEFAULT) -> dict:
    """Dupletas, TSV (>= 3) y taquicardia auricular (>= 4 y FC >= 100 lpm).

    ``classes`` puede ser una lista de nombres ("N", "S", "V") o de códigos
    enteros (1 = S), como en el resto del proyecto.
    """
    cls = list(classes)
    rr = np.asarray(rr_prev, dtype=np.float64)
    is_s = np.array([(c == 1 or c == "S") for c in cls], dtype=bool)
    runs: list[dict] = []
    i = 0
    n = len(cls)
    while i < n:
        if not is_s[i]:
            i += 1
            continue
        j = i
        while j < n and is_s[j]:
            j += 1
        rrs = [float(v) for v in rr[i:j] if 0.2 <= float(v) <= 3.0]
        rate = 60.0 / (sum(rrs) / len(rrs)) if rrs else 0.0
        runs.append({"length": j - i, "start": i, "end": j - 1, "rate_bpm": rate})
        i = j
    lengths = [r["length"] for r in runs]
    at = [r for r in runs if r["length"] >= 4 and r["rate_bpm"] >= 100.0]
    return {
        "duplets": sum(1 for v in lengths if v == 2),
        "tsv": sum(1 for v in lengths if v >= 3),
        "runs": lengths,
        "longest": max(lengths) if lengths else 0,
        "atrial_tachycardia": len(at),
        "atrial_tachycardia_beats": sum(r["length"] for r in at),
        "fastest_bpm": round(max((r["rate_bpm"] for r in runs), default=0.0), 1),
    }


# ---------------------------------------------------------------------------
# Resumen para el informe
# ---------------------------------------------------------------------------
def summary(decision: dict, classes, p_feat: dict | None = None,
           cfg: SvpbConfig = DEFAULT) -> dict:
    """Números agregados de la capa S (para ``report.json``)."""
    feats = decision["features"]
    score = decision["score"]
    is_s = np.asarray(decision["is_s"], dtype=bool)
    runs = run_summary(classes, feats[:, IDX["rr_prev"]], cfg)
    n_runs = int(np.sum(np.asarray(decision.get("runs", np.zeros(is_s.size, bool)),
                                  dtype=bool)))
    out = {
        "threshold": round(float(cfg.threshold), 3),
        "prematurity": float(cfg.prematurity),
        "n_flagged": int(is_s.sum()),
        "n_runs": n_runs,
        "frac_flagged": round(float(is_s.mean()), 4) if is_s.size else 0.0,
        "score_p50": round(float(np.percentile(score, 50)), 3) if score.size else 0.0,
        "score_p95": round(float(np.percentile(score, 95)), 3) if score.size else 0.0,
        "rr_long_s": round(float(decision.get("rr_long", 0.0)), 3),
        "atrial_evidence": {
            "p_available": decision.get("p_available", 0.0),
            "af_record": bool(decision.get("af_record", False)),
        },
        "criteria": {
            "premature": int(np.sum(decision["gates"]["premature"])),
            "mild": int(np.sum(decision["gates"].get(
                "mild", np.zeros(is_s.size, dtype=bool)))),
            "qrs_ok": int(np.sum(decision["gates"]["qrs_ok"])),
            "pause_ok": int(np.sum(decision["gates"]["pause_ok"])),
            "irregular": int(np.sum(decision["gates"].get(
                "irregular", np.zeros(is_s.size, dtype=bool)))),
            "p_evidence": int(np.sum(feats[:, IDX["p_score"]] >= 0.5)),
        },
        "runs": runs,
    }
    if p_feat is not None:
        st = np.asarray(p_feat["p_state"])
        out["p_wave"] = {
            "present": int(np.sum(st == P_PRESENT)),
            "absent": int(np.sum(st == P_ABSENT)),
            "unknown": int(np.sum(st == P_UNKNOWN)),
            "pr_ref_ms": round(float(p_feat["pr_ref_ms"]), 1)
            if np.isfinite(p_feat["pr_ref_ms"]) else None,
        }
    return out
