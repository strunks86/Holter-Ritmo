"""Morfología del QRS ancho: bloqueo de rama, IVCD y preexcitación.

Duración
--------
La duración se mide con `features.qrs_duration_multichannel` sobre el latido
promediado (mediana por minuto): inicio más temprano y fin más tardío entre
los canales disponibles. `>= 120 ms` es QRS ancho y `110-119 ms` incompleto.

Derivaciones del H3+
--------------------
Lo único que declara HScribe es el nombre de las dos derivaciones, y está
verificado en los `ExamSettings.xml` de los 15 exámenes de `FDrepo` que traen
ese archivo: ``<Lead>II</Lead>`` y ``<Lead>V</Lead>``. No dice cuál de los
canales exportados es ``II``, ni numera la precordial (V1 frente a V5), ni hay
documento de HScribe que lo aclare. Por eso:

* la duración y la detección de "QRS ancho" **no** dependen de esa asignación;
* para dar nombre al patrón (BRD frente a BRI) hace falta saber qué canal es
  una precordial derecha (V1/V2) y cuál una lateral (I, V5, V6). Si el
  llamador no lo declara (`roles`), el módulo infiere la posición por la
  polaridad del latido estrecho de referencia y lo anota en `assumed`. Si no
  hay base para inferirla, el patrón queda `indeterminado` con la lista
  `missing`: no se adivina.

Criterios (documentados en el docstring de `classify_beat`)
----------------------------------------------------------
* BRD: QRS ancho con deflexión terminal positiva en precordial derecha
  (R' de un rsR' o R terminal dominante) y S terminal ancha en la lateral.
* BRI: QRS ancho con complejo negativo dominante (QS o rS con r pequeña) en
  precordial derecha y R ancha monofásica o con muescas en la lateral, con
  tiempo de deflexión intrinsecoide >= 60 ms.
* IVCD: 110 ms o más sin cumplir ninguno de los dos (definición ESC, que
  también incluye el QRS de 110-119 ms no bloqueado).
* Preexcitación probable: 110-160 ms con inicio empastado (onda delta) y sin
  patrón de bloqueo de rama. El PR corto solo se informa si hay P medida.

Calibración (T44)
-----------------
Los umbrales se ajustaron con 570 registros de PTB-XL a 500 Hz (100 CLBBB,
100 CRBBB, 100 IVCD, 100 NORM, 100 1AVB, 70 WPW) separando por paciente en
dos mitades iguales: se elige en una mitad y se informa en la otra (ver el
informe de la tarea). El criterio de rama exige la **morfología**, no solo el
ancho, y el ancho de cada derivación cuenta por separado porque la unión de
canales se infla cuando una derivación recoge la onda T o ruido.

| combinación | BRI Se/Esp | BRD Se/Esp | IVCD Se |
| --- | --- | --- | --- |
| II+V1 (2 derivaciones) | 0,83/0,89 | 0,92/0,96 | 0,40 |
| I+II+V1 | 0,98/0,89 | 0,94/0,96 | 0,50 |
| II+V1+V5 | 0,94/0,89 | 0,96/0,96 | 0,35 |
| II+V5 (2 derivaciones) | 0,56/0,91 | 0,28/0,88 | 0,27 |

Cifras de la mitad de prueba (285 registros, pacientes no usados para elegir
los umbrales); la mitad de desarrollo da 0,83/0,93 y 0,92/0,97.
La especificidad del BRI se queda en 0,89 en **todas** las combinaciones,
también con tres derivaciones: los falsos positivos son 13 registros
etiquetados IVCD, 9 WPW y 4 de 1AVB con V1 de morfología izquierda. Con dos
derivaciones no se separan del BRI (el IVCD de PTB-XL es, por definición, el
que no cumple criterios de rama en las 12 derivaciones) y añadir una tercera
no los separa tampoco: es un límite de la etiqueta, no del umbral; el módulo
informa la morfología medida y esta limitación, no la etiqueta clínica. El WPW
solo se detecta cuando hay preexcitación en el registro (el PR mediano de esos
79 registros es 151 ms, igual que en NORM: la mayoría no está preexcitada en
reposo). En el IVCD el resto se reparte entre registros que miden 110 ms o
menos y casos con V1 de morfología izquierda. La duración del QRS medida así
tiene un error absoluto medio de 10,4-11,1 ms frente a las anotaciones de LUDB
(199 registros; ver el informe de la tarea).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import yaml

from .features import (
    QRS_INCOMPLETE_MS,
    QRS_WIDE_MS,
    frontal_axis_deg,
    median_beat,
    qrs_duration_multichannel,
    qrs_onset_shape,
)

RIGHT_PRECORDIAL = ("V1", "V2")
LEFT_LATERAL = ("I", "V5", "V6", "AVL")

# --- criterios de rama (calibrados en PTB-XL, ver docstring del módulo) ------
#: R' o R terminal que ocupa al menos esta fracción de la amplitud del QRS.
BRD_TERMINAL_POS = 0.25
#: media del último 35 % del QRS en la precordial derecha (positiva en el R').
BRD_TERMINAL = 0.10
#: deflexiones relevantes mínimas en la precordial derecha (rsR' = 2).
BRD_EXTREMA = 2
#: o tiempo al pico terminal (R' tardío) cuando el rsR' queda fusionado.
BRD_PEAK_TIME = 80.0
#: S terminal de la derivación lateral / ancho de la precordial derecha.
BRD_LATERAL_S = 0.05
#: fuerzas netas negativas de la precordial derecha en el BRI (QS/rS amplio).
BRI_NET = -0.25
#: media del tramo final en la precordial derecha: en el BRI no hay R terminal.
BRI_TERMINAL = 0.05
#: deflexión intrinsecoide de la derivación izquierda (R ancha del BRI).
BRI_PEAK_TIME = 50.0
#: S terminal máxima de la derivación lateral para el BRI (R monofásica). Se
#: calibra en 0,60 y no en 0,20: con la ventana terminal actual (último 4 % del
#: QRS) una S de 0,2-0,5 recorta la sensibilidad del BRI 10-25 puntos sin ganar
#: especificidad (II+V1+V5: 0,94/0,89 con 0,60 frente a 0,73/0,91 con 0,20 en
#: la mitad de prueba). Solo se rechaza una S inequívoca.
BRI_LATERAL_S = 0.60
#: Por debajo de este ancho, con solo II como referencia lateral, el BRI y el
#: IVCD de PTB-XL no se separan (ver la calibración del módulo).
BRI_IVCD_OVERLAP_MS = 150.0

# --- onda delta (preexcitación) --------------------------------------------
#: pendiente del primer tramo sobre la pendiente máxima del complejo.
DELTA_HEAD_RATIO = 0.40
#: ascenso hasta la mitad del pico (lento en la onda delta).
DELTA_UPSTROKE_MS = 40.0
#: posición del máximo de pendiente dentro del QRS (tardía en la onda delta).
DELTA_SLOPE_FRAC = 0.44
#: fracción del área del QRS que aporta el primer tramo (pequeña con delta).
DELTA_INITIAL_FRAC = 0.28
#: ancho en el que se busca la preexcitación (el delta puro no llega a 200 ms).
DELTA_MIN_WIDTH = 110.0
DELTA_MAX_WIDTH = 185.0
#: PR corto que refuerza la preexcitación cuando hay onda P medida.
DELTA_SHORT_PR = 130.0

#: Lo que HScribe permite afirmar sobre los canales del H3+. El `ExamSettings`
#: de los 15 exámenes de `FDrepo` que lo traen declara exactamente dos
#: derivaciones, `<Lead>II</Lead>` y `<Lead>V</Lead>`; no identifica qué canal
#: exportado es cada una ni cuál es la precordial (V1 o V5).
H3P_LEAD_NOTE = (
    "HScribe declara solo dos derivaciones, 'II' y 'V' (ExamSettings); no dice "
    "qué canal es cada una ni qué precordial es: no se asigna por conjetura"
)

PATTERN_NAMES = {
    "estrecho": "QRS estrecho",
    "brd": "bloqueo de rama derecha",
    "brd_incompleto": "bloqueo de rama derecha incompleto",
    "bri": "bloqueo de rama izquierda",
    "bri_incompleto": "bloqueo de rama izquierda incompleto",
    "ivcd": "trastorno inespecífico de la conducción intraventricular",
    "preexcitacion": "preexcitación probable (onda delta)",
    "indeterminado": "morfología no concluyente",
}

# --- criterios AHA 2009 (T128): umbrales en configs/criteria/aha_ramas.yaml ---
AHA_RULES_PATH = Path(__file__).resolve().parents[5] / "configs" / "criteria" / "aha_ramas.yaml"


@dataclass(frozen=True)
class AhaRules:
    """Umbrales AHA de bloqueo de rama y preexcitación (ms salvo indicación)."""

    qrs_complete_ms: float = 120.0
    qrs_incomplete_ms: float = 110.0
    tciv_min_ms: float = 110.0
    v1_broad_r_peak_ms: float = 50.0
    lateral_s_ms: float = 40.0
    lateral_s_longer_than_r: bool = True
    lateral_r_peak_ms: float = 60.0
    pr_short_ms: float = 120.0
    delta_qrs_min_ms: float = 120.0
    q_amp_frac: float = 0.10

    @classmethod
    def from_mapping(cls, data: dict) -> "AhaRules":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"claves desconocidas en los criterios AHA: {unknown}")
        return cls(**{k: data[k] for k in data})


def load_aha_rules(path: str | Path | None = None) -> AhaRules:
    """Lee los umbrales AHA del YAML de criterios (por defecto el del repositorio)."""
    p = Path(path) if path is not None else AHA_RULES_PATH
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return AhaRules.from_mapping(data)


@dataclass
class Morphology:
    """Resultado de la clasificación morfológica de un latido promediado."""

    width_ms: float
    wide: bool
    incomplete: bool
    pattern: str
    confidence: float
    complete: bool = True
    measurable: bool = True
    roles: dict[int, str] = field(default_factory=dict)
    assumed: bool = False
    criteria: dict[str, float] = field(default_factory=dict)
    applied: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "width_ms": self.width_ms,
            "wide": self.wide,
            "incomplete": self.incomplete,
            "pattern": self.pattern,
            "pattern_name": PATTERN_NAMES.get(self.pattern, self.pattern),
            "confidence": round(self.confidence, 3),
            "complete": self.complete,
            "measurable": self.measurable,
            "roles": {str(k): v for k, v in self.roles.items()},
            "assumed_lead_position": self.assumed,
            "criteria": {k: round(float(v), 3) for k, v in self.criteria.items()},
            "applied": list(self.applied),
            "missing": list(self.missing),
            "notes": list(self.notes),
        }


def channel_features(y: np.ndarray, fs: float, onset: int, offset: int) -> dict:
    """Rasgos de morfología de un canal dentro de los límites del QRS."""
    y = np.asarray(y, dtype=np.float64).ravel()
    onset = max(0, int(onset))
    offset = min(y.size, int(offset))
    if offset - onset < 3:
        return {"amp_mv": 0.0, "valid": False}
    seg = y[onset:offset]
    base = float(np.median(np.concatenate([y[max(0, onset - int(0.03 * fs)):onset],
                                           y[offset:offset + int(0.03 * fs)]])))
    seg = seg - base
    amp = float(np.max(seg) - np.min(seg))
    out = {"amp_mv": amp, "valid": amp > 1e-6}
    if not out["valid"]:
        return out
    n = seg.size
    k1 = max(1, int(round(0.30 * n)))
    k2 = max(1, int(round(0.35 * n)))
    out["initial"] = float(np.mean(seg[:k1])) / amp
    out["terminal"] = float(np.mean(seg[-k2:])) / amp
    last = seg[-max(1, int(round(0.04 * fs))):]
    out["terminal_pos"] = float(np.max(last)) / amp
    out["terminal_neg"] = float(-np.min(last)) / amp
    out["peak_time_ms"] = float(np.argmax(np.abs(seg)) / fs * 1000.0)
    out["end_ms"] = float(n / fs * 1000.0)
    d = np.diff(seg, prepend=seg[:1])
    ad = np.abs(d)
    dmax = float(ad.max())
    out["slope_max"] = dmax
    head = ad[: max(2, int(round(0.04 * fs)))]
    out["onset_slur"] = float(head.max() / dmax) if dmax > 0 else 1.0
    # número de deflexiones relevantes (muescas, R'): extremos locales de |señal|
    # separados al menos 20 ms y por encima del 20 % de la amplitud
    out["extrema"] = float(_count_deflections(seg, fs, 0.20 * amp, 0.020))
    pos_area = float(np.sum(np.clip(seg, 0, None)))
    neg_area = float(-np.sum(np.clip(seg, None, 0)))
    out["pos_area"] = pos_area
    out["neg_area"] = neg_area
    out["net"] = (pos_area - neg_area) / max(amp * n, 1e-12)
    return out


def _count_deflections(seg: np.ndarray, fs: float, thr: float, min_sep_s: float) -> int:
    """Cuenta picos y valles relevantes del segmento QRS."""
    n = seg.size
    if n < 3:
        return 0
    sep = max(1, int(round(min_sep_s * fs)))
    cand = []
    for i in range(1, n - 1):
        if abs(seg[i]) < thr:
            continue
        if (seg[i] >= seg[i - 1] and seg[i] > seg[i + 1]) or (
                seg[i] <= seg[i - 1] and seg[i] < seg[i + 1]):
            cand.append(i)
    if not cand:
        return 0
    cand.sort(key=lambda i: -abs(seg[i]))
    taken: list[int] = []
    for i in cand:
        if all(abs(i - j) >= sep for j in taken):
            taken.append(i)
    return len(taken)


def _resolve_roles(n_channels: int, roles: dict[int, str] | None,
                   baseline: np.ndarray | None, fs: float, center: int) -> tuple[dict[int, str], bool, list[str]]:
    """Posición de cada canal: declarada por el llamador o inferida de la polaridad."""
    notes: list[str] = []
    if roles:
        return {int(k): str(v).upper() for k, v in roles.items()}, False, notes
    if baseline is None or n_channels < 2:
        return {}, False, ["sin mapa de derivaciones declarado"]
    feats = []
    for ch in range(n_channels):
        res = qrs_duration_multichannel(baseline[:, [ch]], fs, center)
        feat = channel_features(baseline[:, ch], fs, res["onset"], res["offset"])
        feats.append(feat)
    big = max(range(n_channels), key=lambda c: feats[c].get("amp_mv", 0.0))
    if feats[big].get("amp_mv", 0.0) <= 1e-6:
        return {}, False, ["canales sin amplitud suficiente para inferir posición"]
    out: dict[int, str] = {}
    if feats[big].get("net", 0.0) < 0:
        out[big] = "V1"
        notes.append(f"canal {big} tratado como precordial derecha por polaridad negativa")
        rest = [c for c in range(n_channels) if c != big]
        if rest:
            other = max(rest, key=lambda c: feats[c].get("amp_mv", 0.0))
            if feats[other].get("net", 0.0) >= 0:
                out[other] = "V5"
                notes.append(f"canal {other} tratado como lateral por polaridad positiva")
    else:
        out[big] = "V5"
        notes.append(f"canal {big} tratado como lateral por polaridad positiva")
        rest = [c for c in range(n_channels) if c != big]
        if rest:
            other = min(rest, key=lambda c: feats[c].get("net", 0.0))
            if feats[other].get("net", 0.0) < 0:
                out[other] = "V1"
                notes.append(f"canal {other} tratado como precordial derecha por polaridad negativa")
    return out, True, notes


def _channel_for(roles: dict[int, str], names: tuple[str, ...]) -> int | None:
    for ch, name in roles.items():
        if name.upper() in names:
            return ch
    return None


def _classify_lateral_only(morph: Morphology, left: dict, width: float) -> Morphology:
    """Patrón estimado solo con una derivación izquierda (I, II, V5, V6).

    Con una sola derivación izquierda el patrón es más débil: la R ancha
    monofásica con deflexión intrinsecoide tardía sugiere BRI y la S terminal
    ancha sugiere BRD. Se marca `confidence` baja y queda anotado en `missing`.
    """
    morph.applied.append("clasificación solo por derivación izquierda")
    if left["peak_time_ms"] >= 60 and left["terminal_neg"] < 0.12:
        morph.pattern = "bri"
        morph.confidence = 0.5
        morph.applied.append("R ancha con deflexión intrinsecoide tardía")
    elif left["terminal_neg"] >= 0.20:
        morph.pattern = "brd"
        morph.confidence = 0.45
        morph.applied.append("S terminal ancha")
    else:
        morph.pattern = "ivcd"
        morph.confidence = 0.35
    if morph.incomplete:
        morph.applied.append("QRS incompleto (110-119 ms)")
        morph.confidence = min(morph.confidence, 0.4)
    return morph


def classify_beat(beat: np.ndarray, fs: float, center: int,
                  roles: dict[int, str] | None = None,
                  baseline: np.ndarray | None = None,
                  pr_ms: float | None = None,
                  rules: AhaRules | None = None) -> Morphology:
    """Clasifica la morfología de un latido promediado `(muestras, canales)`.

    `baseline` es el latido promediado del ritmo de base (estrecho); se usa
    solo para inferir la posición de los canales cuando `roles` es None.
    `pr_ms` es opcional: sin onda P medida no se afirma preexcitación por PR.
    Con `rules` (criterios AHA, `load_aha_rules()`) la decisión sigue la
    sección H del tratado; sin `rules` queda el clasificador calibrado de T44.
    """
    y = np.asarray(beat, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    n_ch = y.shape[1]
    dur = qrs_duration_multichannel(y, fs, center)
    width = float(dur["width_ms"])
    morph = Morphology(width_ms=width, wide=bool(dur["wide"]),
                       incomplete=bool(dur["incomplete"]), pattern="estrecho",
                       confidence=0.5, measurable=bool(dur.get("measurable", True)))
    if not morph.measurable:
        morph.pattern = "indeterminado"
        morph.confidence = 0.0
        morph.missing.append("QRS no medible: canal(es) dominados por ruido")
        return morph
    morph.criteria["width_ms"] = width
    for p in dur["per_channel"]:
        morph.criteria[f"width_ch{p['channel']}_ms"] = p["width_ms"]

    role_map, assumed, notes = _resolve_roles(n_ch, roles, baseline, fs, center)
    morph.roles = role_map
    morph.assumed = assumed
    morph.notes.extend(notes)
    if not role_map.get(0):
        morph.missing.append(H3P_LEAD_NOTE)

    feats = {ch: channel_features(y[:, ch], fs, dur["onset"], dur["offset"])
             for ch in range(n_ch)}
    for ch, feat in feats.items():
        if feat.get("valid"):
            morph.criteria[f"amp_ch{ch}_mv"] = feat["amp_mv"]
            morph.criteria[f"peak_time_ch{ch}_ms"] = feat["peak_time_ms"]
            morph.criteria[f"terminal_ch{ch}"] = feat["terminal"]
            morph.criteria[f"extrema_ch{ch}"] = feat["extrema"]

    if rules is not None:
        return _classify_aha(morph, y, fs, dur, role_map, pr_ms, rules)

    if width < QRS_INCOMPLETE_MS:
        morph.pattern = "estrecho"
        morph.confidence = 0.9 if width >= 60 else 0.6
        return morph

    right = _channel_for(role_map, RIGHT_PRECORDIAL)
    left = _channel_for(role_map, LEFT_LATERAL)
    if left is None:
        limb = _channel_for(role_map, ("II",))
        if limb is not None and limb != right:
            left = limb
            morph.applied.append("sin lateral: se usa II como referencia izquierda")
    right_feat = feats.get(right) if right is not None else None
    left_feat = feats.get(left) if left is not None else None
    if right_feat is None or not right_feat.get("valid"):
        if left_feat is not None and left_feat.get("valid"):
            morph.missing.append("precordial derecha (V1/V2) no identificada: patrón "
                                 "estimado solo con la referencia izquierda")
            return _classify_lateral_only(morph, left_feat, width)
        morph.missing.append("precordial derecha (V1/V2) no identificada: no se distingue BRD de BRI")
        morph.pattern = "indeterminado"
        morph.confidence = 0.2
        return morph

    morph.applied.append("QRS ancho con precordial derecha identificada")
    lateral_ok = left_feat is not None and left_feat.get("valid", False)
    if not lateral_ok:
        morph.missing.append("lateral (I/V5/V6) no identificada: criterio de S ancha no evaluado")

    width_right = float(dur["per_channel"][right]["width_ms"]) if right is not None else width
    width_left = (float(dur["per_channel"][left]["width_ms"]) if lateral_ok else 0.0)
    terminal = right_feat["terminal"]
    terminal_pos = right_feat["terminal_pos"]
    net = right_feat["net"]
    morph.criteria["width_right_ms"] = width_right
    morph.criteria["width_left_ms"] = width_left

    # --- bloqueo de rama derecha: R' terminal en la precordial derecha ------
    brd_core = (terminal_pos >= BRD_TERMINAL_POS and terminal >= BRD_TERMINAL
                and (right_feat["extrema"] >= BRD_EXTREMA
                     or right_feat["peak_time_ms"] >= BRD_PEAK_TIME))
    brd = (brd_core and (width >= QRS_WIDE_MS or width_right >= QRS_INCOMPLETE_MS)
           and net >= -0.10)
    if brd_core:
        morph.applied.append("R' dominante al final del QRS en precordial derecha")
    if brd and lateral_ok and left_feat["terminal_neg"] >= BRD_LATERAL_S:
        morph.applied.append("S terminal en la referencia izquierda")

    # --- bloqueo de rama izquierda: QS/rS amplio y R ancha lateral ---------
    left_name = str(role_map.get(left, "")).upper() if left is not None else ""
    if lateral_ok:
        # la R del BRI es ancha y sin S terminal; con II como referencia (no es
        # una lateral verdadera) solo se exige el retraso intrinsecoide, porque
        # en el BRI es frecuente el rS en II con eje izquierdo
        lateral_s_ok = (left_feat["terminal_neg"] <= BRI_LATERAL_S
                        if left_name in LEFT_LATERAL else True)
        lateral_ridge = (left_feat["peak_time_ms"] >= BRI_PEAK_TIME and lateral_s_ok)
    else:
        lateral_ridge = False
    right_left_ratio = width_right / width_left if width_left > 0 else 1.0
    bri_core = (net <= BRI_NET and terminal <= BRI_TERMINAL and lateral_ridge)
    bri = (bri_core and (width >= QRS_WIDE_MS or width_right >= QRS_INCOMPLETE_MS))
    morph.criteria["width_ratio_right_left"] = round(right_left_ratio, 3)
    if net <= BRI_NET:
        morph.applied.append("complejo negativo dominante en precordial derecha")
    if bri and lateral_ok:
        morph.applied.append("deflexión intrinsecoide >= 60 ms en la referencia izquierda")

    # --- eje frontal (solo si están I y II) --------------------------------
    axis = None
    ch_i = _channel_for(role_map, ("I",))
    ch_ii = _channel_for(role_map, ("II",))
    if ch_i is not None and ch_ii is not None and ch_i != ch_ii:
        f_i, f_ii = feats.get(ch_i), feats.get(ch_ii)
        if f_i and f_ii and f_i.get("valid") and f_ii.get("valid"):
            axis = frontal_axis_deg(f_i["net"], f_ii["net"])
            morph.criteria["frontal_axis_deg"] = round(axis, 1)

    # --- preexcitación: inicio empastado (onda delta) ----------------------
    shapes = {ch: qrs_onset_shape(y[:, ch], fs, dur["onset"], dur["offset"])
              for ch in range(n_ch)}
    delta = _delta_evidence(feats, shapes, width, pr_ms)
    morph.criteria.update(delta["criteria"])
    morph.applied.extend(delta["applied"])
    morph.missing.extend(delta["missing"])
    if delta["strong"] and not brd_core and not bri_core:
        morph.pattern = "preexcitacion"
        morph.confidence = 0.45 if pr_ms is None else (0.75 if delta["short_pr"] else 0.6)
        if delta["short_pr"]:
            morph.applied.append("PR corto")
        if pr_ms is None:
            morph.missing.append("PR no medido: preexcitación solo probable")
        # eje desviado: apoya la preexcitación (una vía accesoria cambia el
        # vector inicial), pero no la decide por sí solo
        if axis is not None and (axis <= -45.0 or axis >= 110.0):
            morph.applied.append(f"eje frontal desviado ({axis:.0f}°)")
            morph.confidence = min(0.85, morph.confidence + 0.1)
    elif brd:
        morph.pattern = "brd"
        morph.confidence = 0.78 if brd_core else 0.6
    elif bri:
        morph.pattern = "bri"
        morph.confidence = 0.75 if bri_core else 0.6
        if lateral_ok and left_name not in LEFT_LATERAL and width < BRI_IVCD_OVERLAP_MS:
            # sin una lateral verdadera (I/V5/V6) y con un QRS no muy ancho, la
            # morfología de V1 sola no separa el BRI del IVCD de PTB-XL: se
            # informa la ambigüedad en vez de elegir por el llamador
            morph.missing.append(
                "BRI sin lateral verdadera (solo II) y QRS < 150 ms: no se "
                "distingue de un IVCD")
    elif width >= QRS_INCOMPLETE_MS:
        # ESC: 110-119 ms sin morfología de rama también es IVCD
        morph.pattern = "ivcd"
        if morph.incomplete:
            morph.applied.append("QRS incompleto (110-119 ms)")
            morph.confidence = 0.35
        else:
            morph.confidence = 0.4
    else:
        morph.pattern = "estrecho"
        morph.confidence = 0.6

    if morph.pattern in ("bri", "brd") and morph.incomplete:
        morph.applied.append("QRS incompleto (110-119 ms)")
        morph.confidence = min(morph.confidence, 0.7)
    if morph.assumed and morph.pattern in ("bri", "brd"):
        # el nombre de las derivaciones se infirió por polaridad: el patrón de
        # rama depende de cuál es la precordial derecha, así que no se informa
        # con la confianza de un caso con derivaciones declaradas
        morph.confidence = min(morph.confidence, 0.6)
        morph.missing.append("patrón de rama con posición de derivaciones inferida: "
                             "confirmar con las derivaciones declaradas")
    return morph


# ---------------------------------------------------------------------------
# Criterios AHA 2009, parte III (T128). Umbrales: configs/criteria/aha_ramas.yaml
# ---------------------------------------------------------------------------
def aha_lead_features(y: np.ndarray, fs: float, onset: int, offset: int,
                      rules: AhaRules) -> dict:
    """Rasgos de una derivación para los criterios AHA, dentro del QRS `[onset, offset)`.

    Además de `channel_features`: ``r_peak_ms`` (pico de la R positiva desde el
    inicio), ``q_present`` (negatividad antes de ese pico mayor que
    ``rules.q_amp_frac``), ``s_final_ms`` (negatividad final hasta el fin) y
    ``r_dur_ms`` (parte positiva). Sin R positiva el pico de R vale 0 y no hay q.
    """
    feat = channel_features(y, fs, onset, offset)
    if not feat.get("valid"):
        return {"valid": False}
    yv = np.asarray(y, dtype=np.float64).ravel()
    on, off = max(0, int(onset)), min(yv.size, int(offset))
    g = int(round(0.03 * fs))
    around = np.concatenate([yv[max(0, on - g):on], yv[off:off + g]])
    base = float(np.median(around)) if around.size else 0.0
    seg = yv[on:off] - base
    amp = feat["amp_mv"]
    n = seg.size
    if float(seg.max()) > 0:
        ipk = int(np.argmax(seg))
        feat["r_peak_ms"] = float(ipk / fs * 1000.0)
        pre = seg[:ipk]
        feat["q_present"] = bool(pre.size and float(pre.min()) < -rules.q_amp_frac * amp)
        pos = np.flatnonzero(seg > 0)
        r_start = int(pos[0])
        neg = np.flatnonzero(seg[ipk:] < 0)
        if neg.size:
            s_start = ipk + int(neg[0])
            feat["s_final_ms"] = float((n - s_start) / fs * 1000.0)
            feat["r_dur_ms"] = float((s_start - r_start) / fs * 1000.0)
        else:
            feat["s_final_ms"] = 0.0
            feat["r_dur_ms"] = float((n - r_start) / fs * 1000.0)
    else:
        feat["r_peak_ms"] = 0.0
        feat["q_present"] = False
        neg = np.flatnonzero(seg < 0)
        feat["s_final_ms"] = float((n - int(neg[0])) / fs * 1000.0) if neg.size else 0.0
        feat["r_dur_ms"] = 0.0
    return feat


def _r_prime_terminal(feat: dict) -> bool:
    """R' terminal en V1 (rsR'/rSR'), con los mismos rasgos del BRD calibrado en T44."""
    return bool(feat.get("terminal_pos", 0.0) >= BRD_TERMINAL_POS
                and feat.get("terminal", 0.0) >= BRD_TERMINAL
                and (feat.get("extrema", 0.0) >= BRD_EXTREMA
                     or feat.get("peak_time_ms", 0.0) >= BRD_PEAK_TIME))


def _classify_aha(morph: Morphology, y: np.ndarray, fs: float, dur: dict,
                  role_map: dict[int, str], pr_ms: float | None,
                  rules: AhaRules) -> Morphology:
    """Decisión AHA 2009 (sección H de tratados/CRITERIOS-TRATADOS.md).

    Orden: preexcitación (PR corto + onda delta + QRS > 120 ms), luego BRD
    (V1 con rsR' o R ancha, y S final lateral), BRI (lateral sin q con pico de R
    > 60 ms), y por último TCIV inespecífico (> 110 ms sin criterios). Hemibloqueos
    y eje no se evalúan: con tres canales no hay plano frontal fiable.
    """
    width = morph.width_ms
    if width < rules.qrs_incomplete_ms:
        morph.pattern = "estrecho"
        morph.confidence = 0.9 if width >= 60 else 0.6
        return morph
    onset, offset = int(dur["onset"]), int(dur["offset"])
    n_ch = y.shape[1]
    feats = {ch: channel_features(y[:, ch], fs, onset, offset) for ch in range(n_ch)}
    shapes = {ch: qrs_onset_shape(y[:, ch], fs, onset, offset) for ch in range(n_ch)}

    # 1) preexcitación: exige PR medido, onda delta y QRS > 120 ms
    delta = _delta_evidence(feats, shapes, width, pr_ms)
    morph.criteria.update(delta["criteria"])
    if delta["strong"] and pr_ms is None:
        morph.missing.append("onda delta sin PR medido: preexcitación no evaluada")
    if (delta["strong"] and pr_ms is not None and pr_ms < rules.pr_short_ms
            and width > rules.delta_qrs_min_ms):
        morph.pattern = "preexcitacion"
        morph.confidence = 0.75
        morph.applied.extend(delta["applied"])
        morph.applied.append(f"PR {pr_ms:.0f} ms < {rules.pr_short_ms:.0f} ms")
        return morph

    # 2) ramas: precordial derecha y lateral deben estar identificadas
    right = _channel_for(role_map, RIGHT_PRECORDIAL)
    left = _channel_for(role_map, LEFT_LATERAL)
    if right is None or left is None:
        if right is None:
            morph.missing.append("precordial derecha (V1/V2) no identificada")
        if left is None:
            morph.missing.append("lateral (I/V5/V6) no identificada")
        morph.pattern = "indeterminado"
        morph.confidence = 0.2
        return morph
    # tiempos y amplitudes dentro del QRS de la unión de canales, como en el
    # tratado. Medir la S final en el QRS propio de V5 la recorta (CRBBB: S > 40
    # ms en 9 % frente a 98 %, muestra de desarrollo); medir el pico de R desde
    # el inicio propio de V5 también empeora la sensibilidad del BRI (CLBBB,
    # desarrollo: 0,61 con la unión frente a 0,44 con el inicio propio)
    rf = aha_lead_features(y[:, right], fs, onset, offset, rules)
    lf = aha_lead_features(y[:, left], fs, onset, offset, rules)
    if not (rf.get("valid") and lf.get("valid")):
        morph.missing.append("V1 o lateral sin amplitud: criterios no evaluables")
        morph.pattern = "indeterminado"
        morph.confidence = 0.2
        return morph
    for tag, feat in (("v1", rf), ("lat", lf)):
        for key in ("r_peak_ms", "s_final_ms", "r_dur_ms", "net"):
            morph.criteria[f"{key}_{tag}"] = round(float(feat[key]), 3)
        morph.criteria[f"q_{tag}"] = float(feat["q_present"])

    rsr = _r_prime_terminal(rf)
    v1_ok = rsr or (rf["r_peak_ms"] > rules.v1_broad_r_peak_ms and rf["net"] > 0)
    s_final, r_dur = lf["s_final_ms"], lf["r_dur_ms"]
    lat_s_ok = (s_final > rules.lateral_s_ms
                or (rules.lateral_s_longer_than_r and s_final > 0 and s_final > r_dur))
    brd = v1_ok and lat_s_ok
    bri_ok = (not lf["q_present"]) and lf["r_peak_ms"] > rules.lateral_r_peak_ms
    complete = width >= rules.qrs_complete_ms
    incomplete = rules.qrs_incomplete_ms <= width < rules.qrs_complete_ms

    if brd and (complete or incomplete):
        morph.pattern = "brd" if complete else "brd_incompleto"
        morph.confidence = 0.8 if complete else 0.6
        morph.applied.append("rsR' en V1" if rsr else "R ancha en V1 con pico > 50 ms")
        morph.applied.append("S final lateral > 40 ms o más larga que la R")
    elif bri_ok and complete and lf["net"] > 0:
        morph.pattern = "bri"
        morph.confidence = 0.8
        morph.applied.append("R ancha dominante positiva en la lateral, sin q")
        morph.applied.append("pico de R lateral > 60 ms")
    elif bri_ok and incomplete:
        morph.pattern = "bri_incompleto"
        morph.confidence = 0.6
        morph.applied.append("pico de R lateral > 60 ms sin q")
    elif width > rules.tciv_min_ms:
        morph.pattern = "ivcd"
        morph.confidence = 0.5
        morph.applied.append("QRS > 110 ms sin criterios de rama")
    else:
        morph.pattern = "estrecho"
        morph.confidence = 0.6
    if morph.pattern in ("brd", "brd_incompleto", "bri", "bri_incompleto", "ivcd") \
            and incomplete:
        morph.applied.append("QRS incompleto (110-119 ms)")
    if morph.assumed and morph.pattern in ("brd", "brd_incompleto", "bri", "bri_incompleto"):
        morph.confidence = min(morph.confidence, 0.6)
        morph.missing.append("patrón de rama con posición de derivaciones inferida: "
                             "confirmar con las derivaciones declaradas")
    return morph


def _delta_evidence(feats: dict, shapes: dict, width: float,
                    pr_ms: float | None) -> dict:
    """Evidencia de onda delta: inicio lento, pendiente máxima tardía y PR corto.

    Se evalúa sobre el canal de mayor amplitud (el QRS más legible) y se exige
    que se cumplan al menos tres de los cuatro rasgos del inicio, con el ancho
    dentro del rango de una preexcitación. `strong` distingue la evidencia
    completa (cuatro rasgos, o tres con PR corto).
    """
    out = {"strong": False, "short_pr": False, "criteria": {},
           "applied": [], "missing": []}
    best = None
    for ch, feat in feats.items():
        shape = shapes.get(ch) or {}
        if not feat.get("valid") or not shape:
            continue
        if best is None or feat["amp_mv"] > best[1]["amp_mv"]:
            best = (ch, feat, shape)
    if best is None:
        return out
    ch, _feat, shape = best
    head = float(shape.get("head_ratio", 1.0))
    upstroke = float(shape.get("upstroke_ms", 0.0))
    slope_frac = float(shape.get("slope_peak_frac", 0.0))
    initial = float(shape.get("initial_frac", 1.0))
    in_window = DELTA_MIN_WIDTH <= width <= DELTA_MAX_WIDTH
    score = 0
    if head <= DELTA_HEAD_RATIO:
        score += 1
        out["applied"].append("inicio del QRS empastado (pendiente inicial baja)")
    if upstroke >= DELTA_UPSTROKE_MS:
        score += 1
        out["applied"].append("ascenso lento hasta la mitad del pico")
    if slope_frac >= DELTA_SLOPE_FRAC:
        score += 1
        out["applied"].append("pendiente máxima tardía en el complejo")
    if initial <= DELTA_INITIAL_FRAC:
        score += 1
    out["short_pr"] = bool(pr_ms is not None and pr_ms <= DELTA_SHORT_PR)
    # la onda delta exige las dos condiciones más específicas: inicio lento
    # (pendiente del primer tramo baja) y máximo de pendiente tardío
    strong = bool(in_window and score >= 3 and head <= DELTA_HEAD_RATIO
                  and slope_frac >= DELTA_SLOPE_FRAC)
    if strong:
        out["applied"].append("inicio empastado (onda delta)")
    out["strong"] = strong
    out["criteria"] = {"delta_ch": float(ch), "delta_score": float(score),
                       "delta_head_ratio": round(head, 3),
                       "delta_upstroke_ms": round(upstroke, 1),
                       "delta_slope_frac": round(slope_frac, 3),
                       "delta_initial_frac": round(initial, 3),
                       "delta_pr_ms": (round(float(pr_ms), 1) if pr_ms is not None
                                       else -1.0)}
    if score >= 3 and not in_window:
        out["missing"].append("inicio empastado fuera del ancho típico de una onda delta")
    return out


# ---------------------------------------------------------------------------
# Minutos, episodios y aberrancia
# ---------------------------------------------------------------------------
@dataclass
class MinuteQrs:
    """QRS promediado de un minuto."""

    minute: int
    n_beats: int
    rr_s: float
    hr_bpm: float
    width_ms: float
    pattern: str
    confidence: float
    detail: dict = field(default_factory=dict)   # Morphology.to_dict() del minuto

    def to_dict(self) -> dict:
        return {
            "minute": self.minute,
            "n_beats": self.n_beats,
            "hr_bpm": round(self.hr_bpm, 1),
            "width_ms": round(self.width_ms, 1),
            "pattern": self.pattern,
            "pattern_name": PATTERN_NAMES.get(self.pattern, self.pattern),
            "confidence": round(self.confidence, 3),
        }


def minute_qrs(x: np.ndarray, fs: float, peaks, *, minute_s: float = 60.0,
               min_beats: int = 5, roles: dict[int, str] | None = None,
               baseline: np.ndarray | None = None,
               rules: AhaRules | None = None,
               exclude_spans=None) -> list[MinuteQrs]:
    """Latido promediado por minuto y su morfología.

    El promedio por mediana dentro del minuto evita que un extrasístole o un
    artefacto decida el patrón del minuto. `exclude_spans` (pares de muestras
    `(inicio, fin)`, p. ej. tramos con ≥2 de 3 canales malos) descarta los
    latidos que caen dentro: no se informa nada sobre ellos. `rules` activa los
    criterios AHA (ver `classify_beat`).
    """
    sig = np.asarray(x, dtype=np.float64)
    if sig.ndim == 1:
        sig = sig[:, None]
    peaks = np.asarray(peaks, dtype=np.int64)
    out: list[MinuteQrs] = []
    if exclude_spans:
        keep = np.ones(peaks.size, dtype=bool)
        for a, b in exclude_spans:
            keep &= ~((peaks >= int(a)) & (peaks < int(b)))
        peaks = peaks[keep]
    if peaks.size == 0:
        return out
    minutes = (peaks / (minute_s * fs)).astype(int)
    for minute in np.unique(minutes):
        sel = peaks[minutes == minute]
        if sel.size < min_beats:
            continue
        beat, center = median_beat(sig, fs, sel)
        morph = classify_beat(beat, fs, center, roles=roles, baseline=baseline,
                              rules=rules)
        if not morph.measurable:
            continue
        rr = float(np.median(np.diff(sel)) / fs) if sel.size > 1 else 0.0
        out.append(MinuteQrs(minute=int(minute), n_beats=int(sel.size), rr_s=rr,
                             hr_bpm=(60.0 / rr if rr > 0 else 0.0),
                             width_ms=morph.width_ms, pattern=morph.pattern,
                             confidence=morph.confidence,
                             detail=morph.to_dict()))
    return out


def rate_dependence(minutes: list[MinuteQrs], min_delta_bpm: float = 5.0) -> dict:
    """Compara la frecuencia de los minutos anchos con la de los estrechos.

    Además de la diferencia de medianas se busca la **frecuencia crítica**: el
    corte de FC que mejor separa los minutos anchos de los estrechos (mayor
    J de Youden). Un bloqueo de rama dependiente de la frecuencia aparece por
    encima de esa FC y desaparece por debajo; si los minutos anchos están
    repartidos a ambos lados del corte, la dependencia no se sostiene y el
    patrón se informa como intermitente, no como dependiente de la FC.
    """
    wide = [m for m in minutes if m.width_ms >= QRS_WIDE_MS]
    narrow = [m for m in minutes if m.width_ms < QRS_INCOMPLETE_MS]
    if not wide or not narrow:
        return {"wide_hr": None, "narrow_hr": None, "delta_bpm": 0.0, "dependent": False}
    hw = float(np.median([m.hr_bpm for m in wide]))
    hn = float(np.median([m.hr_bpm for m in narrow]))
    top = float(np.percentile([m.hr_bpm for m in narrow], 75))
    high_wide = float(np.mean([m.hr_bpm >= top for m in wide]))
    delta = hw - hn
    # fase 3: el ancho aparece por encima de un corte de FC; fase 4: por debajo
    crit_hi = _critical_rate(wide, narrow, above=True)
    crit_lo = _critical_rate(wide, narrow, above=False)
    crit = crit_hi
    if crit_lo is not None and (crit_hi is None or crit_lo["youden"] > crit_hi["youden"]):
        crit = crit_lo
    dependent = bool(delta >= min_delta_bpm and high_wide >= 0.6)
    phase = "fase3" if dependent else None
    if crit is not None and crit["youden"] >= 0.5 and crit["share_wide_side"] >= 0.8:
        dependent = True
        phase = "fase3" if crit["above"] else "fase4"
    return {
        "wide_hr": round(hw, 1),
        "narrow_hr": round(hn, 1),
        "delta_bpm": round(delta, 1),
        "high_rate_share": round(high_wide, 3),
        "critical_bpm": (None if crit is None else crit["critical_bpm"]),
        "critical_youden": (None if crit is None else crit["youden"]),
        "critical_share_wide": (None if crit is None else crit["share_wide_side"]),
        "phase": phase if dependent else None,
        "dependent": dependent,
    }


def _critical_rate(wide: list[MinuteQrs], narrow: list[MinuteQrs], *,
                   above: bool = True) -> dict | None:
    """Corte de FC que mejor separa minutos anchos de estrechos (J de Youden).

    `above=True` busca un ancho que aparece por encima del corte (fase 3);
    `above=False`, uno que aparece por debajo (fase 4).
    """
    hw = np.asarray([m.hr_bpm for m in wide], dtype=float)
    hn = np.asarray([m.hr_bpm for m in narrow], dtype=float)
    if hw.size < 2 or hn.size < 2:
        return None
    best = None
    for cut in np.unique(np.round(np.concatenate([hw, hn]), 1)):
        if above:
            se = float(np.mean(hw >= cut))
            sp = float(np.mean(hn < cut))
        else:
            se = float(np.mean(hw < cut))
            sp = float(np.mean(hn >= cut))
        j = se + sp - 1.0
        if best is None or j > best["youden"] + 1e-9:
            best = {"critical_bpm": round(float(cut), 1), "youden": round(j, 3),
                    "share_wide_side": round(se, 3), "share_narrow_other": round(sp, 3),
                    "above": above}
    return best


def summarise_episode(minutes: list[MinuteQrs], *, persistent_frac: float = 0.8,
                      rare_frac: float = 0.02) -> dict:
    """Decide si el QRS ancho es persistente, intermitente o dependiente de FC.

    * `br_persistente`: ancho en >= 80 % de los minutos medidos.
    * `br_frecuencia_dependiente`: ancho en los minutos de frecuencia alta.
    * `br_intermitente`: ancho en una minoría de minutos sin relación con la FC.
    * `aberrancia_probable`: ancho en < 2 % de los minutos con patrón distinto
      al dominante; se confirma por latido con `is_aberrant`.
    """
    res: dict = {"n_minutes": len(minutes), "pattern": "estrecho", "kind": "sin_qrs_ancho",
                 "wide_fraction": 0.0, "consistency": 0.0}
    if not minutes:
        return res
    wide = [m for m in minutes if m.width_ms >= QRS_WIDE_MS]
    res["wide_fraction"] = round(len(wide) / len(minutes), 3)
    if not wide:
        res["kind"] = "sin_qrs_ancho"
        return res
    patterns = [m.pattern for m in wide]
    dominant = max(set(patterns), key=patterns.count)
    res["pattern"] = dominant
    res["pattern_name"] = PATTERN_NAMES.get(dominant, dominant)
    res["consistency"] = round(patterns.count(dominant) / len(patterns), 3)
    res["n_wide_minutes"] = len(wide)
    rate = rate_dependence(minutes)
    res.update(rate)
    res["rate_dependent"] = bool(rate["dependent"])
    # ¿el ancho se interrumpe al bajar la FC? (BR intermitente dependiente de FC)
    intermitente_fc = False
    if rate.get("critical_bpm") is not None and res["wide_fraction"] < persistent_frac:
        below = [m for m in minutes
                 if m.hr_bpm < float(rate["critical_bpm"])
                 and m.width_ms >= QRS_WIDE_MS]
        narrow_below = [m for m in minutes
                        if m.hr_bpm < float(rate["critical_bpm"])
                        and m.width_ms < QRS_INCOMPLETE_MS]
        if narrow_below and len(below) <= 0.1 * len(narrow_below):
            intermitente_fc = True
    res["intermittent_rate_dependent"] = bool(intermitente_fc)
    if res["wide_fraction"] >= persistent_frac:
        res["kind"] = "br_persistente"
    elif res["wide_fraction"] <= rare_frac:
        res["kind"] = "aberrancia_probable"
    elif (rate["dependent"] or intermitente_fc) and res["consistency"] >= 0.6:
        res["kind"] = "br_frecuencia_dependiente"
    elif res["consistency"] >= 0.5:
        res["kind"] = "br_intermitente"
    else:
        res["kind"] = "morfologia_variable"
    if res["kind"] == "br_frecuencia_dependiente":
        # fase 3: el ancho aparece por encima de la FC de corte; fase 4: por debajo
        res["phase"] = rate.get("phase") or "fase3"
        res["appearance_bpm"] = rate.get("critical_bpm")
    # preexcitación que alterna con minutos de QRS normal: intermitente
    pre = [m for m in minutes if m.pattern == "preexcitacion"]
    normal = [m for m in minutes if m.width_ms < QRS_INCOMPLETE_MS]
    res["preexcitation_minutes"] = len(pre)
    res["preexcitation_intermittent"] = bool(pre and normal)
    return res


def is_aberrant(rr_prev: float, rr_prev2: float, rr_median: float,
                same_pattern: bool) -> dict:
    """Fenómeno de Ashman: ciclo largo-corto con morfología de rama típica.

    Un latido ancho aislado tras un ciclo largo-corto, con el mismo patrón
    que el dominante y en un registro de base estrecha, se informa como
    aberrancia probable; si el patrón es distinto del dominante, como
    extrasístole ventricular morfológicamente atípica.

    El criterio exige tres cosas a la vez (`ashman_score` = cuántas cumple):
    ciclo previo largo (>= 1,08 × RR mediano), acoplamiento corto
    (rr_prev / rr_prev2 <= 0,78) y prematuridad absoluta frente a la mediana
    (rr_prev <= 0,90 × RR mediano). El segundo requisito solo no basta: un
    latido prematuro tras un ciclo normal no es Ashman.
    """
    long_cycle = rr_prev2 >= 1.08 * rr_median
    short_coupling = rr_prev2 > 0 and rr_prev <= 0.78 * rr_prev2
    premature = rr_prev <= 0.90 * rr_median
    score = int(long_cycle) + int(short_coupling) + int(premature)
    long_short = bool(long_cycle and short_coupling and premature)
    return {
        "long_short": long_short,
        "aberrant_probable": bool(long_short and same_pattern),
        "atypical_ventricular": bool(long_short and not same_pattern),
        "ashman_score": score,
        "coupling_ratio": round(float(rr_prev / rr_prev2), 3) if rr_prev2 > 0 else None,
        "long_cycle_s": round(float(rr_prev2), 3),
        "rr_prev": round(float(rr_prev), 3),
        "rr_prev2": round(float(rr_prev2), 3),
        "rr_median": round(float(rr_median), 3),
    }
