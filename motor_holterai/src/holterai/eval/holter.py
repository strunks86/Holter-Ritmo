"""Evaluación de un ``report.json`` de HolterIA contra una referencia.

La referencia puede ser HScribe o cualquier lista de latidos con tiempo y
clase AAMI. Se informa:

* bxb (ventana 150 ms, matriz AAMI) global y por estrato de calidad cuando el
  reporte trae ``features.noise`` / ``features.artifact`` por latido;
* rxr de V y S sobre la referencia, sobre lo detectado y sobre la secuencia
  alineada;
* epicmp de FA si hay episodios de referencia (``--af-ref``) y el reporte
  trae tramos de ritmo o eventos de FA.

No se copia contenido de exámenes: la función devuelve solo números y listas
de tiempos/clases que el llamador decide dónde guardar.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from holterai.eval.bxb import (
    WINDOW_S,
    QualityStrata,
    match_beats,
    score_bxb,
    score_bxb_by_stratum,
)
from holterai.eval.epicmp import Episode, compare_episodes, parse_episodes
from holterai.eval.rxr import DEFAULT_TARGETS, score_rxr

AF_LABELS = {"AF", "AFIB", "FA", "FIBRILACION", "FIBRILACIÓN"}


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _pair_from_item(item: Any) -> tuple[float, Any]:
    if isinstance(item, Mapping):
        t = item.get("t", item.get("time_s", item.get("time")))
        cls = item.get("cls", item.get("class", item.get("label")))
        if t is None:
            raise ValueError("latido de referencia sin tiempo")
        return float(t), cls
    seq = list(item) if isinstance(item, (list, tuple)) else [item]
    if len(seq) < 2:
        raise ValueError("latido de referencia sin clase")
    return float(seq[0]), seq[1]


def load_reference_beats(path: str | Path) -> tuple[list[float], list[str]]:
    """Lee una referencia en JSON (``times``/``classes`` o pares) o CSV ``t,cls``."""
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig").strip()
    if not text:
        return [], []
    if text[0] in "[{":
        data = json.loads(text)
        if isinstance(data, Mapping):
            if "times" in data and "classes" in data:
                times = [float(t) for t in data["times"]]
                classes = [str(c) for c in data["classes"]]
                if len(times) != len(classes):
                    raise ValueError("times y classes no coinciden en longitud")
                return times, classes
            items: Iterable[Any] = data.get("beats", [])
        else:
            items = data
        pairs = [_pair_from_item(item) for item in items]
    else:
        pairs = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x for x in line.replace(";", ",").split(",") if x.strip()]
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0])
            except ValueError:  # cabecera del CSV u otra línea no numérica
                continue
            pairs.append((t, parts[1].strip()))
    return [t for t, _ in pairs], [str(c) for _, c in pairs]


def report_beats(
    report: Mapping[str, Any],
) -> tuple[list[float], list[str], list[float], list[bool]]:
    """Tiempos, clases, ruido y bandera de artefacto de los latidos del reporte."""
    times: list[float] = []
    classes: list[str] = []
    noise: list[float] = []
    artifact: list[bool] = []
    for beat in report.get("beats", []):
        times.append(float(beat.get("t", 0.0)))
        classes.append(str(beat.get("cls", "")))
        feats = beat.get("features") or {}
        noise.append(float(feats.get("noise", 0.0)))
        artifact.append(bool(feats.get("artifact", False)))
    return times, classes, noise, artifact


def has_quality(report: Mapping[str, Any]) -> bool:
    """True si el reporte trae ruido por latido (``features.noise``)."""
    for beat in report.get("beats", []) or []:
        feats = beat.get("features") or {}
        if "noise" in feats:
            return True
    return False


def report_af_episodes(report: Mapping[str, Any]) -> list[Episode]:
    """Episodios de FA declarados en ``rhythm`` o en los eventos del reporte."""
    out: list[Episode] = []
    for seg in report.get("rhythm", []) or []:
        label = str(seg.get("label", "")).strip().upper()
        if label in AF_LABELS:
            out.append(Episode(float(seg["start_s"]), float(seg["end_s"]), "AF"))
    for event in report.get("events", []) or []:
        kind = str(event.get("kind", "")).strip().lower()
        if kind in {"af", "afib", "fa"}:
            out.append(Episode(float(event["start_s"]), float(event["end_s"]), "AF"))
    return sorted(out, key=lambda e: (e.start_s, e.end_s))


def load_af_reference(path: str | Path) -> list[Episode]:
    """Lee episodios de FA de referencia (pares o diccionarios con tiempos)."""
    data = load_json(path)
    if isinstance(data, Mapping):
        data = data.get("episodes", data.get("af", []))
    return parse_episodes(data)


def evaluate_report(
    report: Mapping[str, Any],
    reference: tuple[Sequence[float], Sequence[Any]] | None = None,
    af_reference: Sequence[Episode | Sequence[float] | Mapping[str, Any]] | None = None,
    window_s: float = WINDOW_S,
    strata: QualityStrata | None = None,
    targets: Sequence[str] = DEFAULT_TARGETS,
) -> dict[str, Any]:
    """bxb + rxr + epicmp de un reporte contra la referencia disponible."""
    det_times, det_classes, noise, artifact = report_beats(report)
    out: dict[str, Any] = {
        "n_det": len(det_times),
        "window_s": float(window_s),
        "has_reference": reference is not None,
    }
    if reference is None:
        ref_times: Sequence[float] = []
        ref_classes: Sequence[Any] = []
        out["rxr"] = score_rxr([], det_classes, targets=targets)
    else:
        ref_times, ref_classes = reference
        result = score_bxb(ref_times, ref_classes, det_times, det_classes,
                           window_s=window_s, label="report")
        out["bxb"] = result.as_dict()
        pairs = match_beats(ref_times, det_times, window_s).pairs
        out["rxr"] = score_rxr(ref_classes, det_classes, targets=targets, pairs=pairs)
        if has_quality(report):
            out["strata"] = {
                name: res.as_dict()
                for name, res in score_bxb_by_stratum(
                    ref_times, ref_classes, det_times, det_classes,
                    det_noise=noise, det_artifact=artifact, strata=strata,
                    window_s=window_s,
                ).items()
            }
    out["n_ref"] = len(ref_times)
    if af_reference is not None:
        out["af"] = compare_episodes(af_reference, report_af_episodes(report))
    return out
