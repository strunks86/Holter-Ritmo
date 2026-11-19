"""Modelos de datos centrales de HolterIA."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class Patient:
    patient_id: str = ""
    name: str = ""
    age: int | None = None
    sex: str = ""  # "M" | "F" | ""
    dob: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "name": self.name,
            "age": self.age,
            "sex": self.sex,
            "dob": self.dob,
            "notes": self.notes,
        }


@dataclass
class Recording:
    """Registro Holter cargado en memoria.

    samples: matriz (n_muestras, n_derivaciones), float32, en milivoltios.
    """

    samples: np.ndarray
    fs: float
    leads: list[str] = field(default_factory=lambda: ["I"])
    start_time: datetime | None = None
    patient: Patient = field(default_factory=Patient)
    source: str = ""
    fmt: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- utilidades -------------------------------------------------
    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def n_leads(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_samples / float(self.fs) if self.fs else 0.0

    def lead(self, idx: int) -> np.ndarray:
        return self.samples[:, idx]

    def segment(self, start_s: float, end_s: float, max_points: int = 4_000_000):
        """Devuelve (t0_muestras, muestras) del intervalo pedido, recortado."""
        a = max(0, int(start_s * self.fs))
        b = min(self.n_samples, int(end_s * self.fs))
        if b <= a:
            return a, self.samples[0:0, :]
        if b - a > max_points:  # límite de seguridad para no saturar la UI
            b = a + max_points
        return a, np.asarray(self.samples[a:b, :], dtype=np.float32)

    def meta_dict(self) -> dict:
        return {
            "source": self.source,
            "format": self.fmt,
            "fs": self.fs,
            "n_samples": self.n_samples,
            "n_leads": self.n_leads,
            "leads": list(self.leads),
            "duration_s": round(self.duration_s, 3),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "patient": self.patient.to_dict(),
            "meta": _jsonable(self.meta),
        }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class JsonMixin:
    def to_dict(self) -> dict:  # pragma: no cover - interfaz
        raise NotImplementedError


@dataclass
class Beat(JsonMixin):
    """Un latido detectado y clasificado."""

    idx: int                      # índice de muestra del pico R
    t: float                      # tiempo en segundos
    cls: str = "N"                # N, S, V, F, Q
    conf: float = 0.0
    rr_prev: float = 0.0          # segundos
    rr_next: float = 0.0
    edited: bool = False
    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "idx": int(self.idx),
            "t": round(float(self.t), 4),
            "cls": self.cls,
            "conf": round(float(self.conf), 3),
            "rr_prev": round(float(self.rr_prev), 4),
            "rr_next": round(float(self.rr_next), 4),
            "edited": bool(self.edited),
        }
        if self.features:
            d["features"] = {
                k: round(float(v), 4)
                for k, v in self.features.items()
                if v is not None
            }
        return d


@dataclass
class Event(JsonMixin):
    """Evento clínico detectado (pausa, FA, taquicardia, etc.)."""

    kind: str                     # pause, afib, brady, tachy, svt, vt, bigeminy...
    start_s: float
    end_s: float
    severity: str = "info"        # info | warn | critical
    detail: str = ""
    beats: list[int] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "start_s": round(float(self.start_s), 3),
            "end_s": round(float(self.end_s), 3),
            "duration_s": round(self.duration_s, 3),
            "severity": self.severity,
            "detail": self.detail,
            "beats": self.beats[:200],
        }


@dataclass
class RhythmSegment(JsonMixin):
    """Tramo de ritmo (para la tira de ritmo del informe)."""

    start_s: float
    end_s: float
    label: str           # SR, AF, PAC, PVC, ...
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "start_s": round(float(self.start_s), 2),
            "end_s": round(float(self.end_s), 2),
            "label": self.label,
            "confidence": round(float(self.confidence), 3),
        }


@dataclass
class Analysis:
    """Resultado completo del análisis de un registro."""

    recording: Recording
    beats: list[Beat] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    rhythm: list[RhythmSegment] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    model_info: dict = field(default_factory=dict)
    notes: str = ""
    diary: str = ""
    deleted_idxs: list[int] = field(default_factory=list)
    inserted_beats: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0

    # ---- conteos ----------------------------------------------------
    def class_counts(self) -> dict[str, int]:
        out = {"N": 0, "S": 0, "V": 0, "F": 0, "Q": 0}
        for b in self.beats:
            out[b.cls] = out.get(b.cls, 0) + 1
        return out

    def rr_series(self) -> np.ndarray:
        return np.array([b.rr_prev for b in self.beats[1:]], dtype=float)

    def to_dict(self, include_beats: bool = True) -> dict:
        d = {
            "meta": self.recording.meta_dict(),
            "summary": _jsonable(self.summary),
            "quality": _jsonable(self.quality),
            "model": _jsonable(self.model_info),
            "events": [e.to_dict() for e in self.events],
            "rhythm": [r.to_dict() for r in self.rhythm],
            "counts": self.class_counts(),
            "n_beats": len(self.beats),
            "elapsed_s": round(self.elapsed_s, 2),
            "notes": self.notes,
            "diary": self.diary,
        }
        if include_beats:
            d["beats"] = [b.to_dict() for b in self.beats]
        return d


@dataclass
class BeatFeatures:
    """Características por latido usadas por el modelo y las reglas."""

    window: np.ndarray            # señal normalizada (256 puntos)
    rr_prev: float
    rr_next: float
    rr_ratio: float
    qrs_width: float
    amplitude: float
    polarity: float
    p_score: float
    t_score: float
    noise: float

    def vector(self) -> np.ndarray:
        return np.array(
            [
                self.rr_prev,
                self.rr_next,
                self.rr_ratio,
                self.qrs_width,
                self.amplitude,
                self.polarity,
                self.p_score,
                self.t_score,
                self.noise,
            ],
            dtype=np.float32,
        )

    FEATURE_NAMES = [
        "rr_prev",
        "rr_next",
        "rr_ratio",
        "qrs_width",
        "amplitude",
        "polarity",
        "p_score",
        "t_score",
        "noise",
    ]

