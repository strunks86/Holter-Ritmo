"""Inventario de modelos locales; no ejecuta inferencia."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ModelInfo:
    name: str
    path: str
    kind: str
    classes: list[str]
    features: list[str]
    installed: bool


def _npz_info(path: Path, name: str, kind: str) -> ModelInfo:
    if not path.is_file():
        return ModelInfo(
            name=name, path=str(path), kind=kind, classes=[], features=[], installed=False
        )
    try:
        z: Any = np.load(path, allow_pickle=True)
        classes = [str(x) for x in z["__classes__"]] if "__classes__" in z else []
        features = [str(x) for x in z["names"]] if "names" in z else []
    except Exception:
        classes, features = [], []
    return ModelInfo(
        name=name,
        path=str(path),
        kind=kind,
        classes=classes,
        features=features,
        installed=True,
    )


def list_models(root: str | Path | None = None) -> list[ModelInfo]:
    base = Path(root) if root else Path("models/pretrained")
    out = [
        _npz_info(base / "beat_cnn.npz", "beat_cnn", "clasificación de latido"),
        _npz_info(base / "pac_model.npz", "pac_model", "probabilidad PAC"),
        _npz_info(base / "rhythm_af.npz", "rhythm_af", "ritmo/FA"),
    ]
    founder = base.parent / "ecgfounder" / "ecg_founder_all71.onnx"
    out.append(
        ModelInfo(
            name="ecgfounder_all71",
            path=str(founder),
            kind="modelo fundacional 12 derivaciones",
            classes=[],
            features=[],
            installed=founder.is_file(),
        )
    )
    return out


def as_json(models: list[ModelInfo]) -> str:
    return json.dumps([asdict(m) for m in models], indent=2, ensure_ascii=False)
