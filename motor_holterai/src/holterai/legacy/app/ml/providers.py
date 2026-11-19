"""Selección del proveedor de ejecución de onnxruntime (CPU / GPU).

onnxruntime trae builds distintos para cada backend y solo se puede importar
uno por proceso, así que cada candidato vive en su propia carpeta de vendor/
y se elige al arrancar:

    vendor/ort_gpu   -> onnxruntime-gpu   (CUDAExecutionProvider, ROCm en Linux)
    vendor/ort_dml   -> onnxruntime-directml (DmlExecutionProvider, DX12)
    vendor/          -> onnxruntime (CPU)

La disponibilidad se comprueba en un subproceso para no contaminar el proceso
principal con importaciones incompatibles.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from holterai.legacy.app.core.logging import get_logger

log = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[5]
VENDOR = ROOT / "vendor"

LABELS = {
    "CUDAExecutionProvider": "CUDA (NVIDIA)",
    "ROCMExecutionProvider": "ROCm (AMD, solo Linux)",
    "DmlExecutionProvider": "DirectML (GPU universal, DX12)",
    "CPUExecutionProvider": "CPU",
}

PREF_ORDER = {
    # CUDA en NVIDIA, ROCm en AMD (Linux), CPU como red de seguridad.
    # DirectML solo si el usuario lo elige explícitamente (Windows).
    "auto": ["cuda", "rocm", "cpu"],
    "cuda": ["cuda", "cpu"],
    "rocm": ["rocm", "cpu"],
    "dml": ["dml", "cpu"],
    "cpu": ["cpu"],
}

_DIRS = {
    "cuda": VENDOR / "ort_gpu",
    "rocm": VENDOR / "ort_rocm",
    "dml": VENDOR / "ort_dml",
    "cpu": VENDOR,
}

_probe_cache: dict[str, list[str]] = {}
_active: dict | None = None


def probe(kind: str) -> list[str]:
    """Proveedores que ofrece ese build (lista vacía si no está instalado)."""
    if kind in _probe_cache:
        return _probe_cache[kind]
    d = _DIRS.get(kind)
    if d is None or not (d / "onnxruntime").exists():
        _probe_cache[kind] = []
        return []
    code = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(d)!r})\n"
        "import onnxruntime as o\n"
        "print(json.dumps(o.get_available_providers()))\n"
    )
    try:
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=120)
        provs = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else []
    except Exception as exc:
        log.debug("probe %s falló: %s", kind, exc)
        provs = []
    _probe_cache[kind] = list(provs)
    return _probe_cache[kind]


def availability() -> dict:
    """Estado de cada backend: qué ofrece y si está instalado."""
    out = {}
    for kind in ("cuda", "rocm", "dml", "cpu"):
        provs = probe(kind)
        out[kind] = {
            "installed": bool(provs),
            "providers": provs,
            "label": LABELS.get(provs[0], kind) if provs else kind.upper(),
        }
    return out


def _label(kind: str, provs: list[str]) -> str:
    if "CUDAExecutionProvider" in provs:
        return LABELS["CUDAExecutionProvider"]
    if "DmlExecutionProvider" in provs:
        return LABELS["DmlExecutionProvider"]
    if "ROCMExecutionProvider" in provs:
        return LABELS["ROCMExecutionProvider"]
    if "CPUExecutionProvider" in provs:
        return "CPU"
    return kind.upper()


def describe(pref: str = "auto") -> dict:
    av = availability()
    kind, provs = _pick(pref, av)
    return {
        "preference": pref,
        "selected": kind,
        "providers": provs,
        "label": _label(kind, provs),
        "available": av,
    }


def _pick(pref: str, av: dict) -> tuple[str, list[str]]:
    for kind in PREF_ORDER.get(pref, PREF_ORDER["auto"]):
        if av.get(kind, {}).get("installed"):
            return kind, av[kind]["providers"]
    return "cpu", ["CPUExecutionProvider"]


def activate(pref: str = "auto") -> dict:
    """Deja el build elegido primero en sys.path (llamar antes de importar ort)."""
    global _active
    if _active is not None and _active.get("preference") == pref:
        return _active
    av = availability()
    kind, provs = _pick(pref, av)
    path = _DIRS[kind]
    s = str(path)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)
    _active = {
        "preference": pref,
        "selected": kind,
        "providers": provs,
        "label": _label(kind, provs),
        "path": s,
    }
    log.info("aceleración: %s (%s)", _active["label"], ",".join(provs))
    return _active


def active() -> dict | None:
    return _active

