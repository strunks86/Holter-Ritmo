"""Autodiagnóstico de instalación del prototipo (sin agente)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from holterai.config import DEFAULT_CRITERIA, DEFAULT_HARDWARE, load_criteria, load_hardware


def doctor(
    *,
    dll_dir: str | Path | None = None,
    exporter: str | Path = "ingest/export180.exe",
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    models_dir = root / "models" / "pretrained"
    if not models_dir.is_dir():
        models_dir = Path("models/pretrained")
    exporter_path = Path(exporter)
    if not exporter_path.is_file():
        exporter_path = root / "ingest" / "export180.exe"
    checks: dict[str, Any] = {
        "python": sys.version.split()[0],
        "cwd": str(Path.cwd()),
        "hardware_config": DEFAULT_HARDWARE.is_file(),
        "criteria_config": DEFAULT_CRITERIA.is_file(),
        "exporter": exporter_path.is_file(),
        "exporter_path": str(exporter_path),
        "models": sorted(p.name for p in models_dir.glob("*.npz")),
    }
    try:
        hw = load_hardware()
        checks["hardware"] = hw.model_dump()
    except Exception as exc:
        checks["hardware_error"] = f"{type(exc).__name__}: {exc}"
    try:
        crit = load_criteria()
        checks["criteria"] = crit.model_dump()
    except Exception as exc:
        checks["criteria_error"] = f"{type(exc).__name__}: {exc}"

    deps: dict[str, bool] = {}
    for mod in ("numpy", "docx", "matplotlib", "PIL"):
        try:
            __import__(mod)
            deps[mod] = True
        except Exception:
            deps[mod] = False
    checks["deps"] = deps

    if dll_dir is not None:
        d = Path(dll_dir)
        checks["dll_dir"] = str(d)
        checks["HW32ExamAPI.dll"] = (d / "HW32ExamAPI.dll").is_file()
    return checks


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dll-dir", default=None)
    args = ap.parse_args()
    print(json.dumps(doctor(dll_dir=args.dll_dir), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
