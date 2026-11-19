"""Configuración y rutas de HolterIA.

La aplicación está pensada para ejecutarse desde una carpeta portable:
todo lo que necesita (runtime, modelo, estáticos) vive dentro de APP_ROOT.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

APP_NAME = "HolterIA"
VERSION = "1.0.0"
DISCLAIMER = (
    "Herramienta de apoyo a la investigación y docencia. "
    "NO es un dispositivo médico y NO debe usarse para diagnóstico clínico."
)


def app_root() -> Path:
    """Carpeta raíz de la aplicación (soporta ejecución empaquetada)."""
    if os.environ.get("HOLTERIA_ROOT"):  # HolterLocal de escritorio fija la carpeta del motor
        return Path(os.environ["HOLTERIA_ROOT"])
    if getattr(sys, "frozen", False):  # PyInstaller, si algún día se usa
        return Path(sys.executable).resolve().parent
    # holteria/app/core/config.py -> holteria/
    return Path(__file__).resolve().parents[5]


APP_ROOT = app_root()
MODELS_DIR = APP_ROOT / "models"
STATIC_DIR = APP_ROOT / "src" / "holterai" / "legacy" / "app" / "static"
DATA_DIR = Path(os.environ.get("HOLTERIA_DATA", str(APP_ROOT / "data")))
REPORTS_DIR = DATA_DIR / "reports"
CACHE_DIR = DATA_DIR / "cache"
LOG_DIR = DATA_DIR / "logs"

for _d in (DATA_DIR, REPORTS_DIR, CACHE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SETTINGS_PATH = DATA_DIR / "settings.json"

DEFAULTS = {
    "fs_display_default": 250.0,
    "pause_threshold_s": 2.0,
    "brady_threshold_bpm": 50.0,
    "tachy_threshold_bpm": 100.0,
    "min_af_burden_s": 30.0,
    "report_language": "es",
    "open_browser": True,
    "host": "127.0.0.1",
    "theme": "dark",
}


def load_settings() -> dict:
    s = dict(DEFAULTS)
    try:
        if SETTINGS_PATH.exists():
            s.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
    except Exception:
        pass
    return s


def save_settings(settings: dict) -> None:
    try:
        merged = dict(DEFAULTS)
        merged.update(settings or {})
        SETTINGS_PATH.write_text(
            json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


SETTINGS = load_settings()




