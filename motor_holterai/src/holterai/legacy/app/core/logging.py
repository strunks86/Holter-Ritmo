"""Registro de eventos de la aplicación (consola + archivo)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from holterai.legacy.app.core.config import LOG_DIR

_CONFIGURED = False


def setup(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger("holteria")
    root.setLevel(level)
    try:
        fh = logging.FileHandler(LOG_DIR / "holteria.log", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception:
        pass
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup()
    return logging.getLogger(name if name.startswith("holteria") else f"holteria.{name}")

