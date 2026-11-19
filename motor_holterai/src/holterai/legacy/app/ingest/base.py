"""Importación de registros Holter: detección de formato y carga unificada."""
from __future__ import annotations

import re
from pathlib import Path

from holterai.legacy.app.core.models import Patient, Recording

from . import csv_import, edf, wfdb

SUPPORTED = {".hea", ".dat", ".edf", ".csv", ".txt", ".tsv", ".asc", ".ecg"}


class ImportError_(Exception):
    """Error de importación con mensaje para el usuario."""


def _sniff_magic(p: Path) -> str:
    try:
        head = p.open("rb").read(16)
    except Exception:
        return ""
    if head[:4] == b"\x00\x00\x00\x00" or head[:4] == b"0   ":
        return "edf"
    if head[:8].isascii() and (b"EDF" in p.open("rb").read(300)):
        return "edf"
    low = head.lower()
    if low.startswith(b"version") or low.startswith(b"\xef\xbb\xbfversion"):
        return "csv"
    try:
        txt = p.open("r", encoding="utf-8", errors="ignore").read(400)
        if re.match(r"^\s*[\w\-\.]+/?\d?\s+\d+\s+[\d\.]+\s+\d+", txt):
            return "wfdb"
        if re.match(r"^\s*[\w\-\.]+\s+[-\d.]+\s*[,;\t]", txt):
            return "csv"
    except Exception:
        pass
    return ""


def find_record_files(folder: Path) -> list[Path]:
    """Busca archivos de registro dentro de una carpeta."""
    out: list[Path] = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            out.append(p)
    return out


def load(path: str | Path, fs: float | None = None, **kwargs) -> Recording:
    """Carga un registro Holter y devuelve un Recording.

    Acepta ruta a archivo o a carpeta (en cuyo caso busca el registro).
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise ImportError_(f"No existe la ruta: {p}")
    if p.is_dir():
        files = find_record_files(p)
        if not files:
            raise ImportError_(f"No se encontraron registros en {p}")
        # preferimos cabeceras .hea y luego EDF
        for pref in (".hea", ".edf"):
            for f in files:
                if f.suffix.lower() == pref:
                    p = f
                    break
            else:
                continue
            break
        else:
            p = files[0]

    ext = p.suffix.lower()
    kind = ""
    if ext == ".hea":
        kind = "wfdb"
    elif ext in (".edf",):
        kind = "edf"
    elif ext in (".csv", ".tsv", ".txt", ".asc"):
        kind = "csv"
    elif ext == ".dat" and p.with_suffix(".hea").exists():
        kind = "wfdb"
    if not kind:
        kind = _sniff_magic(p) or "unknown"

    try:
        if kind == "wfdb":
            rec = wfdb.read(p)
        elif kind == "edf":
            rec = edf.read(p)
        elif kind == "csv":
            rec = csv_import.read(p, fs=fs)
        else:
            from . import proprietary
            rec = proprietary.read(p, fs=fs)
    except ImportError_:
        raise
    except Exception as exc:  # mensaje amable al usuario
        raise ImportError_(f"No se pudo interpretar {p.name}: {exc}") from exc

    if rec is None or rec.n_samples == 0:
        raise ImportError_(f"El archivo {p.name} no contiene señal legible")
    rec.source = str(p)
    # frecuencia implausible -> intento de rescate
    if not (20.0 <= rec.fs <= 10000.0):
        raise ImportError_(
            f"Frecuencia de muestreo implausible ({rec.fs} Hz) en {p.name}"
        )
    return rec


def probe(path: str | Path) -> dict:
    """Diagnóstico de un archivo desconocido (para adaptar formatos nuevos)."""
    from . import proprietary
    return proprietary.probe(Path(path))

