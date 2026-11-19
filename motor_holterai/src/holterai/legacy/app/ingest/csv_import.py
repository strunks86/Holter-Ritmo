"""Importación de exportaciones CSV/TXT de Holter.

Acepta exportaciones con o sin cabecera, separadas por coma, punto y coma,
tabulador o espacios, con o sin columna de tiempo.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from holterai.legacy.app.core.models import Patient, Recording

TIME_HINTS = ("time", "tiempo", "t_sec", "t (s)", "time_s", "second", "seg", "ms", "timestamp")
NON_SIGNAL_HINTS = ("time", "tiempo", "index", "idx", "n", "num", "beat", "event", "label", "annotation")


class CSVError(Exception):
    pass


def _detect_sep(sample: str) -> str:
    best, best_n = ",", -1
    for sep in (",", ";", "\t", "|"):
        n = sample.count(sep)
        if n > best_n:
            best, best_n = sep, n
    return best


def _read_text(path: Path, max_chars: int = 200_000) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return f.read(max_chars)


def _guess_fs_from_name(name: str) -> float | None:
    m = re.search(r"(\d{2,5})\s*(?:hz|Hz|HZ)", name)
    if m:
        return float(m.group(1))
    m = re.search(r"(?:fs|frecuencia|rate)[_\-]?(\d{2,5})", name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _infer_fs_from_time(t: np.ndarray) -> float | None:
    """Infiere fs probando unidades (s, ms, µs) y quedándose con la plausible."""
    dt = np.diff(np.asarray(t, dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if not dt.size:
        return None
    med = float(np.median(dt))
    for factor in (1.0, 1e-3, 1e-6):
        fs = 1.0 / (med * factor)
        if 20.0 <= fs <= 5000.0:
            return float(fs)
    return None


def read(path: str | Path, fs: float | None = None) -> Recording:
    path = Path(path)
    head = _read_text(path)
    if not head.strip():
        raise CSVError("archivo vacío")

    # localizar la primera línea con datos numéricos
    lines = head.splitlines()
    data_start = 0
    header_line = ""
    for i, ln in enumerate(lines[:30]):
        s = ln.strip()
        if not s:
            continue
        nums = re.findall(r"[-+]?\d+(?:[.,]\d+)?", s.replace("\t", " "))
        tokens = re.split(r"[,;\t| ]+", s)
        if len(nums) >= len(tokens) * 0.8 and len(tokens) >= 2:
            data_start = i
            break
        header_line = s
        data_start = i + 1
    sep = _detect_sep(lines[data_start] if data_start < len(lines) else head)

    # cabecera de columnas
    cols: list[str] = []
    if header_line:
        cols = [c.strip().strip('"').strip("'") for c in re.split(r"[,;\t|]+|\s{2,}", header_line)]
        cols = [c for c in cols if c]
    try:
        table = np.genfromtxt(
            path, delimiter=sep, skip_header=data_start, comments="#",
            dtype=np.float64, invalid_raise=False,
        )
    except Exception as exc:
        raise CSVError(f"no se pudo leer la tabla numérica: {exc}") from exc
    if table.ndim == 1:
        table = table[:, None]
    table = np.atleast_2d(table)
    table = table[~np.isnan(table).all(axis=1)]
    if table.size == 0:
        raise CSVError("no hay filas numéricas")

    # separar columna de tiempo
    time_col = -1
    if cols and len(cols) == table.shape[1]:
        for j, c in enumerate(cols):
            if any(h in c.lower() for h in TIME_HINTS):
                time_col = j
                break

    fs_val = fs
    if time_col >= 0:
        fs_time = _infer_fs_from_time(table[:, time_col])
        if fs_time:
            fs_val = fs_time
    if fs_val is None:
        fs_val = _guess_fs_from_name(path.name) or 250.0

    if time_col >= 0:
        table = np.delete(table, time_col, axis=1)
        if cols:
            cols.pop(time_col)

    # descartar columnas que claramente no son señal
    keep = list(range(table.shape[1]))
    if cols and len(cols) == table.shape[1]:
        keep = [j for j, c in enumerate(cols)
                if not any(h == c.lower().strip() or c.lower().strip().startswith(h) for h in NON_SIGNAL_HINTS)]
        if not keep:
            keep = list(range(table.shape[1]))
    table = table[:, keep]
    if cols and len(cols) == table.shape[1]:
        leads = [c[:20] for c in cols]
    else:
        leads = [f"Canal {i+1}" for i in range(table.shape[1])]

    # unidad: si la amplitud típica es enorme, probablemente µV o cuentas ADC
    med_abs = float(np.median(np.abs(table))) + 1e-9
    unit = "mV"
    if np.nanpercentile(np.abs(table), 99) > 30.0:
        table = table / 1000.0
        unit = "uV->mV"
    elif np.nanpercentile(np.abs(table), 99) > 5000.0:
        table = table / 200.0
        unit = "ADC->mV"

    return Recording(
        samples=np.ascontiguousarray(table.astype(np.float32)),
        fs=float(fs_val),
        leads=leads,
        patient=Patient(),
        fmt="CSV",
        meta={"columns": cols, "unit": unit, "separator": sep},
    )


def write(path: str | Path, rec: Recording, decimals: int = 4) -> Path:
    path = Path(path)
    n = rec.n_samples
    t = np.arange(n) / rec.fs
    arr = np.column_stack([t, rec.samples])
    header = "time_s," + ",".join(rec.leads)
    np.savetxt(str(path), arr, delimiter=",", header=header, comments="",
               fmt=f"%.{decimals}f")
    return path

