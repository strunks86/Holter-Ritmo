"""Exportación EDF+ mínima de un Holter WFDB.

Solo escribe señal y cabecera; no añade anotaciones ni diagnóstico. El objetivo
es interoperabilidad, no sustituir un visualizador clínico.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def _pad(text: str, n: int) -> bytes:
    return text.encode("ascii", "replace")[:n].ljust(n, b" ")


def _num(value: float, n: int = 8) -> bytes:
    return f"{value:.6g}".encode("ascii")[:n].ljust(n, b" ")


def write_edf(
    samples: Any,
    fs: float,
    leads: list[str],
    path: str | Path,
    *,
    patient_id: str = "",
    sex: str = "",
    dob: str = "",
    name: str = "",
    start: datetime | None = None,
) -> Path:
    """Escribe EDF+ con registros de 1 segundo y resolución int16.

    `samples` debe ser (n_muestras, n_derivaciones) en mV.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[1] != len(leads):
        raise ValueError("samples y leads no coinciden")
    n, ns = x.shape
    if fs <= 0:
        raise ValueError("fs inválida")
    spr = max(1, int(round(fs)))
    n_rec = max(1, int(np.ceil(n / spr)))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = start or datetime.now()
    # EDF+ patient: code sex birthdate name
    patient = " ".join(x for x in (patient_id, sex, dob, name) if x) or "X"
    recording = f"Startdate {t0:%d-%b-%Y} {t0:%H:%M:%S} HolterIA"
    header_bytes = 256 + ns * 256

    h = bytearray()
    h += _pad("0", 8)
    h += _pad(patient, 80)
    h += _pad(recording, 80)
    h += _pad(t0.strftime("%d.%m.%y"), 8)
    h += _pad(t0.strftime("%H.%M.%S"), 8)
    h += _pad(str(header_bytes), 8)
    h += _pad("", 44)
    h += _pad(str(n_rec), 8)
    h += _pad("1", 8)  # 1 s por registro
    h += _pad(str(ns), 4)

    for j, lead in enumerate(leads):
        col = x[:, j]
        pmin = float(np.min(col)) if col.size else -1.0
        pmax = float(np.max(col)) if col.size else 1.0
        if pmax <= pmin:
            pmax = pmin + 1.0
        h += _pad(str(lead), 16)
        h += _pad("", 80)
        h += _pad("mV", 8)
        h += _num(pmin)
        h += _num(pmax)
        h += _pad("-32768", 8)
        h += _pad("32767", 8)
        h += _pad("", 80)
        h += _pad(str(spr), 8)
        h += _pad("", 32)

    out.write_bytes(bytes(h))
    with out.open("ab") as f:
        for r in range(n_rec):
            chunk = x[r * spr : (r + 1) * spr]
            if chunk.shape[0] < spr:
                chunk = np.vstack([chunk, np.zeros((spr - chunk.shape[0], ns))])
            for j in range(ns):
                col = chunk[:, j]
                pmin = float(np.min(x[:, j])) if n else -1.0
                pmax = float(np.max(x[:, j])) if n else 1.0
                if pmax <= pmin:
                    pmax = pmin + 1.0
                dig = np.round((col - pmin) / (pmax - pmin) * 65535.0 - 32768.0)
                f.write(np.clip(dig, -32768, 32767).astype("<i2").tobytes())
    return out
