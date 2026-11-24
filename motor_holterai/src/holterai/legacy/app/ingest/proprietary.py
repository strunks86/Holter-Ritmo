"""Formatos propietarios y archivos desconocidos.

Estrategia: (1) sonda forense del archivo, (2) intento de reconocer un
contenedor conocido (EDF incrustado, WFDB, tabla ASCII), (3) decodificación
heurística de datos binarios int16 con parámetros tomados de la cabecera.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from holterai.legacy.app.core.models import Patient, Recording

from . import csv_import, edf, wfdb

KNOWN_MAGIC = {
    b"\x00\x00\x00\x00": "posible EDF (cabecera con ceros)",
    b"PK\x03\x04": "archivo ZIP (puede contener exportación)",
    b"\x1f\x8b": "gzip",
    b"BM": "imagen BMP",
    b"%PDF": "PDF",
    b"RIFF": "contenedor RIFF (WAV/AVI)",
}


class ProprietaryError(Exception):
    pass


# ---------------------------------------------------------------------------
# Sonda
# ---------------------------------------------------------------------------
def _entropy(b: bytes) -> float:
    if not b:
        return 0.0
    counts = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256).astype(float)
    p = counts[counts > 0] / len(b)
    return float(-(p * np.log2(p)).sum())


def _strings(b: bytes, min_len: int = 5, limit: int = 40) -> list[str]:
    out: list[str] = []
    for m in re.finditer(rb"[\x20-\x7e]{%d,}" % min_len, b):
        s = m.group().decode("latin-1").strip()
        if s:
            out.append(s[:120])
        if len(out) >= limit:
            break
    return out


def probe(path: str | Path, max_bytes: int = 1 << 20) -> dict:
    p = Path(path)
    size = p.stat().st_size
    with p.open("rb") as f:
        head = f.read(min(size, max_bytes))
    magic = head[:8]
    info = {
        "path": str(p),
        "name": p.name,
        "size": size,
        "ext": p.suffix.lower(),
        "magic_hex": magic.hex(),
        "magic_guess": KNOWN_MAGIC.get(magic[:2], "") or KNOWN_MAGIC.get(magic[:4], ""),
        "entropy": round(_entropy(head[:200_000]), 3),
        "first_text": _strings(head[:4096], 6, 12),
        "header_strings": _strings(head[:65536], 6, 40),
        "candidates": [],
        "size_mod_2": size % 2,
        "size_mod_3": size % 3,
        "size_mod_4": size % 4,
    }

    # ¿EDF en algún desplazamiento?
    edf_at = head.find(b"0       ")
    if edf_at < 0 and head[:8].startswith(b"\x00"):
        edf_at = 0
    if edf_at >= 0:
        info["candidates"].append({"format": "EDF", "offset": int(edf_at)})

    # ¿tabla ASCII?
    text_ratio = sum(ch in b"0123456789.,;:\t\r\n -+eE" for ch in head[:20000]) / max(1, min(20000, len(head)))
    if text_ratio > 0.85:
        info["candidates"].append({"format": "CSV/TXT", "confidence": round(text_ratio, 3)})

    # parámetros típicos en las cadenas de cabecera
    joined = " ".join(info["header_strings"])
    for key in ("FS", "FS ", "SAMPLE", "RATE", "NCH", "CHANNEL", "GAIN", "ADC", "SAMPLING"):
        if key in joined.upper():
            info.setdefault("param_hints", {})[key] = True

    # entropía por bloques (detecta zona de cabecera vs datos)
    blocks = []
    step = max(1, len(head) // 16)
    for i in range(0, min(len(head), 16 * step), step):
        blocks.append(round(_entropy(head[i:i + step]), 2))
    info["entropy_blocks"] = blocks
    return info


# ---------------------------------------------------------------------------
# Lectura
# ---------------------------------------------------------------------------
def _ascii_params(path: Path) -> dict:
    """Busca parámetros de muestreo en cabeceras de texto."""
    with path.open("rb") as f:
        head = f.read(65536)
    txt = head.decode("latin-1", "replace")
    params: dict = {}
    pats = {
        "fs": r"(?:sampling\s*rate|sample\s*rate|frecuencia|fs|rate)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        "n_ch": r"(?:channels|canales|nch|n_sig|leads)\s*[:=]\s*([0-9]+)",
        "gain": r"(?:gain|ganancia)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        "offset": r"(?:offset|zero|cero)\s*[:=]\s*(-?[0-9]+)",
    }
    for k, pat in pats.items():
        m = re.search(pat, txt, re.IGNORECASE)
        if m:
            try:
                params[k] = float(m.group(1))
            except Exception:
                pass
    m = re.search(r"(?:header|data)\s*(?:bytes|offset)\s*[:=]\s*([0-9]+)", txt, re.IGNORECASE)
    if m:
        params["header_bytes"] = int(m.group(1))
    return params


def _read_binary_int16(path: Path, n_ch: int, fs: float, gain: float = 200.0,
                       offset: int = 0, header_bytes: int = 0) -> Recording:
    data = np.fromfile(str(path), dtype="<i2", offset=header_bytes)
    if data.size < n_ch * 10:
        raise ProprietaryError("datos binarios insuficientes")
    if data.size % n_ch:
        data = data[: data.size - (data.size % n_ch)]
    mat = data.reshape(-1, n_ch).astype(np.float32)
    mat = (mat + offset) / gain
    if float(np.percentile(np.abs(mat), 99)) > 20:
        mat = mat / 1000.0
    return Recording(
        samples=mat,
        fs=fs,
        leads=[f"Canal {i+1}" for i in range(n_ch)],
        fmt="BIN16 (heurístico)",
        meta={"header_bytes": header_bytes, "gain": gain, "warning": "decodificación heurística"},
    )


def read(path: str | Path, fs: float | None = None) -> Recording:
    p = Path(path)
    if p.suffix.lower() == ".hea" or p.with_suffix(".hea").exists():
        return wfdb.read(p)

    with p.open("rb") as f:
        magic = f.read(8)

    # EDF aunque la extensión no lo diga
    if magic.startswith(b"\x00") or b"EDF" in magic:
        try:
            return edf.read(p)
        except Exception:
            pass

    # ZIP: buscar un EDF/CSV dentro
    if magic.startswith(b"PK\x03\x04"):
        import zipfile
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            for n in names:
                if n.lower().endswith((".edf", ".csv", ".txt", ".hea")):
                    tmp = z.extract(n, p.parent / "_holteria_extract")
                    return read(tmp, fs=fs)

    info = probe(p)
    joined = " ".join(info["header_strings"]).upper()

    # exportación ASCII con cabecera
    if any(c["format"] == "CSV/TXT" for c in info["candidates"]):
        try:
            rec = csv_import.read(p, fs=fs)
            rec.fmt = "TXT/CSV (propietario)"
            return rec
        except Exception:
            pass

    # binario int16 con parámetros en cabecera
    params = _ascii_params(p)
    n_ch = int(params.get("n_ch", 0) or 0)
    fs_val = fs or params.get("fs") or _fs_from_strings(joined) or 250.0
    header_bytes = int(params.get("header_bytes", 0) or 0)
    if n_ch <= 0:
        n_ch = _infer_channels(p, header_bytes, fs_val)
    if n_ch > 0:
        gain = float(params.get("gain", 200.0) or 200.0)
        try:
            rec = _read_binary_int16(p, n_ch, float(fs_val), gain=gain,
                                     offset=int(params.get("offset", 0) or 0),
                                     header_bytes=header_bytes)
            rec.meta["probe"] = info
            rec.fmt = "BIN16 (heurístico)"
            return rec
        except ProprietaryError:
            pass

    raise ProprietaryError(
        "Formato no reconocido. Ejecuta la sonda de diagnóstico (botón "
        "'Analizar archivo') y comparte el resultado para añadir soporte."
    )


def _fs_from_strings(s: str) -> float | None:
    m = re.search(r"(\d{2,4})\s*(?:HZ|HZ\.)", s)
    return float(m.group(1)) if m else None


def _infer_channels(p: Path, header_bytes: int, fs: float) -> int:
    """Infiere el número de canales asumiendo 16 bits y 24 h de registro."""
    size = p.stat().st_size - header_bytes
    for ch in (1, 2, 3, 4, 5, 6):
        if size % (2 * ch):
            continue
        n = size // (2 * ch)
        hours = n / fs / 3600.0
        if 0.02 <= hours <= 72.0:
            return ch
    return 0

