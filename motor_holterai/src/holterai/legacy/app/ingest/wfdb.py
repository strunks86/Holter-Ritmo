"""Lectura/escritura del formato WFDB (PhysioNet).

Soporta los formatos de señal 8, 16, 24, 32, 61, 80, 160 y 212, con
ganancias, líneas base y resolución ADC por derivación.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np

from holterai.legacy.app.core.models import Patient, Recording


class WFDBError(Exception):
    pass


def _parse_header(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not text:
        raise WFDBError("cabecera vacía")
    first = text[0].split()
    if len(first) < 4:
        raise WFDBError(f"cabecera inválida: {text[0][:80]!r}")
    name, n_sig, fs, n_samples = first[0], int(first[1]), float(first[2]), int(first[3])
    base_time = first[4] if len(first) > 4 else ""
    base_date = first[5] if len(first) > 5 else ""

    sigs = []
    for i in range(n_sig):
        parts = text[1 + i].split()
        if len(parts) < 2:
            raise WFDBError(f"línea de señal inválida: {text[1 + i][:80]!r}")
        fname = parts[0]
        fmt = parts[1]
        gain, adc_zero = 200.0, 0
        if len(parts) > 2 and parts[2] not in ("", "0"):
            gm = re.match(r"\s*(-?[0-9.]+)\s*(?:\(\s*(-?[0-9]+)\s*\))?", parts[2])
            if gm:
                try:
                    gain = float(gm.group(1))
                except ValueError:
                    gain = 200.0
                if gm.group(2):
                    adc_zero = int(gm.group(2))
        adc_res = int(parts[3]) if len(parts) > 3 else 16
        if len(parts) > 4 and parts[4].lstrip("-").isdigit():
            adc_zero = int(parts[4])
        init_val = int(parts[5]) if len(parts) > 5 else 0
        # descripción: tras nombre, formato, ganancia, resolución, cero, valor
        # inicial, checksum y (opcional) tamaño de bloque
        if len(parts) > 8:
            desc = " ".join(parts[8:])
        elif len(parts) > 7:
            desc = " ".join(parts[7:])
        else:
            desc = f"sig{i}"
        sigs.append({
            "file": fname,
            "fmt": fmt.strip(),
            "gain": gain if gain else 200.0,
            "adc_res": adc_res,
            "adc_zero": adc_zero,
            "init": init_val,
            "desc": desc or f"sig{i}",
        })

    comments = [ln[1:].strip() for ln in text[1 + n_sig:] if ln.startswith("#")]
    return {
        "name": name,
        "n_sig": n_sig,
        "fs": fs,
        "n_samples": n_samples,
        "base_time": base_time,
        "base_date": base_date,
        "sigs": sigs,
        "comments": comments,
        "dir": path.parent,
    }


def _read_signal_file(path: Path, fmt: str, n_samples: int, n_sig: int) -> np.ndarray:
    """Devuelve una matriz (n_samples, n_sig) de valores ADC."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    fmt = fmt.strip()

    if fmt in ("16", "16x"):
        a = raw.view("<i2")
        if a.size < n_samples * n_sig:
            raise WFDBError(f"{path.name}: datos insuficientes (formato 16)")
        return a[: n_samples * n_sig].reshape(-1, n_sig).astype(np.int64)

    if fmt in ("61",):
        a = raw.view(">i2")
        return a[: n_samples * n_sig].reshape(-1, n_sig).astype(np.int64)

    if fmt in ("80",):
        a = raw.astype(np.int64)
        a = np.where(a > 127, a - 256, a)
        return a[: n_samples * n_sig].reshape(-1, n_sig)

    if fmt in ("160",):
        a = raw.view("<u2").astype(np.int64)
        a = np.where(a > 32767, a - 65536, a)
        return a[: n_samples * n_sig].reshape(-1, n_sig)

    if fmt in ("8",):
        a = raw.view(np.int8).astype(np.int64)
        # formato 8 = diferencia de primer orden
        out = np.cumsum(a.reshape(-1, n_sig), axis=0)
        return out[:n_samples]

    if fmt in ("24",):
        if raw.size % 3:
            raw = raw[: raw.size - (raw.size % 3)]
        b = raw.reshape(-1, 3).astype(np.int64)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v >= (1 << 23), v - (1 << 24), v)
        return v[: n_samples * n_sig].reshape(-1, n_sig)

    if fmt in ("32",):
        a = raw.view("<i4")
        return a[: n_samples * n_sig].reshape(-1, n_sig).astype(np.int64)

    if fmt in ("212",):
        # dos muestras de 12 bits por cada 3 bytes
        n_pairs = raw.size // 3
        if n_pairs == 0:
            raise WFDBError(f"{path.name}: datos insuficientes (formato 212)")
        b = raw[: n_pairs * 3].reshape(-1, 3).astype(np.int64)
        b0, b1, b2 = b[:, 0], b[:, 1], b[:, 2]
        s1 = b0 | ((b1 & 0x0F) << 8)
        s2 = b2 | ((b1 & 0xF0) << 4)
        s1 = np.where(s1 > 2047, s1 - 4096, s1)
        s2 = np.where(s2 > 2047, s2 - 4096, s2)
        inter = np.empty(n_pairs * 2, dtype=np.int64)
        inter[0::2] = s1
        inter[1::2] = s2
        return inter[: n_samples * n_sig].reshape(-1, n_sig)

    raise WFDBError(f"formato de señal WFDB no soportado: {fmt}")


def read(path: str | Path) -> Recording:
    path = Path(path)
    if path.suffix.lower() != ".hea":
        path = path.with_suffix(".hea")
    head = _parse_header(path)
    sigs = head["sigs"]

    # agrupar señales por archivo .dat
    counts: dict[str, int] = {}
    fmt_of: dict[str, str] = {}
    for s in sigs:
        if s["file"] == "~":
            raise WFDBError("señales multiplexadas en un solo archivo no soportadas")
        counts[s["file"]] = counts.get(s["file"], 0) + 1
        fmt_of.setdefault(s["file"], s["fmt"])

    arrays: dict[str, np.ndarray] = {}
    for fname, count in counts.items():
        fpath = head["dir"] / fname
        if not fpath.exists():
            raise WFDBError(f"falta el archivo de señal {fname}")
        arrays[fname] = _read_signal_file(fpath, fmt_of[fname], head["n_samples"], count)

    cursor = {f: 0 for f in counts}
    cols: list[np.ndarray] = []
    for s in sigs:
        arr = arrays[s["file"]]
        k = cursor[s["file"]]
        cursor[s["file"]] = k + 1
        cols.append(arr[:, k])

    adc = np.stack(cols, axis=1) if cols else np.zeros((0, 0))
    phys = np.empty_like(adc, dtype=np.float32)
    for i, sig in enumerate(sigs):
        phys[:, i] = (adc[:, i].astype(np.float64) - sig["adc_zero"]) / sig["gain"]

    leads = [s["desc"] for s in head["sigs"]]
    leads = [re.sub(r"[^\w\+\-/\. ]", "", l) or f"Canal {i+1}" for i, l in enumerate(leads)]

    start = None
    try:
        if head["base_date"] and head["base_time"]:
            start = datetime.strptime(
                f"{head['base_date']} {head['base_time']}", "%d/%m/%Y %H:%M:%S"
            )
        elif head["base_date"]:
            start = datetime.strptime(head["base_date"], "%d/%m/%Y")
    except Exception:
        start = None

    patient = Patient()
    for c in head["comments"]:
        low = c.lower()
        if low.startswith("name:"):
            patient.name = c.split(":", 1)[1].strip()[:60]
            continue
        if low.startswith("id:"):
            patient.patient_id = c.split(":", 1)[1].strip()[:40]
            continue
        m = re.match(r"^(\d{1,3})\s+([MF])\b", c)
        if m:
            patient.age = int(m.group(1))
            patient.sex = m.group(2)
            continue
        if low.startswith("age:"):
            try:
                patient.age = int(re.search(r"\d+", c).group())
            except Exception:
                pass
    return Recording(
        samples=np.ascontiguousarray(phys, dtype=np.float32),
        fs=float(head["fs"]),
        leads=leads,
        start_time=start,
        patient=patient,
        fmt="WFDB",
        meta={"header": str(path), "comments": head["comments"][:20]},
    )


# ---------------------------------------------------------------------------
# Anotaciones (formato MIT): tipo 6 bits + intervalo 10 bits
# ---------------------------------------------------------------------------
ANN_CODE = {"N": 1, "S": 8, "V": 5, "F": 6, "Q": 13, "L": 2, "R": 3, "A": 8,
            "a": 4, "J": 7, "E": 10, "j": 11, "B": 25, "/": 12}
SKIP = 59


def _ann_word(code: int, interval: int) -> bytes:
    w = ((int(code) & 0x3F) << 10) | (int(interval) & 0x3FF)
    return bytes([w & 0xFF, (w >> 8) & 0xFF])


def write_annotations(path: str | Path, samples, symbols) -> Path:
    """Escribe un .atr compatible con WFDB (WFDB/LightWAVE pueden leerlo).

    samples: índices de muestra ordenados; symbols: 'N','S','V','F','Q'.
    """
    path = Path(path)
    if path.suffix.lower() != ".atr":
        path = path.with_suffix(".atr")
    out = bytearray()
    prev = 0
    for s, sym in zip(samples, symbols):
        s = int(s)
        iv = s - prev
        if iv < 0:
            iv = 0
        while iv > 1023:  # SKIP: intervalo largo en formato PDP-11
            chunk = min(iv, 1 << 31)
            hi = (chunk >> 16) & 0xFFFF
            lo = chunk & 0xFFFF
            out += _ann_word(SKIP, 0)
            out += bytes([hi & 0xFF, (hi >> 8) & 0xFF, lo & 0xFF, (lo >> 8) & 0xFF])
            iv -= chunk
            prev += chunk
        out += _ann_word(ANN_CODE.get(str(sym), 13), iv)
        prev = s
    out += b"\x00\x00"
    path.write_bytes(bytes(out))
    return path


# ---------------------------------------------------------------------------
# Escritura (formato 16), útil para exportar y para pruebas
# ---------------------------------------------------------------------------
def write(path: str | Path, rec: Recording, fmt: str = "16", gain: float = 200.0) -> Path:
    """Escribe un registro WFDB de una sola señal por archivo .dat."""
    path = Path(path)
    base = path.with_suffix("")
    if path.suffix.lower() == ".hea":
        base = path.with_suffix("")
    dat_name = base.name + ".dat"
    samples = np.asarray(rec.samples, dtype=np.float64)
    n, L = samples.shape
    adc = np.round(samples * gain).astype(np.int64)

    hdr = [f"{base.name} {L} {rec.fs:g} {n}"]
    for i, lead in enumerate(rec.leads):
        hdr.append(
            f"{dat_name} {fmt} {gain:g}(0)/mV 16 0 0 0 {lead.replace(' ', '_')}"
        )
    if rec.patient.name:
        hdr.append(f"# name: {rec.patient.name}")
    if rec.patient.patient_id:
        hdr.append(f"# id: {rec.patient.patient_id}")
    if rec.patient.age or rec.patient.sex:
        hdr.append(f"# {rec.patient.age or ''} {rec.patient.sex or ''}".rstrip())
    if rec.start_time:
        hdr.append(f"# {rec.start_time.strftime('%H:%M:%S')} {rec.start_time.strftime('%d/%m/%Y')}")
    hdr.append(f"# HolterIA {datetime.now().isoformat(timespec='seconds')}")
    (base.with_suffix(".hea")).write_text("\n".join(hdr) + "\n", encoding="utf-8")

    if fmt != "16":
        raise WFDBError("solo se soporta escritura en formato 16")
    adc.astype("<i2").tofile(str(base.with_suffix(".dat")))
    return base.with_suffix(".hea")

