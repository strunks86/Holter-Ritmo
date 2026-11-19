"""Lectura y escritura de EDF / EDF+ (European Data Format)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from holterai.legacy.app.core.models import Patient, Recording


class EDFError(Exception):
    pass


def _f(buf: bytes, default=0.0) -> float:
    s = buf.decode("latin-1").strip().strip("\x00")
    try:
        return float(s)
    except Exception:
        return default


def _i(buf: bytes, default=0) -> int:
    s = buf.decode("latin-1").strip().strip("\x00")
    try:
        return int(float(s))
    except Exception:
        return default


def _parse_time(date_s: str, time_s: str) -> datetime | None:
    try:
        d = date_s.strip().strip("\x00")
        t = time_s.strip().strip("\x00")
        return datetime.strptime(f"{d} {t}", "%d.%m.%y %H.%M.%S")
    except Exception:
        return None


def read_header(fh) -> dict:
    head = fh.read(256)
    if len(head) < 256:
        raise EDFError("archivo demasiado corto para ser EDF")
    # Cabecera EDF: versión(8) paciente(80) grabación(80) fecha(8) hora(8)
    # bytes_cabecera(8) reservado(44) nº_registros(8) dur_registro(8) nº_señales(4)
    n_sig = _i(head[252:256])
    if not (1 <= n_sig <= 640):
        raise EDFError(f"número de señales inválido: {n_sig}")
    head_bytes = _i(head[184:192], 256 + n_sig * 256)
    if head_bytes < 256 + n_sig * 256 or head_bytes > 10_000_000:
        head_bytes = 256 + n_sig * 256
    n_records = _i(head[236:244], -1)
    rec_dur = _f(head[244:252], 1.0)
    if rec_dur <= 0:
        rec_dur = 1.0

    fields = fh.read(head_bytes - 256)
    if len(fields) < n_sig * 256:
        raise EDFError("cabecera EDF truncada")

    # distribución de campos por señal
    layout = [
        ("label", 16), ("transducer", 80), ("dim", 8), ("pmin", 8), ("pmax", 8),
        ("dmin", 8), ("dmax", 8), ("prefilter", 80), ("spr", 8), ("reserved", 32),
    ]
    sigs = [{} for _ in range(n_sig)]
    off = 0
    for k in range(n_sig):  # los campos van por señal, no por campo
        for key, width in layout:
            sigs[k][key] = fields[off:off + width].decode("latin-1")
            off += width

    for s in sigs:
        s["label"] = s["label"].strip().strip("\x00")
        s["dim"] = s["dim"].strip().strip("\x00")
        s["pmin"] = _f(s["pmin"].encode("latin-1"))
        s["pmax"] = _f(s["pmax"].encode("latin-1"))
        s["dmin"] = _i(s["dmin"].encode("latin-1"))
        s["dmax"] = _i(s["dmax"].encode("latin-1"))
        s["spr"] = _i(s["spr"].encode("latin-1"))
        if s["spr"] <= 0:
            raise EDFError(f"señal {s['label']!r} sin muestras")
        if s["dmax"] == s["dmin"]:
            s["dmax"] = s["dmin"] + 1
        s["scale"] = (s["pmax"] - s["pmin"]) / (s["dmax"] - s["dmin"])

    return {
        "version": head[0:8].decode("latin-1").strip(),
        "patient": head[8:88].decode("latin-1").strip(),
        "recording": head[88:168].decode("latin-1").strip(),
        "start": _parse_time(head[168:176].decode("latin-1"), head[176:184].decode("latin-1")),
        "reserved": head[192:236].decode("latin-1").strip(),
        "n_records": n_records,
        "rec_dur": rec_dur,
        "n_sig": n_sig,
        "head_bytes": head_bytes,
        "sigs": sigs,
    }


def _is_ecg(label: str, dim: str) -> bool:
    lab = label.upper()
    d = dim.upper()
    if any(k in lab for k in ("ECG", "EKG", "HOLTER", "RITMO")):
        return True
    if d in ("MV", "UV", "µV", "V") and not any(
        k in lab for k in ("EEG", "EOG", "EMG", "RESP", "FLOW", "SPO2", "TEMP", "ART")
    ):
        return True
    return False


def read(path: str | Path) -> Recording:
    path = Path(path)
    with path.open("rb") as fh:
        h = read_header(fh)
        sigs = h["sigs"]
        sel = [k for k, s in enumerate(sigs) if _is_ecg(s["label"], s["dim"])]
        if not sel:
            sel = [k for k in range(len(sigs)) if "ANNOT" not in sigs[k]["label"].upper()]
        if not sel:
            raise EDFError("no se encontraron canales de ECG en el EDF")

        spr = sigs[sel[0]]["spr"]
        fs = spr / h["rec_dur"]
        # si las señales elegidas tienen distinta frecuencia, nos quedamos con la moda
        fss = [sigs[k]["spr"] / h["rec_dur"] for k in sel]
        if max(fss) - min(fss) > 0.01:
            fs = float(np.median(fss))
            sel = [k for k in sel if abs(sigs[k]["spr"] / h["rec_dur"] - fs) < 0.01]
            spr = sigs[sel[0]]["spr"]

        heads = [(s, s["scale"], s["pmin"], s["dmin"], s["spr"]) for s in (sigs[k] for k in sel)]
        sprs = [s["spr"] for s in sigs]
        total_spr = sum(sprs)
        offs = np.concatenate([[0], np.cumsum(sprs)])
        rec_len = total_spr * 2
        n_rec = h["n_records"]
        file_size = path.stat().st_size
        max_rec = max(0, (file_size - h["head_bytes"]) // rec_len) if rec_len else 0
        if n_rec < 0 or n_rec > max_rec:
            n_rec = max_rec
        if n_rec <= 0:
            raise EDFError("el EDF no contiene registros de datos")

        total = n_rec * spr
        out = np.empty((total, len(sel)), dtype=np.float32)

        # leer por bloques de registros para no cargar todo el archivo
        per_rec = int(max(1, (1 << 24) // max(1, rec_len)))
        pos = h["head_bytes"]
        fh.seek(pos)
        for start in range(0, n_rec, per_rec):
            k = min(per_rec, n_rec - start)
            buf = fh.read(k * rec_len)
            if len(buf) < k * rec_len:
                k = len(buf) // rec_len
                if k == 0:
                    break
                buf = buf[: k * rec_len]
            arr = np.frombuffer(buf, dtype="<i2").reshape(k, total_spr)
            for j, (s, scale, pmin, dmin, sp) in enumerate(heads):
                d = arr[:, int(offs[sel[j]]):int(offs[sel[j]] + sp)].astype(np.float32)
                out[start * spr:(start + k) * spr, j] = ((d - dmin) * scale + pmin).reshape(-1)
        # EDF unidades físicas: mV o uV
        for j, (s, *_rest) in enumerate(heads):
            if s["dim"].upper() in ("UV", "µV"):
                out[:, j] /= 1000.0
            elif s["dim"].upper() == "V":
                out[:, j] *= 1000.0

    patient = Patient()
    m = h["patient"]
    # EDF+ : "code sex birthdate name" ; clásico: "name sex age ..."
    try:
        parts = m.split()
        if len(parts) >= 4:
            for tok in parts:
                if tok.upper() in ("M", "F", "H", "MASCULINO", "FEMENINO"):
                    patient.sex = "M" if tok.upper().startswith(("M", "H")) else "F"
        name = " ".join(p for p in parts[:3] if p.upper() not in ("M", "F", "X"))
        patient.name = name.strip()[:60]
    except Exception:
        patient.name = m[:60]

    leads = [sigs[k]["label"].strip() or f"Canal {i+1}" for i, k in enumerate(sel)]
    return Recording(
        samples=np.ascontiguousarray(out, dtype=np.float32),
        fs=float(fs),
        leads=leads,
        start_time=h["start"],
        patient=patient,
        fmt="EDF" if "EDF+" not in h["reserved"] else "EDF+",
        meta={
            "edf_patient": h["patient"],
            "edf_recording": h["recording"],
            "n_records": n_rec,
            "record_dur": h["rec_dur"],
            "all_channels": [s["label"] for s in sigs],
        },
    )


def write(path: str | Path, rec: Recording, record_dur: float = 1.0) -> Path:
    """Escribe un EDF+ mínimo (16 bits) con los canales del registro."""
    path = Path(path)
    samples = np.asarray(rec.samples, dtype=np.float32)
    n, L = samples.shape
    spr = int(round(rec.fs * record_dur))
    if spr <= 0:
        raise EDFError("frecuencia de muestreo inválida")
    n_rec = int(np.ceil(n / spr))
    pad = n_rec * spr - n
    if pad:
        samples = np.vstack([samples, np.zeros((pad, L), dtype=np.float32)])

    head = b"0       "
    head += rec.patient.name[:80].ljust(80).encode("latin-1", "replace")
    head += b"HolterIA".ljust(80)
    st = rec.start_time or datetime.now()
    head += st.strftime("%d.%m.%y").encode("latin-1")
    head += st.strftime("%H.%M.%S").encode("latin-1")
    head += str(256 + L * 256).encode("latin-1").ljust(8)
    head += b"EDF+C".ljust(44) + str(n_rec).encode("latin-1").ljust(8)
    head += f"{record_dur:g}".encode("latin-1").ljust(8)
    head += str(L).encode("latin-1").ljust(4)

    pmin, pmax = -5.0, 5.0
    dmin, dmax = -32768, 32767
    for i in range(L):
        head += rec.leads[i][:16].ljust(16).encode("latin-1", "replace")
        head += b"".ljust(80) + b"mV".ljust(8)
        head += f"{pmin:g}".ljust(8).encode("latin-1")
        head += f"{pmax:g}".ljust(8).encode("latin-1")
        head += f"{dmin}".ljust(8).encode("latin-1")
        head += f"{dmax}".ljust(8).encode("latin-1")
        head += b"".ljust(80) + str(spr).ljust(8).encode("latin-1") + b"".ljust(32)

    scale = (dmax - dmin) / (pmax - pmin)
    digital = np.round((samples - pmin) * scale + dmin).clip(dmin, dmax).astype("<i2")
    # EDF almacena por registros: para cada registro, todas las señales seguidas
    digital = np.ascontiguousarray(
        digital.reshape(n_rec, spr, L).transpose(0, 2, 1)
    )
    with path.open("wb") as f:
        f.write(head)
        f.write(digital.tobytes())
    return path

