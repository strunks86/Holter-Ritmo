"""Carga WFDB formato 16 (int16) a un `Recording` del legado.

El exportador de HScribe escribe 3 canales a 180 Hz con ganancia 1000/mV
(muestras en µV). Este lector no toca los archivos originales.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from holterai.legacy.app.core.models import Recording


def _parse_header(hea: Path) -> tuple[int, float, int, list[tuple[str, float, str]]]:
    lines = [ln.strip() for ln in hea.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"header vacío: {hea}")
    first = lines[0].split()
    if len(first) < 4:
        raise ValueError(f"header WFDB inválido: {lines[0]}")
    n_sig, fs, n_samples = int(first[1]), float(first[2]), int(first[3])
    channels: list[tuple[str, float, str]] = []
    for line in lines[1 : 1 + n_sig]:
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"canal inválido: {line}")
        gain_field = parts[2].split("/")[0].split("(")[0]
        gain = float(gain_field) if gain_field else 1.0
        name = parts[-1] if len(parts) >= 9 else f"ch{len(channels)}"
        channels.append((parts[0], gain, name))
    return n_sig, fs, n_samples, channels


def load_wfdb16(path: str | Path) -> Recording:
    """Carga un registro WFDB de 3 canales, formato 16, a mV."""
    p = Path(path)
    hea = p if p.suffix.lower() == ".hea" else p.with_suffix(".hea")
    n_sig, fs, n_samples, channels = _parse_header(hea)
    files = {c[0] for c in channels}
    if len(files) != 1:
        raise ValueError("el lector prototipo solo soporta un .dat intercalado")
    dat_path = hea.parent / channels[0][0]
    raw = np.fromfile(dat_path, dtype="<i2")
    need = n_sig * n_samples
    if raw.size < need:
        raise ValueError(f"{dat_path}: {raw.size} muestras < {need} esperadas")
    digital = raw[:need].reshape(n_samples, n_sig)
    gains = np.asarray([c[1] for c in channels], dtype=np.float64)
    samples = digital.astype(np.float32) / gains.astype(np.float32)
    leads = [c[2] for c in channels]
    return Recording(
        samples=samples,
        fs=fs,
        leads=leads,
        source=str(hea),
        fmt="WFDB16",
    )
