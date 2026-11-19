"""Registro de demostración.

Si están disponibles los datos de MIT-BIH (descargados con
app.tools.fetch_datasets), el ejemplo usa un tramo REAL anotado; si no,
genera uno sintético. Así la primera impresión de la app es representativa.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from holterai.legacy.app.core.config import DATA_DIR
from holterai.legacy.app.core.models import Patient, Recording
from holterai.legacy.app.ingest import wfdb

DEMO_SOURCES = [
    ("svdb", "821", "MIT-BIH SVDB 821 (demostración)", "Arritmias supraventriculares"),
    ("mitdb", "119", "MIT-BIH 119 (demostración)", "Ectopia ventricular"),
]
DEMO_SECONDS = 30 * 60


def _from_dataset(out: Path) -> Path | None:
    for db, rec_name, label, note in DEMO_SOURCES:
        src = DATA_DIR / "datasets" / db / f"{rec_name}.hea"
        if not src.exists():
            continue
        rec = wfdb.read(src)
        n = min(rec.n_samples, int(DEMO_SECONDS * rec.fs))
        meta = dict(rec.meta)
        meta["attribution"] = (
            f"MIT-BIH {db.upper()} {rec_name}, PhysioNet. Licencia ODC-BY 1.0. "
            "Muestra de demostración."
        )
        meta["demo"] = True
        demo = Recording(
            samples=rec.samples[:n].copy(),
            fs=rec.fs,
            leads=rec.leads,
            start_time=rec.start_time,
            patient=Patient(name=label, patient_id=f"{db.upper()}-{rec_name}", notes=note),
            fmt="WFDB",
            meta=meta,
        )
        wfdb.write(out, demo)
        return out
    return None


def _synthetic(out: Path) -> Path:
    from holterai.legacy.app.signal.synth import generate_segment
    plan = [
        dict(duration=600, mode="nsr", hr_bpm=68, pvc_burden=0.01, noise=0.8, seed=11),
        dict(duration=300, mode="af", hr_bpm=96, noise=1.0, seed=12),
        dict(duration=600, mode="nsr", hr_bpm=72, pvc_burden=0.08, pac_burden=0.02, noise=1.1, seed=13),
    ]
    fs, parts = 250.0, []
    for p in plan:
        sig, _, _, _ = generate_segment(fs=fs, n_leads=3, **p)
        parts.append(sig)
    fade = int(0.5 * fs)
    merged = parts[0].copy()
    for nxt in parts[1:]:
        w = np.linspace(0, 1, fade, dtype=np.float32)[:, None]
        merged[-fade:] = merged[-fade:] * (1 - w) + nxt[:fade] * w
        merged = np.vstack([merged, nxt[fade:]])
    rec = Recording(
        samples=merged.astype(np.float32), fs=fs, leads=["II", "V1", "V5"],
        patient=Patient(name="Paciente de ejemplo", age=71, sex="M", patient_id="DEMO-001"),
        fmt="DEMO",
        meta={"note": "Registro sintético con FA y ectopia", "demo": True},
    )
    wfdb.write(out, rec)
    return out


def build(path: Path | None = None, force: bool = False) -> Path:
    out = Path(path) if path else (DATA_DIR / "demo" / "demo.hea")
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    return _from_dataset(out) or _synthetic(out)


if __name__ == "__main__":
    print("demo:", build(force=True))

