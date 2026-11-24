"""Banco de datos para detección de PAC (S vs N) con anotaciones reales.

Fuentes: MIT-BIH Arrhythmia (360 Hz) y MIT-BIH Supraventricular Arrhythmia
DB (128 Hz). Se calculan las 14 características orientadas a PAC por latido.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from holterai.legacy.app.ingest import annotations, wfdb
from holterai.legacy.app.ml.rhythm_model import RhythmModel
from holterai.legacy.app.signal import features as FT
from holterai.legacy.app.signal import pac as P
from holterai.legacy.app.signal.filters import preprocess
from holterai.legacy.app.signal.leads import pick_p_lead
from holterai.legacy.app.signal.qrs import lead_score
from holterai.legacy.app.signal.rhythm import af_probabilities

ROOT = Path(__file__).resolve().parents[5]
DATA = ROOT / "data" / "datasets"
CACHE = ROOT / "data" / "cache"


def record_features(hea: Path, af_model=None) -> tuple[np.ndarray, np.ndarray, str] | None:
    rec = wfdb.read(hea)
    ann = annotations.read_ann(hea.with_suffix(".atr"))
    beats = [b for b in ann.beats() if 0 <= b.sample < rec.n_samples]
    if len(beats) < 60:
        return None
    labels = np.array([b.aami or "" for b in beats], dtype=object)
    keep = np.isin(labels, ["N", "S"])
    beats = [b for b, k in zip(beats, keep) if k]
    labels = labels[keep]
    if len(beats) < 60 or int((labels == "S").sum()) == 0:
        return None

    leads = list(rec.leads)
    lead = pick_p_lead(leads)
    if lead is None:
        lead = int(np.argmax([lead_score(rec.samples[: min(rec.n_samples, 30000), j], rec.fs)
                              for j in range(rec.n_leads)]))
    x = rec.samples[:, lead]
    xf = preprocess(x, rec.fs, "display")

    peaks = np.array([b.sample for b in beats], dtype=int)
    fs = rec.fs
    rr = np.diff(peaks).astype(float) / fs
    rr_prev = np.concatenate([[0.0], rr])
    rr_next = np.concatenate([rr, [0.0]])
    rr_prev2 = np.concatenate([[0.0, 0.0], rr[:-1]]) if rr.size > 1 else np.zeros(len(peaks))
    rr_med = np.zeros(len(peaks))
    for i in range(len(peaks)):
        a, b = max(0, i - 5), min(len(rr_prev), i + 6)
        vals = rr_prev[a:b]
        vals = vals[(vals > 0.15) & (vals < 3.0)]
        # percentil 75: en bigeminismo la mediana se contamina con la ectopia
        rr_med[i] = np.percentile(vals, 75) if vals.size else (rr_prev[i] or 0.8)

    widths = np.array([FT.qrs_width(xf, fs, int(p)) for p in peaks])
    noises = np.array([FT.noise_level(x, fs, int(p)) for p in peaks])
    afp = af_probabilities(rr_prev, af_model) if af_model is not None else np.zeros(len(peaks))

    tpl = P.build_templates(xf, fs, peaks)
    feats = P.beat_features_batch(xf, fs, peaks, tpl, rr_prev, rr_next, rr_med,
                                  widths, noises, afp, rr_prev2)
    y = (labels == "S").astype(np.int8)
    return feats, y, hea.stem


def build(out: Path | None = None, limit_records: int | None = None) -> Path:
    out = out or (CACHE / "pac_dataset.npz")
    try:
        af_model = RhythmModel.load(ROOT / "models" / "pretrained" / "rhythm_af.npz")
    except Exception:
        af_model = None

    files: list[Path] = []
    for db in ("mitdb", "svdb"):
        d = DATA / db
        if d.exists():
            files += sorted(d.glob("*.hea"))
    if limit_records:
        files = files[:limit_records]

    X, Y, R = [], [], []
    for hea in files:
        try:
            res = record_features(hea, af_model)
        except Exception as exc:
            print(f"  {hea.stem}: error {type(exc).__name__}: {exc}")
            continue
        if res is None:
            continue
        x, y, name = res
        X.append(x); Y.append(y); R.append(np.array([f"{hea.parent.name}:{name}"] * len(y)))
        print(f"  {hea.parent.name}/{name}: {len(y)} latidos ({int(y.sum())} S)")

    if not X:
        raise RuntimeError("no se generó ningún dato")
    X = np.concatenate(X); Y = np.concatenate(Y); R = np.concatenate(R)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=Y, rec=R, names=np.array(P.PAC_FEATURES))
    print(f"total {len(Y)} latidos ({int(Y.sum())} S) -> {out}")
    return out


if __name__ == "__main__":
    build()


