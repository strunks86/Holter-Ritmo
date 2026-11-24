"""Conjunto de datos de ritmo (FA vs no FA) a partir de MIT-BIH AFDB.

Usa solo las anotaciones (no hace falta descargar las señales de 10 h):
de cada registro se extraen los intervalos RR y el ritmo anotado en cada
momento, y se construyen ventanas deslizantes de 30 RR.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from holterai.legacy.app.ingest import annotations
from holterai.legacy.app.signal.rhythm import AF_FEATURES, rr_features

AF_TAGS = ("(AFIB", "(AFL")
SR_TAGS = ("(N", "(SBR", "(B", "(J", "(T")


def _rhythm_timeline(ann: annotations.Annotations) -> list[tuple[int, str]]:
    """Lista ordenada (muestra, etiqueta de ritmo)."""
    out: list[tuple[int, str]] = [(0, "SR")]
    for a in ann.items:
        if not a.aux:
            continue
        tag = a.aux.strip()
        if tag.startswith("(AFIB"):
            out.append((a.sample, "AF"))
        elif tag.startswith("(AFL"):
            out.append((a.sample, "AFL"))
        elif tag.startswith("(N"):
            out.append((a.sample, "SR"))
        elif tag.startswith("("):
            out.append((a.sample, "OTHER"))
    return out


def _label_at(timeline: list[tuple[int, str]], sample: int) -> str:
    lab = "SR"
    for s, l in timeline:
        if s <= sample:
            lab = l
        else:
            break
    return lab


def record_windows(atr_path: Path, win: int = 30, stride: int = 5):
    ann = annotations.read_ann(atr_path)
    beats = [a for a in ann.beats() if a.sample >= 0]
    if len(beats) < win + 2:
        # en AFDB los latidos están en un flujo .qrs aparte
        qrs = atr_path.with_suffix(".qrs")
        if qrs.exists():
            beats = [a for a in annotations.read_ann(qrs).beats() if a.sample >= 0]
    if len(beats) < win + 2:
        return None
    samples = np.array([b.sample for b in beats], dtype=np.int64)
    fs = 250.0
    rr = np.diff(samples).astype(np.float64) / fs
    timeline = _rhythm_timeline(ann)

    rows, labels = [], []
    for end in range(win, len(rr) + 1, stride):
        seg = rr[end - win:end]
        if np.any((seg < 0.2) | (seg > 3.0)):
            continue
        labs = [_label_at(timeline, s) for s in samples[end - win:end + 1]]
        labs_set = set(labs)
        if "OTHER" in labs_set or len(labs_set) > 1:
            continue  # evita ventanas a caballo entre ritmos
        lab = labs[0]
        if lab not in ("AF", "SR"):
            continue
        rows.append(rr_features(seg))
        labels.append(1 if lab == "AF" else 0)
    if not rows:
        return None
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int8)


def build_afdb(afdb_dir: Path):
    X, y, recs = [], [], []
    for atr in sorted(afdb_dir.glob("*.atr")):
        res = record_windows(atr)
        if res is None:
            print(f"  {atr.stem}: sin ventanas")
            continue
        Xi, yi = res
        X.append(Xi); y.append(yi)
        recs.append(np.array([atr.stem] * len(yi)))
        print(f"  {atr.stem}: {len(yi)} ventanas ({int(yi.sum())} FA)")
    if not X:
        return np.zeros((0, len(AF_FEATURES)), np.float32), np.zeros(0, np.int8), np.zeros(0, "U8")
    return np.concatenate(X).astype(np.float32), np.concatenate(y), np.concatenate(recs)


def build_mitdb_negatives(mitdb_dir: Path, records: set[str], per_rec: int = 400,
                          seed: int = 0):
    """Ventanas negativas con ectopia real (PVC/PAC) desde MIT-BIH."""
    rng = np.random.default_rng(seed)
    X, y, recs = [], [], []
    for rec in sorted(records):
        hea = mitdb_dir / f"{rec}.hea"
        if not hea.exists():
            continue
        ann = annotations.read_ann(hea.with_suffix(".atr"))
        beats = [a for a in ann.beats()]
        if len(beats) < 40:
            continue
        samples = np.array([b.sample for b in beats], dtype=np.int64)
        rr = np.diff(samples).astype(np.float64) / 360.0
        if rr.size < 31:
            continue
        starts = rng.integers(0, rr.size - 30, min(per_rec, max(1, rr.size - 30)))
        for s in starts:
            seg = rr[s:s + 30]
            if np.any((seg < 0.2) | (seg > 3.0)):
                continue
            X.append(rr_features(seg))
            y.append(0)
            recs.append(rec)
    if not X:
        return np.zeros((0, len(AF_FEATURES)), np.float32), np.zeros(0, np.int8), np.zeros(0, "U8")
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8),
            np.asarray(recs))


def build_svdb_negatives(svdb_dir: Path, max_s_fraction: float = 0.25):
    """Ventanas sin FA procedentes de SVDB (PAC/SVT), el caso que confunde al modelo.

    Se usan solo registros con ectopia auricular moderada: los que tienen
    taquicardias sostenidas no sirven como negativos limpios.
    """
    import os
    from holterai.legacy.app.ingest import annotations as ann_mod
    X, y, recs = [], [], []
    for atr in sorted(svdb_dir.glob("*.atr")):
        try:
            ann = ann_mod.read_ann(atr)
        except Exception:
            continue
        beats = [b for b in ann.beats()]
        if len(beats) < 40:
            continue
        labels = [b.aami or "" for b in beats]
        s_frac = sum(1 for l in labels if l == "S") / max(1, len(labels))
        if s_frac > max_s_fraction:
            continue
        samples = np.array([b.sample for b in beats], dtype=np.int64)
        fs = 128.0
        try:
            fs = float(atr.with_suffix(".hea").read_text(encoding="latin-1").split()[2])
        except Exception:
            pass
        rr = np.diff(samples).astype(np.float64) / fs
        if rr.size < 32:
            continue
        n = 0
        for end in range(30, len(rr) + 1, 5):
            seg = rr[end - 30:end]
            if np.any((seg < 0.2) | (seg > 3.0)):
                continue
            X.append(rr_features(seg)); y.append(0); recs.append(atr.stem); n += 1
        if n:
            print(f"  {atr.stem}: {n} ventanas (S {s_frac*100:.0f}%)")
    if not X:
        return (np.zeros((0, len(AF_FEATURES)), np.float32), np.zeros(0, np.int8),
                np.zeros(0, "U16"))
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8),
            np.asarray(recs, dtype="U16"))


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parents[5]
    d = root / "data" / "datasets"
    X1, y1, r1 = build_afdb(d / "afdb")
    from holterai.legacy.app.ml.dataset_mitdb import DS1
    X2, y2, r2 = build_mitdb_negatives(d / "mitdb", DS1)
    print("negativos SVDB (PAC sin FA):")
    X3, y3, r3 = build_svdb_negatives(d / "svdb")
    X = np.concatenate([X1, X2, X3]); y = np.concatenate([y1, y2, y3])
    r = np.concatenate([r1, r2, r3])
    out = root / "data" / "cache" / "rhythm_af.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, rec=r, names=np.array(AF_FEATURES))
    print(f"total {len(y)} ventanas ({int(y.sum())} FA) -> {out}")

