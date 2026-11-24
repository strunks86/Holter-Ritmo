"""Construcción del conjunto de latidos anotados a partir de MIT-BIH.

Genera ventanas normalizadas + características clínicas + etiqueta AAMI,
y guarda todo en un .npz para entrenar el clasificador.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from holterai.legacy.app.core.models import Recording
from holterai.legacy.app.ingest import annotations, wfdb
from holterai.legacy.app.signal import features as F
from holterai.legacy.app.signal.filters import preprocess

# Partición inter-paciente clásica (de Chazal et al.)
DS1 = {"101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
       "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
       "223", "230"}
DS2 = {"100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
       "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
       "233", "234"}
PACED = {"102", "104", "107", "217"}
CLASSES = ["N", "S", "V", "F", "Q"]


def _local_rr_median(rr: np.ndarray, i: int, half: int = 5) -> float:
    a = max(0, i - half)
    b = min(len(rr), i + half + 1)
    vals = rr[a:b]
    vals = vals[np.isfinite(vals) & (vals > 0.15) & (vals < 3.0)]
    return float(np.median(vals)) if vals.size else 0.0


def record_beats(rec_path: Path, lead: int = 0, jitter: int = 0):
    """Extrae ventanas y etiquetas de un registro MIT-BIH."""
    rec: Recording = wfdb.read(rec_path)
    ann = annotations.read_ann(rec_path.with_suffix(".atr"))
    beats = [a for a in ann.beats() if a.sample < rec.n_samples]
    if not beats:
        return None
    x = rec.samples[:, lead]
    xf = preprocess(x, rec.fs, "display")

    idx = np.array([a.sample for a in beats], dtype=int)
    y = np.array([a.aami for a in beats])
    rr = np.diff(idx).astype(np.float64) / rec.fs
    rr_prev = np.concatenate([[0.0], rr])
    rr_next = np.concatenate([rr, [0.0]])

    wins = np.zeros((len(idx), F.WIN_POINTS), dtype=np.float32)
    feats = np.zeros((len(idx), 5), dtype=np.float32)
    for k, p in enumerate(idx):
        wins[k] = F.extract_window(xf, rec.fs, int(p), jitter=jitter)
        med = _local_rr_median(rr_prev, k)
        feats[k], _ = F.beat_features(
            xf, rec.fs, int(p), float(rr_prev[k]), float(rr_next[k]), med,
            x_raw=x,
        )
    return wins, feats, y, np.array([rec_path.stem] * len(idx))


def _augment_q(wins: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Genera latidos 'no clasificables' corrompiendo latidos normales."""
    pick = rng.integers(0, len(wins), n)
    out = wins[pick].copy()
    for i in range(n):
        amp = rng.uniform(0.10, 0.35)
        noise = rng.standard_normal(out.shape[1]).astype(np.float32) * amp
        drift = np.linspace(0, rng.uniform(-0.6, 0.6), out.shape[1]).astype(np.float32)
        out[i] = np.clip(out[i] + noise + drift, -3, 3)
    return out


def build(out_path: Path, data_dir: Path | None = None, jitter: int = 3,
          n_q: int = 4000, seed: int = 0) -> Path:
    root = data_dir or (Path(__file__).resolve().parents[5] / "data" / "datasets" / "mitdb")
    rng = np.random.default_rng(seed)
    all_w, all_f, all_y, all_r = [], [], [], []
    recs = sorted(p.stem for p in root.glob("*.hea"))
    for rec in recs:
        if rec in PACED:
            continue
        try:
            res = record_beats(root / f"{rec}.hea", jitter=jitter)
        except Exception as exc:
            print(f"  {rec}: error {exc}")
            continue
        if res is None:
            continue
        w, f, y, r = res
        all_w.append(w); all_f.append(f); all_y.append(y); all_r.append(r)
        counts = {c: int((y == c).sum()) for c in CLASSES}
        print(f"  {rec}: {len(y)} latidos {counts}")

    X = np.concatenate(all_w, axis=0)
    Fe = np.concatenate(all_f, axis=0)
    Y = np.concatenate(all_y, axis=0)
    R = np.concatenate(all_r, axis=0)

    # clase Q: latidos corrompidos (no clasificables) sintéticos
    qw = _augment_q(X[Y == "N"], n_q, rng)
    base_feat = Fe[Y == "N"].mean(axis=0) if (Y == "N").any() else np.zeros(5, np.float32)
    qf = np.tile(base_feat, (n_q, 1)).astype(np.float32)
    qf += rng.normal(0, 0.05, qf.shape).astype(np.float32)
    X = np.concatenate([X, qw], axis=0)
    Fe = np.concatenate([Fe, qf], axis=0)
    Y = np.concatenate([Y, np.array(["Q"] * n_q)])
    R = np.concatenate([R, np.array(["synthetic"] * n_q)])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X.astype(np.float16),
        F=Fe.astype(np.float32),
        y=np.array([CLASSES.index(c) for c in Y], dtype=np.int8),
        rec=R.astype("U8"),
    )
    print(f"total {len(Y)} latidos -> {out_path}")
    return out_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[5]
    build(root / "data" / "cache" / "mitdb_beats.npz")

