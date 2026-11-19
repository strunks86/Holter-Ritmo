"""Modelo de ritmo (FA vs no FA) sobre características de la serie RR.

Regresión logística entrenada en numpy; se guarda como .npz y se ejecuta
sin dependencias externas.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from holterai.legacy.app.signal.rhythm import AF_FEATURES

ROOT = Path(__file__).resolve().parents[5]
MODELS = ROOT / "models" / "pretrained"


class RhythmModel:
    def __init__(self, w: np.ndarray, b: float, mean: np.ndarray, std: np.ndarray):
        self.w = np.asarray(w, dtype=np.float64).ravel()
        self.b = float(b)
        self.mean = np.asarray(mean, dtype=np.float64).ravel()
        self.std = np.asarray(std, dtype=np.float64).ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        z = ((X - self.mean) / np.maximum(self.std, 1e-9)) @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, w=self.w, b=self.b, mean=self.mean,
                            std=self.std, names=np.array(AF_FEATURES))
        return path

    @classmethod
    def load(cls, path: Path) -> "RhythmModel":
        d = np.load(str(path), allow_pickle=False)
        return cls(d["w"], float(d["b"]), d["mean"], d["std"])


def _standardize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
    return (X - mean) / np.maximum(std, 1e-9), mean, std


def train(X: np.ndarray, y: np.ndarray, rec: np.ndarray,
          lr: float = 0.05, epochs: int = 300, l2: float = 1e-3,
          seed: int = 0) -> tuple[RhythmModel, dict]:
    rng = np.random.default_rng(seed)
    # partición por registro (inter-paciente)
    uniq = np.unique(rec)
    af_recs = [r for r in uniq if (r.startswith("04") or r.startswith("05")
                                   or r.startswith("06") or r.startswith("07")
                                   or r.startswith("08"))]
    rng.shuffle(af_recs)
    n_test = max(4, int(round(len(af_recs) * 0.25)))
    test_recs = set(af_recs[:n_test])
    # negativos de MIT-BIH: 25% a test
    mit_recs = [r for r in uniq if r not in af_recs]
    rng.shuffle(mit_recs)
    test_recs |= set(mit_recs[: max(3, len(mit_recs) // 4)])

    te = np.isin(rec, list(test_recs))
    tr = ~te

    Xtr, Ftr, _ = _standardize(X[tr])
    Xte, _, _ = _standardize(X[te], Ftr[0], Ftr[1])

    # pesos de clase
    pos = float(y[tr].sum())
    neg = float(len(y[tr]) - pos)
    w_pos = 0.5 * (pos + neg) / max(pos, 1)
    w_neg = 0.5 * (pos + neg) / max(neg, 1)

    w = np.zeros(X.shape[1])
    b = 0.0
    m_w = np.zeros_like(w); v_w = np.zeros_like(w)
    m_b = 0.0; v_b = 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    n = len(Xtr)

    for t in range(1, epochs + 1):
        z = Xtr @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        sw = np.where(y[tr] == 1, w_pos, w_neg)
        g = (p - y[tr]) * sw / n
        gw = Xtr.T @ g + l2 * w
        gb = g.sum()
        m_w = beta1 * m_w + (1 - beta1) * gw
        v_w = beta2 * v_w + (1 - beta2) * gw ** 2
        m_b = beta1 * m_b + (1 - beta1) * gb
        v_b = beta2 * v_b + (1 - beta2) * gb ** 2
        w -= lr * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)
        b -= lr * (m_b / (1 - beta1 ** t)) / (np.sqrt(v_b / (1 - beta2 ** t)) + eps)

    model = RhythmModel(w, b, Ftr[0], Ftr[1])
    metrics = {
        "train": _metrics(model.predict(X[tr]), y[tr]),
        "test": _metrics(model.predict(X[te]), y[te]),
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "test_records": sorted(test_recs),
    }
    return model, metrics


def _metrics(p: np.ndarray, y: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    se = tp / max(1, tp + fn)
    sp = tn / max(1, tn + fp)
    ppv = tp / max(1, tp + fp)
    # AUC
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    n1 = int(y.sum()); n0 = len(y) - n1
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0)
    return {
        "n": int(len(y)), "se": round(se, 4), "sp": round(sp, 4),
        "ppv": round(ppv, 4), "auc": round(float(auc), 4),
        "prev": round(float(y.mean()), 4),
    }


if __name__ == "__main__":
    d = np.load(ROOT / "data" / "cache" / "rhythm_af.npz", allow_pickle=False)
    X, y, rec = d["X"], d["y"], d["rec"]
    model, metrics = train(X, y, rec)
    model.save(MODELS / "rhythm_af.npz")
    (MODELS / "rhythm_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics["test"], indent=2))
    print("pesos:", dict(zip(AF_FEATURES, np.round(model.w, 3))))

