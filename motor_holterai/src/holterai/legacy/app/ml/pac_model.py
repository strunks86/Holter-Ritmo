"""Clasificador específico de PAC (S vs N) sobre características de ritmo y P.

Es un modelo aparte del CNN de latidos: el CNN aporta morfología, este
aporta el contexto (prematuridad, pausa, onda P) que distingue una PAC.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from holterai.legacy.app.signal.pac import PAC_FEATURES

ROOT = Path(__file__).resolve().parents[5]
MODELS = ROOT / "models" / "pretrained"


class PACModel:
    """Regresión logística + estandarización, en numpy."""

    def __init__(self, w, b, mean, std):
        self.w = np.asarray(w, dtype=np.float64)
        self.b = float(b)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        z = ((X - self.mean) / np.maximum(self.std, 1e-9)) @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, w=self.w, b=self.b, mean=self.mean,
                            std=self.std, names=np.array(PAC_FEATURES))
        return path

    @classmethod
    def load(cls, path: Path) -> "PACModel":
        d = np.load(str(path), allow_pickle=False)
        return cls(d["w"], float(d["b"]), d["mean"], d["std"])


def _fit(X, y, lr=0.05, epochs=400, l2=1e-3, pos_weight=None, seed=0,
         sample_weight=None):
    rng = np.random.default_rng(seed)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    Z = (X - mean) / np.maximum(std, 1e-9)
    n, d = Z.shape
    w = np.zeros(d)
    b = float(np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6)))
    m_w = np.zeros(d); v_w = np.zeros(d)
    m_b = v_b = 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    sw = np.where(y == 1, pos_weight or 1.0, 1.0)
    if sample_weight is not None:
        sw = sw * np.asarray(sample_weight, dtype=float)
    sw = sw / sw.mean()
    for t in range(1, epochs + 1):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        g = (p - y) * sw / n
        gw = Z.T @ g + l2 * w
        gb = g.sum()
        m_w = beta1 * m_w + (1 - beta1) * gw
        v_w = beta2 * v_w + (1 - beta2) * gw ** 2
        m_b = beta1 * m_b + (1 - beta1) * gb
        v_b = beta2 * v_b + (1 - beta2) * gb ** 2
        w -= lr * (m_w / (1 - beta1 ** t)) / (np.sqrt(v_w / (1 - beta2 ** t)) + eps)
        b -= lr * (m_b / (1 - beta1 ** t)) / (np.sqrt(v_b / (1 - beta2 ** t)) + eps)
    return PACModel(w, b, mean, std)


def _metrics(p, y, thr):
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    se = tp / max(1, tp + fn)
    ppv = tp / max(1, tp + fp)
    f1 = 2 * se * ppv / max(1e-9, se + ppv)
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    n1, n0 = int(y.sum()), len(y) - int(y.sum())
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0)
    return {"se": round(se, 3), "ppv": round(ppv, 3), "f1": round(f1, 3),
            "auc": round(float(auc), 3), "n": int(len(y)), "n_pos": n1,
            "tp": tp, "fp": fp, "fn": fn}


def _best_threshold(p, y):
    best = (0.5, -1.0)
    for thr in np.linspace(0.1, 0.9, 33):
        m = _metrics(p, y, thr)
        if m["f1"] > best[1]:
            best = (float(thr), m["f1"])
    return best[0]


def train(npz: Path, seed: int = 0) -> tuple[PACModel, dict]:
    d = np.load(npz, allow_pickle=False)
    X = d["X"].astype(np.float64)
    y = d["y"].astype(int)
    rec = d["rec"].astype(str)
    uniq = np.unique(rec)

    rng = np.random.default_rng(seed)
    # partición por registro y por base de datos (inter-paciente)
    def is_svdb(r):
        return r.startswith("svdb")

    mit = [r for r in uniq if not is_svdb(r)]
    sv = [r for r in uniq if is_svdb(r)]
    rng.shuffle(mit); rng.shuffle(sv)
    test_recs = set(mit[: max(4, len(mit) // 3)]) | set(sv[: max(8, len(sv) // 4)])
    test = np.isin(rec, list(test_recs))
    tr = ~test

    pos = float(y[tr].sum())
    neg = float(len(y[tr]) - pos)
    # cada base de datos pesa lo mismo, para que svdb (183 registros) no
    # aplaste a MIT-BIH (44)
    rc = rec[tr]
    w_db = np.where(np.char.startswith(rc, "svdb"),
                    1.0 / max(1, np.sum(np.char.startswith(rc, "svdb"))),
                    1.0 / max(1, np.sum(~np.char.startswith(rc, "svdb"))))
    model = _fit(X[tr], y[tr], pos_weight=neg / max(1.0, pos), sample_weight=w_db * len(rc))

    p_tr = model.predict(X[tr])
    thr = _best_threshold(p_tr, y[tr])
    metrics = {
        "threshold": thr,
        "train": _metrics(p_tr, y[tr], thr),
        "test": _metrics(model.predict(X[test]), y[test], thr),
        "test_records": sorted(test_recs),
        "n_train": int(tr.sum()), "n_test": int(test.sum()),
    }
    return model, metrics


if __name__ == "__main__":
    npz = ROOT / "data" / "cache" / "pac_dataset.npz"
    model, metrics = train(npz)
    model.save(MODELS / "pac_model.npz")
    (MODELS / "pac_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("\npesos:")
    for n, w in sorted(zip(PAC_FEATURES, model.w), key=lambda kv: -abs(kv[1])):
        print(f"  {n:12s} {w:+.3f}")

