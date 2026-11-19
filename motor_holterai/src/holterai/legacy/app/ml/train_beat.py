"""Entrenamiento del clasificador de latidos (CNN numpy).

Uso:  python -m app.ml.train_beat
Guarda models/beat_cnn.npz y models/beat_metrics.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from holterai.legacy.app.ml.dataset_mitdb import CLASSES, DS1, DS2
from holterai.legacy.app.ml.net import Adam, BeatNet, cross_entropy, grad_logits, softmax

ROOT = Path(__file__).resolve().parents[5]
MODELS = ROOT / "models" / "pretrained"


def augment_batch(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    B = X.shape[0]
    scale = 1.0 + rng.normal(0, 0.08, (B, 1)).astype(np.float32)
    shift = int(rng.integers(-3, 4))
    noise = rng.normal(0, 0.02, X.shape).astype(np.float32)
    out = X * scale + noise
    if shift:
        out = np.roll(out, shift, axis=1)
    # invariancia de polaridad: muchas derivaciones Holter invierten el QRS
    flip = rng.random(B) < 0.5
    out[flip] = -out[flip]
    return out


def evaluate(net: BeatNet, X: np.ndarray, F: np.ndarray,
             y: np.ndarray) -> dict:
    probs = net.predict_proba(X, F, batch=512)
    pred = probs.argmax(axis=1)
    K = len(CLASSES)
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y, pred):
        cm[int(t), int(p)] += 1
    per_class = {}
    f1s = []
    for k, c in enumerate(CLASSES):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        se = tp / (tp + fn) if (tp + fn) else 0.0
        ppv = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * se * ppv / (se + ppv) if (se + ppv) else 0.0
        per_class[c] = {
            "n": int(cm[k, :].sum()), "se": round(float(se), 4),
            "ppv": round(float(ppv), 4), "f1": round(float(f1), 4),
        }
        if cm[k, :].sum() > 0:
            f1s.append(f1)
    acc = float(np.trace(cm) / max(1, cm.sum()))
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 4),
        "per_class": per_class,
        "confusion": cm.tolist(),
        "classes": CLASSES,
    }


def train(npz: Path, epochs: int = 14, batch: int = 128, lr: float = 2e-3,
          seed: int = 42) -> dict:
    d = np.load(npz, allow_pickle=False)
    X = d["X"].astype(np.float32)
    F = d["F"].astype(np.float32)
    y = d["y"].astype(np.int64)
    rec = d["rec"]

    # Los latidos de fusión son muy raros y contaminan el modelo: se agrupan
    # con los ventriculares (criterio habitual en análisis de Holter).
    y = np.where(y == CLASSES.index("F"), CLASSES.index("V"), y)

    tr = np.isin(rec, list(DS1)) | (rec == "synthetic")
    va = np.isin(rec, list(DS2))
    print(f"entrenamiento {tr.sum()} latidos, validación {va.sum()} latidos")

    counts = np.bincount(y[tr], minlength=len(CLASSES)).astype(np.float64)
    w = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)
    w = w / w.mean()
    # refuerzo de la clase minoritaria difícil (S)
    w[CLASSES.index("S")] *= 2.5
    w[CLASSES.index("Q")] *= 0.5
    w = w / w.mean()
    print("pesos de clase:", dict(zip(CLASSES, np.round(w, 3))))

    rng = np.random.default_rng(seed)
    net = BeatNet(seed=seed)
    opt = Adam(net.p, lr=lr, weight_decay=2e-4)
    idx_tr = np.flatnonzero(tr)
    idx_s = idx_tr[y[idx_tr] == CLASSES.index("S")]
    if idx_s.size:
        idx_tr = np.concatenate([idx_tr] + [idx_s] * 2)
    best = {"macro_f1": -1.0, "epoch": 0, "params": None}

    t0 = time.time()
    for ep in range(1, epochs + 1):
        rng.shuffle(idx_tr)
        total = 0.0
        nb = 0
        for s in range(0, len(idx_tr), batch):
            b = idx_tr[s:s + batch]
            if len(b) < 4:
                continue
            xb = augment_batch(X[b], rng)
            fb = F[b]
            yb = y[b]
            cache: dict = {}
            logits = net.forward(xb, fb, cache)
            probs = softmax(logits)
            loss = cross_entropy(probs, yb, w[yb])
            grads = net.backward(grad_logits(probs, yb, w[yb]), cache)
            opt.step(net.p, grads)
            total += loss
            nb += 1
        msg = f"época {ep:2d} loss={total / max(1, nb):.4f}"
        if ep % 2 == 0 or ep == epochs:
            m = evaluate(net, X[va], F[va], y[va])
            msg += f" | val acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f}"
            if m["macro_f1"] > best["macro_f1"]:
                best = {"macro_f1": m["macro_f1"], "epoch": ep,
                        "params": {k: v.copy() for k, v in net.p.items()}}
            for c in CLASSES:
                pc = m["per_class"][c]
                if pc["n"]:
                    msg += f" {c}:Se={pc['se']:.2f}/PPV={pc['ppv']:.2f}"
        print(msg, flush=True)

    if best["params"] is not None:
        net.p = best["params"]
    metrics = evaluate(net, X[va], F[va], y[va])
    metrics["train"] = evaluate(net, X[tr], F[tr], y[tr])
    metrics["dataset"] = str(npz.name)
    metrics["epochs"] = epochs
    metrics["seconds"] = round(time.time() - t0, 1)
    metrics["best_epoch"] = best["epoch"]
    metrics["best_macro_f1"] = best["macro_f1"]

    MODELS.mkdir(parents=True, exist_ok=True)
    # F nunca se predice (no se entrena): se fuerza su logit a -10
    net.p["b5"][CLASSES.index("F")] = -10.0
    net.save(MODELS / "beat_cnn.npz", meta={"classes": CLASSES})
    (MODELS / "beat_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"guardado {MODELS/'beat_cnn.npz'} ({metrics['seconds']}s)")
    print(json.dumps(metrics["per_class"], indent=2, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    train(ROOT / "data" / "cache" / "mitdb_beats.npz")


