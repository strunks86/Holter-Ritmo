"""Red neuronal convolucional 1-D implementada en numpy puro.

Sin PyTorch/TensorFlow: se incluye el modelo ya entrenado en models/,
y aquí viven el forward, el backward y el guardado/carga de pesos.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CLASSES = ["N", "S", "V", "F", "Q"]
N_CLASSES = len(CLASSES)


# ---------------------------------------------------------------------------
# Primitivas
# ---------------------------------------------------------------------------
def im2col_1d(x: np.ndarray, k: int, stride: int):
    """(B,C,T) -> (B,T_out,C*k) + índices base."""
    B, C, T = x.shape
    T_out = (T - k) // stride + 1
    if T_out <= 0:
        raise ValueError("ventana demasiado corta para el kernel")
    i0 = np.arange(T_out) * stride
    idx = i0[:, None] + np.arange(k)[None, :]
    patches = x[:, :, idx]  # (B, C, T_out, k)
    return patches.transpose(0, 2, 1, 3).reshape(B, T_out, C * k), i0


def conv_forward(x, W, b, stride):
    cols, i0 = im2col_1d(x, W.shape[1] // x.shape[1], stride)
    y = cols @ W.T + b  # (B,T_out,F)
    return y.transpose(0, 2, 1), cols, i0


def conv_backward(dy, cols, W, x_shape, stride, i0):
    B, C, T = x_shape
    F, Ck = W.shape
    k = Ck // C
    T_out = dy.shape[2]
    dy2 = dy.transpose(0, 2, 1).reshape(-1, F)      # (B*T_out, F)
    dW = dy2.T @ cols.reshape(-1, Ck)               # (F, Ck)
    db = dy2.sum(axis=0)
    dcols = (dy2 @ W).reshape(B, T_out, C, k).transpose(0, 2, 1, 3)  # (B,C,T_out,k)
    dx = np.zeros((B, C, T), dtype=dy.dtype)
    for j in range(k):
        dx[:, :, i0 + j] += dcols[:, :, :, j]
    return dx, dW, db


def pool_forward(y):
    """Pool global: media y máximo por canal."""
    m = y.mean(axis=2)
    mx = y.max(axis=2)
    arg = y.argmax(axis=2)
    return np.concatenate([m, mx], axis=1), (m, mx, arg)


def pool_backward(dg, cache, shape):
    m, mx, arg = cache
    B, F, T = shape
    F2 = dg.shape[1] // 2
    dmean = dg[:, :F2]
    dmax = dg[:, F2:]
    dy = np.zeros((B, F, T), dtype=dg.dtype)
    np.put_along_axis(dy, arg[:, :, None], dmax[:, :, None], axis=2)
    dy += dmean[:, :, None] / T
    return dy


def initialize(rng: np.random.Generator) -> dict:
    def he(shape, fan_in):
        return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)

    p = {
        "W1": he((8, 1 * 15), 15), "b1": np.zeros(8, np.float32),
        "W2": he((16, 8 * 9), 8 * 9), "b2": np.zeros(16, np.float32),
        "W3": he((24, 16 * 7), 16 * 7), "b3": np.zeros(24, np.float32),
        "W4": he((32, 48 + 5), 53), "b4": np.zeros(32, np.float32),
        "W5": he((N_CLASSES, 32), 32), "b5": np.zeros(N_CLASSES, np.float32),
    }
    return p


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
class BeatNet:
    """CNN de 3 capas + rama de características clínicas."""

    def __init__(self, params: dict | None = None, seed: int = 0):
        self.p = params if params is not None else initialize(np.random.default_rng(seed))

    # -- forward ------------------------------------------------------
    def forward(self, x: np.ndarray, f: np.ndarray, cache: dict | None = None):
        p = self.p
        x3 = x[:, None, :].astype(np.float32) if x.ndim == 2 else x.astype(np.float32)
        h1, c1, i1 = conv_forward(x3, p["W1"], p["b1"], 2)
        r1 = np.maximum(h1, 0)
        h2, c2, i2 = conv_forward(r1, p["W2"], p["b2"], 2)
        r2 = np.maximum(h2, 0)
        h3, c3, i3 = conv_forward(r2, p["W3"], p["b3"], 2)
        r3 = np.maximum(h3, 0)
        g, pc = pool_forward(r3)
        gf = np.concatenate([g, np.asarray(f, np.float32)], axis=1)
        a1 = gf @ p["W4"].T + p["b4"]
        r4 = np.maximum(a1, 0)
        logits = r4 @ p["W5"].T + p["b5"]

        if cache is not None:
            cache.update({
                "x3": x3, "c1": c1, "i1": i1, "r1": r1,
                "c2": c2, "i2": i2, "r2": r2,
                "c3": c3, "i3": i3, "r3": r3, "pc": pc,
                "gf": gf, "a1": a1, "r4": r4, "f": np.asarray(f, np.float32),
            })
        return logits

    # -- backward -----------------------------------------------------
    def backward(self, dlogits: np.ndarray, cache: dict) -> dict:
        p = self.p
        g = {}
        g["W5"] = dlogits.T @ cache["r4"]
        g["b5"] = dlogits.sum(axis=0)
        dr4 = dlogits @ p["W5"]
        da1 = dr4 * (cache["a1"] > 0)
        g["W4"] = da1.T @ cache["gf"]
        g["b4"] = da1.sum(axis=0)
        dgf = da1 @ p["W4"]
        dg = dgf[:, :48]
        dr3 = pool_backward(dg, cache["pc"], cache["r3"].shape)

        # conv3
        dh3 = dr3 * (cache["r3"] > 0)
        dr2, g["W3"], g["b3"] = conv_backward(
            dh3, cache["c3"], p["W3"], cache["r2"].shape, 2, cache["i3"])
        dh2 = dr2 * (cache["r2"] > 0)
        dr1, g["W2"], g["b2"] = conv_backward(
            dh2, cache["c2"], p["W2"], cache["r1"].shape, 2, cache["i2"])
        dh1 = dr1 * (cache["r1"] > 0)
        dx, g["W1"], g["b1"] = conv_backward(
            dh1, cache["c1"], p["W1"], cache["x3"].shape, 2, cache["i1"])
        return g

    # -- utilidades ---------------------------------------------------
    def predict_proba(self, x: np.ndarray, f: np.ndarray, batch: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(x), batch):
            logits = self.forward(x[i:i + batch], f[i:i + batch])
            out.append(softmax(logits))
        return np.concatenate(out, axis=0) if out else np.zeros((0, N_CLASSES))

    def save(self, path: str | Path, meta: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in self.p.items()}
        payload["__classes__"] = np.array(CLASSES)
        payload["__meta__"] = np.array([str(meta or {})])
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BeatNet":
        d = np.load(str(path), allow_pickle=False)
        params = {k: d[k] for k in d.files if not k.startswith("__")}
        return cls(params=params)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
    n = len(y)
    ll = -np.log(np.clip(probs[np.arange(n), y], 1e-9, 1.0))
    if w is not None:
        return float((ll * w).sum() / w.sum())
    return float(ll.mean())


def grad_logits(probs: np.ndarray, y: np.ndarray,
                w: np.ndarray | None = None) -> np.ndarray:
    n = len(y)
    d = probs.copy()
    d[np.arange(n), y] -= 1.0
    if w is not None:
        d = d * (w[:, None] / w.sum())
    else:
        d = d / n
    return d.astype(np.float32)


# ---------------------------------------------------------------------------
# Verificación de gradientes (se ejecuta en las pruebas)
# ---------------------------------------------------------------------------
def gradient_check(seed: int = 0, eps: float = 1e-6) -> float:
    rng = np.random.default_rng(seed)
    net = BeatNet(seed=seed)
    net.p = {k: v.astype(np.float64) for k, v in net.p.items()}
    x = rng.standard_normal((3, 256))
    f = rng.standard_normal((3, 5))
    y = np.array([0, 2, 3])
    cache: dict = {}
    logits = net.forward(x, f, cache)
    grads = net.backward(grad_logits(softmax(logits), y), cache)

    worst = 0.0
    for name in ("W1", "W2", "W3", "W4", "W5", "b1", "b2", "b3", "b4", "b5"):
        arr = net.p[name]
        idxs = [tuple(rng.integers(0, s) for s in arr.shape) for _ in range(4)]
        for idx in idxs:
            old = arr[idx]
            arr[idx] = old + eps
            lp = cross_entropy(softmax(net.forward(x, f)), y)
            arr[idx] = old - eps
            lm = cross_entropy(softmax(net.forward(x, f)), y)
            arr[idx] = old
            num = (lp - lm) / (2 * eps)
            ana = float(grads[name][idx])
            denom = abs(num) + abs(ana)
            if denom < 1e-8:
                continue
            worst = max(worst, abs(num - ana) / denom)
    return worst


class Adam:
    def __init__(self, params: dict, lr: float = 1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay: float = 1e-4):
        self.lr = lr
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.wd = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict):
        self.t += 1
        for k in params:
            g = grads[k]
            if k.startswith("W"):
                g = g + self.wd * params[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)

