"""ECGFounder ajustado en T158 (3 canales; salidas FA, flutter, PVC, PAC).

Candidato ``full_s0`` de T158: Net1D de ECGFounder con cabeza de 4 salidas,
ajustado con bases públicas (PTB-XL, CPSC2018, AFDB, LTAFDB). Entrada igual que
en el entrenamiento (``scripts/t158_build.py``): ventanas de 10 s a 500 Hz,
remuestreo lineal, paso de banda FIR 0,5-50 Hz de fase cero y z-score por
derivación y ventana; ranuras 12-derivaciones II (1), V1 (6) y V5 (10). Las
derivaciones sin nombre conocido llenan las ranuras libres en orden (como en
AFDB/LTAFDB, cuyos 2 canales van a II y V1 y la ranura V5 queda a cero).

Inferencia en GPU si torch la ve; si no (o si falla), en CPU.
El archivo de pesos NO va en git: se busca en ``HOLTERIA_T158_PT``, luego en
``<repo>/models/ecgfounder/t158_full_s0.pt`` y en
``/root/holter/modelos/ecgfounder/t158_full_s0.pt``.
"""
from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path

import numpy as np

from holterai.legacy.app.signal.filters import filtfilt, fir_bandpass

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[5]
TARGETS = ("FA", "FLUT", "PVC", "PAC")
FS_MODEL = 500.0
WIN = 5000
WIN_S = 10.0
SLOTS_12 = (1, 6, 10)  # II, V1, V5
_SLOT_BY_NAME = {"II": 0, "MLII": 0, "V1": 1, "MLV1": 1, "V5": 2, "MLV5": 2}


def model_path() -> Path | None:
    cands = [os.environ.get("HOLTERIA_T158_PT", ""),
             str(ROOT / "models" / "ecgfounder" / "t158_full_s0.pt"),
             "/root/holter/modelos/ecgfounder/t158_full_s0.pt"]
    for c in cands:
        if c and Path(c).is_file():
            return Path(c)
    return None


def slot_map(leads: list[str], n_leads: int) -> list[tuple[int, int]]:
    """[(canal, ranura)] para hasta 3 canales: primero por nombre, luego en orden."""
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    rest: list[int] = []
    for j in range(min(n_leads, 3)):
        name = str(leads[j] if j < len(leads) else "").strip().upper().replace(" ", "")
        s = _SLOT_BY_NAME.get(name)
        if s is not None and s not in used:
            pairs.append((j, s))
            used.add(s)
        else:
            rest.append(j)
    free = [s for s in range(3) if s not in used]
    for j, s in zip(rest, free, strict=False):
        pairs.append((j, s))
    return sorted(pairs)


def _resample(x: np.ndarray, fs: float) -> np.ndarray:
    n_out = int(round(x.shape[0] * FS_MODEL / fs))
    t_src = np.arange(x.shape[0]) / fs
    t_dst = np.arange(n_out) / FS_MODEL
    out = np.empty((n_out, x.shape[1]), dtype=np.float64)
    for j in range(x.shape[1]):
        out[:, j] = np.interp(t_dst, t_src, x[:, j])
    return out


def window_batch(samples: np.ndarray, fs: float, leads: list[str],
                 starts_s: np.ndarray, block_s: float = 600.0) -> np.ndarray:
    """Ventanas (N, 3, 5000) float32 normalizadas, preparadas por bloques con margen."""
    pairs = slot_map(leads, samples.shape[1] if samples.ndim > 1 else 1)
    sig = np.asarray(samples, dtype=np.float32)
    if sig.ndim == 1:
        sig = sig[:, None]
    n = sig.shape[0]
    h = fir_bandpass(FS_MODEL, 0.5, 50.0, 501)
    out = np.zeros((len(starts_s), 3, WIN), dtype=np.float32)
    margin_s = 5.0
    i = 0
    while i < len(starts_s):
        b0 = float(starts_s[i])
        j = i
        while j < len(starts_s) and float(starts_s[j]) - b0 < block_s:
            j += 1
        a = max(0, int((b0 - margin_s) * fs))
        b = min(n, int((float(starts_s[j - 1]) + WIN_S + margin_s) * fs) + 1)
        raw = np.zeros((b - a, 3), dtype=np.float64)
        for ch, s in pairs:
            raw[:, s] = sig[a:b, ch]
        x = _resample(raw, fs)
        for _, s in pairs:
            x[:, s] = filtfilt(x[:, s], h)
        for k in range(i, j):
            off = int(round((float(starts_s[k]) - a / fs) * FS_MODEL))
            w = x[off:off + WIN]
            if w.shape[0] < WIN:
                w = np.vstack([w, np.zeros((WIN - w.shape[0], 3))]) if w.size else np.zeros((WIN, 3))
            for _, s in pairs:
                c = w[:, s]
                out[k, s] = ((c - c.mean()) / (c.std() + 1e-6)).astype(np.float32)
        i = j
    return out


class T158Model:
    """Envoltorio de inferencia del candidato T158 (torch, GPU con respaldo a CPU)."""

    def __init__(self, path: Path | None = None, device: str | None = None):
        self.path = Path(path) if path else model_path()
        self.device_pref = device or os.environ.get("HOLTERIA_T158_DEVICE", "auto")
        self.net = None
        self.device = "cpu"

    def load(self):
        if self.net is not None:
            return self
        if self.path is None or not self.path.is_file():
            raise RuntimeError("no se encuentra el modelo T158 (HOLTERIA_T158_PT)")
        import torch
        spec = importlib.util.spec_from_file_location(
            "holterai_net1d", Path(__file__).with_name("ecgfounder_src") / "net1d.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        net = mod.Net1D(in_channels=12, base_filters=64, ratio=1,
                        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
                        m_blocks_list=[2, 2, 2, 3, 3, 4, 4], kernel_size=16, stride=2,
                        groups_width=16, verbose=False, use_bn=False, use_do=False,
                        n_classes=4)
        ck = torch.load(str(self.path), map_location="cpu", weights_only=False)
        net.load_state_dict(ck["state_dict"])
        net.eval()
        dev = "cpu"
        if self.device_pref != "cpu":
            try:
                if torch.cuda.is_available():
                    net = net.to("cuda")
                    dev = "cuda"
            except Exception as exc:  # GPU llena u ocupada: se sigue en CPU
                log.warning("T158 en GPU falló (%s); se usa CPU", exc)
                net = net.to("cpu")
        self.net, self.device = net, dev
        return self

    def _run(self, xb: np.ndarray):
        import torch
        x = torch.zeros((xb.shape[0], 12, WIN), dtype=torch.float32)
        x[:, list(SLOTS_12), :] = torch.from_numpy(xb)
        if self.device == "cuda":
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                return self.net(x.to("cuda")).float().cpu().numpy()
        with torch.no_grad():
            return self.net(x).numpy()

    def predict(self, batch: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """(N, 3, 5000) -> probabilidades (N, 4) en el orden de ``TARGETS``."""
        self.load()
        outs = []
        for i in range(0, len(batch), batch_size):
            xb = np.ascontiguousarray(batch[i:i + batch_size], dtype=np.float32)
            try:
                lg = self._run(xb)
            except RuntimeError as exc:
                if self.device != "cuda":
                    raise
                log.warning("T158: GPU falló en inferencia (%s); se pasa a CPU", exc)
                import torch
                self.net = self.net.to("cpu")
                self.device = "cpu"
                torch.cuda.empty_cache()
                lg = self._run(xb)
            outs.append(1.0 / (1.0 + np.exp(-np.clip(lg, -30, 30))))
        return np.concatenate(outs, 0) if outs else np.zeros((0, 4), np.float32)

    def predict_recording(self, samples: np.ndarray, fs: float, leads: list[str],
                          stride_s: float = 5.0, block_s: float = 600.0) -> dict:
        """Ventanas deslizantes de 10 s: {'starts_s', 'probs' (N,4), 'device'}."""
        dur = samples.shape[0] / float(fs)
        last = max(0.0, dur - WIN_S)
        starts = np.arange(0.0, last + 1e-9, stride_s)
        if starts.size == 0:
            starts = np.zeros(1)
        probs = np.zeros((0, 4), np.float32)
        parts = []
        step = max(1, int(block_s / stride_s))
        for i in range(0, len(starts), step):
            st = starts[i:i + step]
            parts.append(self.predict(window_batch(samples, fs, leads, st, block_s)))
        if parts:
            probs = np.concatenate(parts, 0).astype(np.float32)
        return {"starts_s": starts, "probs": probs, "device": self.device}


def beat_probs(beat_t: np.ndarray, starts_s: np.ndarray, probs: np.ndarray,
               reduce: str = "max") -> np.ndarray:
    """Probabilidad por latido a partir de las ventanas de 10 s que lo contienen.

    ``reduce`` = max o mean sobre las ventanas que cubren el latido (con un
    margen de 0,5 s en los bordes). Sin ventana que lo cubra: la más cercana.
    """
    t = np.asarray(beat_t, dtype=np.float64)
    out = np.zeros((t.size, probs.shape[1]), dtype=np.float32)
    if t.size == 0 or len(starts_s) == 0:
        return out
    lo = np.searchsorted(starts_s, t - WIN_S + 0.5, side="left")
    hi = np.searchsorted(starts_s, t - 0.5, side="right")
    near = np.clip(np.searchsorted(starts_s, t - WIN_S / 2), 0, len(starts_s) - 1)
    for i in range(t.size):
        a, b = int(lo[i]), int(hi[i])
        if b <= a:
            out[i] = probs[near[i]]
        elif reduce == "mean":
            out[i] = probs[a:b].mean(axis=0)
        else:
            out[i] = probs[a:b].max(axis=0)
    return out
