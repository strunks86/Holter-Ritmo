"""ECGFounder (fundación, 71 diagnósticos PTB-XL) sobre señales Holter.

El modelo espera 12 derivaciones × 10 s a 500 Hz. Un Holter trae 1-3
derivaciones, así que se colocan en sus posiciones anatómicas y el resto
se rellena con ceros, que es la práctica estándar de transferencia
12-derivaciones -> pocas derivaciones.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from holterai.legacy.app.signal.filters import fir_bandpass, filtfilt

ROOT = Path(__file__).resolve().parents[5]
MODEL_DIR = ROOT / "models" / "ecgfounder"
VENDOR = ROOT / "vendor"

FS_MODEL = 500.0
WIN_S = 10.0
N_SAMPLES = int(FS_MODEL * WIN_S)  # 5000

# posiciones anatómicas de una Holter de 2-3 canales dentro del estándar de 12
LEAD_INDEX = {
    "I": 0, "II": 1, "III": 2, "AVR": 3, "AVL": 4, "AVF": 5,
    "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11,
    "MLII": 1, "MLI": 0, "ECG1": 1, "ECG2": 6, "CH1": 1, "CH2": 6,
    "C1": 1, "C2": 6, "CANAL 1": 1, "CANAL 2": 6,
}


def _lead_position(name: str, fallback: int) -> int:
    key = name.strip().upper().replace(" ", "")
    if key in LEAD_INDEX:
        return LEAD_INDEX[key]
    for k, v in LEAD_INDEX.items():
        if key.startswith(k):
            return v
    return fallback


def _resample_to(x: np.ndarray, fs: float) -> np.ndarray:
    n_out = int(round(x.shape[0] * FS_MODEL / fs))
    if n_out <= 0:
        return np.zeros((0, x.shape[1]), np.float32)
    t_src = np.arange(x.shape[0]) / fs
    t_dst = np.arange(n_out) / FS_MODEL
    out = np.empty((n_out, x.shape[1]), dtype=np.float32)
    for j in range(x.shape[1]):
        out[:, j] = np.interp(t_dst, t_src, x[:, j]).astype(np.float32)
    return out


class ECGFounder:
    """Envoltorio de inferencia por lotes."""

    def __init__(self, task: str = "all71", model_dir: Path | None = None):
        self.task = task
        d = Path(model_dir or MODEL_DIR)
        fname = "ecg_founder_all71.onnx" if task == "all71" else "ecg_founder_12lead.onnx"
        self.model_path = d / fname
        labels_file = d / ("labels_all71.json" if task == "all71" else "labels_150.json")
        self.labels: list[str] = []
        if labels_file.exists():
            self.labels = json.loads(labels_file.read_text(encoding="utf-8"))["labels"]
        self.session = None
        self._filter = fir_bandpass(FS_MODEL, 0.5, 50.0, 501)

    # -- carga diferida (no bloquea el arranque de la app) ------------
    def load(self):
        if self.session is not None:
            return self
        from holterai.legacy.app.ml import providers
        from holterai.legacy.app.core import config
        import os
        pref = (os.environ.get("HOLTERIA_ACCEL")
                or str(config.load_settings().get("accel", "auto")))
        self.accel = providers.activate(pref)
        import onnxruntime as ort
        if not self.model_path.exists():
            raise RuntimeError(f"no se encuentra el modelo {self.model_path}")
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (int(__import__("os").cpu_count() or 4) - 1))
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        want = list(self.accel.get("providers") or ["CPUExecutionProvider"])
        try:
            self.session = ort.InferenceSession(
                str(self.model_path), sess_options=so, providers=want)
        except Exception as exc:
            log.warning("proveedor %s falló (%s); se usa CPU", want, exc)
            self.session = ort.InferenceSession(
                str(self.model_path), sess_options=so,
                providers=["CPUExecutionProvider"])
            self.accel = {"selected": "cpu", "providers": ["CPUExecutionProvider"],
                          "label": "CPU"}
        self.input_name = self.session.get_inputs()[0].name
        try:
            used = self.session.get_providers()
            self.accel["used"] = used
        except Exception:
            pass
        return self

    # -- preprocesado --------------------------------------------------
    def prepare(self, seg: np.ndarray, fs: float, leads: list[str] | None = None) -> np.ndarray:
        """(n, L) en mV -> (12, 5000) normalizado. Para una ventana suelta."""
        seg = np.asarray(seg, dtype=np.float64)
        if seg.ndim == 1:
            seg = seg[:, None]
        x = _resample_to(seg, fs)
        for j in range(x.shape[1]):
            x[:, j] = filtfilt(x[:, j], self._filter)
        return self._to_12(x, leads)

    def _to_12(self, x: np.ndarray, leads: list[str] | None = None) -> np.ndarray:
        """Coloca las derivaciones disponibles en el estándar de 12 y normaliza."""
        out = np.zeros((12, N_SAMPLES), dtype=np.float32)
        if x is None or len(x) == 0:
            return out
        names = leads or [f"Canal {i+1}" for i in range(x.shape[1])]
        used = set()
        for j in range(x.shape[1]):
            pos = _lead_position(names[j] if j < len(names) else "", j)
            while pos in used and pos < 11:
                pos += 1
            used.add(pos)
            col = x[:, j]
            n = min(N_SAMPLES, col.size)
            out[pos, :n] = col[:n]
        if x.shape[0] < N_SAMPLES:  # ventana incompleta: se repite el final
            out[:, x.shape[0]:] = out[:, x.shape[0] - 1:x.shape[0]]
        for k in range(12):
            if k in used:
                mu = float(out[k].mean())
                sd = float(out[k].std()) + 1e-6
                out[k] = (out[k] - mu) / sd
        return out

    def _prep_block(self, seg: np.ndarray, fs: float) -> np.ndarray:
        """Resamplea y filtra un bloque completo UNA sola vez.

        Es la diferencia clave: antes se filtrabá cada ventana de 10 s por
        separado (60 veces más trabajo de filtrado).
        """
        seg = np.asarray(seg, dtype=np.float64)
        if seg.ndim == 1:
            seg = seg[:, None]
        x = _resample_to(seg, fs)
        for j in range(x.shape[1]):
            x[:, j] = filtfilt(x[:, j], self._filter)
        return x

    # -- inferencia -----------------------------------------------------
    def predict_array(self, batch: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """batch (N,12,5000) -> probabilidades (N, 71).

        El lote es 32 por defecto: en CPU no cambia y en GPU (DirectML/CUDA)
        es el punto donde la aceleración compensa.
        """
        self.load()
        outs = []
        for i in range(0, len(batch), batch_size):
            chunk = np.ascontiguousarray(batch[i:i + batch_size], dtype=np.float32)
            logits = self.session.run(None, {self.input_name: chunk})[0]
            outs.append(1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30))))
        return np.concatenate(outs, axis=0) if outs else np.zeros((0, len(self.labels)))

    def predict_recording(self, samples: np.ndarray, fs: float, leads: list[str],
                          stride_s: float = 5.0, max_windows: int = 4000,
                          block_s: float = 600.0, progress=None) -> dict:
        """Ventanas deslizantes sobre todo el registro.

        El filtrado y el resampleo se hacen por bloques de ~10 min, no por
        ventana: es ~60 veces menos trabajo de filtrado.
        """
        n = samples.shape[0]
        win = int(round(WIN_S * fs))
        stride = max(1, int(round(stride_s * fs)))
        if n < win:
            segs_t = [0]
        else:
            segs_t = list(range(0, n - win + 1, stride))
            if not segs_t:
                segs_t = [0]
            if len(segs_t) > max_windows:
                step = len(segs_t) / max_windows
                segs_t = [segs_t[int(i * step)] for i in range(max_windows)]

        ratio = FS_MODEL / float(fs)
        block = max(1, int(block_s * fs))
        batch = np.empty((len(segs_t), 12, N_SAMPLES), dtype=np.float32)

        i = 0
        while i < len(segs_t):
            t_start = segs_t[i]
            j = i
            while j < len(segs_t) and (segs_t[j] - t_start) < block:
                j += 1
            t_end = min(n, segs_t[j - 1] + win)
            x = self._prep_block(samples[t_start:t_end], fs)
            for k in range(i, j):
                off = int(round((segs_t[k] - t_start) * ratio))
                seg500 = x[off:off + N_SAMPLES]
                if seg500.shape[0] < N_SAMPLES:
                    pad = N_SAMPLES - seg500.shape[0]
                    if seg500.shape[0] == 0:
                        seg500 = np.zeros((N_SAMPLES, x.shape[1]), dtype=np.float32)
                    else:
                        seg500 = np.vstack([seg500, np.repeat(seg500[-1:], pad, axis=0)])
                batch[k] = self._to_12(seg500, leads)
            if progress:
                progress(i, len(segs_t))
            i = j

        probs = self.predict_array(batch)
        return {
            "starts": np.asarray(segs_t, dtype=np.int64),
            "times": np.asarray(segs_t) / fs,
            "probs": probs,
            "labels": self.labels,
        }

    def top_labels(self, probs: np.ndarray, k: int = 6) -> list[tuple[str, float]]:
        idx = np.argsort(probs)[::-1][:k]
        out = []
        for i in idx:
            name = self.labels[i] if i < len(self.labels) else str(i)
            out.append((name, float(probs[i])))
        return out

