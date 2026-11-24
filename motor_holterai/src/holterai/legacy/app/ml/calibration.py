"""Calibración del clasificador de latidos: temperatura, prevalencia y Q.

El clasificador (`beat_cnn`) se entrenó con pesos por clase (S reforzada ×2,5)
y se usa sin corregir la prevalencia, así que sus probabilidades están sesgadas
hacia S y V: el inventario lo ve como exceso de ectópicos y como una clase Q
que nunca se activa porque las confianzas son altas. Este módulo aplica las
tres correcciones del protocolo, en este orden:

1. **Escalado de temperatura** (Guo et al. 2017): un único parámetro `T` que
   minimiza la NLL en un conjunto de validación separado por paciente
   (MIT-BIH DS1). No cambia el argmax, solo la confianza.
2. **Corrección de prevalencia por logits** (Saerens et al. 2002): con la
   prevalencia de entrenamiento `p_from` y la prevalencia objetivo `p_target`,
   `p_new ∝ p · p_target / p_from` — equivalente a sumar
   `log(p_target) − log(p_from)` a los logits. También está el EM de Saerens,
   que estima la prevalencia de cada examen sin etiquetas.
3. **Abstención (clase Q)**: además de la probabilidad máxima baja, se marca Q
   cuando el margen entre las dos clases primeras es < 0,1, cuando el latido
   cae en ruido de alta frecuencia, o cuando el ancho y la correlación con la
   plantilla normal no son compatibles con la clase propuesta (S ancha, V
   estrecha). Q no se cuenta como S/V: es "no clasificable".

Todo es numpy puro: el pipeline legado no depende de scipy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from holterai.legacy.app.ml.net import CLASSES, N_CLASSES

SCHEMA = "holterai/beat_calibration/1"
TEMPERATURE_BOUNDS = (0.5, 10.0)

#: Prevalencia objetivo en un Holter de población general: N 98,5 %, S 0,8 %,
#: V 0,7 %. F no se predice con este modelo (su logit quedó fijado en -10), así
#: que se deja igual que en el entrenamiento en vez de inventarle prevalencia;
#: Q es la abstención y se mantiene muy baja.
TARGET_PRIOR = (0.985, 0.008, 0.007, 1e-6, 5e-4)

#: Prevalencia con la que se entrenó `beat_cnn`: MIT-BIH DS1 (50.577 latidos,
#: F agrupada con V) más los 4.000 latidos Q sintéticos. F nunca se etiquetó
#: (su logit quedó fijado en -10), así que lleva solo un suelo numérico.
DEFAULT_TRAIN_PRIOR = (0.83253, 0.017297, 0.076882, 1e-6, 0.073291)

PRIOR_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Temperatura
# ---------------------------------------------------------------------------
def log_softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def nll(logits: np.ndarray, y: np.ndarray, temperature: float = 1.0) -> float:
    """Entropía cruzada media de `softmax(logits / T)`."""
    y = np.asarray(y, dtype=np.int64)
    if y.size == 0:
        return 0.0
    ls = log_softmax(np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6))
    return float(-ls[np.arange(y.size), y].mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray,
                    bounds: tuple[float, float] = TEMPERATURE_BOUNDS,
                    tol: float = 1e-4) -> float:
    """T que minimiza la NLL por búsqueda dorada (sin scipy).

    La NLL de `softmax(z/T)` es unimodal en `T` para un conjunto fijo de
    logits, así que la búsqueda dorada encuentra el óptimo global.
    """
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if logits.shape[0] == 0 or np.unique(y).size < 2:
        return 1.0
    a, b = float(bounds[0]), float(bounds[1])
    gr = (5.0 ** 0.5 - 1.0) / 2.0
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = nll(logits, y, c), nll(logits, y, d)
    for _ in range(100):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = nll(logits, y, c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = nll(logits, y, d)
        if b - a <= tol:
            break
    return round(float(0.5 * (a + b)), 4)


def scale_probs(probs: np.ndarray, temperature: float) -> np.ndarray:
    """`softmax(log(p) / T)` sobre probabilidades ya normalizadas."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, None)
    t = max(float(temperature), 1e-6)
    if abs(t - 1.0) < 1e-9:
        out = p
    else:
        out = p ** (1.0 / t)
    return out / out.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Prevalencia
# ---------------------------------------------------------------------------
def empirical_prior(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)[:n_classes]
    if counts.sum() == 0:
        return np.full(n_classes, 1.0 / n_classes)
    return counts / counts.sum()


def adjust_prior(probs: np.ndarray, p_from: np.ndarray,
                 p_target: np.ndarray, floor: float = PRIOR_FLOOR) -> np.ndarray:
    """Corrige la prevalencia con `p · p_target / p_from` (equivale a logits)."""
    p = np.asarray(probs, dtype=np.float64)
    src = np.maximum(np.asarray(p_from, dtype=np.float64).ravel(), floor)
    dst = np.maximum(np.asarray(p_target, dtype=np.float64).ravel(), 0.0)
    if src.size != p.shape[1] or dst.size != p.shape[1]:
        raise ValueError("p_from y p_target deben tener una entrada por clase")
    if dst.sum() <= 0:
        return p
    out = p * (dst / src)
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def prior_em(probs: np.ndarray, p_train: np.ndarray, iters: int = 50,
             tol: float = 1e-5, damping: float = 0.5, max_ratio: float = 10.0,
             freeze: tuple[int, ...] = (3, 4)) -> tuple[np.ndarray, np.ndarray]:
    """EM de Saerens: prevalencia del examen y probabilidades reajustadas.

    Devuelve `(Q, p)` con `Q` las probabilidades corregidas y `p` la
    prevalencia estimada del examen. Sin etiquetas la escala solo puede
    apoyarse en `p_train`, así que el resultado depende de ella.

    El EM clásico se realimenta con su propia estimación y, con un
    clasificador miscalibrado como `beat_cnn` (sesgo hacia Q/S), puede divergir
    hacia una clase: por eso cada paso se amortigua (`damping`), la razón
    `p / p_train` se limita a `max_ratio` (equivale a no mover ningún logit más
    de `log(max_ratio)`) y las clases de `freeze` no se corrigen: F no se
    entrena y Q es la abstención, así que no tienen prevalencia poblacional que
    estimar. Con `damping=1`, `max_ratio=inf` y `freeze=()` se recupera el EM
    original del protocolo.
    """
    src = np.maximum(np.asarray(p_train, dtype=np.float64).ravel(), PRIOR_FLOOR)
    src = src / src.sum()
    P = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, None)
    if P.ndim != 2 or P.shape[1] != src.size:
        raise ValueError("probs debe ser (n_latidos, n_clases)")
    lo = 1.0 / max(float(max_ratio), 1.0)
    hi = max(float(max_ratio), 1.0)
    alpha = float(np.clip(damping, 1e-3, 1.0))
    fixed = np.zeros(src.size, dtype=bool)
    for k in freeze:
        if 0 <= int(k) < src.size:
            fixed[int(k)] = True
    p = src.copy()
    Q = P / P.sum(axis=1, keepdims=True)
    for _ in range(max(1, int(iters))):
        ratio = np.ones(src.size)
        ratio[~fixed] = np.clip(p[~fixed] / src[~fixed], lo, hi)
        Q = P * ratio
        Q /= np.maximum(Q.sum(axis=1, keepdims=True), 1e-12)
        p_new = np.maximum(Q.mean(axis=0), PRIOR_FLOOR)
        p_new /= p_new.sum()
        delta = float(np.abs(p_new - p).max())
        p = (1.0 - alpha) * p + alpha * p_new
        p /= p.sum()
        if delta < float(tol):
            break
    ratio = np.ones(src.size)
    ratio[~fixed] = np.clip(p[~fixed] / src[~fixed], lo, hi)
    Q = P * ratio
    Q /= np.maximum(Q.sum(axis=1, keepdims=True), 1e-12)
    return Q, p


# ---------------------------------------------------------------------------
# Abstención (Q)
# ---------------------------------------------------------------------------
def abstain_triggers(probs: np.ndarray, cls: np.ndarray | None = None,
                     noise: np.ndarray | None = None,
                     widths: np.ndarray | None = None,
                     corr: np.ndarray | None = None, *,
                     max_prob: float = 0.45, margin: float = 0.10,
                     noise_thr: float = 0.20, corr_low: float = 0.85,
                     narrow: float = 0.08, wide: float = 0.12) -> dict[str, np.ndarray]:
    """Máscara de abstención desglosada por disparador (para informes/pruebas).

    Se mira solo la ectopia propuesta: una N dudosa se deja como N (no se
    pierde el latido y no se inventa una extrasístole). Disparadores:

    - `confianza`: probabilidad máxima < `max_prob`;
    - `margen`: diferencia entre las dos clases primeras < `margin`;
    - `ruido`: ruido de alta frecuencia por latido ≥ `noise_thr` (calidad 2);
    - `morfologia`: S con QRS ancho y poca correlación con la plantilla
      normal, o V con QRS estrecho y correlación casi perfecta.
    """
    P = np.asarray(probs, dtype=np.float64)
    n = P.shape[0] if P.ndim == 2 else 0
    empty = np.zeros(n, dtype=bool)
    out = {"confianza": empty.copy(), "margen": empty.copy(),
           "ruido": empty.copy(), "morfologia": empty.copy()}
    if n == 0:
        return out
    c = P.argmax(axis=1) if cls is None else np.asarray(cls, dtype=np.int64)
    srt = np.sort(P, axis=1)
    conf = srt[:, -1]
    gap = srt[:, -1] - srt[:, -2]
    ect = np.isin(c, (1, 2, 3))
    out["confianza"] = ect & (conf < float(max_prob))
    out["margen"] = ect & (gap < float(margin))
    if noise is not None:
        out["ruido"] = ect & (np.asarray(noise, dtype=np.float64) >= float(noise_thr))
    if widths is not None and corr is not None:
        w = np.asarray(widths, dtype=np.float64)
        r = np.asarray(corr, dtype=np.float64)
        out["morfologia"] = ((c == 1) & (w >= float(wide)) & (r > 0) & (r < float(corr_low))) | (
            (c == 2) & (w < float(narrow)) & (r >= 0.94))
    return out


def abstain_mask(probs: np.ndarray, cls: np.ndarray | None = None,
                 noise: np.ndarray | None = None,
                 widths: np.ndarray | None = None,
                 corr: np.ndarray | None = None, *,
                 max_prob: float = 0.45, margin: float = 0.10,
                 noise_thr: float = 0.20, corr_low: float = 0.85,
                 narrow: float = 0.08, wide: float = 0.12) -> np.ndarray:
    """OR de todos los disparadores de `abstain_triggers`."""
    trig = abstain_triggers(probs, cls=cls, noise=noise, widths=widths, corr=corr,
                            max_prob=max_prob, margin=margin, noise_thr=noise_thr,
                            corr_low=corr_low, narrow=narrow, wide=wide)
    mask = np.zeros(0, dtype=bool)
    for m in trig.values():
        mask = m.copy() if mask.size == 0 else (mask | m)
    return mask


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Calibration:
    """Parámetros de calibración del clasificador de latidos."""

    temperature: float = 1.0
    train_prior: tuple[float, ...] = DEFAULT_TRAIN_PRIOR
    target_prior: tuple[float, ...] = TARGET_PRIOR
    max_prob: float = 0.45
    margin: float = 0.10
    noise_thr: float = 0.20
    enabled: bool = True

    def apply(self, probs: np.ndarray) -> np.ndarray:
        """Temperatura + corrección de prevalencia, en ese orden."""
        p = np.asarray(probs, dtype=np.float64)
        if not self.enabled or p.size == 0:
            return p
        return adjust_prior(scale_probs(p, self.temperature),
                            self.train_prior, self.target_prior)

    def abstain_triggers(self, probs: np.ndarray, cls: np.ndarray | None = None,
                         noise: np.ndarray | None = None,
                         widths: np.ndarray | None = None,
                         corr: np.ndarray | None = None) -> dict[str, np.ndarray]:
        if not self.enabled:
            return {}
        return abstain_triggers(probs, cls=cls, noise=noise, widths=widths, corr=corr,
                                max_prob=self.max_prob, margin=self.margin,
                                noise_thr=self.noise_thr)

    def abstain(self, probs: np.ndarray, cls: np.ndarray | None = None,
                noise: np.ndarray | None = None,
                widths: np.ndarray | None = None,
                corr: np.ndarray | None = None) -> np.ndarray:
        trig = self.abstain_triggers(probs, cls=cls, noise=noise, widths=widths, corr=corr)
        if not trig:
            return np.zeros(0 if probs is None else len(probs), dtype=bool)
        mask = None
        for m in trig.values():
            mask = m.copy() if mask is None else (mask | m)
        return mask

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "temperature": round(float(self.temperature), 4),
            "train_prior": [round(float(v), 6) for v in self.train_prior],
            "target_prior": [round(float(v), 6) for v in self.target_prior],
            "max_prob": float(self.max_prob),
            "margin": float(self.margin),
            "noise_thr": float(self.noise_thr),
            "classes": list(CLASSES),
            "enabled": bool(self.enabled),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        def _prior(key: str, default: tuple[float, ...]) -> tuple[float, ...]:
            raw = data.get(key)
            if raw is None:
                return default
            vals = tuple(float(v) for v in raw)
            if len(vals) != N_CLASSES:
                raise ValueError(f"{key} debe tener {N_CLASSES} valores")
            return vals

        return cls(
            temperature=float(data.get("temperature", 1.0)),
            train_prior=_prior("train_prior", DEFAULT_TRAIN_PRIOR),
            target_prior=_prior("target_prior", TARGET_PRIOR),
            max_prob=float(data.get("max_prob", 0.45)),
            margin=float(data.get("margin", 0.10)),
            noise_thr=float(data.get("noise_thr", 0.20)),
            enabled=bool(data.get("enabled", True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        """Lee `beat_calibration.json`; si no está, no calibra (identidad)."""
        p = Path(path)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return cls(enabled=False)
        try:
            return cls.from_dict(data)
        except Exception:
            return cls(enabled=False)


IDENTITY = Calibration(enabled=False)
