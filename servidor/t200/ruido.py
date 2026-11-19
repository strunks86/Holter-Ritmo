"""Banco de ruido real (NSTDB: em = electrodo, ma = musculo, bw = linea base) y mezcla sobre ventanas.

Cada registro de ruido (30 min, 2 canales, 360 Hz) se preprocesa igual que las ventanas
(remuestreo a 500 Hz + FIR 0,5-50 Hz) y se parte en el tiempo: primeros 21 min = entrenamiento,
ultimos 9 min = prueba (asi el ruido de prueba nunca se vio).

Etiqueta RUIDO (5a salida): 1 si >=2 canales usados quedan con SNR <= 6 dB (o el unico canal si
la base es de 1 canal); 0 si nada queda por debajo de 12 dB o el ruido cae en 1 solo canal de >=2;
-1 (enmascarado) en la zona gris. Con SNR < 0 dB en todos los canales usados, los positivos de
FA/FLUT/PVC/PAC se enmascaran (no se pueden leer), pero los negativos se mantienen: asi el modelo
aprende a NO inventar V/S/FA sobre ruido.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

BASES = Path(os.environ.get("BASES", "/root/holter/data/HolterIA/bases"))
TIPOS = ("em", "ma", "bw")
PESO_TIPO = np.array([0.5, 0.35, 0.15])
SPLIT_S = 21 * 60


def cargar_banco(tb):
    import wfdb
    banco = {"train": {}, "test": {}}
    for t in TIPOS:
        r = wfdb.rdrecord(str(BASES / "nstdb" / t))
        x = tb.prep_record(np.concatenate([r.p_signal[:, :2], np.zeros((r.p_signal.shape[0], 1))], 1),
                           float(r.fs))[:, :2].astype(np.float32)
        cut = int(SPLIT_S * tb.FS)
        banco["train"][t] = x[:cut]
        banco["test"][t] = x[cut:]
    return banco


def _z(c):
    s = c.std()
    return (c - c.mean()) / (s + 1e-6) if s > 1e-8 else c * 0


def mezclar(x, y, banco_split, rng, snr=None, todos=None, puro=False):
    """x: (3, L) z-score por canal (canales vacios = 0); y: (5,) etiquetas. Devuelve (x2, y2)."""
    L = x.shape[1]
    used = [j for j in range(3) if np.abs(x[j]).max() > 0]
    if not used:
        return x, y
    t = TIPOS[rng.choice(3, p=PESO_TIPO)]
    nb = banco_split[t]
    y2 = y.copy()
    x2 = x.copy()
    if puro:
        for j in used:
            o = rng.integers(0, nb.shape[0] - L)
            x2[j] = _z(nb[o:o + L, rng.integers(0, 2)])
        y2[:] = 0.0
        y2[4] = 1.0
        return x2, y2
    if todos is None:
        todos = rng.random() < 0.7
    canales = used if (todos or len(used) == 1) else [used[rng.integers(0, len(used))]]
    if snr is None:
        snr = rng.uniform(-6.0, 18.0)
    snrs = {}
    for j in canales:
        s = snr + (rng.uniform(-2, 2) if len(canales) > 1 else 0.0)
        o = rng.integers(0, nb.shape[0] - L)
        n = _z(nb[o:o + L, rng.integers(0, 2)])
        x2[j] = _z(x[j] + n * 10 ** (-s / 20.0))
        snrs[j] = s
    malos = sum(1 for s in snrs.values() if s <= 6.0)
    grises = sum(1 for s in snrs.values() if 6.0 < s < 12.0)
    if malos >= 2 or (len(used) == 1 and malos == 1):
        y2[4] = 1.0
    elif malos == 0 and grises == 0:
        y2[4] = 0.0
    elif len(used) >= 2 and len(canales) == 1:
        y2[4] = 0.0
    else:
        y2[4] = -1.0
    if len(snrs) == len(used) and max(snrs.values()) < 0.0:
        for k in range(4):
            if y2[k] == 1.0:
                y2[k] = -1.0
    return x2.astype(np.float32), y2
