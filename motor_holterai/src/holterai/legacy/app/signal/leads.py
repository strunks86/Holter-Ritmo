"""Reconstrucción de derivaciones y normalización de nombres.

Con las derivaciones I y II se obtienen de forma EXACTA las otras cuatro
de miembros (Einthoven y Goldberger):

    III = II - I
    aVR = -(I + II) / 2
    aVL =  I - II/2
    aVF =  II - I/2

Esto convierte un Holter de 2-3 canales en 6-7 derivaciones reales, lo que
mejora el análisis de la onda P y la transferencia a modelos de 12
derivaciones.
"""
from __future__ import annotations

import re

import numpy as np

LIMB_FROM_I_II = {
    "III": lambda i, ii: ii - i,
    "AVR": lambda i, ii: -(i + ii) / 2.0,
    "AVL": lambda i, ii: i - ii / 2.0,
    "AVF": lambda i, ii: ii - i / 2.0,
}

ALIASES = {
    "I": ("I", "DERIVACION I", "LEAD I", "D1"),
    "II": ("II", "DERIVACION II", "LEAD II", "D2", "MLII", "ECG2"),
    "III": ("III", "DERIVACION III", "LEAD III", "D3"),
    "AVR": ("AVR", "A V R", "VR"),
    "AVL": ("AVL", "A V L", "VL"),
    "AVF": ("AVF", "A V F", "VF"),
    "V1": ("V1", "V 1", "C1"),
    "V2": ("V2",),
    "V3": ("V3",),
    "V4": ("V4",),
    "V5": ("V5", "V 5", "C5"),
    "V6": ("V6",),
}


def normalize_name(name: str) -> str:
    """'0_MLII' -> 'II', 'Lead V1' -> 'V1'."""
    s = re.sub(r"^[0-9]+\s*[_\-\.]?\s*", "", str(name or "").strip())
    s = s.upper().replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    for canon, aliases in ALIASES.items():
        for a in aliases:
            if s == a or s.replace(" ", "") == a.replace(" ", ""):
                return canon
    # coincidencia por contenido, de alias más largo a más corto
    for canon in sorted(ALIASES, key=lambda k: -len(k)):
        for a in ALIASES[canon]:
            if a in s:
                return canon
    return ""


def map_leads(names: list[str]) -> list[str]:
    """Nombres canónicos de las derivaciones disponibles ('', si se desconoce)."""
    return [normalize_name(n) for n in names]


def reconstruct(samples: np.ndarray, leads: list[str],
                add_chest: bool = True) -> tuple[np.ndarray, list[str], dict]:
    """Añade las derivaciones de miembros que se puedan derivar.

    Devuelve (señal, nombres, informe). No modifica la señal original.
    """
    samples = np.asarray(samples, dtype=np.float32)
    canon = map_leads(leads)
    info = {"input": list(leads), "mapped": canon, "derived": [], "warnings": []}
    if samples.ndim == 1:
        samples = samples[:, None]

    idx = {c: i for i, c in enumerate(canon) if c}
    out_cols = [samples[:, i] for i in range(samples.shape[1])]
    out_names = list(leads)

    have_i = "I" in idx
    have_ii = "II" in idx
    if have_i and have_ii:
        i = samples[:, idx["I"]].astype(np.float64)
        ii = samples[:, idx["II"]].astype(np.float64)
        for name, fn in LIMB_FROM_I_II.items():
            if name in idx:
                continue
            out_cols.append(fn(i, ii).astype(np.float32))
            out_names.append(name)
            info["derived"].append(name)
        # comprobación de coherencia: aVR suele ser predominantemente negativo
        avr = -(i + ii) / 2.0
        info["avr_negative_fraction"] = round(float(np.mean(avr < 0)), 3)
        if info["avr_negative_fraction"] < 0.35:
            info["warnings"].append(
                "aVR derivada poco negativa: revisa la correspondencia de canales"
            )
    else:
        if not have_i:
            info["warnings"].append("no se identificó la derivación I")
        if not have_ii:
            info["warnings"].append("no se identificó la derivación II")

    if not add_chest:
        return np.ascontiguousarray(np.stack(out_cols, axis=1)), out_names, info
    return np.ascontiguousarray(np.stack(out_cols, axis=1)), out_names, info


def pick_p_lead(leads: list[str]) -> int | None:
    """Derivación más informativa para la onda P."""
    canon = map_leads(leads)
    for pref in ("II", "V1", "I", "AVF"):
        if pref in canon:
            return canon.index(pref)
    return None


def best_p_leads(leads: list[str], limit: int = 2) -> list[int]:
    canon = map_leads(leads)
    order = ["II", "V1", "I", "AVF", "III"]
    out = [canon.index(k) for k in order if k in canon]
    return out[:limit]

