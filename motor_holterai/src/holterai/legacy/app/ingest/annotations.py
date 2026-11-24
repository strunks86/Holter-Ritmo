"""Lectura de archivos de anotaciones WFDB (.atr), p. ej. MIT-BIH.

Formato MIT (spec WFDB): cada anotación ocupa un número par de bytes; el
primer byte del par es el menos significativo. Los 6 bits más altos del
par son el tipo (A) y los 10 restantes el intervalo (I) en muestras desde
la anotación anterior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Códigos WFDB (wfdb/ecgcodes.h)
CODE_NAME = {
    0: "NOTQRS", 1: "N", 2: "L", 3: "R", 4: "a", 5: "V", 6: "F", 7: "J",
    8: "A", 9: "S", 10: "E", 11: "j", 12: "/", 13: "Q", 14: "~", 15: "|",
    16: "|", 17: "X", 18: "s", 19: "T", 20: "*", 21: "D", 22: '"', 23: "=",
    24: "p", 25: "B", 26: "?", 27: "t", 28: "+", 29: "u", 30: "?", 31: "!",
    32: "[", 33: "]", 34: "r", 35: "(", 36: ")", 37: ",", 38: "(", 39: "~",
}

AAMI_MAP = {
    1: "N", 2: "N", 3: "N", 11: "N", 25: "N",
    4: "S", 7: "S", 8: "S", 9: "S",
    5: "V", 10: "V",
    6: "F",
    12: "Q", 13: "Q", 26: "Q",
}

SKIP, NUM, SUB, CHN, AUX = 59, 60, 61, 62, 63
RHYTHM_CODES = {28, 34, 35, 36, 37, 38, 39}
AUX_EXCLUDE = ("(N", "(AFIB", "(AFL", "(J", "(B", "(VT", "(VFL", "(SBR", "(SVTA")


@dataclass
class Annotation:
    sample: int
    code: int
    subtype: int = 0
    chan: int = 0
    num: int = 0
    aux: str = ""
    name: str = ""

    @property
    def aami(self) -> str | None:
        return AAMI_MAP.get(self.code)


@dataclass
class Annotations:
    items: list[Annotation] = field(default_factory=list)

    def beats(self) -> list[Annotation]:
        return [a for a in self.items if a.aami is not None]

    def rhythm_changes(self) -> list[Annotation]:
        return [a for a in self.items if a.code in RHYTHM_CODES or a.aux.startswith("(")]


def _pdp11_long(b: np.ndarray) -> int:
    """Entero de 32 bits en formato PDP-11: mitades altas primero, byte bajo primero."""
    hi = int(b[0]) | (int(b[1]) << 8)
    lo = int(b[2]) | (int(b[3]) << 8)
    return (hi << 16) | lo


def read_ann(path: str | Path) -> Annotations:
    path = Path(path)
    if not path.exists() or (path.suffix.lower() == "" and path.with_suffix(".atr").exists()):
        path = path.with_suffix(".atr")
    d = np.fromfile(str(path), dtype=np.uint8)
    out = Annotations()
    i, t = 0, 0
    sub = chan = num = 0
    n = d.size
    while i + 1 < n:
        b0, b1 = int(d[i]), int(d[i + 1])
        i += 2
        word = b0 | (b1 << 8)
        typ = word >> 10
        iv = word & 0x3FF

        if typ == 0:
            if iv == 0:
                break
            continue
        if typ == SKIP:
            if i + 3 >= n:
                break
            t += _pdp11_long(d[i:i + 4])
            i += 4
            continue
        if typ == NUM:
            num = iv
            continue
        if typ == SUB:
            sub = iv
            continue
        if typ == CHN:
            chan = iv
            continue
        if typ == AUX:
            ln = iv
            aux = d[i:i + ln].tobytes().decode("latin-1", "replace")
            i += ln + (ln % 2)
            out.items.append(Annotation(
                sample=t, code=AUX, subtype=sub, chan=chan, num=num,
                aux=aux.replace("\x00", "").strip(), name="AUX",
            ))
            sub = 0
            continue

        t += iv
        out.items.append(Annotation(
            sample=t, code=typ, subtype=sub, chan=chan, num=num,
            aux="", name=CODE_NAME.get(typ, f"code{typ}"),
        ))
        sub = 0
    return out


def inspect(path: str | Path) -> dict:
    ann = read_ann(path)
    codes: dict[str, int] = {}
    for a in ann.items:
        codes[a.name or str(a.code)] = codes.get(a.name or str(a.code), 0) + 1
    beats = ann.beats()
    return {
        "n": len(ann.items),
        "beats": len(beats),
        "codes": dict(sorted(codes.items(), key=lambda kv: -kv[1])),
        "first_beats": [a.sample for a in beats[:5]],
        "rhythms": [(a.sample, a.aux) for a in ann.items if a.aux][:8],
    }

