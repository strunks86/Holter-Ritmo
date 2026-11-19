"""Inventario de etiquetas de bloqueo AV / pausas en las bases (ritmos aux_note y cabeceras)."""
import sys, collections
from pathlib import Path
import wfdb
R = Path("/root/holter/data/HolterIA")
for base, ext in [("bases/mitdb", "atr"), ("bases/incartdb", "atr"), ("bases/ltdb", "atr"), ("bases/sddb", "atr"), ("bases/svdb", "atr"), ("bases/edb", "atr"), ("bases/ltafdb", "atr"), ("bases2/vfdb", "atr"), ("bases2/mghdb", "ari"), ("bases2/ltstdb", "atr"), ("bases2/stdb", "atr"), ("bases2/nsrdb", "atr"), ("bases2/chfdb", "ecg")]:
    c = collections.Counter(); recs = collections.defaultdict(set)
    for h in sorted((R / base).rglob("*." + ext)):
        try: a = wfdb.rdann(str(h)[: -len(ext) - 1], ext)
        except Exception as e: continue
        for s, x in zip(a.symbol, a.aux_note):
            x = (x or "").strip("\x00 ")
            if s == "+" and x: c[x] += 1; recs[x].add(h.stem)
    print(base, {k: (v, len(recs[k])) for k, v in c.most_common() if any(t in k.upper() for t in ("B", "PAUS", "SA", "NOD", "IVR", "P"))}, flush=True)
