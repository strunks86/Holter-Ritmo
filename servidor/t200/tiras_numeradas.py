"""Hojas con tiras de 10 s (todos los canales) y el latido dudoso numerado en grande, para que Jorge responda por numero."""
import sys, json
from pathlib import Path
import numpy as np, wfdb, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = Path("/root/holter/data/HolterIA/bases"); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
sel = json.load(open(sys.argv[1])); POR = 4
for h0 in range(0, len(sel), POR):
    grp = sel[h0:h0 + POR]; filas = []
    for db, rec, st, toff in grp:
        p = str(list((B / db).rglob(f"{rec}.hea"))[0])[:-4]; fs = wfdb.rdheader(p).fs
        s, f = wfdb.rdsamp(p, sampfrom=int(st * fs), sampto=int((st + 10) * fs)); filas.append((db, rec, st, toff, np.nan_to_num(s)[:, :3], fs, f["sig_name"][:3]))
    nax = sum(x[4].shape[1] for x in filas)
    fig, ax = plt.subplots(nax, 1, figsize=(18, 1.55 * nax + 0.4 * len(filas)), gridspec_kw={"hspace": 0.08})
    k = 0
    for n, (db, rec, st, toff, s, fs, nm) in enumerate(filas, h0 + 1):
        tt = np.arange(len(s)) / fs
        for c in range(s.shape[1]):
            a = ax[k]; y = s[:, c] - np.median(s[:, c]); a.plot(tt, y, "k", lw=0.8)
            a.set_xlim(0, 10); a.set_xticks(np.arange(0, 10.01, 1)); a.set_xticks(np.arange(0, 10, 0.2), minor=True)
            a.grid(True, which="both", color="#f3c0c0", lw=0.4); a.set_yticks([]); a.set_ylabel(nm[c], fontsize=8)
            a.axvline(toff, color="r", lw=2, alpha=0.5)
            if c == 0:
                a.set_title(f"Latido {n}   ({db} {rec}, segundo {st:.0f})", loc="left", fontsize=11, fontweight="bold")
                a.text(toff + 0.08, np.percentile(y, 99) * 0.9, str(n), color="r", fontsize=26, fontweight="bold", va="top")
            if c < s.shape[1] - 1: a.set_xticklabels([])
            k += 1
    fig.savefig(out / f"latidos_{h0 + 1:02d}-{h0 + len(grp):02d}.png", dpi=100, bbox_inches="tight"); plt.close(fig); print("ok", h0 + 1, flush=True)
