"""Tiras de 3 canales (10 s) de S falsas para que las revise Jorge. Marca el latido que el modelo llamó S y las anotaciones de la base."""
import sys, json
from pathlib import Path
import numpy as np, wfdb, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = Path("/root/holter/data/HolterIA/bases"); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
sel = json.load(open(sys.argv[1]))
for n, (db, rec, st, toff) in enumerate(sel, 1):
    h = list((B / db).rglob(f"{rec}.hea"))[0]; p = str(h)[:-4]; fs = wfdb.rdheader(p).fs
    a0 = int((st - 1) * fs); a1 = int((st + 11) * fs)
    s, f = wfdb.rdsamp(p, sampfrom=max(0, a0), sampto=a1); s = np.nan_to_num(s)[:, :3]; tt = np.arange(len(s)) / fs + max(0, a0) / fs
    an = wfdb.rdann(p, "atr", sampfrom=max(0, a0), sampto=a1)
    fig, ax = plt.subplots(s.shape[1], 1, figsize=(16, 2.2 * s.shape[1]), sharex=True)
    ax = np.atleast_1d(ax)
    for c in range(s.shape[1]):
        y = s[:, c] - np.median(s[:, c]); ax[c].plot(tt, y, "k", lw=0.7); ax[c].set_ylabel(f["sig_name"][c])
        ax[c].set_xticks(np.arange(np.floor(tt[0]), tt[-1], 0.2), minor=True); ax[c].grid(True, which="both", color="#f3c0c0", lw=0.4)
        top = np.percentile(y, 99.5) * 1.15
        for sm, sy, au in zip(an.sample, an.symbol, an.aux_note):
            if c == 0: ax[c].text(sm / fs, top, sy + (au.strip("\x00") if sy == "+" else ""), fontsize=8, ha="center", color="b")
    for c in range(s.shape[1]): ax[c].axvline(st + toff, color="r", lw=1.2, alpha=0.6)
    ax[0].set_title(f"Tira {n}: base {db}, registro {rec}, segundo {st:.0f}. Rojo = latido que el modelo marca S. Azul = anotación de la base")
    ax[-1].set_xlabel("segundos"); plt.tight_layout(); plt.savefig(out / f"tira_{n:02d}_{db}_{rec}.png", dpi=110); plt.close()
    print("ok", n, db, rec, flush=True)
