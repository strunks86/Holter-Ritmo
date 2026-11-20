"""Promedio de probabilidades de varios modelos: S (PAC) en la prueba antigua. Se a VPP fijo y S falsas sobre V."""
import sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t202_train as S, diag205_lib as DL
T = S.T; dev = torch.device("cuda:0")
tags = sys.argv[1].split(","); models = dict(a.split("=") for a in sys.argv[2].split(",")); split = sys.argv[3]
srcs = [T.Src(T.NEW / f"{t}_X.npy", T.NEW / f"{t}_meta.npz") for t in tags]
for s, t in zip(srcs, tags): s.TQ = np.load(T.NEW / f"{t}_tq.npz")["TQ"]
sel = []
for src in srcs:
    for db in sorted(set(src.db)):
        idx = np.where((src.db == db) & (src.split == split))[0]
        if len(idx) > 12000: idx = np.sort(np.random.default_rng(1).choice(idx, 12000, replace=False))
        if len(idx): sel.append((src, idx))
L = {}; Y = None
for name, path in models.items():
    net, _ = S.make_model7(dev, Path(path)); ls, ys = [], []
    for src, idx in sel:
        l, y = T.predict(net, DataLoader(DL.D(src, idx), batch_size=256, num_workers=3), dev); ls.append(l); ys.append(y)
    L[name] = np.concatenate(ls); Y = np.concatenate(ys); del net; torch.cuda.empty_cache()
def rep(name, l):
    for k, nm in ((3, "S"), (2, "V")):
        m = Y[:, k] >= 0; y = Y[m, k]; s = l[m, k]; o = np.argsort(-s); ys = y[o]; tp = np.cumsum(ys); ppv = tp / np.arange(1, len(ys) + 1); se = tp / max(1, ys.sum())
        out = [f"{nm} AP {float(ppv[ys == 1].mean()):.4f}"]
        for v in (0.90, 0.95, 0.97):
            ok = np.where(ppv >= v)[0]; out.append(f"Se@VPP{int(v * 100)} {se[ok[-1]] if len(ok) else 0:.3f}")
        if k == 3:
            for t in (0.80, 0.85):
                i = int(np.searchsorted(se, t)); thr = s[o][min(i, len(o) - 1)]; sv = m & (Y[:, 2] == 1) & (Y[:, 3] == 0)
                out.append(f"a Se {t}: VPP {ppv[min(i, len(o) - 1)]:.3f}, S falsa en ventanas con V {1000 * (l[sv, 3] >= thr).mean():.1f}/1000")
        print(name, " | ".join(out), flush=True)
for n, l in L.items(): rep(n, l)
names = list(L)
rep("PROMEDIO(" + "+".join(names) + ")", np.mean([L[n] for n in names], 0))
for i in range(len(names)):
    for j in range(i + 1, len(names)): rep(f"PROMEDIO({names[i]}+{names[j]})", (L[names[i]] + L[names[j]]) / 2)
rep("MINIMO(todos) solo S", np.min([L[n] for n in names], 0))
print("TERMINADO"); import os; os._exit(0)
