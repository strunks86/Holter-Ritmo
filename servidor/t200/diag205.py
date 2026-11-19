"""Diagnostico: T204 vs T205 por base nueva (split val), AUROC/AP por salida, y cruces FA/FLUT."""
import sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t202_train as S
T, tt = S.T, S.tt
N = ["FA", "FLUT", "PVC", "PAC", "RUIDO", "TV", "TSV"]

class D(Dataset):
    def __init__(s, src, idx): s.src, s.idx = src, idx
    def __len__(s): return len(s.idx)
    def __getitem__(s, i):
        j = int(s.idx[i]); y = S.y7(s.src, j); y0 = y.copy(); y0[0] = s.src.Y[j, 0]
        return torch.from_numpy(np.asarray(s.src.X[j], dtype=np.float32).T.copy()), torch.from_numpy(y0)

def ap(y, s):
    o = np.argsort(-s); y = y[o]; tp = np.cumsum(y); return float((tp / np.arange(1, len(y) + 1))[y == 1].mean())

dev = torch.device("cuda:0")
tags = sys.argv[1].split(","); split = sys.argv[3] if len(sys.argv) > 3 else "val"
models = dict(a.split("=") for a in sys.argv[2].split(","))
srcs = [T.Src(T.NEW / f"{t}_X.npy", T.NEW / f"{t}_meta.npz") for t in tags]
for s, t in zip(srcs, tags): s.TQ = np.load(T.NEW / f"{t}_tq.npz")["TQ"]
for name, path in models.items():
    net, _ = S.make_model7(dev, Path(path))
    for src, tag in zip(srcs, tags):
        for db in sorted(set(src.db)):
            idx = np.where((src.db == db) & (src.split == split))[0]
            if len(idx) > 6000: idx = np.sort(np.random.default_rng(1).choice(idx, 6000, replace=False))
            if not len(idx): continue
            l, y = T.predict(net, DataLoader(D(src, idx), batch_size=256, num_workers=3), dev)
            p = 1 / (1 + np.exp(-l)); cells = []
            for k, nm in enumerate(N):
                m = y[:, k] >= 0; pos = int((y[m, k] == 1).sum()); neg = int((y[m, k] == 0).sum())
                if pos and neg: cells.append(f"{nm} AUROC {tt.auroc(y[m, k], l[m, k]):.3f} AP {ap(y[m, k], l[m, k]):.3f} (+{pos}/{neg}) media+ {p[m, k][y[m, k] == 1].mean():.2f} media0 {p[m, k][y[m, k] == 0].mean():.3f}")
                elif neg: cells.append(f"{nm} sin+ FP>0.5 {1000 * (p[m, k] > 0.5).mean():.0f}/1000 (n{neg})")
                elif pos: cells.append(f"{nm} solo+ >0.5 {(p[m, k] > 0.5).mean():.2f} (n{pos})")
            fl = y[:, 1] == 1
            if fl.any(): cells.append(f"[en FLUT+: pFA>0.5 {(p[fl, 0] > 0.5).mean():.2f}, pFLUT>0.5 {(p[fl, 1] > 0.5).mean():.2f}]")
            fa = y[:, 0] == 1
            if fa.any(): cells.append(f"[en FA+: pFA>0.5 {(p[fa, 0] > 0.5).mean():.2f}, pFLUT>0.5 {(p[fa, 1] > 0.5).mean():.2f}]")
            print(name, db, split, "n", len(idx), " | ".join(cells), flush=True)
    del net; torch.cuda.empty_cache()
import os; os._exit(0)
