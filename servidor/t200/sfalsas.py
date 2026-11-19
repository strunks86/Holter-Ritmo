"""De donde salen las S falsas: ventanas con PAC=0 y p(PAC)>umbral, por base y por lo que si tienen."""
import sys, collections
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t202_train as S, diag205_lib as DL
T = S.T
dev = torch.device("cuda:0")
tags = sys.argv[1].split(","); models = dict(a.split("=") for a in sys.argv[2].split(",")); split = sys.argv[3]; thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
srcs = [T.Src(T.NEW / f"{t}_X.npy", T.NEW / f"{t}_meta.npz") for t in tags]
for s, t in zip(srcs, tags): s.TQ = np.load(T.NEW / f"{t}_tq.npz")["TQ"]
for name, path in models.items():
    net, _ = S.make_model7(dev, Path(path)); tot = collections.Counter(); totn = 0; totpos = 0; tp = 0
    for src, tag in zip(srcs, tags):
        for db in sorted(set(src.db)):
            idx = np.where((src.db == db) & (src.split == split))[0]
            if len(idx) > 12000: idx = np.sort(np.random.default_rng(1).choice(idx, 12000, replace=False))
            if not len(idx): continue
            l, y = T.predict(net, DataLoader(DL.D(src, idx), batch_size=256, num_workers=3), dev)
            p = 1 / (1 + np.exp(-l)); neg = y[:, 3] == 0; fp = neg & (p[:, 3] > thr); pos = y[:, 3] == 1
            c = collections.Counter()
            for i in np.where(fp)[0]:
                k = "V" if y[i, 2] == 1 else "FA" if y[i, 0] == 1 else "FLUT" if y[i, 1] == 1 else "RUIDO" if y[i, 4] == 1 else "TV/TSV" if (y[i, 5] == 1 or y[i, 6] == 1) else "nada(normal)"
                c[k] += 1
            tot.update({f"{k}": v for k, v in c.items()}); totn += int(neg.sum()); totpos += int(pos.sum()); tp += int((pos & (p[:, 3] > thr)).sum())
            print(name, db, f"neg {int(neg.sum())} S_falsas {int(fp.sum())} ({1000 * fp.sum() / max(1, neg.sum()):.1f}/1000) pos {int(pos.sum())} Se {((pos & (p[:, 3] > thr)).sum() / max(1, pos.sum())):.3f}", dict(c), flush=True)
    print(name, "TOTAL neg", totn, "falsas", sum(tot.values()), dict(tot), "pos", totpos, "VP", tp, "VPP", round(tp / max(1, tp + sum(tot.values())), 3), flush=True)
    del net; torch.cuda.empty_cache()
import os; os._exit(0)
