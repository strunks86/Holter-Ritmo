"""Diagnostico: por que bajo FA en shdb-af (T158 vs T200). Separa por ritmo real de la ventana."""
import sys, collections
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t200_train as T, t200_eval as E
tt = T.tt
dev = torch.device("cuda:0")
srcs = T.load_sources(["holter"])
te = [(s, j) for s, j in T.pairs_for(srcs, "test") if srcs[s].db[j] == "shdb-af"]
tr = [(s, j) for s, j in T.pairs_for(srcs, "train") if srcs[s].db[j] == "shdb-af"]
Ytr = np.stack([srcs[s].Y[j] for s, j in tr])
print("TRAIN shdb-af: n", len(tr), "FA1", int((Ytr[:,0]==1).sum()), "FA0", int((Ytr[:,0]==0).sum()),
      "FLUT1", int((Ytr[:,1]==1).sum()), "FLUT1 con FA0", int(((Ytr[:,1]==1)&(Ytr[:,0]==0)).sum()))
out = {}
for name, p in (("T158", T.T158_PT), ("T200", "/root/holter/t200/runs/t200/best.pt")):
    net = E.load(p, dev)
    l, y = T.predict(net, DataLoader(T.DS(srcs, te), batch_size=256, num_workers=4), dev)
    out[name] = l
fl = y[:, 1] == 1
print("TEST shdb-af: n", len(y), "FA1", int((y[:,0]==1).sum()), "FLUT1", int(fl.sum()), "resto(no FA no flutter)", int(((y[:,0]==0)&~fl).sum()))
recs = np.array([srcs[s].rec[j] for s, j in te])
for name, l in out.items():
    m = y[:, 0] >= 0
    print(name, "AUROC FA con flutter como no-FA:", round(tt.auroc(y[m,0], l[m,0]), 4),
          "| sin ventanas de flutter:", round(tt.auroc(y[m & ~fl,0], l[m & ~fl,0]), 4),
          "| logit FA medio: FA", round(float(l[y[:,0]==1,0].mean()),2), "flutter", round(float(l[fl,0].mean()),2) if fl.any() else None,
          "otros", round(float(l[(y[:,0]==0)&~fl,0].mean()),2))
    for r in sorted(set(recs)):
        k = (recs == r) & ~fl
        if (y[k,0]==1).any() and (y[k,0]==0).any():
            print("   ", name, r, "n", int(k.sum()), "pos", int((y[k,0]==1).sum()), "AUROC", round(tt.auroc(y[k,0], l[k,0]),4))
import os; os._exit(0)
