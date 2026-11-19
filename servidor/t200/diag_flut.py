import sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t202_train as S, t202_eval as E
T, tt = S.T, S.tt
dev = torch.device("cuda:0")
srcs = S.load_sources7(["holter", "icentia", "taqui"])
te = T.pairs_for(srcs, "test", regular_only=True)
y = np.stack([S.y7(srcs[s], j) for s, j in te]); db = np.array([srcs[s].db[j] for s, j in te]); rec = np.array([srcs[s].rec[j] for s, j in te])
keep = np.where(y[:, 1] >= 0)[0]
rng = np.random.default_rng(1)
pos = keep[y[keep, 1] == 1]; neg = rng.choice(keep[y[keep, 1] == 0], 15000, replace=False)
ix = np.concatenate([pos, neg]); te2 = [te[i] for i in ix]; y2 = y[ix]; db2 = db[ix]; rec2 = rec[ix]
print("flutter positivos test por base:", {d: int(((db2 == d) & (y2[:, 1] == 1)).sum()) for d in sorted(set(db2))})
print("por registro:", {r: int(((rec2 == r) & (y2[:, 1] == 1)).sum()) for r in sorted(set(rec2[y2[:, 1] == 1]))})
tr = T.pairs_for(srcs, "train"); ytr = np.stack([srcs[s].Y[j] for s, j in tr]); dtr = np.array([srcs[s].db[j] for s, j in tr]); rtr = np.array([srcs[s].rec[j] for s, j in tr])
print("flutter positivos TRAIN por base:", {d: (int(((dtr == d) & (ytr[:, 1] == 1)).sum()), len(set(rtr[(dtr == d) & (ytr[:, 1] == 1)]))) for d in sorted(set(dtr)) if ((dtr == d) & (ytr[:, 1] == 1)).any()})
for name, p in (("T158", T.T158_PT), ("T202", "/root/holter/t200/runs/t202/best.pt")):
    net, n0 = S.make_model7(dev, p)
    l, _ = T.predict(net, DataLoader(E.DSN(srcs, te2, None), batch_size=256, num_workers=4), dev)
    print(name, "AUROC flutter global", round(tt.auroc(y2[:, 1], l[:, 1]), 4))
    for d in sorted(set(db2)):
        m = db2 == d
        if (y2[m, 1] == 1).any() and (y2[m, 1] == 0).any():
            print("  ", d, "AUROC", round(tt.auroc(y2[m, 1], l[m, 1]), 4), "logit FLUT medio en flutter", round(float(l[m & (y2[:, 1] == 1), 1].mean()), 2),
                  "en no-flutter", round(float(l[m & (y2[:, 1] == 0), 1].mean()), 2), "logit FA medio en flutter", round(float(l[m & (y2[:, 1] == 1), 0].mean()), 2))
    for r in sorted(set(rec2[y2[:, 1] == 1])):
        m = rec2 == r
        if (y2[m, 1] == 0).any():
            print("     reg", r, "n+", int((y2[m, 1] == 1).sum()), "AUROC", round(tt.auroc(y2[m, 1], l[m, 1]), 4))
import os; os._exit(0)
