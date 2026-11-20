"""Ventana (red) + latido a latido: una ventana es S solo si la red la marca y algun latido (fuera de FA) tiene p(S) alta.
Compara con la red sola a igual sensibilidad. Bases holter, split test."""
import sys, types
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t202_train as S7, diag205_lib as DL, lat_build as LB
LB.TODOS = True
dev = torch.device("cuda:0"); T = S7.T
win_model, lat_model, dbs = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
src_code = open(Path(__file__).resolve().parent / "lat_train2.py").read()
ns = {"nn": torch.nn, "torch": torch}; exec(src_code[src_code.index("class Blk"):src_code.index("net = Net()")], ns)
lat = ns["Net"]().to(dev); lat.load_state_dict(torch.load(lat_model, map_location=dev)); lat.eval()
win, _ = S7.make_model7(dev, Path(win_model))
src = T.Src(T.NEW / "holter_X.npy", T.NEW / "holter_meta.npz"); src.TQ = np.load(T.NEW / "holter_tq.npz")["TQ"]
rows = []
for db in dbs:
    for rec in sorted(set(src.rec[(src.db == db) & (src.split == "test")])):
        hs = list((LB.R / "bases" / db).rglob(f"{rec}.atr"))
        if not hs: continue
        o = LB.one((db, str(hs[0])[:-4], "atr", "test"))
        if o is None: continue
        _, _, _, X, F, yb, t = o
        F = F.copy(); F[:, :2] = np.clip(F[:, :2], 0, 3); F[:, 2] = np.clip(F[:, 2], 0.2, 2.5); F[:, 3] = np.clip(F[:, 3], 0, 1); F[:, 5] /= 3
        with torch.no_grad():
            P = np.concatenate([torch.softmax(lat(torch.from_numpy(X[i:i + 4096].astype(np.float32)).to(dev), torch.from_numpy(F[i:i + 4096]).to(dev)), 1).cpu().numpy() for i in range(0, len(X), 4096)])
        ps = np.where(F[:, 4] > 0, 0.0, P[:, 1])  # sin S en FA anotada
        idx = np.where((src.db == db) & (src.rec == rec) & (src.split == "test"))[0]
        l, y = T.predict(win, DataLoader(DL.D(src, idx), batch_size=256, num_workers=3), dev); p = 1 / (1 + np.exp(-l))
        for k, j in enumerate(idx):
            m = (t >= src.start[j]) & (t < src.start[j] + 10.0)
            rows.append((db, y[k, 3], y[k, 2], p[k, 3], p[k, 0], ps[m].max() if m.any() else 0.0))
        print(db, rec, len(idx), flush=True)
D = np.array([r[0] for r in rows]); A = np.array([r[1:] for r in rows], float)
yS, yV, pS, pFA, pL = A.T
def rep(nm, pred, sel=None):
    sel = np.ones(len(A), bool) if sel is None else sel
    pos = sel & (yS == 1); neg = sel & (yS == 0); sv = neg & (yV == 1); tp = (pred & pos).sum(); fp = (pred & neg).sum()
    print(f"{nm}: S+ {pos.sum()} Se {tp / max(1, pos.sum()):.3f} VPP {tp / max(1, tp + fp):.3f} falsas {fp} | en ventanas con V {(pred & sv).sum()}/{sv.sum()}", flush=True)
nofa = pFA <= 0.5; pos = yS == 1
for wt in (0.5, 0.3, 0.2):
    for lt in (0.5, 0.7, 0.9, 0.95):
        comb = (pS > wt) & nofa & (pL > lt); se = (comb & pos).sum() / pos.sum()
        sc = np.where(nofa, pS, 0.0); thr = np.sort(sc[pos])[::-1][max(0, int(round(se * pos.sum())) - 1)]
        rep(f"COMBINADO red>{wt} latido>{lt}", comb); rep(f"   red sola a igual Se (umbral {thr:.3f})", sc >= thr)
sc = np.where(nofa, np.sqrt(pS * pL), 0.0)
for thr in (0.5, 0.7, 0.8, 0.9):
    rep(f"MEDIA GEOMETRICA > {thr}", sc > thr)
for db in dbs:
    rep(f"[{db}] red sola 0.86", (pS > 0.86) & nofa, D == db); rep(f"[{db}] red>0.3 y latido>0.9", (pS > 0.3) & nofa & (pL > 0.9), D == db)
print("TERMINADO"); import os; os._exit(0)
