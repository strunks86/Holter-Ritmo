"""combo_p: igual que combo_lat, mas veto por onda P (detector p2): se descarta una S si su P es sinusal (misma forma), con PR habitual y no prematura.
Ventana (red) + latido a latido: una ventana es S solo si la red la marca y algun latido (fuera de FA) tiene p(S) alta.
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
lats = []
for lm in lat_model.split("+"):
    n_ = ns["Net"]().to(dev); n_.load_state_dict(torch.load(lm, map_location=dev)); n_.eval(); lats.append(n_)
def lat(x, f): return torch.log(torch.stack([torch.softmax(n_(x, f), 1) for n_ in lats]).mean(0))
win, _ = S7.make_model7(dev, Path(win_model))
import wfdb
from scipy.signal import butter, sosfiltfilt, resample_poly, find_peaks
from fractions import Fraction
PM = sys.argv[4] if len(sys.argv) > 4 else "runs/p2/best.pt"
VETO = sys.argv[5] if len(sys.argv) > 5 else "P"
pc = open(Path(__file__).resolve().parent / "p_train.py").read(); pns = {"nn": torch.nn, "torch": torch}
exec(pc[pc.index("class Net"):pc.index("dev = torch.device")], pns)
pnet = pns["Net"]().to(dev); pnet.load_state_dict(torch.load(PM, map_location=dev)); pnet.eval()
FP = 250; WP = 1000
def pdet(path):
    s, f = wfdb.rdsamp(path); fs = f["fs"]; s = np.nan_to_num(s)[:, :2]
    if s.shape[1] < 2: s = np.concatenate([s, s], 1)
    x = sosfiltfilt(butter(2, [0.5, 40], "bp", fs=fs, output="sos"), s, axis=0)
    fr = Fraction(FP, int(round(fs))).limit_denominator(100); x = resample_poly(x, fr.numerator, fr.denominator, axis=0).astype(np.float32)
    x /= np.percentile(np.abs(x), 99, axis=0) + 1e-6
    n = len(x); o = np.zeros(n, np.float32); c = np.zeros(n, np.float32); st = list(range(0, max(1, n - WP + 1), WP // 2))
    with torch.no_grad():
        for b in range(0, len(st), 512):
            ii = st[b:b + 512]; seg = torch.from_numpy(np.stack([x[i:i + WP].T for i in ii])).to(dev)
            pr = torch.sigmoid(pnet(seg)).cpu().numpy()
            for k, i in enumerate(ii): o[i:i + WP] += pr[k]; c[i:i + WP] += 1
    o /= np.maximum(c, 1); P, pk = find_peaks(o, height=0.4, distance=int(0.12 * FP)); return x, P
def veto(path, t, cand):
    """t: tiempos de latido (s). Devuelve mascara de latidos candidatos vetados (P sinusal a tiempo con PR habitual)."""
    x, P = pdet(path); n = len(x); qb = np.round(t * FP).astype(int)
    # P de cada latido: la ultima P entre 400 y 80 ms antes del QRS
    pb = np.full(len(qb), -1)
    j = np.searchsorted(P, qb - int(0.08 * FP)) - 1
    ok = (j >= 0) & (qb - P[np.clip(j, 0, None)] <= int(0.40 * FP)); pb[ok] = P[j[ok]]
    pr = np.where(pb >= 0, qb - pb, -1)
    v = np.zeros(len(qb), bool); nv = 0
    for i in np.where(cand)[0]:
        if pb[i] < 0 or i < 9: continue
        prev = [k for k in range(i - 8, i) if pb[k] >= 0 and not cand[k]]
        if len(prev) < 4: continue
        prm = np.median(pr[prev]); pp_prev = np.median(np.diff(pb[prev])) if len(prev) > 1 else None
        if pp_prev is None or abs(pr[i] - prm) > 0.03 * FP: continue
        lastp = pb[[k for k in range(i - 1, i - 4, -1) if pb[k] >= 0][0]] if any(pb[k] >= 0 for k in range(i - 1, i - 4, -1)) else -1
        if lastp < 0 or (pb[i] - lastp) < 0.88 * pp_prev: continue  # P prematura: se mantiene la S
        a, b = pb[i] - 25, pb[i] + 25
        if a < 0 or b > n: continue
        tpl = np.median(np.stack([x[pb[k] - 25:pb[k] + 25].T.ravel() for k in prev if 25 <= pb[k] < n - 25]), 0)
        if np.corrcoef(x[a:b].T.ravel(), tpl)[0, 1] >= 0.85: v[i] = True
    return v
src = T.Src(T.NEW / "holter_X.npy", T.NEW / "holter_meta.npz"); src.TQ = np.load(T.NEW / "holter_tq.npz")["TQ"]
rows = []; DG = []
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
        try: vt = veto(str(hs[0])[:-4], t, ps > 0.3)
        except Exception as e: print("veto fallo", rec, e, flush=True); vt = np.zeros(len(ps), bool)
        trasV = np.zeros(len(ps), bool); trasV[1:] = P[:-1, 2] > 0.5  # latido justo despues de un V (segun el modelo)
        psv = np.where(trasV if VETO == "trasV" else (vt | trasV if VETO == "ambos" else vt), 0.0, ps)
        idx = np.where((src.db == db) & (src.rec == rec) & (src.split == "test"))[0]
        l, y = T.predict(win, DataLoader(DL.D(src, idx), batch_size=256, num_workers=3), dev); p = 1 / (1 + np.exp(-l))
        for k, j in enumerate(idx):
            m = (t >= src.start[j]) & (t < src.start[j] + 10.0)
            rows.append((db, y[k, 3], y[k, 2], p[k, 3], p[k, 0], ps[m].max() if m.any() else 0.0, psv[m].max() if m.any() else 0.0))
            if m.any():
                ii = np.where(m)[0]; tb = ii[np.argmax(ps[ii])]
                DG.append((db, rec, float(src.start[j]), int((yb[ii] == 1).sum()), int(yb[tb]), float(F[tb, 0]), float(t[tb] - src.start[j]), float(F[tb, 4]), int(vt[tb])))
            else: DG.append((db, rec, float(src.start[j]), 0, -9, 0.0, 0.0, 0.0, 0))
        print(db, rec, len(idx), "vetados", int((psv < ps).sum()), "de", int((ps > 0.3).sum()), flush=True)
D = np.array([r[0] for r in rows]); A = np.array([r[1:] for r in rows], float)
yS, yV, pS, pFA, pL, pLv = A.T
def rep(nm, pred, sel=None):
    sel = np.ones(len(A), bool) if sel is None else sel
    pos = sel & (yS == 1); neg = sel & (yS == 0); sv = neg & (yV == 1); tp = (pred & pos).sum(); fp = (pred & neg).sum()
    print(f"{nm}: S+ {pos.sum()} Se {tp / max(1, pos.sum()):.3f} VPP {tp / max(1, tp + fp):.3f} falsas {fp} | en ventanas con V {(pred & sv).sum()}/{sv.sum()}", flush=True)
nofa = pFA <= 0.5; pos = yS == 1
for nm, L in (("SIN veto P", pL), ("CON veto P", pLv)):
    sc = np.where(nofa, np.sqrt(pS * L), 0.0)
    for thr in (0.5, 0.7, 0.8, 0.9): rep(f"{nm} MEDIA GEOMETRICA > {thr}", sc > thr)
sc0 = np.where(nofa, np.sqrt(pS * pL), 0.0); sc1 = np.where(nofa, np.sqrt(pS * pLv), 0.0)
for thr in (0.7, 0.8):
    se = ((sc1 > thr) & pos).sum(); t0 = np.sort(sc0[pos])[::-1][max(0, se - 1)]
    rep(f"   SIN veto a igual Se que CON veto>{thr} (umbral {t0:.3f})", sc0 >= t0)
for db in dbs:
    rep(f"[{db}] SIN veto >0.8", sc0 > 0.8, D == db); rep(f"[{db}] CON veto >0.8", sc1 > 0.8, D == db)
np.savez("/root/holter/t200/runs/combo_p_rows.npz", D=D, A=A)
import json; json.dump(DG, open("/root/holter/t200/runs/combo_p_diag.json", "w"))
sc1 = np.where(nofa, np.sqrt(pS * pLv), 0.0); fl = np.where((sc1 > 0.8) & (yS == 0))[0]
print("FALSAS >0.8 (db, rec, inicio s, n S latido en ventana, etiqueta latido top, prem top, t top en ventana, enFA, vetado, pS red, pL):")
for i in fl: print("  ", DG[i], round(pS[i], 2), round(pL[i], 2))
print("TERMINADO"); import os; os._exit(0)
