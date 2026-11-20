"""Red (ventana) + filtro latido a latido: S de la red solo si no hay FA y hay un latido prematuro de morfologia habitual."""
import sys, collections
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t202_train as S, diag205_lib as DL, regla_s as RS
RS.RET_ALL = True
T = S.T; dev = torch.device("cuda:0")
model = sys.argv[1]; dbs = sys.argv[2].split(","); split = sys.argv[3]
src = T.Src(T.NEW / "holter_X.npy", T.NEW / "holter_meta.npz"); src.TQ = np.load(T.NEW / "holter_tq.npz")["TQ"]
net, _ = S.make_model7(dev, Path(model))
rows = []
for db in dbs:
    for rec in sorted(set(src.rec[(src.db == db) & (src.split == split)])):
        idx = np.where((src.db == db) & (src.rec == rec) & (src.split == split))[0]
        hs = list((RS.R / db).rglob(f"{rec}.hea"))
        if not hs: continue
        try: t, sym, prem, cmin, fa = RS.feats(hs[0].with_suffix(""))
        except Exception as e: print("omitido", db, rec, e); continue
        l, y = T.predict(net, DataLoader(DL.D(src, idx), batch_size=256, num_workers=3), dev); p = 1 / (1 + np.exp(-l))
        for k, j in enumerate(idx):
            a, b = src.start[j], src.start[j] + 10.0; m = (t >= a) & (t < b)
            pm = prem[m]; cm = cmin[m]
            # mejor latido candidato de la ventana: el mas prematuro entre los de morfologia habitual
            best = {ct: (pm[cm >= ct].min() if (cm >= ct).any() else 9.0) for ct in (0.85, 0.90, 0.95)}
            rows.append((db, y[k, 3], y[k, 2], y[k, 0], p[k, 3], p[k, 0], best[0.85], best[0.90], best[0.95]))
A = np.array([r[1:] for r in rows], float); D = np.array([r[0] for r in rows])
yS, yV, yFA, pS, pFA = A[:, 0], A[:, 1], A[:, 2], A[:, 3], A[:, 4]
def rep(nm, pred, sel=None):
    sel = np.ones(len(A), bool) if sel is None else sel
    pos = sel & (yS == 1); neg = sel & (yS == 0); sv = neg & (yV == 1)
    tp = (pred & pos).sum(); fp = (pred & neg).sum()
    print(f"{nm}: S+ {pos.sum()} Se {tp / max(1, pos.sum()):.3f} VPP {tp / max(1, tp + fp):.3f} falsas {fp} ({1000 * fp / max(1, neg.sum()):.1f}/1000) | en ventanas con V: {(pred & sv).sum()}/{sv.sum()} ({1000 * (pred & sv).sum() / max(1, sv.sum()):.1f}/1000)", flush=True)
for thr in (0.5, 0.3):
    base = pS > thr
    rep(f"RED sola (umbral {thr})", base)
    rep(f"RED + sin S en FA (pFA>0.5)", base & (pFA <= 0.5))
    for ci, ct in ((5, 0.85), (6, 0.90), (7, 0.95)):
        for pt in (0.85, 0.90, 0.95):
            rep(f"RED + sin FA + latido prematuro<{pt} con QRS habitual (corr>={ct})", base & (pFA <= 0.5) & (A[:, ci] < pt))
print("--- comparacion justa: red sola (sin S en FA) con el umbral que da la misma sensibilidad que cada combinacion")
pos = yS == 1; nofa = pFA <= 0.5
for ci, ct, pt in ((5, 0.85, 0.85), (5, 0.85, 0.95), (6, 0.90, 0.85), (7, 0.95, 0.85)):
    comb = (pS > 0.5) & nofa & (A[:, ci] < pt); se = (comb & pos).sum() / pos.sum()
    sc = np.where(nofa, pS, 0.0); thr = np.sort(sc[pos])[::-1][max(0, int(round(se * pos.sum())) - 1)]
    rep(f"COMBINADO prem<{pt} corr>={ct}", comb); rep(f"  RED sola a igual Se (umbral {thr:.2f})", sc >= thr)
# puntaje combinado: curva completa
for ci, ct, pt in ((5, 0.85, 0.95), (5, 0.85, 0.85)):
    for thr in (0.2, 0.1, 0.05):
        rep(f"COMBINADO umbral red {thr} prem<{pt} corr>={ct}", (pS > thr) & nofa & (A[:, ci] < pt))
for db in dbs:
    rep(f"[{db}] RED sola 0.5", pS > 0.5, D == db); rep(f"[{db}] RED + sin FA + prem<0.9 corr>=0.9", (pS > 0.5) & (pFA <= 0.5) & (A[:, 6] < 0.9), D == db)
print("TERMINADO"); import os; os._exit(0)
