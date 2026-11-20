"""Clasificador latido a latido N/S/V con referencia del paciente (QRS + plantilla habitual + RR)."""
import sys, time, argparse, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
D = Path("/root/holter/t200/data")
ap = argparse.ArgumentParser(); ap.add_argument("--name", default="lat1"); ap.add_argument("--minutes", type=float, default=40)
ap.add_argument("--sin_plantilla", action="store_true"); ap.add_argument("--bs", type=int, default=512); ap.add_argument("--lr", type=float, default=1e-3)
a = ap.parse_args()
out = Path("/root/holter/t200/runs") / a.name; out.mkdir(parents=True, exist_ok=True)
def log(*x): print(time.strftime("[%H:%M:%S]"), *x, flush=True)
X = np.load(D / "lat_X.npy", mmap_mode="r"); m = np.load(D / "lat_meta.npz"); F, y, split, db = m["F"], m["y"].astype(np.int64), m["split"], m["db"]
F = F.copy(); F[:, :2] = np.clip(F[:, :2], 0, 3); F[:, 2] = np.clip(F[:, 2], 0.2, 2.5); F[:, 3] = np.clip(F[:, 3], 0, 1); F[:, 5] /= 3
ok = y >= 0; tr = np.where(ok & (split == "train"))[0]; va = np.where(ok & (split == "val"))[0]; te = np.where(ok & (split == "test"))[0]
log("train", len(tr), np.bincount(y[tr]).tolist(), "val", len(va), np.bincount(y[va]).tolist(), "test", len(te), np.bincount(y[te]).tolist())
dev = torch.device("cuda:0")
class Blk(nn.Module):
    def __init__(s, c, k=7): super().__init__(); s.a = nn.Conv1d(c, c, k, padding=k // 2); s.b = nn.Conv1d(c, c, k, padding=k // 2); s.n1 = nn.BatchNorm1d(c); s.n2 = nn.BatchNorm1d(c)
    def forward(s, x): return torch.relu(x + s.n2(s.b(torch.relu(s.n1(s.a(x))))))
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.cnn = nn.Sequential(nn.Conv1d(9, 48, 9, padding=4), nn.BatchNorm1d(48), nn.ReLU(), Blk(48), nn.MaxPool1d(2), nn.Conv1d(48, 96, 7, padding=3), nn.BatchNorm1d(96), nn.ReLU(), Blk(96), nn.MaxPool1d(2), nn.Conv1d(96, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), Blk(128, 5))
        s.f = nn.Sequential(nn.Linear(6, 32), nn.ReLU()); s.h = nn.Sequential(nn.Linear(128 * 2 + 32, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 3))
    def forward(s, x, f):
        x = torch.cat([x, x[:, :3] - x[:, 3:]], 1)  # latido, plantilla y diferencia
        z = s.cnn(x); return s.h(torch.cat([z.mean(2), z.amax(2), s.f(f)], 1))
net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
Ft = torch.tensor(F); yt = torch.tensor(y)
def batch(idx, train):
    idx = np.sort(idx); x = torch.from_numpy(np.asarray(X[idx], dtype=np.float32)); f = Ft[idx].clone()
    if a.sin_plantilla: x[:, 3:] = 0
    if train:
        B = len(idx)
        # robustez de electrodos: permutar ranuras (igual en latido y plantilla), invertir polaridad, apagar canales, ganancia
        for i in range(B):
            if np.random.rand() < 0.5:
                p = np.random.permutation(3); x[i] = torch.cat([x[i, :3][p], x[i, 3:][p]])
        sg = torch.where(torch.rand(B, 3, 1) < 0.3, -1.0, 1.0) * (0.7 + 0.6 * torch.rand(B, 3, 1)); x = x * torch.cat([sg, sg], 1)
        x = x + 0.03 * torch.randn_like(x) * (torch.rand(B, 1, 1) < 0.5)
    return x.to(dev), f.to(dev), yt[idx].to(dev)
# muestreo equilibrado por clase y base
w = np.zeros(len(tr)); 
for c in range(3):
    for d in set(db[tr]):
        mm = (y[tr] == c) & (db[tr] == d)
        if mm.any(): w[mm] = 1.0 / np.sqrt(mm.sum())
for c, k in ((0, 1.0), (1, 1.0), (2, 1.0)): w[y[tr] == c] *= k / w[y[tr] == c].sum()
w /= w.sum()
def evalua(idx, nm):
    net.eval(); P = []
    with torch.no_grad():
        for i in range(0, len(idx), 4096):
            x, f, _ = batch(idx[i:i + 4096], False); P.append(torch.softmax(net(x, f), 1).cpu().numpy())
    P = np.concatenate(P); yy = y[np.sort(idx)]; net.train(); r = {}
    pr = P.argmax(1)
    for c, n in ((1, "S"), (2, "V")):
        tp = ((pr == c) & (yy == c)).sum(); r[n + "_Se"] = tp / max(1, (yy == c).sum()); r[n + "_VPP"] = tp / max(1, (pr == c).sum())
    r["V_marcada_S_por_1000"] = 1000 * ((pr == 1) & (yy == 2)).sum() / max(1, (yy == 2).sum()); r["N_marcado_S_por_1000"] = 1000 * ((pr == 1) & (yy == 0)).sum() / max(1, (yy == 0).sum())
    # S a precision fija
    o = np.argsort(-P[:, 1]); t = (yy[o] == 1); tp = np.cumsum(t); ppv = tp / np.arange(1, len(o) + 1); se = tp / max(1, t.sum())
    for v in (0.90, 0.95):
        k = np.where(ppv >= v)[0]; r[f"S_Se@VPP{int(v * 100)}"] = float(se[k[-1]]) if len(k) else 0.0
    r["S_AP"] = float(ppv[t].mean())
    log(nm, {k: round(float(v), 3) for k, v in r.items()}); return r, P, yy
t0 = time.time(); step = 0; best = -1; nxt = 5 * 60
crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 1.0], device=dev))
vs = np.sort(np.random.default_rng(0).choice(va, min(len(va), 300000), replace=False))
while time.time() - t0 < a.minutes * 60:
    idx = np.random.choice(tr, a.bs, p=w); x, f, yb = batch(idx, True)
    loss = crit(net(x, f), yb); opt.zero_grad(); loss.backward(); opt.step(); step += 1
    if step % 500 == 0: log("paso", step, "loss", round(loss.item(), 4))
    if time.time() - t0 > nxt:
        nxt += 5 * 60; r, _, _ = evalua(vs, f"VAL min={(time.time() - t0) / 60:.0f}")
        if r["S_AP"] > best: best = r["S_AP"]; torch.save(net.state_dict(), out / "best.pt"); log("  nuevo mejor")
net.load_state_dict(torch.load(out / "best.pt")); res = {}
r, P, yy = evalua(te, "TEST total"); res["total"] = r
for d in sorted(set(db[te])):
    r, _, _ = evalua(te[db[te] == d], f"TEST {d}"); res[d] = r
json.dump({k: {a_: float(b) for a_, b in v.items()} for k, v in res.items()}, open(out / "test.json", "w"), indent=1)
log("TERMINADO")
