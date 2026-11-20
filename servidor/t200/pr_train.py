"""PR largo / BAV de 1er grado: modelo aparte (1 salida), espalda de SOPA, ventanas PTB-XL de 10 s en ranuras II/V1/V5.
Positivos: 1AVB o LPR. Negativos: resto. Sin etiqueta: 2AVB, 3AVB, FA, flutter, marcapasos (PR no definido)."""
import sys, time, ast, csv, json, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent))
import t202_train as S
T, tt = S.T, S.tt
ap = argparse.ArgumentParser(); ap.add_argument("--init", default="/root/holter/t200/runs/t205b/sopa50.pt"); ap.add_argument("--minutes", type=float, default=40)
ap.add_argument("--name", default="pr1"); ap.add_argument("--nhdb", action="store_true"); ap.add_argument("--excluir_suj", default=""); a = ap.parse_args()
out = Path("/root/holter/t200/runs") / a.name; out.mkdir(parents=True, exist_ok=True)
def log(*x): print(time.strftime("[%H:%M:%S]"), *x, flush=True)
X = np.load("/root/holter/t158/data/windows_X.npy", mmap_mode="r"); m = np.load("/root/holter/t158/data/windows_meta.npz")
idx = np.where(m["db"] == "ptbxl")[0]; rec = m["rec"][idx]; split = m["split"][idx]
scp = {}
for r in csv.DictReader(open("/root/holter/data/HolterIA/bases/ptb-xl/ptbxl_database.csv")):
    scp[f"{int(r['ecg_id']):05d}_hr"] = (ast.literal_eval(r["scp_codes"]), r["pacemaker"].strip())
y = np.zeros(len(idx), np.float32)
for i, rr in enumerate(rec):
    c, pm = scp[rr]
    if any(k in c for k in ("2AVB", "3AVB", "AFIB", "AFLT", "PACE")) or pm: y[i] = -1
    elif any(k in c and c[k] >= 50 for k in ("1AVB", "LPR")): y[i] = 1
    elif any(k in c for k in ("1AVB", "LPR")): y[i] = -1  # dudoso (<50 % de certeza)
# fuentes: lista de (X, indice) para unificar PTB-XL y NHDB
SRC = [(X, int(j)) for j in idx]; db_ = np.array(["ptbxl"] * len(idx))
if a.nhdb:
    import glob, os
    Dn = glob.glob("/root/holter/data/HolterIA/bases2/noise-holter-database/noise*/wfdb_files")[0]; ev = {}
    for h in glob.glob(Dn + "/*.hea"):
        com = [x for x in open(h).read().split("\n") if x.startswith("#")]; parts = [q.strip() for q in com[0][1:].split("|")] if com else []
        k_ = next((k for k, q in enumerate(parts) if q.startswith("EVENTS")), None)
        ev[os.path.basename(h)[:-4]] = set([parts[k_].split(":", 1)[1].strip()] + parts[k_ + 1:]) if k_ is not None else set()
    Xn = np.load(T.NEW / "n_nhdb_X.npy", mmap_mode="r"); mn = np.load(T.NEW / "n_nhdb_meta.npz"); TQn = np.load(T.NEW / "n_nhdb_tq.npz")["TQ"]
    yn = np.full(len(Xn), -1.0, np.float32)
    for i, r in enumerate(mn["rec"]):
        e = ev.get(str(r), set()) - {""}
        if TQn[i, 3] == 1: continue
        if "AVB" in e and not e & {"A. Fib.", "A. Flutter"}: yn[i] = 1
        elif e <= {"None"}: yn[i] = 0
    exc = set(a.excluir_suj.split(",")) - {""}; spn = np.array([("excl" if str(r)[:3] in exc and sp_ == "train" else sp_) for r, sp_ in zip(mn["rec"], mn["split"])])
    SRC += [(Xn, i) for i in range(len(Xn))]; y = np.concatenate([y, yn]); split = np.concatenate([split, spn]); db_ = np.concatenate([db_, ["nhdb"] * len(Xn)])
for s in ("train", "val", "test"): log("nhdb" if a.nhdb else "", s, "pos nhdb", int(((y == 1) & (split == s) & (db_ == "nhdb")).sum()), "neg nhdb", int(((y == 0) & (split == s) & (db_ == "nhdb")).sum()))
for s in ("train", "val", "test"): log(s, "pos", int((y[split == s] == 1).sum()), "neg", int((y[split == s] == 0).sum()))
dev = torch.device("cuda:0")
net, _ = S.make_model7(dev, Path(a.init)); net.dense = nn.Linear(1024, 1).to(dev)
head = list(net.dense.parameters()); hid = {id(p) for p in head}; bb = [p for p in net.parameters() if id(p) not in hid]
opt = torch.optim.AdamW([{"params": bb, "lr": 1e-5}, {"params": head, "lr": 2e-4}], weight_decay=1e-5)
tr = np.where((split == "train") & (y >= 0))[0]; va = np.where((split == "val") & (y >= 0))[0]; te = np.where((split == "test") & (y >= 0))[0]
pw = torch.tensor([np.sqrt((y[tr] == 0).sum() / (y[tr] == 1).sum())], device=dev)
w = np.where(y[tr] == 1, 3.0, 1.0) * np.where(db_[tr] == 'nhdb', (db_[tr] == 'ptbxl').sum() / max(1, (db_[tr] == 'nhdb').sum()), 1.0); w /= w.sum()
rng = np.random.default_rng(0)
def get(ii, train):
    xb = np.stack([np.asarray(SRC[j][0][SRC[j][1]], dtype=np.float32).T for j in ii])
    if train:
        for k in range(len(xb)):
            if rng.random() < 0.5: xb[k] = S.perturbar(xb[k], rng)
            xb[k] *= 0.8 + 0.4 * rng.random()
    return torch.from_numpy(xb).to(dev)
def pred(ii):
    net.eval(); L = []
    with torch.no_grad():
        for k in range(0, len(ii), 256):
            with torch.autocast("cuda", dtype=torch.float16): L.append(net(tt.to12(get(ii[k:k + 256], False))).float().cpu().numpy()[:, 0])
    net.train(); return np.concatenate(L)
def mide(ii, nm):
    l = pred(ii); yy = y[ii]; o = np.argsort(-l); t = yy[o] == 1; tp = np.cumsum(t); ppv = tp / np.arange(1, len(o) + 1); se = tp / t.sum()
    r = {"AUROC": float(tt.auroc(yy, l)), "AP": float(ppv[t].mean())}
    for v in (0.8, 0.9):
        k = np.where(ppv >= v)[0]; r[f"Se@VPP{int(v * 100)}"] = float(se[k[-1]]) if len(k) else 0.0
    p = 1 / (1 + np.exp(-l)); r["Se@0.5"] = float((p[yy == 1] > 0.5).mean()); r["FP/1000@0.5"] = float(1000 * (p[yy == 0] > 0.5).mean())
    log(nm, {k: round(v, 3) for k, v in r.items()}); return r
t0 = time.time(); nxt = 300; best = -1; step = 0; net.train()
while time.time() - t0 < a.minutes * 60:
    fr = (time.time() - t0) / (a.minutes * 60)
    for g, b in zip(opt.param_groups, (1e-5, 2e-4)): g["lr"] = b * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * fr)))
    ii = rng.choice(tr, 64, p=w); xb = get(ii, True); yb = torch.from_numpy(y[ii]).to(dev)
    with torch.autocast("cuda", dtype=torch.float16): lo = net(tt.to12(xb)).float()[:, 0]
    loss = nn.functional.binary_cross_entropy_with_logits(lo, yb, pos_weight=pw); opt.zero_grad(); loss.backward(); opt.step(); step += 1
    if step % 300 == 0: log("paso", step, round(loss.item(), 4))
    if time.time() - t0 > nxt:
        nxt += 300; r = mide(va, f"VAL min={(time.time() - t0) / 60:.0f}")
        if a.nhdb: mide(va[db_[va] == "nhdb"], "   VAL solo nhdb")
        if r["AP"] > best: best = r["AP"]; torch.save({"state_dict": net.state_dict(), "targets": ["PR_LARGO"]}, out / "best.pt"); log("  nuevo mejor")
net.load_state_dict(torch.load(out / "best.pt", weights_only=False)["state_dict"]); r = mide(te[db_[te] == "ptbxl"], "TEST ptbxl")
if a.nhdb: r = {"ptbxl": r, "nhdb": mide(te[db_[te] == "nhdb"], "TEST nhdb (Holter real, sujetos no vistos)")}
json.dump(r, open(out / "test.json", "w")); log("TERMINADO")
