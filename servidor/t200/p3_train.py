"""Detector de onda P (segmentacion): mapa de calor de picos P a 250 Hz sobre ventanas de 4 s, 2 canales (pwave + BUT PDB + LUDB; p2: validacion separada y peso 5 a las P bloqueadas).
Particion por registro de origen (mitdb/svdb/ltafdb). Mide Se/VPP de P con tolerancia 50 ms en registros de prueba
y, en los registros de prueba, el recuento de P por cada QRS (base para bloqueo AV)."""
import sys, time, json, re
from pathlib import Path
import numpy as np, torch, torch.nn as nn, wfdb
from scipy.signal import butter, sosfiltfilt, resample_poly, find_peaks
from fractions import Fraction
B = Path("/root/holter/data/HolterIA/bases2"); FS = 250; W = 4 * FS
NAME = sys.argv[2] if len(sys.argv) > 2 else "p2"; out = Path("/root/holter/t200/runs") / NAME; out.mkdir(parents=True, exist_ok=True)
def log(*x): print(time.strftime("[%H:%M:%S]"), *x, flush=True)
orig = {}
for line in open(B / "but-pdb/README.txt"):
    m = re.match(r"\s*(\d+)\s+'([^']*)'\s+(.+?Database)\s+(\d+)\s+\[", line)
    if m: dbn = m.group(3); pre = "MIT-BIH" if "Arrhythmia" in dbn else ("SVDB" if "Supraventricular" in dbn else "LongTerm"); orig[f"{int(m.group(1)):02d}"] = (m.group(2), pre + "_" + m.group(4))
recs = []
for h in sorted((B / "but-pdb").glob("*.hea")):
    if h.stem in orig: recs.append(("but-pdb", h.stem, orig[h.stem][1], orig[h.stem][0]))
for h in sorted((B / "pwave").glob("*.hea")): recs.append(("pwave", h.stem, "MIT-BIH_" + h.stem, ""))
LU = sorted(B.glob("ludb_x/*/data"))[0]
for h in sorted(LU.glob("*.hea"), key=lambda z: int(z.stem)): recs.append(("ludb", h.stem, "LUDB_" + h.stem, ""))
TEST = {"MIT-BIH_231", "MIT-BIH_111", "MIT-BIH_124", "MIT-BIH_208", "MIT-BIH_119", "MIT-BIH_222", "MIT-BIH_203", "MIT-BIH_101", "SVDB_801", "LongTerm_120"}
def load(db, r):
    p = str((LU if db == "ludb" else B / db) / r); s, f = wfdb.rdsamp(p); fs = f["fs"]; s = np.nan_to_num(s)
    if db == "ludb":
        nm = [z.lower() for z in f["sig_name"]]; s = s[:, [nm.index("ii"), nm.index("v1")]]
    else: s = s[:, :2]
    if s.shape[1] < 2: s = np.concatenate([s, np.zeros_like(s)], 1)
    x = sosfiltfilt(butter(2, [0.5, 40], "bp", fs=fs, output="sos"), s, axis=0)
    fr = Fraction(FS, int(round(fs))).limit_denominator(100); x = resample_poly(x, fr.numerator, fr.denominator, axis=0).astype(np.float32)
    x /= np.percentile(np.abs(x), 99, axis=0) + 1e-6
    if db == "ludb":
        a = wfdb.rdann(p, "ii"); sy = np.array(a.symbol); P = (a.sample[sy == "p"] * FS / fs).astype(int); Q = (a.sample[sy == "N"] * FS / fs).astype(int)
        return x, P, Q
    pa = wfdb.rdann(p, "pwave"); P = (pa.sample * FS / fs).astype(int)
    try:
        q = wfdb.rdann(p, "qrs" if db == "but-pdb" else "atr"); Q = (q.sample[[c in "NLRBAaJSVEFejn/fQ" for c in q.symbol]] * FS / fs).astype(int)
    except Exception: Q = np.array([], int)
    return x, P, Q
data = {}
for db, r, o, pat in recs:
    try: data[(db, r)] = load(db, r) + (o, pat)
    except Exception as e: log("omitido", db, r, e)
log("registros", len(data), "P anotadas", sum(len(v[1]) for v in data.values()))
VAL = {"MIT-BIH_100", "MIT-BIH_105", "MIT-BIH_116", "MIT-BIH_213", "MIT-BIH_232", "SVDB_820"} | {f"LUDB_{i}" for i in range(181, 201)}
tr = [k for k, v in data.items() if v[3] not in TEST | VAL]; va = [k for k, v in data.items() if v[3] in VAL]; te = [k for k, v in data.items() if v[3] in TEST]
log("val", len(va), [k for k in va if k[0] != "ludb"])
log("train", len(tr), "test", len(te), [k for k in te])
def target(n, P):
    t = np.zeros(n, np.float32); g = np.exp(-0.5 * (np.arange(-15, 16) / 5.0) ** 2)
    for p in P:
        a, b = max(0, p - 15), min(n, p + 16); t[a:b] = np.maximum(t[a:b], g[a - p + 15:b - p + 15])
    return t
T = {k: target(len(v[0]), v[1]) for k, v in data.items()}
def wmap(n, P, Q):
    w = np.ones(n, np.float32); nb = 0
    for p in P:
        if len(Q) and not np.any((Q > p) & (Q <= p + int(0.45 * FS))): w[max(0, p - 20):p + 21] = 5.0; nb += 1
    return w, nb
WM = {}; NB = 0
for k, v in data.items():
    WM[k], nb = wmap(len(v[0]), v[1], v[2]); NB += nb
log("P bloqueadas (sin QRS en 450 ms) con peso 5:", NB)
# zonas anotadas: entre la primera y la ultima P (pwave anota solo partes del registro)
Z = {k: (max(0, v[1].min() - 2 * FS), min(len(v[0]), v[1].max() + 2 * FS)) for k, v in data.items() if len(v[1])}
class Net(nn.Module):
    def __init__(s, c=48):
        super().__init__()
        def blk(i, o, d): return nn.Sequential(nn.Conv1d(i, o, 7, padding=3 * d, dilation=d), nn.BatchNorm1d(o), nn.ReLU())
        s.f = nn.Sequential(blk(2, c, 1), blk(c, c, 2), blk(c, c, 4), blk(c, c, 8), blk(c, c, 16), blk(c, c, 1), nn.Conv1d(c, 1, 1))
    def forward(s, x): return s.f(x)[:, 0]
dev = torch.device("cuda:0"); net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
rng = np.random.default_rng(0); trk = [k for k in tr if k in Z]
trL = [k for k in trk if k[0] == "ludb"]; trO = [k for k in trk if k[0] != "ludb"]
TPL = {}
for k in trk:
    x, P = data[k][0], data[k][1]; P = P[(P > 40) & (P < len(x) - 40)]
    if len(P) < 5: continue
    t = np.median(np.stack([x[p - 30:p + 30] for p in P[:400]]), 0)
    t = t - np.linspace(t[0], t[-1], 60); TPL[k] = (t * np.hanning(60)[:, None]).T.astype(np.float32)
SYN = float(__import__("os").environ.get("PSYN", "0.5")); PLU = float(__import__("os").environ.get("PLUDB", "0.2"))
log("plantillas P", len(TPL), "prob sinteticas", SYN, "prob LUDB", PLU)
def batch(n=64):
    X, Y, Wt = [], [], []
    for _ in range(n):
        L = trL if rng.random() < PLU else trO; k = L[rng.integers(len(L))]; a, b = Z[k]; i = rng.integers(a, max(a + 1, b - W))
        x = data[k][0][i:i + W].T.copy(); y = T[k][i:i + W].copy(); wt = WM[k][i:i + W].copy()
        if k in TPL and rng.random() < SYN and x.shape[1] == W:
            Qw = data[k][2]; Qw = Qw[(Qw >= i) & (Qw < i + W)] - i; Pw = data[k][1]; Pw = Pw[(Pw >= i) & (Pw < i + W)] - i
            for _ in range(rng.integers(1, 3)):
                c = int(rng.integers(40, W - 40))
                if (len(Qw) and np.min(np.abs(Qw - c)) < 25) or (len(Pw) and np.min(np.abs(Pw - c)) < 40): continue
                x[:, c - 30:c + 30] += TPL[k] * (0.7 + 0.6 * rng.random()); g = target(W, np.array([c])); y = np.maximum(y, g); wt[c - 20:c + 21] = 3.0
        if x.shape[1] < W: continue
        if rng.random() < 0.5: x = x[::-1].copy()
        if x.shape[1] < W: continue
        x *= np.where(rng.random((2, 1)) < 0.2, -1, 1) * (0.7 + 0.6 * rng.random((2, 1)))
        if rng.random() < 0.15: x[rng.integers(2)] = 0
        x += 0.03 * rng.standard_normal(x.shape) * (rng.random() < 0.5)
        X.append(x); Y.append(y); Wt.append(wt)
    return torch.tensor(np.array(X)).to(dev), torch.tensor(np.array(Y)).to(dev), torch.tensor(np.array(Wt)).to(dev)
NETS = []
def infer(x):
    nets = NETS or [net]
    for m in nets: m.eval()
    o = np.zeros(len(x)); c = np.zeros(len(x))
    with torch.no_grad():
        for i in range(0, max(1, len(x) - W + 1), W // 2):
            seg = torch.tensor(x[i:i + W].T[None]).to(dev); o[i:i + W] += sum(torch.sigmoid(m(seg))[0].cpu().numpy() for m in nets) / len(nets); c[i:i + W] += 1
    net.train(); return o / np.maximum(c, 1)
def evalua(keys, nm, thr=0.4):
    tp = fp = fn = 0; err = []; ratio = []; btp = bn = 0
    for k in keys:
        if k not in Z: continue
        x, P, Q, o, pat = data[k]; a, b = Z[k]; pr = infer(x[a:b]); pk, _ = find_peaks(pr, height=thr, distance=int(0.12 * FS)); pk = pk + a
        Pi = P[(P >= a) & (P < b)]; used = np.zeros(len(pk), bool)
        for p in Pi:
            d = np.abs(pk - p); j = int(np.argmin(d)) if len(pk) else -1
            bl = len(Q) and not np.any((Q > p) & (Q <= p + int(0.45 * FS))); bn += bool(bl)
            if j >= 0 and d[j] <= 0.05 * FS and not used[j]: tp += 1; used[j] = True; err.append((pk[j] - p) / FS * 1000); btp += bool(bl)
            else: fn += 1
        fp += int((~used).sum())
        Qi = Q[(Q >= a) & (Q < b)]
        if len(Qi) > 5: ratio.append((k[1], pat, round(len(Pi) / len(Qi), 2), round(len(pk) / len(Qi), 2)))
    r = {"Se": tp / max(1, tp + fn), "VPP": tp / max(1, tp + fp), "error_ms_mediana": float(np.median(np.abs(err))) if err else None, "Se_P_bloqueadas": btp / max(1, bn), "n_bloqueadas": bn}
    log(nm, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}); return r, ratio
t0 = time.time(); best = -1; nxt = 300; step = 0; MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 40
while time.time() - t0 < MIN * 60:
    for g in opt.param_groups: g["lr"] = 1e-3 * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * (time.time() - t0) / (MIN * 60))))
    xb, yb, wb = batch(); lo = net(xb); loss = (nn.functional.binary_cross_entropy_with_logits(lo, yb, pos_weight=torch.tensor(5.0, device=dev), reduction="none") * wb).mean()
    opt.zero_grad(); loss.backward(); opt.step(); step += 1
    if step % 500 == 0: log("paso", step, round(loss.item(), 4))
    if time.time() - t0 > nxt:
        nxt += 300; r, _ = evalua(va, f"VAL min={(time.time() - t0) / 60:.0f}")
        f1 = 2 * r["Se"] * r["VPP"] / max(1e-6, r["Se"] + r["VPP"])
        if f1 > best: best = f1; torch.save(net.state_dict(), out / "best.pt"); log("  nuevo mejor F1", round(f1, 3))
if "+" in NAME:
    for nm in NAME.split("+"):
        m = Net().to(dev); m.load_state_dict(torch.load(out.parent / nm / "best.pt")); NETS.append(m)
else: net.load_state_dict(torch.load(out / "best.pt"))
for thr in (0.3, 0.4, 0.5): evalua(te, f"FINAL umbral {thr}", thr)
for thr in (0.3, 0.4, 0.5): evalua(va, f"VAL FINAL umbral {thr}", thr)
r, ratio = evalua(te, "FINAL 0.4 con P/QRS")
for x in ratio: log("  P por QRS (registro, patologia, anotado, detectado):", x)
json.dump(r, open(out / "test.json", "w")); log("TERMINADO")
