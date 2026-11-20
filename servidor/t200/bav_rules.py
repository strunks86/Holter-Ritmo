"""Reglas de bloqueo AV 2o/3er grado, bloqueo SA y pausas a partir de las P del detector (p2) y de los QRS.
Mide en MIT-BIH completo (48 registros, 30 min): episodios detectados vs ritmo anotado (BII, BI, SBR...) y 'x' (P no conducida),
y falsas alarmas por hora en los registros sin bloqueo. Uso: bav_rules.py runs/p2/best.pt [umbral_P]"""
import sys, time, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn, wfdb
from scipy.signal import butter, sosfiltfilt, resample_poly, find_peaks
from fractions import Fraction
FS = 250; W = 4 * FS; M = Path("/root/holter/data/HolterIA/bases/mitdb")
class Net(nn.Module):
    def __init__(s, c=48):
        super().__init__()
        def blk(i, o, d): return nn.Sequential(nn.Conv1d(i, o, 7, padding=3 * d, dilation=d), nn.BatchNorm1d(o), nn.ReLU())
        s.f = nn.Sequential(blk(2, c, 1), blk(c, c, 2), blk(c, c, 4), blk(c, c, 8), blk(c, c, 16), blk(c, c, 1), nn.Conv1d(c, 1, 1))
    def forward(s, x): return s.f(x)[:, 0]
dev = torch.device("cuda:0"); nets = []
for pth in sys.argv[1].split("+"):
    n = Net().to(dev); n.load_state_dict(torch.load(pth, map_location=dev)); n.eval(); nets.append(n)
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
STRICT = len(sys.argv) > 3 and sys.argv[3] == "estricto"
def infer(x):
    o = np.zeros(len(x)); c = np.zeros(len(x))
    with torch.no_grad():
        for i in range(0, max(1, len(x) - W + 1), W // 2):
            seg = torch.tensor(x[i:i + W].T[None].copy()).to(dev)
            o[i:i + W] += sum(torch.sigmoid(n(seg))[0].cpu().numpy() for n in nets) / len(nets); c[i:i + W] += 1
    return o / np.maximum(c, 1)
def load(r):
    p = str(M / r); s, f = wfdb.rdsamp(p); fs = f["fs"]; s = np.nan_to_num(s)[:, :2]
    x = sosfiltfilt(butter(2, [0.5, 40], "bp", fs=fs, output="sos"), s, axis=0)
    fr = Fraction(FS, int(round(fs))).limit_denominator(100); x = resample_poly(x, fr.numerator, fr.denominator, axis=0).astype(np.float32)
    x /= np.percentile(np.abs(x), 99, axis=0) + 1e-6
    a = wfdb.rdann(p, "atr"); sy = np.array(a.symbol); smp = (a.sample * FS / fs).astype(int)
    isq = np.isin(sy, list("NLRBAaJSVEFejn/fQ")); Q = smp[isq]; QS = sy[isq]
    X = smp[sy == "x"]
    rit = []; cur = None; t0 = 0
    for s_, sm, au in zip(sy, smp, a.aux_note):
        if s_ == "+" and au.startswith("("):
            if cur is not None: rit.append((cur, t0, sm))
            cur = au.strip("(\x00 "); t0 = sm
    if cur is not None: rit.append((cur, t0, len(x)))
    return x, Q, QS, X, rit
def reglas(P, Q, QS, n, H=None, x=None):
    """Devuelve lista de eventos (tipo, muestra, detalle)."""
    ev = []
    if len(Q) < 10: return ev
    # P validas: fuera del QRS (de -40 ms a +100 ms del QRS)
    keep = np.ones(len(P), bool)
    for i, p in enumerate(P):
        j = np.searchsorted(Q, p); 
        for q in (Q[j - 1] if j > 0 else None, Q[j] if j < len(Q) else None):
            if q is not None and -0.04 * FS <= p - q <= 0.10 * FS: keep[i] = False
    P = P[keep]; H = H[keep] if H is not None else np.ones(len(P))
    if len(P) < 10: return ev
    # plantilla de P conducidas (seguidas de QRS a 120-300 ms)
    cond = [p for p in P if np.any((Q - p >= 0.10 * FS) & (Q - p <= 0.30 * FS)) and 30 <= p < n - 30]
    TP = np.median(np.stack([x[p - 25:p + 25].T.ravel() for p in cond[:500]]), 0) if len(cond) > 10 else None
    def corr(p):
        if TP is None or p < 25 or p >= n - 25: return 0.0
        v = x[p - 25:p + 25].T.ravel(); return float(np.corrcoef(v, TP)[0, 1])
    PP = np.diff(P); medPP = np.median(PP)
    # P bloqueada: sin QRS en los 450 ms siguientes y no prematura (PP >= 0.85 x mediana local)
    for i in range(1, len(P)):
        p = P[i]; j = np.searchsorted(Q, p)
        nxt = Q[j] if j < len(Q) else n
        if nxt - p <= 0.45 * FS: continue
        loc = PP[max(0, i - 8):i + 8]; m = np.median(loc)
        if PP[i - 1] < 0.85 * m: ev.append(("P_bloqueada_prematura", p, round(PP[i - 1] / m, 2))); continue
        if STRICT:
            if abs(PP[i - 1] / m - 1) > 0.15 or (i < len(PP) and abs(PP[i] / m - 1) > 0.15): continue
            if H[i] < 0.6 or corr(p) < 0.7: continue
        # PR antes y despues
        pr_b = None; pr_a = None
        if j > 0:
            prevp = P[(P < Q[j - 1]) & (P > Q[j - 1] - 0.4 * FS)]
            if len(prevp): pr_b = (Q[j - 1] - prevp[-1]) / FS
        if j < len(Q):
            np_ = P[(P < Q[j]) & (P > Q[j] - 0.4 * FS)]
            if len(np_): pr_a = (Q[j] - np_[-1]) / FS
        tipo = "BAV2"
        if pr_b and pr_a: tipo = "BAV2_Mobitz_I" if pr_b - pr_a > 0.04 else "BAV2_Mobitz_II_o_2:1"
        ev.append((tipo, p, (pr_b, pr_a)))
    # BAV 3: ventanas de 20 s con P regulares, mas P que QRS y PR que no se repite
    for a in range(0, n - 20 * FS, 10 * FS):
        Pw = P[(P >= a) & (P < a + 20 * FS)]; Qw = Q[(Q >= a) & (Q < a + 20 * FS)]
        if len(Pw) < 10 or len(Qw) < 6: continue
        if np.std(np.diff(Pw)) / np.mean(np.diff(Pw)) > 0.15 or len(Pw) < 1.3 * len(Qw): continue
        prs = []
        for q in Qw:
            pp = Pw[Pw < q]
            if len(pp): prs.append((q - pp[-1]) / FS)
        if len(prs) >= 5 and np.std(prs) > 0.10: ev.append(("BAV3", a, round(float(np.std(prs)), 3)))
    # pausas y bloqueo SA
    RR = np.diff(Q)
    for i, rr in enumerate(RR):
        if rr < 2.0 * FS: continue
        inP = P[(P > Q[i] + 0.1 * FS) & (P < Q[i + 1] - 0.1 * FS)]
        k = rr / medPP
        if len(inP) == 0 and 1.8 <= k <= 2.2: ev.append(("Bloqueo_SA_2", Q[i], round(rr / FS, 2)))
        elif len(inP) == 0: ev.append(("Pausa_sinusal", Q[i], round(rr / FS, 2)))
        else: ev.append(("Pausa_con_P", Q[i], round(rr / FS, 2)))
    return ev
recs = sorted(h.stem for h in M.glob("*.hea"))
if len(sys.argv) > 4: recs = sys.argv[4].split(",")
res = {}; tot = {}
for r in recs:
    try: x, Q, QS, X, rit = load(r)
    except Exception as e: print("omitido", r, e); continue
    pr = infer(x); P, pk = find_peaks(pr, height=THR, distance=int(0.12 * FS))
    ev = reglas(P, Q, QS, len(x), pk["peak_heights"], x); h = len(x) / FS / 3600
    # episodios: BAV2 a menos de 30 s entre si; aislado = episodio de 1 sola P bloqueada
    b2 = sorted(s_ for t, s_, d in ev if t.startswith("BAV2")); epi = []
    for s_ in b2:
        if epi and s_ - epi[-1][-1] < 30 * FS: epi[-1].append(s_)
        else: epi.append([s_])
    ev += [("EPISODIO_BAV2" + ("_aislado" if len(e) == 1 else ""), e[0], len(e)) for e in epi]
    cnt = {}
    for t, s_, d in ev: cnt[t] = cnt.get(t, 0) + 1
    rr = sorted({c for c, _, _ in rit})
    def en(tipo, rit_nm):
        e = [s_ for t, s_, d in ev if t.startswith(tipo)]
        z = [(a, b) for c, a, b in rit if c == rit_nm]
        return sum(any(a <= s_ < b for a, b in z) for s_ in e), len(e)
    bii = (en("BAV2", "BII"), en("EPISODIO_BAV2", "BII"))
    print(r, f"h={h:.2f}", "ritmos", rr, "x anotadas", len(X), "eventos", cnt, "BAV2 dentro de BII", bii, "episodios_min", [(round(e[0] / FS / 60, 2), round(e[-1] / FS / 60, 2), len(e)) for e in epi], flush=True)
    res[r] = {"ritmos": rr, "x": int(len(X)), "eventos": cnt, "bav2_en_BII": bii}
    for t, c in cnt.items(): tot[t] = tot.get(t, 0) + c
json.dump(res, open("/root/holter/t200/runs/bav_rules_mitdb" + ("_estricto" if STRICT else "") + ".json", "w"), default=str)
print("TOTAL", tot)
