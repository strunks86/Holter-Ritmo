"""Datos latido a latido con referencia del paciente: cada latido = su QRS (3 ranuras) + plantilla habitual local + RR.
Etiquetas N/S/V de las anotaciones; latidos dentro de FA anotada: S pasa a N (regla: no hay S en FA)."""
import sys, collections
from pathlib import Path
import numpy as np, wfdb
from scipy.signal import butter, sosfiltfilt, resample_poly
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor
import os
R = Path("/root/holter/data/HolterIA"); OUT = Path("/root/holter/t200/data")
SUF = os.environ.get("LATSUF", "")  # "_r2": referencia RR = mediana de promedios de pares de RR (robusta a bigeminismo)
FS = 250; W0, W1 = int(0.30 * FS), int(0.40 * FS)  # 175 muestras
TODOS = False
NORMAL = set("NLRejB"); SV = set("AaJS"); VE = set("VE")
def one(args):
    db, path, ext, split = args
    try:
        s, f = wfdb.rdsamp(path); a = wfdb.rdann(path, ext)
    except Exception as e: return None
    fs = f["fs"]; s = np.nan_to_num(s)[:, :3]
    fr = Fraction(FS, int(round(fs))).limit_denominator(100)
    x = sosfiltfilt(butter(2, [0.7, 40], "bp", fs=fs, output="sos"), s, axis=0)
    x = resample_poly(x, fr.numerator, fr.denominator, axis=0).astype(np.float32)
    ritmo = ""; enfa = np.zeros(len(a.symbol), bool)
    for i, (c, n) in enumerate(zip(a.symbol, a.aux_note)):
        if c == "+" and n: ritmo = n.strip("\x00 ")
        enfa[i] = ritmo.startswith("(AF")
    keep = [i for i, c in enumerate(a.symbol) if c in NORMAL | SV | VE | set("F/fQ!")]
    if len(keep) < 50: return None
    r = (a.sample[keep] * FS / fs).astype(int); sym = np.array([a.symbol[i] for i in keep]); fa = enfa[keep]
    ok = (r > W0 + 2) & (r < len(x) - W1 - 2); r, sym, fa = r[ok], sym[ok], fa[ok]
    sc = np.percentile(np.abs(x), 99, axis=0) + 1e-6; x = x / sc
    B = np.stack([x[i - W0:i + W1] for i in r])  # n, 175, c
    rr = np.diff(r, prepend=r[0]) / FS; rr[0] = np.median(rr)
    if SUF == "_r2":
        pa = np.concatenate([[rr[0]], (rr[1:] + rr[:-1]) / 2])
        ref = np.array([np.median(pa[max(1, i - 9):i]) if i > 3 else np.median(rr) for i in range(len(rr))])
    else:
        ref = np.array([np.median(rr[max(0, i - 9):i]) if i > 2 else np.median(rr) for i in range(len(rr))])
    prem = rr / np.maximum(ref, 1e-3); post = np.append(rr[1:], rr[-1]) / np.maximum(ref, 1e-3)
    cv = np.array([np.std(rr[max(0, i - 9):i + 1]) / (np.mean(rr[max(0, i - 9):i + 1]) + 1e-6) for i in range(len(rr))])
    # plantilla habitual por tramos de 5 min: mediana de latidos no prematuros, refinada con los mas parecidos (no usa etiquetas)
    Tm = np.zeros_like(B); seg = r // (300 * FS)
    for g in np.unique(seg):
        m = np.where(seg == g)[0]; c = m[prem[m] > 0.9] if (prem[m] > 0.9).sum() >= 5 else m
        t = np.median(B[c], 0)
        for _ in range(2):
            d = ((B[c] - t) ** 2).sum((1, 2)); t = np.median(B[c[d <= np.quantile(d, 0.5)]], 0)
        Tm[m] = t
    y = np.where(np.isin(sym, list(SV)), 1, np.where(np.isin(sym, list(VE)), 2, np.where(np.isin(sym, list(NORMAL)), 0, -1))).astype(np.int8)
    y[(y == 1) & fa] = 0
    nc = B.shape[2]; X = np.zeros((len(r), 6, W0 + W1), np.float16)
    X[:, :nc] = B.transpose(0, 2, 1); X[:, 3:3 + nc] = Tm.transpose(0, 2, 1)
    F = np.stack([prem, post, ref, cv, fa.astype(float), np.full(len(r), nc)], 1).astype(np.float32)
    # submuestreo de normales: todos los S/V, y N vecinos de ectopicos + 1 de cada 6
    rng = np.random.default_rng(abs(hash(path)) % 2**31); nb = np.zeros(len(r), bool); e = np.where(y > 0)[0]
    for d in (-2, -1, 1, 2): nb[np.clip(e + d, 0, len(r) - 1)] = True
    kp = (y > 0) | ((y == 0) & (nb | (rng.random(len(r)) < 1 / 6)))
    if TODOS: return db, Path(path).name, split, X, F, y, r / FS
    return db, Path(path).name, split, X[kp], F[kp], y[kp]
def main():
    hm = np.load(OUT / "holter_meta.npz"); sp = {(d, r): s for d, r, s in zip(hm["db"], hm["rec"], hm["split"])}
    jobs = []
    for db in ("mitdb", "svdb", "incartdb", "edb", "ltdb", "sddb"):
        for h in sorted((R / "bases" / db).rglob("*.atr")):
            s = sp.get((db, h.stem))
            if s: jobs.append((db, str(h)[:-4], "atr", s))
    lm = np.load(OUT / "n_ltstdb_meta.npz"); sp2 = {r: s for r, s in zip(lm["rec"], lm["split"])}
    for h in sorted((R / "bases2" / "ltstdb").rglob("*.atr")):
        if h.stem in sp2: jobs.append(("ltstdb", str(h)[:-4], "atr", sp2[h.stem]))
    print("registros", len(jobs), collections.Counter(j[3] for j in jobs), flush=True)
    Xs, Fs, Ys, Ds, Rs, Ss = [], [], [], [], [], []
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as ex:
        for k, o in enumerate(ex.map(one, jobs)):
            if o is None: continue
            db, rec, split, X, F, y = o; n = len(y)
            Xs.append(X); Fs.append(F); Ys.append(y); Ds += [db] * n; Rs += [rec] * n; Ss += [split] * n
            if k % 20 == 0: print(k, len(jobs), db, rec, n, flush=True)
    X = np.concatenate(Xs); np.save(OUT / f"lat_X{SUF}.npy", X)
    y = np.concatenate(Ys); S = np.array(Ss); D = np.array(Ds)
    np.savez(OUT / f"lat_meta{SUF}.npz", F=np.concatenate(Fs), y=y, db=D, rec=np.array(Rs), split=S)
    for s in ("train", "val", "test"):
        for d in sorted(set(D)):
            m = (S == s) & (D == d)
            if m.any(): print(s, d, "N", int((y[m] == 0).sum()), "S", int((y[m] == 1).sum()), "V", int((y[m] == 2).sum()), flush=True)
    print("LISTO", X.shape)
if __name__ == "__main__": main()
