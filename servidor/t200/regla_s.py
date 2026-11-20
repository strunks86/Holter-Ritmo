"""Prototipo regla latido a latido: S = prematura + QRS igual a la plantilla normal del paciente (todos los canales).
Usa las posiciones R de referencia (mide la regla, no el detector). Sin entrenamiento."""
import sys, collections
from pathlib import Path
import numpy as np, wfdb
from scipy.signal import butter, sosfiltfilt
R = Path("/root/holter/data/HolterIA/bases")
RET_ALL = False
NORMAL = set("NLRejB"); SV = set("AaJS"); VE = set("VE"); OTROS = set("F/fQ!")
def feats(rec):
    s, f = wfdb.rdsamp(str(rec)); fs = f["fs"]; a = wfdb.rdann(str(rec), "atr")
    # ritmo de fondo: los latidos dentro de FA/flutter no se evaluan (criterio de Jorge: primero el ritmo)
    ritmo = ""; enfa = np.zeros(len(a.symbol), bool)
    for i, (c, x) in enumerate(zip(a.symbol, a.aux_note)):
        if c == "+" and x: ritmo = x.strip("\x00 ")
        enfa[i] = ritmo.startswith("(AF")
    keep = [i for i, c in enumerate(a.symbol) if c in NORMAL | SV | VE | OTROS]
    fa = enfa[keep]
    r = a.sample[keep]; sym = np.array([a.symbol[i] for i in keep])
    x = sosfiltfilt(butter(2, [1, 40], "bp", fs=fs, output="sos"), np.nan_to_num(s), axis=0)
    w0, w1 = int(0.08 * fs), int(0.10 * fs); ok = (r > w0 + 5) & (r < len(x) - w1 - 5); r, sym, fa = r[ok], sym[ok], fa[ok]
    B = np.stack([x[i - w0:i + w1] for i in r])  # (n, w, c)
    rr = np.diff(r, prepend=r[0]) / fs; rr[0] = np.median(rr)
    # RR de referencia: mediana movil de los 9 RR previos
    ref = np.array([np.median(rr[max(0, i - 9):i]) if i > 2 else np.median(rr) for i in range(len(rr))])
    prem = rr / np.maximum(ref, 1e-3)
    post = np.append(rr[1:], rr[-1]) / np.maximum(ref, 1e-3)
    # plantilla normal del paciente: mediana de los latidos no prematuros mas parecidos a la mediana global (2 pasadas), por tramos de 5 min
    cor = np.zeros((len(r), B.shape[2])); 
    seg = (r // int(300 * fs)).astype(int)
    for g in np.unique(seg):
        m = seg == g; Bg = B[m]; cand = Bg[prem[m] > 0.9] if (prem[m] > 0.9).sum() >= 5 else Bg
        t = np.median(cand, 0)
        for _ in range(2):
            c = np.array([[np.corrcoef(b[:, k], t[:, k])[0, 1] for k in range(B.shape[2])] for b in cand]).min(1)
            t = np.median(cand[c >= np.quantile(c, 0.5)], 0)
        tn = t - t.mean(0); bn = Bg - Bg.mean(1, keepdims=True)
        num = (bn * tn).sum(1); den = np.sqrt((bn ** 2).sum(1) * (tn ** 2).sum(0)) + 1e-9
        cor[m] = num / den
        amp = np.abs(bn).max(1) / (np.abs(tn).max(0) + 1e-9)
        cor[m] = np.where((amp > 0.6) & (amp < 1.6), cor[m], np.minimum(cor[m], 0.5))
    if RET_ALL: return r / fs, sym, prem, np.nan_to_num(cor).min(1), fa
    return sym[~fa], prem[~fa], post[~fa], np.nan_to_num(cor).min(1)[~fa]
def main():
    tot = collections.defaultdict(lambda: collections.Counter())
    for db in sys.argv[1].split(","):
        for h in sorted((R / db).rglob("*.hea")):
            try: sym, prem, post, cmin = feats(h.with_suffix(""))
            except Exception as e: print("omitido", h.stem, e); continue
            cls = np.where(np.isin(sym, list(SV)), "S", np.where(np.isin(sym, list(VE)), "V", np.where(np.isin(sym, list(NORMAL)), "N", "O")))
            pv = (prem < 0.85) & (cmin >= 0.95) & (cls == "V")
            if pv.sum() >= 20: print("  registro", h.stem, "V marcadas S", int(pv.sum()), "de V", int((cls == "V").sum()), "N", int((cls == "N").sum()), flush=True)
            pn = (prem < 0.85) & (cmin >= 0.95) & (cls == "N")
            if pn.sum() >= 150: print("  registro", h.stem, "N marcados S", int(pn.sum()), "de N", int((cls == "N").sum()), flush=True)
            for pt in (0.80, 0.85, 0.90):
                for ct in (0.90, 0.95, 0.97):
                    pred = (prem < pt) & (cmin >= ct)
                    for c in "SVNO": tot[(db, pt, ct)][c + "_marc"] += int((pred & (cls == c)).sum()); tot[(db, pt, ct)][c] += int((cls == c).sum())
        for (d, pt, ct), c in tot.items():
            if d != db: continue
            fp = c["V_marc"] + c["N_marc"] + c["O_marc"]
            print(f"{db} prem<{pt} corr>={ct}: S {c['S']} Se {c['S_marc'] / max(1, c['S']):.3f} VPP {c['S_marc'] / max(1, c['S_marc'] + fp):.3f} | V marcadas S {c['V_marc']}/{c['V']} ({1000 * c['V_marc'] / max(1, c['V']):.1f}/1000) | N marcados S {c['N_marc']}/{c['N']} ({1000 * c['N_marc'] / max(1, c['N']):.2f}/1000)", flush=True)
    print("TERMINADO")


if __name__ == '__main__': main()
