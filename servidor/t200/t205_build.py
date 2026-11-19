"""T205: ventanas de las bases nuevas (19-nov). Un tag por grupo de bases:
  n_leipzig  Leipzig Heart Center (AVNRT/AVRT/TA ectopica/flutter/TV latido a latido, 12 deriv.)
  n_vf       MIT-BIH Malignant Ventricular (vfdb) + Creighton (cudb): TV/FV
  n_nhdb     Noise Holter Database (calidad Clean/N1/N2 + latidos + eventos)
  n_cpsc20   CPSC2020 (10 Holter 24 h, S y V)
  n_butqdb   BUT QDB (calidad 1/2/3)
  n_chapman  Chapman-Ningbo (ECG 10 s: flutter, TA, AVNRT, AVRT, TSV)
Salida: <tag>_X.npy, <tag>_meta.npz (Y = FA, FLUT, PVC, PAC) y <tag>_tq.npz con
TQ (N,4) = [TV, TSV, taquicardia_sinusal, RUIDO_real].  -1 = sin etiqueta.
Particion por paciente/registro (70/15/15, semilla 200), igual que el resto.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import re
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t200_build as B  # noqa: E402

tb = B.tb
FS, WIN, WIN_S = B.FS, B.WIN, B.WIN_S
B2 = Path(os.environ.get("BASES2", "/root/holter/data/HolterIA/bases2"))
OUT = B.OUT


def cut(db, rec, split, sig, fs0, slots, wins, toff=0.0):
    """sig (n, C) crudo; wins = [(t0 relativo, y4, tq4)]."""
    sig = np.nan_to_num(np.asarray(sig, dtype=np.float64))
    raw = np.zeros((sig.shape[0], 3))
    used = []
    for j, c in enumerate(slots):
        if c >= 0:
            raw[:, j] = sig[:, c]
            used.append(j)
    x = tb.prep_record(raw, fs0)
    out = []
    for t0, y, q in wins:
        i0 = int(round(t0 * FS))
        w = x[i0:i0 + WIN]
        if w.shape[0] != WIN or np.abs(w).max() == 0:
            continue
        out.append((db, rec, split, float(toff + t0), tb.zwin(w, used).astype(np.float16),
                    np.array(y, np.float32), np.array(q, np.float32)))
    return out


def tri(n, k=3):
    return 1.0 if n >= k else (0.0 if n == 0 else -1.0)


# ---------------------------------------------------------------- Leipzig
LZ_X = {"AVRT": "SVT", "AVNRT": "SVT", "AVNRT+BII": "SVT", "EAT": "SVT", "VT": "VT",
        "AFIB": "AF", "AFL": "AFL", "IVR": "IVR"}
LZ_S = {"/": "PACE", "V": "PVC", "F": "PVC", "f": "PVC", "A": "PAC", "a": "PAC", "J": "PAC", "j": "PAC",
        "N": "N", "R": "N", "L": "N", "b": "N", "Q": "Q", "~": "RUIDO"}


def lz_label(kinds):
    c = collections.Counter(kinds)
    nb = sum(v for k, v in c.items() if k != "RUIDO")
    if nb == 0:
        return None
    fa, fl = tri(c["AF"]), tri(c["AFL"])
    tsv, tv = tri(c["SVT"]), tri(c["VT"])
    if c["IVR"] and tv != 1:
        tv = -1.0
    if c["PACE"]:
        if tsv != 1:
            tsv = -1.0
        if tv != 1:
            tv = -1.0
    pvc = 1.0 if c["PVC"] else (-1.0 if (c["PACE"] or c["VT"] or c["IVR"]) else 0.0)
    pac = 1.0 if c["PAC"] else (-1.0 if (c["PACE"] or c["SVT"] or c["AF"] or c["AFL"]) else 0.0)
    ru = -1.0 if c["RUIDO"] else 0.0
    return [fa, fl, pvc, pac], [tv, tsv, 0.0, ru]


def job_leipzig(item):
    path, rec, split = item["path"], item["rec"], item["split"]
    hdr = wfdb.rdheader(path)
    ann = wfdb.rdann(path, "atr")
    fs0 = float(hdr.fs)
    names = [n.upper() for n in hdr.sig_name]
    chans = [names.index(n) for n in ("II", "V1", "V5")]
    bt, bk = [], []
    for s, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note, strict=False):
        a = (aux or "").strip("\x00 ").upper()
        k = LZ_X.get(a) if sym == "X" else LZ_S.get(sym)
        if k:
            bt.append(s / fs0)
            bk.append(k)
    bt = np.array(bt)
    n_sec = hdr.sig_len / fs0
    out = []
    CH = 1200
    for c0 in range(0, int(n_sec), CH):
        a = int(c0 * fs0)
        b = int(min(hdr.sig_len, (c0 + CH + WIN_S + 2) * fs0))
        sig = wfdb.rdrecord(path, sampfrom=a, sampto=b, channels=chans).p_signal
        wins = []
        for t0 in range(c0, min(c0 + CH, int(n_sec - WIN_S)), 5):
            lo, hi = np.searchsorted(bt, [t0, t0 + WIN_S])
            lab = lz_label(bk[lo:hi])
            if lab:
                wins.append((t0 - c0, lab[0], lab[1]))
        out += cut("leipzig", rec, split, sig, fs0, [0, 1, 2], wins, toff=c0)
    return out


def items_leipzig():
    d = B2 / "leipzig-heart-center-ecg"
    recs = [r for r in open(d / "RECORDS").read().split() if (d / f"{r}.atr").exists()]
    sp = B.split_of(recs)
    return [{"path": str(d / r), "rec": r, "split": sp[r]} for r in recs]


# ---------------------------------------------------------------- vfdb / cudb
def rhythm_iv(ann, fs0, n_sec):
    ch = []
    for s, aux in zip(ann.sample, ann.aux_note, strict=False):
        a = (aux or "").strip("\x00 ").strip()
        if a.startswith("("):
            ch.append((s / fs0, a[1:].split()[0]))
    return [(t, ch[i + 1][0] if i + 1 < len(ch) else n_sec, r) for i, (t, r) in enumerate(ch)]


VF_TV = {"VT", "VFL", "VF", "VFIB"}


def job_vfdb(item):
    path, rec, split = item["path"], item["rec"], item["split"]
    hdr = wfdb.rdheader(path)
    ann = wfdb.rdann(path, "atr")
    fs0 = float(hdr.fs)
    n_sec = hdr.sig_len / fs0
    iv = rhythm_iv(ann, fs0, n_sec)
    sig = wfdb.rdrecord(path).p_signal
    wins = []
    for t0 in range(0, int(n_sec - WIN_S), 5):
        rs = {r for s, e, r in iv if e > t0 and s < t0 + WIN_S}
        if len(rs) != 1:
            continue
        r = rs.pop()
        if r == "NOISE":  # "ruido" de vfdb no equivale a ilegible (T204 AUROC 0,58): se omite
            continue
        tv = 1.0 if r in VF_TV else (-1.0 if r in ("HGEA", "VER") else 0.0)
        fa = 1.0 if r == "AFIB" else 0.0
        tsv = 1.0 if r == "SVTA" else (-1.0 if r == "NOD" else 0.0)
        pvc = 1.0 if r in ("HGEA", "BI") else -1.0
        wins.append((t0, [fa, 0.0, pvc, -1.0], [tv, tsv, 0.0, 0.0]))
    return cut("vfdb", rec, split, sig, fs0, B.slots_for(hdr.sig_name), wins)


def job_cudb(item):
    path, rec, split = item["path"], item["rec"], item["split"]
    hdr = wfdb.rdheader(path)
    ann = wfdb.rdann(path, "atr")
    fs0 = float(hdr.fs)
    n_sec = hdr.sig_len / fs0
    eps, op, bad = [], None, []
    for s, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note, strict=False):
        t = s / fs0
        a = (aux or "").strip("\x00 ")
        if sym == "[" or a in ("(VT", "(VF"):
            op = t if op is None else op
        elif sym == "]" or (a.startswith("(") and op is not None):
            if op is not None:
                eps.append((op, t))
                op = None
        if sym in ("~", "|"):
            bad.append(t)
    if op is not None:
        eps.append((op, n_sec))
    bad = np.array(bad)
    sig = wfdb.rdrecord(path).p_signal
    wins = []
    for t0 in range(0, int(n_sec - WIN_S), 5):
        if len(bad) and np.any((bad >= t0) & (bad < t0 + WIN_S)):
            continue
        ov = sum(max(0.0, min(e, t0 + WIN_S) - max(s, t0)) for s, e in eps)
        if 0 < ov < 3.0:
            continue
        wins.append((t0, [-1, -1, -1, -1], [1.0 if ov >= 3.0 else 0.0, -1, 0, 0]))
    return cut("cudb", rec, split, sig, fs0, B.slots_for(hdr.sig_name), wins)


def items_vf():
    out = []
    for db in ("vfdb",):  # cudb fuera: solo anota FV, las TV sin anotar quedaban como negativos
        ps = sorted(p[:-4] for p in glob.glob(str(B2 / db / "*.hea")) if os.path.exists(p[:-4] + ".atr"))
        sp = B.split_of([os.path.basename(p) for p in ps])
        out += [{"path": p, "rec": os.path.basename(p), "split": sp[os.path.basename(p)], "db": db} for p in ps]
    return out


def job_vf(item):
    return job_vfdb(item) if item["db"] == "vfdb" else job_cudb(item)


GROUPS = {"n_leipzig": (items_leipzig, job_leipzig), "n_vf": (items_vf, job_vf)}


def safe(a):
    fn, item = a
    try:
        return fn(item)
    except Exception as e:  # noqa: BLE001
        print("fallo", item.get("rec"), repr(e)[:200], flush=True)
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    try:
        import t205_more  # noqa: F401  (registra mas grupos)
    except ImportError:
        pass
    items_fn, fn = GROUPS[args.tag]
    items = items_fn()
    print(args.tag, "registros", len(items), flush=True)
    tmp = OUT / f"{args.tag}_X.f16"
    meta = {"db": [], "rec": [], "split": [], "start": [], "Y": [], "TQ": []}
    n = 0
    with open(tmp, "wb") as fx, Pool(args.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(safe, [(fn, it) for it in items], chunksize=1)):
            for db, rec, split, st, w, y, q in res:
                fx.write(np.ascontiguousarray(w, np.float16).tobytes())
                for key, v in zip(("db", "rec", "split", "start", "Y", "TQ"), (db, rec, split, st, y, q), strict=True):
                    meta[key].append(v)
                n += 1
            if k % 50 == 0:
                print(f"{k}/{len(items)} registros, {n} ventanas", flush=True)
    xa = np.memmap(tmp, dtype=np.float16, mode="r", shape=(n, WIN, 3))
    dst = np.lib.format.open_memmap(OUT / f"{args.tag}_X.npy", mode="w+", dtype=np.float16, shape=(n, WIN, 3))
    for i in range(0, n, 20000):
        dst[i:i + 20000] = xa[i:i + 20000]
    dst.flush()
    del dst
    os.remove(tmp)
    Y, TQ = np.stack(meta["Y"]).astype(np.float32), np.stack(meta["TQ"]).astype(np.float32)
    db, split = np.array(meta["db"]), np.array(meta["split"])
    np.savez(OUT / f"{args.tag}_meta.npz", db=db, rec=np.array(meta["rec"]), split=split,
             start=np.array(meta["start"], np.float64), Y=Y)
    np.savez(OUT / f"{args.tag}_tq.npz", TQ=TQ)
    names = ["FA", "FLUT", "PVC", "PAC", "TV", "TSV", "STACH", "RUIDO"]
    A = np.concatenate([Y, TQ], 1)
    for d in sorted(set(db)):
        for s in ("train", "val", "test"):
            m = (db == d) & (split == s)
            print(d, s, "n", int(m.sum()), " ".join(f"{nm}+{int((A[m, i] == 1).sum())}/0:{int((A[m, i] == 0).sum())}"
                                                     for i, nm in enumerate(names)), flush=True)
    print("LISTO", args.tag, n, flush=True)


if __name__ == "__main__":
    main()
