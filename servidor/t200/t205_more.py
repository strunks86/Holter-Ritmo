"""Grupos adicionales de T205 (se registran en t205_build.GROUPS)."""
from __future__ import annotations

import csv
import glob
import os
import zlib
from pathlib import Path

import numpy as np
import wfdb

import t200_build as B
import __main__ as TB

if not hasattr(TB, "GROUPS"):
    import t205_build as TB
B2, WIN_S = TB.B2, TB.WIN_S


# ---------------------------------------------------------------- BUT QDB
def items_butqdb():
    recs = sorted(os.path.basename(p) for p in glob.glob(str(B2 / "butqdb/*")) if os.path.isdir(p))
    sp = B.split_of(sorted({r[:3] for r in recs}))
    return [{"path": str(B2 / "butqdb" / r / f"{r}_ECG"), "ann": str(B2 / "butqdb" / r / f"{r}_ANN.csv"),
             "rec": r, "split": sp[r[:3]]} for r in recs]


def job_butqdb(item):
    """Calidad de consenso (ultimas 3 columnas): 1 = limpio -> RUIDO 0; 3 = ilegible -> RUIDO 1; 2 se omite."""
    path, rec, split = item["path"], item["rec"], item["split"]
    hdr = wfdb.rdheader(path)
    fs0 = float(hdr.fs)
    segs = []
    for row in csv.reader(open(item["ann"])):
        if len(row) >= 12 and row[9].strip() and row[11].strip():
            segs.append((int(float(row[9])) / fs0, int(float(row[10])) / fs0, int(float(row[11]))))
    rng = np.random.default_rng(zlib.crc32(rec.encode()))
    c1, c3 = [], []
    for s, e, c in segs:
        if c == 1:
            c1 += list(range(int(s) + 1, int(e - WIN_S), 60))
        elif c == 3:
            c3 += list(range(int(s) + 1, int(e - WIN_S), 10))
    if len(c1) > 500:
        c1 = list(rng.choice(c1, 500, replace=False))
    if len(c3) > 500:
        c3 = list(rng.choice(c3, 500, replace=False))
    out = []
    for t0, ru in sorted([(t, 0.0) for t in c1] + [(t, 1.0) for t in c3]):
        a = int(t0 * fs0)
        b = a + int((WIN_S + 1) * fs0)
        if b > hdr.sig_len:
            continue
        # margen de 2 s a cada lado para el filtro
        a2 = max(0, a - int(2 * fs0))
        sig = wfdb.rdrecord(path, sampfrom=a2, sampto=min(hdr.sig_len, b + int(2 * fs0))).p_signal
        out += TB.cut("butqdb", rec, split, sig, fs0, [0, -1, -1],
                      [((a - a2) / fs0, [-1, -1, -1, -1], [-1, -1, 0, ru])], toff=t0 - (a - a2) / fs0)
    return out


TB.GROUPS["n_butqdb"] = (items_butqdb, job_butqdb)


# ---------------------------------------------------------------- Noise Holter Database
def _nh_dir():
    return glob.glob(str(B2 / "noise-holter-database/noise*/wfdb_files"))[0]


def items_nhdb():
    hs = sorted(p[:-4] for p in glob.glob(_nh_dir() + "/*.hea"))
    sp = B.split_of(sorted({os.path.basename(h)[:3] for h in hs}))
    by = {}
    for h in hs:
        by.setdefault(os.path.basename(h)[:3], []).append(h)
    return [{"subj": s, "paths": ps, "split": sp[s], "rec": s} for s, ps in sorted(by.items())]


def job_nhdb(item):
    """Segmentos >= 10 s. RUIDO: Clean=0, N1=-1 (ruido con QRS visible), N2=1 (QRS ilegible)."""
    out = []
    for path in item["paths"]:
        txt = open(path + ".hea").read().split("\n")
        hd = txt[0].split()
        fs0, n = float(hd[2]), int(hd[3])
        if n / fs0 < WIN_S:
            continue
        com = [x for x in txt if x.startswith("#")]
        parts = [p.strip() for p in com[0][1:].split("|")] if com else []
        cls = next((p.split(":", 1)[1].strip() for p in parts if p.startswith("CLASS")), "")
        i = next((k for k, p in enumerate(parts) if p.startswith("EVENTS")), None)
        evs = set([parts[i].split(":", 1)[1].strip()] + parts[i + 1:]) if i is not None else set()
        ru = {"Clean": 0.0, "N1": -1.0, "N2": 1.0}.get(cls)
        if ru is None:
            continue
        r = wfdb.rdrecord(path)
        try:
            a = wfdb.rdann(path, "atr")
            bt = np.array(a.sample) / fs0
            bs = list(a.symbol)
            bx = [(x or "").strip("\x00 ") for x in a.aux_note]
        except Exception:  # noqa: BLE001
            bt, bs, bx = np.array([]), [], []
        dur = n / fs0
        starts = list(range(0, int(dur - WIN_S) + 1, 10))
        if dur - WIN_S - starts[-1] >= 3:
            starts.append(dur - WIN_S)
        wins = []
        for t0 in starts:
            lo, hi = np.searchsorted(bt, [t0, t0 + WIN_S])
            sy, ax = bs[lo:hi], bx[lo:hi]
            fa = 1.0 if "A. Fib." in evs else 0.0
            fl = 1.0 if "A. Flutter" in evs else 0.0
            ev_v = bool(evs & {"VPB", "V Couplet", "MB"})
            ev_s = bool(evs & {"SVPB", "SV Couplet", "TgSV"})
            pvc = 1.0 if "V" in sy else (-1.0 if ev_v else 0.0)
            pac = 1.0 if "A" in sy else (-1.0 if ev_s else 0.0)
            tv = 1.0 if any("TachyV" in x for x in ax) else 0.0
            if evs & {"TachySV", "TachyA"}:
                run = best = 0
                for k in range(lo, hi):
                    run = run + 1 if (bs[k] == "A" and (run == 0 or bt[k] - bt[k - 1] < 0.6)) else (1 if bs[k] == "A" else 0)
                    best = max(best, run)
                tsv = 1.0 if best >= 3 else -1.0
                pac = -1.0 if tsv == 1.0 else pac
            else:
                tsv = 0.0
            if ru == 1.0:  # ilegible: no se afirma ningun positivo de latido
                pvc = -1.0 if pvc == 1.0 else pvc
                pac = -1.0 if pac == 1.0 else pac
            wins.append((float(t0), [fa, fl, pvc, pac], [tv, tsv, 0.0, ru]))
        slots = [0, 1, -1] if r.p_signal.shape[1] >= 2 else [0, -1, -1]
        out += TB.cut("nhdb", os.path.basename(path), item["split"], r.p_signal, fs0, slots, wins)
    return out


TB.GROUPS["n_nhdb"] = (items_nhdb, job_nhdb)


# ---------------------------------------------------------------- CPSC2020
def items_cpsc20():
    d = B2 / "cpsc2020/TrainingSet"
    recs = sorted(os.path.basename(p)[:-4] for p in glob.glob(str(d / "data/A*.mat")))
    sp = B.split_of(recs)
    out = []
    for r in recs:
        for h in range(0, 27):
            out.append({"rec": r, "split": sp[r], "hour": h, "d": str(d)})
    return out


def _runs(t):
    """Salvas de >=3 latidos con intervalos < 0,6 s."""
    eps, i = [], 0
    while i < len(t):
        j = i
        while j + 1 < len(t) and t[j + 1] - t[j] < 0.6:
            j += 1
        if j - i + 1 >= 3:
            eps.append((t[i], t[j]))
        i = j + 1
    return eps


def job_cpsc20(item):
    import scipy.io as sio
    rec, h = item["rec"], item["hour"]
    fs0 = 400.0
    ecg = sio.loadmat(f"{item['d']}/data/{rec}.mat")["ecg"][:, 0]
    ref = sio.loadmat(f"{item['d']}/ref/R{rec[1:]}.mat")["ref"][0, 0]
    S = np.sort(np.asarray(ref["S_ref"]).ravel()) / fs0
    V = np.sort(np.asarray(ref["V_ref"]).ravel()) / fs0
    n_sec = len(ecg) / fs0
    c0 = h * 3600
    if c0 >= n_sec - WIN_S:
        return []
    c1 = min(c0 + 3600, n_sec - WIN_S - 1)
    rng = np.random.default_rng(zlib.crc32(f"{rec}/{h}".encode()))
    starts = set(range(c0, int(c1), 60))
    ect = np.concatenate([S[(S >= c0 + 5) & (S < c1)], V[(V >= c0 + 5) & (V < c1)]])
    if len(ect) > 70:
        ect = rng.choice(ect, 70, replace=False)
    starts |= {int(t - 5) for t in ect}
    rs, rv = _runs(S), _runs(V)
    for s, e in rs + rv:
        if c0 + 5 <= s < c1:
            starts.add(int((s + e) / 2 - 5))
    wins = []
    for t0 in sorted(starts):
        t1 = t0 + WIN_S
        pvc = float(np.any((V >= t0) & (V < t1)))
        pac = float(np.any((S >= t0) & (S < t1)))
        tv = float(any(e > t0 and s < t1 and min(e, t1) - max(s, t0) >= min(1.0, e - s) * 0.999 for s, e in rv))
        tsv = float(any(e > t0 and s < t1 and min(e, t1) - max(s, t0) >= min(1.0, e - s) * 0.999 for s, e in rs))
        wins.append((float(t0 - c0), [-1, -1, pvc, pac], [tv, tsv, 0.0, 0.0]))
    a, b = int(c0 * fs0), int(min(len(ecg), (c0 + 3600 + WIN_S + 5) * fs0))
    return TB.cut("cpsc2020", rec, item["split"], ecg[a:b, None], fs0, [0, -1, -1], wins, toff=c0)


TB.GROUPS["n_cpsc20"] = (items_cpsc20, job_cpsc20)


# ---------------------------------------------------------------- Chapman-Ningbo (ECG de reposo, solo refuerzo)
CH = {"AFIB": "164889003", "AFL": "164890007", "SVT": "426761007", "AT": "713422000", "AVNRT": "233896004",
      "AVRT": "233897008", "ST": "427084000"}
CH_PVC = {"17338001", "75532003", "11157007", "251180001"}
CH_PAC = {"284470004", "251173003", "251164006", "426995002"}


def items_chapman():
    hs = sorted(glob.glob(str(B2 / "ecg-arrhythmia/WFDBRecords/*/*/*.hea")))
    rng = np.random.default_rng(205)
    sel = []
    for h in hs:
        dx = set()
        for ln in open(h, errors="replace"):
            if ln.startswith("#Dx:"):
                dx = set(x.strip() for x in ln[4:].split(","))
        raro = dx & {CH["AFL"], CH["SVT"], CH["AT"], CH["AVNRT"], CH["AVRT"]}
        u = rng.random()
        if raro or (CH["AFIB"] in dx and u < 0.4) or u < 0.12:
            sel.append((h[:-4], sorted(dx)))
    sp = B.split_of([os.path.basename(p) for p, _ in sel])
    return [{"batch": sel[i:i + 200], "sp": {os.path.basename(p): sp[os.path.basename(p)] for p, _ in sel[i:i + 200]},
             "rec": f"lote{i}"} for i in range(0, len(sel), 200)]


def job_chapman(item):
    out = []
    for path, dx in item["batch"]:
        dx = set(dx)
        try:
            r = wfdb.rdrecord(path)
        except Exception:  # noqa: BLE001
            continue
        names = [n.upper() for n in r.sig_name]
        if r.p_signal.shape[0] < int(WIN_S * r.fs) or not all(n in names for n in ("II", "V1", "V5")):
            continue
        sig = r.p_signal[:, [names.index("II"), names.index("V1"), names.index("V5")]]
        fa = float(CH["AFIB"] in dx)
        fl = float(CH["AFL"] in dx)
        tsv = float(bool(dx & {CH["SVT"], CH["AT"], CH["AVNRT"], CH["AVRT"]}))
        arr = fa or fl or tsv
        pvc = float(bool(dx & CH_PVC))
        pac = 1.0 if dx & CH_PAC else (-1.0 if arr else 0.0)
        rec = os.path.basename(path)
        # relleno espejo de 1 s a cada lado para el filtro; la ventana son los 10 s originales
        n1 = int(r.fs)
        sig2 = np.concatenate([sig[n1:0:-1], sig, sig[-2:-n1 - 2:-1]])
        out += TB.cut("chapman", rec, item["sp"][rec], sig2, float(r.fs), [0, 1, 2],
                      [(1.0, [fa, fl, pvc, pac], [0.0, tsv, float(CH["ST"] in dx), 0.0])], toff=-1.0)
    return out


TB.GROUPS["n_chapman"] = (items_chapman, job_chapman)
