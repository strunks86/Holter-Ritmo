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


# ---------------------------------------------------------------- bases con latidos anotados (tanda 2)
KIND = {"V": "V", "F": "V", "E": "V", "S": "S", "A": "S", "a": "S", "J": "J", "j": "J", "/": "P", "f": "P",
        "N": "N", "L": "N", "R": "N", "B": "N", "e": "N", "n": "N", "Q": "N", "r": "N"}
BAD_SYM = {"~", '"', "|", "=", "!", "[", "]"}
BAD_AUX = ("ARTIFACT", "OFF", "ZERO", "CAUTERY", "NOISE", "UNREADABLE")
CFG2 = {
    "mghdb": {"dir": "mghdb", "ext": "ari", "stride": 30, "cap": 150, "fa": -1.0,
              "leads": (("ECG LEAD II", "II"), ("ECG LEAD V", "V", "ECG LEAD V5", "ECG LEAD V1"), ("ECG LEAD I", "I"))},
    "nsrdb": {"dir": "nsrdb", "ext": "atr", "stride": 180, "cap": 100, "fa": 0.0, "leads": None},
    "chfdb": {"dir": "chfdb", "ext": "ecg", "stride": 180, "cap": 300, "fa": -1.0, "leads": None},
    "stdb": {"dir": "stdb", "ext": "atr", "stride": 60, "cap": 100, "fa": -1.0, "leads": None},
    "excluded": {"dir": "excluded", "ext": "atr", "stride": 120, "cap": 300, "fa": -1.0, "leads": None},
    "ltstdb": {"dir": "ltstdb", "ext": "atr", "stride": 120, "cap": 200, "fa": -1.0, "leads": None},
}


def items_beats(db):
    def f():
        d = B2 / CFG2[db]["dir"]
        hs = sorted(glob.glob(str(d / "**/*.hea"), recursive=True))
        ps = [h[:-4] for h in hs if os.path.exists(h[:-4] + "." + CFG2[db]["ext"]) and os.path.exists(h[:-4] + ".dat")]
        sp = B.split_of([os.path.basename(p) for p in ps])
        return [{"db": db, "path": p, "rec": os.path.basename(p), "split": sp[os.path.basename(p)]} for p in ps]
    return f


def job_beats(item):
    db, path, rec, split = item["db"], item["path"], item["rec"], item["split"]
    cfg = CFG2[db]
    hdr = wfdb.rdheader(path)
    ann = wfdb.rdann(path, cfg["ext"])
    fs0 = float(hdr.fs)
    n_sec = hdr.sig_len / fs0
    names = [str(n).upper().strip() for n in hdr.sig_name]
    if cfg["leads"]:
        slots = []
        for cands in cfg["leads"]:
            slots.append(next((names.index(c) for c in cands if c in names), -1))
        if slots[0] < 0 and slots[1] < 0 and slots[2] < 0:
            return []
    else:
        ecg = [i for i, n in enumerate(names) if "ECG" in n or n in ("II", "V1", "V5", "MLII", "ML2", "MV1", "MV5", "V2", "V3", "V4", "V6", "I", "III", "AVF", "MLIII", "V", "ML5")]
        ecg = ecg or list(range(len(names)))
        sub = B.slots_for([names[i] for i in ecg])
        slots = [ecg[s] if s >= 0 else -1 for s in sub]
    bt, bk, bad = [], [], []
    for s, sym, aux in zip(ann.sample, ann.symbol, ann.aux_note, strict=False):
        sym = str(sym)
        a = (aux or "").strip("\x00 ").upper()
        if sym in BAD_SYM or any(x in a for x in BAD_AUX):
            bad.append(s / fs0)
        k = KIND.get(sym)
        if k:
            bt.append(s / fs0)
            bk.append(k)
    if len(bt) < 50:
        return []
    bt, bad = np.array(bt), np.array(bad)

    def runs(kind):
        eps, i, n = [], 0, len(bt)
        while i < n:
            if bk[i] == kind:
                j = i
                while j + 1 < n and bk[j + 1] == kind and bt[j + 1] - bt[j] < 0.6:
                    j += 1
                if j - i + 1 >= 3:
                    eps.append((bt[i], bt[j]))
                i = j + 1
            else:
                i += 1
        return eps
    rv, rs = runs("V"), runs("S")
    rng = np.random.default_rng(zlib.crc32(f"{db}/{rec}".encode()))
    starts = set(range(0, int(n_sec - WIN_S), cfg["stride"]))
    ect = [t for t, k in zip(bt, bk, strict=True) if k in ("V", "S")]
    if len(ect) > cfg["cap"]:
        ect = list(rng.choice(ect, cfg["cap"], replace=False))
    starts |= {int(t - 5) for t in ect if 5 <= t < n_sec - WIN_S}
    starts |= {int((s + e) / 2 - 5) for s, e in rv + rs if 5 <= (s + e) / 2 < n_sec - WIN_S}

    def ov(eps, t0):
        t1 = t0 + WIN_S
        return any(e > t0 and s < t1 and min(e, t1) - max(s, t0) >= min(1.0, e - s) * 0.999 for s, e in eps)
    out = []
    CH = 1800
    for c0 in range(0, int(n_sec), CH):
        st = sorted(t for t in starts if c0 <= t < c0 + CH)
        wins = []
        for t0 in st:
            if len(bad) and np.any((bad >= t0 - 1) & (bad < t0 + WIN_S + 1)):
                continue
            lo, hi = np.searchsorted(bt, [t0, t0 + WIN_S])
            ks = bk[lo:hi]
            if len(ks) < 3:
                continue
            pace = "P" in ks
            pvc = 1.0 if "V" in ks else (-1.0 if pace else 0.0)
            pac = 1.0 if "S" in ks else (-1.0 if (pace or "J" in ks) else 0.0)
            tv = 1.0 if ov(rv, t0) else (-1.0 if pace else 0.0)
            tsv = 1.0 if ov(rs, t0) else (-1.0 if (pace or "J" in ks) else 0.0)
            wins.append((float(t0 - c0), [cfg["fa"], cfg["fa"], pvc, pac], [tv, tsv, 0.0, -1.0]))
        if not wins:
            continue
        a, b = int(c0 * fs0), int(min(hdr.sig_len, (c0 + CH + WIN_S + 5) * fs0))
        sig = wfdb.rdrecord(path, sampfrom=a, sampto=b).p_signal
        out += TB.cut(db, rec, split, sig, fs0, slots, wins, toff=c0)
    return out


for _db in CFG2:
    TB.GROUPS["n_" + _db] = (items_beats(_db), job_beats)


# ---------------------------------------------------------------- IRIDIA-AF (Holter 2 canales, solo FA)
def items_iridia():
    d = B2 / "iridia-af"
    pat = {}
    for r in csv.DictReader(open(d / "iridia-af-metadata-v1.0.1.csv")):
        pat[r["record_id"]] = r["patient_id"]
    sp = B.split_of(sorted(set(pat.values())))
    out = []
    for rec in sorted(pat):
        rd = d / "iridia-af-records-v1.0.1" / rec
        if rd.is_dir():
            out.append({"dir": str(rd), "rec": rec, "split": sp[pat[rec]]})
    return out


def job_iridia(item):
    import h5py
    rd, rec, split = item["dir"], item["rec"], item["split"]
    fs0 = 200.0
    eps = {}
    for r in csv.DictReader(open(f"{rd}/{rec}_ecg_labels.csv")):
        a, b = int(r["start_file_index"]), int(r["end_file_index"])
        s, e = int(r["start_qrs_index"]) / fs0, int(r["end_qrs_index"]) / fs0
        for f in range(a, b + 1):
            eps.setdefault(f, []).append((s if f == a else 0.0, e if f == b else 1e12))
    rng = np.random.default_rng(zlib.crc32(rec.encode()))
    out = []
    files = sorted(glob.glob(f"{rd}/{rec}_ecg_*.h5"))
    for fi, fp in enumerate(files):
        with h5py.File(fp, "r") as h:
            key = "ecg" if "ecg" in h else list(h.keys())[0]
            ds = h[key]
            n_sec = ds.shape[0] / fs0
            ev = eps.get(fi, [])

            def frac(t0):
                return sum(max(0.0, min(e, t0 + WIN_S) - max(s, t0)) for s, e in ev) / WIN_S
            pos, neg = [], []
            cand = set(range(30, int(n_sec - WIN_S - 30), 120))
            for s, e in ev:  # inicios y finales de episodio: ventanas justo antes y despues
                for t in (s - 25, s - 12, s + 2, s + 15, e - 25, e - 12, e + 2, e + 15):
                    if 30 <= t < n_sec - WIN_S - 30:
                        cand.add(int(t))
            for t0 in cand:
                f = frac(t0)
                if f >= 0.999:
                    pos.append(t0)
                elif f <= 0.001:
                    neg.append(t0)
            cap = max(40, 240 // max(1, len(files)))
            if len(pos) > cap:
                pos = list(rng.choice(pos, cap, replace=False))
            if len(neg) > cap:
                neg = list(rng.choice(neg, cap, replace=False))
            for t0, fa in sorted([(t, 1.0) for t in pos] + [(t, 0.0) for t in neg]):
                a = int((t0 - 2) * fs0)
                sig = np.asarray(ds[a:a + int((WIN_S + 4) * fs0)], dtype=np.float64)
                if sig.ndim == 1:
                    sig = sig[:, None]
                slots = [0, 1, -1] if sig.shape[1] >= 2 else [0, -1, -1]
                out += TB.cut("iridia", rec, split, sig, fs0, slots,
                              [(2.0, [fa, -1, -1, -1], [-1, -1, 0, -1])], toff=fi * 100000.0 + t0 - 2.0)
    return out


TB.GROUPS["n_iridia"] = (items_iridia, job_iridia)
