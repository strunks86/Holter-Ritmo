"""T202: etiquetas TV y TSV por ventana, a partir de las anotaciones publicas.

TV  = ritmo anotado (VT / (VFL, o salva de >=3 latidos V seguidos con intervalos < 600 ms (>100/min).
TSV = ritmo anotado (SVTA / (AT / (PAT, o salva de >=3 latidos supraventriculares (A, a, J, S)
      seguidos con intervalos < 600 ms. La taquicardia sinusal NO cuenta (queda como negativo;
      en PTB-XL el codigo STACH se guarda aparte para medir falsos positivos).
      Ritmo nodal anotado ((NOD) sin salva rapida -> TSV enmascarada (-1).
Negativo (0) solo donde la base trae latidos anotados (o, para TSV, ritmo anotado en SHDB-AF).
Salida por fuente: <tag>_tq.npz con TQ (N,3) = [TV, TSV, es_taquicardia_sinusal_PTBXL].
"""
from __future__ import annotations

import ast
import csv
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t200_build as B  # noqa: E402

BASES = B.BASES
OLD = Path("/root/holter/t158/data")
NEW = B.OUT
V_RUN = {"V"}
S_RUN = {"A", "a", "J", "S"}
TV_RH = {"(VT", "(VFL"}
TSV_RH = {"(SVTA", "(AT", "(PAT"}
NOD_RH = {"(NOD"}
BEATS_DB = {"mitdb", "svdb", "incartdb", "ltdb", "edb", "sddb", "icentia11k", "ltafdb"}
WIN_S = 10.0


def rec_path(db, rec):
    if db == "icentia11k":
        return str(BASES / "icentia11k" / f"p{rec[1:3]}" / rec[:6] / rec)
    return str(BASES / db / rec)


def load_events(db, path):
    hdr = wfdb.rdheader(path)
    ann = wfdb.rdann(path, "atr")
    fs = float(hdr.fs)
    n_sec = hdr.sig_len / fs
    beats = [(s / fs, sym) for s, sym in zip(ann.sample, ann.symbol, strict=False) if sym in B.BEAT_SYM]
    if db not in BEATS_DB:
        beats = []
    rh = []
    for s, aux in zip(ann.sample, ann.aux_note, strict=False):
        a = (aux or "").strip("\x00 ").strip()
        if a.startswith("("):
            rh.append((s / fs, a.split()[0].split("\x00")[0]))
        elif a == ")":
            rh.append((s / fs, None))

    def rh_iv(tags):
        return [(t, rh[i + 1][0] if i + 1 < len(rh) else n_sec) for i, (t, g) in enumerate(rh) if g in tags]

    def runs(symset):
        out, i, n = [], 0, len(beats)
        while i < n:
            if beats[i][1] in symset:
                j = i
                while j + 1 < n and beats[j + 1][1] in symset and beats[j + 1][0] - beats[j][0] < 0.6:
                    j += 1
                if j - i + 1 >= 3:
                    out.append((beats[i][0], beats[j][0]))
                i = j + 1
            else:
                i += 1
        return out
    return {"bt": np.array([b[0] for b in beats]), "tv": rh_iv(TV_RH) + runs(V_RUN),
            "tsv": rh_iv(TSV_RH) + runs(S_RUN), "nod": rh_iv(NOD_RH),
            "rh0": rh[0][0] if rh else None, "n_sec": n_sec}


def _ov(eps, t0):
    t1 = t0 + WIN_S
    for s, e in eps:
        o = min(e, t1) - max(s, t0)
        if e > t0 and s < t1 and o >= min(1.0, e - s) * 0.999:
            return True
    return False


def label(db, ev, t0):
    lo, hi = np.searchsorted(ev["bt"], [t0, t0 + WIN_S]) if len(ev["bt"]) else (0, 0)
    hb = hi > lo
    tv = 1.0 if _ov(ev["tv"], t0) else (0.0 if hb else -1.0)
    if _ov(ev["tsv"], t0):
        tsv = 1.0
    elif _ov(ev["nod"], t0):
        tsv = -1.0
    elif hb or (db == "shdb-af" and ev["rh0"] is not None and t0 >= ev["rh0"]):
        tsv = 0.0
    else:
        tsv = -1.0
    return tv, tsv


def job(a):
    db, rec, idx, starts = a
    out = np.full((len(idx), 3), -1.0, np.float32)
    out[:, 2] = 0.0
    if db in ("afdb", "cpsc2021", "cpsc2018"):
        return idx, out
    try:
        ev = load_events(db, rec_path(db, rec))
    except Exception as e:  # noqa: BLE001
        print("fallo", db, rec, e, flush=True)
        return idx, out
    for k, t0 in enumerate(starts):
        out[k, 0], out[k, 1] = label(db, ev, float(t0))
    return idx, out


def run(meta_path, out_path, pool):
    m = np.load(meta_path)
    db, rec, start = m["db"], m["rec"], m["start"]
    n = len(db)
    tq = np.full((n, 3), -1.0, np.float32)
    tq[:, 2] = 0.0
    isp = db == "ptbxl"
    if isp.any():
        codes = {}
        for r in csv.DictReader(open(BASES / "ptb-xl/ptbxl_database.csv", encoding="utf-8", errors="replace")):
            codes[os.path.basename(r["filename_hr"])] = set(ast.literal_eval(r["scp_codes"]).keys())
        for i in np.where(isp)[0]:
            c = codes.get(str(rec[i]), set())
            tq[i] = [0.0, 1.0 if c & {"SVTAC", "PSVT"} else 0.0, 1.0 if "STACH" in c else 0.0]
    groups = {}
    for i in np.where(~isp)[0]:
        groups.setdefault((str(db[i]), str(rec[i])), []).append(i)
    jobs = [(d, r, np.array(ix), start[ix]) for (d, r), ix in groups.items()]
    for idx, out in pool.imap_unordered(job, jobs, chunksize=4):
        tq[idx] = out
    np.savez(out_path, TQ=tq)
    for d in sorted(set(db)):
        s = db == d
        print(f"{Path(out_path).name} {d}: n={int(s.sum())} TV+={int((tq[s, 0] == 1).sum())} TV0={int((tq[s, 0] == 0).sum())} "
              f"TSV+={int((tq[s, 1] == 1).sum())} TSV0={int((tq[s, 1] == 0).sum())} STACH={int(tq[s, 2].sum())}", flush=True)


if __name__ == "__main__":
    tags = sys.argv[1].split(",")
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as pool:
        for t in tags:
            if t == "old":
                run(OLD / "windows_meta.npz", OLD / "windows_tq.npz", pool)
            elif (NEW / f"{t}_meta.npz").exists():
                run(NEW / f"{t}_meta.npz", NEW / f"{t}_tq.npz", pool)
    print("ETIQUETAS_OK", flush=True)
