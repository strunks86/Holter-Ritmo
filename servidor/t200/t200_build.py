"""T200: ventanas de 10 s (500 Hz, 3 ranuras II/V1/V5) de las bases Holter que T158 no uso.

Mismo preprocesado que T158 (t158_build.prep_record + zwin), mismas etiquetas FA/FLUT/PVC/PAC
(-1 = enmascarado). Solo bases publicas anotadas, nada de pacientes.

Bases (en BASES/<carpeta>):
  mitdb, svdb, incartdb, ltdb, edb, sddb  -> latidos (V = PVC; A/a/J/S = PAC) y ritmo si lo hay
  shdb-af, cpsc2021                        -> solo ritmo (sus latidos son todos N: PVC/PAC enmascarados)
  icentia11k                               -> 1 canal, latidos N/S/V y ritmo (N/(AFIB/(AFL ... ")")
Particion por registro/paciente (semilla 200): 70 % train, 15 % val, 15 % test.
Registros largos se procesan por bloques de 1 h para no gastar RAM.
Salida: OUT/new_X.npy (N,5000,3) float16 + OUT/new_meta.npz (db, rec, split, start, Y).
"""
from __future__ import annotations

import argparse
import glob
import zlib
import importlib.util
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import wfdb

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("HOLTERAI", "/root/holter/holterai"))
_spec = importlib.util.spec_from_file_location("t158_build", REPO / "scripts/t158_build.py")
tb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tb)

BASES = Path(os.environ.get("BASES", "/root/holter/data/HolterIA/bases"))
OUT = Path(os.environ.get("T200_OUT", "/root/holter/t200/data"))
FS, WIN, WIN_S = tb.FS, tb.WIN, tb.WIN_S
PVC_SYM = {"V"}
PAC_SYM = {"A", "a", "J", "S"}
BEAT_SYM = {"N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q"}
BEATS_OK = {"mitdb", "svdb", "incartdb", "ltdb", "edb", "sddb", "icentia11k"}
RHYTHM_OK = {"mitdb", "edb", "shdb-af", "cpsc2021", "icentia11k"}
CHUNK_S = 3600


def slots_for(names):
    """Indice de canal para cada ranura (II, V1, V5); -1 = vacia."""
    up = [n.upper().strip() for n in names]

    def find(*cands):
        for c in cands:
            if c in up:
                return up.index(c)
        return -1
    s0 = find("II", "MLII", "ML2")
    s1 = find("V1", "MV1")
    s2 = find("V5", "MV5", "V4", "V6")
    used = {s0, s1, s2} - {-1}
    rest = [i for i in range(len(names)) if i not in used]
    if s0 < 0 and rest:
        s0 = rest.pop(0)
    if s1 < 0 and rest:
        s1 = rest.pop(0)
    return [s0, s1, s2]


def rhythm_changes(ann, fs0):
    ch = []
    for s, aux in zip(ann.sample, ann.aux_note, strict=False):
        a = (aux or "").strip("\x00 ").strip()
        if a.startswith("("):
            r = {"(AFIB": "AF", "(AFL": "AFL", "(N": "N"}.get(a.split()[0], "O")
            ch.append((s / fs0, r))
        elif a == ")":
            ch.append((s / fs0, None))
    ch.sort(key=lambda z: z[0])
    return ch


def rhythm_of_window(ch, t0, t1):
    """Ritmo unico en [t0,t1) o None si cambia o es desconocido."""
    times = [c[0] for c in ch]
    k = np.searchsorted(times, t0, side="right") - 1
    cur = ch[k][1] if k >= 0 else None
    j = k + 1
    while j < len(ch) and ch[j][0] < t1:
        if ch[j][1] != cur:
            return None
        j += 1
    return cur


def job(item):
    db, path, rec, split = item["db"], item["path"], item["rec"], item["split"]
    try:
        hdr = wfdb.rdheader(path)
        ann = wfdb.rdann(path, "atr")
    except Exception as e:  # noqa: BLE001
        print("fallo", db, rec, e, flush=True)
        return []
    fs0 = float(hdr.fs)
    n_sec = int(hdr.sig_len / fs0)
    slots = slots_for(hdr.sig_name)
    ch = rhythm_changes(ann, fs0) if db in RHYTHM_OK else []
    bt, bs = [], []
    for s, sym in zip(ann.sample, ann.symbol, strict=False):
        if sym in BEAT_SYM:
            bt.append(s / fs0)
            bs.append(sym)
    bt = np.array(bt)
    rng = np.random.default_rng(zlib.crc32(f"{db}/{rec}".encode()))

    starts = set(range(0, max(0, n_sec - int(WIN_S)), int(item["stride_s"])))
    if db in BEATS_OK and len(bt):
        ect = [t for t, s in zip(bt, bs, strict=False) if s in PVC_SYM or s in PAC_SYM]
        if len(ect) > item["ect_cap"]:
            ect = list(rng.choice(ect, item["ect_cap"], replace=False))
        last = -1e9
        for t in sorted(ect):
            if t - last >= WIN_S:
                t0 = int(round(t - 5.0))
                if 0 <= t0 <= n_sec - int(WIN_S):
                    starts.add(t0)
                    last = t
    starts = sorted(starts)
    out = []
    for c0 in range(0, n_sec, CHUNK_S):
        st = [t for t in starts if c0 <= t < c0 + CHUNK_S]
        if not st:
            continue
        a = int(c0 * fs0)
        b = int(min(hdr.sig_len, (c0 + CHUNK_S + WIN_S + 20) * fs0))
        try:
            r = wfdb.rdrecord(path, sampfrom=a, sampto=b)
        except Exception as e:  # noqa: BLE001
            print("fallo lectura", db, rec, e, flush=True)
            break
        sig = np.nan_to_num(r.p_signal)
        raw = np.zeros((sig.shape[0], 3))
        used = []
        for j, c in enumerate(slots):
            if c >= 0:
                raw[:, j] = sig[:, c]
                used.append(j)
        x = tb.prep_record(raw, fs0)
        for t0 in st:
            y = np.full(4, -1.0, np.float32)
            rhy = rhythm_of_window(ch, t0, t0 + WIN_S) if ch else None
            if rhy in ("AF", "N", "O"):
                y[0] = 1.0 if rhy == "AF" else 0.0
                y[1] = 0.0
            elif rhy == "AFL":
                y[1] = 1.0
                y[0] = 0.0
            if db in BEATS_OK and len(bt):
                lo, hi = np.searchsorted(bt, [t0, t0 + WIN_S])
                syms = bs[lo:hi]
                if syms:
                    y[2] = float(any(s in PVC_SYM for s in syms))
                    y[3] = float(any(s in PAC_SYM for s in syms))
            if np.all(y < 0):
                continue
            i0 = int((t0 - c0) * FS)
            w = x[i0:i0 + WIN]
            if w.shape[0] != WIN or np.abs(w).max() == 0:
                continue
            out.append((db, rec, split, float(t0), tb.zwin(w, used).astype(np.float16), y))
    return out


def split_of(names, seed=200):
    rng = np.random.default_rng(seed)
    names = sorted(names)
    order = rng.permutation(len(names))
    n_tr, n_va = int(round(0.70 * len(names))), int(round(0.15 * len(names)))
    return {names[i]: ("train" if p < n_tr else ("val" if p < n_tr + n_va else "test"))
            for p, i in enumerate(order)}


def items_flat(db, folder, stride, cap, pattern="*.hea"):
    paths = sorted(glob.glob(str(BASES / folder / pattern)))
    paths = [p[:-4] for p in paths if os.path.exists(p[:-4] + ".atr")]
    sp = split_of([os.path.basename(p) for p in paths])
    return [{"db": db, "path": p, "rec": os.path.basename(p), "split": sp[os.path.basename(p)],
             "stride_s": stride, "ect_cap": cap} for p in paths]


def items_cpsc2021(stride):
    paths = sorted(glob.glob(str(BASES / "cpsc2021/*/*.hea")))
    paths = [p[:-4] for p in paths if os.path.exists(p[:-4] + ".atr")]
    pat = {os.path.basename(p): "_".join(os.path.basename(p).split("_")[:2]) for p in paths}
    sp = split_of(set(pat.values()))  # data_<paciente>_<n>: particion por paciente
    return [{"db": "cpsc2021", "path": p, "rec": os.path.basename(p),
             "split": sp[pat[os.path.basename(p)]], "stride_s": stride, "ect_cap": 0}
            for p in paths]


def items_icentia(n_segs, stride, cap, max_pat):
    pats = sorted(glob.glob(str(BASES / "icentia11k/p*/p*")))
    pats = [p for p in pats if os.path.isdir(p)]
    if max_pat:
        pats = pats[:max_pat]
    sp = split_of([os.path.basename(p) for p in pats])
    out = []
    for p in pats:
        segs = sorted(s[:-4] for s in glob.glob(p + "/*.atr")
                      if os.path.exists(s[:-4] + ".dat"))
        if not segs:
            continue
        rng = np.random.default_rng(int(os.path.basename(p)[1:]))
        pick = rng.choice(segs, min(n_segs, len(segs)), replace=False)
        for s in pick:
            out.append({"db": "icentia11k", "path": s, "rec": os.path.basename(s),
                        "split": sp[os.path.basename(p)], "stride_s": stride, "ect_cap": cap})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--dbs", default="mitdb,svdb,incartdb,ltdb,edb,sddb,shdb-af,cpsc2021,icentia11k")
    ap.add_argument("--icentia_segs", type=int, default=2)
    ap.add_argument("--icentia_max", type=int, default=0)
    ap.add_argument("--tag", default="new")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dbs = args.dbs.split(",")
    items = []
    cfg = {"mitdb": (60, 300), "svdb": (60, 300), "incartdb": (60, 300), "ltdb": (120, 400),
           "edb": (60, 200), "sddb": (120, 400), "shdb-af": (120, 0)}
    for db, (stride, cap) in cfg.items():
        if db in dbs:
            items += items_flat(db, db, stride, cap)
    if "cpsc2021" in dbs:
        items += items_cpsc2021(30)
    if "icentia11k" in dbs:
        items += items_icentia(args.icentia_segs, 300, 10, args.icentia_max)
    print("registros", len(items), flush=True)

    # escritura incremental a un .npy por trozos para no juntar todo en RAM
    tmp = OUT / f"{args.tag}_X.f16"
    meta = {"db": [], "rec": [], "split": [], "start": [], "Y": []}
    n = 0
    with open(tmp, "wb") as fx, Pool(args.workers) as pool:
        for k, res in enumerate(pool.imap_unordered(job, items, chunksize=1)):
            for db, rec, split, st, w, y in res:
                fx.write(np.ascontiguousarray(w, np.float16).tobytes())
                meta["db"].append(db)
                meta["rec"].append(rec)
                meta["split"].append(split)
                meta["start"].append(st)
                meta["Y"].append(y)
                n += 1
            if k % 200 == 0:
                print(f"{k}/{len(items)} registros, {n} ventanas", flush=True)
    xa = np.memmap(tmp, dtype=np.float16, mode="r", shape=(n, WIN, 3))
    dst = np.lib.format.open_memmap(OUT / f"{args.tag}_X.npy", mode="w+", dtype=np.float16,
                                    shape=(n, WIN, 3))
    for i in range(0, n, 20000):
        dst[i:i + 20000] = xa[i:i + 20000]
    dst.flush()
    del dst
    os.remove(tmp)
    np.savez(OUT / f"{args.tag}_meta.npz", db=np.array(meta["db"]), rec=np.array(meta["rec"]),
             split=np.array(meta["split"]), start=np.array(meta["start"], np.float64),
             Y=np.stack(meta["Y"]).astype(np.float32))
    print("listo", n, "ventanas", flush=True)


if __name__ == "__main__":
    main()
