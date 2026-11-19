"""T202: ventanas extra centradas en episodios de TV y TSV (tag 'taqui'), mismas particiones
por registro que ya existen (no se crean registros nuevos ni se cambia train/val/test)."""
from __future__ import annotations

import os
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t200_build as B  # noqa: E402
import t202_labels as L  # noqa: E402

B.BEATS_OK |= {"ltafdb"}
B.RHYTHM_OK |= {"ltafdb"}
CAP = 150


def plan(a):
    db, rec, split, have = a
    try:
        ev = L.load_events(db, L.rec_path(db, rec))
    except Exception as e:  # noqa: BLE001
        print("fallo", db, rec, e, flush=True)
        return None
    rng = np.random.default_rng(zlib.crc32(f"tq/{db}/{rec}".encode()))
    starts = set()
    for eps in (ev["tv"], ev["tsv"]):
        c = []
        for s, e in eps:
            if e - s <= WIN:
                c.append(int(round((s + e) / 2 - 5)))
            else:
                c += list(range(int(s), int(e - WIN) + 1, 10))
        c = sorted({t for t in c if 0 <= t <= ev["n_sec"] - WIN - 1 and t not in have})
        if len(c) > CAP:
            c = list(rng.choice(c, CAP, replace=False))
        starts |= set(int(t) for t in c)
    if not starts:
        return None
    return {"db": db, "path": L.rec_path(db, rec), "rec": rec, "split": split,
            "stride_s": 10, "ect_cap": 0, "starts": sorted(starts)}


WIN = 10.0

if __name__ == "__main__":
    recs = {}
    for mp in (L.OLD / "windows_meta.npz", B.OUT / "holter_meta.npz", B.OUT / "icentia_meta.npz"):
        m = np.load(mp)
        for d, r, s, t in zip(m["db"], m["rec"], m["split"], m["start"], strict=True):
            d = str(d)
            if d in L.BEATS_DB or d == "shdb-af":
                recs.setdefault((d, str(r)), [str(s), set()])[1].add(int(round(t)))
    jobs = [(d, r, v[0], v[1]) for (d, r), v in recs.items()]
    print("registros a revisar", len(jobs), flush=True)
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    meta = {"db": [], "rec": [], "split": [], "start": [], "Y": []}
    n = 0
    tmp = B.OUT / "taqui_X.f16"
    with Pool(nw) as pool:
        items = [x for x in pool.imap_unordered(plan, jobs, chunksize=8) if x]
        print("registros con episodios", len(items), "ventanas planeadas", sum(len(i["starts"]) for i in items), flush=True)
        with open(tmp, "wb") as fx:
            for k, res in enumerate(pool.imap_unordered(B.job, items, chunksize=1)):
                for db, rec, split, st, w, y in res:
                    if db == "ltafdb":
                        y[1] = -1.0
                    fx.write(np.ascontiguousarray(w, np.float16).tobytes())
                    for key, v in zip(("db", "rec", "split", "start", "Y"), (db, rec, split, st, y), strict=True):
                        meta[key].append(v)
                    n += 1
                if k % 100 == 0:
                    print(f"{k}/{len(items)} registros, {n} ventanas", flush=True)
    xa = np.memmap(tmp, dtype=np.float16, mode="r", shape=(n, B.WIN, 3))
    dst = np.lib.format.open_memmap(B.OUT / "taqui_X.npy", mode="w+", dtype=np.float16, shape=(n, B.WIN, 3))
    for i in range(0, n, 20000):
        dst[i:i + 20000] = xa[i:i + 20000]
    dst.flush()
    del dst
    os.remove(tmp)
    np.savez(B.OUT / "taqui_meta.npz", db=np.array(meta["db"]), rec=np.array(meta["rec"]),
             split=np.array(meta["split"]), start=np.array(meta["start"], np.float64),
             Y=np.stack(meta["Y"]).astype(np.float32))
    print("TAQUI_OK", n, flush=True)
