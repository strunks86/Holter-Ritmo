"""Descarga de bases de datos públicas de ECG para entrenamiento/validación.

Requiere red. Usa urllib (pip/curl están roto en este equipo).
  python -m app.tools.fetch_datasets mitdb afdb
"""
from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "datasets"

MITDB_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]

AFDB_RECORDS = [
    "04015", "04043", "04048", "04126", "04746", "04908", "05121", "05261",
    "06426", "07162", "07859", "07879", "07910", "08215", "08219", "08378",
    "08405", "08434", "08455",
]

SVDB_RECORDS = [str(n) for n in range(800, 878)]  # MIT-BIH Supraventricular Arrhythmia DB

BASES = {
    "mitdb": ("https://physionet.org/files/mitdb/1.0.0/", MITDB_RECORDS,
              (".hea", ".dat", ".atr")),
    "afdb": ("https://physionet.org/files/afdb/1.0.0/", AFDB_RECORDS,
             (".hea", ".atr")),
    "svdb": ("https://physionet.org/files/svdb/1.0.0/", SVDB_RECORDS,
             (".hea", ".dat", ".atr")),
}


def fetch(url: str, dest: Path, retries: int = 3) -> tuple[bool, str]:
    if dest.exists() and dest.stat().st_size > 0:
        return True, "skip"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HolterIA/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return True, f"{len(data)} B"
        except urllib.error.HTTPError as e:
            return False, f"HTTP {e.code}"
        except Exception as e:
            if attempt == retries - 1:
                return False, type(e).__name__
            time.sleep(1.5 * (attempt + 1))
    return False, "fallo"


def download(name: str, workers: int = 8) -> None:
    base, records, exts = BASES[name]
    outdir = DATA / name
    tasks = []
    for rec in records:
        for ext in exts:
            tasks.append((f"{base}{rec}{ext}", outdir / f"{rec}{ext}"))

    ok = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch, u, d): (u, d) for u, d in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            okk, msg = fut.result()
            if okk:
                ok += 1
            else:
                fail += 1
                print(f"  FALLO {futs[fut][1].name}: {msg}")
            if i % 20 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} archivos ({time.time()-t0:.0f}s)")
    print(f"{name}: {ok} ok, {fail} fallos en {time.time()-t0:.0f}s -> {outdir}")


if __name__ == "__main__":
    names = sys.argv[1:] or ["mitdb"]
    for n in names:
        if n not in BASES:
            print("unknown:", n)
            continue
        download(n)

