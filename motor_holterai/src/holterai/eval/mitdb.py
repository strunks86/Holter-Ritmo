"""Evaluación del modelo estable sobre MIT-BIH (MIT-BIH Arrhythmia Database).

Flujo: descarga los registros que falten (MIT-BIH es de acceso libre), corre
el detector y el clasificador desplegados, y compara contra las anotaciones
AAMI con ``bxb``, ``rxr`` y estratos de calidad por ruido del latido.

No se copian datos de pacientes: solo se informan números agregados y el
identificador público del registro (101, 202…).
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from holterai.eval.bxb import (
    AAMI_CLASSES,
    WINDOW_S,
    BxbResult,
    QualityStrata,
    RecordBeats,
    combine_results,
    mean_metrics,
    score_bxb_by_stratum,
)
from holterai.eval.rxr import DEFAULT_TARGETS, RunSummary, pool_runs, scan_runs
from holterai.eval.table import fmt, markdown_table

MITDB_URL = "https://physionet-open.s3.amazonaws.com/mitdb/1.0.0/"
MITDB_EXTS = (".hea", ".dat", ".atr")

#: Partición inter-paciente clásica (de Chazal et al.); igual que en el
#: conjunto de entrenamiento, sin los cuatro registros con marcapasos.
DS1: tuple[str, ...] = (
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
)
DS2: tuple[str, ...] = (
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
)
PACED: tuple[str, ...] = ("102", "104", "107", "217")
SPLITS: dict[str, tuple[str, ...]] = {"ds1": DS1, "ds2": DS2, "paced": PACED}


def split_records(split: str = "ds2") -> tuple[str, ...]:
    """Registros de la partición pedida (``ds1``, ``ds2``, ``paced`` o ``all``)."""
    key = split.strip().lower()
    if key in SPLITS:
        return SPLITS[key]
    if key == "all":
        return tuple(sorted({*DS1, *DS2, *PACED}))
    raise ValueError(f"partición desconocida: {split!r} (ds1, ds2, paced, all)")


def default_mitdb_dir() -> Path:
    """Carpeta de MIT-BIH: ``HOLTERAI_MITDB``, ``D:/HolterIA`` o ``data/``."""
    env = os.environ.get("HOLTERAI_MITDB")
    if env:
        return Path(env)
    for cand in (
        Path("D:/HolterIA/datasets/mitdb"),
        Path("D:/HolterIA/mitdb"),
        Path("data/datasets/mitdb"),
    ):
        if cand.is_dir():
            return cand
    return Path("data/datasets/mitdb")


def default_model_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "models" / "pretrained" / "beat_cnn.npz"


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------
def _fetch(url: str, dest: Path, retries: int = 3, timeout_s: int = 60) -> tuple[bool, str]:
    if dest.exists() and dest.stat().st_size > 0:
        return True, "skip"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HolterIA/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = resp.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return True, f"{len(data)} B"
        except urllib.error.HTTPError as exc:
            return False, f"HTTP {exc.code}"
        except Exception as exc:  # red inestable: reintenta
            if attempt == retries - 1:
                return False, type(exc).__name__
            time.sleep(1.5 * (attempt + 1))
    return False, "fallo"


def ensure_records(
    data_dir: str | Path,
    records: Sequence[str],
    base_url: str = MITDB_URL,
    exts: Sequence[str] = MITDB_EXTS,
    workers: int = 8,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Descarga los ficheros que falten; devuelve los que no se pudieron bajar."""
    root = Path(data_dir)
    tasks = [(f"{base_url}{rec}{ext}", root / f"{rec}{ext}") for rec in records for ext in exts]
    failed: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_fetch, url, dest): dest for url, dest in tasks}
        for fut in futures:
            ok, msg = fut.result()
            done += 1
            if not ok:
                failed.append(f"{futures[fut].name}: {msg}")
            if progress and (done % 20 == 0 or done == len(tasks)):
                progress(f"{done}/{len(tasks)} archivos ({len(failed)} fallos)")
    return {"downloaded_or_present": len(tasks) - len(failed), "failed": failed}


# ---------------------------------------------------------------------------
# Predicción (detector + clasificador desplegados)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RecordPrediction:
    """Latidos detectados y clasificados de un registro."""

    times: tuple[float, ...]
    classes: tuple[str, ...]
    noise: tuple[float, ...]
    width_ms: tuple[float, ...] = ()

    def __len__(self) -> int:
        return len(self.times)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": len(self.times),
            "classes": {c: self.classes.count(c) for c in AAMI_CLASSES},
        }


def predict_record(
    hea: str | Path,
    model_path: str | Path | None = None,
    lead: int = 0,
) -> RecordPrediction:
    """Corre el detector y el clasificador estables sobre un WFDB."""
    import numpy as np

    from holterai.legacy.app.ingest import wfdb
    from holterai.legacy.app.ml.net import CLASSES
    from holterai.legacy.app.pipeline import classify_beats
    from holterai.legacy.app.signal import features
    from holterai.legacy.app.signal.filters import preprocess
    from holterai.legacy.app.signal.qrs import detect_qrs

    path = Path(hea)
    rec = wfdb.read(path)
    if rec.n_leads == 0:
        return RecordPrediction((), (), ())
    x_raw = np.asarray(rec.samples[:, lead], dtype=np.float64)
    xf = preprocess(x_raw, rec.fs, "display")
    idx = detect_qrs(xf, rec.fs)
    if idx.size == 0:
        return RecordPrediction((), (), ())
    wins = features.extract_windows_batch(xf, rec.fs, idx)
    widths = np.array([features.qrs_width(xf, rec.fs, int(p)) for p in idx], dtype=np.float64)
    half = int(0.1 * rec.fs)
    amps = np.array(
        [float(np.max(np.abs(xf[max(0, int(p) - half):int(p) + half]))) for p in idx],
        dtype=np.float64,
    )
    cls, _conf, _rr, _probs = classify_beats(rec, idx, wins, widths, amps)
    noise = features.noise_batch(x_raw, rec.fs, idx)
    return RecordPrediction(
        times=tuple(float(p) / rec.fs for p in idx),
        classes=tuple(CLASSES[int(c)] for c in cls),
        noise=tuple(float(v) for v in noise),
        width_ms=tuple(float(w) * 1000.0 for w in widths),
    )


def _header_fs(hea: Path) -> float:
    """Frecuencia de muestreo del encabezado WFDB (3.er campo de la 1.ª línea)."""
    first = hea.read_text(encoding="utf-8", errors="replace").splitlines()[0].split()
    try:
        return float(first[2])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"cabecera WFDB inválida: {hea}") from exc


def load_reference(hea: str | Path) -> tuple[list[float], list[str]]:
    """Tiempos y clases AAMI de las anotaciones de referencia."""
    from holterai.legacy.app.ingest import annotations

    path = Path(hea)
    fs = _header_fs(path)
    ann = annotations.read_ann(path.with_suffix(".atr"))
    times: list[float] = []
    classes: list[str] = []
    for beat in ann.beats():
        aami = beat.aami
        if aami is None:
            continue
        times.append(beat.sample / fs)
        classes.append(aami)
    return times, classes


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
def evaluate_records(
    paths: Iterable[str | Path],
    predictor: Callable[[Path], RecordPrediction] = predict_record,
    ref_loader: Callable[[Path], tuple[list[float], list[str]]] = load_reference,
    window_s: float = WINDOW_S,
    strata: QualityStrata | None = None,
    targets: Sequence[str] = DEFAULT_TARGETS,
) -> dict[str, Any]:
    """Evalúa una lista de registros WFDB con detector/clasificador inyectables."""
    conf = strata or QualityStrata()
    per_record: dict[str, BxbResult] = {}
    per_record_rxr: dict[str, dict[str, Any]] = {}
    strata_rows: dict[str, BxbResult] = {}
    rxr_ref: dict[str, list[RunSummary]] = {t: [] for t in targets}
    rxr_det: dict[str, list[RunSummary]] = {t: [] for t in targets}
    errors: dict[str, str] = {}

    for path in paths:
        path = Path(path)
        name = path.stem
        try:
            ref_times, ref_classes = ref_loader(path)
            pred = predictor(path)
        except Exception as exc:  # un registro roto no invalida el resto
            errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        res = RecordBeats(ref_times, ref_classes, pred.times, pred.classes).score(
            window_s=window_s, label=name
        )
        per_record[name] = res
        ref_runs = {t: scan_runs(ref_classes, t) for t in targets}
        det_runs = {t: scan_runs(pred.classes, t) for t in targets}
        for target in targets:
            rxr_ref[target].append(ref_runs[target])
            rxr_det[target].append(det_runs[target])
        per_record_rxr[name] = {
            "ref": {t: ref_runs[t].as_dict() for t in targets},
            "det": {t: det_runs[t].as_dict() for t in targets},
        }
        for bucket, partial in score_bxb_by_stratum(
            ref_times,
            ref_classes,
            pred.times,
            pred.classes,
            det_noise=pred.noise,
            strata=conf,
            window_s=window_s,
        ).items():
            if bucket in strata_rows:
                strata_rows[bucket] = combine_results(
                    [strata_rows[bucket], partial], label=bucket
                )
            else:
                strata_rows[bucket] = partial

    pooled = combine_results(per_record.values(), label="agrupado")
    means = mean_metrics(per_record.values())
    rxr_totals = {
        target: {
            "ref": pool_runs(rxr_ref[target]),
            "det": pool_runs(rxr_det[target]),
        }
        for target in targets
    }
    return {
        "window_s": float(window_s),
        "n_records": len(per_record),
        "errors": errors,
        "records": {name: res.as_dict() for name, res in per_record.items()},
        "rxr": {"per_record": per_record_rxr, "totals": rxr_totals},
        "pooled": pooled.as_dict(),
        "mean": {
            "n_records": means["n_records"],
            "detection": {k: round(v, 4) for k, v in means["detection"].items()},
            "classes": {
                c: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}
                for c, s in means["classes"].items()
            },
        },
        "strata": {name: res.as_dict() for name, res in strata_rows.items()},
        "strata_names": list(conf.all_names()),
    }


def evaluate_mitdb(
    data_dir: str | Path | None = None,
    split: str = "ds2",
    model_path: str | Path | None = None,
    window_s: float = WINDOW_S,
    limit: int = 0,
    download: bool = True,
    strata: QualityStrata | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evalúa MIT-BIH con el modelo estable y devuelve métricas y tablas."""
    root = Path(data_dir) if data_dir is not None else default_mitdb_dir()
    records = split_records(split)
    if limit > 0:
        records = records[:limit]
    info: dict[str, Any] = {
        "data_dir": str(root),
        "split": split.lower(),
        "model": "",
        "strata_cuts": list((strata or QualityStrata()).cuts),
    }
    if download:
        info["fetch"] = ensure_records(root, records, progress=progress)
    paths = [root / f"{rec}.hea" for rec in records]
    missing = [p.stem for p in paths if not p.is_file()]
    info["missing"] = missing
    model = Path(model_path) if model_path is not None else default_model_path()
    info["model"] = str(model)
    predictor = _bound_predictor(model)
    result = evaluate_records([p for p in paths if p.is_file()], predictor=predictor,
                              window_s=window_s, strata=strata)
    result.update(info)
    result["table"] = render_markdown(result)
    return result


def _bound_predictor(model: Path) -> Callable[[Path], RecordPrediction]:
    def run(path: Path) -> RecordPrediction:
        return predict_record(path, model_path=model)

    return run


# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------
_HEAD_CLASSES = AAMI_CLASSES


def _per_record_table(result: dict[str, Any]) -> str:
    headers = ["registro", "n_ref", "n_det", "Se_QRS", "+P_QRS", "FPR"]
    for cls in _HEAD_CLASSES:
        headers += [f"Se {cls}", f"+P {cls}"]
    rows = []
    for name, rec in result["records"].items():
        det = rec["detection"]
        row: list[Any] = [
            name, det["n_ref"], det["n_det"], fmt(det["se"]), fmt(det["ppv"]), fmt(det["fpr"]),
        ]
        for cls in _HEAD_CLASSES:
            stats = rec["classes"][cls]
            row += [fmt(stats["se"]), fmt(stats["ppv"])]
        rows.append(row)
    rows.append(["**agrupado**", *_pooled_row(result["pooled"])])
    return markdown_table(headers, rows)


def _pooled_row(pooled: dict[str, Any]) -> list[Any]:
    det = pooled["detection"]
    row: list[Any] = [
        det["n_ref"], det["n_det"], fmt(det["se"]), fmt(det["ppv"]), fmt(det["fpr"]),
    ]
    for cls in _HEAD_CLASSES:
        stats = pooled["classes"][cls]
        row += [fmt(stats["se"]), fmt(stats["ppv"])]
    return row


def _class_table(result: dict[str, Any]) -> str:
    pooled = result["pooled"]
    mean = result["mean"]
    headers = ["clase", "n_ref", "n_det", "tp", "fp", "fn",
               "Se agrupada", "+P agrupada", "FPR agrupada",
               "Se promedio", "+P promedio", "FPR promedio"]
    rows = []
    for cls in _HEAD_CLASSES:
        p = pooled["classes"][cls]
        m = mean["classes"][cls]
        rows.append([cls, p["n_ref"], p["n_det"], p["tp"], p["fp"], p["fn"],
                     fmt(p["se"]), fmt(p["ppv"]), fmt(p["fpr"]),
                     fmt(m["se"]), fmt(m["ppv"]), fmt(m["fpr"])])
    det_p, det_m = pooled["detection"], mean["detection"]
    rows.append(["QRS", det_p["n_ref"], det_p["n_det"], det_p["tp"], det_p["fp"], det_p["fn"],
                 fmt(det_p["se"]), fmt(det_p["ppv"]), fmt(det_p["fpr"]),
                 fmt(det_m["se"]), fmt(det_m["ppv"]), fmt(det_m["fpr"])])
    return markdown_table(headers, rows)


def _strata_table(result: dict[str, Any]) -> str:
    headers = ["estrato", "n_ref", "n_det", "tp", "fp", "fn",
               "Se_QRS", "+P_QRS", "Se V", "+P V", "Se S", "+P S"]
    rows = []
    names = [*result.get("strata_names", []), "sin_medicion"]
    for name in names:
        rec = result["strata"].get(name)
        if not rec:
            continue
        det = rec["detection"]
        rows.append([name, det["n_ref"], det["n_det"], det["tp"], det["fp"], det["fn"],
                     fmt(det["se"]), fmt(det["ppv"]),
                     fmt(rec["classes"]["V"]["se"]), fmt(rec["classes"]["V"]["ppv"]),
                     fmt(rec["classes"]["S"]["se"]), fmt(rec["classes"]["S"]["ppv"])])
    if not rows:
        return "_(sin estratos)_"
    return markdown_table(headers, rows)


def _rxr_table(result: dict[str, Any]) -> str:
    headers = ["clase", "dupletas ref", "dupletas det", "3-5 ref", "3-5 det",
               ">=6 ref", ">=6 det", "latidos en rachas ref", "latidos en rachas det"]
    rows = []
    for target, totals in result["rxr"]["totals"].items():
        ref, det = totals["ref"], totals["det"]
        rows.append([target, ref["dupletas"], det["dupletas"], ref["rachas_3_5"],
                     det["rachas_3_5"], ref["rachas_6"], det["rachas_6"],
                     ref["latidos_en_rachas"], det["latidos_en_rachas"]])
    return markdown_table(headers, rows)


def render_markdown(result: dict[str, Any]) -> str:
    """Tablas Markdown del resultado: por registro, por clase, estratos y rachas."""
    n = result.get("n_records", 0)
    title = (
        f"Evaluación MIT-BIH · partición {result.get('split', '?')} · "
        f"{n} registros · ventana {result.get('window_s', WINDOW_S):.3f} s"
    )
    parts = [
        f"## {title}",
        "",
        f"Modelo: `{result.get('model', '')}`.",
        "",
        "### bxb por registro",
        "",
        _per_record_table(result),
        "",
        "### bxb por clase (agrupado y promedio por registro)",
        "",
        _class_table(result),
        "",
        "### Estratos por calidad del latido (cortes de ruido en mV: "
        + "/".join(f"{c:g}" for c in result.get("strata_cuts", ()) or ())
        + ")",
        "",
        _strata_table(result),
        "",
        "### rxr (rachas)",
        "",
        _rxr_table(result),
    ]
    if result.get("missing"):
        parts += ["", f"Registros no encontrados: {', '.join(result['missing'])}"]
    if result.get("errors"):
        parts += ["", f"Errores: {result['errors']}"]
    return "\n".join(parts)
