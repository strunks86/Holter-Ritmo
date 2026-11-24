"""CLI de HolterIA."""

from __future__ import annotations

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Annotated, NoReturn

import typer

app = typer.Typer(
    name="holterai",
    help="HolterIA — análisis local de Holter HScribe.",
    no_args_is_help=True,
)


def _pending(phase: str) -> NoReturn:
    typer.echo(f"no implementado (fase {phase})", err=True)
    raise typer.Exit(code=2)


@app.command()
def inventory(
    interim: Annotated[Path, typer.Option("--interim", help="Directorio WFDB.")] = Path(
        "data/interim"
    ),
) -> None:
    """Lista los exámenes WFDB disponibles (F01)."""
    exams = sorted(p.stem for p in interim.glob("*.hea"))
    typer.echo(json.dumps({"interim": str(interim), "exams": exams}, indent=2))
    if not exams:
        raise typer.Exit(code=2)


@app.command()
def worklist(
    db: Annotated[Path | None, typer.Option("--db", help="Ruta de holterai.db.")] = None,
) -> None:
    """Lista pacientes/exámenes registrados en SQLite."""
    from holterai.store import list_exams

    typer.echo(json.dumps(list_exams(db), indent=2, ensure_ascii=False))


@app.command()
def export(
    report: Annotated[Path, typer.Option("--report", help="report.json.")],
    out: Annotated[Path, typer.Option("--out", help="Directorio de exportación.")] = Path(
        "data/exports"
    ),
    formats: Annotated[str, typer.Option("--formats", help="csv,json.")] = "csv,json",
) -> None:
    """Exporta beats/eventos/resumen a CSV/JSON."""
    from holterai.export import export_report

    files = export_report(report, out, tuple(x.strip() for x in formats.split(",") if x.strip()))
    typer.echo(json.dumps(files, indent=2, ensure_ascii=False))


@app.command()
def models(
    root: Annotated[Path | None, typer.Option("--root", help="Carpeta de modelos.")] = None,
) -> None:
    """Lista los modelos locales y su estado."""
    from holterai.model_registry import as_json, list_models

    typer.echo(as_json(list_models(root)))


@app.command()
def edf(
    exam: Annotated[Path, typer.Option("--exam", help="WFDB .hea de entrada.")],
    out: Annotated[Path | None, typer.Option("--out", help="Salida .edf.")] = None,
) -> None:
    """Exporta la señal a EDF+ (1 s por registro, int16)."""
    from holterai.edf_export import write_edf
    from holterai.legacy.loader import load_wfdb16

    rec = load_wfdb16(exam)
    target = out or Path("data/exports") / (Path(exam).stem + ".edf")
    p = write_edf(
        rec.samples,
        rec.fs,
        rec.leads,
        target,
        patient_id=rec.patient.patient_id,
        sex=rec.patient.sex,
        dob=rec.patient.dob,
        name=rec.patient.name,
        start=rec.start_time,
    )
    typer.echo(json.dumps({"edf": str(p)}, indent=2))


@app.command("import-ecg")
def import_ecg(
    src: Annotated[Path, typer.Option("--src", help="EDF/CSV/WFDB de entrada.")],
    out: Annotated[Path, typer.Option("--out", help="WFDB de salida, sin extensión.")] = Path(
        "data/interim/imported"
    ),
    fs: Annotated[
        float | None, typer.Option("--fs", help="Frecuencia si el CSV no la trae.")
    ] = None,
) -> None:
    """Convierte EDF/CSV/WFDB a WFDB16 para el pipeline."""
    from holterai.legacy.app.ingest import base as ingest
    from holterai.legacy.app.ingest import wfdb

    rec = ingest.load(src, fs=fs)
    p = wfdb.write(out, rec, fmt="16")
    typer.echo(json.dumps({"wfdb": str(p), "fs": rec.fs, "leads": rec.leads}, indent=2))


@app.command()
def ingest(
    src: Annotated[Path, typer.Option("--src", help="Carpeta del examen HScribe.")],
    dll_dir: Annotated[
        Path, typer.Option("--dll-dir", help="Carpeta de HScribe con HW32ExamAPI.dll.")
    ],
    out: Annotated[Path, typer.Option("--out", help="Salida WFDB sin extensión.")] = Path(
        "data/interim/exam"
    ),
    seconds: Annotated[int, typer.Option("--seconds", help="Segundos; 0 = todo.")] = 600,
) -> None:
    """Ingesta HScribe -> WFDB 180 Hz (Windows + HScribe)."""
    from holterai.ingest_hscribe import export_hscribe

    hea = export_hscribe(src, out, seconds=seconds, dll_dir=dll_dir)
    typer.echo(str(hea))


@app.command()
def convert(
    src: Annotated[Path, typer.Option("--src", help="Carpeta del examen HScribe.")],
    dll_dir: Annotated[
        Path, typer.Option("--dll-dir", help="Carpeta de HScribe con HW32ExamAPI.dll.")
    ],
    out: Annotated[Path, typer.Option("--out", help="Salida WFDB sin extensión.")] = Path(
        "data/interim/exam"
    ),
    seconds: Annotated[int, typer.Option("--seconds", help="Segundos; 0 = todo.")] = 600,
) -> None:
    """Convierte HScribe -> WFDB 180 Hz (alias explícito de ingest)."""
    from holterai.ingest_hscribe import export_hscribe

    typer.echo(str(export_hscribe(src, out, seconds=seconds, dll_dir=dll_dir)))


@app.command()
def analyze(
    exam: Annotated[Path, typer.Option("--exam", help="Examen WFDB (.hea).")],
    out: Annotated[Path, typer.Option("--out", help="Directorio de salida.")] = Path(
        "data/reports"
    ),
) -> None:
    """Analiza un examen y genera report.json + DOCX/HTML/PDF."""
    from holterai.prototype import analyze_exam

    result = analyze_exam(exam, out)
    typer.echo(json.dumps(result["files"], indent=2, ensure_ascii=False))


@app.command()
def report(
    exam: Annotated[Path, typer.Option("--exam", help="Examen WFDB (.hea).")],
    out: Annotated[Path, typer.Option("--out", help="Directorio de salida.")] = Path(
        "data/reports"
    ),
) -> None:
    """Genera el reporte de un examen ya analizado o lo analiza."""
    from holterai.prototype import analyze_exam

    result = analyze_exam(exam, out)
    typer.echo(json.dumps(result["files"], indent=2, ensure_ascii=False))


@app.command()
def batch(
    src: Annotated[Path, typer.Option("--src", help="Examen o repositorio FDrepo.")],
    dll_dir: Annotated[
        Path, typer.Option("--dll-dir", help="Carpeta de HScribe con HW32ExamAPI.dll.")
    ],
    out: Annotated[Path, typer.Option("--out", help="Directorio de reportes.")] = Path(
        "data/reports"
    ),
    interim: Annotated[Path, typer.Option("--interim", help="Directorio WFDB intermedio.")] = Path(
        "data/interim"
    ),
    seconds: Annotated[int, typer.Option("--seconds", help="Segundos; 0 = todo.")] = 600,
    limit: Annotated[int, typer.Option("--limit", help="Máximo de exámenes; 0 = todos.")] = 0,
    skip_existing: Annotated[
        bool, typer.Option("--skip-existing", help="No repetir reportes existentes.")
    ] = False,
) -> None:
    """Procesa un examen o todo FDrepo sin intervención manual."""
    from holterai.batch import process_batch

    summary = process_batch(
        src,
        dll_dir=dll_dir,
        out=out,
        interim=interim,
        seconds=seconds,
        limit=limit,
        skip_existing=skip_existing,
    )
    typer.echo(
        json.dumps({k: summary[k] for k in ("n_exams", "ok", "errors", "skipped")}, indent=2)
    )


eval_app = typer.Typer(
    name="eval",
    help=(
        "Evaluación EC57: bxb (latido a latido), rxr (rachas) y epicmp "
        "(episodios). Tabla bxb de MIT-BIH DS2: "
        "holterai eval mitdb --split ds2 --no-download"
    ),
    no_args_is_help=True,
)
app.add_typer(eval_app, name="eval")


@eval_app.command("mitdb")
def eval_mitdb(
    data_dir: Annotated[
        Path | None, typer.Option("--data-dir", help="Carpeta de MIT-BIH.")
    ] = None,
    split: Annotated[str, typer.Option("--split", help="ds1, ds2, paced o all.")] = "ds2",
    model: Annotated[Path | None, typer.Option("--model", help="Modelo de latidos.")] = None,
    window_s: Annotated[float, typer.Option("--window-s", help="Ventana bxb (s).")] = 0.150,
    limit: Annotated[int, typer.Option("--limit", help="Máximo de registros; 0 = todos.")] = 0,
    download: Annotated[bool, typer.Option("--download/--no-download")] = True,
    strata_cuts: Annotated[
        str,
        typer.Option(
            "--strata-cuts",
            help="Cortes de ruido por latido en mV, separados por coma (estratos).",
        ),
    ] = "0.04,0.08",
    out: Annotated[
        Path | None, typer.Option("--out", help="Directorio para .md y .json.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Imprime JSON en vez de tabla.")] = False,
) -> None:
    """Evalúa el modelo estable sobre MIT-BIH y escribe la tabla bxb."""
    from holterai.eval.bxb import QualityStrata
    from holterai.eval.mitdb import evaluate_mitdb

    try:
        strata = QualityStrata.from_cuts(
            [float(x) for x in strata_cuts.replace(";", ",").split(",") if x.strip()]
        )
    except ValueError as exc:
        typer.echo(f"--strata-cuts inválido: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    result = evaluate_mitdb(
        data_dir=data_dir,
        split=split,
        model_path=model,
        window_s=window_s,
        limit=limit,
        download=download,
        strata=strata,
        progress=lambda msg: typer.echo(msg, err=True),
    )
    if result["n_records"] == 0:
        typer.echo(
            json.dumps({"error": "sin registros evaluables", **{
                k: result[k] for k in ("data_dir", "split", "missing", "errors")
            }}, indent=2, ensure_ascii=False),
            err=True,
        )
        raise typer.Exit(code=2)
    if as_json:
        payload = {k: v for k, v in result.items() if k != "table"}
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        typer.echo(result["table"])
    target = out or Path("data/reports/eval")
    target.mkdir(parents=True, exist_ok=True)
    stem = f"mitdb_{split.lower()}"
    (target / f"{stem}.md").write_text(result["table"] + "\n", encoding="utf-8")
    (target / f"{stem}.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "table"}, indent=2,
                   ensure_ascii=False),
        encoding="utf-8",
    )
    typer.echo(f"tablas en {target}", err=True)


@eval_app.command("beats")
def eval_beats(
    ref: Annotated[Path, typer.Option("--ref", help="Referencia: JSON o CSV t,clase.")],
    test: Annotated[Path, typer.Option("--test", help="Detectado: JSON o CSV t,clase.")],
    window_s: Annotated[float, typer.Option("--window-s", help="Ventana bxb (s).")] = 0.150,
    as_json: Annotated[bool, typer.Option("--json", help="Imprime JSON completo.")] = False,
) -> None:
    """bxb + rxr entre dos listas de latidos con clase AAMI."""
    from holterai.eval.bxb import score_bxb
    from holterai.eval.holter import load_reference_beats
    from holterai.eval.rxr import score_rxr
    from holterai.eval.table import fmt

    ref_t, ref_c = load_reference_beats(ref)
    det_t, det_c = load_reference_beats(test)
    result = score_bxb(ref_t, ref_c, det_t, det_c, window_s=window_s, label="beats")
    pairs = [(i, j) for i, j in _pairs(ref_t, det_t, window_s)]
    payload = {"bxb": result.as_dict(), "rxr": score_rxr(ref_c, det_c, pairs=pairs)}
    if as_json:
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    det = result.detection()
    typer.echo(
        f"bxb ventana {window_s:.3f} s: n_ref={det['n_ref']} n_det={det['n_det']} "
        f"Se={fmt(det['se'])} +P={fmt(det['ppv'])} FPR={fmt(det['fpr'])}"
    )
    for cls in ("N", "S", "V", "F", "Q"):
        st = result.class_stats(cls)
        typer.echo(
            f"  {cls}: n_ref={st['n_ref']:5d} n_det={st['n_det']:5d} tp={st['tp']:5d} "
            f"Se={fmt(st['se'])} +P={fmt(st['ppv'])} FPR={fmt(st['fpr'])}"
        )


def _pairs(
    ref_times: list[float], det_times: list[float], window_s: float
) -> list[tuple[int, int]]:
    from holterai.eval.bxb import match_beats

    return list(match_beats(ref_times, det_times, window_s).pairs)


@eval_app.command("report")
def eval_report(
    report: Annotated[Path, typer.Option("--report", help="report.json de HolterIA.")],
    ref: Annotated[
        Path | None, typer.Option("--ref", help="Referencia de latidos (JSON/CSV).")
    ] = None,
    af_ref: Annotated[
        Path | None, typer.Option("--af-ref", help="Episodios de FA de referencia (JSON).")
    ] = None,
    window_s: Annotated[float, typer.Option("--window-s", help="Ventana bxb (s).")] = 0.150,
    out: Annotated[Path | None, typer.Option("--out", help="Guarda el JSON.")] = None,
) -> None:
    """bxb, rxr, epicmp y estratos por calidad de un reporte."""
    from holterai.eval.holter import (
        evaluate_report,
        load_af_reference,
        load_json,
        load_reference_beats,
    )

    payload = evaluate_report(
        load_json(report),
        reference=load_reference_beats(ref) if ref is not None else None,
        af_reference=load_af_reference(af_ref) if af_ref is not None else None,
        window_s=window_s,
    )
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        typer.echo(str(out))
    else:
        typer.echo(text)


@eval_app.command("episodes")
def eval_episodes(
    ref: Annotated[Path, typer.Option("--ref", help="Episodios de referencia (JSON).")],
    det: Annotated[Path, typer.Option("--det", help="Episodios detectados (JSON).")],
    overlap_s: Annotated[float, typer.Option("--overlap-s", help="Solapamiento mínimo (s).")] = 1.0,
    merge_gap_s: Annotated[
        float, typer.Option("--merge-gap-s", help="Fusión de cortes (s).")
    ] = 0.0,
) -> None:
    """epicmp: compara episodios por episodio y por duración."""
    from holterai.eval.epicmp import compare_episodes
    from holterai.eval.holter import load_json

    def _episodes(path: Path) -> list:
        data = load_json(path)
        if isinstance(data, dict):
            data = data.get("episodes", data.get("af", []))
        return data

    result = compare_episodes(
        _episodes(ref), _episodes(det), overlap_s=overlap_s, merge_gap_s=merge_gap_s
    )
    result.pop("episodes", None)
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


@app.command()
def serve(
    directory: Annotated[Path, typer.Option("--dir", help="Directorio con reportes.")] = Path(
        "data/reports"
    ),
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
) -> None:
    """Sirve los reportes locales (visor mínimo, sin red externa)."""
    directory.mkdir(parents=True, exist_ok=True)
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    with ThreadingHTTPServer((host, port), handler) as httpd:
        typer.echo(f"HolterIA en http://{host}:{port}/  (Ctrl+C para salir)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            typer.echo("servidor detenido")


@app.command()
def doctor(
    dll_dir: Annotated[
        Path | None, typer.Option("--dll-dir", help="Carpeta opcional de HScribe.")
    ] = None,
) -> None:
    """Comprueba instalación, modelos y dependencias."""
    from holterai.doctor import doctor as run_doctor

    typer.echo(json.dumps(run_doctor(dll_dir=dll_dir), indent=2, ensure_ascii=False))


@app.command()
def ui(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8765,
    no_browser: Annotated[bool, typer.Option("--no-browser")] = False,
) -> None:
    """Abre la interfaz local: worklist, visor, edición e informes."""
    import webbrowser

    from holterai.legacy.app.server import serve

    httpd, url = serve(host, port)
    typer.echo(f"HolterIA UI en {url}  (Ctrl+C para salir)")
    if not no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        typer.echo("servidor detenido")


@app.command()
def desktop(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8765,
    no_open: Annotated[bool, typer.Option("--no-open")] = False,
) -> None:
    """Abre HolterIA en una ventana tipo aplicación (Edge/Chrome --app)."""
    import os
    import subprocess
    import webbrowser

    from holterai.legacy.app.server import serve

    httpd, url = serve(host, port)
    typer.echo(f"HolterIA escritorio en {url}  (Ctrl+C para salir)")
    if not no_open:
        candidates = [
            Path(os.environ.get("PROGRAMFILES", "")) / "Microsoft/Edge/Application/msedge.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft/Edge/Application/msedge.exe",
            Path(os.environ.get("PROGRAMFILES", "")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
        ]
        browser = next((p for p in candidates if p.is_file()), None)
        try:
            if browser is not None:
                subprocess.Popen([str(browser), f"--app={url}", "--window-size=1500,950"])
            else:
                webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        typer.echo("servidor detenido")


@app.command()
def ec57(
    ref: Annotated[Path, typer.Option("--ref", help="JSON/CSV con tiempos de referencia.")],
    test: Annotated[Path, typer.Option("--test", help="JSON/CSV con tiempos detectados.")],
    tol: Annotated[float, typer.Option("--tol", help="Tolerancia en segundos.")] = 0.150,
) -> None:
    """Evalúa QRS con métricas EC57 (sin WFDB)."""
    import json

    from holterai.ec57 import score_qrs

    def load(path: Path) -> list[float]:
        text = path.read_text(encoding="utf-8-sig").strip()
        if text.startswith("["):
            return [float(x) for x in json.loads(text)]
        return [float(x.strip()) for x in text.replace(";", ",").split(",") if x.strip()]

    typer.echo(json.dumps(score_qrs(load(ref), load(test), tol).as_dict(), indent=2))


@app.command()
def train(
    config: Annotated[Path, typer.Option("--config")] = Path("configs/model"),
    with_corrections: Annotated[bool, typer.Option("--with-corrections")] = False,
) -> None:
    """Entrena un modelo (F04–F07/F10)."""
    _ = (config, with_corrections)
    _pending("F04")


if __name__ == "__main__":
    app()
