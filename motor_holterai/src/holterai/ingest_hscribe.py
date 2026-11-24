"""Ingesta HScribe: ejecuta el helper x86 y escribe WFDB 180 Hz.

Soporta dos layouts:
- examen standalone (todos los .dat/.xml en una carpeta);
- repositorio FDrepo (`Exam_X/slot_*/...`), que se fusiona en una carpeta
  temporal porque HW32ExamAPI no abre los slots por separado.

La ventana exportada se define con `start_s` (inicio, en s) y `seconds`
(duración; 0 = hasta el final real del registro). `start_s` queda como
comentario en el `.hea` para que la ventana se pueda auditar después.

No modifica la carpeta del examen. La carpeta temporal se borra al terminar.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def slot_dirs(exam: Path) -> list[Path]:
    """Devuelve los `slot_*` de un examen de repositorio, si existen."""
    return sorted(p for p in exam.iterdir() if p.is_dir() and p.name.startswith("slot_"))


def _resolve_exporter(exporter: str | Path) -> Path:
    """Busca el exportador en cwd o en la raíz del repo."""
    p = Path(exporter)
    if p.is_file():
        return p
    root = Path(__file__).resolve().parents[2]
    candidate = root / "ingest" / "export180.exe"
    return candidate if candidate.is_file() else p


def _merge_slots(exam: Path, parent: Path) -> Path:
    """Copia los slots a una carpeta temporal y devuelve esa carpeta."""
    tmp = Path(tempfile.mkdtemp(prefix=f"holterai_{exam.name}_", dir=parent))
    try:
        for slot in slot_dirs(exam):
            for p in slot.iterdir():
                if p.is_file():
                    shutil.copy2(p, tmp / p.name)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return tmp


def header_samples(hea: Path) -> int | None:
    """Muestras del primer canal según el `.hea`, o None si no se puede leer."""
    try:
        first = hea.read_text(encoding="utf-8", errors="replace").splitlines()[0].split()
        return int(first[3])
    except (OSError, IndexError, ValueError):
        return None


def export_hscribe(
    src: str | Path,
    out_base: str | Path,
    *,
    seconds: int = 600,
    start_s: float = 0.0,
    dll_dir: str | Path,
    exporter: str | Path = "ingest/export180.exe",
) -> Path:
    """Exporta `src` a `<out_base>.dat/.hea` y devuelve el .hea.

    Exporta la ventana `[start_s, start_s + seconds)`. Con `seconds=0` se
    exporta el registro completo: el exportador avanza hasta encontrar el
    final real de la señal (GetSummary no es fiable como duración).
    """
    exam = Path(src)
    out = Path(out_base)
    exe = _resolve_exporter(exporter)
    dll = Path(dll_dir)
    if not exam.is_dir():
        raise FileNotFoundError(f"no existe el examen: {exam}")
    if not exe.is_file():
        raise FileNotFoundError(f"no existe el exportador: {exe}")
    if not dll.is_dir():
        raise FileNotFoundError(f"no existe la carpeta de HScribe: {dll}")
    if seconds < 0:
        raise ValueError(f"seconds no puede ser negativo: {seconds}")
    if start_s < 0:
        raise ValueError(f"start_s no puede ser negativo: {start_s}")
    out.parent.mkdir(parents=True, exist_ok=True)

    working = exam
    merged: Path | None = None
    if slot_dirs(exam):
        merged = _merge_slots(exam, out.parent)
        working = merged
    try:
        cmd = [
            str(exe),
            str(working),
            str(out),
            str(int(seconds)),
            str(dll),
            f"{float(start_s):.3f}",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"exportador falló ({proc.returncode}): {proc.stdout}\n{proc.stderr}"
            )
        # Copiar Beats.dat antes de borrar la carpeta temporal fusionada.
        beats = working / "Beats.dat"
        if beats.is_file():
            shutil.copy2(beats, out.with_suffix(".beats.dat"))
    finally:
        if merged is not None:
            shutil.rmtree(merged, ignore_errors=True)

    hea = out.with_suffix(".hea")
    if not hea.is_file():
        raise RuntimeError(f"el exportador no creó {hea}")
    n_samples = header_samples(hea)
    if n_samples is not None and n_samples <= 0:
        raise RuntimeError(
            f"ventana vacía: start_s={start_s:g} s, seconds={seconds} s ({hea})"
        )
    with hea.open("a", encoding="utf-8") as fh:
        fh.write(f"# start_s: {start_s:g}\n")
    return hea
