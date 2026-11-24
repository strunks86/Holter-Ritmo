"""Instalación de dependencias binarias sin pip (descarga + extracción).

pip está roto en este equipo; los wheels de PyPI son ZIP, así que se
descargan con urllib y se extraen en vendor/ (añadido a sys.path).
"""
from __future__ import annotations

import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor"


def _pick_wheel(meta: dict) -> dict:
    """Elige el wheel adecuado para este sistema (Windows o Linux)."""
    import platform
    urls = meta.get("urls") or []
    win = platform.system() == "Windows"
    tags = (
        ("cp312-cp312-win_amd64", "py3-none-any", "none-any") if win
        else ("cp312-cp312-manylinux", "cp312-cp312-linux_x86_64", "py3-none-any", "none-any")
    )
    for tag in tags:
        for f in urls:
            if tag in f["filename"] and f["filename"].endswith(".whl"):
                return f
    raise RuntimeError(f"sin wheel compatible para {meta['info']['name']}")


def _wheel_url(pkg: str, tag_part: str) -> tuple[str, str]:
    meta = json.load(urllib.request.urlopen(
        f"https://pypi.org/pypi/{pkg}/json", timeout=60))
    f = _pick_wheel(meta)
    return f["url"], f["filename"]


def install(pkg: str, tag_part: str = "cp312-cp312-win_amd64") -> Path:
    url, fn = _wheel_url(pkg, tag_part)
    VENDOR.mkdir(parents=True, exist_ok=True)
    dest = VENDOR / fn
    if not dest.exists():
        print(f"descargando {fn} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "HolterIA/1.0"})
        data = urllib.request.urlopen(req, timeout=600).read()
        dest.write_bytes(data)
        print(f"  {len(data)/1e6:.1f} MB")
    target = VENDOR
    with zipfile.ZipFile(dest) as z:
        z.extractall(target)
    # .data/ o rutas con prefijo
    for moved in target.rglob("*.data/purelib"):
        pass
    print(f"extraído en {target}")
    return target


def ensure_path() -> None:
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))


if __name__ == "__main__":
    args = sys.argv[1:] or ["onnxruntime"]
    install(args[0] if args[0] != "onnxruntime" else "onnxruntime")

