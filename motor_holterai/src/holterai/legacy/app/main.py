"""Punto de entrada de HolterIA: arranca el servidor local y abre la UI."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "vendor")):
    if p not in sys.path:
        sys.path.insert(0, p)

from holterai.legacy.app.core import config  # noqa: E402
from holterai.legacy.app.core.logging import get_logger  # noqa: E402
from holterai.legacy.app.server import serve  # noqa: E402

log = get_logger(__name__)

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


def open_ui(url: str, app_mode: bool = True) -> None:
    if app_mode:
        for exe in BROWSERS:
            if Path(exe).exists():
                try:
                    subprocess.Popen([
                        exe, f"--app={url}", "--window-size=1600,980",
                        "--user-data-dir=" + str(config.DATA_DIR / "browser_profile"),
                    ], close_fds=True)
                    return
                except Exception:
                    continue
    webbrowser.open(url)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="HolterIA")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--host", default=None)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--window", action="store_true", default=True)
    ap.add_argument("--open", default="")
    args = ap.parse_args(argv)

    httpd, url = serve(host=args.host, port=args.port)
    print(f"[HolterIA] {url}", flush=True)

    if not args.no_browser:
        threading.Timer(0.8, lambda: open_ui(url, app_mode=True)).start()

    if args.open:
        # carga automática opcional de un registro
        import json
        import urllib.request
        payload = json.dumps({"path": args.open}).encode()
        req = urllib.request.Request(url + "/api/import", data=payload,
                                     headers={"Content-Type": "application/json"})
        threading.Timer(3.0, lambda: _try(req)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
    return 0


def _try(req) -> None:
    import urllib.request
    try:
        urllib.request.urlopen(req, timeout=600)
    except Exception as exc:
        log.warning("no se pudo abrir el registro inicial: %s", exc)


if __name__ == "__main__":
    raise SystemExit(main())

