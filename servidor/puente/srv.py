"""Puente HolterLocal: servicio minimo para que Claude trabaje en este servidor.

Escucha solo en 127.0.0.1:8765 (sale a internet unicamente por el tunel de Cloudflare)
y exige la clave del encabezado X-T en cada pedido. Autorizado por Jorge el 19-nov-2026.
  POST /run?t=600   cuerpo = comando bash; devuelve stdout+stderr y el codigo de salida
  POST /put?f=ruta  cuerpo = bytes del archivo (&ap=1 agrega al final)
  GET  /get?f=ruta&o=desde&n=cuantos  lee bytes de un archivo
Para cerrarlo: pkill -f puente/vig.sh; pkill -f puente/srv.py; pkill cloudflared
"""
import http.server
import os
import subprocess
import sys
import urllib.parse as up

TOKEN = open(sys.argv[1]).read().strip()


class H(http.server.BaseHTTPRequestHandler):
    def reply(self, body, code=200):
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def args(self):
        if self.headers.get("X-T") != TOKEN:
            self.reply(b"clave incorrecta", 403)
            return None
        u = up.urlparse(self.path)
        return u.path, dict(up.parse_qsl(u.query))

    def do_POST(self):
        r = self.args()
        if not r:
            return
        path, a = r
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if path == "/run":
            try:
                o = subprocess.run(["bash", "-c", body.decode()], capture_output=True,
                                   timeout=float(a.get("t", 600)))
                self.reply(o.stdout + o.stderr + b"\n[rc=%d]" % o.returncode)
            except Exception as e:  # noqa: BLE001
                self.reply(("error: %s" % e).encode())
        elif path == "/put":
            os.makedirs(os.path.dirname(a["f"]) or ".", exist_ok=True)
            with open(a["f"], "ab" if a.get("ap") else "wb") as f:
                f.write(body)
            self.reply(b"ok %d" % len(body))
        else:
            self.reply(b"?", 404)

    def do_GET(self):
        r = self.args()
        if not r:
            return
        _, a = r
        with open(a["f"], "rb") as f:
            f.seek(int(a.get("o", 0)))
            self.reply(f.read(int(a.get("n", 50_000_000))))

    def log_message(self, *x):
        pass


http.server.ThreadingHTTPServer(("127.0.0.1", 8765), H).serve_forever()
