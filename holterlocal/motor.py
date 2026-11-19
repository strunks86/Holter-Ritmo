"""Puente con el motor de interpretación (servidor.js en Node) y el examen cargado en memoria."""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
from PySide6.QtCore import QObject, QProcess, Signal

DTYPES = {
    'Int16Array': np.int16, 'Int32Array': np.int32, 'Uint8Array': np.uint8, 'Int8Array': np.int8,
    'Uint16Array': np.uint16, 'Uint32Array': np.uint32, 'Float32Array': np.float32, 'Float64Array': np.float64,
}

CLS_SYM = ['N', 'S', 'V', 'F', 'X', 'Q', 'B', 'A', 'E', 'P']
CLS_NAME = ['Normal', 'Supraventricular', 'Ventricular', 'Fusión', 'Artefacto', 'No clasificable',
            'Bloqueo de rama', 'Aberrante', 'Escape ventricular', 'Estimulado']

EVENT_NAMES = [
    ('vt', 'Taquicardia ventricular'), ('aivr', 'Ritmo idioventricular acelerado'), ('ivr', 'Ritmo idioventricular'),
    ('vPair', 'Pares ventriculares'), ('vIso', 'Ventriculares aisladas'), ('vBig', 'Bigeminismo ventricular'),
    ('vTrig', 'Trigeminismo ventricular'), ('rOnT', 'R sobre T'), ('interp', 'Ventriculares interpoladas'),
    ('fus', 'Latidos de fusión'), ('svt', 'Taquicardia supraventricular'), ('sRun', 'Rachas supraventriculares lentas'),
    ('sPair', 'Pares supraventriculares'), ('sIso', 'Supraventriculares aisladas'), ('sBig', 'Bigeminismo supraventricular'),
    ('sTrig', 'Trigeminismo supraventricular'), ('pause', 'Pausas'), ('brady', 'Bradicardia'), ('tachy', 'Taquicardia'),
    ('af', 'Fibrilación auricular'), ('stDep', 'Descenso del ST'), ('stEle', 'Elevación del ST'),
    ('doubt', 'Latidos dudosos'), ('nosig', 'Sin señal'), ('artifact', 'Artefacto'),
]
EVENT_LABEL = dict(EVENT_NAMES)


def recurso(*partes):
    """Ruta a un archivo que viaja con la aplicación (también dentro del .exe)."""
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *partes)


def buscar_node():
    for p in (recurso('node', 'node.exe'), recurso('node', 'bin', 'node'), recurso('node', 'node')):
        if os.path.isfile(p):
            return p
    return shutil.which('node') or shutil.which('node.exe')


def leer_arreglo(info):
    return np.fromfile(info['file'], dtype=DTYPES[info['dtype']], count=info['n'])


class Examen:
    """Señal y resultados del análisis de un registro."""

    def __init__(self, res, rutas):
        self.rutas = rutas
        r = res['rec']
        self.nombre = r.get('name') or os.path.basename(rutas[0])
        self.fuente = r.get('source') or ''
        self.fs = float(r['fs'])
        self.n = int(r['n'])
        self.nsig = int(r['nsig'])
        self.gain = [float(g) or 1.0 for g in r['gain']]
        self.baseline = [float(b) for b in r['baseline']]
        self.names = list(r.get('names') or [f'Canal {i + 1}' for i in range(self.nsig)])
        self.start = r.get('start')
        self.sig = [leer_arreglo(s) for s in r['sig']]
        self.dur = self.n / self.fs
        ref = res.get('ref')
        self.ref = None
        if ref:
            self.ref = {'file': ref['file'], 't': np.asarray(ref['t'], float), 'sym': ref['sym']}
        self.actualizar(res['A'])

    def actualizar(self, A):
        self.A = A
        f = A['files']
        self.t = leer_arreglo(f['t'])
        self.cls = leer_arreglo(f['cls'])
        self.auto = leer_arreglo(f['auto'])
        self.edited = leer_arreglo(f['edited'])
        self.tid = leer_arreglo(f['tid'])
        self.pV = leer_arreglo(f['pV'])
        self.pX = leer_arreglo(f['pX'])
        self.rr = leer_arreglo(f['rr'])
        self.prem = leer_arreglo(f['prem'])
        self.hr8 = leer_arreglo(f['hr8'])
        self.afBeat = leer_arreglo(f['afBeat'])
        self.best = leer_arreglo(f['best'])
        self.excl = leer_arreglo(f['excl'])
        self.qual = [leer_arreglo(f['qual%d' % i]) for i in range(A['nch'])]
        self.chans = A['chans']
        self.eventos = A['events']

    # ---- utilidades
    def mv(self, ch, i0, i1):
        i0 = max(0, int(i0)); i1 = min(self.n, int(i1))
        if i1 <= i0:
            return np.zeros(0, np.float32)
        return (self.sig[ch][i0:i1].astype(np.float32) - self.baseline[ch]) / self.gain[ch]

    def seg_inicio(self):
        s = self.start
        if not s:
            return 0
        return (s.get('h', 0) * 3600 + s.get('mi', 0) * 60 + s.get('s', 0))

    def reloj(self, t, segundos=True):
        x = int(round(self.seg_inicio() + t))
        dia, x = divmod(x, 86400)
        h, r = divmod(x, 3600); m, s = divmod(r, 60)
        txt = f'{h:02d}:{m:02d}:{s:02d}' if segundos else f'{h:02d}:{m:02d}'
        return (f'D{dia + 1} ' if dia > 0 else '') + txt

    def latido_cercano(self, t, tol=0.2):
        if not len(self.t):
            return -1
        i = int(np.searchsorted(self.t, t))
        best, bd = -1, tol
        for j in (i - 1, i):
            if 0 <= j < len(self.t) and abs(self.t[j] - t) <= bd:
                best, bd = j, abs(self.t[j] - t)
        return best


class Motor(QObject):
    """Mantiene vivo el proceso del motor y le pasa órdenes."""
    progreso = Signal(str, float)
    listo = Signal(str)
    fallo = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.proc = None
        self.buf = b''
        self.pend = {}
        self.sig_id = 1
        self.version = ''
        self.tmp = tempfile.mkdtemp(prefix='holterlocal_')
        self.n_open = 0

    def iniciar(self):
        node = buscar_node()
        if not node:
            self.fallo.emit('No encontré el motor (falta node).')
            return False
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.SeparateChannels)
        self.proc.readyReadStandardOutput.connect(self._leer)
        self.proc.readyReadStandardError.connect(self._err)
        self.proc.finished.connect(self._fin)
        self.proc.start(node, ['--max-old-space-size=6144', recurso('motor_js', 'servidor.js')])
        return self.proc.waitForStarted(15000)

    def _err(self):
        data = bytes(self.proc.readAllStandardError()).decode('utf-8', 'replace')
        if data.strip():
            sys.stderr.write(data)

    def _fin(self, *a):
        for cb, eb in list(self.pend.values()):
            if eb:
                eb('El motor se cerró inesperadamente.')
        self.pend.clear()

    def _leer(self):
        self.buf += bytes(self.proc.readAllStandardOutput())
        while b'\n' in self.buf:
            line, self.buf = self.buf.split(b'\n', 1)
            if not line.strip():
                continue
            try:
                m = json.loads(line.decode('utf-8'))
            except Exception:
                continue
            ty = m.get('type')
            if ty == 'ready':
                self.version = m.get('version', '')
                self.listo.emit(self.version)
            elif ty == 'progress':
                self.progreso.emit(m.get('msg', ''), float(m.get('p', 0)))
            elif ty == 'result':
                cb, eb = self.pend.pop(m.get('id'), (None, None))
                if m.get('error'):
                    if eb:
                        eb(m['error'])
                elif cb:
                    cb(m)

    def llamar(self, cmd, cb=None, eb=None, **kw):
        if not self.proc or self.proc.state() != QProcess.Running:
            if not self.iniciar():
                if eb:
                    eb('No se pudo iniciar el motor.')
                return
        i = self.sig_id; self.sig_id += 1
        self.pend[i] = (cb, eb)
        kw.update(cmd=cmd, id=i)
        self.proc.write((json.dumps(kw) + '\n').encode('utf-8'))

    def abrir(self, rutas, cb, eb, settings=None, opt=None):
        self.n_open += 1
        out = os.path.join(self.tmp, 'ex%d' % self.n_open)
        self.llamar('open', cb, eb, paths=rutas, out=out, settings=settings or {}, opt=opt or {})

    def cerrar(self):
        if self.proc:
            self.proc.kill(); self.proc.waitForFinished(2000)
        shutil.rmtree(self.tmp, ignore_errors=True)
