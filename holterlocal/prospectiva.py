"""Pestañas Prospectiva (revisión cronológica página a página con criterios de parada) y Superposición
(latidos superpuestos que acumulan brillo; los ventriculares a la derecha)."""
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
                               QRadioButton, QSplitter, QVBoxLayout, QWidget)

from .estilo import PAPEL
from .motor import VENTRICULAR
from .tira import linea, senal

CRITERIOS = [
    ('v', 'Ventricular aislada'), ('vpar', 'Par ventricular'), ('vrun', 'Racha ventricular / TV'), ('vbt', 'Bigeminismo o trigeminismo V'),
    ('ront', 'R sobre T'), ('s', 'Supraventricular aislada'), ('spar', 'Par supraventricular'), ('srun', 'Racha supraventricular / TSV'),
    ('sbt', 'Bigeminismo o trigeminismo S'), ('pausa', 'Pausa'), ('fa', 'Inicio de fibrilación auricular'), ('art', 'Artefacto'),
    ('brady', 'Bradicardia'), ('tachy', 'Taquicardia'), ('st', 'Cambio del ST'), ('morf', 'Nueva morfología'),
    ('estim', 'Latido estimulado'), ('usuario', 'Suceso del usuario o del diario'), ('otros', 'Hallazgos de holterai (bloqueos AV, etc.)'),
]
EV_CRIT = {'vpar': ['vPair'], 'vrun': ['vt', 'aivr', 'ivr'], 'vbt': ['vBig', 'vTrig'], 'ront': ['rOnT'], 'spar': ['sPair'],
           'srun': ['svt', 'sRun'], 'sbt': ['sBig', 'sTrig'], 'pausa': ['pause'], 'fa': ['af'], 'art': ['artifact', 'nosig'],
           'brady': ['brady'], 'tachy': ['tachy'], 'st': ['stDep', 'stEle'], 'usuario': ['u1', 'u2', 'u3', 'diary']}
VELOCIDADES = [2600, 1600, 1000, 600, 300]   # ms por página


class VistaProspectiva(QWidget):
    estado = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.panel = None; self.corriendo = False
        lay = QHBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        izq = QVBoxLayout()
        nav = QHBoxLayout()
        self.b_ini = QPushButton('Inicio (F7)'); self.b_ini.setObjectName('primario'); self.b_par = QPushButton('Parar (F8)')
        self.b_ini.clicked.connect(self.iniciar); self.b_par.clicked.connect(self.parar)
        nav.addWidget(self.b_ini); nav.addWidget(self.b_par)
        self.b_reset = QPushButton('Volver al principio'); self.b_reset.clicked.connect(self.reiniciar); nav.addWidget(self.b_reset)
        nav.addStretch(1); izq.addLayout(nav)
        vel = QHBoxLayout(); vel.addWidget(QLabel('Velocidad'))
        self.bg = QButtonGroup(self)
        for i, n in enumerate(['1 (lenta)', '2', '3', '4', '5 (rápida)', 'InstaPágina']):
            r = QRadioButton(n); r.setChecked(i == 2); self.bg.addButton(r, i); vel.addWidget(r)
        vel.addStretch(1); izq.addLayout(vel)
        g = QGroupBox('Detenerse en'); gl = QGridLayout(g); self.chk = {}
        for i, (k, n) in enumerate(CRITERIOS):
            c = QCheckBox(n); c.setChecked(k not in ('v', 's', 'estim')); self.chk[k] = c; gl.addWidget(c, i // 3, i % 3)
        b1 = QPushButton('Todos'); b2 = QPushButton('Ninguno')
        b1.clicked.connect(lambda: [c.setChecked(True) for c in self.chk.values()]); b2.clicked.connect(lambda: [c.setChecked(False) for c in self.chk.values()])
        fila = QHBoxLayout(); fila.addWidget(b1); fila.addWidget(b2); fila.addStretch(1); gl.addLayout(fila, 99, 0, 1, 3)
        izq.addWidget(g)
        self.sup_chk = QCheckBox('Mostrar la superposición de la página'); izq.addWidget(self.sup_chk)
        self.motivo = QLabel('La revisión prospectiva recorre el ECG en orden y se detiene donde marcaste.'); self.motivo.setWordWrap(True)
        self.motivo.setObjectName('titulo'); izq.addWidget(self.motivo)
        izq.addWidget(QLabel('Al detenerse puedes etiquetar latidos en la tira de abajo. Clic derecho en un latido → «Aprender» cambia toda su morfología.'))
        izq.addStretch(1)
        lay.addLayout(izq, 3)
        self.sup = Superpuesto(self); self.sup.setVisible(False); lay.addWidget(self.sup, 2)
        self.sup_chk.toggled.connect(self._sup)
        self.timer = QTimer(self); self.timer.timeout.connect(self._paso)

    def set_panel(self, panel):
        self.panel = panel

    def mostrar(self, ex):
        self.ex = ex
        tid = ex.tid
        ok = tid >= 0
        self.primeras = np.array([], float)
        if ok.any():
            _, first = np.unique(tid[ok], return_index=True)
            self.primeras = np.sort(ex.t[np.nonzero(ok)[0][first]])
        self.sup.ex = ex

    def _sup(self, on):
        self.sup.setVisible(on); self._pinta_sup()

    def _pinta_sup(self):
        if self.sup.isVisible() and self.ex and self.panel:
            t0 = self.panel.tira.t0; t1 = t0 + self.panel.tira.ancho_seg()
            b0, b1 = np.searchsorted(self.ex.t, [t0, t1])
            self.sup.limpiar(); self.sup.agregar(range(int(b0), int(b1)))

    def motivos(self, ta, tb):
        ex = self.ex; out = []
        on = {k for k, c in self.chk.items() if c.isChecked()}
        for k, tipos in EV_CRIT.items():
            if k in on:
                for e in ex.eventos:
                    if e['type'] in tipos and ta <= e['t0'] < tb:
                        out.append((e['t0'], e.get('label') or e['type'], k)); break
        if 'otros' in on:
            for e in ex.eventos:
                if str(e['type']).startswith('ha_') and ta <= e['t0'] < tb:
                    out.append((e['t0'], e.get('label', e['type']), 'otros')); break
        b0, b1 = np.searchsorted(ex.t, [ta, tb])
        seg = ex.cls[b0:b1]
        for k, clases, n in (('v', VENTRICULAR, 'Ventricular'), ('s', (1,), 'Supraventricular'), ('estim', (9, 12, 13), 'Estimulado')):
            if k in on:
                idx = np.nonzero(np.isin(seg, clases))[0]
                if len(idx):
                    out.append((float(ex.t[b0 + idx[0]]), n, k))
        if 'morf' in on and len(self.primeras):
            i = np.searchsorted(self.primeras, ta)
            if i < len(self.primeras) and self.primeras[i] < tb:
                out.append((float(self.primeras[i]), 'Nueva morfología', 'morf'))
        out.sort()
        return out

    def iniciar(self):
        if not self.ex or not self.panel:
            return
        self.corriendo = True; self.b_ini.setEnabled(False)
        self.motivo.setText('Revisando…')
        self.timer.start(30 if self.bg.checkedId() == 5 else VELOCIDADES[max(0, self.bg.checkedId())])

    def parar(self, motivo=''):
        self.timer.stop(); self.corriendo = False; self.b_ini.setEnabled(True)
        if motivo:
            self.motivo.setText(motivo); self.estado.emit(motivo)
        elif self.ex:
            self.motivo.setText('Detenido en ' + self.ex.reloj(self.panel.tira.t0))
        self._pinta_sup()

    def reiniciar(self):
        self.parar()
        if self.panel:
            self.panel.tira.ir_a(0, centrar=False)
        self.motivo.setText('Listo para empezar desde el principio.')

    def alternar(self):
        self.parar() if self.corriendo else self.iniciar()

    def _paso(self):
        tira = self.panel.tira; ex = self.ex
        w = tira.ancho_seg()
        insta = self.bg.checkedId() == 5
        for _ in range(400 if insta else 1):
            t0 = tira.t0 + w
            if t0 >= ex.dur - 0.5:
                tira.ir_a(max(0, ex.dur - w), centrar=False); self.parar('Llegó al final del registro.'); return
            tira.ir_a(t0, centrar=False)
            m = self.motivos(t0, t0 + w)
            if m:
                t, n, _ = m[0]
                b = ex.latido_cercano(t, 0.3)
                if b >= 0:
                    tira.sel = b; tira.sels = {b}; tira.update()
                self.parar('Detenido: %s a las %s' % (n, ex.reloj(t))); return
        self._pinta_sup()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_F7:
            self.alternar()
        elif e.key() == Qt.Key_F8:
            self.parar()
        else:
            super().keyPressEvent(e)


class Superpuesto(QWidget):
    """Lienzo que acumula latidos: normales y supraventriculares a la izquierda, ventriculares a la derecha."""
    PRE, POST = 0.25, 0.45

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.img = None; self.ultimo = ''; self.oscuro = True; self.n = [0, 0]
        self.setMinimumSize(260, 200)

    def resizeEvent(self, e):
        self.limpiar(); super().resizeEvent(e)

    def limpiar(self):
        self.img = QImage(max(10, self.width()), max(10, self.height()), QImage.Format_ARGB32_Premultiplied)
        self.img.fill(QColor('#05080d')); self.n = [0, 0]; self.update()

    def agregar(self, beats):
        ex = self.ex
        if ex is None or self.img is None:
            return
        p = QPainter(self.img); p.setRenderHint(QPainter.Antialiasing, True)
        W, H = self.img.width(), self.img.height(); nch = min(3, len(ex.chans)); hc = (H - 18) / max(1, nch)
        half = W / 2
        pxs = (half - 10) / (self.PRE + self.POST); pxmv = hc * 0.35
        for b in beats:
            k = int(ex.cls[b])
            if k == 4:
                continue
            lado = 1 if k in VENTRICULAR else 0
            col = QColor(255, 90, 90, 26) if lado else QColor(120, 230, 160, 26)
            p.setPen(QPen(col, 1.0))
            t = float(ex.t[b]); i0 = int((t - self.PRE) * ex.fs); i1 = int((t + self.POST) * ex.fs)
            for ci in range(nch):
                y = senal(ex, ex.chans[ci], i0, i1)
                if len(y) < 4:
                    continue
                y = y - np.median(y[:max(2, int(0.06 * ex.fs))])
                mid = 18 + hc * (ci + 0.6)
                p.drawPolyline(linea(y, ex.fs, i0 / ex.fs, t - self.PRE, lado * half + 5, pxs, mid, pxmv, 18 + ci * hc, 18 + (ci + 1) * hc))
            self.n[lado] += 1; self.ultimo = ex.reloj(t)
        p.end(); self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        if self.img is not None:
            p.drawImage(0, 0, self.img)
        p.setPen(QColor('#e5e7eb')); f = QFont(self.font()); f.setPointSizeF(8.5); p.setFont(f)
        p.drawText(QPointF(6, 13), 'Normales y SV: %d' % self.n[0]); p.drawText(QPointF(self.width() / 2 + 6, 13), 'Ventriculares: %d' % self.n[1])
        if self.ultimo:
            p.drawText(QPointF(self.width() - 70, self.height() - 6), self.ultimo)
        p.setPen(QPen(QColor('#374151'), 1)); p.drawLine(QPointF(self.width() / 2, 0), QPointF(self.width() / 2, self.height()))
        p.end()


class VistaSuperposicion(QWidget):
    ir = Signal(float, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.pos = 0; self.dir = 1
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        bar = QHBoxLayout()
        self.b_atras = QPushButton('◀ Hacia atrás'); self.b_av = QPushButton('Avanzar ▶ (F7)'); self.b_av.setObjectName('primario')
        self.b_stop = QPushButton('■ Parar'); self.b_clr = QPushButton('Borrar la pantalla')
        for b in (self.b_atras, self.b_av, self.b_stop, self.b_clr):
            bar.addWidget(b)
        bar.addSpacing(12); bar.addWidget(QLabel('Velocidad'))
        self.bg = QButtonGroup(self)
        for i in range(5):
            r = QRadioButton(str(i + 1)); r.setChecked(i == 2); self.bg.addButton(r, i); bar.addWidget(r)
        self.lbl = QLabel(''); self.lbl.setObjectName('reloj'); bar.addStretch(1); bar.addWidget(self.lbl)
        lay.addLayout(bar)
        self.lienzo = Superpuesto(self); lay.addWidget(self.lienzo, 1)
        lay.addWidget(QLabel('Sirve para ver cambios de PR, QRS y ST-T a medida que ocurren. Al parar, la tira de abajo muestra el último latido.'))
        self.timer = QTimer(self); self.timer.timeout.connect(self._tick)
        self.b_av.clicked.connect(lambda: self.correr(1)); self.b_atras.clicked.connect(lambda: self.correr(-1))
        self.b_stop.clicked.connect(self.parar); self.b_clr.clicked.connect(self.lienzo.limpiar)

    def mostrar(self, ex):
        if self.ex is not ex:
            self.pos = 0
        self.ex = ex; self.lienzo.ex = ex

    def desde(self, t):
        if self.ex:
            self.pos = int(np.searchsorted(self.ex.t, t))

    def correr(self, d):
        if not self.ex:
            return
        self.dir = d; self.timer.start(60)

    def alternar(self):
        self.parar() if self.timer.isActive() else self.correr(1)

    def parar(self):
        if self.timer.isActive():
            self.timer.stop()
            if self.ex and len(self.ex.t):
                b = int(np.clip(self.pos, 0, len(self.ex.t) - 1)); self.ir.emit(float(self.ex.t[b]), b)

    def _tick(self):
        ex = self.ex; n = [2, 5, 12, 30, 80][max(0, self.bg.checkedId())]
        if self.dir > 0:
            a, z = self.pos, min(len(ex.t), self.pos + n)
            beats = range(a, z); self.pos = z
        else:
            a, z = max(0, self.pos - n), self.pos
            beats = range(z - 1, a - 1, -1); self.pos = a
        if not len(beats):
            self.parar(); return
        self.lienzo.agregar(beats); self.lbl.setText(self.lienzo.ultimo)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_F7, Qt.Key_F8):
            self.alternar()
        else:
            super().keyPressEvent(e)
