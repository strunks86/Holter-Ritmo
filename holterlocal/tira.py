"""Tira de ECG en papel milimetrado: varios canales, etiquetas de latido, calibrador y edición."""
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QMenu, QPushButton, QScrollBar,
                               QSizePolicy, QToolButton, QVBoxLayout, QWidget)
from scipy.signal import butter, sosfiltfilt

from .estilo import COLOR_CLASE, C
from .motor import CLS_NAME, CLS_SYM

TECLAS = {Qt.Key_N: 0, Qt.Key_S: 1, Qt.Key_V: 2, Qt.Key_F: 3, Qt.Key_X: 4, Qt.Key_Q: 5,
          Qt.Key_B: 6, Qt.Key_A: 7, Qt.Key_E: 8, Qt.Key_P: 9}


class Tira(QWidget):
    tiempo_cambiado = Signal(float)
    latido_elegido = Signal(int)
    reetiquetar = Signal(list, int)
    excluir = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None
        self.t0 = 0.0
        self.vel = 25.0      # mm/s
        self.gan = 10.0      # mm/mV
        self.filtro = True
        self.sel = -1
        self.cal = None      # (t_a, t_b, y_a, y_b)
        self.rango = None    # selección con Mayúscula
        self._arr = None
        self.ver_ref = False
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # ---- geometría
    def pxmm(self):
        return max(3.0, self.logicalDpiX() / 25.4)

    def ancho_seg(self):
        return self.width() / (self.vel * self.pxmm())

    def x_de(self, t):
        return (t - self.t0) * self.vel * self.pxmm()

    def t_de(self, x):
        return self.t0 + x / (self.vel * self.pxmm())

    def set_examen(self, ex):
        self.ex = ex; self.t0 = 0.0; self.sel = -1; self.cal = None; self.rango = None
        self.update()

    def ir_a(self, t, centrar=True):
        if not self.ex:
            return
        w = self.ancho_seg()
        t0 = t - w * 0.4 if centrar else t
        self.t0 = float(min(max(0.0, t0), max(0.0, self.ex.dur - w)))
        self.update(); self.tiempo_cambiado.emit(self.t0)

    def seleccionar(self, b, centrar=True):
        if not self.ex or b < 0 or b >= len(self.ex.t):
            return
        self.sel = int(b)
        t = float(self.ex.t[b])
        if centrar or not (self.t0 + 0.3 < t < self.t0 + self.ancho_seg() - 0.3):
            self.ir_a(t)
        self.update(); self.latido_elegido.emit(self.sel)

    # ---- señal visible
    def senal(self, ch, i0, i1):
        ex = self.ex
        if not self.filtro:
            return ex.mv(ch, i0, i1), i0
        m = int(2 * ex.fs)
        a0 = max(0, i0 - m); a1 = min(ex.n, i1 + m)
        y = ex.mv(ch, a0, a1).astype(np.float64)
        if len(y) > 30:
            hi = min(40.0, 0.45 * ex.fs)
            sos = butter(2, [0.5, hi], btype='bandpass', fs=ex.fs, output='sos')
            try:
                y = sosfiltfilt(sos, y)
            except ValueError:
                pass
        return y[i0 - a0:i0 - a0 + (i1 - i0)], i0

    # ---- dibujo
    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor(C['papel']))
        ex = self.ex
        if not ex:
            p.setPen(QColor(C['texto2']))
            p.drawText(self.rect(), Qt.AlignCenter, 'Abre un examen con Archivo → Abrir examen (Ctrl+O)')
            return
        pm = self.pxmm()
        top = 30  # banda de etiquetas
        # cuadrícula
        mm = pm
        fino = QPen(QColor(C['rejilla1'])); fino.setWidthF(0.6)
        grueso = QPen(QColor(C['rejilla2'])); grueso.setWidthF(1.0)
        off = (self.t0 * self.vel * pm) % (5 * mm)
        k = 0
        x = -off
        while x < W:
            for j in range(5):
                xx = x + j * mm
                if 0 <= xx < W:
                    p.setPen(grueso if j == 0 else fino); p.drawLine(QPointF(xx, top), QPointF(xx, H))
            x += 5 * mm
        y = top; j = 0
        while y < H:
            p.setPen(grueso if j % 5 == 0 else fino); p.drawLine(QPointF(0, y), QPointF(W, y)); y += mm; j += 1
        # calidad por segundo (gris = inservible en todos los canales)
        t1 = self.t0 + self.ancho_seg()
        s0, s1 = int(self.t0), min(int(t1) + 1, len(ex.best))
        nch = ex.A['nch']
        chs = ex.chans
        hc = (H - top) / max(1, nch)
        for s in range(max(0, s0), s1):
            xa, xb = self.x_de(s), self.x_de(s + 1)
            if ex.excl[s]:
                p.fillRect(QRectF(xa, top, xb - xa, H - top), QColor(120, 120, 120, 45))
            for ci in range(nch):
                q = ex.qual[ci][s] if s < len(ex.qual[ci]) else 0
                if q == 2:
                    p.fillRect(QRectF(xa, top + ci * hc, xb - xa, hc), QColor(230, 160, 40, 40))
        # selección de rango
        if self.rango:
            a, b = sorted(self.rango)
            p.fillRect(QRectF(self.x_de(a), top, self.x_de(b) - self.x_de(a), H - top), QColor(60, 120, 220, 50))
        # canales
        i0 = int(self.t0 * ex.fs); i1 = int(t1 * ex.fs) + 2
        self._arr = []
        fnt = QFont(self.font()); fnt.setPointSizeF(8.5)
        for ci, ch in enumerate(chs):
            mid = top + hc * (ci + 0.55)
            y, _ = self.senal(ch, i0, i1)
            self._arr.append((mid, y))
            if len(y) < 2:
                continue
            xs = (np.arange(len(y)) / ex.fs + i0 / ex.fs - self.t0) * self.vel * pm
            ys = mid - y * self.gan * pm
            ys = np.clip(ys, top + ci * hc - 0.5 * hc, top + (ci + 1) * hc + 0.5 * hc)
            poly = QPolygonF([QPointF(float(a), float(b)) for a, b in zip(xs, ys)])
            pen = QPen(QColor(C['trazo'])); pen.setWidthF(1.15)
            p.setPen(pen); p.drawPolyline(poly)
            # pulso de calibración de 1 mV
            p.setPen(QPen(QColor(C['texto']), 1.2))
            cx = 4; base = mid
            path = QPainterPath(QPointF(cx, base)); path.lineTo(cx + 2, base); path.lineTo(cx + 2, base - self.gan * pm)
            path.lineTo(cx + 2 + 0.2 * self.vel * pm, base - self.gan * pm); path.lineTo(cx + 2 + 0.2 * self.vel * pm, base)
            path.lineTo(cx + 4 + 0.2 * self.vel * pm, base); p.drawPath(path)
            p.setFont(fnt); p.setPen(QColor(C['texto2']))
            p.drawText(QPointF(8, top + ci * hc + 14), ex.names[ch] if ch < len(ex.names) else f'Canal {ch + 1}')
        # latidos
        b0 = int(np.searchsorted(ex.t, self.t0)); b1 = int(np.searchsorted(ex.t, t1))
        f2 = QFont(self.font()); f2.setBold(True); f2.setPointSizeF(9.5)
        f3 = QFont(self.font()); f3.setPointSizeF(7.5)
        for b in range(b0, b1):
            x = self.x_de(ex.t[b]); k = int(ex.cls[b])
            if b == self.sel:
                p.fillRect(QRectF(x - 9, 0, 18, H), QColor(255, 220, 0, 70))
            col = QColor(COLOR_CLASE[k])
            p.setPen(col); p.setFont(f2)
            p.drawText(QRectF(x - 10, 2, 20, 14), Qt.AlignCenter, CLS_SYM[k] + ('*' if ex.edited[b] else ''))
            if ex.afBeat[b]:
                p.fillRect(QRectF(x - 6, 16, 12, 2), QColor(C['fa']))
            rr = ex.rr[b]
            if rr == rr and b > b0:
                p.setFont(f3); p.setPen(QColor(C['texto2']))
                xp = self.x_de(ex.t[b - 1])
                p.drawText(QRectF((xp + x) / 2 - 20, 17, 40, 12), Qt.AlignCenter, '%d' % round(rr * 1000))
        # anotaciones de referencia (si hay)
        if self.ver_ref and ex.ref is not None:
            rt = ex.ref['t']; r0 = int(np.searchsorted(rt, self.t0)); r1 = int(np.searchsorted(rt, t1))
            p.setFont(f3)
            for i in range(r0, r1):
                x = self.x_de(rt[i]); p.setPen(QColor(C['ref'])); p.drawText(QRectF(x - 8, H - 14, 16, 12), Qt.AlignCenter, ex.ref['sym'][i])
        # calibrador
        if self.cal:
            ta, tb, ya, yb = self.cal
            xa, xb = self.x_de(ta), self.x_de(tb)
            pen = QPen(QColor(C['calibrador'])); pen.setWidthF(1.3); pen.setStyle(Qt.DashLine)
            p.setPen(pen); p.drawLine(QPointF(xa, top), QPointF(xa, H)); p.drawLine(QPointF(xb, top), QPointF(xb, H))
            p.drawLine(QPointF(xa, ya), QPointF(xb, yb))
            dt = abs(tb - ta)
            dmv = (ya - yb) / (self.gan * pm)
            txt = '%d ms' % round(dt * 1000)
            if dt > 0.05:
                txt += '  ·  %d lpm' % round(60 / dt)
            txt += '  ·  %+.2f mV' % dmv
            p.setFont(f2)
            r = QRectF(min(xa, xb) + 4, max(top + 2, min(ya, yb) - 22), 210, 18)
            p.fillRect(r, QColor(255, 255, 255, 230)); p.setPen(QColor(C['calibrador'])); p.drawText(r, Qt.AlignLeft | Qt.AlignVCenter, ' ' + txt)
        p.end()

    # ---- ratón y teclado
    def mousePressEvent(self, e):
        self.setFocus()
        if not self.ex:
            return
        t = self.t_de(e.position().x())
        if e.button() == Qt.LeftButton:
            if e.modifiers() & Qt.ShiftModifier:
                self.rango = [t, t]; self.cal = None
            else:
                self._press = (t, e.position().y()); self.cal = None; self.rango = None
        self.update()

    def mouseMoveEvent(self, e):
        if not self.ex or not (e.buttons() & Qt.LeftButton):
            return
        t = self.t_de(e.position().x())
        if self.rango is not None:
            self.rango[1] = t
        elif getattr(self, '_press', None):
            ta, ya = self._press
            if abs(self.x_de(t) - self.x_de(ta)) > 4:
                self.cal = (ta, t, ya, e.position().y())
        self.update()

    def mouseReleaseEvent(self, e):
        if not self.ex:
            return
        t = self.t_de(e.position().x())
        if self.rango is not None:
            a, b = sorted(self.rango)
            if b - a > 0.2:
                self._menu_rango(e, a, b)
            return
        if getattr(self, '_press', None) and self.cal is None:
            b = self.ex.latido_cercano(t, 0.25)
            if b >= 0:
                self.seleccionar(b, centrar=False)
        self._press = None

    def _menu_rango(self, e, a, b):
        ex = self.ex
        m = QMenu(self)
        b0 = int(np.searchsorted(ex.t, a)); b1 = int(np.searchsorted(ex.t, b))
        m.addAction('Marcar %.1f s como artefacto (no se informa)' % (b - a), lambda: self.excluir.emit(a, b))
        sub = m.addMenu('Etiquetar los %d latidos como…' % (b1 - b0))
        for k in (0, 1, 2, 3, 4, 6, 7, 9):
            sub.addAction(CLS_NAME[k], lambda k=k: self.reetiquetar.emit(list(range(b0, b1)), k))
        m.addAction('Medir: %.0f ms, %d latidos' % ((b - a) * 1000, b1 - b0))
        m.exec(e.globalPosition().toPoint())
        self.rango = None; self.update()

    def wheelEvent(self, e):
        if not self.ex:
            return
        d = e.angleDelta().y() or e.angleDelta().x()
        if e.modifiers() & Qt.ControlModifier:
            self.gan = float(min(40, max(2.5, self.gan * (1.25 if d > 0 else 0.8)))); self.update(); return
        self.ir_a(self.t0 - np.sign(d) * self.ancho_seg() * 0.25, centrar=False)

    def keyPressEvent(self, e):
        if not self.ex:
            return
        k = e.key()
        if k in TECLAS and self.sel >= 0 and not (e.modifiers() & Qt.ControlModifier):
            self.reetiquetar.emit([self.sel], TECLAS[k]); return
        if k == Qt.Key_Right:
            self.seleccionar(self.sel + 1 if self.sel >= 0 else int(np.searchsorted(self.ex.t, self.t0)), centrar=False)
        elif k == Qt.Key_Left:
            self.seleccionar(max(0, self.sel - 1), centrar=False)
        elif k in (Qt.Key_PageDown, Qt.Key_Space):
            self.ir_a(self.t0 + self.ancho_seg() * 0.9, centrar=False)
        elif k == Qt.Key_PageUp:
            self.ir_a(self.t0 - self.ancho_seg() * 0.9, centrar=False)
        elif k == Qt.Key_Escape:
            self.cal = None; self.rango = None; self.update()
        else:
            super().keyPressEvent(e)


class PanelECG(QWidget):
    """Tira con su barra de herramientas y desplazamiento por todo el registro."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None
        self.tira = Tira(self)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(2)
        bar = QHBoxLayout(); bar.setContentsMargins(6, 4, 6, 0)
        self.lbl = QLabel('—'); self.lbl.setObjectName('reloj')
        self.vel = QComboBox(); [self.vel.addItem(f'{v} mm/s', v) for v in (6.25, 12.5, 25.0, 50.0)]; self.vel.setCurrentIndex(2)
        self.gan = QComboBox(); [self.gan.addItem(f'{v:g} mm/mV', v) for v in (2.5, 5.0, 10.0, 20.0)]; self.gan.setCurrentIndex(2)
        self.fil = QCheckBox('Filtro 0,5–40 Hz'); self.fil.setChecked(True)
        self.ref = QCheckBox('Anotaciones de referencia'); self.ref.setVisible(False)
        bar.addWidget(self.lbl); bar.addStretch(1)
        for nombre, k in (('◀ V', 2), ('V ▶', 2), ('◀ S', 1), ('S ▶', 1)):
            b = QToolButton(); b.setText(nombre); b.setToolTip('Latido %s %s' % (CLS_NAME[k].lower(), 'anterior' if nombre.startswith('◀') else 'siguiente'))
            b.clicked.connect(lambda _=False, k=k, d=(-1 if nombre.startswith('◀') else 1): self.saltar_clase(k, d)); bar.addWidget(b)
        bar.addSpacing(10)
        for w in (QLabel('Velocidad'), self.vel, QLabel('Ganancia'), self.gan, self.fil, self.ref):
            bar.addWidget(w)
        lay.addLayout(bar)
        lay.addWidget(self.tira, 1)
        self.scroll = QScrollBar(Qt.Horizontal); lay.addWidget(self.scroll)
        self.ayuda = QLabel('Clic: elegir latido · teclas N S V F B A P X: reetiquetar · arrastrar: calibrador · Mayús+arrastrar: tramo · rueda: avanzar · Ctrl+rueda: ganancia')
        self.ayuda.setObjectName('ayuda'); lay.addWidget(self.ayuda)
        self.vel.currentIndexChanged.connect(self._opc); self.gan.currentIndexChanged.connect(self._opc)
        self.fil.toggled.connect(self._opc); self.ref.toggled.connect(self._opc)
        self.scroll.valueChanged.connect(self._scroll)
        self.tira.tiempo_cambiado.connect(self._sync)

    def set_examen(self, ex):
        self.ex = ex; self.tira.set_examen(ex)
        self.scroll.setRange(0, int(max(0, ex.dur - 5) * 10)); self.scroll.setPageStep(100)
        self.ref.setVisible(ex.ref is not None)
        self._sync(0)

    def _opc(self):
        self.tira.vel = self.vel.currentData(); self.tira.gan = self.gan.currentData(); self.tira.filtro = self.fil.isChecked()
        self.tira.ver_ref = self.ref.isChecked(); self.tira.update(); self._sync(self.tira.t0)

    def _scroll(self, v):
        if abs(v / 10 - self.tira.t0) > 0.05:
            self.tira.t0 = v / 10; self.tira.update(); self._sync(self.tira.t0, mover=False)

    def _sync(self, t0, mover=True):
        if not self.ex:
            return
        if mover:
            self.scroll.blockSignals(True); self.scroll.setValue(int(t0 * 10)); self.scroll.blockSignals(False)
        w = self.tira.ancho_seg()
        self.lbl.setText('%s – %s' % (self.ex.reloj(t0), self.ex.reloj(min(self.ex.dur, t0 + w))))

    def saltar_clase(self, k, d):
        ex = self.ex
        if not ex:
            return
        cur = self.tira.sel if self.tira.sel >= 0 else int(np.searchsorted(ex.t, self.tira.t0))
        idx = np.nonzero(ex.cls == k)[0]
        if not len(idx):
            return
        j = np.searchsorted(idx, cur)
        if d > 0:
            j = j + 1 if j < len(idx) and idx[j] == cur else j
            if j >= len(idx):
                return
        else:
            j -= 1
            if j < 0:
                return
        self.tira.seleccionar(int(idx[j]))

    def ir_a(self, t, b=-1):
        if b >= 0:
            self.tira.seleccionar(b)
        else:
            self.tira.ir_a(t)
        self.tira.setFocus()
