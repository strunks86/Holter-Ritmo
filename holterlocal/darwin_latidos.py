"""Vistas latido a latido: cascada (Editor de plantillas, vista Cascada), ECHOView, análisis auricular (histograma de PR en el
tiempo) y medidor de latidos (manual de medilog Darwin2, 4.2.2, 4.2.10, 4.2.18 y 4.2.19)."""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMenu, QPushButton, QScrollBar,
                               QSlider, QSpinBox, QSplitter, QVBoxLayout, QWidget)

from .darwin_comun import fmt
from .darwin_vfc import Perezosa, mapa_fuego
from .estilo import COLOR_CLASE
from .motor import CLS_NAME, CLS_SYM, ORDEN_ETIQUETAS
from .tira import TECLAS, senal


def _por_tiempo(ex, t_src, v_src, tol=0.06):
    if t_src is None or not len(t_src) or not len(ex.t):
        return np.full(len(ex.t), np.nan, np.float32)
    j = np.clip(np.searchsorted(t_src, ex.t), 1, len(t_src) - 1)
    jj = np.where(np.abs(t_src[j - 1] - ex.t) < np.abs(t_src[j] - ex.t), j - 1, j)
    out = np.asarray(v_src, np.float32)[jj].copy()
    out[np.abs(t_src[jj] - ex.t) > tol] = np.nan
    return out


def pr_latidos(ex):
    """PR (ms) de cada latido según holterai (NaN donde no se midió o si holterai no corrió)."""
    ha = getattr(ex, 'ha_lat', None)
    if not ha or ha.get('pr') is None:
        return None
    c = getattr(ex, '_pr_cache', None)
    if c and c[0] == len(ex.t) and c[1] is ex.t:
        return c[2]
    v = _por_tiempo(ex, ha['t'], ha['pr'])
    ex._pr_cache = (len(ex.t), ex.t, v)
    return v


def ancho_latidos(ex):
    ha = getattr(ex, 'ha_lat', None)
    if not ha or ha.get('ancho') is None:
        return None
    return _por_tiempo(ex, ha['t'], ha['ancho'])


# --------------------------------------------------------------------------- cascada
class LienzoCascada(QWidget):
    elegido = Signal(int)
    reetiquetar = Signal(list, int)
    ir = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.lat = np.zeros(0, int); self.off = 0; self.alto = 48; self.chans = [0]; self.sel = set(); self.ancla = -1; self.cur = -1
        self.gan = 10.0; self.media = 1.0
        self.setFocusPolicy(Qt.StrongFocus); self.setMinimumHeight(200)

    def filas_visibles(self):
        return max(1, self.height() // self.alto)

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); p.fillRect(self.rect(), QColor('#fffdfb'))
        ex = self.ex
        if ex is None or not len(self.lat):
            p.setPen(QColor('#6b7280')); p.drawText(self.rect(), Qt.AlignCenter, 'Sin latidos para mostrar'); p.end(); return
        W = self.width(); x0 = 190; w = W - x0 - 6; px_s = w / (2 * self.media); f = QFont(); f.setPointSizeF(8); p.setFont(f)
        p.setPen(QPen(QColor('#e9a9a9'), 1, Qt.DashLine)); p.drawLine(int(x0 + w / 2), 0, int(x0 + w / 2), self.height())
        nch = len(self.chans)
        for r in range(self.filas_visibles() + 1):
            i = self.off + r
            if i >= len(self.lat):
                break
            b = int(self.lat[i]); y = r * self.alto; k = int(ex.cls[b])
            if i in self.sel:
                p.fillRect(QRectF(0, y, W, self.alto), QColor('#fde2e2'))
            if i == self.cur:
                p.setPen(QPen(QColor('#0b57d0'), 1.2)); p.drawRect(QRectF(1, y + 1, W - 2, self.alto - 2))
            p.setPen(QColor('#e3e6eb')); p.drawLine(0, y + self.alto - 1, W, y + self.alto - 1)
            rr = ex.rr[b]; pm = ex.prem[b]
            p.setPen(QColor(COLOR_CLASE[k]))
            p.drawText(QRectF(4, y + 2, x0 - 8, 14), Qt.AlignLeft, f'{i + 1}. {CLS_SYM[k]} · {ex.reloj(float(ex.t[b]))}')
            p.setPen(QColor('#5b6573'))
            txt = (f'RR {fmt(rr * 1000)} ms' if rr == rr else '') + (f' · prem. {fmt(pm * 100)} %' if pm == pm else '')
            p.drawText(QRectF(4, y + 16, x0 - 8, 14), Qt.AlignLeft, txt)
            p.drawText(QRectF(4, y + 30, x0 - 8, 14), Qt.AlignLeft, f'morf. {int(ex.tid[b]) + 1} · pV {ex.pV[b]:.2f}')
            i0 = int((ex.t[b] - self.media) * ex.fs); i1 = int((ex.t[b] + self.media) * ex.fs)
            hc = self.alto / nch
            for ci, ch in enumerate(self.chans):
                yy = senal(ex, ex.chans[ch], i0, i1)
                if not len(yy):
                    continue
                xs = x0 + np.arange(len(yy)) / ex.fs * px_s; mid = y + hc * (ci + 0.6); pxmv = self.gan * hc / 18
                ys = np.clip(mid - yy * pxmv, y, y + self.alto)
                p.setPen(QPen(QColor('#111111' if i not in self.sel else '#b91c1c'), 1.0))
                p.drawPolyline(QPolygonF([QPointF(float(a), float(c)) for a, c in zip(xs[::max(1, len(xs) // 900)], ys[::max(1, len(xs) // 900)])]))
        p.end()

    def _fila(self, y):
        return self.off + int(y // self.alto)

    def mousePressEvent(self, e):
        i = self._fila(e.position().y())
        if not (0 <= i < len(self.lat)):
            return
        mod = e.modifiers()
        if mod & Qt.ShiftModifier and self.ancla >= 0:
            a, b = sorted((self.ancla, i)); self.sel = set(range(a, b + 1))
        elif mod & Qt.ControlModifier:
            self.sel ^= {i}; self.ancla = i
        else:
            self.sel = {i}; self.ancla = i
        self.cur = i; self.update(); self.elegido.emit(int(self.lat[i]))
        if e.button() == Qt.RightButton:
            self._menu(e)

    def mouseDoubleClickEvent(self, e):
        i = self._fila(e.position().y())
        if 0 <= i < len(self.lat):
            self.ir.emit(int(self.lat[i]))

    def _menu(self, e):
        bs = self.elegidos(); m = QMenu(self)
        sub = m.addMenu('Etiquetar %d latido(s) como' % len(bs))
        for k in ORDEN_ETIQUETAS + [4]:
            sub.addAction(f'{CLS_NAME[k]}\t{CLS_SYM[k]}', lambda k=k: self.reetiquetar.emit(bs, k))
        m.addAction('Ir al ECG\tIntro', lambda: self.ir.emit(bs[0]))
        m.exec(e.globalPosition().toPoint())

    def elegidos(self):
        return [int(self.lat[i]) for i in sorted(self.sel) if i < len(self.lat)]

    def keyPressEvent(self, e):
        k = e.key()
        if k in TECLAS and not (e.modifiers() & (Qt.ControlModifier | Qt.AltModifier)):
            bs = self.elegidos()
            if bs:
                self.reetiquetar.emit(bs, TECLAS[k]); return
        if k in (Qt.Key_Down, Qt.Key_Up):
            i = int(np.clip(self.cur + (1 if k == Qt.Key_Down else -1), 0, len(self.lat) - 1))
            self.cur = i; self.sel = {i}; self.ancla = i
            if i < self.off:
                self.off = i
            elif i >= self.off + self.filas_visibles():
                self.off = i - self.filas_visibles() + 1
            self.parent().barra.setValue(self.off); self.update(); self.elegido.emit(int(self.lat[i])); return
        if k in (Qt.Key_Return, Qt.Key_Enter) and self.cur >= 0:
            self.ir.emit(int(self.lat[self.cur])); return
        super().keyPressEvent(e)

    def wheelEvent(self, e):
        self.parent().barra.setValue(self.off - int(np.sign(e.angleDelta().y())) * 3)


FUENTES = [('1', 'Supraventriculares (S)'), ('2v', 'Ventriculares (V, R sobre T, interpoladas, escape)'), ('0', 'Normales (N)'),
           ('7', 'Aberrantes'), ('6', 'Bloqueo de rama'), ('3', 'Fusión'), ('5', 'Desconocidos'), ('4', 'Artefacto'), ('est', 'Estimulados'),
           ('tpl', 'Plantilla elegida en Plantillas'), ('todo', 'Todos los latidos')]
ORDENES = [('t', 'Cronológico'), ('prem', 'Prematuridad (más prematuros arriba)'), ('prem_inv', 'Prematuridad (menos prematuros arriba)'),
           ('pv', 'Probabilidad ventricular'), ('rr', 'RR más corto arriba')]


class VistaCascada(Perezosa, QWidget):
    ir = Signal(float, int)
    reetiquetar = Signal(list, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.sucio = False; self.tid = -1
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        top = QHBoxLayout(); top.addWidget(QLabel('Latidos'))
        self.fuente = QComboBox(); [self.fuente.addItem(n, k) for k, n in FUENTES]; top.addWidget(self.fuente)
        top.addWidget(QLabel('Orden')); self.orden = QComboBox(); [self.orden.addItem(n, k) for k, n in ORDENES]; top.addWidget(self.orden)
        top.addWidget(QLabel('Canales')); self.canal = QComboBox(); top.addWidget(self.canal)
        top.addWidget(QLabel('Ganancia')); self.gan = QComboBox(); [self.gan.addItem(f'×{v / 10:g}', v) for v in (5.0, 10.0, 20.0, 40.0)]; self.gan.setCurrentIndex(1); top.addWidget(self.gan)
        top.addWidget(QLabel('Espacio')); self.esp = QSlider(Qt.Horizontal); self.esp.setRange(34, 160); self.esp.setValue(52); self.esp.setMaximumWidth(120); top.addWidget(self.esp)
        self.info = QLabel(''); top.addWidget(self.info, 1)
        lay.addLayout(top)
        cuerpo = QHBoxLayout(); self.lienzo = LienzoCascada(self); self.barra = QScrollBar(Qt.Vertical)
        cuerpo.addWidget(self.lienzo, 1); cuerpo.addWidget(self.barra); lay.addLayout(cuerpo, 1)
        ay = QLabel('Un segundo antes y después de cada latido, alineados en la R. Clic, Ctrl+clic y Mayús+clic eligen; la letra de la etiqueta '
                    '(N, S, V, A…) reclasifica los elegidos; ↑ ↓ recorren; Intro o doble clic van al ECG. Úsala para confirmar que las S son supraventriculares verdaderas.')
        ay.setWordWrap(True); ay.setObjectName('ayuda'); lay.addWidget(ay)
        self.barra.valueChanged.connect(self._scroll)
        for w in (self.fuente, self.orden, self.canal):
            w.currentIndexChanged.connect(lambda *_: self._calcular_seguro())
        self.gan.currentIndexChanged.connect(lambda *_: self._vista())
        self.esp.valueChanged.connect(lambda *_: self._vista())
        self.lienzo.ir.connect(lambda b: self.ir.emit(float(self.ex.t[b]), b))
        self.lienzo.reetiquetar.connect(self.reetiquetar.emit)

    def set_plantilla(self, tid):
        self.tid = int(tid)

    def calcular(self):
        ex = self.ex; L = self.lienzo
        cur = self.canal.currentData(); self.canal.blockSignals(True); self.canal.clear(); self.canal.addItem('Todos', -1)
        for i, c in enumerate(ex.chans):
            self.canal.addItem(ex.names[c] if c < len(ex.names) else f'Canal {i + 1}', i)
        self.canal.setCurrentIndex(max(0, self.canal.findData(cur if cur is not None else -1))); self.canal.blockSignals(False)
        f = self.fuente.currentData()
        if f == '2v':
            m = np.isin(ex.cls, (2, 8, 10, 11))
        elif f == 'est':
            m = np.isin(ex.cls, (9, 12, 13))
        elif f == 'tpl':
            m = ex.tid == self.tid
        elif f == 'todo':
            m = np.ones(len(ex.t), bool)
        else:
            m = ex.cls == int(f)
        idx = np.nonzero(m)[0]
        o = self.orden.currentData()
        if o == 'prem':
            idx = idx[np.argsort(-np.nan_to_num(ex.prem[idx], nan=-9))]
        elif o == 'prem_inv':
            idx = idx[np.argsort(np.nan_to_num(ex.prem[idx], nan=9))]
        elif o == 'pv':
            idx = idx[np.argsort(-ex.pV[idx])]
        elif o == 'rr':
            idx = idx[np.argsort(np.nan_to_num(ex.rr[idx], nan=9))]
        L.ex = ex; L.lat = idx; L.sel = {i for i in L.sel if i < len(idx)}; L.cur = min(L.cur, len(idx) - 1)
        self.info.setText(f'{fmt(len(idx))} latidos' + (f' · plantilla {self.tid + 1}' if f == 'tpl' else ''))
        self._vista()

    def _vista(self):
        L = self.lienzo
        if L.ex is None:
            return
        c = self.canal.currentData()
        L.chans = list(range(len(L.ex.chans))) if c is None or c < 0 else [c]
        L.alto = self.esp.value() * (1 if len(L.chans) == 1 else max(1, len(L.chans)) * 0.75)
        L.alto = int(max(L.alto, 46)); L.gan = self.gan.currentData()
        self.barra.setRange(0, max(0, len(L.lat) - L.filas_visibles())); self.barra.setPageStep(L.filas_visibles())
        L.off = min(L.off, self.barra.maximum()); L.update()

    def _scroll(self, v):
        self.lienzo.off = int(v); self.lienzo.update()

    def resizeEvent(self, e):
        super().resizeEvent(e); self._vista()


# --------------------------------------------------------------------------- ECHOView
def lut_echo():
    n = 256; out = np.zeros((2 * n - 1, 3), np.uint8)
    for i in range(2 * n - 1):
        v = (i - (n - 1)) / (n - 1)
        if v >= 0:
            if v < 0.33:
                a = v / 0.33; c = (255, 255, 255 - 255 * a)                 # blanco → amarillo
            elif v < 0.66:
                a = (v - 0.33) / 0.33; c = (255, 255 - 90 * a, 0)          # amarillo → naranja
            else:
                a = (v - 0.66) / 0.34; c = (255 - 120 * a, 165 - 165 * a, 0)  # naranja → rojo oscuro
        else:
            a = -v; c = (255 - 235 * a, 255 - 200 * a, 255 - 90 * a)        # blanco → azul oscuro
        out[i] = c
    return out


LUT_ECHO = lut_echo()


class LienzoECHO(QWidget):
    clic = Signal(int, float)
    rango = Signal(int, int, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.b0 = 0; self.px = 2; self.pre = 250.0; self.post = 450.0; self.canal = 0; self.contraste = 1.0; self.filtro = True
        self.arr = True; self.img = None; self.cur = -1; self.sel = None; self._arr0 = None
        self.setMouseTracking(True); self.setMinimumHeight(240)

    def ncol(self):
        return max(1, (self.width() - 50) // self.px)

    def construir(self):
        ex = self.ex
        if ex is None or not len(ex.t):
            self.img = None; return
        n = min(self.ncol(), len(ex.t) - self.b0); H = max(40, self.height() - 34)
        bs = np.arange(self.b0, self.b0 + n)
        ch = ex.chans[min(self.canal, len(ex.chans) - 1)]
        a = float(ex.t[bs[0]]) - self.pre / 1000 - 0.1; z = float(ex.t[bs[-1]]) + self.post / 1000 + 0.1
        i0 = max(0, int(a * ex.fs)); i1 = min(ex.n, int(z * ex.fs) + 1)
        y = senal(ex, ch, i0, i1, self.filtro)
        off = np.linspace(-self.pre, self.post, H) / 1000.0
        idx = np.clip(((ex.t[bs][None, :] + off[:, None]) * ex.fs).astype(int) - i0, 0, max(0, len(y) - 1))
        if len(y):
            base = np.median(y[idx[:max(1, int(H * 0.1))]], axis=0) if H > 10 else 0
            v = (y[idx] - base) / max(0.05, self.contraste)
        else:
            v = np.zeros((H, n))
        q = np.clip(np.round(v * 255) + 255, 0, 510).astype(int)
        rgb = LUT_ECHO[q]
        if self.px > 1:
            rgb = np.repeat(rgb, self.px, axis=1)
        self.rgb = np.ascontiguousarray(rgb)
        h, w, _ = self.rgb.shape
        self.img = QImage(self.rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.n = n; self.H = H

    def paintEvent(self, e):
        p = QPainter(self); p.fillRect(self.rect(), QColor('#ffffff'))
        ex = self.ex
        if ex is None or self.img is None:
            p.end(); return
        x0 = 46; p.drawImage(x0, 0, self.img)
        f = QFont(); f.setPointSizeF(7.5); p.setFont(f); p.setPen(QColor('#374151'))
        tot = self.pre + self.post
        for ms in range(-int(self.pre // 100) * 100, int(self.post) + 1, 100):
            y = (ms + self.pre) / tot * (self.H - 1)
            p.drawLine(x0 - 4, int(y), x0, int(y)); p.drawText(QRectF(0, y - 6, x0 - 5, 12), Qt.AlignRight | Qt.AlignVCenter, f'{ms}')
        yR = self.pre / tot * (self.H - 1); p.setPen(QPen(QColor(0, 0, 0, 90), 1, Qt.DotLine)); p.drawLine(x0, int(yR), self.width(), int(yR))
        yb = self.H + 2
        if self.arr:
            for j in range(self.n):
                b = self.b0 + j; k = int(ex.cls[b])
                if k != 0:
                    p.fillRect(QRectF(x0 + j * self.px, yb, self.px, 10), QColor(COLOR_CLASE[k]))
        p.setPen(QColor('#374151'))
        p.drawText(QRectF(x0, yb + 11, 200, 14), Qt.AlignLeft, ex.reloj(float(ex.t[self.b0])))
        p.drawText(QRectF(self.width() - 210, yb + 11, 200, 14), Qt.AlignRight, ex.reloj(float(ex.t[min(len(ex.t) - 1, self.b0 + self.n - 1)])))
        if self.sel:
            a, b = sorted(self.sel); p.fillRect(QRectF(x0 + a * self.px, 0, (b - a + 1) * self.px, self.H), QColor(11, 87, 208, 50))
        if 0 <= self.cur - self.b0 < self.n:
            x = x0 + (self.cur - self.b0) * self.px + self.px / 2; p.setPen(QPen(QColor('#16a34a'), 1)); p.drawLine(int(x), 0, int(x), self.H)
        p.end()

    def _col(self, x):
        return int((x - 46) // self.px)

    def mousePressEvent(self, e):
        j = self._col(e.position().x())
        if self.ex is None or not (0 <= j < getattr(self, 'n', 0)):
            return
        if e.modifiers() & Qt.ControlModifier:
            self.sel = [j, j]; self._arr0 = j; self.update(); return
        ms = e.position().y() / max(1, self.H - 1) * (self.pre + self.post) - self.pre
        self.cur = self.b0 + j; self.update(); self.clic.emit(self.cur, ms)

    def mouseMoveEvent(self, e):
        if self._arr0 is not None and e.buttons() & Qt.LeftButton:
            self.sel = [self._arr0, max(0, min(self.n - 1, self._col(e.position().x())))]; self.update()

    def mouseReleaseEvent(self, e):
        if self._arr0 is not None and self.sel:
            a, b = sorted(self.sel); self._arr0 = None
            self.rango.emit(self.b0 + a, self.b0 + b, e.globalPosition().toPoint())

    def resizeEvent(self, e):
        super().resizeEvent(e); self.construir()


class VistaECHO(Perezosa, QWidget):
    ir = Signal(float, int)
    reetiquetar = Signal(list, int)
    excluir = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.sucio = False
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        top = QHBoxLayout(); top.addWidget(QLabel('Canal')); self.canal = QComboBox(); top.addWidget(self.canal)
        top.addWidget(QLabel('Ancho de banda')); self.pre = QSpinBox(); self.pre.setRange(100, 600); self.pre.setValue(250); self.pre.setSuffix(' ms antes de R')
        self.post = QSpinBox(); self.post.setRange(200, 900); self.post.setValue(450); self.post.setSuffix(' ms después')
        top.addWidget(self.pre); top.addWidget(self.post)
        top.addWidget(QLabel('Escala')); self.px = QComboBox(); [self.px.addItem(f'{v} px por latido', v) for v in (1, 2, 3, 4, 6)]; self.px.setCurrentIndex(1); top.addWidget(self.px)
        top.addWidget(QLabel('Contraste')); self.cont = QSlider(Qt.Horizontal); self.cont.setRange(1, 40); self.cont.setValue(10); self.cont.setMaximumWidth(120); top.addWidget(self.cont)
        self.c_fil = QCheckBox('Filtro'); self.c_fil.setChecked(True); self.c_arr = QCheckBox('Arritmias'); self.c_arr.setChecked(True)
        top.addWidget(self.c_fil); top.addWidget(self.c_arr)
        self.info = QLabel(''); top.addWidget(self.info, 1); lay.addLayout(top)
        self.lienzo = LienzoECHO(); lay.addWidget(self.lienzo, 1)
        self.barra = QScrollBar(Qt.Horizontal); lay.addWidget(self.barra)
        ay = QLabel('Cada columna es un latido alineado en la R (rojo: positivo, azul: negativo, blanco: línea de base). Permite ver de un vistazo '
                    'cambios del PR, del QRS, del ST y del QT, y la onda P que desaparece en la FA. Clic: distancia al pico R y va al ECG · '
                    'Ctrl+arrastrar: reclasificar o excluir los latidos elegidos.')
        ay.setWordWrap(True); ay.setObjectName('ayuda'); lay.addWidget(ay)
        self.barra.valueChanged.connect(self._mover)
        for w in (self.canal, self.px):
            w.currentIndexChanged.connect(lambda *_: self._opc())
        for w in (self.pre, self.post):
            w.valueChanged.connect(lambda *_: self._opc())
        self.cont.valueChanged.connect(lambda *_: self._opc())
        self.c_fil.toggled.connect(lambda *_: self._opc()); self.c_arr.toggled.connect(lambda *_: self._opc())
        self.lienzo.clic.connect(self._clic); self.lienzo.rango.connect(self._rango)

    def calcular(self):
        ex = self.ex
        cur = self.canal.currentData(); self.canal.blockSignals(True); self.canal.clear()
        for i, c in enumerate(ex.chans):
            self.canal.addItem(ex.names[c] if c < len(ex.names) else f'Canal {i + 1}', i)
        self.canal.setCurrentIndex(max(0, self.canal.findData(cur if cur is not None else 0))); self.canal.blockSignals(False)
        self.lienzo.ex = ex; self._opc()

    def _opc(self):
        L = self.lienzo
        if L.ex is None:
            return
        L.canal = self.canal.currentData() or 0; L.px = self.px.currentData(); L.pre = float(self.pre.value()); L.post = float(self.post.value())
        L.contraste = self.cont.value() / 10.0; L.filtro = self.c_fil.isChecked(); L.arr = self.c_arr.isChecked()
        self.barra.setRange(0, max(0, len(L.ex.t) - L.ncol())); self.barra.setPageStep(L.ncol())
        L.b0 = min(L.b0, self.barra.maximum()); L.construir(); L.update()

    def set_tiempo(self, t):
        if self.ex is None or not self.isVisible():
            return
        b = int(np.searchsorted(self.ex.t, t)); L = self.lienzo
        if not (L.b0 <= b < L.b0 + L.ncol()):
            self.barra.setValue(max(0, b - L.ncol() // 2))

    def _mover(self, v):
        self.lienzo.b0 = int(v); self.lienzo.construir(); self.lienzo.update()

    def _clic(self, b, ms):
        self.info.setText(f'Latido {b + 1} · {CLS_NAME[int(self.ex.cls[b])]} · {self.ex.reloj(float(self.ex.t[b]))} · distancia al pico R: {ms:+.0f} ms')
        self.ir.emit(float(self.ex.t[b]), b)

    def _rango(self, a, b, pos):
        bs = list(range(a, b + 1)); m = QMenu(self)
        sub = m.addMenu('Etiquetar %d latidos como' % len(bs))
        for k in ORDEN_ETIQUETAS + [4]:
            sub.addAction(f'{CLS_NAME[k]}\t{CLS_SYM[k]}', lambda k=k: self.reetiquetar.emit(bs, k))
        t0 = float(self.ex.t[a]) - 0.3; t1 = float(self.ex.t[b]) + 0.3
        m.addAction('Excluir el tramo (artefacto)', lambda: self.excluir.emit(t0, t1))
        m.exec(pos); self.lienzo.sel = None; self.lienzo.update()


# --------------------------------------------------------------------------- análisis auricular
class VistaAuricular(Perezosa, QWidget):
    ir = Signal(float, int)
    nav = Signal(str, list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.sucio = False; self.pr = None
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        top = QHBoxLayout()
        top.addWidget(QLabel('Resolución')); self.res = QComboBox(); [self.res.addItem(f'{v} ms', v) for v in (5, 10, 20)]; self.res.setCurrentIndex(1); top.addWidget(self.res)
        top.addWidget(QLabel('PR máximo')); self.prmax = QSpinBox(); self.prmax.setRange(250, 800); self.prmax.setValue(400); self.prmax.setSuffix(' ms'); top.addWidget(self.prmax)
        top.addWidget(QLabel('Segmento')); self.seg = QComboBox(); [self.seg.addItem(n, s) for n, s in (('1 min', 60), ('2 min', 120), ('5 min', 300), ('10 min', 600), ('15 min', 900))]
        self.seg.setCurrentIndex(2); top.addWidget(self.seg)
        self.c_inv = QCheckBox('Invertir colores'); self.c_fa = QCheckBox('Mostrar episodios de FA'); self.c_fa.setChecked(True)
        top.addWidget(self.c_inv); top.addWidget(self.c_fa)
        self.info = QLabel(''); self.info.setWordWrap(True); top.addWidget(self.info, 1); lay.addLayout(top)
        sp = QSplitter(Qt.Vertical)
        self.g = pg.PlotWidget(); self.g.setLabel('left', 'PR (ms)'); self.g.setLabel('bottom', 'horas desde el inicio')
        self.img = pg.ImageItem(); self.g.addItem(self.img); self.g.scene().sigMouseClicked.connect(self._clic_mapa)
        self.cur = pg.InfiniteLine(angle=90, pen=pg.mkPen('#16a34a', width=1.5)); self.g.addItem(self.cur)
        sp.addWidget(self.g)
        abajo = QWidget(); al = QHBoxLayout(abajo); al.setContentsMargins(0, 0, 0, 0)
        self.h = pg.PlotWidget(); self.h.setLabel('bottom', 'PR (ms)'); self.h.setLabel('left', 'latidos'); self.h.showGrid(x=True, y=True, alpha=0.25)
        self.h.scene().sigMouseClicked.connect(self._clic_hist); al.addWidget(self.h, 2)
        self.tr = pg.PlotWidget(); self.tr.setLabel('left', '% PR > 200 ms'); self.tr.setLabel('bottom', 'horas'); self.tr.showGrid(x=True, y=True, alpha=0.25)
        al.addWidget(self.tr, 3); sp.addWidget(abajo); sp.setSizes([420, 260]); lay.addWidget(sp, 1)
        ay = QLabel('Histograma de intervalos PR a lo largo del registro (azul: pocos latidos, rojo: muchos). La línea punteada marca 200 ms '
                    '(sobre ella, bloqueo AV de primer grado). Un PR que se dispersa o desaparece sugiere FA o flutter. Clic en el mapa: '
                    'histograma de ese segmento · clic en una barra: recorre sus latidos con Tab. El PR lo mide holterai, latido a latido y con los canales disponibles.')
        ay.setWordWrap(True); ay.setObjectName('ayuda'); lay.addWidget(ay)
        for w in (self.res, self.seg):
            w.currentIndexChanged.connect(lambda *_: self._calcular_seguro())
        self.prmax.valueChanged.connect(lambda *_: self._calcular_seguro())
        self.c_inv.toggled.connect(lambda *_: self._calcular_seguro()); self.c_fa.toggled.connect(lambda *_: self._calcular_seguro())

    def calcular(self):
        ex = self.ex; self.g.clear(); self.h.clear(); self.tr.clear(); self.g.addItem(self.img); self.g.addItem(self.cur)
        self.pr = pr = pr_latidos(ex)
        if pr is None:
            self.img.clear()
            self.info.setText('<b>El PR latido a latido lo entrega holterai.</b> Usa el motor «Mixto» o «Solo holterai» (menú Motor) y vuelve a abrir el examen.')
            return
        ok = np.isfinite(pr) & np.isin(ex.cls, (0, 1, 6, 7)) & (pr > 40) & (pr < self.prmax.value())
        res = self.res.currentData(); S = self.seg.currentData()
        nseg = int(np.ceil(ex.dur / S)); nb = int(self.prmax.value() // res)
        si = (ex.t[ok] // S).astype(int); bi = np.clip((pr[ok] // res).astype(int), 0, nb - 1)
        M = np.zeros((nseg, nb)); np.add.at(M, (np.clip(si, 0, nseg - 1), bi), 1)
        self.M = M; self.ok = ok
        L = np.log1p(M)
        cm = pg.ColorMap([0, 0.01, 0.4, 0.75, 1.0], [(255, 255, 255), (191, 219, 254), (59, 130, 246), (250, 204, 21), (220, 38, 38)])
        if self.c_inv.isChecked():
            cm = pg.ColorMap([0, 0.01, 0.4, 0.75, 1.0], [(255, 255, 255), (254, 202, 202), (220, 38, 38), (250, 204, 21), (37, 99, 235)])
        from PySide6.QtCore import QRectF as R_
        self.img.setImage(L, levels=(0, max(1e-6, L.max()))); self.img.setLookupTable(cm.getLookupTable(nPts=256))
        self.img.setRect(R_(0, 0, nseg * S / 3600.0, nb * res))
        self.g.addItem(pg.InfiniteLine(pos=200, angle=0, pen=pg.mkPen('#111111', width=1.2, style=Qt.DashLine)))
        if self.c_fa.isChecked():
            for e in ex.eventos:
                if e['type'] == 'af':
                    r = pg.LinearRegionItem([e['t0'] / 3600, e['t1'] / 3600], movable=False, brush=pg.mkBrush(13, 148, 136, 45), pen=pg.mkPen(None))
                    self.g.addItem(r)
        self.g.setYRange(0, self.prmax.value())
        tot = M.sum(1); largo = M[:, int(200 // res):].sum(1)
        pct = np.where(tot >= 5, 100 * largo / np.maximum(tot, 1), np.nan)
        self.tr.plot((np.arange(nseg) + 0.5) * S / 3600, pct, pen=None, symbol='o', symbolSize=4, symbolBrush='#be185d')
        n_ok = int(ok.sum()); n_val = int(np.isin(ex.cls, (0, 1, 6, 7)).sum())
        med = np.nanmedian(pr[ok]) if n_ok else np.nan
        self.info.setText(f'PR medido en {fmt(n_ok)} de {fmt(n_val)} latidos N/S ({fmt(100 * n_ok / max(1, n_val), 1)} %) · mediana {fmt(med)} ms · '
                          f'PR &gt; 200 ms en {fmt(100 * (pr[ok] > 200).mean() if n_ok else np.nan, 1)} %')
        self._hist_seg(None)

    def _hist_seg(self, j):
        ex = self.ex; res = self.res.currentData(); S = self.seg.currentData(); self.h.clear()
        if j is None:
            y = self.M.sum(0); tit = 'Registro completo'; self.j = None
        else:
            y = self.M[j]; tit = f'{ex.reloj(j * S, False)} – {ex.reloj((j + 1) * S, False)}'; self.j = j
            self.cur.setValue((j + 0.5) * S / 3600)
        x = (np.arange(len(y)) + 0.5) * res
        self.h.addItem(pg.LinearRegionItem([0, 200], movable=False, brush=pg.mkBrush(22, 163, 74, 35), pen=pg.mkPen(None)))
        cols = [pg.mkBrush('#1d4ed8' if c < 200 else '#be185d') for c in x]
        self.h.addItem(pg.BarGraphItem(x=x, height=y, width=res * 0.9, brushes=cols, pen=None)); self.h.setTitle('Histograma P-R detallado · ' + tit)

    def _clic_mapa(self, ev):
        if self.pr is None or not self.g.sceneBoundingRect().contains(ev.scenePos()):
            return
        pt = self.g.getPlotItem().vb.mapSceneToView(ev.scenePos()); S = self.seg.currentData()
        j = int(pt.x() * 3600 // S)
        if 0 <= j < len(self.M):
            self._hist_seg(j); self.ir.emit(j * S + 1.0, -1)

    def _clic_hist(self, ev):
        if self.pr is None or not self.h.sceneBoundingRect().contains(ev.scenePos()):
            return
        x = self.h.getPlotItem().vb.mapSceneToView(ev.scenePos()).x(); res = self.res.currentData(); S = self.seg.currentData()
        k = int(x // res); ex = self.ex
        m = self.ok & (np.clip((np.nan_to_num(self.pr) // res).astype(int), 0, 10 ** 6) == k)
        if self.j is not None:
            m &= (ex.t // S).astype(int) == self.j
        idx = np.nonzero(m)[0]
        self.nav.emit(f'PR {k * res}–{(k + 1) * res} ms', [(float(ex.t[b]), int(b)) for b in idx[:20000]])


# --------------------------------------------------------------------------- medidor de latidos
PUNTOS = [('p', 'Inicio P', '#0d9488'), ('q', 'Inicio QRS', '#1d4ed8'), ('j', 'Punto J', '#dc2626'), ('t', 'Final T', '#7c3aed')]


class LienzoMedidor(QWidget):
    cambio = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.b = -1; self.pos = {}; self.pre = 0.45; self.post = 0.65; self.gan = 20.0; self._drag = None
        self.setMouseTracking(True); self.setMinimumHeight(260)

    def x_de(self, ms):
        return 50 + (ms / 1000 + self.pre) / (self.pre + self.post) * (self.width() - 60)

    def ms_de(self, x):
        return ((x - 50) / max(1, self.width() - 60) * (self.pre + self.post) - self.pre) * 1000

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); p.fillRect(self.rect(), QColor('#fffdfb'))
        ex = self.ex
        if ex is None or self.b < 0:
            p.setPen(QColor('#6b7280')); p.drawText(self.rect(), Qt.AlignCenter, 'Elige un latido en el ECG'); p.end(); return
        W, H = self.width(), self.height()
        p.setPen(QPen(QColor('#f6d9d9'), 1))
        for ms in range(-int(self.pre * 1000), int(self.post * 1000) + 1, 40):
            x = self.x_de(ms); p.setPen(QPen(QColor('#e9a9a9' if ms % 200 == 0 else '#f6d9d9'), 1)); p.drawLine(int(x), 0, int(x), H)
        nch = len(ex.chans); hc = (H - 20) / max(1, nch); tb = float(ex.t[self.b])
        i0 = int((tb - self.pre) * ex.fs); i1 = int((tb + self.post) * ex.fs)
        f = QFont(); f.setPointSizeF(8); p.setFont(f)
        for ci, ch in enumerate(ex.chans):
            y = senal(ex, ch, i0, i1); mid = 14 + hc * (ci + 0.6); pxmv = self.gan * (W - 60) / ((self.pre + self.post) * 1000) * 40 / 10
            pxmv = min(pxmv, hc * 0.35)
            xs = self.x_de((np.arange(len(y)) / ex.fs - self.pre) * 1000); ys = mid - y * pxmv
            p.setPen(QPen(QColor('#111111'), 1.2)); p.drawPolyline(QPolygonF([QPointF(float(a), float(b)) for a, b in zip(xs, ys)]))
            p.setPen(QColor('#5b6573')); p.drawText(QPointF(4, 14 + hc * ci + 12), ex.names[ch] if ch < len(ex.names) else str(ci + 1))
        for k, n, col in PUNTOS:
            if k in self.pos:
                x = self.x_de(self.pos[k]); p.setPen(QPen(QColor(col), 1.6, Qt.DashLine)); p.drawLine(int(x), 12, int(x), H)
                p.setPen(QColor(col)); p.drawText(QRectF(x - 40, 0, 80, 12), Qt.AlignCenter, n)
        p.end()

    def mousePressEvent(self, e):
        x = e.position().x(); best = None
        for k, _, _ in PUNTOS:
            if k in self.pos and abs(self.x_de(self.pos[k]) - x) < 8:
                best = k
        self._drag = best

    def mouseMoveEvent(self, e):
        if self._drag and e.buttons() & Qt.LeftButton:
            self.pos[self._drag] = float(np.clip(self.ms_de(e.position().x()), -self.pre * 1000, self.post * 1000)); self.update(); self.cambio.emit()
        else:
            x = e.position().x()
            self.setCursor(Qt.SizeHorCursor if any(abs(self.x_de(v) - x) < 8 for v in self.pos.values()) else Qt.ArrowCursor)

    def mouseReleaseEvent(self, e):
        self._drag = None


class VistaMedidor(QWidget):
    cambio = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None
        lay = QHBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        izq = QVBoxLayout()
        top = QHBoxLayout(); top.addWidget(QLabel('Ganancia')); self.gan = QComboBox(); [self.gan.addItem(f'×{v / 10:g}', v) for v in (10.0, 20.0, 40.0)]; self.gan.setCurrentIndex(1)
        top.addWidget(self.gan); self.titulo = QLabel(''); top.addWidget(self.titulo, 1)
        b = QPushButton('Reiniciar puntos'); b.clicked.connect(self._iniciales); top.addWidget(b)
        bg = QPushButton('Guardar medición'); bg.setObjectName('primario'); bg.clicked.connect(self.guardar); top.addWidget(bg)
        izq.addLayout(top)
        self.lienzo = LienzoMedidor(); izq.addWidget(self.lienzo, 1)
        ay = QLabel('Arrastra las líneas de inicio de P, inicio del QRS, punto J y final de T. Los puntos parten de lo que midió el motor '
                    '(holterai cuando está disponible) y se pueden corregir a mano. El nivel del ST se mide contra el segmento PR.')
        ay.setWordWrap(True); ay.setObjectName('ayuda'); izq.addWidget(ay)
        lay.addLayout(izq, 3)
        der = QVBoxLayout(); self.vals = QLabel(''); self.vals.setTextFormat(Qt.RichText); self.vals.setAlignment(Qt.AlignTop); der.addWidget(self.vals)
        der.addWidget(QLabel('Elementos guardados')); self.lista = QListWidget(); der.addWidget(self.lista, 1)
        bb = QPushButton('Borrar el elegido'); bb.clicked.connect(self._borrar); der.addWidget(bb)
        lay.addLayout(der, 2)
        self.lienzo.cambio.connect(self._valores); self.gan.currentIndexChanged.connect(lambda *_: self._gan())
        self.lista.currentItemChanged.connect(lambda it, _: self._cargar(it))

    def _gan(self):
        self.lienzo.gan = self.gan.currentData(); self.lienzo.update()

    def mostrar(self, ex):
        self.ex = ex; self.lienzo.ex = ex; self._lista()
        if not (0 <= self.lienzo.b < len(ex.t)):
            self.lienzo.b = -1
        self.lienzo.update()

    def set_latido(self, b):
        if self.ex is None or not (0 <= b < len(self.ex.t)):
            return
        self.lienzo.b = int(b); self._iniciales()

    def _iniciales(self):
        ex = self.ex; b = self.lienzo.b
        if ex is None or b < 0:
            return
        an = ancho_latidos(ex); pr = pr_latidos(ex)
        w = float(an[b]) if an is not None and an[b] == an[b] else 100.0
        prv = float(pr[b]) if pr is not None and pr[b] == pr[b] else 160.0
        rr = float(ex.rr[b]) if ex.rr[b] == ex.rr[b] else 0.85
        M = ex.A.get('minute') or {}; qtm = M.get('qt') or []; mi = int(ex.t[b] // 60)
        qt = float(qtm[mi]) if mi < len(qtm) and qtm[mi] else 400.0 * np.sqrt(max(0.3, rr))
        q = -0.45 * w
        self.lienzo.pos = {'q': q, 'j': q + w, 'p': q - prv, 't': min(self.lienzo.post * 1000 - 5, q + qt)}
        self.titulo.setText(f'Latido {b + 1} · {CLS_NAME[int(ex.cls[b])]} · {ex.reloj(float(ex.t[b]))}')
        self.lienzo.update(); self._valores()

    def medidas(self):
        ex = self.ex; b = self.lienzo.b; P = self.lienzo.pos
        if ex is None or b < 0 or not P:
            return None
        rr = float(ex.rr[b]) if ex.rr[b] == ex.rr[b] else np.nan
        d = {'t': float(ex.t[b]), 'hora': ex.reloj(float(ex.t[b])), 'clase': CLS_SYM[int(ex.cls[b])], 'pr': P['q'] - P['p'], 'qrs': P['j'] - P['q'],
             'qt': P['t'] - P['q'], 'rr': rr * 1000 if rr == rr else np.nan, 'puntos': dict(P), 'canales': []}
        d['qtc'] = d['qt'] / np.sqrt(rr) if rr == rr and rr > 0 else np.nan
        tb = float(ex.t[b])
        for ci, ch in enumerate(ex.chans):
            i0 = int((tb - 0.6) * ex.fs); y = senal(ex, ch, i0, int((tb + 0.7) * ex.fs))
            at = lambda ms: y[int(np.clip((tb + ms / 1000) * ex.fs - i0, 0, len(y) - 1))] if len(y) else np.nan
            base = np.mean([at(P['q'] - 30), at(P['q'] - 20), at(P['q'] - 10)])
            qa = int(np.clip((tb + P['q'] / 1000) * ex.fs - i0, 0, len(y) - 1)); ja = int(np.clip((tb + P['j'] / 1000) * ex.fs - i0, 0, len(y) - 1))
            seg = y[qa:ja + 1] if ja > qa else y[qa:qa + 1]
            d['canales'].append({'canal': ex.names[ch] if ch < len(ex.names) else str(ci + 1), 'r_mv': float(seg.max() - base) if len(seg) else np.nan,
                                 'st60': float(at(P['j'] + 60) - base), 'st80': float(at(P['j'] + 80) - base)})
        return d

    def _valores(self):
        d = self.medidas()
        if not d:
            self.vals.setText(''); return
        h = (f'<b>{d["hora"]} · {d["clase"]}</b><table>'
             f'<tr><td>PR</td><td align="right"><b>{fmt(d["pr"])}</b> ms</td></tr><tr><td>QRS</td><td align="right"><b>{fmt(d["qrs"])}</b> ms</td></tr>'
             f'<tr><td>QT</td><td align="right"><b>{fmt(d["qt"])}</b> ms</td></tr><tr><td>QTc (Bazett)</td><td align="right"><b>{fmt(d["qtc"])}</b> ms</td></tr>'
             f'<tr><td>RR</td><td align="right">{fmt(d["rr"])} ms</td></tr></table><br><table><tr><th>Canal</th><th>R (mV)</th><th>ST J+60</th><th>ST J+80</th></tr>')
        for c in d['canales']:
            h += f'<tr><td>{c["canal"]}</td><td align="right">{fmt(c["r_mv"], 2)}</td><td align="right">{fmt(c["st60"], 2)}</td><td align="right">{fmt(c["st80"], 2)}</td></tr>'
        if d['pr'] > 200:
            h += '</table><p style="color:#be185d">PR mayor de 200 ms</p>'
        if d['qrs'] >= 120:
            h += '<p style="color:#be185d">QRS de 120 ms o más</p>'
        self.vals.setText(h + '</table>')

    def guardar(self):
        d = self.medidas()
        if d and getattr(self.ex, 'sesion', None):
            L = self.ex.sesion.d.setdefault('mediciones', []); L[:] = [x for x in L if abs(x['t'] - d['t']) > 0.01]; L.append(d)
            L.sort(key=lambda x: x['t']); self._lista(); self.cambio.emit()

    def _lista(self):
        self.lista.blockSignals(True); self.lista.clear()
        for d in (self.ex.sesion.d.get('mediciones') or []) if getattr(self.ex, 'sesion', None) else []:
            it = QListWidgetItem(f'{d["hora"]} {d["clase"]} · PR {fmt(d["pr"])} · QRS {fmt(d["qrs"])} · QT {fmt(d["qt"])} ms'); it.setData(Qt.UserRole, d['t'])
            self.lista.addItem(it)
        self.lista.blockSignals(False)

    def _cargar(self, it):
        if not it:
            return
        t = it.data(Qt.UserRole); b = self.ex.latido_cercano(t, 0.05)
        for d in self.ex.sesion.d.get('mediciones') or []:
            if abs(d['t'] - t) < 0.01 and b >= 0:
                self.lienzo.b = b; self.lienzo.pos = dict(d['puntos']); self.titulo.setText(f'Medición guardada · {d["hora"]}')
                self.lienzo.update(); self._valores()

    def _borrar(self):
        it = self.lista.currentItem()
        if it:
            t = it.data(Qt.UserRole); L = self.ex.sesion.d.get('mediciones') or []
            L[:] = [x for x in L if abs(x['t'] - t) > 0.01]; self._lista(); self.cambio.emit()


def html_mediciones(ex):
    L = (ex.sesion.d.get('mediciones') or []) if getattr(ex, 'sesion', None) else []
    if not L:
        return '<p class="m">Sin mediciones guardadas.</p>'
    h = '<table><tr><th>Hora</th><th>Clase</th><th>PR</th><th>QRS</th><th>QT</th><th>QTc</th><th>RR</th><th>ST J+80 por canal (mV)</th></tr>'
    for d in L:
        st = ' / '.join(f'{c["canal"]} {fmt(c["st80"], 2)}' for c in d['canales'])
        h += (f'<tr><td>{d["hora"]}</td><td>{d["clase"]}</td><td class="n">{fmt(d["pr"])}</td><td class="n">{fmt(d["qrs"])}</td><td class="n">{fmt(d["qt"])}</td>'
              f'<td class="n">{fmt(d["qtc"])}</td><td class="n">{fmt(d["rr"])}</td><td>{st}</td></tr>')
    return h + '</table>'
