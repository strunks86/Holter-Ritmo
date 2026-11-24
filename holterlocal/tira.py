"""Vista de ECG: tira de varios canales con herramientas de latidos, calibrador, tira y sucesos; barras de sucesos,
barra de tiempo, vista de contexto y una función de dibujo que también usan el informe y la impresión de pantalla."""
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QHBoxLayout, QInputDialog, QLabel, QMenu, QSizePolicy,
                               QToolButton, QToolTip, QVBoxLayout, QWidget)
from scipy.signal import butter, sosfiltfilt

from .estilo import ALIAS_BARRA, COLOR_CLASE, PAPEL, SUCESOS_BARRA, C
from .motor import CLS_NAME, CLS_SYM, ORDEN_ETIQUETAS

# tecla -> clase (las mismas letras de la etiqueta); X también marca artefacto
TECLAS = {getattr(Qt, 'Key_' + s): k for k, s in enumerate(CLS_SYM)}
TECLAS[Qt.Key_X] = 4
COLOR_BARRA = {k: QColor(c) for k, _, c, _ in SUCESOS_BARRA}
PRIO_BARRA = {k: p for k, _, _, p in SUCESOS_BARRA}
NOMBRE_BARRA = {k: n for k, n, _, _ in SUCESOS_BARRA}
EDITABLES = ('artifact', 'af', 'u1', 'u2', 'u3', 'stDep', 'stEle')

# duración por página: (texto, segundos) o mm/s del papel
DURACIONES = [('25 mm/s', ('mm', 25.0)), ('50 mm/s', ('mm', 50.0)), ('12,5 mm/s', ('mm', 12.5)), ('5 s', ('s', 5)),
              ('7,5 s', ('s', 7.5)), ('10 s', ('s', 10)), ('15 s', ('s', 15)), ('30 s', ('s', 30)), ('1 min', ('s', 60)),
              ('1,5 min', ('s', 90)), ('2 min', ('s', 120)), ('2,5 min', ('s', 150)), ('5 min', ('s', 300)),
              ('10 min', ('s', 600)), ('15 min', ('s', 900)), ('20 min', ('s', 1200)), ('30 min', ('s', 1800))]
GANANCIAS = [('×½', 5.0), ('×1', 10.0), ('×2', 20.0), ('×4', 40.0)]
_SOS = {}


def filtro_sos(fs):
    if fs not in _SOS:
        _SOS[fs] = butter(2, [0.5, min(40.0, 0.45 * fs)], btype='bandpass', fs=fs, output='sos')
    return _SOS[fs]


def senal(ex, ch, i0, i1, filtro=True):
    """Señal en mV de un canal entre dos muestras, filtrada 0,5–40 Hz con margen para evitar bordes."""
    if not filtro:
        return ex.mv(ch, i0, i1).astype(np.float64)
    m = int(2 * ex.fs)
    a0 = max(0, i0 - m); a1 = min(ex.n, i1 + m)
    y = ex.mv(ch, a0, a1).astype(np.float64)
    if len(y) > 30:
        try:
            y = sosfiltfilt(filtro_sos(ex.fs), y)
        except ValueError:
            pass
    out = y[max(0, i0 - a0):max(0, i0 - a0) + max(0, min(i1, ex.n) - max(0, i0))]
    return out


def linea(y, fs, t_ini, ta, xa, px_s, ymid, px_mv, ylo, yhi):
    """Polilínea de una señal (muestras desde t_ini) dibujada desde ta en xa; resume en mínimo y máximo por píxel."""
    n = len(y)
    if n < 2:
        return QPolygonF()
    spp = fs / max(1e-6, px_s)
    xs0 = xa + (t_ini - ta) * px_s
    if spp > 2.5:
        ncol = max(1, int(n / spp))
        cut = int(ncol * spp)
        if cut > n:
            ncol -= 1; cut = int(ncol * spp)
        if ncol < 1:
            return QPolygonF()
        idx = (np.arange(ncol + 1) * spp).astype(int)
        mins = np.minimum.reduceat(y[:idx[-1]], idx[:-1]) if idx[-1] > 0 else y[:1]
        maxs = np.maximum.reduceat(y[:idx[-1]], idx[:-1]) if idx[-1] > 0 else y[:1]
        xs = xs0 + np.arange(len(mins))
        pts = []
        for j in range(len(mins)):
            a = min(max(ymid - maxs[j] * px_mv, ylo), yhi); b = min(max(ymid - mins[j] * px_mv, ylo), yhi)
            if j % 2:
                pts.append(QPointF(xs[j], a)); pts.append(QPointF(xs[j], b))
            else:
                pts.append(QPointF(xs[j], b)); pts.append(QPointF(xs[j], a))
        return QPolygonF(pts)
    xs = xs0 + np.arange(n) / fs * px_s
    ys = np.clip(ymid - y * px_mv, ylo, yhi)
    return QPolygonF([QPointF(float(a), float(b)) for a, b in zip(xs, ys)])


def tramos_color(ex, ta, tb):
    """Tramos de la señal que se colorean según la etiqueta del latido, y pausas largas en gris."""
    b0 = max(0, int(np.searchsorted(ex.t, ta)) - 1); b1 = min(len(ex.t), int(np.searchsorted(ex.t, tb)) + 1)
    out = []
    for b in range(b0, b1):
        k = int(ex.cls[b]); t = float(ex.t[b])
        tp = float(ex.t[b - 1]) if b > 0 else t - 1; tn = float(ex.t[b + 1]) if b + 1 < len(ex.t) else t + 1
        if k != 0:
            a = max(t - 0.25, (tp + t) / 2, t - 0.6); z = min(t + 0.45, (t + tn) / 2, t + 0.6)
            out.append((a, z, COLOR_CLASE[k]))
        if tn - t > 2.0:
            out.append((t + 0.45, tn - 0.25, '#9ca3af'))
    return out


def dibujar_ecg(p, rect, ex, t0, dur, chans, mm_mv=10.0, etiquetas=True, rejilla=True, oscuro=False, filtro=True,
                titulo_canal=True, colorear=True, grosor=1.0):
    """Dibuja un tramo de ECG en un rectángulo (papel milimetrado escalado al ancho). Lo usan el informe y la impresión."""
    pal = PAPEL[bool(oscuro)]
    p.save(); p.setClipRect(rect)
    p.fillRect(rect, QColor(pal['papel']))
    px_s = rect.width() / max(0.1, dur)
    px_mm = px_s / 25.0
    top = rect.top() + (14 if etiquetas else 2)
    if rejilla:
        fino = QPen(QColor(pal['rejilla1'])); fino.setWidthF(0.4 * grosor)
        grueso = QPen(QColor(pal['rejilla2'])); grueso.setWidthF(0.8 * grosor)
        step = px_mm if px_mm >= 2.5 else px_mm * 5
        x = rect.left(); j = 0
        while x <= rect.right():
            p.setPen(grueso if (j % 5 == 0 or step > px_mm) else fino); p.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom())); x += step; j += 1
        y = rect.top(); j = 0
        while y <= rect.bottom():
            p.setPen(grueso if (j % 5 == 0 or step > px_mm) else fino); p.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y)); y += step; j += 1
    nch = max(1, len(chans)); hc = (rect.bottom() - top) / nch
    f = QFont(); f.setPointSizeF(6.5)
    for ci, ch in enumerate(chans):
        mid = top + hc * (ci + 0.55)
        i0 = int(t0 * ex.fs); i1 = int((t0 + dur) * ex.fs) + 2
        y = senal(ex, ch, i0, i1, filtro)
        pxmv = mm_mv * px_mm
        pen = QPen(QColor(pal['trazo'])); pen.setWidthF(0.9 * grosor); p.setPen(pen)
        lo, hi = top + ci * hc - 0.3 * hc, top + (ci + 1) * hc + 0.3 * hc
        p.drawPolyline(linea(y, ex.fs, i0 / ex.fs, t0, rect.left(), px_s, mid, pxmv, lo, hi))
        if colorear:
            for a, z, col in tramos_color(ex, t0, t0 + dur):
                ja = max(0, int((a - i0 / ex.fs) * ex.fs)); jz = min(len(y), int((z - i0 / ex.fs) * ex.fs) + 1)
                if jz - ja > 1:
                    pc = QPen(QColor(col)); pc.setWidthF(1.1 * grosor); p.setPen(pc)
                    p.drawPolyline(linea(y[ja:jz], ex.fs, (i0 + ja) / ex.fs, t0, rect.left(), px_s, mid, pxmv, lo, hi))
        if titulo_canal:
            p.setFont(f); p.setPen(QColor(pal['texto2']))
            p.drawText(QPointF(rect.left() + 3, top + ci * hc + 9), ex.names[ch] if ch < len(ex.names) else f'Canal {ch + 1}')
    if etiquetas:
        b0 = int(np.searchsorted(ex.t, t0)); b1 = int(np.searchsorted(ex.t, t0 + dur))
        f2 = QFont(); f2.setPointSizeF(7); f2.setBold(True); p.setFont(f2)
        for b in range(b0, b1):
            k = int(ex.cls[b]); x = rect.left() + (ex.t[b] - t0) * px_s
            p.setPen(QColor(pal['normal'] if k == 0 else COLOR_CLASE[k]))
            p.drawText(QRectF(x - 8, rect.top() + 1, 16, 12), Qt.AlignCenter, CLS_SYM[k])
    p.restore()


# =========================================================================== tira interactiva
class Tira(QWidget):
    tiempo_cambiado = Signal(float)
    latido_elegido = Signal(int)
    reetiquetar = Signal(list, int)
    excluir = Signal(float, float)
    insertar = Signal(float, int)
    borrar = Signal(list)
    aprender = Signal(int, int)
    mostrar_plantilla = Signal(int)
    pedir_tira = Signal(float, float, int)        # inicio, duración, minutos de tira de página (0 si es normal)
    agregar_suceso = Signal(str, float, float)
    borrar_suceso = Signal(dict)
    editar_suceso = Signal(dict)
    detalles_st = Signal(dict)
    navegar = Signal(int)                         # Tab / Mayús+Tab
    vista_cambiada = Signal()
    intervalo = Signal(str, float, float)         # herramientas de intervalo de Darwin2: acción, desde, hasta

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None
        self.t0 = 0.0
        self.modo = ('mm', 25.0)
        self.gan = 10.0
        self.filtro = True
        self.rejilla = True
        self.etiquetas = True
        self.oscuro = False
        self.visibles = None          # canales visibles (índices en ex.chans); None = todos
        self.herramienta = 'latidos'
        self.sel = -1
        self.sels = set()
        self.ancla = -1
        self.cal = None               # [ta, tb, ya, yb]
        self.cal_amp = None           # [x, y1, y2]
        self.marcha = False
        self.cal_activo = 'b'
        self.rango = None
        self.ver_ref = False
        self.barras_on = {k for k, *_ in SUCESOS_BARRA}
        self.tira_dur = 7.5
        self.t_raton = None
        self.inicio_suceso = None
        self.titulo_nav = ''
        self.marcador = None                      # marcador de intervalo (Darwin2: [marcador – actual])
        self._drag = None
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # ---- geometría
    def pxmm(self):
        return max(3.0, self.logicalDpiX() / 25.4)

    def top(self):
        return 30 if self._con_etiquetas() else 6

    def lineas(self):
        """Segundos por línea y número de líneas de la página."""
        if self.modo[0] == 'mm':
            return self.width() / (self.modo[1] * self.pxmm()), 1
        d = float(self.modo[1])
        if d <= 30:
            return d, 1
        L = 30.0 if d <= 300 else 60.0
        return L, int(round(d / L))

    def _con_etiquetas(self):
        return self.etiquetas and self.lineas()[1] <= 4

    def ancho_seg(self):
        L, nl = self.lineas()
        return L * nl

    def alto_linea(self):
        return (self.height() - self.top()) / max(1, self.lineas()[1])

    def px_s(self):
        return self.width() / self.lineas()[0]

    def x_de(self, t):
        L, _ = self.lineas()
        dt = t - self.t0
        li = int(np.floor(dt / L)) if dt >= 0 else 0
        return (dt - li * L) * self.px_s()

    def pos(self, t):
        L, nl = self.lineas()
        dt = t - self.t0
        li = int(min(max(np.floor(dt / L), 0), nl - 1))
        return li, (dt - li * L) * self.px_s()

    def t_de(self, x, y=None):
        L, nl = self.lineas()
        li = 0
        if y is not None and nl > 1:
            li = int(min(max((y - self.top()) // self.alto_linea(), 0), nl - 1))
        return self.t0 + li * L + x / self.px_s()

    def canales(self):
        ex = self.ex
        idx = list(range(len(ex.chans))) if self.visibles is None else [i for i in self.visibles if i < len(ex.chans)]
        return idx or [0]

    def px_mv(self):
        nl = self.lineas()[1]; nc = len(self.canales())
        h = self.alto_linea() / max(1, nc)
        base = self.gan * self.pxmm()
        return base if nl == 1 else base * min(1.0, h / (25 * self.pxmm()))

    def set_examen(self, ex):
        self.ex = ex; self.t0 = 0.0; self.sel = -1; self.sels = set(); self.cal = None; self.cal_amp = None; self.rango = None
        self.inicio_suceso = None
        self.update()

    def ir_a(self, t, centrar=True):
        if not self.ex:
            return
        w = self.ancho_seg()
        t0 = t - w * (0.4 if self.lineas()[1] == 1 else 0.5) if centrar else t
        self.t0 = float(min(max(0.0, t0), max(0.0, self.ex.dur - w)))
        self.update(); self.tiempo_cambiado.emit(self.t0); self.vista_cambiada.emit()

    def seleccionar(self, b, centrar=True, agregar=False):
        if not self.ex or b < 0 or b >= len(self.ex.t):
            return
        self.sel = int(b)
        if agregar:
            self.sels.add(self.sel)
        else:
            self.sels = {self.sel}
        t = float(self.ex.t[b])
        if centrar or not (self.t0 + 0.3 < t < self.t0 + self.ancho_seg() - 0.3):
            self.ir_a(t)
        self.update(); self.latido_elegido.emit(self.sel)

    def seleccion(self):
        s = sorted(b for b in self.sels if 0 <= b < len(self.ex.t)) if self.ex else []
        if not s and self.sel >= 0:
            s = [self.sel]
        return s

    # ---- dibujo
    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        pal = PAPEL[self.oscuro]
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor(pal['papel']))
        ex = self.ex
        if not ex:
            p.setPen(QColor(pal['texto2']))
            p.drawText(self.rect(), Qt.AlignCenter, 'Abre un examen con Examen → Abrir examen (Ctrl+O)')
            return
        L, nl = self.lineas(); top = self.top(); lh = self.alto_linea(); pxs = self.px_s(); pm = self.pxmm()
        chs = self.canales(); nc = len(chs); hc = lh / nc; pxmv = self.px_mv()
        fnt = QFont(self.font()); fnt.setPointSizeF(8.5)
        f2 = QFont(self.font()); f2.setBold(True); f2.setPointSizeF(9.5 if nl == 1 else 8)
        f3 = QFont(self.font()); f3.setPointSizeF(7.5)
        evs = self._eventos_vista()
        for li in range(nl):
            ta = self.t0 + li * L; tb = ta + L
            y0 = top + li * lh
            if tb > ex.dur + L:
                break
            # cuadrícula
            if self.rejilla:
                self._rejilla(p, pal, ta, y0, lh, pxs, pxmv)
            # calidad por segundo y tramos excluidos
            for s in range(max(0, int(ta)), min(int(tb) + 1, len(ex.best))):
                xa, xb = (s - ta) * pxs, (s + 1 - ta) * pxs
                if ex.excl[s]:
                    p.fillRect(QRectF(xa, y0, xb - xa, lh), QColor(120, 120, 120, 45))
                for ci, c in enumerate(chs):
                    q = ex.qual[c][s] if s < len(ex.qual[c]) else 0
                    if q == 2:
                        p.fillRect(QRectF(xa, y0 + ci * hc, xb - xa, hc), QColor(230, 160, 40, 40))
            # rango elegido
            if self.rango:
                a, b = sorted(self.rango)
                a, b = max(a, ta), min(b, tb)
                if b > a:
                    p.fillRect(QRectF((a - ta) * pxs, y0, (b - a) * pxs, lh), QColor(60, 120, 220, 50))
            if self.inicio_suceso is not None and ta <= self.inicio_suceso[1] <= tb:
                x = (self.inicio_suceso[1] - ta) * pxs
                p.setPen(QPen(QColor('#d946ef'), 2)); p.drawLine(QPointF(x, y0), QPointF(x, y0 + lh))
            # latidos elegidos
            b0 = int(np.searchsorted(ex.t, ta)); b1 = int(np.searchsorted(ex.t, tb))
            for b in range(b0, b1):
                if b in self.sels or b == self.sel:
                    x = (ex.t[b] - ta) * pxs
                    p.fillRect(QRectF(x - 8, y0, 16, lh), QColor(255, 220, 0, 80 if b == self.sel else 55))
            # canales
            tramos = tramos_color(ex, ta, tb)
            i0 = int(ta * ex.fs); i1 = int(tb * ex.fs) + 2
            for ci, c in enumerate(chs):
                mid = y0 + hc * (ci + 0.55)
                lo, hi = y0 + ci * hc - 0.4 * hc, y0 + (ci + 1) * hc + 0.4 * hc
                y = senal(ex, ex.chans[c], i0, i1, self.filtro)
                pen = QPen(QColor(pal['trazo'])); pen.setWidthF(1.15 if nl == 1 else 0.9); p.setPen(pen)
                p.drawPolyline(linea(y, ex.fs, i0 / ex.fs, ta, 0, pxs, mid, pxmv, lo, hi))
                for a, z, col in tramos:
                    ja = max(0, int((a - i0 / ex.fs) * ex.fs)); jz = min(len(y), int((z - i0 / ex.fs) * ex.fs) + 1)
                    if jz - ja > 1:
                        pc = QPen(QColor(col)); pc.setWidthF(1.3 if nl == 1 else 1.0); p.setPen(pc)
                        p.drawPolyline(linea(y[ja:jz], ex.fs, (i0 + ja) / ex.fs, ta, 0, pxs, mid, pxmv, lo, hi))
                if li == 0:
                    if nl == 1:
                        self._pulso(p, pal, mid, pm)
                    p.setFont(fnt); p.setPen(QColor(pal['texto2']))
                    nombre = ex.names[ex.chans[c]] if ex.chans[c] < len(ex.names) else f'Canal {c + 1}'
                    p.drawText(QPointF(22 if nl == 1 else 4, y0 + ci * hc + 13), nombre)
            if nl > 1:
                p.setFont(f3); p.setPen(QColor(pal['texto2']))
                p.drawText(QPointF(4, y0 + lh - 3), ex.reloj(ta))
            # etiquetas de latido
            if self._con_etiquetas():
                for b in range(b0, b1):
                    x = (ex.t[b] - ta) * pxs; k = int(ex.cls[b])
                    p.setPen(QColor(pal['normal'] if k == 0 else COLOR_CLASE[k])); p.setFont(f2)
                    p.drawText(QRectF(x - 10, y0 - 28 if li == 0 else y0 + 1, 20, 14), Qt.AlignCenter, CLS_SYM[k] + ('*' if ex.edited[b] else ''))
                    if ex.afBeat[b] and li == 0:
                        p.fillRect(QRectF(x - 6, y0 - 14, 12, 2), QColor(C['fa']))
                    rr = ex.rr[b]
                    if nl == 1 and rr == rr and b > b0 and pxs * rr > 34:
                        p.setFont(f3); p.setPen(QColor(pal['texto2']))
                        xp = (ex.t[b - 1] - ta) * pxs
                        p.drawText(QRectF((xp + x) / 2 - 20, y0 - 13, 40, 12), Qt.AlignCenter, '%d' % round(rr * 1000))
            else:
                for b in range(b0, b1):
                    k = int(ex.cls[b])
                    if k != 0:
                        x = (ex.t[b] - ta) * pxs
                        p.fillRect(QRectF(x - 1.5, y0 + 1, 3, 6), QColor(COLOR_CLASE[k]))
            # barras de sucesos (debajo de los canales)
            self._barras(p, evs, ta, tb, pxs, y0 + lh - 7, 6)
            # anotaciones de referencia
            if self.ver_ref and ex.ref is not None:
                rt = ex.ref['t']; r0 = int(np.searchsorted(rt, ta)); r1 = int(np.searchsorted(rt, tb))
                p.setFont(f3)
                for i in range(r0, r1):
                    x = (rt[i] - ta) * pxs; p.setPen(QColor(C['ref'])); p.drawText(QRectF(x - 8, y0 + lh - 22, 16, 12), Qt.AlignCenter, ex.ref['sym'][i])
            if li > 0:
                p.setPen(QPen(QColor(pal['rejilla2']), 1.2)); p.drawLine(QPointF(0, y0), QPointF(W, y0))
        self._calibradores(p, pal, top, H, f2)
        # herramienta de tira: marco rojo que sigue al ratón
        if self.herramienta == 'tira' and self.t_raton is not None and nl == 1:
            xa = self.x_de(self.t_raton); w = self.tira_dur * pxs
            p.setPen(QPen(QColor('#dc2626'), 2)); p.setBrush(Qt.NoBrush); p.drawRect(QRectF(xa, top, w, H - top - 2))
            p.setFont(f3); p.drawText(QPointF(xa + 4, top + 12), '%s · %s s' % (ex.reloj(self.t_raton), str(self.tira_dur).replace('.', ',')))
        if self.marcador is not None and self.t0 <= self.marcador <= self.t0 + self.ancho_seg():
            li, xm = self.pos(self.marcador); ya = self.top() + li * self.alto_linea()
            p.setPen(QPen(QColor('#7c3aed'), 1.5, Qt.DashLine)); p.drawLine(QPointF(xm, ya), QPointF(xm, ya + self.alto_linea()))
            p.setFont(f3); p.setPen(QColor('#7c3aed')); p.drawText(QPointF(xm + 3, ya + 11), 'Marcador ' + ex.reloj(self.marcador))
        if self.titulo_nav:
            p.setFont(f2); r = QRectF(W / 2 - 180, 1, 360, 16)
            p.fillRect(r, QColor(255, 255, 255, 200)); p.setPen(QColor('#0b57d0')); p.drawText(r, Qt.AlignCenter, self.titulo_nav)
        p.end()

    def _rejilla(self, p, pal, ta, y0, lh, pxs, pxmv):
        fino = QPen(QColor(pal['rejilla1'])); fino.setWidthF(0.6)
        grueso = QPen(QColor(pal['rejilla2'])); grueso.setWidthF(1.0)
        W = self.width()
        mm_t = pxs * 0.04   # 1 mm = 40 ms
        if mm_t >= 3:
            paso, cada = 0.04, 5
        elif mm_t * 5 >= 4:
            paso, cada = 0.2, 5
        else:
            paso, cada = 1.0, 5
        k0 = int(np.floor(ta / paso)); x = (k0 * paso - ta) * pxs; k = k0
        while x < W:
            if x >= 0:
                p.setPen(grueso if k % cada == 0 else fino); p.drawLine(QPointF(x, y0), QPointF(x, y0 + lh))
            x += paso * pxs; k += 1
        mmv = pxmv * 0.1
        pa = mmv if mmv >= 3 else mmv * 5
        y = y0; j = 0
        while y < y0 + lh:
            p.setPen(grueso if (j % 5 == 0 or pa > mmv) else fino); p.drawLine(QPointF(0, y), QPointF(W, y)); y += pa; j += 1

    def _pulso(self, p, pal, base, pm):
        p.setPen(QPen(QColor(pal['texto']), 1.2))
        cx = 4; hh = self.px_mv(); w = 0.2 * self.px_s()
        path = QPainterPath(QPointF(cx, base)); path.lineTo(cx + 2, base); path.lineTo(cx + 2, base - hh)
        path.lineTo(cx + 2 + w, base - hh); path.lineTo(cx + 2 + w, base); path.lineTo(cx + 4 + w, base); p.drawPath(path)

    def _eventos_vista(self):
        ex = self.ex
        t0, t1 = self.t0, self.t0 + self.ancho_seg()
        out = []
        for e in ex.eventos:
            k = ALIAS_BARRA.get(e['type'], e['type'])
            if k not in PRIO_BARRA or k not in self.barras_on:
                continue
            a, b = float(e['t0']), float(max(e['t1'], e['t0'] + 0.3))
            if b >= t0 and a <= t1:
                out.append((PRIO_BARRA[k], k, a, b, e))
        out.sort(key=lambda x: -x[0])   # la de más prioridad se dibuja al final, encima
        return out

    def tipos_en_vista(self):
        ex = self.ex
        if not ex:
            return set()
        t0, t1 = self.t0, self.t0 + self.ancho_seg()
        return {ALIAS_BARRA.get(e['type'], e['type']) for e in ex.eventos if e['t1'] >= t0 and e['t0'] <= t1}

    def _barras(self, p, evs, ta, tb, pxs, y, h):
        for _, k, a, b, e in evs:
            a2, b2 = max(a, ta), min(b, tb)
            if b2 > a2:
                p.fillRect(QRectF((a2 - ta) * pxs, y, max(2.0, (b2 - a2) * pxs), h), COLOR_BARRA[k])

    def _calibradores(self, p, pal, top, H, f2):
        if self.cal:
            ta, tb, ya, yb = self.cal
            xa, xb = self.x_de(ta), self.x_de(tb)
            pen = QPen(QColor(C['calibrador'])); pen.setWidthF(1.3); pen.setStyle(Qt.DashLine)
            p.setPen(pen); p.drawLine(QPointF(xa, top), QPointF(xa, H)); p.drawLine(QPointF(xb, top), QPointF(xb, H))
            pen.setStyle(Qt.SolidLine); p.setPen(pen); p.drawLine(QPointF(xa, ya), QPointF(xb, ya))
            for x in (xa, xb):
                p.setBrush(QColor(C['calibrador'])); p.drawEllipse(QPointF(x, ya), 3, 3)
            dt = abs(tb - ta)
            if self.marcha and dt > 0.05:
                pm = QPen(QColor(C['calibrador'])); pm.setWidthF(1.0); pm.setStyle(Qt.DotLine); p.setPen(pm)
                k = 1
                while True:
                    xm = self.x_de(min(ta, tb) + k * dt) if False else xa + (xb - xa) * (k + 1)
                    xm2 = xa - (xb - xa) * k
                    if (xm > self.width() or xm < 0) and (xm2 < 0 or xm2 > self.width()):
                        break
                    for xx in (xm, xm2):
                        if 0 <= xx <= self.width():
                            p.drawLine(QPointF(xx, top), QPointF(xx, H))
                    k += 1
                    if k > 200:
                        break
            txt = '%d ms' % round(dt * 1000)
            if dt > 0.05:
                txt += '  ·  %d lpm' % round(60 / dt)
            p.setFont(f2)
            r = QRectF(min(xa, xb) + 4, max(top + 2, ya - 22), 170, 18)
            p.fillRect(r, QColor(255, 255, 255, 230)); p.setPen(QColor(C['calibrador'])); p.drawText(r, Qt.AlignLeft | Qt.AlignVCenter, ' ' + txt)
        if self.cal_amp:
            x, y1, y2 = self.cal_amp
            pen = QPen(QColor('#0f766e')); pen.setWidthF(1.3); pen.setStyle(Qt.DashLine); p.setPen(pen)
            p.drawLine(QPointF(x - 40, y1), QPointF(x + 40, y1)); p.drawLine(QPointF(x - 40, y2), QPointF(x + 40, y2))
            pen.setStyle(Qt.SolidLine); p.setPen(pen); p.drawLine(QPointF(x, y1), QPointF(x, y2))
            dmv = (y2 - y1) / max(1e-6, self.px_mv())
            r = QRectF(x + 44, (y1 + y2) / 2 - 9, 90, 18)
            p.fillRect(r, QColor(255, 255, 255, 230)); p.setPen(QColor('#0f766e')); p.setFont(f2)
            p.drawText(r, Qt.AlignLeft | Qt.AlignVCenter, ' %+.2f mV' % dmv)

    # ---- ratón
    def _cerca_cal(self, x, y):
        if self.cal:
            xa, xb = self.x_de(self.cal[0]), self.x_de(self.cal[1])
            if abs(x - xa) < 6:
                return 'a'
            if abs(x - xb) < 6:
                return 'b'
            if min(xa, xb) < x < max(xa, xb) and abs(y - self.cal[2]) < 6:
                return 'h'
        if self.cal_amp:
            xx, y1, y2 = self.cal_amp
            if abs(x - xx) < 44:
                if abs(y - y1) < 6:
                    return 'y1'
                if abs(y - y2) < 6:
                    return 'y2'
        return None

    def mousePressEvent(self, e):
        self.setFocus()
        if not self.ex:
            return
        x, y = e.position().x(), e.position().y()
        t = self.t_de(x, y); mods = e.modifiers()
        if e.button() == Qt.RightButton:
            self._menu_contexto(e, t, x, y); return
        if e.button() != Qt.LeftButton:
            return
        if mods & Qt.AltModifier:
            self.ir_a(t); return
        h = self.herramienta
        if h == 'calibrador':
            c = self._cerca_cal(x, y)
            if c:
                self._drag = ('cal', c, x, y); self.cal_activo = c
            else:
                self.cal = [t, t, y, y]; self._drag = ('cal', 'b', x, y); self.cal_activo = 'b'
            self.update(); return
        if h == 'tira':
            self.pedir_tira.emit(float(t), float(self.tira_dur), 0); return
        if h == 'suceso':
            if mods & Qt.ShiftModifier and self.inicio_suceso is not None:
                self._menu_final_suceso(e, t); return
            self.rango = [t, t]; self._drag = ('rango', None, x, y); self.update(); return
        # herramienta de latidos
        b = self.ex.latido_cercano(t, 0.25)
        if mods & Qt.ShiftModifier and b >= 0 and self.ancla >= 0:
            a, z = sorted((self.ancla, b)); self.sels = set(range(a, z + 1)); self.sel = b; self.update(); self.latido_elegido.emit(b); return
        if mods & Qt.ControlModifier and b >= 0:
            if b in self.sels:
                self.sels.discard(b)
            else:
                self.sels.add(b)
            self.sel = b; self.update(); self.latido_elegido.emit(b); return
        self._drag = ('lat', b, x, y); self.rango = None
        self.update()

    def mouseDoubleClickEvent(self, e):
        if not self.ex or self.herramienta != 'latidos':
            return
        b = self.ex.latido_cercano(self.t_de(e.position().x(), e.position().y()), 0.25)
        if b >= 0 and self.ex.tid[b] >= 0:
            self.mostrar_plantilla.emit(int(self.ex.tid[b]))

    def mouseMoveEvent(self, e):
        if not self.ex:
            return
        x, y = e.position().x(), e.position().y()
        t = self.t_de(x, y); self.t_raton = t
        if self.herramienta == 'tira':
            self.update()
        if not (e.buttons() & Qt.LeftButton) or not self._drag:
            return
        kind, c, x0, y0 = self._drag
        if kind == 'cal' and self.cal is not None:
            if c == 'a':
                self.cal[0] = t
            elif c == 'b':
                self.cal[1] = t; self.cal[3] = y
            elif c == 'h':
                self.cal[2] = y
            elif c in ('y1', 'y2') and self.cal_amp:
                self.cal_amp[1 if c == 'y1' else 2] = y
        elif kind in ('lat', 'rango'):
            if abs(x - x0) > 5 or self.rango is not None:
                t0 = self.t_de(x0, y0)
                self.rango = [t0, t]
        self.update()

    def mouseReleaseEvent(self, e):
        if not self.ex or not self._drag:
            return
        kind, c, x0, y0 = self._drag; self._drag = None
        t = self.t_de(e.position().x(), e.position().y())
        if kind == 'cal':
            return
        if kind == 'rango' or (kind == 'lat' and self.rango is not None):
            a, b = sorted(self.rango)
            if b - a < 0.15:
                self.rango = None
            elif self.herramienta == 'suceso':
                self._menu_suceso(e, a, b); return
            else:
                b0 = int(np.searchsorted(self.ex.t, a)); b1 = int(np.searchsorted(self.ex.t, b))
                self.sels = set(range(b0, b1))
                if b1 > b0:
                    self.sel = b0; self.ancla = b0; self.latido_elegido.emit(b0)
                self.update(); return
        if kind == 'lat' and c is not None and c >= 0:
            self.ancla = c; self.seleccionar(c, centrar=False)
        self.update()

    def leaveEvent(self, e):
        self.t_raton = None
        if self.herramienta == 'tira':
            self.update()

    # ---- menús
    def _submenu_etiquetas(self, menu, titulo, fn):
        sub = menu.addMenu(titulo)
        for k in ORDEN_ETIQUETAS:
            a = sub.addAction(f'{CLS_NAME[k]}\t{CLS_SYM[k]}', lambda k=k: fn(k))
        sub.addSeparator(); sub.addAction(f'{CLS_NAME[4]}\tA', lambda: fn(4))
        return sub

    def _menu_contexto(self, e, t, x, y):
        ex = self.ex; m = QMenu(self); g = e.globalPosition().toPoint()
        h = self.herramienta
        if h == 'calibrador' and self.cal:
            a = m.addAction('Marcha (marcas equidistantes)', self._toggle_marcha); a.setCheckable(True); a.setChecked(self.marcha)
            m.addAction('Calibrador de amplitud aquí', lambda: self._amp_en(x, y))
            m.addAction('Quitar calibradores', self._sin_cal)
            m.exec(g); return
        if h == 'tira':
            m.addAction('Ampliar 7,5 s', lambda: self._tira_dur(7.5))
            if self.tira_dur > 7.5:
                m.addAction('Encoger 7,5 s', lambda: self._tira_dur(-7.5))
            sub = m.addMenu('Tira de página de un canal')
            for mi in (5, 10, 15, 20, 30, 45, 60):
                sub.addAction(f'{mi} minutos', lambda mi=mi: self.pedir_tira.emit(float(t), mi * 60.0, mi))
            m.addAction('Mover al centro', lambda: self.ir_a(t))
            m.exec(g); return
        # sucesos bajo el puntero
        evs = [ev for ev in ex.eventos if ev['t0'] - 0.3 <= t <= ev['t1'] + 0.3 and ALIAS_BARRA.get(ev['type'], ev['type']) in EDITABLES]
        if h == 'suceso':
            if evs:
                sb = m.addMenu('Borrar suceso')
                for ev in evs:
                    sb.addAction(self._nombre_ev(ev), lambda ev=ev: self.borrar_suceso.emit(ev))
                se = m.addMenu('Editar tiempos del suceso')
                for ev in evs:
                    se.addAction(self._nombre_ev(ev), lambda ev=ev: self.editar_suceso.emit(ev))
                st = [ev for ev in evs if ev['type'] in ('stDep', 'stEle')]
                for ev in st:
                    m.addAction('Editar detalles del suceso ST…', lambda ev=ev: self.detalles_st.emit(ev))
            else:
                m.addAction('(no hay sucesos editables aquí)').setEnabled(False)
            if self.inicio_suceso is not None:
                self._acciones_final(m, t)
            m.addAction('Mover al centro', lambda: self.ir_a(t))
            m.exec(g); return
        # herramienta de latidos
        sel = self.seleccion()
        b = ex.latido_cercano(t, 0.25)
        if b >= 0 and b not in sel:
            sel = [b]; self.sel = b; self.sels = {b}; self.update(); self.latido_elegido.emit(b)
        if sel:
            self._submenu_etiquetas(m, 'Etiquetar %d latido(s) como' % len(sel), lambda k: self.reetiquetar.emit(sel, k))
            if len(sel) == 1 and ex.tid[sel[0]] >= 0:
                tid = int(ex.tid[sel[0]])
                self._submenu_etiquetas(m, 'Aprender: toda su morfología como', lambda k: self.aprender.emit(tid, k))
                m.addAction('Ver su plantilla', lambda: self.mostrar_plantilla.emit(tid))
            m.addAction('Borrar latido(s)\tSupr', lambda: self.borrar.emit(sel))
        cerca = ex.latido_cercano(t, 0.1)
        if cerca < 0:
            si = m.addMenu('Insertar latido aquí')
            for k in ORDEN_ETIQUETAS:
                si.addAction(f'{CLS_NAME[k]}\tMayús+{CLS_SYM[k]}', lambda k=k: self.insertar.emit(float(t), k))
        if self.rango:
            a, z = sorted(self.rango)
            m.addSeparator()
            m.addAction('Marcar %.1f s como artefacto (no se informa)' % (z - a), lambda: self.excluir.emit(a, z))
            m.addAction('Medir: %.0f ms, %d latidos' % ((z - a) * 1000, len(sel)))
        m.addSeparator()
        self._menu_intervalos(m, float(t))
        m.addSeparator()
        m.addAction('Mover al centro\tAlt+clic', lambda: self.ir_a(t))
        m.addAction('Agregar una tira de 7,5 s aquí', lambda: self.pedir_tira.emit(float(t), 7.5, 0))
        m.exec(g)

    def _menu_intervalos(self, m, t):
        """Herramientas de selección de Darwin2: marcador e intervalos [marcador–actual], [inicio–actual] y [actual–final]."""
        mi = m.addMenu('Intervalo')
        mi.addAction('Poner el marcador aquí (%s)' % self.ex.reloj(t), lambda: self._poner_marcador(t))
        if self.marcador is not None:
            mi.addAction('Quitar el marcador', lambda: self._poner_marcador(None))
        rangos = []
        if self.marcador is not None and abs(self.marcador - t) > 0.05:
            rangos.append(('[marcador – aquí]', min(self.marcador, t), max(self.marcador, t)))
        rangos += [('[inicio – aquí]', 0.0, t), ('[aquí – final]', t, float(self.ex.dur))]
        acciones = [('artifact', 'Marcar como artefacto (excluir)'), ('incluir', 'Devolver al análisis (incluir)'), ('af', 'Marcar fibrilación auricular'),
                    ('noaf', 'Quitar fibrilación auricular'), ('borrar', 'Borrar todos los latidos')]
        for nom, a, b in rangos:
            sub = mi.addMenu('%s  %s → %s' % (nom, self.ex.reloj(a), self.ex.reloj(b)))
            for k, txt in acciones:
                sub.addAction(txt, lambda k=k, a=a, b=b: self.intervalo.emit(k, a, b))
            se = sub.addMenu('Etiquetar todos los latidos como')
            for k in ORDEN_ETIQUETAS:
                se.addAction(f'{CLS_NAME[k]}', lambda k=k, a=a, b=b: self.intervalo.emit('cls%d' % k, a, b))
            sub.addAction('Comentario en el diario…', lambda a=a, b=b: self.intervalo.emit('diario', a, b))
        mi.addSeparator()
        mi.addAction('Fijar este latido como FC máxima', lambda: self.intervalo.emit('fc_max', t, t))
        mi.addAction('Fijar este latido como FC mínima', lambda: self.intervalo.emit('fc_min', t, t))

    def _poner_marcador(self, t):
        self.marcador = t; self.update()

    def _nombre_ev(self, ev):
        k = ALIAS_BARRA.get(ev['type'], ev['type'])
        return '%s (%s)' % (ev.get('label') or NOMBRE_BARRA.get(k, k), self.ex.reloj(ev['t0']))

    def _menu_suceso(self, e, a, b):
        m = QMenu(self)
        sub = m.addMenu('Agregar suceso de %.1f s' % (b - a))
        for k in ('artifact', 'af', 'u1', 'u2', 'u3'):
            sub.addAction(self.nombre_tipo(k), lambda k=k: self.agregar_suceso.emit(k, a, b))
        sub2 = m.addMenu('Quitar fibrilación auricular de este tramo')
        sub2.addAction('Quitar', lambda: self.agregar_suceso.emit('noaf', a, b))
        sub3 = m.addMenu('Ajustar inicio de suceso (sigue en otra página)')
        for k in ('artifact', 'af', 'u1', 'u2', 'u3'):
            sub3.addAction(self.nombre_tipo(k), lambda k=k: self._fijar_inicio(k, a))
        m.exec(e.globalPosition().toPoint())
        self.rango = None; self.update()

    def nombre_tipo(self, k):
        if k.startswith('u') and getattr(self.ex, 'sesion', None):
            return self.ex.sesion.etiqueta_usuario(int(k[1]))
        return NOMBRE_BARRA.get(k, k)

    def _fijar_inicio(self, k, a):
        self.inicio_suceso = (k, a); self.update()

    def _acciones_final(self, m, t):
        k, a = self.inicio_suceso
        if t > a:
            m.addAction('Ajustar final del suceso: %s' % self.nombre_tipo(k), lambda: self._cerrar_suceso(t))
        m.addAction('Cancelar el suceso pendiente', lambda: self._fijar_inicio_none())

    def _fijar_inicio_none(self):
        self.inicio_suceso = None; self.update()

    def _menu_final_suceso(self, e, t):
        m = QMenu(self); self._acciones_final(m, t); m.exec(e.globalPosition().toPoint())

    def _cerrar_suceso(self, t):
        k, a = self.inicio_suceso; self.inicio_suceso = None
        self.agregar_suceso.emit(k, float(a), float(t)); self.update()

    def _toggle_marcha(self):
        self.marcha = not self.marcha; self.update()

    def _amp_en(self, x, y):
        self.cal_amp = [x, y - 20, y + 20]; self.cal_activo = 'y2'; self.update()

    def _sin_cal(self):
        self.cal = None; self.cal_amp = None; self.marcha = False; self.update()

    def _tira_dur(self, d):
        self.tira_dur = max(7.5, min(60.0, self.tira_dur + d)); self.update()

    # ---- teclado y rueda
    def wheelEvent(self, e):
        if not self.ex:
            return
        d = e.angleDelta().y() or e.angleDelta().x()
        if e.modifiers() & Qt.ControlModifier:
            self.gan = float(min(40, max(2.5, self.gan * (1.25 if d > 0 else 0.8)))); self.update(); self.vista_cambiada.emit(); return
        self.ir_a(self.t0 - np.sign(d) * self.ancho_seg() * 0.25, centrar=False)

    def event(self, e):
        # el Tab navega por sucesos y no cambia el foco
        if e.type() == e.Type.KeyPress and e.key() in (Qt.Key_Tab, Qt.Key_Backtab):
            self.navegar.emit(-1 if (e.key() == Qt.Key_Backtab or e.modifiers() & Qt.ShiftModifier) else 1)
            return True
        return super().event(e)

    def _mover_cal(self, dx, dy):
        c = self.cal_activo
        if self.cal and c in ('a', 'b', 'h'):
            dt = dx / self.px_s()
            if c == 'a':
                self.cal[0] += dt
            elif c == 'b':
                self.cal[1] += dt
            self.cal[2] += dy
        elif self.cal_amp and c in ('y1', 'y2'):
            self.cal_amp[1 if c == 'y1' else 2] += dy
        self.update()

    def keyPressEvent(self, e):
        if not self.ex:
            return
        k = e.key(); mods = e.modifiers()
        if self.herramienta == 'calibrador' and (self.cal or self.cal_amp) and k in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down, Qt.Key_Plus, Qt.Key_Minus) and mods & (Qt.ControlModifier | Qt.ShiftModifier):
            paso = 10 if mods & Qt.ShiftModifier else 1
            if k in (Qt.Key_Plus, Qt.Key_Minus) and self.cal:
                self.cal[1] += (1 if k == Qt.Key_Plus else -1) * (1 if self.cal[1] >= self.cal[0] else -1) / self.px_s(); self.update(); return
            dx = {Qt.Key_Left: -paso, Qt.Key_Right: paso}.get(k, 0); dy = {Qt.Key_Up: -paso, Qt.Key_Down: paso}.get(k, 0)
            self._mover_cal(dx, dy); return
        if k in TECLAS and not (mods & (Qt.ControlModifier | Qt.AltModifier)):
            if mods & Qt.ShiftModifier:
                t = self.t_raton
                if t is not None and self.ex.latido_cercano(t, 0.1) < 0 and TECLAS[k] != 4:
                    self.insertar.emit(float(t), TECLAS[k])
                return
            sel = self.seleccion()
            if sel:
                self.reetiquetar.emit(sel, TECLAS[k])
            return
        if k in (Qt.Key_Delete, Qt.Key_Backspace):
            sel = self.seleccion()
            if sel:
                self.borrar.emit(sel)
            return
        if k == Qt.Key_Right:
            nb = self.sel + 1 if self.sel >= 0 else int(np.searchsorted(self.ex.t, self.t0))
            self.ancla = nb; self.seleccionar(nb, centrar=False)
        elif k == Qt.Key_Left:
            nb = max(0, self.sel - 1); self.ancla = nb; self.seleccionar(nb, centrar=False)
        elif k in (Qt.Key_PageDown, Qt.Key_Space):
            self.pagina(1)
        elif k == Qt.Key_PageUp:
            self.pagina(-1)
        elif k == Qt.Key_Home:
            self.ir_a(0, centrar=False)
        elif k == Qt.Key_End:
            self.ir_a(self.ex.dur, centrar=False)
        elif k == Qt.Key_Escape:
            self.cal = None; self.cal_amp = None; self.rango = None; self.sels = set(); self.inicio_suceso = None; self.update()
        else:
            super().keyPressEvent(e)

    def pagina(self, d):
        self.ir_a(self.t0 + d * self.ancho_seg() * (0.9 if self.lineas()[1] == 1 else 1.0), centrar=False)

    def linea_sig(self, d):
        L, nl = self.lineas()
        self.ir_a(self.t0 + d * (L if nl > 1 else L / 4), centrar=False)


# =========================================================================== barra de tiempo
class BarraTiempo(QWidget):
    ir = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.t0 = 0.0; self.w = 10.0
        self.setFixedHeight(28); self.setMouseTracking(True)
        self.setToolTip('Barra de tiempo: clic para ir a esa hora')

    def set_vista(self, t0, w):
        self.t0, self.w = t0, w; self.update()

    def _t(self, x):
        return max(0.0, min(self.ex.dur, x / max(1, self.width()) * self.ex.dur))

    def paintEvent(self, ev):
        p = QPainter(self); W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor('#eef1f5'))
        ex = self.ex
        if not ex:
            return
        k = W / max(1.0, ex.dur)
        for e in ex.eventos:
            ty = ALIAS_BARRA.get(e['type'], e['type'])
            if ty in ('af', 'artifact'):
                p.fillRect(QRectF(e['t0'] * k, 2, max(1.0, (e['t1'] - e['t0']) * k), 7), COLOR_BARRA[ty])
            elif ty in ('vt', 'pause', 'svt'):
                p.fillRect(QRectF(e['t0'] * k - 0.5, 9, 1.5, 6), COLOR_BARRA[ty])
        f = QFont(self.font()); f.setPointSizeF(7); p.setFont(f)
        q = 900.0; n = int(ex.dur // q) + 1
        off = ex.seg_inicio() % 900
        for i in range(-1, n + 1):
            t = i * q - off
            if t < 0 or t > ex.dur:
                continue
            x = t * k; reloj = int(round(ex.seg_inicio() + t)) % 86400
            hora = reloj % 3600 == 0
            p.setPen(QPen(QColor('#4b5563' if hora else '#9ca3af'), 1)); p.drawLine(QPointF(x, H - (12 if hora else 7)), QPointF(x, H))
            if hora and W / max(1, ex.dur / 3600) > 26:
                p.drawText(QPointF(x + 2, H - 3), '%02d' % (reloj // 3600))
        p.setPen(QPen(QColor('#dc2626'), 1.5)); p.setBrush(QColor(220, 38, 38, 60))
        p.drawRect(QRectF(self.t0 * k, 1, max(3.0, self.w * k), H - 2))
        p.end()

    def mousePressEvent(self, e):
        if self.ex:
            self.ir.emit(self._t(e.position().x()))

    def mouseMoveEvent(self, e):
        if not self.ex:
            return
        t = self._t(e.position().x())
        if e.buttons() & Qt.LeftButton:
            self.ir.emit(t)
        QToolTip.showText(e.globalPosition().toPoint(), self.ex.reloj(t), self)


# =========================================================================== vista de contexto
class VistaContexto(QWidget):
    """Un canal, 60 s por fila, alrededor de lo que muestra la tira; rectángulo rosa = vista actual."""
    centrar = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.canal = 0; self.t_vista = 0.0; self.w_vista = 10.0; self.oscuro = False; self.tiras = []
        self.setMinimumHeight(120); self.setMouseTracking(True)

    def filas(self):
        return max(2, int(self.height() // 42))

    def inicio(self):
        n = self.filas(); c = self.t_vista + self.w_vista / 2
        r0 = int(c // 60) - n // 2
        return max(0, min(r0, int(self.ex.dur // 60) - n + 1)) * 60.0

    def paintEvent(self, ev):
        p = QPainter(self); pal = PAPEL[self.oscuro]; W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor(pal['papel']))
        ex = self.ex
        if not ex:
            return
        n = self.filas(); rh = H / n; s0 = self.inicio(); pxs = (W - 54) / 60.0
        f = QFont(self.font()); f.setPointSizeF(7.5); p.setFont(f)
        c = ex.chans[min(self.canal, len(ex.chans) - 1)]
        sc = np.nanpercentile(np.abs(senal(ex, c, int(self.t_vista * ex.fs), int((self.t_vista + 10) * ex.fs))), 99) if ex.n else 1
        pxmv = 0.42 * rh / max(0.2, sc)
        for r in range(n):
            ta = s0 + r * 60
            if ta >= ex.dur:
                break
            y0 = r * rh; mid = y0 + rh * 0.55
            for t in self.tiras:
                a, b = max(t['t0'], ta), min(t['t0'] + t['dur'], ta + 60)
                if b > a:
                    p.fillRect(QRectF(54 + (a - ta) * pxs, y0, (b - a) * pxs, rh), QColor(37, 99, 235, 40))
            a, b = max(self.t_vista, ta), min(self.t_vista + self.w_vista, ta + 60)
            if b > a:
                p.setPen(QPen(QColor('#ec4899'), 1.5)); p.setBrush(QColor(236, 72, 153, 35))
                p.drawRect(QRectF(54 + (a - ta) * pxs, y0 + 1, (b - a) * pxs, rh - 2)); p.setBrush(Qt.NoBrush)
            y = senal(ex, c, int(ta * ex.fs), int((ta + 60) * ex.fs))
            p.setPen(QPen(QColor(pal['trazo']), 0.8))
            p.drawPolyline(linea(y, ex.fs, ta, ta, 54, pxs, mid, pxmv, y0, y0 + rh))
            b0 = int(np.searchsorted(ex.t, ta)); b1 = int(np.searchsorted(ex.t, ta + 60))
            for bb in range(b0, b1):
                k = int(ex.cls[bb])
                if k:
                    p.fillRect(QRectF(54 + (ex.t[bb] - ta) * pxs - 1, y0 + 1, 2.5, 5), QColor(COLOR_CLASE[k]))
            for e in ex.eventos:
                ty = ALIAS_BARRA.get(e['type'], e['type'])
                if ty in COLOR_BARRA and e['t1'] >= ta and e['t0'] <= ta + 60:
                    a2, b2 = max(e['t0'], ta), min(max(e['t1'], e['t0'] + 0.5), ta + 60)
                    p.fillRect(QRectF(54 + (a2 - ta) * pxs, y0 + rh - 4, max(2, (b2 - a2) * pxs), 3), COLOR_BARRA[ty])
            p.setPen(QColor(pal['texto2'])); p.drawText(QPointF(3, mid + 4), ex.reloj(ta, False))
            hrs = ex.A.get('minute', {}).get('hr', [])
            mi = int(ta // 60)
            if mi < len(hrs) and hrs[mi]:
                p.drawText(QPointF(3, mid + 15), '%d lpm' % round(hrs[mi]))
        p.end()

    def _t(self, x, y):
        n = self.filas(); rh = self.height() / n
        return self.inicio() + int(y // rh) * 60 + max(0, (x - 54)) / ((self.width() - 54) / 60.0)

    def mousePressEvent(self, e):
        if self.ex:
            self.centrar.emit(float(min(self.ex.dur, self._t(e.position().x(), e.position().y()))))


# =========================================================================== panel completo
class PanelECG(QWidget):
    """Tira con barras de herramientas, sucesos, barra de tiempo y contexto."""
    herramienta_cambiada = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None
        self.tira = Tira(self)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(2)
        # fila 1: hora, herramientas, canales, formato
        bar = QHBoxLayout(); bar.setContentsMargins(6, 4, 6, 0)
        self.lbl = QLabel('—'); self.lbl.setObjectName('reloj')
        self.fc = QLabel(''); self.fc.setObjectName('reloj')
        bar.addWidget(self.lbl); bar.addSpacing(8); bar.addWidget(self.fc); bar.addStretch(1)
        self.grupo_h = QButtonGroup(self); self.grupo_h.setExclusive(True); self.bot_h = {}
        for k, txt, tip in (('latidos', 'Latidos', 'Elegir y etiquetar latidos (arrastrar elige varios)'), ('calibrador', 'Calibrador', 'Medir tiempo y amplitud'),
                            ('tira', 'Tira', 'Elegir tiras para el informe'), ('suceso', 'Suceso', 'Agregar, borrar o editar sucesos')):
            b = QToolButton(); b.setText(txt); b.setCheckable(True); b.setToolTip(tip); b.setChecked(k == 'latidos')
            b.clicked.connect(lambda _=False, k=k: self.set_herramienta(k)); self.grupo_h.addButton(b); bar.addWidget(b); self.bot_h[k] = b
        bar.addSpacing(10); bar.addWidget(QLabel('Canales'))
        self.chk_c = []
        for i in range(3):
            c = QCheckBox(str(i + 1)); c.setChecked(True); c.toggled.connect(self._opc); self.chk_c.append(c); bar.addWidget(c)
        bar.addSpacing(10)
        self.dur = QComboBox(); [self.dur.addItem(a, b) for a, b in DURACIONES]
        self.gan = QComboBox(); [self.gan.addItem(a, b) for a, b in GANANCIAS]; self.gan.setCurrentIndex(1)
        self.fil = QCheckBox('Filtro'); self.fil.setChecked(True); self.fil.setToolTip('Filtro de 0,5 a 40 Hz para la pantalla')
        self.ref = QCheckBox('Referencia'); self.ref.setVisible(False)
        for w in (QLabel('Página'), self.dur, QLabel('Ganancia'), self.gan, self.fil, self.ref):
            bar.addWidget(w)
        lay.addLayout(bar)
        # fila 2: navegación y casillas de sucesos
        bar2 = QHBoxLayout(); bar2.setContentsMargins(6, 0, 6, 0)
        for txt, tip, fn in (('|◀', 'Ir al inicio', lambda: self.tira.ir_a(0, centrar=False)), ('−1 h', 'Una hora atrás', lambda: self.mover(-3600)),
                             ('◀', 'Página anterior', lambda: self.tira.pagina(-1)), ('▶', 'Página siguiente', lambda: self.tira.pagina(1)),
                             ('+1 h', 'Una hora adelante', lambda: self.mover(3600)), ('▶|', 'Ir al final', lambda: self.tira.ir_a(self.ex.dur if self.ex else 0, centrar=False))):
            b = QToolButton(); b.setText(txt); b.setToolTip(tip); b.clicked.connect(fn); bar2.addWidget(b)
        bar2.addSpacing(8)
        for nombre, k in (('◀ V', 2), ('V ▶', 2), ('◀ S', 1), ('S ▶', 1)):
            b = QToolButton(); b.setText(nombre); b.setToolTip('Latido %s %s' % (CLS_NAME[k].lower(), 'anterior' if nombre.startswith('◀') else 'siguiente'))
            b.clicked.connect(lambda _=False, k=k, d=(-1 if nombre.startswith('◀') else 1): self.saltar_clase(k, d)); bar2.addWidget(b)
        bar2.addSpacing(10)
        self.chk_ev = {}
        for k, n, col, _ in SUCESOS_BARRA:
            c = QCheckBox(n); c.setChecked(True); c.setVisible(False)
            c.setStyleSheet(f'QCheckBox {{ color: {col}; font-weight: 600; }}')
            c.toggled.connect(self._barras); self.chk_ev[k] = c; bar2.addWidget(c)
        bar2.addStretch(1)
        self.nav_lbl = QLabel(''); self.nav_lbl.setObjectName('reloj'); bar2.addWidget(self.nav_lbl)
        lay.addLayout(bar2)
        lay.addWidget(self.tira, 1)
        self.barra = BarraTiempo(self); lay.addWidget(self.barra)
        self.contexto = VistaContexto(self); self.contexto.setVisible(False); self.contexto.setFixedHeight(170); lay.addWidget(self.contexto)
        self.ayuda = QLabel('')
        self.ayuda.setObjectName('ayuda'); lay.addWidget(self.ayuda)
        self.dur.currentIndexChanged.connect(self._opc); self.gan.currentIndexChanged.connect(self._opc)
        self.fil.toggled.connect(self._opc); self.ref.toggled.connect(self._opc)
        self.barra.ir.connect(lambda t: self.tira.ir_a(t))
        self.contexto.centrar.connect(lambda t: self.tira.ir_a(t))
        self.tira.tiempo_cambiado.connect(self._sync)
        self.tira.vista_cambiada.connect(lambda: self._sync(self.tira.t0))
        self.set_herramienta('latidos')

    AYUDAS = {
        'latidos': 'Clic: elegir · arrastrar, Mayús+clic, Ctrl+clic: varios · N S B T V R I E C P D F U A: etiquetar · Mayús+letra: insertar · Supr: borrar · doble clic: plantilla · clic derecho: menú · Tab: suceso siguiente',
        'calibrador': 'Arrastrar: medir tiempo y frecuencia · arrastrar los extremos para ajustar · clic derecho: marcha y calibrador de amplitud · Ctrl/Mayús+flechas: mover 1/10 píxeles',
        'tira': 'El marco rojo es la tira: clic para agregarla al informe · clic derecho: ampliar, encoger o tira de página',
        'suceso': 'Arrastrar: agregar un suceso (artefacto, FA o del usuario) · clic derecho sobre su barra: borrar o editar · Mayús+clic: final de un suceso largo',
    }

    def set_herramienta(self, k):
        self.tira.herramienta = k; self.bot_h[k].setChecked(True)
        self.tira.setCursor(Qt.CrossCursor if k in ('calibrador', 'suceso') else Qt.PointingHandCursor if k == 'tira' else Qt.ArrowCursor)
        self.ayuda.setText(self.AYUDAS[k]); self.tira.update(); self.herramienta_cambiada.emit(k)

    def set_examen(self, ex):
        self.ex = ex; self.tira.set_examen(ex); self.barra.ex = ex; self.contexto.ex = ex
        self.ref.setVisible(ex.ref is not None)
        for i, c in enumerate(self.chk_c):
            c.setVisible(i < len(ex.chans)); c.setText(ex.names[ex.chans[i]] if i < len(ex.chans) and ex.chans[i] < len(ex.names) else str(i + 1))
        self._sync(0)

    def _opc(self):
        t = self.tira
        t.modo = self.dur.currentData(); t.gan = self.gan.currentData(); t.filtro = self.fil.isChecked(); t.ver_ref = self.ref.isChecked()
        vis = [i for i, c in enumerate(self.chk_c) if c.isChecked() and c.isVisible()]
        t.visibles = vis if self.ex and len(vis) < len(self.ex.chans) else None
        t.update(); self._sync(t.t0)

    def _barras(self):
        self.tira.barras_on = {k for k, c in self.chk_ev.items() if c.isChecked()}; self.tira.update()

    def mover(self, d):
        self.tira.ir_a(self.tira.t0 + d, centrar=False)

    def _sync(self, t0, mover=True):
        if not self.ex:
            return
        w = self.tira.ancho_seg()
        self.lbl.setText('%s – %s' % (self.ex.reloj(t0), self.ex.reloj(min(self.ex.dur, t0 + w))))
        hrs = self.ex.A.get('minute', {}).get('hr', []); mi = int((t0 + w / 2) // 60)
        self.fc.setText(('FC %d lpm' % round(hrs[mi])) if mi < len(hrs) and hrs[mi] else '')
        self.barra.set_vista(t0, w)
        self.contexto.t_vista, self.contexto.w_vista = t0, w; self.contexto.oscuro = self.tira.oscuro; self.contexto.update()
        presentes = self.tira.tipos_en_vista()
        for k, c in self.chk_ev.items():
            c.setVisible(k in presentes)

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
