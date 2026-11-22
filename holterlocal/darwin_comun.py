"""Piezas comunes de los módulos que siguen el manual de medilog Darwin2: día y noche según el periodo de sueño,
orden de gravedad de las arritmias, tipos derivados (TV sostenida y no sostenida) y una tira de ECG pequeña."""
import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from .estilo import SUCESOS_BARRA
from .motor import EVENT_LABEL
from .tira import dibujar_ecg

SUENO_DEF = {'luces': '22:30', 'dormir': '23:00', 'despierto': '07:00'}


def hm(txt, d=0):
    try:
        h, m = str(txt).split(':')[:2]
        return int(h) * 3600 + int(m) * 60
    except (ValueError, AttributeError):
        return d


def sueno(ex):
    ses = ex.sesion.d if getattr(ex, 'sesion', None) else {}
    s = dict(SUENO_DEF); s.update(ses.get('sueno') or {})
    return s


def es_noche(ex, t):
    """Verdadero para los tiempos (s desde el inicio) que caen en el periodo de sueño del diario (por defecto 23:00 a 07:00)."""
    s = sueno(ex); a = hm(s['dormir'], 23 * 3600); b = hm(s['despierto'], 7 * 3600)
    tod = (ex.seg_inicio() + np.asarray(t, float)) % 86400
    return ((tod >= a) | (tod < b)) if a > b else ((tod >= a) & (tod < b))


# orden de gravedad (de más grave a menos), como la barra lateral del resumen de arritmias de Darwin
GRAVEDAD = ['vt_s', 'vt_ns', 'ha_no_organized', 'ha_asystole', 'ha_av_block_complete', 'ha_av_block_advanced', 'ha_av_block_mobitz2',
            'ha_av_block_2to1', 'pause', 'ha_av_block_mobitz1', 'aivr', 'ivr', 'af', 'ha_flutter', 'svt', 'qtLong', 'rOnT', 'vPair', 'vBig', 'vTrig',
            'stDep', 'stEle', 'sRun', 'sPair', 'sBig', 'sTrig', 'brady', 'tachy', 'ha_av_block_1', 'vIso', 'sIso', 'interp', 'fus',
            'ha_pacing_failure', 'ha_pacing_sensing', 'ha_pacing_fusion', 'u1', 'u2', 'u3', 'doubt']
NOMBRES = dict(EVENT_LABEL)
NOMBRES.update({'vt_s': 'TV sostenida (30 s o más)', 'vt_ns': 'TV no sostenida (menos de 30 s)', 'qtLong': 'QT largo',
                'artifact': 'Artefacto', 'nosig': 'Sin señal'})
COLORES = {k: c for k, _, c, _ in SUCESOS_BARRA}
COLORES.update({'vt_s': '#7c3aed', 'vt_ns': '#a78bfa', 'aivr': '#c4b5fd', 'ivr': '#ddd6fe', 'vPair': '#ef4444', 'vIso': '#fca5a5',
                'sRun': '#22c55e', 'sPair': '#3b82f6', 'sIso': '#93c5fd', 'rOnT': '#e8735e', 'interp': '#d97706', 'fus': '#92400e',
                'qtLong': '#9333ea', 'ha_flutter': '#14b8a6', 'doubt': '#9ca3af'})
for _k in GRAVEDAD:
    if _k.startswith('ha_av'):
        COLORES.setdefault(_k, '#be185d')
    COLORES.setdefault(_k, '#64748b')
NO_ARRITMIA = ('diary', 'artifact', 'nosig')


def rango(k):
    return GRAVEDAD.index(k) if k in GRAVEDAD else len(GRAVEDAD)


def eventos_darwin(ex, extra=None):
    """Sucesos del examen con la TV separada en sostenida y no sostenida (criterio de Darwin: 30 s)."""
    out = []
    for e in ex.eventos:
        ty = e['type']
        if ty in NO_ARRITMIA:
            continue
        if ty == 'vt':
            e = dict(e); e['type'] = 'vt_s' if (e.get('dur') or (e['t1'] - e['t0'])) >= 30 else 'vt_ns'
            e['label'] = NOMBRES[e['type']]
        out.append(e)
    out += list(extra or [])
    out.sort(key=lambda e: e['t0'])
    return out


def nombre(ex, k, e=None):
    if k.startswith('u') and len(k) == 2 and getattr(ex, 'sesion', None):
        return ex.sesion.etiqueta_usuario(int(k[1]))
    if k in NOMBRES:
        return NOMBRES[k]
    return (e or {}).get('label') or k


def fmt(v, d=0):
    if v is None or (isinstance(v, float) and v != v):
        return '—'
    try:
        s = f'{v:,.{d}f}'
    except (TypeError, ValueError):
        return str(v)
    return s.replace(',', 'X').replace('.', ',').replace('X', '.')


def fc_evento(ex, e):
    """FC mínima, media y máxima de los latidos de un suceso (lpm)."""
    b0, b1 = np.searchsorted(ex.t, [e['t0'], e['t1'] + 0.01])
    rr = ex.rr[b0:b1 + 1]; rr = rr[np.isfinite(rr) & (rr > 0.2)]
    if not len(rr):
        return None, None, None
    f = 60.0 / rr
    return float(f.min()), float(60.0 / rr.mean()), float(f.max())


class MiniTira(QWidget):
    """Tira de ECG pequeña de solo lectura (directorio de bandas, ruido, QT)."""
    clic = Signal(float)

    def __init__(self, parent=None, alto=150):
        super().__init__(parent)
        self.ex = None; self.t0 = 0.0; self.dur = 10.0; self.chans = None; self.marca = None; self.titulo = ''; self.gan = 10.0
        self.etiquetas = True; self.alto = alto
        self.setMinimumHeight(alto); self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def sizeHint(self):
        return QSize(600, self.alto)

    def mostrar(self, ex, t0, dur, chans=None, marca=None, titulo=''):
        self.ex = ex; self.t0 = max(0.0, float(t0)); self.dur = float(dur); self.chans = chans; self.marca = marca; self.titulo = titulo
        self.update()

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(0, 0, self.width(), self.height())
        if not self.ex:
            p.fillRect(r, QColor('#ffffff')); p.end(); return
        chs = self.chans if self.chans is not None else list(self.ex.chans)
        dibujar_ecg(p, r, self.ex, self.t0, self.dur, chs, mm_mv=self.gan, etiquetas=self.etiquetas)
        if self.marca:
            a, b = self.marca; px = self.width() / self.dur
            xa = (a - self.t0) * px; xb = max(xa + 2, (b - self.t0) * px)
            p.fillRect(QRectF(xa, 0, xb - xa, self.height()), QColor(255, 165, 0, 45))
            p.setPen(QPen(QColor('#f97316'), 1.5)); p.drawRect(QRectF(xa, 0.75, xb - xa, self.height() - 1.5))
        if self.titulo:
            f = QFont(); f.setPointSizeF(8); f.setBold(True); p.setFont(f)
            w = p.fontMetrics().horizontalAdvance(self.titulo) + 10
            p.fillRect(QRectF(self.width() - w - 4, 2, w, 15), QColor(255, 255, 255, 215))
            p.setPen(QColor('#1f2933')); p.drawText(QRectF(self.width() - w - 4, 2, w, 15), Qt.AlignCenter, self.titulo)
        p.end()

    def mousePressEvent(self, e):
        if self.ex:
            self.clic.emit(self.t0 + e.position().x() / max(1, self.width()) * self.dur)
