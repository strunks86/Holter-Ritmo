"""Pestañas de la aplicación: eventos, plantillas, tendencias, histogramas, criterios y validación."""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem, QMenu, QPushButton, QRadioButton,
                               QSpinBox, QSplitter, QTextBrowser, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .estilo import COLOR_CLASE, RESUMEN_CSS
from .motor import CLS_NAME, CLS_SYM, EVENT_LABEL, EVENT_NAMES, GRUPO_CLASE, ORDEN_ETIQUETAS

pg.setConfigOptions(antialias=True, background='w', foreground='#1f2933')


def fmt(v, d=0):
    if v is None or (isinstance(v, float) and v != v):
        return '—'
    try:
        s = f'{v:,.{d}f}'
    except (TypeError, ValueError):
        return str(v)
    return s.replace(',', 'X').replace('.', ',').replace('X', '.')


def dur_txt(s):
    s = int(round(s or 0)); h, r = divmod(s, 3600); m, s = divmod(r, 60)
    return f'{h} h {m:02d} min' if h else (f'{m} min {s:02d} s' if m else f'{s} s')


# --------------------------------------------------------------------------- Eventos
class VistaEventos(QWidget):
    ir = Signal(float, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        self.arbol = QTreeWidget(); self.arbol.setHeaderLabels(['Evento', 'Hora', 'Latidos', 'FC (lpm)', 'Duración', 'Detalle'])
        self.arbol.setAlternatingRowColors(True); self.arbol.setUniformRowHeights(True)
        self.arbol.header().setSectionResizeMode(5, QHeaderView.Stretch); self.arbol.setColumnWidth(0, 300)
        lay.addWidget(self.arbol)
        self.arbol.itemClicked.connect(self._clic)
        self.arbol.currentItemChanged.connect(lambda it, _: self._clic(it))

    def mostrar(self, ex):
        self.arbol.clear()
        by = {}
        for e in ex.eventos:
            by.setdefault(e['type'], []).append(e)
        extra = []
        for k, evs in by.items():
            if k not in EVENT_LABEL:
                extra.append((k, evs[0].get('label', k)))
        for k, nombre in EVENT_NAMES[:-2] + sorted(extra, key=lambda x: x[1]) + EVENT_NAMES[-2:]:
            evs = by.get(k)
            if not evs:
                continue
            top = QTreeWidgetItem([f'{nombre}  ({len(evs)})'])
            f = top.font(0); f.setBold(True); top.setFont(0, f)
            hijos = []
            for e in evs[:5000]:
                it = QTreeWidgetItem([nombre, ex.reloj(e['t0']), fmt(e.get('n')) if e.get('n') else '',
                                      fmt(e.get('rate')) if e.get('rate') else '', dur_txt(e.get('dur') or (e['t1'] - e['t0'])), e.get('detail', '')])
                if e.get('severity') == 'critical':
                    it.setForeground(0, QColor('#c81e1e'))
                it.setData(0, Qt.UserRole, (float(e['t0']), int(e.get('b0', -1)) if e.get('n') and e.get('b0', -1) >= 0 else -1))
                hijos.append(it)
            top.addChildren(hijos)
            self.arbol.addTopLevelItem(top)

    def _clic(self, it):
        if it is None:
            return
        d = it.data(0, Qt.UserRole)
        if d:
            self.ir.emit(d[0], d[1])


# --------------------------------------------------------------------------- Plantillas
def icono_plantilla(T, nch, M, k, sz=QSize(150, 110), sel=False):
    pm = QPixmap(sz); pm.fill(QColor('#fff8d6' if sel else '#ffffff'))
    p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor('#f0d4d4'), 1))
    for x in range(0, sz.width(), 10):
        p.drawLine(x, 0, x, sz.height())
    w = np.asarray(T['wave'], float)
    h = sz.height() / max(1, nch)
    col = QColor(COLOR_CLASE[k])
    for c in range(nch):
        y = w[c * M:(c + 1) * M]
        if not len(y):
            continue
        rng = max(1e-6, np.nanmax(np.abs(y)))
        xs = np.linspace(6, sz.width() - 6, len(y)); ys = h * (c + 0.55) - y / rng * h * 0.42
        p.setPen(QPen(col, 1.6)); p.drawPolyline(QPolygonF([QPointF(a, b) for a, b in zip(xs, ys)]))
    p.end()
    return QIcon(pm)


class ListaPlantillas(QListWidget):
    tecla = Signal(int)
    tab = Signal(int)

    def keyPressEvent(self, e):
        from .tira import TECLAS
        if e.key() in TECLAS and not (e.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.ShiftModifier)):
            self.tecla.emit(TECLAS[e.key()]); return
        super().keyPressEvent(e)

    def event(self, e):
        if e.type() == e.Type.KeyPress and e.key() in (Qt.Key_Tab, Qt.Key_Backtab):
            self.tab.emit(-1 if (e.key() == Qt.Key_Backtab or e.modifiers() & Qt.ShiftModifier) else 1); return True
        return super().event(e)


GRUPOS_PLANTILLA = [('*', 'Todas'), ('N', 'Normal'), ('S', 'Supraventricular'), ('V', 'Ventricular'), ('P', 'Estimulado'),
                    ('U', 'Desconocido'), ('X', 'Artefacto')]


class VistaPlantillas(QWidget):
    ir = Signal(float, int)
    reetiquetar_plantillas = Signal(list, int)
    combinar = Signal(list)
    borrar_plantillas = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.pos = {}; self.grupo = '*'; self.orden = []
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        top = QHBoxLayout(); self.bg = QButtonGroup(self); self.radios = {}
        for k, n in GRUPOS_PLANTILLA:
            r = QRadioButton(n); r.setChecked(k == '*'); r.toggled.connect(lambda on, k=k: on and self._grupo(k)); self.bg.addButton(r); top.addWidget(r); self.radios[k] = r
        self.info = QLabel('Elige una morfología.'); self.info.setWordWrap(True)
        top.addSpacing(10); top.addWidget(self.info, 1)
        lay.addLayout(top)
        self.lista = ListaPlantillas(); self.lista.setViewMode(QListWidget.IconMode); self.lista.setIconSize(QSize(150, 110))
        self.lista.setResizeMode(QListWidget.Adjust); self.lista.setMovement(QListWidget.Static); self.lista.setSpacing(6)
        self.lista.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.lista.setContextMenuPolicy(Qt.CustomContextMenu); self.lista.customContextMenuRequested.connect(self._menu)
        lay.addWidget(self.lista, 1)
        bot = QHBoxLayout()
        self.bprev = QPushButton('◀ Latido anterior'); self.bnext = QPushButton('Latido siguiente ▶')
        bot.addWidget(self.bprev); bot.addWidget(self.bnext); bot.addSpacing(16)
        bot.addWidget(QLabel('Letra = etiquetar las elegidas · Ctrl/Mayús+clic o arrastrar = varias · clic derecho = combinar, borrar, artefacto · Tab = latido siguiente'))
        bot.addStretch(1)
        lay.addLayout(bot)
        self.lista.currentItemChanged.connect(lambda it, _: self._sel(it))
        self.lista.tecla.connect(self._rel); self.lista.tab.connect(self._nav)
        self.bprev.clicked.connect(lambda: self._nav(-1)); self.bnext.clicked.connect(lambda: self._nav(1))

    def _grupo(self, k):
        self.grupo = k
        if self.ex:
            self.mostrar(self.ex)

    def dominante(self, i):
        tc = self.ex.A['tcls'][i] if i < len(self.ex.A.get('tcls', [])) else None
        return int(np.argmax(tc)) if tc and sum(tc) else 0

    def grupo_de(self, k):
        g = GRUPO_CLASE.get(k, 'U')
        if g == 'S' and not (self.ex.A.get('settings') or {}).get('svTemplates', True):
            g = 'N'
        return g

    def mostrar(self, ex):
        self.ex = ex
        self.radios['S'].setVisible((ex.A.get('settings') or {}).get('svTemplates', True))
        cur = self.lista.currentItem().data(Qt.UserRole) if self.lista.currentItem() else None
        self.lista.blockSignals(True); self.lista.clear()
        T = ex.A.get('templates') or []; nch = ex.A['nch']; M = ex.A.get('morphM', {}).get('M', 56)
        tcls = ex.A.get('tcls') or []
        cuenta = [int(sum(tcls[i])) if i < len(tcls) else T[i]['count'] for i in range(len(T))]
        self.orden = []
        for i in sorted(range(len(T)), key=lambda i: -cuenta[i]):
            k = self.dominante(i); n = cuenta[i]
            if n == 0 or (self.grupo != '*' and self.grupo_de(k) != self.grupo):
                continue
            self.orden.append(i)
            it = QListWidgetItem(icono_plantilla(T[i], nch, M, k), f'#{i + 1} · {CLS_SYM[k]} · {fmt(n)}')
            it.setData(Qt.UserRole, i); it.setToolTip(f'{CLS_NAME[k]} · {fmt(n)} latidos · ancho {fmt(T[i].get("width"))} ms')
            it.setForeground(QColor(COLOR_CLASE[k]))
            self.lista.addItem(it)
            if cur == i:
                self.lista.setCurrentItem(it)
        self.lista.blockSignals(False)

    def elegidas(self):
        return [it.data(Qt.UserRole) for it in self.lista.selectedItems()] or ([self.lista.currentItem().data(Qt.UserRole)] if self.lista.currentItem() else [])

    def elegir(self, tid):
        for j in range(self.lista.count()):
            if self.lista.item(j).data(Qt.UserRole) == tid:
                self.lista.setCurrentRow(j); return True
        if self.grupo != '*':
            self.radios['*'].setChecked(True); return self.elegir(tid)
        return False

    def _sel(self, it):
        if not it or not self.ex:
            return
        i = it.data(Qt.UserRole); idx = np.nonzero(self.ex.tid == i)[0]
        T = self.ex.A['templates'][i]; k = self.dominante(i)
        self.info.setText(f'<b>Morfología {i + 1}</b>: {fmt(len(idx))} latidos ({fmt(100 * len(idx) / max(1, len(self.ex.t)), 2)} %), '
                          f'clase {CLS_NAME[k].lower()}, ancho {fmt(T.get("width"))} ms, prob. ventricular {fmt(T.get("pV"), 2)}, artefacto {fmt(T.get("pX"), 2)}')
        self.pos[i] = 0
        if len(idx):
            self.ir.emit(float(self.ex.t[idx[0]]), int(idx[0]))

    def _nav(self, d):
        it = self.lista.currentItem()
        if not it:
            return
        i = it.data(Qt.UserRole); idx = np.nonzero(self.ex.tid == i)[0]
        if not len(idx):
            return
        self.pos[i] = int(np.clip(self.pos.get(i, 0) + d, 0, len(idx) - 1))
        b = int(idx[self.pos[i]]); self.ir.emit(float(self.ex.t[b]), b)
        self.info.setText(f'<b>Morfología {i + 1}</b>: latido {self.pos[i] + 1} de {fmt(len(idx))}')

    def _rel(self, k):
        tids = self.elegidas()
        if tids:
            self.reetiquetar_plantillas.emit(tids, k)

    def _menu(self, pos):
        tids = self.elegidas()
        if not tids:
            return
        m = QMenu(self)
        sub = m.addMenu('Etiquetar %d plantilla(s) como' % len(tids))
        for k in ORDEN_ETIQUETAS:
            sub.addAction(f'{CLS_NAME[k]}\t{CLS_SYM[k]}', lambda k=k: self.reetiquetar_plantillas.emit(tids, k))
        m.addAction('Artefacto en todos sus latidos (se excluyen de los cálculos)\tA', lambda: self.reetiquetar_plantillas.emit(tids, 4))
        cur = self.lista.currentItem().data(Qt.UserRole) if self.lista.currentItem() else tids[0]
        if cur in self.orden:
            sig = self.orden[self.orden.index(cur):]
            m.addAction('Reetiquetar esta y todas las siguientes (%d) como artefacto' % len(sig), lambda: self.reetiquetar_plantillas.emit(sig, 4))
        if len(tids) > 1:
            m.addAction('Combinar plantillas', lambda: self.combinar.emit(tids))
        m.addSeparator()
        m.addAction('Eliminar todos los latidos de la(s) plantilla(s)', lambda: self.borrar_plantillas.emit(tids))
        m.exec(self.lista.viewport().mapToGlobal(pos))


# --------------------------------------------------------------------------- Tendencias
class EjeReloj(pg.AxisItem):
    def __init__(self, ex_fn, **kw):
        super().__init__(orientation='bottom', **kw); self.ex_fn = ex_fn

    def tickStrings(self, values, scale, spacing):
        ex = self.ex_fn()
        return [ex.reloj(v * 60, False) if ex else '' for v in values]


def _cinco(arr, f=np.nanmean):
    n = int(np.ceil(len(arr) / 5)); out = np.full(n, np.nan)
    for i in range(n):
        s = arr[i * 5:(i + 1) * 5]
        if len(s) and np.isfinite(s).any():
            out[i] = f(s)
    return out


def series_tendencia(ex):
    """Series de 5 minutos (en minutos desde el inicio)."""
    M = ex.A.get('minute') or {}
    arr = lambda k: np.array([np.nan if v is None else v for v in M.get(k, [])], float)
    hr = arr('hr'); n = int(np.ceil(len(hr) / 5))
    S = {'hr': _cinco(hr), 'hrmin': _cinco(arr('hrMin'), np.nanmin), 'hrmax': _cinco(arr('hrMax'), np.nanmax)}
    S['rr'] = 60000.0 / S['hr']
    S['v'] = _cinco(arr('v'), np.nansum); S['s'] = _cinco(arr('s'), np.nansum)
    S['vmin'] = S['v'] / 5; S['smin'] = S['s'] / 5
    S['af'] = _cinco(arr('af')) * 100
    S['rmssd'] = _cinco(arr('rmssd')); S['sdnn'] = _cinco(arr('sdnn'))
    S['qtc'] = _cinco(np.array([np.nan if v is None else v for v in (ex.A.get('qtc') or [])], float))
    S['qt'] = _cinco(arr('qt'))
    S['st'] = [_cinco(np.array([np.nan if v is None else v for v in s], float)) for s in (M.get('st') or [])]
    idx5 = (ex.t // 300).astype(int) if len(ex.t) else np.zeros(0, int)

    def latidos_en(tipos):
        out = np.zeros(n)
        for e in ex.eventos:
            if e['type'] in tipos and e.get('b0', -1) >= 0:
                b0, b1 = int(e['b0']), int(e.get('b1', e['b0']))
                ii = idx5[b0:b1 + 1]; ii = ii[ii < n]
                np.add.at(out, ii, 1)
        return out

    def eventos_en(tipos):
        out = np.zeros(n)
        for e in ex.eventos:
            if e['type'] in tipos:
                i = int(e['t0'] // 300)
                if i < n:
                    out[i] += 1
        return out
    S['vpar'] = eventos_en(['vPair']); S['vrun'] = eventos_en(['vt', 'aivr', 'ivr']); S['spar'] = eventos_en(['sPair']); S['srun'] = eventos_en(['svt', 'sRun'])
    for k, ts in (('brady', ['brady']), ('tachy', ['tachy']), ('vbig', ['vBig']), ('vtrig', ['vTrig']), ('sbig', ['sBig']), ('strig', ['sTrig']),
                  ('u1', ['u1']), ('u2', ['u2']), ('u3', ['u3'])):
        S[k] = latidos_en(ts)
    return S


GRUPOS_TENDENCIA = {
    'Ritmo': [('v', 'V (cuenta)', 'bar', '#dc2626'), ('vmin', 'V por minuto', 'bar', '#ef4444'), ('s', 'S (cuenta)', 'bar', '#1d4ed8'),
              ('smin', 'S por minuto', 'bar', '#3b82f6'), ('vpar', 'Pares V', 'bar', '#b91c1c'), ('vrun', 'Rachas V', 'bar', '#7f1d1d'),
              ('spar', 'Pares S', 'bar', '#1e40af'), ('srun', 'Rachas S', 'bar', '#1e3a8a'), ('af', 'FA (%)', 'bar', '#0d9488'),
              ('rr', 'RR (ms)', 'line', '#374151'), ('hr', 'FC (lpm)', 'band', '#1f5fbf')],
    'QT': [('qt', 'QT (ms)', 'dots', '#374151'), ('qtc', 'QTc (ms)', 'dots', '#7c3aed'), ('hr', 'FC (lpm)', 'band', '#1f5fbf'), ('rr', 'RR (ms)', 'line', '#374151')],
    'ST': [('st', 'ST (µV)', 'st', ''), ('hr', 'FC (lpm)', 'band', '#1f5fbf'), ('rr', 'RR (ms)', 'line', '#374151')],
    'Otros': [('brady', 'Latidos en bradicardia', 'bar', '#16a34a'), ('tachy', 'Latidos en taquicardia', 'bar', '#a16207'),
              ('vbig', 'Bigeminismo V (latidos)', 'bar', '#b07a7a'), ('vtrig', 'Trigeminismo V (latidos)', 'bar', '#fdba74'),
              ('sbig', 'Bigeminismo S (latidos)', 'bar', '#06b6d4'), ('strig', 'Trigeminismo S (latidos)', 'bar', '#808000'),
              ('u1', 'Usuario 1', 'bar', '#b08968'), ('u2', 'Usuario 2', 'bar', '#c2410c'), ('u3', 'Usuario 3', 'bar', '#fb7185'),
              ('hr', 'FC (lpm)', 'band', '#1f5fbf'), ('rr', 'RR (ms)', 'line', '#374151')],
    'VFC': [('rmssd', 'RMSSD (ms)', 'line', '#0f766e'), ('sdnn', 'SDNN (ms)', 'line', '#7c3aed'), ('hr', 'FC (lpm)', 'band', '#1f5fbf'), ('rr', 'RR (ms)', 'line', '#374151')],
}


class VistaTendencias(QWidget):
    ir = Signal(float, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.grupo = 'Ritmo'; self.cursor_min = 0.0; self.plots = []; self.lineas = []
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        top = QHBoxLayout(); self.bg = QButtonGroup(self)
        for g in GRUPOS_TENDENCIA:
            r = QRadioButton(g); r.setChecked(g == 'Ritmo'); r.toggled.connect(lambda on, g=g: on and self._grupo(g)); self.bg.addButton(r); top.addWidget(r)
        top.addSpacing(16); top.addWidget(QLabel('Resolución'))
        self.res = QComboBox(); [self.res.addItem(f'×{i}', i) for i in (1, 2, 3, 4)]; self.res.currentIndexChanged.connect(self._zoom); top.addWidget(self.res)
        top.addStretch(1); lay.addLayout(top)
        cuerpo = QHBoxLayout()
        self.g = pg.GraphicsLayoutWidget(); cuerpo.addWidget(self.g, 1)
        self.valores = QLabel(''); self.valores.setMinimumWidth(190); self.valores.setAlignment(Qt.AlignTop); self.valores.setTextFormat(Qt.RichText)
        cuerpo.addWidget(self.valores)
        lay.addLayout(cuerpo, 1)

    def _grupo(self, g):
        self.grupo = g
        if self.ex:
            self.mostrar(self.ex)

    def mostrar(self, ex):
        self.ex = ex; self.g.clear(); self.plots = []; self.lineas = []
        self.S = S = series_tendencia(ex)
        n = len(S['hr'])
        if not n:
            return
        x = np.arange(n) * 5 + 2.5
        for k, tit, tipo, col in GRUPOS_TENDENCIA[self.grupo]:
            if tipo == 'st' and not S['st']:
                continue
            p = self.g.addPlot(row=len(self.plots), col=0, axisItems={'bottom': EjeReloj(lambda: self.ex)})
            p.setLabel('left', tit); p.showGrid(x=True, y=True, alpha=0.25); p.setMouseEnabled(y=False); p.getAxis('left').setWidth(70)
            if self.plots:
                p.setXLink(self.plots[0])
            if tipo == 'bar':
                p.addItem(pg.BarGraphItem(x=x, height=np.nan_to_num(S[k]), width=4.5, brush=col, pen=None))
            elif tipo == 'band':
                f1 = pg.PlotDataItem(x, S['hrmin'], pen=pg.mkPen('#9fb7e3')); f2 = pg.PlotDataItem(x, S['hrmax'], pen=pg.mkPen('#9fb7e3'))
                p.addItem(pg.FillBetweenItem(f1, f2, brush=pg.mkBrush(31, 95, 191, 50))); p.addItem(f1); p.addItem(f2)
                p.plot(x, S['hr'], pen=pg.mkPen(col, width=1.6))
            elif tipo == 'dots':
                p.plot(x, S[k], pen=None, symbol='o', symbolSize=4, symbolBrush=col)
            elif tipo == 'st':
                cols = ['#111111', '#0f766e', '#7c3aed']
                for c, s in enumerate(S['st']):
                    p.plot(x[:len(s)], s, pen=pg.mkPen(cols[c % 3], width=1.3))
            else:
                p.plot(x, S[k], pen=pg.mkPen(col, width=1.4))
            ln = pg.InfiniteLine(pos=self.cursor_min, angle=90, movable=True, pen=pg.mkPen('#dc2626', width=1.5))
            ln.sigDragged.connect(self._arrastre); ln.sigPositionChangeFinished.connect(self._soltar)
            p.addItem(ln); self.lineas.append(ln)
            p.scene().sigMouseClicked.connect(self._clic) if not self.plots else None
            self.plots.append(p)
        self._zoom(); self._valores()

    def _zoom(self, *_):
        if not self.plots or not self.ex:
            return
        z = self.res.currentData() or 1; tot = self.ex.dur / 60; w = tot / z
        a = min(max(0, self.cursor_min - w / 2), max(0, tot - w))
        self.plots[0].setXRange(a, a + w, padding=0)

    def _valores(self):
        S = getattr(self, 'S', None)
        if not S or not self.ex:
            return
        i = int(self.cursor_min // 5)
        filas = [f'<b>{self.ex.reloj(i * 300, False)} – {self.ex.reloj(i * 300 + 300, False)}</b>']
        for k, tit, tipo, _ in GRUPOS_TENDENCIA[self.grupo]:
            if tipo == 'st':
                for c, s in enumerate(S['st']):
                    v = s[i] if i < len(s) else np.nan
                    filas.append(f'ST canal {c + 1}: <b>{fmt(v)}</b> µV')
                continue
            arr = S.get(k)
            v = arr[i] if arr is not None and i < len(arr) else np.nan
            filas.append(f'{tit}: <b>{fmt(v, 1 if tipo == "line" and k != "rr" else 0)}</b>')
        self.valores.setText('<br>'.join(filas))

    def set_tiempo(self, t):
        self.cursor_min = t / 60.0
        for ln in self.lineas:
            ln.blockSignals(True); ln.setValue(self.cursor_min); ln.blockSignals(False)
        self._valores()

    def _arrastre(self, ln):
        self.cursor_min = max(0.0, ln.value())
        for o in self.lineas:
            if o is not ln:
                o.blockSignals(True); o.setValue(self.cursor_min); o.blockSignals(False)
        self._valores()

    def _soltar(self, ln):
        self.ir.emit(self.cursor_min * 60, -1)

    def _clic(self, ev):
        if not self.ex or not self.plots:
            return
        for p in self.plots:
            if p.sceneBoundingRect().contains(ev.scenePos()):
                m = p.vb.mapSceneToView(ev.scenePos()).x()
                self.set_tiempo(max(0.0, float(m) * 60)); self.ir.emit(max(0.0, float(m) * 60), -1); return


# --------------------------------------------------------------------------- Histogramas
class VistaHistogramas(QWidget):
    ir = Signal(float, int)
    nav = Signal(str, list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.modo = 'rr'; self.bins = {}; self.sel = None
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        top = QHBoxLayout(); self.bg = QButtonGroup(self)
        for k, n in (('rr', 'RR y prematuridad supraventricular'), ('rachas', 'Rachas'), ('estim', 'Estimulados'), ('lorenz', 'Diagrama de Lorenz')):
            r = QRadioButton(n); r.setChecked(k == 'rr'); r.toggled.connect(lambda on, k=k: on and self._modo(k)); self.bg.addButton(r); top.addWidget(r)
            if k == 'estim':
                self.r_estim = r
        self.info = QLabel('Clic en una columna: va al primer latido · Tab: siguiente · ← →: columna vecina')
        self.info.setObjectName('ayuda'); top.addSpacing(10); top.addWidget(self.info, 1)
        lay.addLayout(top)
        fila = QHBoxLayout(); self.g1 = pg.PlotWidget(); self.g2 = pg.PlotWidget()
        for g in (self.g1, self.g2):
            g.showGrid(x=True, y=True, alpha=0.25); g.scene().sigMouseClicked.connect(lambda ev, g=g: self._clic(g, ev)); fila.addWidget(g, 1)
        lay.addLayout(fila, 1)
        self.setFocusPolicy(Qt.StrongFocus)

    def _modo(self, k):
        self.modo = k
        if self.ex:
            self.mostrar(self.ex)

    def mostrar(self, ex):
        self.ex = ex; self.g1.clear(); self.g2.clear(); self.bins = {}
        S = ex.A.get('settings') or {}
        self.r_estim.setVisible(bool(np.isin(ex.cls, (9, 12, 13)).any()))
        ok = np.isfinite(ex.rr) & (ex.cls != 4)
        if self.modo == 'rr':
            rr = ex.rr * 1000.0
            edges = np.arange(200, 3020, 20)
            self._hist(self.g1, 'g1', 'Intervalos RR (ms)', rr, ok, edges, lambda c: c >= S.get('pauseSec', 2) * 1000 or c < 300, 'RR')
            pr = ex.prem * 100
            okp = np.isfinite(pr) & np.isin(ex.cls, (0, 1)) & (pr > 0)
            self._hist(self.g2, 'g2', 'Prematuridad supraventricular (%)', pr, okp, np.arange(0, 82, 2), lambda c: c >= S.get('svPrematurity', 0.25) * 100, 'Prematuridad')
        elif self.modo == 'rachas':
            for g, nombre, tipos, cl in ((self.g1, 'Longitud de las rachas ventriculares (latidos)', ['vPair', 'vt', 'aivr', 'ivr'], 'g1'),
                                         (self.g2, 'Longitud de las rachas supraventriculares (latidos)', ['sPair', 'svt', 'sRun'], 'g2')):
                evs = [e for e in ex.eventos if e['type'] in tipos]
                lens = np.array([e.get('n', 2) for e in evs], float); t0 = np.array([e['t0'] for e in evs], float)
                b0 = np.array([e.get('b0', -1) for e in evs], int)
                edges = np.arange(1.5, max(12, int(lens.max()) + 2) + 0.5, 1) if len(lens) else np.arange(1.5, 12.5, 1)
                g.setTitle(nombre); g.setLabel('bottom', 'latidos'); g.setLabel('left', 'rachas')
                y, _ = np.histogram(lens, bins=edges)
                g.addItem(pg.BarGraphItem(x=edges[:-1] + 0.5, height=y, width=0.8, brush='#dc2626' if cl == 'g1' else '#1d4ed8', pen=None))
                self.bins[cl] = ('rachas', edges, lens, t0, b0, nombre)
        elif self.modo == 'estim':
            m = np.isin(ex.cls, (9, 12, 13)) & ok
            rr = ex.rr * 1000.0
            self._hist(self.g1, 'g1', 'RR de latidos estimulados (ms)', rr, m, np.arange(200, 2020, 20), lambda c: False, 'RR estimulado')
            self.g2.setTitle('Sin marcas de espiga de marcapasos en este formato')
        else:
            rr = ex.rr * 1000.0; a, b = rr[:-1], rr[1:]
            good = np.isfinite(a) & np.isfinite(b); idx = np.nonzero(good)[0]
            if len(idx) > 60000:
                idx = np.sort(np.random.default_rng(0).choice(idx, 60000, replace=False))
            brushes = [pg.mkBrush(COLOR_CLASE[int(k)]) for k in ex.cls[idx + 1]]
            sp = pg.ScatterPlotItem(a[idx], b[idx], size=3, pen=None, brush=brushes, data=idx + 1)
            sp.sigClicked.connect(lambda _, pts: self.ir.emit(float(ex.t[int(pts[0].data())]), int(pts[0].data())) if len(pts) else None)
            self.g1.setTitle('Diagrama de Lorenz (RRn frente a RRn+1)'); self.g1.setLabel('bottom', 'RR n (ms)'); self.g1.setLabel('left', 'RR n+1 (ms)')
            self.g1.addItem(sp); self.g1.setAspectLocked(True)
            okn = np.isfinite(rr) & (ex.cls == 0)
            y, e = np.histogram(rr[okn], bins=np.arange(200, 2600, 20))
            self.g2.setTitle('RR de latidos normales (ms)'); self.g2.addItem(pg.BarGraphItem(x=e[:-1] + 10, height=y, width=18, brush='#1f5fbf', pen=None))
        if self.modo != 'lorenz':
            self.g1.setAspectLocked(False)

    def _hist(self, g, cl, titulo, vals, mask, edges, fuera, nombre):
        g.setTitle(titulo); g.setLabel('left', 'latidos')
        v = vals[mask]; idx = np.nonzero(mask)[0]
        y, _ = np.histogram(np.clip(v, edges[0], edges[-1] - 1e-6), bins=edges)
        cent = (edges[:-1] + edges[1:]) / 2; w = (edges[1] - edges[0]) * 0.9
        col = ['#dc2626' if fuera(c) else '#1f5fbf' for c in cent]
        g.addItem(pg.BarGraphItem(x=cent, height=y, width=w, brushes=col, pen=None))
        self.bins[cl] = ('lat', edges, np.clip(v, edges[0], edges[-1] - 1e-6), None, idx, nombre)

    def _clic(self, g, ev):
        cl = 'g1' if g is self.g1 else 'g2'
        if cl not in self.bins or not g.sceneBoundingRect().contains(ev.scenePos()):
            return
        x = g.getPlotItem().vb.mapSceneToView(ev.scenePos()).x()
        edges = self.bins[cl][1]
        j = int(np.searchsorted(edges, x) - 1)
        if 0 <= j < len(edges) - 1:
            self.sel = (cl, j); self._ir_bin(cl, j)

    def _ir_bin(self, cl, j):
        kind, edges, vals, t0, idx, nombre = self.bins[cl]
        m = (vals >= edges[j]) & (vals < edges[j + 1])
        if kind == 'lat':
            bs = idx[m]
            its = [(float(self.ex.t[b]), int(b)) for b in bs[:20000]]
        else:
            its = [(float(t), int(b)) for t, b in zip(t0[m], idx[m])]
        a, b = edges[j], edges[j + 1]
        self.nav.emit(f'{nombre} {fmt(a)}–{fmt(b)} ({len(its)})', its)

    def keyPressEvent(self, e):
        if self.sel and e.key() in (Qt.Key_Left, Qt.Key_Right):
            cl, j = self.sel; edges = self.bins[cl][1]
            j = int(np.clip(j + (1 if e.key() == Qt.Key_Right else -1), 0, len(edges) - 2)); self.sel = (cl, j); self._ir_bin(cl, j)
        else:
            super().keyPressEvent(e)


# --------------------------------------------------------------------------- Criterios de escaneo
class VistaCriterios(QWidget):
    aplicar = Signal(dict)
    reescanear = Signal()

    CAMPOS = [('svPrematurity', 'Supraventricular: prematuridad mínima (fracción)', 0.25, 0.05, 0.6, 0.01),
              ('pauseSec', 'Pausa: RR mayor que (s)', 2.0, 1.0, 6.0, 0.1),
              ('stDepUv', 'Descenso del ST (µV)', 100, 50, 400, 10), ('stEleUv', 'Elevación del ST (µV)', 100, 50, 400, 10),
              ('stPointMs', 'ST medido en J + (ms)', 60, 40, 120, 20),
              ('tachyBpm', 'Taquicardia sobre (lpm)', 100, 80, 200, 1), ('bradyBpm', 'Bradicardia bajo (lpm)', 50, 20, 80, 1),
              ('minRateDurSec', 'Bradi/taqui: duración mínima (s)', 30, 5, 3600, 5),
              ('vtBpm', 'TV: frecuencia mínima (lpm)', 100, 60, 250, 1), ('vtBeats', 'TV: latidos consecutivos', 3, 3, 20, 1),
              ('svtBpm', 'TSV: frecuencia mínima (lpm)', 100, 60, 250, 1), ('svtBeats', 'TSV: latidos consecutivos', 3, 3, 20, 1),
              ('pacerRate', 'Marcapasos: frecuencia programada (lpm)', 60, 30, 150, 1)]
    CASILLAS = [('pauseNNOnly', 'La pausa se mide solo entre latidos normales (N a N)', False),
                ('afDetect', 'Detectar automáticamente la fibrilación auricular', True),
                ('svTemplates', 'Habilitar el grupo de plantillas supraventricular', True),
                ('hrvNormalOnly', 'VFC solo con latidos normales (si no, normales y supraventriculares)', True),
                ('hrNormalOnly', 'Frecuencia cardíaca solo con latidos normales', False),
                ('hrExcludePauses', 'Excluir las pausas del cálculo de la frecuencia cardíaca', False),
                ('pacing', 'Análisis de marcapasos activado', False)]

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self); lay.setContentsMargins(10, 10, 10, 10)
        g = QGroupBox('Criterios de escaneo'); f = QFormLayout(g)
        self.w = {}
        for k, lbl, d, lo, hi, st in self.CAMPOS:
            s = QDoubleSpinBox() if isinstance(st, float) else QSpinBox()
            s.setRange(lo, hi); s.setSingleStep(st)
            if isinstance(s, QDoubleSpinBox):
                s.setDecimals(2)
            s.setValue(d); self.w[k] = s; f.addRow(lbl, s)
        self.qtc = QComboBox(); [self.qtc.addItem(n, k) for k, n in (('bazett', 'Bazett'), ('fridericia', 'Fridericia'), ('framingham', 'Lineal (Framingham)'), ('hodges', 'Hodges'))]
        f.addRow('Fórmula del QTc', self.qtc)
        f.addRow('RR para el QTc', QLabel('RR medio de cada minuto (el QT se mide sobre el latido promedio del minuto)'))
        lay.addWidget(g)
        der = QVBoxLayout(); g2 = QGroupBox('Opciones'); v = QVBoxLayout(g2); self.c = {}
        for k, lbl, d in self.CASILLAS:
            c = QCheckBox(lbl); c.setChecked(d); self.c[k] = c; v.addWidget(c)
        der.addWidget(g2)
        b = QPushButton('Aplicar criterios y recalcular'); b.setObjectName('primario'); b.clicked.connect(self._ap)
        b2 = QPushButton('Volver a escanear (excluir canales o acortar)…'); b2.clicked.connect(self.reescanear.emit)
        der.addWidget(b); der.addWidget(b2)
        der.addWidget(QLabel('Los cambios valen para este paciente y se guardan con el examen.'))
        der.addStretch(1); lay.addLayout(der)

    def cargar(self, s):
        for k, w in self.w.items():
            v = s.get(k)
            if k in ('stDepUv', 'stEleUv') and not v:
                v = s.get('stThreshUv')
            if v is not None:
                w.setValue(v)
        i = self.qtc.findData(s.get('qtcFormula', 'bazett')); self.qtc.setCurrentIndex(max(0, i))
        for k, c in self.c.items():
            if k in s:
                c.setChecked(bool(s[k]))

    def valores(self):
        d = {k: w.value() for k, w in self.w.items()}
        d['stThreshUv'] = min(d['stDepUv'], d['stEleUv'])
        d['qtcFormula'] = self.qtc.currentData()
        d.update({k: c.isChecked() for k, c in self.c.items()})
        return d

    def _ap(self):
        self.aplicar.emit(self.valores())


# --------------------------------------------------------------------------- Validación con anotaciones de referencia
def emparejar(ref, det, tol=0.15):
    i = j = 0; mr = np.full(len(ref), -1); md = np.full(len(det), -1)
    while i < len(ref) and j < len(det):
        d = det[j] - ref[i]
        if abs(d) <= tol:
            if j + 1 < len(det) and abs(det[j + 1] - ref[i]) < abs(d):
                j += 1; continue
            mr[i] = j; md[j] = i; i += 1; j += 1
        elif d < 0:
            j += 1
        else:
            i += 1
    return mr, md


class VistaValidacion(QTextBrowser):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.document().setDefaultStyleSheet(RESUMEN_CSS)

    def mostrar(self, ex):
        if ex.ref is None:
            self.setHtml('<p class="m">Este examen no trae anotaciones de referencia (archivo .atr junto al .hea).</p>'); return
        rt = ex.ref['t']; sym = ex.ref['sym']; det = ex.t
        keep_d = ex.cls != 4
        dt = det[keep_d]; dc = ex.cls[keep_d]
        mr, md = emparejar(rt, dt)
        tp = int((mr >= 0).sum()); fn = len(rt) - tp; fp = int((md < 0).sum())
        rows = [('Detección de latidos', tp / max(1, tp + fn), tp / max(1, tp + fp), len(rt))]
        rv = np.array([s in 'VE' for s in sym]); rs = np.array([s in 'AaJS' for s in sym])
        dv = np.isin(dc, (2, 8, 10, 11)); ds = dc == 1
        for nombre, R, D in (('Ventriculares', rv, dv), ('Supraventriculares', rs, ds)):
            m = mr >= 0
            tpk = int((R & m & D[np.clip(mr, 0, None)]).sum())
            fnk = int(R.sum()) - tpk
            fpk = int(((md >= 0) & D & ~R[np.clip(md, 0, None)]).sum() + ((md < 0) & D).sum())
            rows.append((nombre, tpk / max(1, tpk + fnk), tpk / max(1, tpk + fpk), int(R.sum())))
        h = f'<h2>Validación contra {ex.ref["file"]}</h2><p class="m">Emparejamiento latido a latido con tolerancia de 150 ms (como bxb de PhysioNet).</p>'
        h += '<table><tr><th>Medida</th><th>Sensibilidad</th><th>Valor predictivo positivo</th><th>Latidos de referencia</th></tr>'
        for n, se, pp, nr in rows:
            h += f'<tr><td>{n}</td><td class="n">{fmt(100 * se, 2)} %</td><td class="n">{fmt(100 * pp, 2)} %</td><td class="n">{fmt(nr)}</td></tr>'
        self.setHtml(h + '</table>')
