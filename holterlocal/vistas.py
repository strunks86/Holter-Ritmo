"""Pestañas de la aplicación: resumen, eventos, plantillas, tendencias, histogramas, criterios y validación."""
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QListWidget, QListWidgetItem, QPushButton, QSpinBox,
                               QSplitter, QTextBrowser, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
                               QCheckBox)

from .estilo import COLOR_CLASE, RESUMEN_CSS
from .motor import CLS_NAME, CLS_SYM, EVENT_LABEL, EVENT_NAMES

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


# --------------------------------------------------------------------------- Resumen
class VistaResumen(QTextBrowser):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenExternalLinks(False)
        self.document().setDefaultStyleSheet(RESUMEN_CSS)
        self.setHtml('<p class="m">Sin examen abierto.</p>')

    def html(self, ex):
        A = ex.A; hr = A.get('hr') or {}; cnt = A.get('cnt') or {}; c = A.get('counts') or [0] * 10
        by = {}
        for e in ex.eventos:
            by.setdefault(e['type'], []).append(e)
        n = lambda k: len(by.get(k, []))
        filas = lambda rows: '<table>' + ''.join(f'<tr><td>{a}</td><td class="n"><b>{b}</b></td><td class="m">{c_ or ""}</td></tr>' for a, b, c_ in rows) + '</table>'
        h = [f'<h2>{ex.nombre}</h2><p class="m">{ex.fuente} · {ex.nsig} canales a {fmt(ex.fs)} Hz · inicio {ex.reloj(0)} · '
             f'duración {dur_txt(ex.dur)} · motor {A.get("motor", "")}</p>']
        h.append('<h3>Calidad de la señal</h3>' + filas([
            ('Tiempo analizado', dur_txt(A.get('analyzedSec', 0)), ''),
            ('Tiempo excluido por artefacto o sin señal', fmt(A.get('exPct', 0), 1) + ' %', dur_txt(A.get('exSec', 0))),
            ('Latidos marcados como artefacto', fmt(c[4]), ''),
        ]))
        h.append('<h3>Frecuencia cardíaca</h3>' + filas([
            ('Latidos totales', fmt(A.get('total')), ''),
            ('FC media', fmt(hr.get('mean')) + ' lpm', ''),
            ('FC mínima', fmt(hr.get('min')) + ' lpm', ex.reloj(ex.t[hr['minB']]) if hr.get('minB') is not None and hr.get('minB', -1) < len(ex.t) else ''),
            ('FC máxima', fmt(hr.get('max')) + ' lpm', ex.reloj(ex.t[hr['maxB']]) if hr.get('maxB') is not None and hr.get('maxB', -1) < len(ex.t) else ''),
            ('RR más largo', fmt(A.get('rrMax'), 2) + ' s', ex.reloj(ex.t[A['rrMaxB']]) if A.get('rrMaxB') is not None and 0 <= A.get('rrMaxB', -1) < len(ex.t) else ''),
            ('Pausas ≥ %s s' % fmt((A.get('settings') or {}).get('pauseSec', 2), 1), fmt(A.get('pauses')), ''),
            ('Bradicardia / taquicardia', f'{n("brady")} / {n("tachy")}', 'episodios'),
        ]))
        vt = by.get('vt', [])
        h.append('<h3>Ectopia ventricular</h3>' + filas([
            ('Extrasístoles ventriculares', fmt(A.get('vTot')), fmt(A.get('vBurden'), 2) + ' % de los latidos'),
            ('Aisladas / pares / rachas', f'{fmt(cnt.get("vIso"))} / {fmt(cnt.get("vPair"))} / {fmt(cnt.get("vRun"))}', ''),
            ('Bigeminismo / trigeminismo', f'{n("vBig")} / {n("vTrig")}', 'tramos'),
            ('Taquicardia ventricular', fmt(len(vt)), ('más larga %d latidos, más rápida %s lpm' % (max(e['n'] for e in vt), fmt(max(e.get('rate', 0) for e in vt)))) if vt else ''),
            ('R sobre T / interpoladas / fusión', f'{fmt(A.get("rOnT"))} / {fmt(A.get("interp"))} / {fmt(c[3])}', ''),
        ]))
        svt = by.get('svt', [])
        h.append('<h3>Ectopia supraventricular</h3>' + filas([
            ('Extrasístoles supraventriculares', fmt(A.get('sTot')), fmt(A.get('sBurden'), 2) + ' % de los latidos'),
            ('Aisladas / pares / rachas', f'{fmt(cnt.get("sIso"))} / {fmt(cnt.get("sPair"))} / {fmt(cnt.get("sRun"))}', ''),
            ('Bigeminismo / trigeminismo', f'{n("sBig")} / {n("sTrig")}', 'tramos'),
            ('Taquicardia supraventricular', fmt(len(svt)), ('más larga %d latidos' % max(e['n'] for e in svt)) if svt else ''),
        ]))
        af = A.get('af') or {}
        h.append('<h3>Fibrilación auricular</h3>' + filas([
            ('Carga de FA', fmt(af.get('pct'), 1) + ' %', dur_txt(af.get('sec', 0))),
            ('Episodios', fmt(af.get('episodes')), ('el más largo ' + dur_txt(af.get('longest', 0))) if af.get('episodes') else ''),
        ]))
        extra = A.get('extra_html')
        if extra:
            h.append(extra)
        qt = A.get('qt') or {}; hrv = A.get('hrv') or {}
        if qt:
            h.append('<h3>QT</h3>' + filas([
                ('QTc medio (%s)' % (A.get('settings') or {}).get('qtcFormula', 'bazett'), fmt(qt.get('mean')) + ' ms', 'mín %s · máx %s' % (fmt(qt.get('min')), fmt(qt.get('max')))),
                ('Minutos con QTc > 450 / 500 ms', f'{fmt(qt.get("over450"), 1)} % / {fmt(qt.get("over500"), 1)} %', ''),
            ]))
        if hrv:
            h.append('<h3>Variabilidad</h3>' + filas([
                ('SDNN / SDANN / índice SDNN', f'{fmt(hrv.get("sdnn"))} / {fmt(hrv.get("sdann"))} / {fmt(hrv.get("sdnnIndex"))} ms', ''),
                ('RMSSD / pNN50', f'{fmt(hrv.get("rmssd"))} ms / {fmt(hrv.get("pnn50"), 1)} %', ''),
                ('LF / HF / cociente', f'{fmt(hrv.get("lf"))} / {fmt(hrv.get("hf"))} / {fmt(hrv.get("lfhf"), 2)}', 'ms²'),
            ]))
        hours = A.get('hours') or []
        if hours:
            cols = [('Hora', lambda x: ex.reloj(x['t0'], False)), ('Latidos', lambda x: fmt(x.get('beats'))),
                    ('FC mín', lambda x: fmt(x.get('hrMin'))), ('FC media', lambda x: fmt(x.get('hrMean'))), ('FC máx', lambda x: fmt(x.get('hrMax'))),
                    ('Pausas', lambda x: fmt(x.get('pauses'))), ('V', lambda x: fmt(x.get('vTot'))), ('V pares', lambda x: fmt(x.get('vPair'))),
                    ('V rachas', lambda x: fmt(x.get('vRun'))), ('S', lambda x: fmt(x.get('sTot'))), ('S rachas', lambda x: fmt(x.get('sRun'))),
                    ('FA %', lambda x: fmt(x.get('afPct'), 0))]
            t = '<table><tr>' + ''.join(f'<th>{a}</th>' for a, _ in cols) + '</tr>'
            for x in hours:
                t += '<tr>' + ''.join(f'<td class="n">{f(x)}</td>' for _, f in cols) + '</tr>'
            h.append('<h3>Tabla horaria</h3>' + t + '</table>')
        return ''.join(h)

    def mostrar(self, ex):
        self.setHtml(self.html(ex))


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
def icono_plantilla(T, nch, M, k, sz=QSize(150, 110)):
    pm = QPixmap(sz); pm.fill(QColor('#ffffff'))
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


class VistaPlantillas(QWidget):
    ir = Signal(float, int)
    reetiquetar_plantilla = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.pos = {}
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        top = QHBoxLayout()
        self.filtro = QComboBox(); self.filtro.addItem('Todas las morfologías', -1)
        for k in (0, 1, 2, 3, 4, 6, 7, 9):
            self.filtro.addItem(CLS_NAME[k], k)
        self.info = QLabel('Elige una morfología.'); self.info.setWordWrap(True)
        top.addWidget(QLabel('Mostrar')); top.addWidget(self.filtro); top.addWidget(self.info, 1)
        lay.addLayout(top)
        self.lista = QListWidget(); self.lista.setViewMode(QListWidget.IconMode); self.lista.setIconSize(QSize(150, 110))
        self.lista.setResizeMode(QListWidget.Adjust); self.lista.setMovement(QListWidget.Static); self.lista.setSpacing(6)
        self.lista.setSelectionMode(QAbstractItemView.SingleSelection)
        lay.addWidget(self.lista, 1)
        bot = QHBoxLayout()
        self.bprev = QPushButton('◀ Latido anterior'); self.bnext = QPushButton('Latido siguiente ▶')
        bot.addWidget(self.bprev); bot.addWidget(self.bnext); bot.addSpacing(16); bot.addWidget(QLabel('Etiquetar toda la morfología como:'))
        for k in (0, 1, 2, 3, 6, 7, 9, 4):
            b = QPushButton(CLS_NAME[k]); b.clicked.connect(lambda _=False, k=k: self._rel(k)); bot.addWidget(b)
        bot.addStretch(1)
        lay.addLayout(bot)
        self.filtro.currentIndexChanged.connect(lambda _: self.mostrar(self.ex) if self.ex else None)
        self.lista.currentItemChanged.connect(lambda it, _: self._sel(it))
        self.bprev.clicked.connect(lambda: self._nav(-1)); self.bnext.clicked.connect(lambda: self._nav(1))

    def dominante(self, i):
        tc = self.ex.A['tcls'][i] if i < len(self.ex.A.get('tcls', [])) else None
        return int(np.argmax(tc)) if tc and sum(tc) else 0

    def mostrar(self, ex):
        self.ex = ex
        cur = self.lista.currentItem().data(Qt.UserRole) if self.lista.currentItem() else None
        self.lista.clear()
        T = ex.A.get('templates') or []; nch = ex.A['nch']; M = ex.A.get('morphM', {}).get('M', 56)
        f = self.filtro.currentData()
        orden = sorted(range(len(T)), key=lambda i: -T[i]['count'])
        for i in orden:
            k = self.dominante(i)
            n = int(sum(ex.A['tcls'][i])) if ex.A.get('tcls') else T[i]['count']
            if f is not None and f >= 0 and k != f:
                continue
            if n == 0:
                continue
            it = QListWidgetItem(icono_plantilla(T[i], nch, M, k), f'#{i + 1} · {CLS_SYM[k]} · {fmt(n)}')
            it.setData(Qt.UserRole, i); it.setToolTip(f'{CLS_NAME[k]} · {fmt(n)} latidos · ancho {fmt(T[i].get("width"))} ms')
            it.setForeground(QColor(COLOR_CLASE[k]))
            self.lista.addItem(it)
            if cur == i:
                self.lista.setCurrentItem(it)

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

    def _rel(self, k):
        it = self.lista.currentItem()
        if it:
            self.reetiquetar_plantilla.emit(int(it.data(Qt.UserRole)), k)


# --------------------------------------------------------------------------- Tendencias
class EjeReloj(pg.AxisItem):
    def __init__(self, ex_fn, **kw):
        super().__init__(orientation='bottom', **kw); self.ex_fn = ex_fn

    def tickStrings(self, values, scale, spacing):
        ex = self.ex_fn()
        return [ex.reloj(v * 60, False) if ex else '' for v in values]


class VistaTendencias(pg.GraphicsLayoutWidget):
    ir = Signal(float, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None

    def mostrar(self, ex):
        self.ex = ex; self.clear()
        M = ex.A.get('minute') or {}
        if not M:
            return
        n = len(M.get('hr', []))
        x = np.arange(n) + 0.5
        arr = lambda k: np.array([np.nan if v is None else v for v in M.get(k, [])], float)
        plots = []

        def nuevo(titulo, alto=None):
            p = self.addPlot(row=len(plots), col=0, axisItems={'bottom': EjeReloj(lambda: self.ex)})
            p.setLabel('left', titulo); p.showGrid(x=True, y=True, alpha=0.25); p.setMouseEnabled(y=False)
            p.getAxis('left').setWidth(64)
            if plots:
                p.setXLink(plots[0])
            p.scene().sigMouseClicked.connect(self._clic) if not plots else None
            plots.append(p); return p

        p = nuevo('FC (lpm)')
        lo, hi, me = arr('hrMin'), arr('hrMax'), arr('hr')
        if len(lo):
            f1 = pg.PlotDataItem(x, lo, pen=pg.mkPen('#9fb7e3')); f2 = pg.PlotDataItem(x, hi, pen=pg.mkPen('#9fb7e3'))
            p.addItem(pg.FillBetweenItem(f1, f2, brush=pg.mkBrush(31, 95, 191, 50))); p.addItem(f1); p.addItem(f2)
        p.plot(x, me, pen=pg.mkPen('#1f5fbf', width=1.6))
        for k, titulo, col in (('v', 'V por minuto', '#c81e1e'), ('s', 'S por minuto', '#1d4ed8')):
            y = arr(k)
            if len(y):
                q = nuevo(titulo); q.addItem(pg.BarGraphItem(x=x, height=np.nan_to_num(y), width=0.9, brush=col, pen=None))
        y = arr('af')
        if len(y):
            q = nuevo('FA (%)'); q.addItem(pg.BarGraphItem(x=x, height=np.nan_to_num(y) * (100 if np.nanmax(np.nan_to_num(y)) <= 1 else 1), width=0.9, brush='#b45309', pen=None))
        st = M.get('st') or []
        if st:
            q = nuevo('ST (µV)')
            cols = ['#111111', '#0f766e', '#7c3aed']
            for c, s in enumerate(st):
                q.plot(x[:len(s)], np.array([np.nan if v is None else v for v in s], float), pen=pg.mkPen(cols[c % 3], width=1.2), name=ex.names[c] if c < len(ex.names) else '')
        qtc = ex.A.get('qtc')
        if qtc:
            q = nuevo('QTc (ms)'); q.plot(x[:len(qtc)], np.array([np.nan if v is None else v for v in qtc], float), pen=None, symbol='o', symbolSize=3, symbolBrush='#374151')
        for p in plots:
            p.setXRange(0, n, padding=0)
        self.plots = plots

    def _clic(self, ev):
        if not self.ex or not getattr(self, 'plots', None):
            return
        for p in self.plots:
            if p.sceneBoundingRect().contains(ev.scenePos()):
                m = p.vb.mapSceneToView(ev.scenePos()).x()
                self.ir.emit(max(0.0, float(m) * 60), -1); return


# --------------------------------------------------------------------------- Histogramas y Lorenz
class VistaHistogramas(QWidget):
    ir = Signal(float, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None
        lay = QHBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        self.g1 = pg.PlotWidget(title='Histograma de RR (latidos normales)'); self.g2 = pg.PlotWidget(title='Diagrama de Lorenz (RRn frente a RRn+1)')
        for g in (self.g1, self.g2):
            g.showGrid(x=True, y=True, alpha=0.25)
        self.g1.setLabel('bottom', 'RR (ms)'); self.g1.setLabel('left', 'latidos')
        self.g2.setLabel('bottom', 'RR n (ms)'); self.g2.setLabel('left', 'RR n+1 (ms)'); self.g2.setAspectLocked(True)
        lay.addWidget(self.g1, 1); lay.addWidget(self.g2, 1)

    def mostrar(self, ex):
        self.ex = ex; self.g1.clear(); self.g2.clear()
        rr = ex.rr * 1000.0; ok = np.isfinite(rr) & (ex.cls == 0)
        if ok.sum() < 10:
            return
        y, edges = np.histogram(rr[ok], bins=np.arange(200, 2600, 20))
        self.g1.addItem(pg.BarGraphItem(x=edges[:-1] + 10, height=y, width=18, brush='#1f5fbf', pen=None))
        a, b = rr[:-1], rr[1:]
        good = np.isfinite(a) & np.isfinite(b)
        idx = np.nonzero(good)[0]
        if len(idx) > 60000:
            idx = np.sort(np.random.default_rng(0).choice(idx, 60000, replace=False))
        brushes = [pg.mkBrush(COLOR_CLASE[int(k)]) for k in ex.cls[idx + 1]]
        sp = pg.ScatterPlotItem(a[idx], b[idx], size=3, pen=None, brush=brushes, data=idx + 1)
        sp.sigClicked.connect(lambda _, pts: self.ir.emit(float(ex.t[int(pts[0].data())]), int(pts[0].data())) if len(pts) else None)
        self.g2.addItem(sp)


# --------------------------------------------------------------------------- Criterios
class VistaCriterios(QWidget):
    aplicar = Signal(dict)

    CAMPOS = [('pauseSec', 'Pausa: RR mayor que (s)', 2.0, 1.0, 6.0, 0.1), ('bradyBpm', 'Bradicardia bajo (lpm)', 50, 20, 80, 1),
              ('tachyBpm', 'Taquicardia sobre (lpm)', 100, 80, 200, 1), ('vtBpm', 'TV: frecuencia mínima (lpm)', 100, 60, 250, 1),
              ('vtBeats', 'TV: latidos mínimos', 3, 3, 20, 1), ('svtBpm', 'TSV: frecuencia mínima (lpm)', 100, 60, 250, 1),
              ('svtBeats', 'TSV: latidos mínimos', 3, 3, 20, 1), ('svPrematurity', 'Supraventricular: prematuridad mínima (fracción)', 0.25, 0.05, 0.6, 0.01),
              ('minRateDurSec', 'Bradi/taqui: duración mínima (s)', 30, 5, 600, 5), ('stThreshUv', 'ST: umbral (µV)', 100, 50, 300, 10),
              ('stPointMs', 'ST: medido en J + (ms)', 60, 40, 120, 20)]

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(10, 10, 10, 10)
        g = QGroupBox('Criterios de análisis'); f = QFormLayout(g)
        self.w = {}
        for k, lbl, d, lo, hi, st in self.CAMPOS:
            s = QDoubleSpinBox() if isinstance(st, float) else QSpinBox()
            s.setRange(lo, hi); s.setSingleStep(st)
            if isinstance(s, QDoubleSpinBox):
                s.setDecimals(2)
            s.setValue(d); self.w[k] = s; f.addRow(lbl, s)
        self.qtc = QComboBox(); [self.qtc.addItem(n, k) for k, n in (('bazett', 'Bazett'), ('fridericia', 'Fridericia'), ('framingham', 'Framingham'), ('hodges', 'Hodges'))]
        f.addRow('Fórmula del QTc', self.qtc)
        self.af = QCheckBox('Detectar fibrilación auricular'); self.af.setChecked(True); f.addRow('', self.af)
        lay.addWidget(g)
        b = QPushButton('Aplicar criterios y recalcular'); b.setObjectName('primario'); b.clicked.connect(self._ap)
        lay.addWidget(b, 0, Qt.AlignLeft); lay.addStretch(1)

    def cargar(self, s):
        for k, w in self.w.items():
            if k in s and s[k] is not None:
                w.setValue(s[k])
        i = self.qtc.findData(s.get('qtcFormula', 'bazett')); self.qtc.setCurrentIndex(max(0, i))
        self.af.setChecked(bool(s.get('afDetect', True)))

    def _ap(self):
        d = {k: w.value() for k, w in self.w.items()}
        d['qtcFormula'] = self.qtc.currentData(); d['afDetect'] = self.af.isChecked()
        self.aplicar.emit(d)


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
        dv = np.isin(dc, (2, 8)); ds = dc == 1
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
