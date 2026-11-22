"""Tendencias de arritmia, resumen de arritmias en bloques de color, resumen tabular con día y noche y directorio de bandas
(manual de medilog Darwin2, 4.2.13, 4.2.16, 4.2.17 y 4.2.20)."""
import numpy as np
from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
                               QListWidget, QListWidgetItem, QMenu, QPushButton, QRadioButton, QScrollArea, QSpinBox, QSplitter,
                               QTableWidget, QTableWidgetItem, QToolTip, QVBoxLayout, QWidget)

from .darwin_comun import COLORES, GRAVEDAD, MiniTira, es_noche, eventos_darwin, fc_evento, fmt, nombre, rango
from .tiras import nueva_tira


def tipos_presentes(evs):
    by = {}
    for e in evs:
        by.setdefault(e['type'], []).append(e)
    return by


# --------------------------------------------------------------------------- tendencias de arritmia y resumen en bloques
class LienzoArritmias(QWidget):
    clic = Signal(str, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.filas = []; self.by = {}; self.modo = 'tend'; self.bloque = 300; self.color = True; self.cantidad = True
        self.eje = True; self.cursor = 0.0
        self.setMouseTracking(True); self.setMinimumHeight(200)

    def datos(self, ex, by, filas):
        self.ex = ex; self.by = by; self.filas = filas
        self.setMinimumHeight(40 + 18 * (len(filas) + (2 if self.modo == 'bloques' else 0)))
        self.update()

    def _geo(self):
        return 230, self.width() - 240, 22 if self.eje else 6

    def paintEvent(self, e):
        p = QPainter(self); p.fillRect(self.rect(), QColor('#ffffff'))
        if not self.ex:
            p.end(); return
        x0, w, y0 = self._geo(); dur = max(1.0, self.ex.dur); f = QFont(); f.setPointSizeF(8); p.setFont(f)
        if self.eje:
            p.setPen(QColor('#6b7280'))
            for h in range(0, int(dur // 3600) + 1):
                x = x0 + h * 3600 / dur * w
                p.drawLine(int(x), y0 - 4, int(x), self.height()); p.drawText(int(x) + 2, 12, self.ex.reloj(h * 3600, False))
        rh = 18
        filas = self.filas
        if self.modo == 'bloques':
            # fila condensada: el bloque toma el color del suceso más grave
            nb = int(np.ceil(dur / self.bloque)); peor = [None] * nb
            for k in filas:
                for ev in self.by.get(k, []):
                    for j in range(int(ev['t0'] // self.bloque), min(nb, int(ev['t1'] // self.bloque) + 1)):
                        if peor[j] is None or rango(k) < rango(peor[j]):
                            peor[j] = k
            p.setPen(QColor('#1f2933')); p.drawText(QRectF(4, y0, x0 - 8, rh), Qt.AlignVCenter | Qt.AlignLeft, 'Todas (la más grave)')
            for j, k in enumerate(peor):
                if k:
                    p.fillRect(QRectF(x0 + j * self.bloque / dur * w, y0 + 2, max(1.5, self.bloque / dur * w - 0.5), rh - 4), QColor(COLORES.get(k, '#64748b')))
            y0 += rh + 8
        for i, k in enumerate(filas):
            y = y0 + i * rh; evs = self.by.get(k, [])
            if i % 2:
                p.fillRect(QRectF(0, y, self.width(), rh), QColor('#f8f9fb'))
            p.setPen(QColor('#1f2933' if evs else '#9ca3af'))
            txt = nombre(self.ex, k, evs[0] if evs else None) + (f'  ({len(evs)})' if self.cantidad else '')
            p.drawText(QRectF(4, y, x0 - 8, rh), Qt.AlignVCenter | Qt.AlignLeft, txt)
            col = QColor(COLORES.get(k, '#64748b') if self.color else '#111111')
            grave = rango(k) < 12
            if self.modo == 'bloques':
                nb = int(np.ceil(dur / self.bloque)); occ = np.zeros(nb, bool)
                for ev in evs:
                    occ[int(ev['t0'] // self.bloque):min(nb, int(ev['t1'] // self.bloque) + 1)] = True
                for j in np.nonzero(occ)[0]:
                    p.fillRect(QRectF(x0 + j * self.bloque / dur * w, y + 3, max(1.5, self.bloque / dur * w - 0.5), rh - 6), col)
            else:
                for ev in evs:
                    xa = x0 + ev['t0'] / dur * w; xb = max(xa + 1.2, x0 + ev['t1'] / dur * w)
                    p.fillRect(QRectF(xa, y + 3, xb - xa, rh - 6), QColor('#f97316') if (grave and not self.color) else col)
        xc = x0 + self.cursor / dur * w
        p.setPen(QPen(QColor('#dc2626'), 1.2)); p.drawLine(int(xc), 0, int(xc), self.height())
        p.end()

    def _hit(self, pos):
        x0, w, y0 = self._geo()
        if self.modo == 'bloques':
            y0 += 18 + 8
        i = int((pos.y() - y0) // 18)
        if pos.x() < x0 or not (0 <= i < len(self.filas)):
            return None, None
        return self.filas[i], (pos.x() - x0) / max(1, w) * self.ex.dur

    def mousePressEvent(self, e):
        k, t = self._hit(e.position())
        if k:
            self.clic.emit(k, t)

    def mouseMoveEvent(self, e):
        k, t = self._hit(e.position())
        if k:
            evs = self.by.get(k, [])
            cerca = min(evs, key=lambda ev: abs(ev['t0'] - t)) if evs else None
            txt = nombre(self.ex, k, cerca) + (f' · {self.ex.reloj(cerca["t0"])}' if cerca else '')
            QToolTip.showText(e.globalPosition().toPoint(), txt, self)


COLS_TAB = [('pause', 'Pausa'), ('tachy', 'Taqui.'), ('svt', 'TSVP'), ('brady', 'Bradi.'), ('sIso', 'SVES'), ('sPair', 'Doblete SV'),
            ('sRun', 'Racha SV'), ('sBig', 'Bigem. SV'), ('sTrig', 'Trigem. SV'), ('af', 'FA'), ('vIso', 'VES'), ('vPair', 'Doblete V'),
            ('vt_ns', 'TVNS'), ('vt_s', 'TV'), ('aivr', 'RIVA'), ('vBig', 'Bigem. V'), ('vTrig', 'Trigem. V'), ('rOnT', 'R sobre T'),
            ('ha_av_block_1', 'BAV 1.º'), ('ha_av_block_mobitz1', 'Mobitz I'), ('ha_av_block_mobitz2', 'Mobitz II'), ('ha_av_block_2to1', 'BAV 2:1'),
            ('ha_av_block_advanced', 'BAV avanz.'), ('ha_av_block_complete', 'BAV compl.'), ('stDep', 'ST ↓'), ('stEle', 'ST ↑'),
            ('u1', 'Usuario 1'), ('u2', 'Usuario 2'), ('u3', 'Usuario 3')]


class VistaArritmias(QWidget):
    ir = Signal(float, int)
    nav = Signal(str, list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.by = {}; self.modo = 'tend'
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        top = QHBoxLayout(); self.bg = QButtonGroup(self)
        for k, n in (('tend', 'Tendencias de arritmia'), ('bloques', 'Resumen de arritmias (bloques)'), ('tabla', 'Resumen tabular')):
            r = QRadioButton(n); r.setChecked(k == 'tend'); r.toggled.connect(lambda on, k=k: on and self._modo(k)); self.bg.addButton(r); top.addWidget(r)
        top.addSpacing(12)
        self.c_vacias = QCheckBox('Ocultar líneas vacías'); self.c_vacias.setChecked(True)
        self.c_color = QCheckBox('Codificadas por color'); self.c_color.setChecked(True)
        self.c_cant = QCheckBox('Mostrar cantidad'); self.c_cant.setChecked(True)
        self.c_eje = QCheckBox('Eje de tiempo'); self.c_eje.setChecked(True)
        for c in (self.c_vacias, self.c_color, self.c_cant, self.c_eje):
            c.toggled.connect(self._refrescar); top.addWidget(c)
        self.l_bloque = QLabel('Tamaño de bloque'); self.bloque = QComboBox()
        [self.bloque.addItem(n, s) for n, s in (('1 min', 60), ('5 min', 300), ('15 min', 900), ('30 min', 1800), ('1 h', 3600))]
        self.bloque.setCurrentIndex(1); self.bloque.currentIndexChanged.connect(self._refrescar); top.addWidget(self.l_bloque); top.addWidget(self.bloque)
        top.addStretch(1); lay.addLayout(top)
        # fila de opciones de la tabla
        t2 = QHBoxLayout(); self.w_tabla = QWidget(); self.w_tabla.setLayout(t2); t2.setContentsMargins(0, 0, 0, 0)
        t2.addWidget(QLabel('Intervalos de')); self.inter = QComboBox()
        [self.inter.addItem(n, s) for n, s in (('15 min', 900), ('30 min', 1800), ('1 h', 3600), ('2 h', 7200), ('4 h', 14400))]
        self.inter.setCurrentIndex(2); t2.addWidget(self.inter)
        t2.addWidget(QLabel('Mostrar')); self.cel = QComboBox()
        [self.cel.addItem(n, k) for k, n in (('n', 'Número de sucesos'), ('dur', 'Duración de la arritmia'), ('rel', 'Duración relativa (%)'), ('ndur', 'Número y duración'))]
        t2.addWidget(self.cel)
        self.c_lat = QCheckBox('Columnas de latidos (N, SV, V, validez)'); self.c_lat.setChecked(True); t2.addWidget(self.c_lat)
        self.c_abs = QCheckBox('Valores absolutos (si no, % de latidos)'); self.c_abs.setChecked(True); t2.addWidget(self.c_abs)
        self.c_vcol = QCheckBox('Esconder columnas vacías'); self.c_vcol.setChecked(True); t2.addWidget(self.c_vcol)
        t2.addStretch(1)
        for w in (self.inter, self.cel):
            w.currentIndexChanged.connect(self._refrescar)
        for w in (self.c_lat, self.c_abs, self.c_vcol):
            w.toggled.connect(self._refrescar)
        lay.addWidget(self.w_tabla)
        self.lienzo = LienzoArritmias(); self.lienzo.clic.connect(self._clic)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll.setWidget(self.lienzo); lay.addWidget(self.scroll, 1)
        self.tabla = QTableWidget(); self.tabla.setEditTriggers(QAbstractItemView.NoEditTriggers); self.tabla.cellClicked.connect(self._celda)
        self.tabla.setAlternatingRowColors(True); lay.addWidget(self.tabla, 1)
        ay = QLabel('Clic en un suceso: va a su ECG y Tab / Mayús+Tab recorren los de la misma fila. En la tabla, clic en una celda recorre sus sucesos. '
                    'Día y noche siguen el periodo de sueño (Ajustes → Periodo de sueño).')
        ay.setObjectName('ayuda'); ay.setWordWrap(True); lay.addWidget(ay)
        self._modo('tend')

    def _modo(self, k):
        self.modo = k
        tabla = k == 'tabla'
        self.scroll.setVisible(not tabla); self.tabla.setVisible(tabla); self.w_tabla.setVisible(tabla)
        for w in (self.c_vacias, self.c_color, self.c_cant, self.c_eje):
            w.setVisible(not tabla)
        self.l_bloque.setVisible(k == 'bloques'); self.bloque.setVisible(k == 'bloques')
        self._refrescar()

    def _refrescar(self, *_):
        if self.ex:
            self.mostrar(self.ex)

    def set_tiempo(self, t):
        self.lienzo.cursor = t; self.lienzo.update()

    def mostrar(self, ex):
        self.ex = ex
        from .darwin_vfc import eventos_qt_largo
        evs = eventos_darwin(ex, eventos_qt_largo(ex))
        self.by = tipos_presentes(evs)
        if self.modo == 'tabla':
            self._tabla(); return
        filas = [k for k in GRAVEDAD if k in self.by or not self.c_vacias.isChecked()]
        filas += sorted(k for k in self.by if k not in GRAVEDAD)
        L = self.lienzo; L.modo = self.modo; L.bloque = self.bloque.currentData(); L.color = self.c_color.isChecked()
        L.cantidad = self.c_cant.isChecked(); L.eje = self.c_eje.isChecked()
        L.datos(ex, self.by, filas)

    def _clic(self, k, t):
        evs = self.by.get(k, [])
        if not evs:
            return
        its = [(float(e['t0']), int(e['b0']) if e.get('n') and e.get('b0', -1) >= 0 else -1) for e in evs]
        j = int(np.argmin([abs(e['t0'] - t) for e in evs]))
        self.nav.emit(nombre(self.ex, k, evs[0]), its[j:] + its[:j])

    # ---- resumen tabular
    def _tabla(self):
        ex = self.ex; paso = self.inter.currentData(); n = int(np.ceil(ex.dur / paso))
        filas = [(f'{ex.reloj(i * paso, False)}', i * paso, min(ex.dur, (i + 1) * paso)) for i in range(n)]
        filas += [('Día', 'dia', None), ('Noche', 'noche', None), ('Registro completo', 'todo', None)]
        M = ex.A.get('minute') or {}
        hr = np.array([np.nan if v is None else v for v in M.get('hr', [])], float)
        hmn = np.array([np.nan if v is None else v for v in M.get('hrMin', [])], float)
        hmx = np.array([np.nan if v is None else v for v in M.get('hrMax', [])], float)
        tm = np.arange(len(hr)) * 60 + 30
        noche_m = es_noche(ex, tm) if len(tm) else np.zeros(0, bool)
        noche_b = es_noche(ex, ex.t) if len(ex.t) else np.zeros(0, bool)
        modo = self.cel.currentData()
        cols = [c for c in COLS_TAB if not self.c_vcol.isChecked() or c[0] in self.by]
        cab = ['Hora', 'FC mín', 'FC media', 'FC máx']
        if self.c_lat.isChecked():
            cab += ['Latidos', 'N', 'SV', 'V', 'Validez %']
        cab += [nombre(ex, k) if k.startswith('u') else n for k, n in cols]
        self.tabla.clear(); self.tabla.setColumnCount(len(cab)); self.tabla.setRowCount(len(filas)); self.tabla.setHorizontalHeaderLabels(cab)
        self.celdas = {}
        cls = ex.cls; ven = np.isin(cls, (2, 3, 8, 10, 11)); sv = cls == 1; nn = cls == 0; val = cls != 4
        for r, (lab, a, b) in enumerate(filas):
            if a == 'dia':
                mm = ~noche_m; mb = ~noche_b; sel = lambda e: not es_noche(ex, [e['t0']])[0]; seg = float((~es_noche(ex, np.arange(0, ex.dur, 60))).sum() * 60)
            elif a == 'noche':
                mm = noche_m; mb = noche_b; sel = lambda e: bool(es_noche(ex, [e['t0']])[0]); seg = float(es_noche(ex, np.arange(0, ex.dur, 60)).sum() * 60)
            elif a == 'todo':
                mm = np.ones(len(hr), bool); mb = np.ones(len(ex.t), bool); sel = lambda e: True; seg = ex.dur
            else:
                mm = (tm >= a) & (tm < b); mb = (ex.t >= a) & (ex.t < b); sel = (lambda e, a=a, b=b: a <= e['t0'] < b); seg = b - a
            v = [lab, fmt(np.nanmin(hmn[mm])) if mm.any() and np.isfinite(hmn[mm]).any() else '—',
                 fmt(np.nanmean(hr[mm])) if mm.any() and np.isfinite(hr[mm]).any() else '—',
                 fmt(np.nanmax(hmx[mm])) if mm.any() and np.isfinite(hmx[mm]).any() else '—']
            if self.c_lat.isChecked():
                tot = int(mb.sum()); nv = int((mb & val).sum())
                if self.c_abs.isChecked():
                    v += [fmt(tot), fmt(int((mb & nn).sum())), fmt(int((mb & sv).sum())), fmt(int((mb & ven).sum()))]
                else:
                    q = max(1, nv); v += [fmt(tot), fmt(100 * (mb & nn).sum() / q, 1), fmt(100 * (mb & sv).sum() / q, 2), fmt(100 * (mb & ven).sum() / q, 2)]
                v.append(fmt(100 * nv / max(1, tot), 1))
            base = len(v)
            for j, (k, _) in enumerate(cols):
                evs = [e for e in self.by.get(k, []) if sel(e)]
                d = sum(max(0.0, e['t1'] - e['t0']) for e in evs)
                if modo == 'n':
                    txt = fmt(len(evs)) if evs else ''
                elif modo == 'dur':
                    txt = (fmt(d / 60, 1) + ' min') if evs else ''
                elif modo == 'rel':
                    txt = (fmt(100 * d / max(1, seg), 2) + ' %') if evs else ''
                else:
                    txt = f'{len(evs)} · {fmt(d / 60, 1)} min' if evs else ''
                v.append(txt); self.celdas[(r, base + j)] = (k, evs)
            for c, txt in enumerate(v):
                it = QTableWidgetItem(str(txt)); it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter if c else Qt.AlignLeft | Qt.AlignVCenter)
                if r >= n:
                    f = it.font(); f.setBold(True); it.setFont(f)
                self.tabla.setItem(r, c, it)
        self.tabla.resizeColumnsToContents()

    def _celda(self, r, c):
        k, evs = self.celdas.get((r, c), (None, []))
        if evs:
            its = [(float(e['t0']), int(e['b0']) if e.get('n') and e.get('b0', -1) >= 0 else -1) for e in evs]
            self.nav.emit(f'{nombre(self.ex, k, evs[0])} · {self.tabla.item(r, 0).text()}', its)


# --------------------------------------------------------------------------- directorio de bandas
def gravedad_evento(e):
    k = e['type']
    if k == 'pause':
        return e.get('rr') or (e['t1'] - e['t0'])
    if k in ('stDep', 'stEle'):
        return abs(e.get('level') or 0)
    if k in ('brady',):
        return -(e.get('rate') or 0)
    if e.get('rate'):
        return e['rate']
    return e.get('dur') or (e['t1'] - e['t0'])


class Banda(QWidget):
    def __init__(self, vista, e, i):
        super().__init__()
        self.vista = vista; self.e = e
        lay = QVBoxLayout(self); lay.setContentsMargins(2, 2, 2, 2); lay.setSpacing(1)
        fila = QHBoxLayout()
        self.chk = QCheckBox('Al informe'); fila.addWidget(self.chk)
        ex = vista.ex; dur = max(0.0, e['t1'] - e['t0'])
        fmn, fmd, fmx = fc_evento(ex, e)
        info = f'<b>{i + 1}.</b> {nombre(ex, e["type"], e)} · {ex.reloj(e["t0"])} · {fmt(dur, 1)} s'
        if e.get('n'):
            info += f' · {e["n"]} latidos'
        if fmd:
            info += f' · FC {fmt(fmn)}/{fmt(fmd)}/{fmt(fmx)} lpm'
        if e['type'] == 'pause' and e.get('rr'):
            info += f' · RR {fmt(e["rr"], 2)} s'
        if e['type'] in ('stDep', 'stEle'):
            info += f' · {fmt(e.get("level"))} µV'
        com = vista.comentario(e)
        self.lbl = QLabel(info + (f' · <i>{com}</i>' if com else '')); fila.addWidget(self.lbl, 1)
        for txt, fn in (('Inicio', lambda: vista.ir.emit(float(e['t0']), -1)), ('Centro', lambda: vista.ir.emit(float((e['t0'] + e['t1']) / 2), -1)),
                        ('Final', lambda: vista.ir.emit(float(e['t1']), -1))):
            b = QPushButton(txt); b.setFlat(True); b.clicked.connect(fn); fila.addWidget(b)
        lay.addLayout(fila)
        self.tira = MiniTira(alto=vista.alto.value())
        t0 = e['t0'] - 2.0; w = vista.ventana.currentData()
        chs = list(ex.chans) if vista.canal.currentData() < 0 else [ex.chans[min(vista.canal.currentData(), len(ex.chans) - 1)]]
        self.tira.gan = vista.gan.currentData(); self.tira.etiquetas = vista.c_lat.isChecked()
        self.tira.mostrar(ex, t0, w, chs, marca=(e['t0'], min(e['t1'], t0 + w)))
        self.tira.clic.connect(lambda t: vista.ir.emit(t, -1))
        lay.addWidget(self.tira)
        self.chk.setChecked(vista.en_informe(e))
        self.chk.toggled.connect(lambda on: vista.al_informe(e, on))
        self.setContextMenuPolicy(Qt.CustomContextMenu); self.customContextMenuRequested.connect(self._menu)

    def _menu(self, pos):
        v = self.vista; e = self.e; m = QMenu(self)
        m.addAction('Comentario…', lambda: v.editar_comentario(e))
        m.addAction('Agregar o quitar del informe', lambda: self.chk.setChecked(not self.chk.isChecked()))
        if e['type'] in ('af', 'stDep', 'stEle', 'u1', 'u2', 'u3') or e['type'] in ('artifact',):
            m.addAction('Borrar este suceso', lambda: v.borrar_suceso.emit(e))
        m.exec(self.mapToGlobal(pos))


class VistaBandas(QWidget):
    ir = Signal(float, int)
    cambio = Signal()
    borrar_suceso = Signal(dict)

    POR_PAGINA = 25

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.by = {}; self.tipo = None; self.pagina = 0
        lay = QHBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        sp = QSplitter(Qt.Horizontal)
        izq = QWidget(); il = QVBoxLayout(izq); il.setContentsMargins(0, 0, 0, 0)
        self.c_todo = QCheckBox('Mostrar todo (también sin sucesos)'); self.c_todo.toggled.connect(lambda *_: self._lista()); il.addWidget(self.c_todo)
        self.lista = QListWidget(); self.lista.currentItemChanged.connect(lambda it, _: self._tipo(it)); il.addWidget(self.lista, 1)
        b4 = QPushButton('Preseleccionar las más graves'); b4.setToolTip('Marca para el informe las bandas más graves de cada tipo (4 por defecto)')
        b4.clicked.connect(self.preseleccionar); il.addWidget(b4)
        fn = QHBoxLayout(); fn.addWidget(QLabel('por tipo')); self.n_pre = QSpinBox(); self.n_pre.setRange(1, 20); self.n_pre.setValue(4); fn.addWidget(self.n_pre); fn.addStretch(1)
        il.addLayout(fn)
        bq = QPushButton('Quitar todas las bandas del informe'); bq.clicked.connect(self.quitar_todas); il.addWidget(bq)
        sp.addWidget(izq)
        der = QWidget(); dl = QVBoxLayout(der); dl.setContentsMargins(0, 0, 0, 0)
        op = QHBoxLayout(); self.bg = QButtonGroup(self); self.orden = 'grav'
        for k, n in (('crono', 'Cronológico'), ('largo', 'Longitud'), ('grav', 'Gravedad')):
            r = QRadioButton(n); r.setChecked(k == 'grav'); r.toggled.connect(lambda on, k=k: on and self._orden(k)); self.bg.addButton(r); op.addWidget(r)
        op.addSpacing(10); op.addWidget(QLabel('Canal'))
        self.canal = QComboBox(); self.canal.addItem('Todos', -1); op.addWidget(self.canal)
        self.gan = QComboBox(); [self.gan.addItem(f'{v} mm/mV', v) for v in (5.0, 10.0, 20.0)]; self.gan.setCurrentIndex(1); op.addWidget(self.gan)
        self.ventana = QComboBox(); [self.ventana.addItem(f'{v} s', float(v)) for v in (8, 10, 15, 20, 30)]; self.ventana.setCurrentIndex(1); op.addWidget(self.ventana)
        op.addWidget(QLabel('Altura')); self.alto = QSpinBox(); self.alto.setRange(80, 400); self.alto.setValue(150); self.alto.setSingleStep(10); op.addWidget(self.alto)
        self.c_lat = QCheckBox('Información de latidos'); self.c_lat.setChecked(True); op.addWidget(self.c_lat)
        self.c_marc = QCheckBox('Solo las marcadas para el informe'); op.addWidget(self.c_marc)
        for w in (self.canal, self.gan, self.ventana):
            w.currentIndexChanged.connect(lambda *_: self._bandas())
        self.alto.valueChanged.connect(lambda *_: self._bandas()); self.c_lat.toggled.connect(lambda *_: self._bandas()); self.c_marc.toggled.connect(lambda *_: self._bandas())
        op.addStretch(1); dl.addLayout(op)
        pg_ = QHBoxLayout(); self.b_ant = QPushButton('◀'); self.b_sig = QPushButton('▶'); self.l_pag = QLabel('')
        self.b_ant.clicked.connect(lambda: self._pag(-1)); self.b_sig.clicked.connect(lambda: self._pag(1))
        pg_.addWidget(self.b_ant); pg_.addWidget(self.l_pag); pg_.addWidget(self.b_sig); pg_.addStretch(1); dl.addLayout(pg_)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); dl.addWidget(self.scroll, 1)
        sp.addWidget(der); sp.setSizes([280, 1000]); lay.addWidget(sp)

    # ---- tiras del informe ligadas a las bandas
    def _clave(self, e):
        return 'banda:%s:%.1f' % (e['type'], e['t0'])

    def en_informe(self, e):
        c = self._clave(e)
        return any(t.get('clave') == c for t in self.ex.sesion.d.get('tiras', []))

    def comentario(self, e):
        return (self.ex.sesion.d.get('banda_comentarios') or {}).get(self._clave(e), '')

    def editar_comentario(self, e):
        d = self.ex.sesion.d.setdefault('banda_comentarios', {}); c = self._clave(e)
        txt, ok = QInputDialog.getText(self, 'Comentario de la banda', 'Comentario:', text=d.get(c, ''))
        if ok:
            d[c] = txt.strip()
            for t in self.ex.sesion.d.get('tiras', []):
                if t.get('clave') == c:
                    t['ann'] = self._anotacion(e)
            self.cambio.emit(); self._bandas()

    def _anotacion(self, e):
        com = self.comentario(e)
        return nombre(self.ex, e['type'], e) + (' · ' + com if com else '')

    def al_informe(self, e, on):
        L = self.ex.sesion.d['tiras']; c = self._clave(e)
        L[:] = [t for t in L if t.get('clave') != c]
        if on:
            t0 = e['t0'] - 2.0
            chs = [0, 1, 2] if self.canal.currentData() < 0 else [self.canal.currentData()]
            L.append(nueva_tira(t0, 7.5, self._anotacion(e), chs, False, c))
            L.sort(key=lambda t: t['t0'])
        self.cambio.emit()

    def ordenados(self, k):
        evs = list(self.by.get(k, []))
        if self.orden == 'crono':
            return sorted(evs, key=lambda e: e['t0'])
        if self.orden == 'largo':
            return sorted(evs, key=lambda e: -(e.get('n') or 0) - (e['t1'] - e['t0']) / 1000)
        return sorted(evs, key=lambda e: -gravedad_evento(e))

    def preseleccionar(self):
        if not self.ex:
            return
        orden = self.orden; self.orden = 'grav'
        for k in self.by:
            for e in self.ordenados(k)[:self.n_pre.value()]:
                if not self.en_informe(e):
                    self.al_informe(e, True)
        self.orden = orden; self._bandas()

    def quitar_todas(self):
        if self.ex:
            L = self.ex.sesion.d['tiras']; L[:] = [t for t in L if not str(t.get('clave', '')).startswith('banda:')]
            self.cambio.emit(); self._bandas()

    # ---- pantalla
    def mostrar(self, ex):
        self.ex = ex
        from .darwin_vfc import eventos_qt_largo
        self.by = tipos_presentes(eventos_darwin(ex, eventos_qt_largo(ex)))
        cur = self.canal.currentData(); self.canal.blockSignals(True); self.canal.clear(); self.canal.addItem('Todos', -1)
        for i, c in enumerate(ex.chans):
            self.canal.addItem(ex.names[c] if c < len(ex.names) else f'Canal {i + 1}', i)
        self.canal.setCurrentIndex(max(0, self.canal.findData(cur))); self.canal.blockSignals(False)
        self._lista()

    def _lista(self):
        if not self.ex:
            return
        tipos = [k for k in GRAVEDAD if k in self.by or self.c_todo.isChecked()] + sorted(k for k in self.by if k not in GRAVEDAD)
        self.lista.blockSignals(True); self.lista.clear(); sel = None
        for k in tipos:
            evs = self.by.get(k, [])
            it = QListWidgetItem(f'{nombre(self.ex, k, evs[0] if evs else None)}  [{len(evs)}]'); it.setData(Qt.UserRole, k)
            it.setForeground(QColor(COLORES.get(k, '#1f2933') if evs else '#9ca3af'))
            self.lista.addItem(it)
            if k == self.tipo:
                sel = it
        self.lista.blockSignals(False)
        if sel:
            self.lista.setCurrentItem(sel)
        elif self.lista.count():
            self.lista.setCurrentRow(0)
        else:
            self.scroll.setWidget(QLabel('Sin sucesos.'))

    def _tipo(self, it):
        if it:
            self.tipo = it.data(Qt.UserRole); self.pagina = 0; self._bandas()

    def _orden(self, k):
        self.orden = k; self.pagina = 0; self._bandas()

    def _pag(self, d):
        self.pagina = max(0, self.pagina + d); self._bandas()

    def _bandas(self):
        if not self.ex or not self.tipo:
            return
        evs = self.ordenados(self.tipo)
        if self.c_marc.isChecked():
            evs = [e for e in evs if self.en_informe(e)]
        npag = max(1, int(np.ceil(len(evs) / self.POR_PAGINA))); self.pagina = min(self.pagina, npag - 1)
        a = self.pagina * self.POR_PAGINA
        self.l_pag.setText(f'Página {self.pagina + 1} de {npag} · {len(evs)} bandas')
        w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(6)
        for i, e in enumerate(evs[a:a + self.POR_PAGINA]):
            v.addWidget(Banda(self, e, a + i))
        if not evs:
            v.addWidget(QLabel('No hay sucesos de este tipo.'))
        v.addStretch(1)
        self.scroll.setWidget(w)
