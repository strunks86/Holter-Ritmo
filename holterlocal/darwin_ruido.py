"""Directorio de ruido y escáner de mínimos y máximos (manual de medilog Darwin2, 4.2.14 y 4.2.15)."""
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel, QMenu, QMessageBox,
                               QPushButton, QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .darwin_comun import MiniTira, fmt
from .motor import CLS_SYM

MOTIVOS = [('limites', 'Límites excedidos'), ('ruido', 'Señal ruidosa'), ('baja', 'ECG de baja amplitud'), ('patron', 'Patrón de latido improbable')]
NOMBRE_MOTIVO = dict(MOTIVOS)


def _tramos(mask, gap=2, minimo=3):
    """Tramos [a, b) de segundos verdaderos, uniendo huecos cortos."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(np.r_[0, m.astype(np.int8), 0]); a = np.nonzero(d == 1)[0]; b = np.nonzero(d == -1)[0]
    out = []
    for x, y in zip(a, b):
        if out and x - out[-1][1] <= gap:
            out[-1][1] = y
        else:
            out.append([x, y])
    return [(int(x), int(y)) for x, y in out if y - x >= minimo]


def amplitud_por_segundo(ex):
    """Por canal: amplitud pico a pico (mV) y fracción de muestras recortadas en cada segundo."""
    cache = getattr(ex, '_amp_seg', None)
    if cache is not None and cache[0] == ex.n:
        return cache[1]
    fs = int(round(ex.fs)); nsec = int(ex.n // max(1, fs)); out = []
    for ch in ex.chans:
        x = ex.sig[ch][:nsec * fs]
        if not len(x):
            out.append((np.zeros(0), np.zeros(0))); continue
        r = x.reshape(nsec, fs)
        ptp = (r.max(1).astype(np.float64) - r.min(1)) / ex.gain[ch]
        lo, hi = x.min(), x.max()
        clip = ((r == lo) | (r == hi)).mean(1) if hi > lo else np.ones(nsec)
        out.append((ptp, clip))
    ex._amp_seg = (ex.n, out)
    return out


def episodios_ruido(ex, op, descartados=()):
    """Episodios agrupados como en Darwin: ruido potencial, excluido automáticamente y excluido manualmente."""
    nsec = len(ex.excl); excl = np.asarray(ex.excl)
    pot, auto, man = [], [], []
    for a, b in _tramos(excl == 1, gap=0, minimo=1):
        auto.append({'cat': 'auto', 't0': float(a), 't1': float(b), 'ch': -1, 'motivo': 'Los canales no se pueden analizar (ruido o sin señal)'})
    for e in (ex.A.get('exclusions') or []):
        man.append({'cat': 'man', 't0': float(e[0]), 't1': float(e[1]), 'ch': -1, 'motivo': 'Excluido por el usuario'})
    amps = amplitud_por_segundo(ex)
    libre = excl == 0
    porcanal = []
    for ci in range(len(ex.chans)):
        q = np.asarray(ex.qual[ci][:nsec])
        ptp, clip = amps[ci] if ci < len(amps) else (np.zeros(nsec), np.zeros(nsec))
        n = min(nsec, len(ptp)); ptp = np.r_[ptp[:n], np.zeros(nsec - n)]; clip = np.r_[clip[:n], np.zeros(nsec - n)]
        crit = {'limites': (clip > 0.02) | (ptp > 8.0), 'ruido': q >= 1, 'baja': (ptp < 0.25) & (q < 2)}
        marca = np.zeros(nsec, bool)
        for k, m in crit.items():
            if op.get(k, True):
                m = m & libre
                for a, b in _tramos(m):
                    pot.append({'cat': 'pot', 't0': float(a), 't1': float(b), 'ch': ci, 'motivo': NOMBRE_MOTIVO[k]})
                marca |= m
        porcanal.append(marca)
    if op.get('patron', True) and len(ex.t) > 10:
        rr = np.where(np.isfinite(ex.rr), ex.rr, 9.0)
        raro = (rr < 0.28) | (ex.cls == 4) | (ex.cls == 5)
        en_racha = np.zeros(len(ex.t), bool)
        for e in ex.eventos:
            if e['type'] in ('vt', 'svt', 'af', 'aivr') and e.get('b0', -1) >= 0:
                en_racha[int(e['b0']):int(e.get('b1', e['b0'])) + 1] = True
        raro &= ~en_racha
        win = (ex.t // 10).astype(int); nw = int(nsec // 10) + 1
        cnt = np.bincount(win[raro], minlength=nw)[:nw]; tot = np.maximum(1, np.bincount(win, minlength=nw)[:nw])
        malo = (cnt >= 3) & (cnt / tot >= 0.3)
        sec = np.repeat(malo, 10)[:nsec] & libre
        for a, b in _tramos(sec, gap=10, minimo=10):
            pot.append({'cat': 'pot', 't0': float(a), 't1': float(b), 'ch': -1, 'motivo': NOMBRE_MOTIVO['patron']})
    if op.get('completas'):
        todos = np.logical_and.reduce(porcanal) if porcanal else np.zeros(nsec, bool)
        pot = [e for e in pot if e['ch'] < 0 or todos[int(e['t0']):int(e['t1'])].mean() > 0.6]
    if op.get('sinusal'):
        def sinusal(e):
            b0, b1 = np.searchsorted(ex.t, [e['t0'], e['t1']])
            if b1 - b0 < 4:
                return False
            rr = ex.rr[b0 + 1:b1]; rr = rr[np.isfinite(rr)]
            return (ex.cls[b0:b1] == 0).mean() >= 0.95 and len(rr) > 2 and np.std(rr) / max(1e-6, np.mean(rr)) < 0.15
        pot = [e for e in pot if not sinusal(e)]
    desc = {(round(a, 0), int(c)) for a, c in descartados}
    pot = [e for e in pot if (round(e['t0'], 0), e['ch']) not in desc]
    for L in (pot, auto, man):
        L.sort(key=(lambda e: -(e['t1'] - e['t0'])) if op.get('por_longitud') else (lambda e: e['t0']))
    return pot, auto, man


class VistaRuido(QWidget):
    ir = Signal(float, int)
    exclusiones = Signal(list, list)

    CATS = [('pot', 'Ruido potencial', '#a16207'), ('auto', 'Excluido automáticamente', '#be185d'), ('man', 'Excluido manualmente', '#4b5563')]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.listas = {}
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        ayuda = QLabel('Revisa los episodios antes de los demás pasos. Supr quita un episodio de la lista: si es ruido potencial no se excluye; '
                       'si estaba excluido vuelve al análisis. «Confirmar» excluye los episodios de ruido potencial que quedan (o los elegidos).')
        ayuda.setWordWrap(True); ayuda.setObjectName('ayuda'); lay.addWidget(ayuda)
        fila = QHBoxLayout(); self.op = {}
        for k, n in MOTIVOS + [('sinusal', 'Ocultar episodios de ritmo sinusal'), ('completas', 'Solo exclusiones completas (todos los canales)'),
                              ('por_longitud', 'Ordenar por duración')]:
            c = QCheckBox(n); c.setChecked(k in dict(MOTIVOS)); c.toggled.connect(self._recalcular); self.op[k] = c; fila.addWidget(c)
        fila.addStretch(1); lay.addLayout(fila)
        sp = QSplitter(Qt.Horizontal)
        self.arbol = QTreeWidget(); self.arbol.setHeaderLabels(['Episodio', 'Hora', 'Canal', 'Motivo', 'Duración'])
        self.arbol.setSelectionMode(QAbstractItemView.ExtendedSelection); self.arbol.setUniformRowHeights(True)
        self.arbol.header().setSectionResizeMode(3, QHeaderView.Stretch); self.arbol.setColumnWidth(0, 210)
        self.arbol.currentItemChanged.connect(lambda it, _: self._ver(it))
        self.arbol.setContextMenuPolicy(Qt.CustomContextMenu); self.arbol.customContextMenuRequested.connect(self._menu)
        sp.addWidget(self.arbol)
        der = QWidget(); dl = QVBoxLayout(der); dl.setContentsMargins(0, 0, 0, 0)
        self.tira = MiniTira(alto=260); self.tira.clic.connect(lambda t: self.ir.emit(t, -1)); dl.addWidget(self.tira)
        self.det = QLabel(''); self.det.setWordWrap(True); dl.addWidget(self.det)
        b = QHBoxLayout()
        self.b_conf = QPushButton('Confirmar (excluir el ruido potencial)'); self.b_conf.setObjectName('primario'); self.b_conf.clicked.connect(self.confirmar)
        self.b_quit = QPushButton('Quitar de la lista / devolver al análisis (Supr)'); self.b_quit.clicked.connect(self.quitar)
        b.addWidget(self.b_conf); b.addWidget(self.b_quit); b.addStretch(1); dl.addLayout(b); dl.addStretch(1)
        sp.addWidget(der); sp.setSizes([560, 700]); lay.addWidget(sp, 1)
        QShortcut(QKeySequence.Delete, self.arbol, self.quitar)

    def opciones(self):
        return {k: c.isChecked() for k, c in self.op.items()}

    def _recalcular(self, *_):
        if self.ex:
            self.mostrar(self.ex)

    def mostrar(self, ex):
        self.ex = ex
        desc = ex.sesion.d.get('ruido_descartado', []) if getattr(ex, 'sesion', None) else []
        pot, auto, man = episodios_ruido(ex, self.opciones(), desc)
        self.listas = {'pot': pot, 'auto': auto, 'man': man}
        self.arbol.clear()
        for k, n, col in self.CATS:
            L = self.listas[k]
            top = QTreeWidgetItem([f'{n}  ({len(L)})']); f = top.font(0); f.setBold(True); top.setFont(0, f); top.setForeground(0, QColor(col))
            for i, e in enumerate(L[:3000]):
                ch = 'todos' if e['ch'] < 0 else (ex.names[ex.chans[e['ch']]] if ex.chans[e['ch']] < len(ex.names) else str(e['ch'] + 1))
                it = QTreeWidgetItem(['', ex.reloj(e['t0']), ch, e['motivo'], fmt(e['t1'] - e['t0']) + ' s'])
                it.setData(0, Qt.UserRole, (k, i)); top.addChild(it)
            self.arbol.addTopLevelItem(top); top.setExpanded(k == 'pot')
        self.b_conf.setText('Confirmar (excluir %d episodios de ruido potencial)' % len(pot))

    def _e(self, it):
        d = it.data(0, Qt.UserRole) if it else None
        return (d[0], self.listas[d[0]][d[1]]) if d else (None, None)

    def _ver(self, it):
        k, e = self._e(it)
        if not e:
            return
        t0 = max(0.0, e['t0'] - 2); dur = min(20.0, e['t1'] - e['t0'] + 4)
        self.tira.mostrar(self.ex, t0, dur, marca=(e['t0'], e['t1']), titulo=f'{self.ex.reloj(e["t0"])} · {fmt(e["t1"] - e["t0"])} s')
        b0, b1 = np.searchsorted(self.ex.t, [e['t0'], e['t1']])
        cl = self.ex.cls[b0:b1]
        txt = ', '.join(f'{CLS_SYM[c]} {n}' for c, n in zip(*np.unique(cl, return_counts=True))) if len(cl) else 'sin latidos'
        self.det.setText(f'<b>{e["motivo"]}</b> · {b1 - b0} latidos ({txt})')
        self.ir.emit(float(e['t0']), -1)

    def _elegidos(self):
        out = []
        for it in self.arbol.selectedItems():
            k, e = self._e(it)
            if e:
                out.append((k, e))
        return out

    def confirmar(self):
        if not self.ex:
            return
        sel = [e for k, e in self._elegidos() if k == 'pot'] or self.listas.get('pot', [])
        if not sel:
            return
        if QMessageBox.question(self, 'Directorio de ruido', f'¿Excluir del análisis {len(sel)} episodios de ruido potencial?') != QMessageBox.Yes:
            return
        excl = [list(map(float, e)) for e in (self.ex.A.get('exclusions') or [])] + [[e['t0'], e['t1']] for e in sel]
        self.exclusiones.emit(excl, [list(map(float, e)) for e in (self.ex.A.get('inclusions') or [])])

    def quitar(self):
        if not self.ex:
            return
        sel = self._elegidos()
        if not sel:
            return
        ses = self.ex.sesion.d
        excl = [list(map(float, e)) for e in (self.ex.A.get('exclusions') or [])]
        incl = [list(map(float, e)) for e in (self.ex.A.get('inclusions') or [])]
        cambio = False
        for k, e in sel:
            if k == 'pot':
                ses.setdefault('ruido_descartado', []).append([round(e['t0'], 0), e['ch']])
            elif k == 'man':
                excl = [x for x in excl if not (abs(x[0] - e['t0']) < 0.01 and abs(x[1] - e['t1']) < 0.01)]; cambio = True
            else:
                incl.append([e['t0'], e['t1']]); cambio = True
        if cambio:
            self.exclusiones.emit(excl, incl)
        else:
            self.mostrar(self.ex)

    def _menu(self, pos):
        sel = self._elegidos()
        if not sel:
            return
        m = QMenu(self)
        if any(k == 'pot' for k, _ in sel):
            m.addAction('Confirmar las entradas elegidas (excluir)', self.confirmar)
        m.addAction('Quitar de la lista / devolver al análisis\tSupr', self.quitar)
        m.exec(self.arbol.viewport().mapToGlobal(pos))


# --------------------------------------------------------------------------- escáner mínimo y máximo
MODOS_MM = [('nn', 'NN: intervalos entre latidos normales'), ('rr', 'RR: intervalos entre latidos de cualquier tipo'),
            ('fcn', 'FC de 3 latidos normales'), ('fc', 'FC de 3 latidos de cualquier tipo')]


def valores_minmax(ex, modo):
    """(índices de latido, valores) válidos para el modo: ms en NN/RR, lpm en FC."""
    n = len(ex.t)
    if n < 4:
        return np.zeros(0, int), np.zeros(0)
    rr = np.asarray(ex.rr, float); cls = np.asarray(ex.cls)
    sec = np.clip(ex.t.astype(int), 0, max(0, len(ex.excl) - 1))
    libre = (np.asarray(ex.excl)[sec] == 0) & (cls != 4) & np.isfinite(rr) & (rr > 0.2)
    prev = np.r_[False, libre[:-1]]
    norm = cls == 0; pnorm = np.r_[False, norm[:-1]]
    if modo in ('nn', 'rr'):
        m = libre & prev
        if modo == 'nn':
            m &= norm & pnorm
        idx = np.nonzero(m)[0]
        return idx, rr[idx] * 1000.0
    ok = libre.copy()
    if modo == 'fcn':
        ok &= norm & pnorm
    w = ok & np.r_[False, ok[:-1]] & np.r_[False, False, ok[:-2]]
    s = rr + np.r_[0, rr[:-1]] + np.r_[0, 0, rr[:-2]]
    idx = np.nonzero(w)[0]
    return idx, 180.0 / s[idx]


class VistaMinMax(QWidget):
    ir = Signal(float, int)
    fijar = Signal(str, int, float)          # 'fc_max' | 'fc_min' | 'rr_max' | 'rr_min', latido, valor

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.idx = np.zeros(0, int); self.val = np.zeros(0)
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        top = QHBoxLayout(); top.addWidget(QLabel('Buscar'))
        self.modo = QComboBox(); [self.modo.addItem(n, k) for k, n in MODOS_MM]; self.modo.setCurrentIndex(2)
        self.modo.currentIndexChanged.connect(lambda *_: self.mostrar(self.ex) if self.ex else None); top.addWidget(self.modo)
        self.info = QLabel(''); self.info.setObjectName('ayuda'); top.addWidget(self.info, 1); lay.addLayout(top)
        fila = QHBoxLayout(); self.arboles = {}
        for k, tit in (('alto', ''), ('bajo', '')):
            col = QVBoxLayout(); lbl = QLabel(tit); lbl.setObjectName('titulo'); col.addWidget(lbl)
            a = QTreeWidget(); a.setHeaderLabels(['#', 'Hora', 'Valor']); a.setRootIsDecorated(False); a.setUniformRowHeights(True)
            a.currentItemChanged.connect(lambda it, _: self._ir(it)); col.addWidget(a, 1)
            b = QPushButton(''); b.clicked.connect(lambda _=False, k=k: self._fijar(k)); col.addWidget(b)
            self.arboles[k] = (a, lbl, b); fila.addLayout(col, 1)
        lay.addLayout(fila, 1)
        lay.addWidget(QLabel('El valor más alto se marca en rojo y el más bajo en verde. Si el primero no es válido (artefacto), pasa al siguiente con '
                             'las flechas y fija el correcto: el resumen y las tiras de FC mínima y máxima usan el que fijes.'))

    def mostrar(self, ex):
        self.ex = ex; m = self.modo.currentData()
        self.idx, self.val = valores_minmax(ex, m)
        es_fc = m.startswith('fc')
        unidad = 'lpm' if es_fc else 'ms'
        tit = {'alto': 'FC más alta' if es_fc else 'Intervalo más largo', 'bajo': 'FC más baja' if es_fc else 'Intervalo más corto'}
        fijo = (ex.sesion.d.get('minmax_fijado') or {}) if getattr(ex, 'sesion', None) else {}
        orden = np.argsort(self.val)
        for k in ('alto', 'bajo'):
            a, lbl, b = self.arboles[k]; a.blockSignals(True); a.clear()
            o = orden[::-1] if k == 'alto' else orden
            for j, i in enumerate(o[:300]):
                b_ = int(self.idx[i])
                it = QTreeWidgetItem([str(j + 1), ex.reloj(float(ex.t[b_])), f'{fmt(self.val[i])} {unidad}'])
                it.setData(0, Qt.UserRole, (b_, float(self.val[i])))
                if j == 0:
                    it.setForeground(2, QColor('#dc2626' if k == 'alto' else '#16a34a'))
                a.addTopLevelItem(it)
            a.blockSignals(False)
            clave = self._clave(k)
            f = fijo.get(clave)
            lbl.setText(tit[k] + (f' · fijado a mano: {f["texto"]}' if f else ''))
            b.setText('Establecer como ' + {'fc_max': 'FC máxima', 'fc_min': 'FC mínima', 'rr_max': 'RR más largo', 'rr_min': 'RR más corto'}[clave] + ' del registro')
        self.info.setText(f'{fmt(len(self.idx))} valores válidos (sin artefactos ni tramos excluidos)')

    def _clave(self, k):
        es_fc = self.modo.currentData().startswith('fc')
        return ('fc_max' if k == 'alto' else 'fc_min') if es_fc else ('rr_max' if k == 'alto' else 'rr_min')

    def _ir(self, it):
        if it:
            b, _ = it.data(0, Qt.UserRole); self.ir.emit(float(self.ex.t[b]), b)

    def _fijar(self, k):
        a = self.arboles[k][0]; it = a.currentItem() or a.topLevelItem(0)
        if not it:
            return
        b, v = it.data(0, Qt.UserRole)
        self.fijar.emit(self._clave(k), int(b), float(v))
