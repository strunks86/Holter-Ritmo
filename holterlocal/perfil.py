"""Pestaña Perfil: resumen hora a hora (o cada 4 horas si el registro pasa de 48 h) con fila de totales, grupos
General / Ritmo / Estimulado, navegación al suceso de cada celda, valores cambiados a mano y columnas eliminadas."""
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QHBoxLayout, QInputDialog, QLabel, QMenu, QRadioButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .motor import VENTRICULAR


def _f(v, d=0):
    if v is None or (isinstance(v, float) and v != v):
        return ''
    s = f'{v:,.{d}f}'
    return s.replace(',', 'X').replace('.', ',').replace('X', '.')


# clave, título, grupos, agregación de la fila superior, decimales, navegación
COLUMNAS = [
    ('min', 'Min', 'G', 'sum', 0, None), ('lat', 'Latidos', 'G', 'sum', 0, None), ('diario', 'Diario', 'GRP', 'sum', 0, ('ev', ['diary'])),
    ('fcmin', 'FC mín', 'GRP', 'min', 0, ('fcmin', None)), ('fcmed', 'FC media', 'GRP', 'mean', 0, None), ('fcmax', 'FC máx', 'GRP', 'max', 0, ('fcmax', None)),
    ('rrmax', 'RR máx (s)', 'G', 'max', 2, ('rrmax', None)), ('pausas', 'Pausas', 'G', 'sum', 0, ('ev', ['pause'])),
    ('stdep', 'ST desc. (µV)', 'G', 'min', 0, ('ev', ['stDep'])), ('stele', 'ST elev. (µV)', 'G', 'max', 0, ('ev', ['stEle'])),
    ('pnn50', 'pNN50 %', 'G', 'none', 1, None), ('rmssd', 'RMSSD', 'G', 'none', 0, None), ('sdnn', 'SDNN', 'G', 'none', 0, None),
    ('qtc', 'QTc', 'G', 'none', 0, None),
    ('s1', 'S aisl.', 'R', 'sum', 0, ('cls1', [1])), ('s2', 'S pares', 'R', 'sum', 0, ('ev', ['sPair'])), ('s3', 'S 3+', 'R', 'sum', 0, ('ev', ['svt', 'sRun'])),
    ('stot', 'S total', 'R', 'sum', 0, ('cls', [1])), ('tsv', 'TSV', 'R', 'sum', 0, ('ev', ['svt'])), ('sbig', 'Big. S', 'R', 'sum', 0, ('ev', ['sBig'])),
    ('strig', 'Trig. S', 'R', 'sum', 0, ('ev', ['sTrig'])), ('aber', 'Aberr.', 'R', 'sum', 0, ('cls', [7])), ('brb', 'BRB', 'R', 'sum', 0, ('cls', [6])),
    ('fa', 'FA %', 'R', 'none', 0, ('ev', ['af'])),
    ('v1', 'V aisl.', 'R', 'sum', 0, ('cls1', list(VENTRICULAR))), ('v2', 'V pares', 'R', 'sum', 0, ('ev', ['vPair'])),
    ('v3', 'V 3+', 'R', 'sum', 0, ('ev', ['vt', 'aivr', 'ivr'])), ('vtot', 'V total', 'R', 'sum', 0, ('cls', list(VENTRICULAR))),
    ('tv', 'TV', 'R', 'sum', 0, ('ev', ['vt'])), ('vbig', 'Big. V', 'R', 'sum', 0, ('ev', ['vBig'])), ('vtrig', 'Trig. V', 'R', 'sum', 0, ('ev', ['vTrig'])),
    ('ront', 'R/T', 'R', 'sum', 0, ('ev', ['rOnT'])), ('fus', 'Fusión', 'R', 'sum', 0, ('cls', [3])), ('interp', 'Interp.', 'R', 'sum', 0, ('ev', ['interp'])),
    ('esc', 'Escape', 'R', 'sum', 0, ('cls', [8])), ('desc', 'Descon.', 'R', 'sum', 0, ('cls', [5])),
    ('pa', 'A estim.', 'P', 'sum', 0, ('cls', [12])), ('pv', 'V estim.', 'P', 'sum', 0, ('cls', [9])), ('pd', 'Doble', 'P', 'sum', 0, ('cls', [13])),
    ('ptot', 'Total estim.', 'P', 'sum', 0, ('cls', [9, 12, 13])), ('pfc', 'Fallo captura', 'P', 'sum', 0, ('ev', ['ha_pacing_failure'])),
    ('pfs', 'Fallo sensado', 'P', 'sum', 0, ('ev', ['ha_pacing_sensing'])),
    ('u1', 'Usuario 1', 'GRP', 'sum', 0, ('ev', ['u1'])), ('u2', 'Usuario 2', 'GRP', 'sum', 0, ('ev', ['u2'])), ('u3', 'Usuario 3', 'GRP', 'sum', 0, ('ev', ['u3'])),
]
COL = {c[0]: c for c in COLUMNAS}
GRUPOS = [('*', 'Todo'), ('G', 'General'), ('R', 'Ritmo'), ('P', 'Estimulado')]


def calcular(ex):
    """Tabla del perfil: (periodo en s, filas[dict], fila de totales dict)."""
    P = 3600 if ex.dur <= 48 * 3600 + 60 else 4 * 3600
    nper = int(np.ceil(ex.dur / P))
    t = ex.t; cls = ex.cls
    hidx = np.minimum((t // P).astype(int), nper - 1) if len(t) else np.zeros(0, int)
    cnt = lambda ks: np.bincount(hidx[np.isin(cls, ks)], minlength=nper)[:nper]
    by = ex.por_tipo()

    def evc(tipos, campo=None):
        out = np.zeros(nper);
        for ty in tipos:
            for e in by.get(ty, []):
                h = min(nper - 1, int(e['t0'] // P))
                out[h] += (e.get(campo) or 0) if campo else 1
        return out

    M = ex.A.get('minute') or {}
    arr = lambda k: np.array([np.nan if v is None else v for v in M.get(k, [])], float)
    hr, hmin, hmax, exc = arr('hr'), arr('hrMin'), arr('hrMax'), arr('excl')
    mper = P // 60
    filas = []
    vt = cnt(list(VENTRICULAR)); st = cnt([1])
    v2 = evc(['vPair']); v3n = evc(['vt', 'aivr', 'ivr'], 'n'); s2 = evc(['sPair']); s3n = evc(['svt', 'sRun'], 'n')
    afb = np.bincount(hidx[ex.afBeat.astype(bool)], minlength=nper)[:nper] if len(t) else np.zeros(nper)
    lat = np.bincount(hidx[cls != 4], minlength=nper)[:nper] if len(t) else np.zeros(nper)
    rrmax = np.zeros(nper)
    ok = np.isfinite(ex.rr)
    if ok.any():
        np.maximum.at(rrmax, hidx[ok], ex.rr[ok])
    hours = ex.A.get('hours') or []
    stdep = np.full(nper, np.nan); stele = np.full(nper, np.nan)
    for e in by.get('stDep', []):
        h = min(nper - 1, int(e['t0'] // P)); stdep[h] = np.nanmin([stdep[h], e.get('level', np.nan)])
    for e in by.get('stEle', []):
        h = min(nper - 1, int(e['t0'] // P)); stele[h] = np.nanmax([stele[h], e.get('level', np.nan)])
    cols = {
        'lat': lat, 'diario': evc(['diary']), 'pausas': evc(['pause']), 'rrmax': rrmax,
        's1': st - 2 * s2 - s3n, 's2': s2, 's3': evc(['svt', 'sRun']), 'stot': st, 'tsv': evc(['svt']), 'sbig': evc(['sBig']), 'strig': evc(['sTrig']),
        'aber': cnt([7]), 'brb': cnt([6]), 'fa': np.where(lat > 0, 100 * afb / np.maximum(1, lat), 0),
        'v1': vt - 2 * v2 - v3n, 'v2': v2, 'v3': evc(['vt', 'aivr', 'ivr']), 'vtot': vt, 'tv': evc(['vt']), 'vbig': evc(['vBig']), 'vtrig': evc(['vTrig']),
        'ront': evc(['rOnT']), 'fus': cnt([3]), 'interp': evc(['interp']), 'esc': cnt([8]), 'desc': cnt([5]),
        'pa': cnt([12]), 'pv': cnt([9]), 'pd': cnt([13]), 'ptot': cnt([9, 12, 13]), 'pfc': evc(['ha_pacing_failure']), 'pfs': evc(['ha_pacing_sensing']),
        'u1': evc(['u1'], 'n'), 'u2': evc(['u2'], 'n'), 'u3': evc(['u3'], 'n'), 'stdep': stdep, 'stele': stele,
    }
    for h in range(nper):
        sl = slice(h * mper, (h + 1) * mper)
        r = {'t0': h * P}
        r['min'] = float(np.nansum(1 - exc[sl])) if len(exc) else 0
        r['fcmin'] = float(np.nanmin(hmin[sl])) if len(hmin[sl]) and np.isfinite(hmin[sl]).any() else np.nan
        r['fcmax'] = float(np.nanmax(hmax[sl])) if len(hmax[sl]) and np.isfinite(hmax[sl]).any() else np.nan
        r['fcmed'] = float(np.nanmean(hr[sl])) if len(hr[sl]) and np.isfinite(hr[sl]).any() else np.nan
        for k, v in cols.items():
            r[k] = float(v[h])
        r['rrmax'] = r['rrmax'] or np.nan
        if P == 3600 and h < len(hours):
            H = hours[h]
            for k, hk in (('pnn50', 'pnn50'), ('rmssd', 'rmssd'), ('sdnn', 'sdnn'), ('qtc', 'qtc')):
                v = H.get(hk); r[k] = np.nan if v is None else v
        filas.append(r)
    tot = {}
    for c in COLUMNAS:
        k, ag = c[0], c[3]
        vals = np.array([r.get(k, np.nan) for r in filas], float)
        if ag == 'sum':
            tot[k] = float(np.nansum(vals))
        elif ag == 'min':
            tot[k] = float(np.nanmin(vals)) if np.isfinite(vals).any() else np.nan
        elif ag == 'max':
            tot[k] = float(np.nanmax(vals)) if np.isfinite(vals).any() else np.nan
        else:
            tot[k] = np.nan
    hrA = ex.A.get('hr') or {}; hrv = ex.A.get('hrv') or {}
    tot['fcmed'] = hrA.get('mean') if hrA.get('mean') is not None else np.nan
    tot['fa'] = (ex.A.get('af') or {}).get('pct', np.nan)
    tot['pnn50'] = hrv.get('pnn50', np.nan); tot['rmssd'] = hrv.get('rmssd', np.nan); tot['sdnn'] = hrv.get('sdnn', np.nan)
    tot['qtc'] = (ex.A.get('qt') or {}).get('mean', np.nan)
    return P, filas, tot


def items_nav(ex, col, t0, t1):
    """Latidos o sucesos que corresponden a una celda, para recorrerlos con Tab."""
    nav = COL[col][5]
    if not nav:
        return []
    tipo, arg = nav
    if tipo == 'ev':
        return [(float(e['t0']), int(e.get('b0', -1)) if e.get('n') else -1) for e in ex.eventos if e['type'] in arg and t0 <= e['t0'] < t1]
    if tipo in ('cls', 'cls1'):
        idx = np.nonzero(np.isin(ex.cls, arg) & (ex.t >= t0) & (ex.t < t1))[0]
        return [(float(ex.t[b]), int(b)) for b in idx[:20000]]
    m = (ex.t >= t0) & (ex.t < t1)
    if tipo == 'rrmax':
        rr = np.where(m & np.isfinite(ex.rr), ex.rr, -1)
        b = int(np.argmax(rr)) if len(rr) else -1
        return [(float(ex.t[b]), b)] if b >= 0 and rr[b] > 0 else []
    h = np.where(m & np.isfinite(ex.hr8), ex.hr8, np.nan)
    if not np.isfinite(h).any():
        return []
    b = int(np.nanargmin(h) if tipo == 'fcmin' else np.nanargmax(h))
    return [(float(ex.t[b]), b)]


class VistaPerfil(QWidget):
    nav = Signal(str, list)
    cambio = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.grupo = '*'
        lay = QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        top = QHBoxLayout(); self.bg = QButtonGroup(self)
        for k, n in GRUPOS:
            r = QRadioButton(n); r.setChecked(k == '*'); r.toggled.connect(lambda on, k=k: on and self._grupo(k)); self.bg.addButton(r); top.addWidget(r)
        self.info = QLabel('Clic en una celda: va al primer suceso · Tab / Mayús+Tab: siguiente / anterior · clic derecho: cambiar o restaurar')
        self.info.setObjectName('ayuda'); top.addSpacing(12); top.addWidget(self.info, 1)
        lay.addLayout(top)
        self.tabla = QTableWidget(); self.tabla.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tabla.setSelectionMode(QAbstractItemView.SingleSelection); self.tabla.setAlternatingRowColors(True)
        self.tabla.verticalHeader().setDefaultSectionSize(22)
        self.tabla.cellClicked.connect(self._clic)
        self.tabla.setContextMenuPolicy(Qt.CustomContextMenu); self.tabla.customContextMenuRequested.connect(self._menu)
        self.tabla.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.tabla.horizontalHeader().customContextMenuRequested.connect(self._menu_col)
        lay.addWidget(self.tabla, 1)

    def _grupo(self, k):
        self.grupo = k
        if self.ex:
            self.mostrar(self.ex)

    def columnas(self):
        return [c for c in COLUMNAS if self.grupo == '*' or self.grupo in c[2]]

    def mostrar(self, ex):
        self.ex = ex
        self.P, self.filas, self.tot = calcular(ex)
        ses = ex.sesion.d if getattr(ex, 'sesion', None) else {}
        ov = ses.get('perfil_override', {}); off = set(ses.get('perfil_cols_off', []))
        cols = self.columnas(); self.cols = cols
        tb = self.tabla; tb.clear(); tb.setColumnCount(len(cols)); tb.setRowCount(len(self.filas) + 1)
        nombres = []
        for c in cols:
            n = c[1]
            if c[0] in ('u1', 'u2', 'u3') and getattr(ex, 'sesion', None):
                n = ex.sesion.etiqueta_usuario(int(c[0][1]))[:14]
            nombres.append(n + (' ✕' if c[0] in off else ''))
        tb.setHorizontalHeaderLabels(nombres)
        tb.setVerticalHeaderLabels(['Total'] + [ex.reloj(r['t0'], False) for r in self.filas])
        gris = QBrush(QColor('#e5e7eb')); amarillo = QBrush(QColor('#fff4c2')); bold = None
        for j, c in enumerate(cols):
            for i, r in enumerate([self.tot] + self.filas):
                v = r.get(c[0], np.nan)
                txt = _f(v, c[4])
                key = '%s@%d' % (c[0], i - 1)
                it = QTableWidgetItem(txt)
                if key in ov:
                    it.setText(ov[key]); it.setBackground(amarillo)
                if c[0] in off:
                    it.setText('—'); it.setForeground(QBrush(QColor('#9ca3af')))
                if i == 0:
                    f = it.font(); f.setBold(True); it.setFont(f); it.setBackground(gris if key not in ov else amarillo)
                if c[5]:
                    it.setForeground(QBrush(QColor('#1f5fbf'))) if c[0] not in off else None
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                tb.setItem(i, j, it)
        tb.resizeColumnsToContents()

    def _rango(self, fila):
        if fila == 0:
            return 0.0, self.ex.dur + 1
        t0 = self.filas[fila - 1]['t0']; return t0, t0 + self.P

    def _clic(self, fila, col):
        if not self.ex:
            return
        c = self.cols[col]
        t0, t1 = self._rango(fila)
        its = items_nav(self.ex, c[0], t0, t1)
        if its:
            quien = 'todo el registro' if fila == 0 else self.ex.reloj(t0, False)
            self.nav.emit(f'{c[1]} · {quien}', its)

    def _menu(self, pos):
        it = self.tabla.itemAt(pos)
        if not it or not self.ex:
            return
        fila, col = it.row(), it.column(); c = self.cols[col]; key = '%s@%d' % (c[0], fila - 1)
        m = QMenu(self)
        m.addAction('Ir al primer suceso', lambda: self._clic(fila, col))
        m.addAction('Cambiar el valor…', lambda: self._cambiar(key, it.text()))
        ov = self.ex.sesion.d['perfil_override']
        a = m.addAction('Restaurar el valor', lambda: self._restaurar(key)); a.setEnabled(key in ov)
        m.addSeparator(); self._acc_col(m, c[0])
        m.exec(self.tabla.viewport().mapToGlobal(pos))

    def _menu_col(self, pos):
        col = self.tabla.horizontalHeader().logicalIndexAt(pos)
        if col < 0 or not self.ex:
            return
        m = QMenu(self); self._acc_col(m, self.cols[col][0]); m.exec(self.tabla.horizontalHeader().mapToGlobal(pos))

    def _acc_col(self, m, k):
        off = self.ex.sesion.d['perfil_cols_off']
        if k in off:
            m.addAction('Restaurar todos los valores de la columna', lambda: self._col(k, False))
        else:
            m.addAction('Eliminar todos los valores de la columna', lambda: self._col(k, True))

    def _col(self, k, quitar):
        off = self.ex.sesion.d['perfil_cols_off']
        if quitar and k not in off:
            off.append(k)
        elif not quitar and k in off:
            off.remove(k)
        self.mostrar(self.ex); self.cambio.emit()

    def _cambiar(self, key, actual):
        txt, ok = QInputDialog.getText(self, 'Perfil', 'Nuevo valor:', text=actual)
        if ok:
            self.ex.sesion.d['perfil_override'][key] = txt; self.mostrar(self.ex); self.cambio.emit()

    def _restaurar(self, key):
        self.ex.sesion.d['perfil_override'].pop(key, None); self.mostrar(self.ex); self.cambio.emit()
