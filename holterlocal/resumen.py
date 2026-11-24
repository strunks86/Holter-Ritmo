"""Pestaña Resumen: valores del examen por grupos (con casillas para el informe), valores cambiados a mano que se
pueden restaurar, enlaces al ECG y campos de diagnóstico, notas y conclusiones. También arma el resumen narrativo."""
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QFormLayout, QHBoxLayout, QInputDialog, QLabel, QMenu, QPlainTextEdit, QPushButton, QSplitter,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .motor import VENTRICULAR

# columnas del perfil que, al eliminarse, borran filas del resumen
PERFIL_A_RESUMEN = {'stot': ['s_tot', 's_iso', 's_par', 's_rachas', 's_burden'], 's1': ['s_iso'], 's2': ['s_par'], 's3': ['s_rachas'],
                    'tsv': ['s_tsv'], 'sbig': ['s_bt'], 'strig': ['s_bt'], 'brb': ['brb'], 'aber': ['aber'],
                    'vtot': ['v_tot', 'v_iso', 'v_par', 'v_rachas', 'v_burden'], 'v1': ['v_iso'], 'v2': ['v_par'], 'v3': ['v_rachas'],
                    'tv': ['v_tv'], 'vbig': ['v_bt'], 'vtrig': ['v_bt'], 'ront': ['v_otros'], 'fus': ['v_otros'], 'interp': ['v_otros'],
                    'fa': ['fa_pct', 'fa_ep', 'fa_fc'], 'pausas': ['pausas'], 'desc': ['desc']}


def _f(v, d=0):
    if v is None or (isinstance(v, float) and v != v):
        return '—'
    try:
        s = f'{v:,.{d}f}'
    except (TypeError, ValueError):
        return str(v)
    return s.replace(',', 'X').replace('.', ',').replace('X', '.')


def hms(s):
    s = int(round(s or 0)); h, r = divmod(s, 3600); m, s = divmod(r, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def filas_resumen(ex):
    """Grupos del resumen: [(clave, título, [(clave, etiqueta, valor, tiempo o None, nota)])]."""
    A = ex.A; hr = A.get('hr') or {}; cnt = A.get('cnt') or {}; c = A.get('counts') or [0] * 14
    by = ex.por_tipo(); n = lambda *ks: sum(len(by.get(k, [])) for k in ks)
    tb = lambda b: float(ex.t[b]) if b is not None and 0 <= b < len(ex.t) else None
    first = lambda *ks: min((e['t0'] for k in ks for e in by.get(k, [])), default=None)
    def ext(ks, campo, mayor=True):
        evs = [e for k in ks for e in by.get(k, []) if e.get(campo) is not None]
        if not evs:
            return None
        return (max if mayor else min)(evs, key=lambda e: e[campo])
    G = []
    G.append(('gen', 'Totales', [
        ('dur', 'Duración del registro', hms(ex.dur), None, ''),
        ('analizado', 'Tiempo analizado', hms(A.get('analyzedSec', 0)), None, ''),
        ('excluido', 'Tiempo excluido (artefacto o sin señal)', _f(A.get('exPct', 0), 1) + ' %', None, hms(A.get('exSec', 0))),
        ('latidos', 'Latidos totales', _f(A.get('total')), None, ''),
        ('artef', 'Latidos marcados como artefacto', _f(c[4]), None, ''),
    ]))
    G.append(('fc', 'Frecuencia cardíaca', [
        ('fc_media', 'FC media', _f(hr.get('mean')) + ' lpm', None, ''),
        ('fc_min', 'FC mínima', _f(hr.get('min')) + ' lpm', tb(hr.get('minB')), ''),
        ('fc_max', 'FC máxima', _f(hr.get('max')) + ' lpm', tb(hr.get('maxB')), ''),
        ('fc_dia', 'FC media de día / de noche', f'{_f(hr.get("day"))} / {_f(hr.get("night"))} lpm', None, '07:00–23:00 / 23:00–07:00'),
        ('taqui', 'Episodios de taquicardia', _f(n('tachy')), first('tachy'), ''),
        ('bradi', 'Episodios de bradicardia', _f(n('brady')), first('brady'), ''),
    ]))
    e = ext(['tachy'], 'dur')
    if e:
        G[-1][2].append(('taqui_larga', 'Taquicardia más larga', hms(e['dur']), e['t0'], '%s lpm' % _f(e.get('rate'))))
    e = ext(['brady'], 'rate', False)
    if e:
        G[-1][2].append(('bradi_lenta', 'Bradicardia más lenta', _f(e.get('rate')) + ' lpm', e['t0'], hms(e.get('dur'))))
    vt = by.get('vt', [])
    vr = ext(['vt', 'aivr', 'ivr'], 'n'); vf = ext(['vt', 'aivr', 'ivr'], 'rate')
    G.append(('v', 'Ectopia ventricular', [
        ('v_tot', 'Latidos ventriculares', _f(A.get('vTot')), first('vIso', 'vPair') if False else None, '%s por hora · %s por 1000' % (
            _f(A.get('vTot', 0) / max(1e-6, A.get('analyzedSec', 1) / 3600), 1), _f(1000 * A.get('vTot', 0) / max(1, A.get('total', 1)), 1))),
        ('v_burden', 'Carga ventricular', _f(A.get('vBurden'), 2) + ' %', None, ''),
        ('v_iso', 'Aisladas', _f(cnt.get('vIso')), None, ''),
        ('v_par', 'Pares', _f(cnt.get('vPair')), first('vPair'), ''),
        ('v_rachas', 'Rachas de 3 o más', _f(cnt.get('vRun')), first('vt', 'aivr', 'ivr'), ''),
        ('v_tv', 'Taquicardia ventricular', _f(len(vt)), first('vt'), ''),
        ('v_larga', 'Racha más larga', ('%d latidos' % vr['n']) if vr else '—', vr['t0'] if vr else None, ''),
        ('v_rapida', 'Racha más rápida', (_f(vf['rate']) + ' lpm') if vf else '—', vf['t0'] if vf else None, ''),
        ('v_bt', 'Bigeminismo / trigeminismo', f'{n("vBig")} / {n("vTrig")}', first('vBig', 'vTrig'), 'episodios'),
        ('v_otros', 'R sobre T / interpoladas / fusión', f'{_f(A.get("rOnT"))} / {_f(A.get("interp"))} / {_f(c[3])}', first('rOnT'), ''),
        ('v_esc', 'Escape ventricular', _f(c[8]), None, ''),
    ]))
    sr = ext(['svt', 'sRun'], 'n'); sf = ext(['svt', 'sRun'], 'rate')
    G.append(('s', 'Ectopia supraventricular', [
        ('s_tot', 'Latidos supraventriculares', _f(A.get('sTot')), None, '%s por hora · %s por 1000' % (
            _f(A.get('sTot', 0) / max(1e-6, A.get('analyzedSec', 1) / 3600), 1), _f(1000 * A.get('sTot', 0) / max(1, A.get('total', 1)), 1))),
        ('s_burden', 'Carga supraventricular', _f(A.get('sBurden'), 2) + ' %', None, ''),
        ('s_iso', 'Aisladas', _f(cnt.get('sIso')), None, ''),
        ('s_par', 'Pares', _f(cnt.get('sPair')), first('sPair'), ''),
        ('s_rachas', 'Rachas de 3 o más', _f(cnt.get('sRun')), first('svt', 'sRun'), ''),
        ('s_tsv', 'Taquicardia supraventricular', _f(n('svt')), first('svt'), ''),
        ('s_larga', 'Racha más larga', ('%d latidos' % sr['n']) if sr else '—', sr['t0'] if sr else None, ''),
        ('s_rapida', 'Racha más rápida', (_f(sf['rate']) + ' lpm') if sf else '—', sf['t0'] if sf else None, ''),
        ('s_bt', 'Bigeminismo / trigeminismo', f'{n("sBig")} / {n("sTrig")}', first('sBig', 'sTrig'), 'episodios'),
        ('aber', 'Latidos aberrantes', _f(c[7]), None, ''),
        ('brb', 'Latidos con bloqueo de rama', _f(c[6]), None, ''),
    ]))
    af = A.get('af') or {}
    afe = by.get('af', [])
    G.append(('fa', 'Fibrilación auricular', [
        ('fa_pct', 'Carga de FA', _f(af.get('pct'), 1) + ' %', first('af'), hms(af.get('sec', 0))),
        ('fa_ep', 'Episodios', _f(af.get('episodes')), first('af'), ('el más largo ' + hms(af.get('longest', 0))) if af.get('episodes') else ''),
        ('fa_fc', 'FC media máxima en FA', (_f(max(e.get('rate') or 0 for e in afe)) + ' lpm') if afe else '—', None, ''),
    ]))
    pz = ext(['pause'], 'rr')
    G.append(('pausas', 'Pausas', [
        ('rr_max', 'RR más largo', _f(A.get('rrMax'), 2) + ' s', tb(A.get('rrMaxB')), ''),
        ('pausas', 'Pausas ≥ %s s' % _f((A.get('settings') or {}).get('pauseSec', 2), 1), _f(A.get('pauses')), first('pause'), ''),
        ('pausa_larga', 'Pausa más larga', (_f(pz['rr'], 2) + ' s') if pz else '—', pz['t0'] if pz else None, pz.get('after', '') if pz else ''),
    ]))
    if np.isin(ex.cls, (9, 12, 13)).any() or (A.get('settings') or {}).get('pacing'):
        tot = max(1, A.get('total', 1))
        G.append(('estim', 'Estimulados', [
            ('pa', 'Auricular estimulado', _f(c[12]), None, _f(100 * c[12] / tot, 1) + ' %'),
            ('pv', 'Ventricular estimulado', _f(c[9]), None, _f(100 * c[9] / tot, 1) + ' %'),
            ('pd', 'Estimulado doble', _f(c[13]), None, _f(100 * c[13] / tot, 1) + ' %'),
            ('pfc', 'Fallo de captura / de sensado', f'{n("ha_pacing_failure")} / {n("ha_pacing_sensing")}', first('ha_pacing_failure', 'ha_pacing_sensing'), ''),
        ]))
    hrv = A.get('hrv') or {}
    if hrv:
        G.append(('vfc', 'Variabilidad del RR', [
            ('sdnn', 'SDNN / SDANN / índice SDNN', f'{_f(hrv.get("sdnn"))} / {_f(hrv.get("sdann"))} / {_f(hrv.get("sdnnIndex"))} ms', None, ''),
            ('rmssd', 'RMSSD / pNN50', f'{_f(hrv.get("rmssd"))} ms / {_f(hrv.get("pnn50"), 1)} %', None, ''),
            ('tri', 'Índice triangular', _f(hrv.get('triIndex') or hrv.get('tri'), 1), None, ''),
            ('lfhf', 'LF / HF / cociente', f'{_f(hrv.get("lf"))} / {_f(hrv.get("hf"))} / {_f(hrv.get("lfhf"), 2)}', None, 'ms²'),
        ]))
    qt = A.get('qt') or {}
    if qt and qt.get('n'):
        S = A.get('settings') or {}
        mx = qt.get('maxMin', -1)
        G.append(('qt', 'QT', [
            ('qt_medio', 'QT medio', _f(qt.get('qtMean')) + ' ms', None, ''),
            ('qtc_medio', 'QTc medio (%s)' % NOMBRE_QTC.get(S.get('qtcFormula', 'bazett'), ''), _f(qt.get('mean')) + ' ms', None, 'mín %s · máx %s' % (_f(qt.get('min')), _f(qt.get('max')))),
            ('qtc_max', 'QTc máximo', _f(qt.get('max')) + ' ms', (mx * 60.0) if mx is not None and mx >= 0 else None, ''),
            ('qtc_450', 'Minutos con QTc > 450 / 500 ms', f'{_f(qt.get("over450"), 1)} % / {_f(qt.get("over500"), 1)} %', None, ''),
        ]))
    st = [e for e in ex.eventos if e['type'] in ('stDep', 'stEle')]
    filas = []
    for ty, nom in (('stDep', 'Descenso del ST'), ('stEle', 'Elevación del ST')):
        evs = [e for e in st if e['type'] == ty]
        if evs:
            e = max(evs, key=lambda e: abs(e.get('level', 0)))
            ch = e.get('ch', 0); nc = ex.names[ex.chans[ch]] if ch < len(ex.chans) and ex.chans[ch] < len(ex.names) else str(ch + 1)
            filas.append((ty, nom + ' máximo', '%s µV' % _f(e.get('level')), e['t0'], f'canal {nc} · {len(evs)} episodios'))
        else:
            filas.append((ty, nom, 'sin episodios', None, ''))
    G.append(('st', 'Segmento ST', filas))
    us = []
    for k in (1, 2, 3):
        evs = by.get('u%d' % k, [])
        if evs:
            us.append(('u%d' % k, ex.sesion.etiqueta_usuario(k) if getattr(ex, 'sesion', None) else f'Usuario {k}',
                        '%d latidos' % sum(e.get('n', 0) for e in evs), evs[0]['t0'], '%s del registro' % (_f(100 * sum(e['dur'] for e in evs) / max(1, ex.dur), 1) + ' %')))
    if us:
        G.append(('usr', 'Sucesos definidos por el usuario', us))
    return G


NOMBRE_QTC = {'bazett': 'Bazett', 'fridericia': 'Fridericia', 'framingham': 'lineal', 'hodges': 'Hodges'}

PLANTILLA_NARRATIVA = (
    'El registro comenzó a las {inicio} y duró {duracion}, con {analizado} de señal analizable. Se contaron {latidos} latidos. '
    'La frecuencia cardíaca media fue de {fc_media} lpm, con una mínima de {fc_min} lpm a las {fc_min_hora} y una máxima de {fc_max} lpm a las {fc_max_hora}.\n\n'
    'Ectopia ventricular: {v_tot} latidos ({v_burden} %), {v_iso} aisladas, {v_par} pares y {v_rachas} rachas de 3 o más; '
    '{v_tv} episodios de taquicardia ventricular. Bigeminismo / trigeminismo ventricular: {v_bt}.\n\n'
    'Ectopia supraventricular: {s_tot} latidos ({s_burden} %), {s_iso} aisladas, {s_par} pares y {s_rachas} rachas; '
    '{s_tsv} episodios de taquicardia supraventricular.\n\n'
    'Fibrilación auricular: {fa_pct} del registro en {fa_ep} episodios.\n\n'
    'El RR más largo fue de {rr_max} y hubo {pausas} pausas.\n\n'
    'Variabilidad: SDNN / SDANN / índice SDNN {sdnn}; RMSSD / pNN50 {rmssd}.\n\n'
    'QTc medio {qtc_medio}.'
)


def campos(ex, overrides=None):
    d = {}
    for _, _, filas in filas_resumen(ex):
        for k, _, v, t, _ in filas:
            d[k] = v.replace(' lpm', '') if k.startswith('fc_') else v
            if t is not None:
                d[k + '_hora'] = ex.reloj(t)
    for k, v in (overrides or {}).items():
        d[k] = v
    d['inicio'] = ex.reloj(0); d['duracion'] = hms(ex.dur); d['analizado'] = hms(ex.A.get('analyzedSec', 0))
    d.setdefault('fc_min_hora', '—'); d.setdefault('fc_max_hora', '—')
    return d


def narrativo(ex, plantilla=None):
    import string
    d = campos(ex, ex.sesion.d.get('resumen_override') if getattr(ex, 'sesion', None) else None)

    class Seguro(dict):
        def __missing__(self, k):
            return '—'
    return string.Formatter().vformat(plantilla or PLANTILLA_NARRATIVA, (), Seguro(d))


def grupos_finales(ex):
    """Grupos con los cambios del usuario aplicados (lo que va al informe)."""
    ses = ex.sesion.d if getattr(ex, 'sesion', None) else {}
    ov = ses.get('resumen_override', {}); off = set(ses.get('resumen_grupos_off', []))
    borradas = set()
    for col in ses.get('perfil_cols_off', []):
        borradas.update(PERFIL_A_RESUMEN.get(col, []))
    out = []
    for g, tit, filas in filas_resumen(ex):
        fs = []
        for k, lab, v, t, nota in filas:
            if k in borradas:
                v, nota = '0 (eliminado)', ''
            if k in ov:
                v = ov[k]
            fs.append((k, lab, v, t, nota))
        out.append((g, tit, fs, g not in off))
    return out


def html_resumen(ex, solo_incluidos=True):
    h = []
    for g, tit, fs, on in grupos_finales(ex):
        if solo_incluidos and not on:
            continue
        h.append(f'<h3>{tit}</h3><table>' + ''.join(
            f'<tr><td>{lab}</td><td class="n"><b>{v}</b></td><td class="m">{(ex.reloj(t) + " ") if t is not None else ""}{nota or ""}</td></tr>'
            for k, lab, v, t, nota in fs) + '</table>')
    extra = ex.A.get('extra_html')
    if extra:
        h.append(extra)
    return ''.join(h)


class VistaResumen(QWidget):
    ir = Signal(float, int)
    cambio = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None
        lay = QHBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        sp = QSplitter(Qt.Horizontal); lay.addWidget(sp)
        izq = QWidget(); il = QVBoxLayout(izq); il.setContentsMargins(0, 0, 0, 0)
        self.titulo = QLabel('Sin examen abierto.'); self.titulo.setObjectName('titulo'); self.titulo.setWordWrap(True); il.addWidget(self.titulo)
        self.arbol = QTreeWidget(); self.arbol.setHeaderLabels(['Medida', 'Valor', 'Hora y detalle']); self.arbol.setColumnWidth(0, 290)
        self.arbol.setColumnWidth(1, 150); self.arbol.setAlternatingRowColors(True)
        self.arbol.itemClicked.connect(self._clic); self.arbol.itemDoubleClicked.connect(self._editar)
        self.arbol.itemChanged.connect(self._check)
        self.arbol.setContextMenuPolicy(Qt.CustomContextMenu); self.arbol.customContextMenuRequested.connect(self._menu)
        il.addWidget(self.arbol, 1)
        il.addWidget(QLabel('Doble clic en un valor para cambiarlo (queda sombreado); clic derecho para restaurarlo. Los valores azules llevan al ECG. '
                            'La casilla de cada grupo decide si va al informe.'))
        self.extra = QLabel(''); self.extra.setWordWrap(True); self.extra.setTextFormat(Qt.RichText)
        sp.addWidget(izq)
        der = QWidget(); f = QFormLayout(der)
        self.diag = QPlainTextEdit(); self.diag.setMaximumHeight(80); self.diag.setPlaceholderText('Hasta unas 4 líneas')
        self.notas = QPlainTextEdit(); self.notas.setMaximumHeight(70)
        self.concl = QPlainTextEdit(); self.concl.setPlaceholderText('Conclusiones del médico')
        f.addRow('Diagnóstico', self.diag); f.addRow('Notas', self.notas); f.addRow('Conclusiones', self.concl)
        bn = QPushButton('Escribir el resumen narrativo en las conclusiones'); bn.clicked.connect(self._narrativo); f.addRow('', bn)
        br = QPushButton('Restaurar todos los valores cambiados'); br.clicked.connect(self._restaurar_todo); f.addRow('', br)
        for w in (self.diag, self.notas, self.concl):
            w.textChanged.connect(self._textos)
        sp.addWidget(der); sp.setSizes([700, 420])
        self._cargando = False

    def mostrar(self, ex):
        self.ex = ex; self._cargando = True
        ses = ex.sesion.d if getattr(ex, 'sesion', None) else {}
        p = ses.get('paciente') or {}
        nombre = ' '.join(x for x in (p.get('nombre', ''), p.get('apellido', '')) if x) or ex.nombre
        self.titulo.setText(f'{nombre} · {ex.fuente} · {ex.nsig} canales a {_f(ex.fs)} Hz · inicio {ex.reloj(0)} · '
                            f'{hms(ex.dur)} · motor {ex.A.get("motor", "")} · estado: {ses.get("estado", "Adquirido")}')
        ov = ses.get('resumen_override', {})
        abiertos = {self.arbol.topLevelItem(i).data(0, Qt.UserRole): self.arbol.topLevelItem(i).isExpanded() for i in range(self.arbol.topLevelItemCount())}
        self.arbol.clear()
        sombra = QBrush(QColor('#fff4c2'))
        for g, tit, fs, on in grupos_finales(ex):
            top = QTreeWidgetItem([tit]); top.setData(0, Qt.UserRole, g)
            top.setFlags(top.flags() | Qt.ItemIsUserCheckable); top.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
            fnt = top.font(0); fnt.setBold(True); top.setFont(0, fnt)
            for k, lab, v, t, nota in fs:
                it = QTreeWidgetItem([lab, v, ((ex.reloj(t) + '  ') if t is not None else '') + (nota or '')])
                it.setData(0, Qt.UserRole, ('fila', k, t)); it.setTextAlignment(1, Qt.AlignRight | Qt.AlignVCenter)
                if k in ov:
                    for j in range(3):
                        it.setBackground(j, sombra)
                if t is not None:
                    it.setForeground(1, QBrush(QColor('#1f5fbf'))); it.setForeground(2, QBrush(QColor('#1f5fbf')))
                    it.setToolTip(1, 'Clic para ver el ECG en ese momento')
                top.addChild(it)
            self.arbol.addTopLevelItem(top); top.setExpanded(abiertos.get(g, True))
        self.diag.setPlainText(ses.get('diagnostico', '')); self.notas.setPlainText(ses.get('notas', '')); self.concl.setPlainText(ses.get('conclusiones', ''))
        self._cargando = False

    def html(self, ex):
        return f'<h2>{ex.nombre}</h2>' + html_resumen(ex)

    def _clic(self, it, col):
        d = it.data(0, Qt.UserRole)
        if isinstance(d, tuple) and d[2] is not None:
            b = self.ex.latido_cercano(d[2], 0.6)
            self.ir.emit(float(d[2]), b)

    def _editar(self, it, col):
        d = it.data(0, Qt.UserRole)
        if not isinstance(d, tuple):
            return
        txt, ok = QInputDialog.getText(self, 'Resumen', f'Valor de «{it.text(0)}»:', text=it.text(1))
        if ok:
            self.ex.sesion.d['resumen_override'][d[1]] = txt; self.mostrar(self.ex); self.cambio.emit()

    def _menu(self, pos):
        it = self.arbol.itemAt(pos)
        if not it:
            return
        d = it.data(0, Qt.UserRole); m = QMenu(self)
        if isinstance(d, tuple):
            m.addAction('Cambiar el valor…', lambda: self._editar(it, 1))
            a = m.addAction('Restaurar', lambda: self._restaurar(d[1])); a.setEnabled(d[1] in self.ex.sesion.d['resumen_override'])
        m.exec(self.arbol.viewport().mapToGlobal(pos))

    def _restaurar(self, k):
        self.ex.sesion.d['resumen_override'].pop(k, None); self.mostrar(self.ex); self.cambio.emit()

    def _restaurar_todo(self):
        if self.ex:
            self.ex.sesion.d['resumen_override'] = {}; self.mostrar(self.ex); self.cambio.emit()

    def _check(self, it, col):
        if self._cargando or not self.ex or it.parent() is not None:
            return
        g = it.data(0, Qt.UserRole); off = self.ex.sesion.d['resumen_grupos_off']
        if it.checkState(0) == Qt.Checked and g in off:
            off.remove(g)
        elif it.checkState(0) != Qt.Checked and g not in off:
            off.append(g)
        self.cambio.emit()

    def _textos(self):
        if self._cargando or not self.ex:
            return
        d = self.ex.sesion.d
        d['diagnostico'] = self.diag.toPlainText(); d['notas'] = self.notas.toPlainText(); d['conclusiones'] = self.concl.toPlainText()
        self.cambio.emit()

    def _narrativo(self):
        if not self.ex:
            return
        from PySide6.QtCore import QSettings
        plantilla = QSettings('HolterLocal', 'HolterLocal').value('narrativa', '') or None
        txt = narrativo(self.ex, plantilla)
        actual = self.concl.toPlainText().strip()
        self.concl.setPlainText((actual + '\n\n' if actual else '') + txt)
