"""Pestaña Tiras: lista de tiras de ECG del informe, vista previa, edición, orden y tiras automáticas."""
import itertools

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QScrollArea, QSpinBox,
                               QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from .motor import VENTRICULAR
from .tira import dibujar_ecg

_ids = itertools.count(int(__import__('time').time() * 1000) % 10 ** 9)

ANOTACIONES = ['Ritmo sinusal', 'Ritmo de base', 'Extrasístole ventricular', 'Par ventricular', 'Taquicardia ventricular',
               'Extrasístole supraventricular', 'Taquicardia supraventricular', 'Fibrilación auricular', 'Flutter auricular',
               'Pausa', 'Bloqueo AV', 'Bradicardia', 'Taquicardia', 'Frecuencia máxima', 'Frecuencia mínima', 'Artefacto',
               'Síntoma del paciente', 'Marcapasos']

MINMAX = [('hrmax', 'Frecuencia cardíaca máxima'), ('hrmin', 'Frecuencia cardíaca mínima'), ('tachy_long', 'Taquicardia más larga'),
          ('tachy_fast', 'Taquicardia más rápida'), ('brady_long', 'Bradicardia más larga'), ('brady_slow', 'Bradicardia más lenta'),
          ('vrun_long', 'Racha ventricular más larga'), ('vrun_fast', 'Racha ventricular más rápida'), ('srun_long', 'Racha supraventricular más larga'),
          ('srun_fast', 'Racha supraventricular más rápida'), ('rr_long', 'RR más largo'), ('pause_long', 'Pausa más larga'), ('rr_short', 'RR más corto')]

EVENTOS_AUTO = [
    ('Ventriculares', [('vIso', 'Ventricular aislada'), ('vPair', 'Par ventricular'), ('vRun', 'Racha ventricular'), ('vBig', 'Bigeminismo ventricular'),
                       ('rOnT', 'R sobre T'), ('vTrig', 'Trigeminismo ventricular'), ('esc', 'Latido de escape'), ('interp', 'Interpolado')]),
    ('Supraventriculares', [('sIso', 'Supraventricular aislada'), ('sPair', 'Par supraventricular'), ('sRun', 'Racha supraventricular'),
                            ('sBig', 'Bigeminismo supraventricular'), ('af', 'Fibrilación auricular'), ('sTrig', 'Trigeminismo supraventricular'),
                            ('aber', 'Latido aberrante')]),
    ('Ritmo y ST', [('stDep', 'Descenso del ST'), ('stEle', 'Elevación del ST'), ('brady', 'Bradicardia'), ('tachy', 'Taquicardia'),
                    ('fus', 'Latido de fusión'), ('brb', 'Latido con bloqueo de rama'), ('desc', 'Latido desconocido'), ('pause', 'Pausa'),
                    ('avb', 'Bloqueo AV (holterai)')]),
    ('Estimulados', [('pa', 'Auricular estimulado'), ('pv', 'Ventricular estimulado'), ('pd', 'Estimulado doble'), ('pfc', 'Fallo de captura'),
                     ('pfs', 'Fallo de sensado')]),
    ('Del usuario', [('u1', 'Suceso del usuario 1'), ('u2', 'Suceso del usuario 2'), ('u3', 'Suceso del usuario 3')]),
]
EV_TIPOS = {'vPair': ['vPair'], 'vRun': ['vt', 'aivr', 'ivr'], 'vBig': ['vBig'], 'rOnT': ['rOnT'], 'vTrig': ['vTrig'], 'interp': ['interp'],
            'sPair': ['sPair'], 'sRun': ['svt', 'sRun'], 'sBig': ['sBig'], 'af': ['af'], 'sTrig': ['sTrig'], 'stDep': ['stDep'], 'stEle': ['stEle'],
            'brady': ['brady'], 'tachy': ['tachy'], 'pause': ['pause'], 'pfc': ['ha_pacing_failure'], 'pfs': ['ha_pacing_sensing'],
            'u1': ['u1'], 'u2': ['u2'], 'u3': ['u3'],
            'avb': ['ha_av_block_1', 'ha_av_block_mobitz1', 'ha_av_block_mobitz2', 'ha_av_block_2to1', 'ha_av_block_advanced', 'ha_av_block_complete']}
EV_CLASES = {'esc': [8], 'aber': [7], 'fus': [3], 'brb': [6], 'desc': [5], 'pa': [12], 'pv': [9], 'pd': [13]}
# Planilla de tiras pedida por Jorge, en su orden. 'uno' = un solo episodio (el más extremo); 'varios' = cantidad elegible.
PLANILLA = [('normal', 'Normal (ritmo sinusal de base)', 'uno'), ('hrmax', 'Frecuencia máxima', 'uno'), ('hrmin', 'Frecuencia mínima', 'uno'),
            ('vIso', 'Extrasístole ventricular', 'varios'), ('vPair', 'Par de extrasístoles ventriculares', 'varios'),
            ('vTrip', 'Tripleta de extrasístoles ventriculares', 'varios'), ('vt', 'Taquicardia ventricular', 'varios'),
            ('vt_fast', 'Taquicardia ventricular más rápida', 'uno'), ('vt_long', 'Taquicardia ventricular más larga', 'uno'),
            ('sIso', 'Extrasístole supraventricular', 'varios'), ('sPair', 'Par de extrasístoles supraventriculares', 'varios'),
            ('svt_fast', 'Taquicardia supraventricular más rápida', 'uno'), ('svt_long', 'Taquicardia supraventricular más larga', 'uno'),
            ('avb23', 'Bloqueo AV de segundo y tercer grado', 'varios')]
AVB_23 = ['ha_av_block_mobitz1', 'ha_av_block_mobitz2', 'ha_av_block_2to1', 'ha_av_block_advanced', 'ha_av_block_complete']
POR_DEFECTO = {'chans': [0, 1, 2], 'skip_fail': True, 'diario': True, 'periodicas': False, 'offset': '00:00:00', 'intervalo': '01:00:00',
               'planilla': {k: [True, 1 if modo == 'uno' else 2, 'examen'] for k, _, modo in PLANILLA},
               'minmax': {k: True for k, _ in MINMAX},
               'eventos': {k: [k in ('vPair', 'vRun', 'sRun', 'af', 'pause', 'avb', 'pfc', 'u1', 'u2', 'u3'), 3, 'examen']
                           for _, g in EVENTOS_AUTO for k, _ in g}}


def nueva_tira(t0, dur=7.5, ann='', chans=None, auto=False, clave='', pagina=0, canal=0):
    return {'id': next(_ids), 't0': float(max(0.0, t0)), 'dur': float(dur), 'ann': ann, 'auto': bool(auto), 'clave': clave,
            'chans': list(chans if chans is not None else [0, 1, 2]), 'pagina': int(pagina), 'canal': int(canal)}


def anotacion_en(ex, t0, dur):
    """Texto sugerido para una tira: el suceso más importante que hay dentro."""
    orden = ['vt', 'svt', 'af', 'pause', 'vPair', 'sPair', 'vBig', 'sBig', 'brady', 'tachy', 'stDep', 'stEle']
    dentro = [e for e in ex.eventos if e['t0'] < t0 + dur and e['t1'] >= t0]
    for ty in orden:
        for e in dentro:
            if e['type'] == ty:
                return e.get('label') or __import__('holterlocal.motor', fromlist=['EVENT_LABEL']).EVENT_LABEL.get(ty, ty)
    for e in dentro:
        if str(e['type']).startswith('ha_'):
            return e.get('label', '')
    b0, b1 = np.searchsorted(ex.t, [t0, t0 + dur])
    seg = ex.cls[b0:b1]
    if np.isin(seg, VENTRICULAR).any():
        return 'Extrasístole ventricular'
    if (seg == 1).any():
        return 'Extrasístole supraventricular'
    return 'Ritmo de base'


def _cands_minmax(ex, k):
    by = ex.por_tipo(); A = ex.A
    def best(tipos, campo, mayor=True):
        evs = [e for ty in tipos for e in by.get(ty, []) if e.get(campo) is not None]
        if not evs:
            return None
        e = max(evs, key=lambda e: e[campo]) if mayor else min(evs, key=lambda e: e[campo])
        return float(e['t0']), e
    hr = A.get('hr') or {}
    if k in ('hrmax', 'hrmin'):
        b = hr.get('maxB' if k == 'hrmax' else 'minB')
        if b is not None and 0 <= b < len(ex.t):
            v = hr.get('max' if k == 'hrmax' else 'min')
            return float(ex.t[b]) - 3.75, 'FC %s %d lpm' % ('máxima' if k == 'hrmax' else 'mínima', round(v or 0))
        return None
    if k in ('rr_long', 'rr_short'):
        b = A.get('rrMaxB' if k == 'rr_long' else 'rrMinB')
        if b is None:
            ok = np.isfinite(ex.rr)
            if not ok.any():
                return None
            b = int(np.nanargmin(np.where(ok, ex.rr, np.nan)))
        if 0 <= b < len(ex.t):
            return float(ex.t[b]) - 3.75, 'RR más %s %d ms' % ('largo' if k == 'rr_long' else 'corto', round(ex.rr[b] * 1000) if ex.rr[b] == ex.rr[b] else 0)
        return None
    tabla = {'tachy_long': (['tachy'], 'dur', True, 'Taquicardia más larga'), 'tachy_fast': (['tachy'], 'rate', True, 'Taquicardia más rápida'),
             'brady_long': (['brady'], 'dur', True, 'Bradicardia más larga'), 'brady_slow': (['brady'], 'rate', False, 'Bradicardia más lenta'),
             'vrun_long': (['vt', 'aivr', 'ivr'], 'n', True, 'Racha ventricular más larga'), 'vrun_fast': (['vt', 'aivr', 'ivr'], 'rate', True, 'Racha ventricular más rápida'),
             'srun_long': (['svt', 'sRun'], 'n', True, 'Racha supraventricular más larga'), 'srun_fast': (['svt', 'sRun'], 'rate', True, 'Racha supraventricular más rápida'),
             'pause_long': (['pause'], 'rr', True, 'Pausa más larga')}
    if k not in tabla:
        return None
    tipos, campo, mayor, nombre = tabla[k]
    r = best(tipos, campo, mayor)
    if not r:
        return None
    t, e = r
    extra = ' %d lpm' % round(e['rate']) if e.get('rate') else (' %.2f s' % e['rr'] if campo == 'rr' else '')
    return t - 1.5, nombre + extra.replace('.', ',')


def _cands_evento(ex, k):
    out = []
    if k in EV_TIPOS:
        for e in ex.eventos:
            if e['type'] in EV_TIPOS[k]:
                out.append((float(e['t0']), e.get('label') or ''))
    elif k in EV_CLASES:
        idx = np.nonzero(np.isin(ex.cls, EV_CLASES[k]))[0]
        out = [(float(ex.t[b]), '') for b in idx]
    elif k in ('vIso', 'sIso'):
        cl = np.isin(ex.cls, VENTRICULAR) if k == 'vIso' else ex.cls == 1
        prev = np.r_[False, cl[:-1]]; nxt = np.r_[cl[1:], False]
        idx = np.nonzero(cl & ~prev & ~nxt)[0]
        out = [(float(ex.t[b]), '') for b in idx]
    return out


def _cand_normal(ex):
    """Tramo de 7,5 s solo con latidos normales, con los canales limpios y sin exclusiones, de día si se puede,
    con la FC más cercana a la mediana del registro."""
    n = len(ex.t)
    if n < 10:
        return None
    vent = 7.5
    okN = (np.asarray(ex.cls) == 0)
    rr = np.asarray(ex.rr, float)
    rrN = rr[okN & np.isfinite(rr) & (rr > 0.25)]
    if not len(rrN):
        return None
    med = 60.0 / float(np.median(rrN))
    nseg = int(ex.dur) + 1
    malo = np.zeros(nseg, bool)
    for q in ex.qual:
        q = np.asarray(q)[:nseg]; malo[:len(q)] |= q != 0
    x = np.asarray(ex.excl)[:nseg]; malo[:len(x)] |= x != 0
    cm = np.r_[0, np.cumsum(malo)]
    cn = np.r_[0, np.cumsum(~okN)]
    t0s = np.arange(5.0, max(5.0, ex.dur - vent - 5.0), vent)
    if not len(t0s):
        return None
    b0 = np.searchsorted(ex.t, t0s); b1 = np.searchsorted(ex.t, t0s + vent)
    s0 = t0s.astype(int); s1 = np.minimum(nseg, (t0s + vent).astype(int) + 1)
    limpio = ((b1 - b0) >= 4) & ((cn[b1] - cn[b0]) == 0) & ((cm[s1] - cm[s0]) == 0)
    if not limpio.any():
        return None
    hr = np.full(len(t0s), np.nan)
    for i in np.nonzero(limpio)[0]:
        r = rr[b0[i] + 1:b1[i]]; r = r[np.isfinite(r) & (r > 0.25)]
        if len(r):
            hr[i] = 60.0 / r.mean()
    pen = np.abs(hr - med)
    try:
        from .darwin_comun import es_noche
        pen = pen + np.where(es_noche(ex, t0s), 1000.0, 0.0)
    except Exception:
        pass
    if not np.isfinite(pen).any():
        return None
    i = int(np.nanargmin(pen))
    return float(t0s[i]), 'Ritmo de base %d lpm' % round(hr[i])


def _cands_planilla(ex, k):
    """Lista de (inicio de la tira, anotación) para un elemento de la planilla."""
    if k == 'normal':
        r = _cand_normal(ex); return [r] if r else []
    if k in ('hrmax', 'hrmin'):
        r = _cands_minmax(ex, k); return [r] if r else []
    if k in ('vIso', 'sIso'):
        return [(t - 2.0, '') for t, _ in _cands_evento(ex, k)]
    tipos = {'vPair': ['vPair'], 'sPair': ['sPair'], 'vt': ['vt'], 'vt_fast': ['vt'], 'vt_long': ['vt'],
             'svt_fast': ['svt', 'sRun'], 'svt_long': ['svt', 'sRun'], 'avb23': AVB_23, 'vTrip': ['vt', 'aivr', 'ivr']}.get(k)
    if not tipos:
        return []
    evs = [e for e in ex.eventos if e['type'] in tipos]
    if k == 'vTrip':
        evs = [e for e in evs if int(e.get('n') or 0) == 3]
    def lpm(e):
        return ' %d lpm' % round(e['rate']) if e.get('rate') else ''
    if k.endswith('_fast') or k.endswith('_long'):
        campo = 'rate' if k.endswith('_fast') else 'n'
        evs = [e for e in evs if e.get(campo) is not None]
        if not evs:
            return []
        e = max(evs, key=lambda e: (e[campo], e.get('dur') or 0))
        nom = dict((a, b) for a, b, _ in PLANILLA)[k]
        extra = lpm(e) if campo == 'rate' else ' %d latidos' % int(e['n']) + lpm(e)
        return [(float(e['t0']) - 1.5, nom + extra)]
    out = []
    for e in evs:
        lab = e.get('label') or ''
        if k in ('vt', 'vTrip') and e.get('n'):
            lab = ('Taquicardia ventricular' if k == 'vt' else 'Tripleta ventricular') + ' %d latidos' % int(e['n']) + lpm(e)
        out.append((float(e['t0']) - 1.5, lab))
    return out


def generar_auto(ex, op):
    """Tiras automáticas según las opciones (episodios mín./máx., sucesos por examen / 24 h / hora, diario y periódicas)."""
    chans = [c for c in op.get('chans', [0, 1, 2]) if c < len(ex.chans)] or [0]
    tiras = []
    pl = dict(POR_DEFECTO['planilla']); pl.update(op.get('planilla') or {})
    noms = {k: n for k, n, _ in PLANILLA}
    for k, _, modo in PLANILLA:
        on, cant, por = pl.get(k, [False, 1, 'examen'])
        if not on:
            continue
        grupo = {'examen': lambda t: 0, '24h': lambda t: int(t // 86400), 'hora': lambda t: int(t // 3600)}.get(por, lambda t: 0)
        cuenta = {}
        for t, lab in sorted(_cands_planilla(ex, k)):
            g = grupo(t)
            lim = 1 if modo == 'uno' else (10 ** 6 if cant == 'todos' else int(cant))
            if cuenta.get(g, 0) >= lim:
                continue
            cuenta[g] = cuenta.get(g, 0) + 1
            tiras.append(nueva_tira(t, 7.5, lab or noms[k], chans, True, 'pl:' + k))
            if len(tiras) > 600:
                break
    for k, n in MINMAX:
        if op['minmax'].get(k):
            r = _cands_minmax(ex, k)
            if r:
                tiras.append(nueva_tira(r[0], 7.5, r[1], chans, True, 'mm:' + k))
    nombres = {k: n for _, g in EVENTOS_AUTO for k, n in g}
    for k, (on, cant, por) in op['eventos'].items():
        if not on:
            continue
        cands = sorted(_cands_evento(ex, k))
        grupo = {'examen': lambda t: 0, '24h': lambda t: int(t // 86400), 'hora': lambda t: int(t // 3600)}[por]
        cuenta = {}
        for t, lab in cands:
            g = grupo(t)
            if cant != 'todos' and cuenta.get(g, 0) >= int(cant):
                continue
            cuenta[g] = cuenta.get(g, 0) + 1
            tiras.append(nueva_tira(t - 2.0, 7.5, lab or nombres.get(k, k), chans, True, 'ev:' + k))
            if len(tiras) > 600:
                break
    if op.get('diario') and getattr(ex, 'sesion', None):
        for d in ex.sesion.d.get('diario', []):
            tiras.append(nueva_tira(float(d['t']) - 3.75, 7.5, 'Diario: ' + d.get('texto', ''), chans, True, 'diario'))
    if op.get('periodicas'):
        a = sum(int(x) * m for x, m in zip(op.get('offset', '0:0:0').split(':'), (3600, 60, 1))) if op.get('offset') else 0
        paso = sum(int(x) * m for x, m in zip(op.get('intervalo', '1:0:0').split(':'), (3600, 60, 1)))
        t = float(a)
        while paso >= 10 and t < ex.dur - 7.5 and len(tiras) < 800:
            malo = False
            if op.get('skip_fail'):
                s0, s1 = int(t), int(t + 7.5) + 1
                malo = all(np.mean(np.asarray(ex.qual[c][s0:s1]) == 2) > 0.5 for c in range(len(ex.chans)))
            if not malo:
                tiras.append(nueva_tira(t, 7.5, 'Tira periódica', chans, True, 'periodica'))
            t += paso
    tiras.sort(key=lambda x: x['t0'])
    # la planilla y las listas antiguas pueden pedir el mismo episodio: se deja una sola tira (la primera pedida)
    unicas = []
    for x in sorted(tiras, key=lambda x: (not str(x['clave']).startswith(('pl:', 'mm:')), x['t0'])):
        if not any(abs(u['t0'] - x['t0']) < 1.0 and u['chans'] == x['chans'] and not u.get('pagina') for u in unicas):
            unicas.append(x)
    unicas.sort(key=lambda x: x['t0'])
    return unicas


# --------------------------------------------------------------------------- diálogos
class DialogoTira(QDialog):
    def __init__(self, ex, tira, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Tira para el informe'); self.ex = ex; self.tira = dict(tira)
        f = QFormLayout(self)
        self.hora = QLineEdit(ex.reloj(tira['t0']) + (' (%d)' % (int((ex.seg_inicio() + tira['t0']) // 86400) + 1) if tira['t0'] + ex.seg_inicio() >= 86400 else ''))
        f.addRow('Hora de inicio', self.hora)
        self.ann = QComboBox(); self.ann.setEditable(True); self.ann.addItems([tira.get('ann') or anotacion_en(ex, tira['t0'], tira['dur'])] + ANOTACIONES)
        f.addRow('Anotación', self.ann)
        if tira.get('pagina'):
            self.canal = QComboBox()
            for i, c in enumerate(ex.chans):
                self.canal.addItem(ex.names[c] if c < len(ex.names) else f'Canal {i + 1}', i)
            self.canal.setCurrentIndex(min(tira.get('canal', 0), len(ex.chans) - 1))
            self.min = QSpinBox(); self.min.setRange(5, 60); self.min.setValue(tira['pagina']); self.min.setSuffix(' min')
            f.addRow('Canal', self.canal); f.addRow('Minutos por página', self.min)
        else:
            self.dur = QComboBox()
            for v in (7.5, 15, 22.5, 30, 37.5, 45, 52.5, 60):
                self.dur.addItem(('%g s' % v).replace('.', ','), v)
            i = self.dur.findData(tira['dur']); self.dur.setCurrentIndex(max(0, i))
            f.addRow('Duración', self.dur)
            fila = QHBoxLayout(); self.ch = []
            for i, c in enumerate(ex.chans):
                cb = QCheckBox(ex.names[c] if c < len(ex.names) else f'Canal {i + 1}'); cb.setChecked(i in tira.get('chans', [0, 1, 2])); self.ch.append(cb); fila.addWidget(cb)
            f.addRow('Canales', fila)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); bb.accepted.connect(self.accept); bb.rejected.connect(self.reject); f.addRow(bb)

    def valor(self):
        t = self.ex.parse_reloj(self.hora.text())
        if t is not None:
            self.tira['t0'] = t
        self.tira['ann'] = self.ann.currentText()
        if self.tira.get('pagina'):
            self.tira['canal'] = self.canal.currentData(); self.tira['pagina'] = self.min.value(); self.tira['dur'] = self.min.value() * 60.0
        else:
            self.tira['dur'] = self.dur.currentData(); self.tira['chans'] = [i for i, c in enumerate(self.ch) if c.isChecked()] or [0]
        return self.tira


class DialogoTirasAuto(QDialog):
    def __init__(self, ex, op, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Tiras automáticas'); self.ex = ex; self.resize(820, 620)
        lay = QVBoxLayout(self)
        top = QHBoxLayout(); top.addWidget(QLabel('Canales'))
        self.ch = []
        for i, c in enumerate(ex.chans):
            cb = QCheckBox(ex.names[c] if c < len(ex.names) else f'Canal {i + 1}'); cb.setChecked(i in op.get('chans', [0, 1, 2])); self.ch.append(cb); top.addWidget(cb)
        top.addSpacing(16)
        self.skip = QCheckBox('Omitir tiras periódicas con fallo de electrodos'); self.skip.setChecked(op.get('skip_fail', True)); top.addWidget(self.skip)
        self.diario = QCheckBox('Tiras de los sucesos del diario'); self.diario.setChecked(op.get('diario', True)); top.addWidget(self.diario)
        top.addStretch(1); lay.addLayout(top)
        per = QHBoxLayout()
        self.per = QCheckBox('Tiras periódicas automáticas'); self.per.setChecked(op.get('periodicas', False))
        self.off = QLineEdit(op.get('offset', '00:00:00')); self.off.setMaximumWidth(90)
        self.iv = QLineEdit(op.get('intervalo', '01:00:00')); self.iv.setMaximumWidth(90)
        for w in (self.per, QLabel('desde el inicio (HH:MM:SS)'), self.off, QLabel('cada'), self.iv):
            per.addWidget(w)
        per.addStretch(1); lay.addLayout(per)
        tabs = QTabWidget(); lay.addWidget(tabs, 1)
        w = QWidget(); g = QGridLayout(w); self.pl = {}
        pl = dict(POR_DEFECTO['planilla']); pl.update(op.get('planilla') or {})
        for i, (k, n, modo) in enumerate(PLANILLA):
            on, cant, por = pl.get(k, [True, 1, 'examen'])
            cb = QCheckBox(n); cb.setChecked(on); g.addWidget(cb, i, 0)
            cc = QComboBox(); pp = QComboBox()
            if modo == 'varios':
                cc.addItem('Imprimir todo', 'todos')
                for v in (1, 2, 3, 5, 10, 20, 50, 100):
                    cc.addItem(str(v), v)
                cc.setCurrentIndex(max(0, cc.findData(cant)))
                [pp.addItem(a, b) for a, b in (('por examen', 'examen'), ('cada 24 horas', '24h'), ('cada hora', 'hora'))]
                pp.setCurrentIndex(max(0, pp.findData(por)))
                g.addWidget(cc, i, 1); g.addWidget(pp, i, 2)
            else:
                cc.addItem('1', 1); pp.addItem('por examen', 'examen')
                g.addWidget(QLabel('una tira representativa, de día' if k == 'normal' else 'el episodio más extremo'), i, 1, 1, 2)
            self.pl[k] = (cb, cc, pp)
        nota = QLabel('Si otra lista pide el mismo episodio, sale una sola tira. El bloqueo AV de 2.º y 3.er grado viene de holterai.')
        nota.setWordWrap(True); g.addWidget(nota, len(PLANILLA), 0, 1, 4)
        g.setRowStretch(30, 1); g.setColumnStretch(3, 1); tabs.addTab(w, 'Planilla del informe')
        w = QWidget(); g = QGridLayout(w); self.mm = {}
        for i, (k, n) in enumerate(MINMAX):
            cb = QCheckBox(n); cb.setChecked(op['minmax'].get(k, True)); self.mm[k] = cb; g.addWidget(cb, i // 2, i % 2)
        g.setRowStretch(20, 1); tabs.addTab(w, 'Episodios mín./máx.')
        self.ev = {}
        for grupo, items in EVENTOS_AUTO:
            w = QWidget(); g = QGridLayout(w)
            for i, (k, n) in enumerate(items):
                on, cant, por = op['eventos'].get(k, [False, 3, 'examen'])
                cb = QCheckBox(n); cb.setChecked(on)
                cc = QComboBox(); cc.addItem('Imprimir todo', 'todos')
                for v in (1, 2, 3, 5, 10, 20, 50, 100):
                    cc.addItem(str(v), v)
                j = cc.findData(cant); cc.setCurrentIndex(max(0, j))
                pp = QComboBox(); [pp.addItem(a, b) for a, b in (('por examen', 'examen'), ('cada 24 horas', '24h'), ('cada hora', 'hora'))]
                pp.setCurrentIndex(max(0, pp.findData(por)))
                g.addWidget(cb, i, 0); g.addWidget(cc, i, 1); g.addWidget(pp, i, 2); self.ev[k] = (cb, cc, pp)
            g.setRowStretch(20, 1); g.setColumnStretch(3, 1); tabs.addTab(w, grupo)
        lay.addWidget(QLabel('Al aceptar se reemplazan todas las tiras automáticas; las que agregaste a mano no se tocan.'))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); bb.accepted.connect(self.accept); bb.rejected.connect(self.reject); lay.addWidget(bb)

    def opciones(self):
        return {'chans': [i for i, c in enumerate(self.ch) if c.isChecked()] or [0], 'skip_fail': self.skip.isChecked(), 'diario': self.diario.isChecked(),
                'periodicas': self.per.isChecked(), 'offset': self.off.text().strip() or '00:00:00', 'intervalo': self.iv.text().strip() or '01:00:00',
                'planilla': {k: [cb.isChecked(), cc.currentData(), pp.currentData()] for k, (cb, cc, pp) in self.pl.items()},
                'minmax': {k: c.isChecked() for k, c in self.mm.items()},
                'eventos': {k: [cb.isChecked(), cc.currentData(), pp.currentData()] for k, (cb, cc, pp) in self.ev.items()}}


# --------------------------------------------------------------------------- pestaña
class Previa(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent); self.ex = None; self.tira = None; self.setMinimumWidth(320)

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing, True)
        if not self.ex or not self.tira:
            p.drawText(self.rect(), Qt.AlignCenter, 'Elige una tira'); return
        t = self.tira
        if t.get('pagina'):
            dibujar_ecg(p, QRectF(0, 0, self.width(), self.height()), self.ex, t['t0'], min(60.0, t['dur']), [self.ex.chans[t.get('canal', 0)]], etiquetas=False)
        else:
            chs = [self.ex.chans[c] for c in t['chans'] if c < len(self.ex.chans)] or [self.ex.chans[0]]
            dibujar_ecg(p, QRectF(0, 0, self.width(), self.height()), self.ex, t['t0'], t['dur'], chs)
        p.end()


class VistaTiras(QWidget):
    ir = Signal(float, int)
    cambio = Signal()
    excluir = Signal(float, float)
    pedir_auto = Signal()

    COLS = ['Hora', 'Anotación', 'Auto', 'Duración (s)', 'Canales']

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None
        lay = QHBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        sp = QSplitter(Qt.Horizontal); lay.addWidget(sp)
        izq = QWidget(); il = QVBoxLayout(izq); il.setContentsMargins(0, 0, 0, 0)
        self.tabla = QTableWidget(0, len(self.COLS)); self.tabla.setHorizontalHeaderLabels(self.COLS)
        self.tabla.setSelectionBehavior(QAbstractItemView.SelectRows); self.tabla.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tabla.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch); self.tabla.verticalHeader().setVisible(False)
        self.tabla.horizontalHeader().sectionClicked.connect(self._ordenar)
        self.tabla.currentCellChanged.connect(lambda r, *_: self._previa(r))
        self.tabla.cellDoubleClicked.connect(lambda r, c: self._ir(r))
        il.addWidget(self.tabla, 1)
        bot = QHBoxLayout()
        for txt, fn in (('Editar…', self._editar), ('Borrar', self._borrar), ('Subir', lambda: self._mover(-1)), ('Bajar', lambda: self._mover(1)),
                        ('Artefacto', self._artefacto), ('Agregar automáticas…', self.pedir_auto.emit), ('Borrar automáticas', self._sin_auto)):
            b = QPushButton(txt); b.clicked.connect(fn); bot.addWidget(b)
        bot.addStretch(1); il.addLayout(bot)
        self.cuenta = QLabel(''); self.cuenta.setObjectName('ayuda'); il.addWidget(self.cuenta)
        sp.addWidget(izq)
        self.previa = Previa(); sp.addWidget(self.previa); sp.setSizes([560, 520])

    def lista(self):
        return self.ex.sesion.d['tiras'] if self.ex and getattr(self.ex, 'sesion', None) else []

    def mostrar(self, ex):
        self.ex = ex; self.previa.ex = ex
        L = self.lista(); tb = self.tabla
        r0 = tb.currentRow()
        tb.setRowCount(len(L))
        for i, t in enumerate(L):
            dia = int((ex.seg_inicio() + t['t0']) // 86400)
            hora = ex.reloj(t['t0']).split(' ')[-1] + (f' ({dia + 1})' if dia else '')
            can = ('canal %d' % (t.get('canal', 0) + 1)) if t.get('pagina') else ', '.join(str(c + 1) for c in t['chans'])
            dur = ('%d min' % t['pagina']) if t.get('pagina') else ('%g' % t['dur']).replace('.', ',')
            for j, v in enumerate([hora, t.get('ann', ''), 'S' if t.get('auto') else '', dur, can]):
                tb.setItem(i, j, QTableWidgetItem(v))
        tb.resizeColumnToContents(0)
        if 0 <= r0 < len(L):
            tb.selectRow(r0)
        n_auto = sum(1 for t in L if t.get('auto'))
        self.cuenta.setText(f'{len(L)} tiras ({n_auto} automáticas). El orden de la lista es el orden en que se imprimen. Clic en un encabezado para ordenar.')
        self._previa(tb.currentRow())

    def _previa(self, r):
        L = self.lista()
        self.previa.tira = L[r] if 0 <= r < len(L) else None; self.previa.update()

    def _ir(self, r):
        L = self.lista()
        if 0 <= r < len(L):
            self.ir.emit(L[r]['t0'] + min(L[r]['dur'], 7.5) / 2, -1)

    def _ordenar(self, col):
        L = self.lista()
        clave = [lambda t: t['t0'], lambda t: t.get('ann', ''), lambda t: not t.get('auto'), lambda t: t['dur'], lambda t: str(t['chans'])][col]
        L.sort(key=clave); self.mostrar(self.ex); self.cambio.emit()

    def _editar(self):
        r = self.tabla.currentRow(); L = self.lista()
        if 0 <= r < len(L):
            d = DialogoTira(self.ex, L[r], self)
            if d.exec():
                L[r] = d.valor(); L[r]['auto'] = False; self.mostrar(self.ex); self.cambio.emit()

    def _borrar(self):
        rows = sorted({i.row() for i in self.tabla.selectedIndexes()}, reverse=True); L = self.lista()
        for r in rows:
            if 0 <= r < len(L):
                L.pop(r)
        self.mostrar(self.ex); self.cambio.emit()

    def _mover(self, d):
        r = self.tabla.currentRow(); L = self.lista()
        if 0 <= r < len(L) and 0 <= r + d < len(L):
            L[r], L[r + d] = L[r + d], L[r]; self.mostrar(self.ex); self.tabla.selectRow(r + d); self.cambio.emit()

    def _artefacto(self):
        r = self.tabla.currentRow(); L = self.lista()
        if 0 <= r < len(L):
            t = L.pop(r); self.mostrar(self.ex); self.excluir.emit(t['t0'], t['t0'] + t['dur'])

    def _sin_auto(self):
        self.ex.sesion.d['tiras'] = [t for t in self.lista() if not t.get('auto')]; self.mostrar(self.ex); self.cambio.emit()
