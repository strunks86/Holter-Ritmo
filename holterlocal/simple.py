"""Modo simple: flujo guiado de cuatro pasos (Exámenes → Hallazgos → Revisar → Informe).

Muestra solo lo que el médico necesita para un Holter de ritmo. Todo lo demás sigue en el modo experto (las pestañas
de siempre), al que se pasa con un botón sin perder nada.
"""
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
                               QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton, QSplitter, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget)

from .resumen import _f, hms, html_resumen

AZUL = '#1f5fbf'
ESTADO_COLOR = {'ok': '#16a34a', 'ver': '#d97706', 'alto': '#dc2626', 'nada': '#9ca3af'}
ESTADO_TXT = {'ok': 'Sin hallazgos', 'ver': 'Revisar', 'alto': 'Importante', 'nada': 'Sin datos'}

QSS_SIMPLE = f"""
QWidget#simple {{ background: #f3f4f6; }}
QFrame#cabecera {{ background: #ffffff; border-bottom: 1px solid #d5d9e0; }}
QLabel#marca {{ font-size: 15pt; font-weight: 700; color: {AZUL}; }}
QLabel#examen {{ font-size: 10.5pt; color: #5b6573; }}
QPushButton#paso {{ border: none; border-radius: 18px; padding: 8px 18px; font-size: 11pt; color: #5b6573; background: transparent; }}
QPushButton#paso:checked {{ background: {AZUL}; color: white; font-weight: 600; }}
QPushButton#paso:hover:!checked {{ background: #e8f0fc; }}
QPushButton#experto {{ border: 1px solid #d5d9e0; border-radius: 6px; padding: 7px 14px; font-size: 10pt; }}
QFrame#tarjeta {{ background: #ffffff; border: 1px solid #d5d9e0; border-radius: 10px; }}
QFrame#tarjeta QLabel, QFrame#cabecera QLabel {{ background: transparent; }}
QLabel#t_titulo {{ font-size: 11.5pt; font-weight: 600; }}
QLabel#t_valor {{ font-size: 24pt; font-weight: 700; }}
QLabel#t_det {{ font-size: 10pt; color: #374151; }}
QPushButton#revisar {{ background: #e8f0fc; color: {AZUL}; border: none; border-radius: 6px; padding: 7px 12px; font-weight: 600; font-size: 10pt; }}
QPushButton#revisar:hover {{ background: #d4e3fa; }}
QPushButton#grande {{ background: #ffffff; border: 1px solid #d5d9e0; border-radius: 10px; padding: 18px; font-size: 12pt; text-align: left; }}
QPushButton#grande:hover {{ border-color: {AZUL}; background: #f7faff; }}
QPushButton#decide {{ border-radius: 6px; padding: 9px 16px; font-size: 10.5pt; font-weight: 600; }}
QListWidget#categorias {{ font-size: 11pt; border: none; background: #ffffff; }}
QListWidget#categorias::item {{ padding: 10px 8px; border-bottom: 1px solid #eef0f4; }}
QListWidget#categorias::item:selected {{ background: #e8f0fc; color: #1f2933; border-left: 4px solid {AZUL}; }}
QLabel#seccion {{ font-size: 13pt; font-weight: 600; }}
QLabel#nota {{ color: #5b6573; font-size: 9.5pt; }}
"""

# categorías clínicas del flujo de revisión: (clave, nombre, tipos de suceso)
CATEGORIAS = [
    ('fa', 'Fibrilación / flutter auricular', ('af', 'ha_flutter')),
    ('s', 'Supraventriculares', ('svt', 'sRun', 'sPair', 'sBig', 'sTrig')),
    ('v', 'Ventriculares', ('vt', 'aivr', 'ivr', 'vPair', 'vBig', 'vTrig', 'rOnT', 'ha_multiforme')),
    ('pausa', 'Pausas', ('pause', 'ha_asystole')),
    ('bloq', 'Bloqueos AV y sinoauriculares', ('ha_av_', 'ha_bsa')),
    ('fc', 'Taquicardia y bradicardia', ('tachy', 'brady')),
    ('art', 'Artefactos y sin señal', ('artifact', 'nosig', 'ha_no_signal')),
]

NOMBRE_TIPO = {'af': 'FA', 'ha_flutter': 'Flutter', 'svt': 'TSV', 'sRun': 'Racha S', 'sPair': 'Par S', 'sBig': 'Bigeminismo S',
               'sTrig': 'Trigeminismo S', 'vt': 'TV', 'aivr': 'RIVA', 'ivr': 'Ritmo idioventricular', 'vPair': 'Par V',
               'vBig': 'Bigeminismo V', 'vTrig': 'Trigeminismo V', 'rOnT': 'R sobre T', 'pause': 'Pausa', 'tachy': 'Taquicardia',
               'brady': 'Bradicardia', 'artifact': 'Artefacto', 'nosig': 'Sin señal'}


def de_categoria(ex, clave):
    tipos = dict((k, t) for k, _, t in CATEGORIAS)[clave]
    return sorted((e for e in ex.eventos if str(e['type']).startswith(tipos)), key=lambda e: e['t0'])


def reloj(ex, t):
    return ex.reloj(t) if hasattr(ex, 'reloj') else hms(t)


class Tarjeta(QFrame):
    revisar = Signal(str)

    def __init__(self, clave, titulo):
        super().__init__(); self.setObjectName('tarjeta'); self.clave = clave
        lay = QVBoxLayout(self); lay.setContentsMargins(16, 12, 16, 14); lay.setSpacing(4)
        top = QHBoxLayout(); self.tit = QLabel(titulo); self.tit.setObjectName('t_titulo')
        self.est = QLabel(''); top.addWidget(self.tit); top.addStretch(1); top.addWidget(self.est); lay.addLayout(top)
        self.valor = QLabel('—'); self.valor.setObjectName('t_valor'); lay.addWidget(self.valor)
        self.det = QLabel(''); self.det.setObjectName('t_det'); self.det.setWordWrap(True); self.det.setTextFormat(Qt.RichText)
        lay.addWidget(self.det); lay.addStretch(1)
        self.bot = QPushButton('Revisar'); self.bot.setObjectName('revisar'); self.bot.setCursor(Qt.PointingHandCursor)
        self.bot.clicked.connect(lambda: self.revisar.emit(self.clave)); lay.addWidget(self.bot, 0, Qt.AlignLeft)

    def poner(self, valor, detalle, estado, n=None):
        c = ESTADO_COLOR[estado]
        self.setStyleSheet(f'QFrame#tarjeta {{ border-left: 6px solid {c}; }}')
        self.est.setText(f'<span style="color:{c}; font-weight:600">● {ESTADO_TXT[estado]}</span>')
        self.valor.setText(valor); self.det.setText(detalle)
        self.bot.setVisible(n is not None); self.bot.setText(f'Revisar {n} episodio(s)  ▸' if n else 'Ver en el ECG  ▸')


class PaginaInicio(QWidget):
    def __init__(self, win):
        super().__init__()
        lay = QVBoxLayout(self); lay.setContentsMargins(80, 50, 80, 40); lay.setSpacing(14)
        t = QLabel('¿Qué examen quieres revisar?'); t.setObjectName('seccion'); t.setStyleSheet('font-size: 18pt;'); lay.addWidget(t)
        n = QLabel('Elige un examen. HolterLocal lo analiza con los tres canales y te lleva a los hallazgos.'); n.setObjectName('nota')
        lay.addWidget(n); lay.addSpacing(10)
        for txt, sub, fn in (('📋  Lista de exámenes', 'Los Holter de la carpeta de HScribe, como en “Buscar examen”', win.lista),
                             ('📂  Abrir un archivo o carpeta', 'HScribe, WFDB, EDF o ISHNE', win.abrir),
                             ('🫀  Abrir el Holter de ejemplo', 'Registro de 24 h con FA y extrasístoles (PhysioNet)', win.abrir_ejemplo)):
            b = QPushButton(f'{txt}\n      {sub}'); b.setObjectName('grande'); b.setCursor(Qt.PointingHandCursor); b.clicked.connect(fn)
            b.setMinimumHeight(78); lay.addWidget(b)
        lay.addStretch(1)


class PaginaHallazgos(QWidget):
    revisar = Signal(str)

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self); lay.setContentsMargins(20, 14, 20, 14); lay.setSpacing(10)
        self.tit = QLabel('Hallazgos clave'); self.tit.setObjectName('seccion'); lay.addWidget(self.tit)
        self.sub = QLabel(''); self.sub.setObjectName('nota'); lay.addWidget(self.sub)
        g = QGridLayout(); g.setSpacing(12); lay.addLayout(g)
        self.t = {}
        for i, (k, tit) in enumerate((('fc', 'Frecuencia cardíaca'), ('fa', 'Fibrilación auricular'), ('s', 'Supraventriculares'),
                                       ('v', 'Ventriculares'), ('pausa', 'Pausas'), ('bloq', 'Bloqueos AV / SA'),
                                       ('cond', 'Conducción (PR y QRS)'), ('art', 'Calidad de señal y artefactos'))):
            c = Tarjeta(k, tit); c.revisar.connect(self.revisar); self.t[k] = c; g.addWidget(c, i // 4, i % 4)
        fin = QHBoxLayout(); fin.addStretch(1)
        b = QPushButton('Empezar la revisión  ▸'); b.setObjectName('primario'); b.setMinimumHeight(42); b.setMinimumWidth(240)
        b.clicked.connect(lambda: self.revisar.emit('')); fin.addWidget(b); lay.addLayout(fin); lay.addStretch(1)

    def mostrar(self, ex):
        A = ex.A; hr = A.get('hr') or {}; cnt = A.get('cnt') or {}; c = A.get('counts') or [0] * 14; af = A.get('af') or {}
        hs = max(1e-6, A.get('analyzedSec', 1) / 3600)
        n = lambda k: len(de_categoria(ex, k))
        self.sub.setText(f'{ex.nombre} · {hms(ex.dur)} de registro · {len(ex.chans)} canales · {_f(A.get("total"))} latidos · '
                         f'motor {A.get("motor") or "estación"}. Verde: nada que revisar · ámbar: revisar · rojo: importante.')
        tb = lambda b: float(ex.t[b]) if b is not None and 0 <= b < len(ex.t) else None
        self.t['fc'].poner(f'{_f(hr.get("mean"))} lpm', f'Mínima <b>{_f(hr.get("min"))}</b> lpm ({reloj(ex, tb(hr.get("minB")) or 0)})<br>'
                           f'Máxima <b>{_f(hr.get("max"))}</b> lpm ({reloj(ex, tb(hr.get("maxB")) or 0)})<br>'
                           f'Día / noche: {_f(hr.get("day"))} / {_f(hr.get("night"))} lpm', 'ok', n('fc'))
        ep = af.get('episodes') or 0
        self.t['fa'].poner(f'{_f(af.get("pct"), 1)} %' if ep else 'No', (f'<b>{ep}</b> episodio(s), {hms(af.get("sec", 0))} en total<br>'
                           f'El más largo: {hms(af.get("longest", 0))}') if ep else 'Sin episodios de FA', 'alto' if ep else 'ok', n('fa'))
        st = A.get('sTot') or 0
        self.t['s'].poner(_f(st), f'{_f(st / hs, 1)} por hora · carga {_f(A.get("sBurden"), 2)} %<br>'
                          f'Pares <b>{_f(cnt.get("sPair"))}</b> · rachas <b>{_f(cnt.get("sRun"))}</b> · aberrantes {_f(c[7])}',
                          'ver' if st else 'ok', n('s'))
        vt_ = A.get('vTot') or 0; ntv = sum(1 for e in ex.eventos if e['type'] == 'vt')
        self.t['v'].poner(_f(vt_), f'{_f(vt_ / hs, 1)} por hora · carga {_f(A.get("vBurden"), 2)} %<br>'
                          f'Pares <b>{_f(cnt.get("vPair"))}</b> · TV <b>{ntv}</b> · bigeminismo {_f(cnt.get("vBig"))}',
                          'alto' if ntv else 'ver' if vt_ else 'ok', n('v'))
        np_ = A.get('pauses') or 0
        self.t['pausa'].poner(f'{_f(A.get("rrMax"), 2)} s', f'RR más largo ({reloj(ex, tb(A.get("rrMaxB")) or 0)})<br>'
                              f'Pausas ≥ {_f((A.get("settings") or {}).get("pauseSec", 2), 1)} s: <b>{np_}</b>',
                              'alto' if np_ else 'ok', n('pausa'))
        nb = n('bloq'); ha = getattr(ex, 'ha', None)
        self.t['bloq'].poner(str(nb) if ha else '—', ('Episodios de bloqueo AV o SA detectados' if nb else 'Sin bloqueos AV ni SA')
                             if ha else 'Este análisis no buscó bloqueos (requiere el motor holterai)',
                             ('alto' if nb else 'ok') if ha else 'nada', nb if ha else None)
        brb = c[6]; q = A.get('qtc')
        if isinstance(q, (list, tuple)):
            q = [x for x in q if x is not None]; q = float(np.median(q)) if q else None
        self.t['cond'].poner(f'{_f(q)} ms' if q else '—', f'QTc medio · latidos con bloqueo de rama <b>{_f(brb)}</b><br>'
                             f'PR: {"medido por holterai" if ha else "requiere el motor holterai"}', 'ver' if brb else 'ok', None)
        exp = A.get('exPct') or 0
        self.t['art'].poner(f'{_f(exp, 1)} %', f'Tiempo excluido: {hms(A.get("exSec", 0))}<br>Latidos marcados como artefacto: '
                            f'<b>{_f(c[4])}</b>', 'alto' if exp > 10 else 'ver' if exp > 1 or c[4] else 'ok', n('art'))


class PaginaRevisar(QWidget):
    def __init__(self, win):
        super().__init__(); self.win = win; self.ex = None; self.lista = []; self.estado = {}
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        self.cat = QListWidget(); self.cat.setObjectName('categorias'); self.cat.setFixedWidth(290)
        self.cat.currentRowChanged.connect(self._categoria); lay.addWidget(self.cat)
        der = QWidget(); dl = QVBoxLayout(der); dl.setContentsMargins(12, 10, 12, 6); dl.setSpacing(6); lay.addWidget(der, 1)
        top = QHBoxLayout(); self.tit = QLabel(''); self.tit.setObjectName('seccion'); top.addWidget(self.tit); top.addStretch(1)
        self.pos = QLabel(''); self.pos.setObjectName('nota'); top.addWidget(self.pos); dl.addLayout(top)
        self.split = QSplitter(Qt.Vertical); dl.addWidget(self.split, 1)
        self.tab = QTableWidget(0, 6); self.tab.setHorizontalHeaderLabels(['Hora', 'Tipo', 'Duración', 'Latidos', 'FC', 'Tu decisión'])
        self.tab.verticalHeader().setVisible(False); self.tab.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tab.setEditTriggers(QAbstractItemView.NoEditTriggers); self.tab.setAlternatingRowColors(True)
        self.tab.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tab.currentCellChanged.connect(lambda r, *_: self._ir(r)); self.split.addWidget(self.tab)
        self.hueco = QWidget(); self.hl = QVBoxLayout(self.hueco); self.hl.setContentsMargins(0, 0, 0, 0); self.split.addWidget(self.hueco)
        self.split.setSizes([200, 520])
        dec = QHBoxLayout(); dec.setSpacing(8)
        for txt, col, fn, sc in (('◀ Anterior', '#ffffff', lambda: self._mover(-1), 'Left'),
                                 ('✓ Correcto', '#16a34a', lambda: self._decidir('Correcto'), 'Return'),
                                 ('✗ Es artefacto', '#d946ef', self._artefacto, 'A'),
                                 ('No corresponde', '#6b7280', lambda: self._decidir('No corresponde'), 'Delete'),
                                 ('＋ Tira al informe', AZUL, self._tira, 'T'),
                                 ('Siguiente ▶', '#ffffff', lambda: self._mover(1), 'Right')):
            b = QPushButton(f'{txt}  ({ {"Return": "Enter", "Delete": "Supr"}.get(sc, sc)})' if sc not in ('Left', 'Right') else txt); b.setObjectName('decide')
            fg = '#1f2933' if col == '#ffffff' else 'white'
            b.setStyleSheet(f'QPushButton#decide {{ background: {col}; color: {fg}; border: 1px solid {"#d5d9e0" if col == "#ffffff" else col}; }}')
            b.clicked.connect(fn); b.setCursor(Qt.PointingHandCursor); dec.addWidget(b)
            QShortcut(QKeySequence(sc), self, activated=fn)
        dl.addLayout(dec)

    def tomar_panel(self, panel):
        self.hl.addWidget(panel); panel.show()

    def mostrar(self, ex, clave=None):
        self.ex = ex; self.cat.blockSignals(True); self.cat.clear()
        for k, nom, _ in CATEGORIAS:
            nn = len(de_categoria(ex, k)); it = QListWidgetItem(f'{nom}\n{nn} episodio(s)' if nn else f'{nom}\nnada que revisar')
            it.setData(Qt.UserRole, k)
            if not nn:
                it.setForeground(QColor('#9ca3af'))
            self.cat.addItem(it)
        self.cat.blockSignals(False)
        claves = [k for k, _, _ in CATEGORIAS]
        if not clave:  # sin categoría: la primera que tenga algo que revisar
            clave = next((k for k in claves if de_categoria(ex, k)), claves[0])
        self.cat.setCurrentRow(claves.index(clave) if clave in claves else 0); self._categoria(self.cat.currentRow())

    def _categoria(self, i):
        if not self.ex or i < 0:
            return
        k, nom, _ = CATEGORIAS[i]; self.clave = k; self.lista = de_categoria(self.ex, k)
        self.tit.setText(nom); self.tab.setRowCount(len(self.lista))
        for r, e in enumerate(self.lista):
            dur = e.get('dur', e['t1'] - e['t0'])
            vals = [reloj(self.ex, e['t0']), e.get('label') or NOMBRE_TIPO.get(e['type'], e['type']), hms(dur) if dur >= 1 else f'{_f(dur, 1)} s',
                    _f(e.get('n')) if e.get('n') else '—', (_f(e['rate']) + ' lpm') if e.get('rate') else '—',
                    self.estado.get((e['type'], round(e['t0'], 1)), 'Pendiente')]
            for c, v in enumerate(vals):
                self.tab.setItem(r, c, QTableWidgetItem(str(v)))
            self._pintar(r)
        if self.lista:
            self.tab.selectRow(0); self._ir(0)
        self.pos.setText(f'{len(self.lista)} episodio(s)')

    def _pintar(self, r):
        it = self.tab.item(r, 5); v = it.text()
        it.setForeground(QColor({'Correcto': '#16a34a', 'Artefacto': '#d946ef', 'No corresponde': '#6b7280'}.get(v, '#d97706')))

    def _ir(self, r):
        if 0 <= r < len(self.lista):
            e = self.lista[r]; b = e.get('b0', -1)
            self.win.ir_a(float(e['t0']), int(b) if b is not None and b >= 0 else -1)
            hechos = sum(1 for x in self.lista if (x['type'], round(x['t0'], 1)) in self.estado)
            self.pos.setText(f'Episodio {r + 1} de {len(self.lista)} · revisados {hechos}')

    def _mover(self, d):
        r = self.tab.currentRow() + d
        if 0 <= r < len(self.lista):
            self.tab.selectRow(r)

    def _decidir(self, txt):
        r = self.tab.currentRow()
        if 0 <= r < len(self.lista):
            e = self.lista[r]; self.estado[(e['type'], round(e['t0'], 1))] = txt
            self.tab.item(r, 5).setText(txt); self._pintar(r); self._mover(1)

    def _artefacto(self):
        r = self.tab.currentRow()
        if 0 <= r < len(self.lista):
            e = self.lista[r]; self._decidir('Artefacto'); self.win.excluir(e['t0'], max(e['t1'], e['t0'] + 1))

    def _tira(self):
        r = self.tab.currentRow()
        if 0 <= r < len(self.lista):
            self.win.pedir_tira(self.lista[r]['t0'], 10, False)


class PaginaInforme(QWidget):
    def __init__(self, win):
        super().__init__(); self.win = win
        lay = QHBoxLayout(self); lay.setContentsMargins(20, 14, 20, 14); lay.setSpacing(14)
        izq = QVBoxLayout(); t = QLabel('Vista previa del informe'); t.setObjectName('seccion'); izq.addWidget(t)
        self.prev = QTextBrowser(); izq.addWidget(self.prev, 1); lay.addLayout(izq, 3)
        der = QVBoxLayout(); der.setSpacing(8)
        for nom, ph in (('Diagnóstico', 'Hasta unas 4 líneas'), ('Conclusiones', 'Conclusiones del médico')):
            lb = QLabel(nom); lb.setObjectName('seccion'); der.addWidget(lb)
            e = QPlainTextEdit(); e.setPlaceholderText(ph); der.addWidget(e, 1 if nom == 'Conclusiones' else 0)
            if nom == 'Diagnóstico':
                e.setMaximumHeight(110); self.diag = e
            else:
                self.conc = e
        self.tiras = QLabel(''); self.tiras.setObjectName('nota'); self.tiras.setWordWrap(True); der.addWidget(self.tiras)
        for txt, fn, prim in (('Elegir las tiras del informe…', lambda: win.tiras_auto(), False),
                              ('Imprimir o guardar PDF…', win.informe, True), ('Finalizar y firmar…', win.finalizar, False)):
            b = QPushButton(txt); b.setObjectName('primario' if prim else ''); b.setMinimumHeight(40); b.clicked.connect(fn); der.addWidget(b)
        lay.addLayout(der, 2)

    def mostrar(self, ex):
        from .estilo import RESUMEN_CSS
        self.prev.document().setDefaultStyleSheet(RESUMEN_CSS)
        self.prev.setHtml(f'<h2>Holter de ritmo · {ex.nombre}</h2><p class="m">{hms(ex.dur)} de registro · {len(ex.chans)} canales</p>'
                          + html_resumen(ex))
        ses = getattr(self.win, 'ses', None); n = len(ses.d.get('tiras', [])) if ses else 0
        self.tiras.setText(f'Tiras en el informe: {n}. Puedes cambiar cada una por otra de las candidatas.')


class ModoSimple(QWidget):
    """Ventana del modo simple: cabecera con los cuatro pasos y las páginas."""
    experto = Signal()
    PASOS = ('Exámenes', 'Hallazgos', 'Revisar', 'Informe')

    def __init__(self, win):
        super().__init__(); self.setObjectName('simple'); self.win = win; self.setStyleSheet(QSS_SIMPLE)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        cab = QFrame(); cab.setObjectName('cabecera'); cl = QHBoxLayout(cab); cl.setContentsMargins(18, 8, 18, 8)
        m = QLabel('HolterLocal'); m.setObjectName('marca'); cl.addWidget(m); cl.addSpacing(16)
        self.info = QLabel('Sin examen abierto'); self.info.setObjectName('examen'); cl.addWidget(self.info); cl.addStretch(1)
        self.grupo = QButtonGroup(self); self.grupo.setExclusive(True); self.bot = []
        for i, p in enumerate(self.PASOS):
            b = QPushButton(f'{i + 1}  {p}'); b.setObjectName('paso'); b.setCheckable(True); b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, i=i: self.ir(i)); self.grupo.addButton(b); self.bot.append(b); cl.addWidget(b)
            if i < 3:
                s = QLabel('›'); s.setStyleSheet('color:#9ca3af; font-size: 14pt;'); cl.addWidget(s)
        cl.addStretch(1)
        e = QPushButton('Modo experto ▸'); e.setObjectName('experto'); e.setToolTip('Todas las pestañas y menús de siempre')
        e.clicked.connect(self.experto); cl.addWidget(e)
        lay.addWidget(cab)
        self.pag = QStackedWidget(); lay.addWidget(self.pag, 1)
        self.p0 = PaginaInicio(win); self.p1 = PaginaHallazgos(); self.p2 = PaginaRevisar(win); self.p3 = PaginaInforme(win)
        for p in (self.p0, self.p1, self.p2, self.p3):
            self.pag.addWidget(p)
        self.p1.revisar.connect(lambda k: self.ir(2, k))
        self.ir(0)

    def ir(self, i, clave=None):
        ex = self.win.ex
        if i > 0 and ex is None:
            i = 0
        self.bot[i].setChecked(True); self.pag.setCurrentIndex(i)
        if i == 1:
            self.p1.mostrar(ex)
        elif i == 2:
            self.p2.tomar_panel(self.win.panel); self.p2.mostrar(ex, clave)
        elif i == 3:
            self.p3.mostrar(ex)

    def examen_abierto(self, ex):
        self.info.setText(f'{ex.nombre} · {hms(ex.dur)} · {len(ex.chans)} canales')
        self.ir(1)
