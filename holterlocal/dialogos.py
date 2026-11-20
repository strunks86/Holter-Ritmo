"""Ventanas de diálogo: información del paciente, diario, etiquetas de sucesos, ir a una hora, deshacer artefactos,
volver a escanear, finalizar el examen, buscar exámenes, ajustes del informe y edición de sucesos."""
import os

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from .sesion import ESTADOS, Registro

CAMPOS_PACIENTE = [
    ('apellido', 'Apellido'), ('nombre', 'Nombre'), ('id', 'ID del paciente'), ('id2', 'ID secundaria'), ('admision', 'ID de admisión'),
    ('nacimiento', 'Fecha de nacimiento'), ('edad', 'Edad'), ('sexo', 'Sexo'), ('raza', 'Etnia'), ('direccion', 'Dirección'),
    ('telefono', 'Teléfono'), ('email', 'Correo electrónico'), ('indicaciones', 'Indicaciones'), ('medicamentos', 'Medicamentos'),
    ('derivador', 'Médico que deriva'), ('procedimiento', 'Tipo de procedimiento'), ('ubicacion', 'Ubicación'),
    ('fecha_registro', 'Fecha del registro'), ('tecnico', 'Técnico'), ('analista', 'Analista'), ('grabadora_tipo', 'Tipo de grabadora'),
    ('grabadora_num', 'Número de grabadora')]


def _botones(dlg, lay):
    bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
    lay.addWidget(bb) if not isinstance(lay, QFormLayout) else lay.addRow(bb)
    return bb


class DialogoPaciente(QDialog):
    def __init__(self, datos, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Información del paciente'); self.resize(640, 560)
        lay = QVBoxLayout(self); g = QGridLayout(); self.w = {}
        for i, (k, n) in enumerate(CAMPOS_PACIENTE):
            if k == 'sexo':
                w = QComboBox(); w.setEditable(True); w.addItems(['', 'Femenino', 'Masculino', 'Otro']); w.setCurrentText(datos.get(k, ''))
            elif k in ('indicaciones', 'medicamentos'):
                w = QPlainTextEdit(datos.get(k, '')); w.setMaximumHeight(52)
            else:
                w = QLineEdit(datos.get(k, ''))
                if k == 'grabadora_tipo' and not datos.get(k):
                    w.setText('Holter de 3 canales')
            self.w[k] = w; g.addWidget(QLabel(n), i // 2, (i % 2) * 2); g.addWidget(w, i // 2, (i % 2) * 2 + 1)
        lay.addLayout(g); _botones(self, lay)

    def valores(self):
        out = {}
        for k, w in self.w.items():
            out[k] = w.currentText() if isinstance(w, QComboBox) else w.toPlainText() if isinstance(w, QPlainTextEdit) else w.text()
        return out


class DialogoDiario(QDialog):
    def __init__(self, ex, diario, parent=None):
        super().__init__(parent)
        self.ex = ex; self.setWindowTitle('Lista del diario'); self.resize(560, 420)
        lay = QVBoxLayout(self)
        self.tabla = QTableWidget(0, 2); self.tabla.setHorizontalHeaderLabels(['Hora (HH:MM:SS (día))', 'Síntoma o actividad'])
        self.tabla.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch); lay.addWidget(self.tabla, 1)
        for d in sorted(diario, key=lambda d: d['t']):
            self._fila(self._hora(d['t']), d.get('texto', ''))
        fila = QHBoxLayout()
        for txt, fn in (('Agregar', lambda: self._fila(self._hora(0), '')), ('Borrar', self._borrar)):
            b = QPushButton(txt); b.clicked.connect(fn); fila.addWidget(b)
        fila.addStretch(1); lay.addLayout(fila)
        lay.addWidget(QLabel('Después de las primeras 24 horas escribe el día entre paréntesis, por ejemplo 08:24:36 (2).'))
        _botones(self, lay)

    def _hora(self, t):
        dia = int((self.ex.seg_inicio() + t) // 86400)
        return self.ex.reloj(t).split(' ')[-1] + (f' ({dia + 1})' if dia else '')

    def _fila(self, h, txt):
        r = self.tabla.rowCount(); self.tabla.insertRow(r)
        self.tabla.setItem(r, 0, QTableWidgetItem(h)); self.tabla.setItem(r, 1, QTableWidgetItem(txt))

    def _borrar(self):
        for r in sorted({i.row() for i in self.tabla.selectedIndexes()}, reverse=True):
            self.tabla.removeRow(r)

    def valores(self):
        out = []
        for r in range(self.tabla.rowCount()):
            h = self.tabla.item(r, 0).text() if self.tabla.item(r, 0) else ''
            t = self.ex.parse_reloj(h)
            if t is not None:
                out.append({'t': t, 'texto': self.tabla.item(r, 1).text() if self.tabla.item(r, 1) else ''})
        return sorted(out, key=lambda d: d['t'])


class DialogoEtiquetas(QDialog):
    def __init__(self, etiquetas, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Etiquetas de sucesos del usuario'); f = QFormLayout(self); self.w = []
        for i in range(3):
            w = QLineEdit(etiquetas[i] if i < len(etiquetas) else ''); w.setMaxLength(16); w.setPlaceholderText(f'Definido por el usuario {i + 1}')
            self.w.append(w); f.addRow(f'Suceso {i + 1}', w)
        f.addRow(QLabel('Hasta 16 caracteres. Se cuentan los latidos de cada suceso en el perfil y el resumen.'))
        _botones(self, f)

    def valores(self):
        return [w.text().strip() for w in self.w]


class DialogoIrHora(QDialog):
    def __init__(self, ex, parent=None):
        super().__init__(parent)
        self.ex = ex; self.setWindowTitle('Ir a una hora'); f = QFormLayout(self)
        self.h = QLineEdit(ex.reloj(0).split(' ')[-1]); f.addRow('Hora (HH:MM:SS)', self.h)
        f.addRow(QLabel(f'El registro va de {ex.reloj(0)} a {ex.reloj(ex.dur)}. Para el segundo día: 08:24:36 (2).'))
        _botones(self, f)

    def tiempo(self):
        return self.ex.parse_reloj(self.h.text())


class DialogoArtefactos(QDialog):
    def __init__(self, ex, lista, parent=None):
        super().__init__(parent)
        self.ex = ex; self.lista = [list(x) for x in lista]; self.hist = []
        self.setWindowTitle('Deshacer artefacto'); self.resize(460, 380)
        lay = QVBoxLayout(self); self.w = QListWidget(); self.w.setSelectionMode(QAbstractItemView.ExtendedSelection); lay.addWidget(self.w, 1)
        fila = QHBoxLayout()
        for txt, fn in (('Eliminar los elegidos', self._quitar), ('Eliminar todos', self._todos), ('Deshacer el último', self._ultimo)):
            b = QPushButton(txt); b.clicked.connect(fn); fila.addWidget(b)
        lay.addLayout(fila); _botones(self, lay); self._pinta()

    def _pinta(self):
        self.w.clear()
        for a, b in self.lista:
            self.w.addItem(f'{self.ex.reloj(a)} – {self.ex.reloj(b)}  ({b - a:.0f} s)')

    def _quitar(self):
        for r in sorted({i.row() for i in self.w.selectedIndexes()}, reverse=True):
            self.lista.pop(r)
        self._pinta()

    def _todos(self):
        if self.lista and QMessageBox.question(self, 'Deshacer artefacto', '¿Eliminar todas las marcas de artefacto?') == QMessageBox.Yes:
            self.lista = []; self._pinta()

    def _ultimo(self):
        if self.lista:
            self.lista.pop(); self._pinta()


class DialogoReescanear(QDialog):
    def __init__(self, ex, previo, parent=None):
        super().__init__(parent)
        self.ex = ex; self.setWindowTitle('Volver a escanear'); f = QFormLayout(self)
        f.addRow(QLabel('<b>Se perderán todas las correcciones de latidos, artefactos, cambios del perfil y del resumen y las tiras automáticas.</b>'))
        fila = QHBoxLayout(); self.ch = []
        excl = set(previo.get('excluir', []))
        nombres = previo.get('nombres') or [ex.names[c] if c < len(ex.names) else str(c + 1) for c in range(ex.nsig)]
        for c in range(ex.nsig):
            cb = QCheckBox(nombres[c] if c < len(nombres) else f'Canal {c + 1}'); cb.setChecked(c not in excl); self.ch.append(cb); fila.addWidget(cb)
        f.addRow('Canales que se analizan', fila)
        self.h = QSpinBox(); self.h.setRange(0, 7 * 24); self.m = QSpinBox(); self.m.setRange(0, 59)
        dur = previo.get('maxSec') or ex.dur
        self.h.setValue(int(dur // 3600)); self.m.setValue(int((dur % 3600) // 60))
        fd = QHBoxLayout(); fd.addWidget(self.h); fd.addWidget(QLabel('horas')); fd.addWidget(self.m); fd.addWidget(QLabel('minutos')); fd.addStretch(1)
        f.addRow('Duración a analizar', fd)
        f.addRow(QLabel('Acortar sirve cuando los electrodos se soltaron antes del final. Se recomienda analizar siempre los 3 canales.'))
        _botones(self, f)

    def valores(self):
        excl = [c for c, cb in enumerate(self.ch) if not cb.isChecked()]
        if len(excl) == len(self.ch):
            excl = []
        sec = self.h.value() * 3600 + self.m.value() * 60
        return {'excluir': excl, 'maxSec': sec if sec < self.ex.dur - 30 else 0}


class DialogoFinalizar(QDialog):
    def __init__(self, ses, presets, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Finalizar el examen'); f = QFormLayout(self); d = ses.d
        p = d.get('paciente') or {}
        f.addRow(QLabel(f"<b>{p.get('nombre', '')} {p.get('apellido', '')}</b> · ID {p.get('id', '—')} · estado actual: {d.get('estado', 'Adquirido')}"))
        self.estado = QComboBox(); self.estado.addItems(ESTADOS)
        i = ESTADOS.index(d.get('estado', 'Adquirido')) if d.get('estado') in ESTADOS else 0
        self.estado.setCurrentIndex(min(len(ESTADOS) - 1, i + 1))
        f.addRow('Nuevo estado', self.estado)
        self.rev = QLineEdit(d.get('revisado_por', '')); self.firma = QLineEdit(d.get('firmado_por', '') or QSettings('HolterLocal', 'HolterLocal').value('medico', ''))
        f.addRow('Revisado por', self.rev); f.addRow('Firmado por', self.firma)
        self.informe = QComboBox(); self.informe.addItems(presets)
        f.addRow('Tipo de informe', self.informe)
        self.imprimir = QComboBox(); self.imprimir.addItems(['No imprimir', 'Guardar el PDF siempre', 'Guardar el PDF si queda firmado'])
        f.addRow('Al actualizar', self.imprimir)
        self.estado.currentTextChanged.connect(self._vis); self._vis(self.estado.currentText())
        bb = QDialogButtonBox(); self.b_prev = bb.addButton('Vista previa…', QDialogButtonBox.ActionRole)
        bb.addButton('Actualizar', QDialogButtonBox.AcceptRole); bb.addButton(QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject); f.addRow(bb)
        self.vista_previa = False
        self.b_prev.clicked.connect(self._prev)

    def _prev(self):
        self.vista_previa = True; self.accept()

    def _vis(self, e):
        self.rev.setEnabled(e in ('Revisado', 'Firmado')); self.firma.setEnabled(e == 'Firmado')

    def valores(self):
        return {'estado': self.estado.currentText(), 'revisado_por': self.rev.text(), 'firmado_por': self.firma.text(),
                'informe': self.informe.currentText(), 'imprimir': self.imprimir.currentIndex()}


class DialogoBuscar(QDialog):
    COLS = ['Apellido', 'Nombre', 'ID', 'Examen', 'Inicio', 'Horas', 'Estado', 'Modificado']

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Buscar examen'); self.resize(900, 520); self.reg = Registro(); self.elegido = None; self.accion = ''
        lay = QVBoxLayout(self)
        fila = QHBoxLayout(); self.q = QLineEdit(); self.q.setPlaceholderText('Apellido, nombre o ID (empieza con…)')
        fila.addWidget(self.q, 1); self.chk = {}
        for e in ['Adquirido', 'Editado', 'Revisado', 'Firmado']:
            c = QCheckBox(e); c.setChecked(True); c.toggled.connect(self._pinta); self.chk[e] = c; fila.addWidget(c)
        self.cuando = QComboBox(); self.cuando.addItems(['Todos', 'Hoy', 'Última semana']); self.cuando.currentIndexChanged.connect(self._pinta)
        fila.addWidget(self.cuando)
        b = QPushButton('Buscar'); b.clicked.connect(self._pinta); fila.addWidget(b)
        lay.addLayout(fila)
        self.t = QTableWidget(0, len(self.COLS)); self.t.setHorizontalHeaderLabels(self.COLS); self.t.setSortingEnabled(True)
        self.t.setSelectionBehavior(QAbstractItemView.SelectRows); self.t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.t.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch); self.t.cellDoubleClicked.connect(lambda *_: self._ok('editar'))
        lay.addWidget(self.t, 1)
        bot = QHBoxLayout()
        for txt, k in (('Editar', 'editar'), ('Informe', 'informe'), ('Quitar de la lista', 'quitar')):
            bb = QPushButton(txt); bb.clicked.connect(lambda _=False, k=k: self._ok(k)); bot.addWidget(bb)
        bot.addStretch(1); cerrar = QPushButton('Cerrar'); cerrar.clicked.connect(self.reject); bot.addWidget(cerrar)
        lay.addLayout(bot)
        self.q.textChanged.connect(self._pinta); self._pinta()

    def _pinta(self):
        q = self.q.text().strip().lower()
        import time
        desde = time.strftime('%Y-%m-%d', time.localtime(time.time() - (0 if self.cuando.currentIndex() == 1 else 6 * 86400)))
        self.t.setSortingEnabled(False); self.t.setRowCount(0)
        for k, it in sorted(self.reg.items.items(), key=lambda x: x[1].get('modificado', ''), reverse=True):
            if not self.chk.get(it.get('estado', 'Adquirido'), self.chk['Adquirido']).isChecked():
                continue
            if q and not any(str(it.get(c, '')).lower().startswith(q) for c in ('apellido', 'nombres', 'id', 'nombre')):
                continue
            if self.cuando.currentIndex() and str(it.get('modificado', '')) < desde:
                continue
            r = self.t.rowCount(); self.t.insertRow(r)
            for j, v in enumerate([it.get('apellido', ''), it.get('nombres', ''), it.get('id', ''), it.get('nombre', '') or os.path.basename(it.get('ruta', '')),
                                   it.get('inicio', ''), str(it.get('duracion_h', '')), it.get('estado', ''), it.get('modificado', '')]):
                c = QTableWidgetItem(v); c.setData(Qt.UserRole, k); self.t.setItem(r, j, c)
        self.t.setSortingEnabled(True)

    def _ok(self, accion):
        r = self.t.currentRow()
        if r < 0:
            return
        k = self.t.item(r, 0).data(Qt.UserRole)
        if accion == 'quitar':
            if QMessageBox.question(self, 'Buscar examen', 'Se borra de la lista y se pierde el trabajo guardado de este examen (no el examen original). ¿Seguir?') == QMessageBox.Yes:
                self.reg.quitar(k); self._pinta()
            return
        self.elegido = self.reg.items[k]; self.accion = accion; self.accept()


SECCIONES = [('portada', 'Información del paciente y estadísticas resumidas'), ('estadisticas', 'Estadísticas completas'),
             ('narrativo', 'Resumen narrativo'), ('perfiles', 'Perfiles hora a hora'), ('tendencias', 'Tendencias'),
             ('plantillas', 'Plantillas'), ('tiras', 'Tiras de ECG'), ('completa', 'Información completa (ECG en miniatura)'),
             ('holterai', 'Hallazgos de conducción (holterai)')]
PRESET_BASE = {'secciones': {k: k not in ('completa',) for k, _ in SECCIONES}, 'condensado': False, 'ganancia': 10, 'tendencias': ['fc', 'ectopia', 'st', 'qt', 'vfc'],
               'completa_canales': [0], 'completa_horas': [], 'rejilla': True}


def presets():
    s = QSettings('HolterLocal', 'HolterLocal')
    import json
    try:
        d = json.loads(s.value('presets_informe', '') or '{}')
    except ValueError:
        d = {}
    if 'Estándar' not in d:
        d['Estándar'] = dict(PRESET_BASE)
    if 'Condensado' not in d:
        c = dict(PRESET_BASE); c['condensado'] = True
        c['secciones'] = {k: k in ('portada', 'tiras') for k, _ in SECCIONES}; d['Condensado'] = c
    return d


def guardar_presets(d):
    import json
    QSettings('HolterLocal', 'HolterLocal').setValue('presets_informe', json.dumps(d))


class DialogoInforme(QDialog):
    """Opciones del informe de este examen: tipo, secciones, ganancia, tendencias e información completa."""

    def __init__(self, ex, actual, parent=None):
        super().__init__(parent)
        self.ex = ex; self.setWindowTitle('Informe final'); self.resize(640, 560)
        self.P = presets(); lay = QVBoxLayout(self)
        fila = QHBoxLayout(); fila.addWidget(QLabel('Tipo de informe'))
        self.tipo = QComboBox(); self.tipo.addItems(list(self.P)); fila.addWidget(self.tipo)
        bg = QPushButton('Guardar como tipo…'); bg.clicked.connect(self._guardar_tipo); fila.addWidget(bg)
        bb_ = QPushButton('Borrar tipo'); bb_.clicked.connect(self._borrar_tipo); fila.addWidget(bb_)
        fila.addStretch(1); lay.addLayout(fila)
        g = QGroupBox('Secciones'); gl = QGridLayout(g); self.sec = {}
        for i, (k, n) in enumerate(SECCIONES):
            c = QCheckBox(n); self.sec[k] = c; gl.addWidget(c, i // 2, i % 2)
        lay.addWidget(g)
        f = QFormLayout()
        self.cond = QCheckBox('Portada condensada (subconjunto de estadísticas en la primera página)'); f.addRow('', self.cond)
        self.gan = QComboBox(); [self.gan.addItem(f'{v} mm/mV', v) for v in (5, 10, 20, 40)]; f.addRow('Ganancia de las tiras', self.gan)
        self.rej = QCheckBox('Cuadrícula del ECG'); f.addRow('', self.rej)
        ft = QHBoxLayout(); self.tend = {}
        for k, n in (('fc', 'FC'), ('ectopia', 'Ectopia'), ('st', 'ST'), ('qt', 'QT'), ('vfc', 'VFC')):
            c = QCheckBox(n); self.tend[k] = c; ft.addWidget(c)
        f.addRow('Tendencias', ft)
        fc = QHBoxLayout(); self.cc = []
        for i, c in enumerate(ex.chans):
            cb = QCheckBox(ex.names[c] if c < len(ex.names) else f'Canal {i + 1}'); self.cc.append(cb); fc.addWidget(cb)
        f.addRow('Información completa: canales (hasta 3)', fc)
        self.horas = QListWidget(); self.horas.setMaximumHeight(110); self.horas.setFlow(QListWidget.LeftToRight); self.horas.setWrapping(True)
        for h in range(int(ex.dur // 3600) + 1):
            it = QListWidgetItem(ex.reloj(h * 3600, False)); it.setFlags(it.flags() | Qt.ItemIsUserCheckable); it.setCheckState(Qt.Unchecked); self.horas.addItem(it)
        hb = QHBoxLayout(); a = QPushButton('Todas'); n = QPushButton('Ninguna')
        a.clicked.connect(lambda: self._horas(True)); n.clicked.connect(lambda: self._horas(False)); hb.addWidget(a); hb.addWidget(n); hb.addStretch(1)
        f.addRow('Horas de información completa', self.horas); f.addRow('', hb)
        lay.addLayout(f)
        bb = QDialogButtonBox(); self.b_prev = bb.addButton('Vista previa', QDialogButtonBox.AcceptRole)
        self.b_pdf = bb.addButton('Guardar PDF…', QDialogButtonBox.ActionRole); self.b_imp = bb.addButton('Imprimir…', QDialogButtonBox.ActionRole)
        bb.addButton(QDialogButtonBox.Cancel); bb.rejected.connect(self.reject)
        self.accion = 'previa'
        self.b_prev.clicked.connect(lambda: self._fin('previa')); self.b_pdf.clicked.connect(lambda: self._fin('pdf')); self.b_imp.clicked.connect(lambda: self._fin('imprimir'))
        lay.addWidget(bb)
        self.tipo.currentTextChanged.connect(lambda t: self.cargar(self.P.get(t, PRESET_BASE)))
        if actual:
            self.tipo.blockSignals(True); self.tipo.setCurrentText(actual.get('tipo', 'Estándar')); self.tipo.blockSignals(False); self.cargar(actual)
        else:
            self.cargar(self.P['Estándar'])

    def _horas(self, on):
        for i in range(self.horas.count()):
            self.horas.item(i).setCheckState(Qt.Checked if on else Qt.Unchecked)

    def cargar(self, o):
        for k, c in self.sec.items():
            c.setChecked(bool(o.get('secciones', {}).get(k, PRESET_BASE['secciones'].get(k))))
        self.cond.setChecked(bool(o.get('condensado'))); self.gan.setCurrentIndex(max(0, self.gan.findData(o.get('ganancia', 10))))
        self.rej.setChecked(o.get('rejilla', True))
        for k, c in self.tend.items():
            c.setChecked(k in o.get('tendencias', PRESET_BASE['tendencias']))
        for i, c in enumerate(self.cc):
            c.setChecked(i in o.get('completa_canales', [0]))
        hs = set(o.get('completa_horas', []))
        for i in range(self.horas.count()):
            self.horas.item(i).setCheckState(Qt.Checked if i in hs else Qt.Unchecked)

    def valores(self):
        return {'tipo': self.tipo.currentText(), 'secciones': {k: c.isChecked() for k, c in self.sec.items()}, 'condensado': self.cond.isChecked(),
                'ganancia': self.gan.currentData(), 'rejilla': self.rej.isChecked(), 'tendencias': [k for k, c in self.tend.items() if c.isChecked()],
                'completa_canales': [i for i, c in enumerate(self.cc) if c.isChecked()][:3] or [0],
                'completa_horas': [i for i in range(self.horas.count()) if self.horas.item(i).checkState() == Qt.Checked]}

    def _guardar_tipo(self):
        from PySide6.QtWidgets import QInputDialog
        n, ok = QInputDialog.getText(self, 'Tipo de informe', 'Nombre del tipo de informe:')
        if ok and n.strip():
            v = self.valores(); v.pop('completa_horas', None); self.P[n.strip()] = v; guardar_presets(self.P)
            self.tipo.addItem(n.strip()); self.tipo.setCurrentText(n.strip())

    def _borrar_tipo(self):
        t = self.tipo.currentText()
        if t in ('Estándar', 'Condensado'):
            return
        self.P.pop(t, None); guardar_presets(self.P); self.tipo.removeItem(self.tipo.currentIndex())

    def _fin(self, a):
        self.accion = a; self.accept()


class DialogoAjustes(QDialog):
    """Datos de la institución, logotipo, médico firmante y texto del resumen narrativo."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from .resumen import PLANTILLA_NARRATIVA
        self.s = QSettings('HolterLocal', 'HolterLocal'); self.setWindowTitle('Ajustes del informe'); self.resize(700, 600)
        lay = QVBoxLayout(self); f = QFormLayout()
        self.inst = QLineEdit(self.s.value('institucion', '')); self.cont = QPlainTextEdit(self.s.value('contacto', '')); self.cont.setMaximumHeight(60)
        self.logo = QLineEdit(self.s.value('logo', '')); bl = QPushButton('Elegir…'); bl.clicked.connect(self._logo)
        fl = QHBoxLayout(); fl.addWidget(self.logo, 1); fl.addWidget(bl)
        self.medico = QLineEdit(self.s.value('medico', ''))
        f.addRow('Institución', self.inst); f.addRow('Dirección y contacto', self.cont); f.addRow('Logotipo (PNG o JPG)', fl); f.addRow('Médico que firma', self.medico)
        lay.addLayout(f)
        lay.addWidget(QLabel('Resumen narrativo: el texto entre llaves {…} se reemplaza por el valor del examen (por ejemplo {fc_media}, {v_tot}, {fa_pct}, {pausas}).'))
        self.narr = QPlainTextEdit(self.s.value('narrativa', '') or PLANTILLA_NARRATIVA); lay.addWidget(self.narr, 1)
        br = QPushButton('Volver al texto original'); br.clicked.connect(lambda: self.narr.setPlainText(PLANTILLA_NARRATIVA)); lay.addWidget(br, 0, Qt.AlignLeft)
        _botones(self, lay)

    def _logo(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Logotipo', '', 'Imágenes (*.png *.jpg *.jpeg *.bmp)')
        if p:
            self.logo.setText(p)

    def guardar(self):
        self.s.setValue('institucion', self.inst.text()); self.s.setValue('contacto', self.cont.toPlainText()); self.s.setValue('logo', self.logo.text())
        self.s.setValue('medico', self.medico.text()); self.s.setValue('narrativa', self.narr.toPlainText())


class DialogoTiempos(QDialog):
    def __init__(self, ex, t0, t1, titulo, parent=None):
        super().__init__(parent)
        self.ex = ex; self.setWindowTitle(titulo); f = QFormLayout(self)
        self.a = QLineEdit(ex.reloj(t0).split(' ')[-1]); self.b = QLineEdit(ex.reloj(t1).split(' ')[-1])
        f.addRow('Inicio (HH:MM:SS (día))', self.a); f.addRow('Final', self.b); _botones(self, f)

    def valores(self):
        a, b = self.ex.parse_reloj(self.a.text()), self.ex.parse_reloj(self.b.text())
        return (a, b) if a is not None and b is not None and b > a else None


class DialogoST(QDialog):
    def __init__(self, ex, ev, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Detalles del suceso ST'); f = QFormLayout(self)
        self.nivel = QDoubleSpinBox(); self.nivel.setRange(-2000, 2000); self.nivel.setValue(float(ev.get('level', 0)))
        self.prom = QDoubleSpinBox(); self.prom.setRange(-2000, 2000); self.prom.setValue(float(ev.get('mean', ev.get('level', 0))))
        self.ch = QComboBox()
        for i, c in enumerate(ex.chans):
            self.ch.addItem(ex.names[c] if c < len(ex.names) else f'Canal {i + 1}', i)
        self.ch.setCurrentIndex(int(ev.get('ch', 0)))
        f.addRow('Máximo (µV)', self.nivel); f.addRow('Promedio (µV)', self.prom); f.addRow('Canal principal', self.ch); _botones(self, f)

    def valores(self):
        return {'level': self.nivel.value(), 'mean': self.prom.value(), 'ch': self.ch.currentData()}
