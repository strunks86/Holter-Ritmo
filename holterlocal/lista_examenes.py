"""Lista de exámenes de una carpeta, como «Buscar examen» de HScribe.

El usuario elige una vez la carpeta donde están los Holter. El programa la recorre, encuentra cada examen de HScribe
y muestra una tabla con ID del paciente, apellidos, nombre, estado, fecha y hora, fecha de nacimiento y grupo.

De dónde sale cada dato (en este orden):
1. Lo que el usuario escribió en HolterLocal para ese examen (Información del paciente y estado).
2. CFV.XML del examen, que traen los exámenes archivados o exportados por HScribe (apellido, nombre, ID, fecha de
   nacimiento, hora de inicio y si está firmado).
3. ExamInfo.dat de HScribe: hora de inicio del registro e ID del paciente. No trae el nombre: en el repositorio de
   HScribe (carpetas Exam_N/slot_N) el nombre vive en la base de datos de HScribe, no en los archivos.
4. Otro examen del mismo paciente (mismo ID) que sí tenga nombre.

Todo se lee y se guarda solo en este computador (la caché va a la carpeta de datos del usuario).
"""
import json
import os
import re
import struct
import time
import xml.etree.ElementTree as ET

from PySide6.QtCore import QSettings, QSize, Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from .sesion import Registro, Sesion, carpeta_datos, clave_examen

VERSION_CACHE = 1
MAX_PROF = 6          # niveles de subcarpetas que se recorren bajo la carpeta elegida
ARCHIVOS_EXAMEN = ('samples.dat', 'examinfo.dat', 'beats.dat')


# ============================================================ recorrer la carpeta
def _listar(d):
    try:
        return list(os.scandir(d))
    except OSError:
        return []


def _archivos(d):
    """Nombre en minúsculas → ruta, de los archivos de la carpeta d."""
    out = {}
    for e in _listar(d):
        try:
            if e.is_file():
                out[e.name.lower()] = e.path
        except OSError:
            pass
    return out


def slots(d):
    """Subcarpetas slot_N del repositorio de HScribe, en orden numérico."""
    out = []
    for e in _listar(d):
        try:
            if e.is_dir() and e.name.lower().startswith('slot_'):
                out.append(e.path)
        except OSError:
            pass

    def num(p):
        m = re.search(r'(\d+)$', p)
        return int(m.group(1)) if m else 0
    return sorted(out, key=num)


def archivos_examen(d):
    """Archivos del examen: los de la carpeta y los de sus slot_N (el último slot gana si se repite un nombre)."""
    out = {}
    for s in slots(d):
        out.update(_archivos(s))
    out.update(_archivos(d))
    return out


def es_examen(d):
    a = archivos_examen(d)
    return any(k in a for k in ARCHIVOS_EXAMEN)


def buscar_examenes(raiz, parar=lambda: False):
    """Carpetas de examen bajo raiz. No entra a una carpeta de examen ni a sus slot_N."""
    encontrados = []
    pila = [(raiz, 0)]
    while pila:
        if parar():
            break
        d, prof = pila.pop()
        hijos = []
        for e in _listar(d):
            try:
                if e.is_dir() and not e.name.startswith(('.', '$')):
                    hijos.append(e)
            except OSError:
                pass
        propios = _archivos(d)
        tiene_slots = any(h.name.lower().startswith('slot_') for h in hijos)
        if any(k in propios for k in ARCHIVOS_EXAMEN) or (tiene_slots and es_examen(d)):
            encontrados.append(d); continue
        if prof < MAX_PROF:
            for h in sorted(hijos, key=lambda h: h.name, reverse=True):
                pila.append((h.path, prof + 1))
    return encontrados


# ============================================================ leer los datos del paciente
def _texto(s):
    return (s or '').replace('[comma]', ',').strip()


def leer_cfv(ruta):
    """Campos de CFV.XML (FieldID → valor)."""
    try:
        with open(ruta, 'rb') as f:
            raw = f.read(2_000_000)
        raiz = ET.fromstring(raw)
    except (OSError, ET.ParseError, ValueError):
        return {}
    out = {}
    for df in raiz.iter('DataField'):
        fid = df.get('FieldID')
        fv = df.find('FieldValue')
        if fid and fv is not None:
            out[fid] = _texto(fv.text)
    return out


def _fecha_iso(s):
    """'2012-08-12T21:50:57-05:00' o '1949-05-18' → (aaaa, mm, dd, hh, mi, ss) o None. La hora es la local del registro."""
    m = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2})(?:[T ](\d{1,2}):(\d{2})(?::(\d{2}))?)?', s or '')
    if not m:
        return None
    return tuple(int(x or 0) for x in m.groups())


def _hora_valida(v):
    a, me, di, h, mi, s = v
    return 1990 <= a <= 2100 and 1 <= me <= 12 and 1 <= di <= 31 and 0 <= h < 24 and 0 <= mi < 60 and 0 <= s < 60


def _utf16(b, i, maximo=64):
    """Texto UTF-16 que empieza en b[i] y termina en el primer carácter nulo."""
    out = []
    for j in range(i, min(len(b) - 1, i + 2 * maximo), 2):
        c = b[j] | (b[j + 1] << 8)
        if c == 0:
            break
        out.append(chr(c))
    return ''.join(out)


def _id_valido(s):
    return 1 <= len(s) <= 32 and all(ch.isprintable() for ch in s) and any(ch.isalnum() for ch in s)


def leer_examinfo(ruta):
    """Hora de inicio e ID del paciente de ExamInfo.dat.

    En los exámenes de muestra de HScribe, la hora de inicio son seis enteros de 32 bits (año, mes, día, hora, minuto y
    segundo) y justo después viene el ID del paciente en UTF-16. La posición cambia entre versiones, así que se busca
    la primera fecha válida en vez de usar una posición fija."""
    try:
        with open(ruta, 'rb') as f:
            b = f.read(8192)
    except OSError:
        return {}
    for i in range(0, len(b) - 24 + 1, 4):
        v = struct.unpack_from('<6i', b, i)
        if _hora_valida(v):
            out = {'inicio': v}
            pid = _utf16(b, i + 24).strip()
            if _id_valido(pid):
                out['id'] = pid
            return out
    return {}


def _fmt_fecha(v, hora=True):
    if not v:
        return ''
    a, me, di, h, mi, s = (list(v) + [0] * 6)[:6]
    return f'{di:02d}-{me:02d}-{a:04d} {h}:{mi:02d}:{s:02d}' if hora else f'{di:02d}-{me:02d}-{a:04d}'


def _clave_orden(v):
    return '%04d%02d%02d%02d%02d%02d' % tuple((list(v) + [0] * 6)[:6]) if v else ''


def _firma(a):
    """Lo que cambia si cambian los datos del examen (para no volver a leerlo)."""
    out = []
    for k in ('examinfo.dat', 'cfv.xml', 'samples.dat', 'beats.dat'):
        p = a.get(k)
        if p:
            try:
                st = os.stat(p); out.append([k, int(st.st_mtime), st.st_size])
            except OSError:
                pass
    return out


def leer_examen(d, raiz):
    """Datos de un examen leídos de sus archivos (sin lo que el usuario escribió en HolterLocal)."""
    a = archivos_examen(d)
    it = {'ruta': d, 'id': '', 'apellidos': '', 'nombre': '', 'nacimiento': '', 'inicio': None, 'firmado': False,
          'analizado': 'beats.dat' in a, 'origen_datos': []}
    if 'cfv.xml' in a:
        c = leer_cfv(a['cfv.xml'])
        if c:
            it['origen_datos'].append('CFV.XML')
            it['apellidos'] = c.get('LastName', '')
            it['nombre'] = ' '.join(x for x in (c.get('FirstName', ''), c.get('MiddleName', '')) if x)
            it['id'] = c.get('PatientID', '')
            nac = _fecha_iso(c.get('DOB', ''))
            it['nacimiento'] = _fmt_fecha(nac, hora=False) if nac else ''
            ini = _fecha_iso(c.get('AcquisitionTimestamp', ''))
            if ini and _hora_valida(ini):
                it['inicio'] = list(ini)
            it['firmado'] = c.get('LegallySignedID', '') == '1'
    if 'examinfo.dat' in a and (not it['inicio'] or not it['id']):
        e = leer_examinfo(a['examinfo.dat'])
        if e:
            it['origen_datos'].append('ExamInfo.dat')
            if not it['inicio'] and e.get('inicio'):
                it['inicio'] = list(e['inicio'])
            if not it['id'] and e.get('id'):
                it['id'] = e['id']
    if not it['inicio']:
        p = a.get('samples.dat') or a.get('beats.dat')
        if p:
            try:
                t = time.localtime(os.path.getmtime(p)); it['inicio'] = list(t[:6]); it['inicio_aprox'] = True
            except OSError:
                pass
    rel = os.path.relpath(os.path.dirname(d), raiz) if raiz else ''
    it['grupo'] = '' if rel in ('.', '') else rel.replace('\\', '/')
    it['carpeta'] = os.path.basename(os.path.normpath(d))
    it['firma'] = _firma(a)
    return it


# ============================================================ caché local
def _archivo_cache():
    return os.path.join(carpeta_datos(), 'lista_examenes.json')


def cargar_cache():
    try:
        with open(_archivo_cache(), encoding='utf-8') as f:
            c = json.load(f)
        if c.get('version') == VERSION_CACHE:
            return c
    except (OSError, ValueError):
        pass
    return {'version': VERSION_CACHE, 'examenes': {}}


def guardar_cache(c):
    try:
        tmp = _archivo_cache() + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(c, f, ensure_ascii=False)
        os.replace(tmp, _archivo_cache())
    except OSError:
        pass


def escanear(raiz, cache=None, progreso=lambda n, total: None, parar=lambda: False):
    """Lista de exámenes de raiz. Solo vuelve a leer los que cambiaron desde la última vez."""
    cache = cache if cache is not None else cargar_cache()
    viejos = cache.get('examenes', {})
    dirs = buscar_examenes(raiz, parar)
    out = []
    for i, d in enumerate(dirs):
        if parar():
            break
        k = os.path.normcase(os.path.abspath(d))
        v = viejos.get(k)
        if v is None or v.get('firma') != _firma(archivos_examen(d)):
            v = leer_examen(d, raiz)
        out.append(v); viejos[k] = v
        if i % 10 == 0:
            progreso(i + 1, len(dirs))
    progreso(len(dirs), len(dirs))
    cache['examenes'] = viejos
    return out


def completar(items, registro=None):
    """Mezcla lo leído de los archivos con lo que el usuario escribió en HolterLocal, y comparte nombres por ID."""
    reg = registro if registro is not None else Registro().items
    for it in items:
        it['clave'] = clave_examen(it['ruta'])
        r = reg.get(it['clave']) or {}
        it['estado'] = r.get('estado') or ('Firmado' if it.get('firmado') else 'Adquirido')
        for k_it, k_reg in (('apellidos', 'apellido'), ('nombre', 'nombres'), ('id', 'id'), ('nacimiento', 'nacimiento')):
            if r.get(k_reg):
                it[k_it] = r[k_reg]
    por_id = {}
    for it in items:
        if it.get('id') and (it.get('apellidos') or it.get('nombre')):
            por_id.setdefault(_norm_id(it['id']), it)
    for it in items:
        if it.get('id') and not (it.get('apellidos') or it.get('nombre')):
            o = por_id.get(_norm_id(it['id']))
            if o:
                it['apellidos'] = o['apellidos']; it['nombre'] = o['nombre']
                it['nacimiento'] = it.get('nacimiento') or o.get('nacimiento', '')
    return items


def _norm_id(s):
    return re.sub(r'[^0-9a-z]', '', (s or '').lower())


def filtrar(items, q):
    """Busca en ID, apellidos, nombre y grupo; varias palabras deben aparecer todas. El RUT se compara sin puntos ni guion."""
    palabras = [p for p in (q or '').lower().split() if p]
    if not palabras:
        return list(items)
    out = []
    for it in items:
        texto = ' '.join(str(it.get(k, '')) for k in ('id', 'apellidos', 'nombre', 'grupo', 'carpeta')).lower()
        plano = _norm_id(texto)
        if all(p in texto or (_norm_id(p) and _norm_id(p) in plano) for p in palabras):
            out.append(it)
    return out


# ============================================================ ventana
class _Item(QTableWidgetItem):
    """Celda que ordena por una clave propia (fechas dd-mm-aaaa, RUT sin puntos)."""

    def __lt__(self, otro):
        a, b = self.data(Qt.UserRole + 1), otro.data(Qt.UserRole + 1)
        if a is not None and b is not None:
            return a < b
        return super().__lt__(otro)


class _Escaneo(QThread):
    avance = Signal(int, int)
    listo = Signal(list)

    def __init__(self, raiz, parent=None):
        super().__init__(parent); self.raiz = raiz; self._parar = False

    def run(self):
        cache = cargar_cache()
        items = escanear(self.raiz, cache, lambda n, t: self.avance.emit(n, t), lambda: self._parar)
        guardar_cache(cache)
        self.listo.emit(items)

    def parar(self):
        self._parar = True


class DialogoListaExamenes(QDialog):
    """Ventana «Lista de exámenes»: carpeta guardada, tabla, búsqueda, orden y doble clic para abrir."""
    COLS = ['ID Paciente', 'Apellidos', 'Nombre', 'Estado', 'Fecha/Hora', 'Fec. Nac.', 'Grupo']
    COLORES = {'Firmado': '#1f7a3a', 'Revisado': '#1d5fa8', 'Editado': '#9a6400', 'Adquirido': ''}

    def __init__(self, parent=None, cfg=None):
        super().__init__(parent)
        self.cfg = cfg or QSettings('HolterLocal', 'HolterLocal')
        self.setWindowTitle('Lista de exámenes'); self.resize(1180, 680)
        self.items = []; self.elegido = None; self.accion = ''; self.hilo = None
        lay = QVBoxLayout(self)
        fila = QHBoxLayout()
        fila.addWidget(QLabel('Carpeta de Holter:'))
        self.l_carpeta = QLabel(); self.l_carpeta.setTextInteractionFlags(Qt.TextSelectableByMouse); fila.addWidget(self.l_carpeta, 1)
        b = QPushButton('Elegir carpeta…'); b.clicked.connect(self.elegir_carpeta); fila.addWidget(b)
        self.b_esc = QPushButton('Volver a escanear'); self.b_esc.clicked.connect(self.escanear); fila.addWidget(self.b_esc)
        lay.addLayout(fila)
        fila = QHBoxLayout()
        self.q = QLineEdit(); self.q.setPlaceholderText('Buscar por RUT, apellidos, nombre o grupo'); self.q.setClearButtonEnabled(True)
        self.q.textChanged.connect(self._pinta); self.q.returnPressed.connect(self._abrir_unico)
        fila.addWidget(self.q, 1)
        self.f_estado = QComboBox(); self.f_estado.addItems(['Todos los estados', 'Adquirido', 'Editado', 'Revisado', 'Firmado'])
        self.f_estado.currentIndexChanged.connect(self._pinta); fila.addWidget(self.f_estado)
        self.f_cuando = QComboBox(); self.f_cuando.addItems(['Todas las fechas', 'Hoy', 'Última semana', 'Último mes'])
        self.f_cuando.currentIndexChanged.connect(self._pinta); fila.addWidget(self.f_cuando)
        lay.addLayout(fila)
        self.t = QTableWidget(0, len(self.COLS)); self.t.setHorizontalHeaderLabels(self.COLS)
        self.t.setSelectionBehavior(QAbstractItemView.SelectRows); self.t.setSelectionMode(QAbstractItemView.SingleSelection)
        self.t.setEditTriggers(QAbstractItemView.NoEditTriggers); self.t.setAlternatingRowColors(True)
        self.t.verticalHeader().setVisible(False); self.t.verticalHeader().setDefaultSectionSize(22)
        h = self.t.horizontalHeader(); h.setSectionResizeMode(QHeaderView.Interactive); h.setStretchLastSection(True)
        for j, w in enumerate((130, 210, 170, 90, 160, 100)):
            self.t.setColumnWidth(j, w)
        self.t.setSortingEnabled(True); self.t.sortByColumn(4, Qt.DescendingOrder)
        self.t.cellDoubleClicked.connect(lambda *_: self._ok('editar'))
        lay.addWidget(self.t, 1)
        bot = QHBoxLayout()
        for txt, k in (('Abrir', 'editar'), ('Informe', 'informe'), ('Datos del paciente…', 'paciente')):
            bb = QPushButton(txt); bb.clicked.connect(lambda _=False, k=k: self._ok(k)); bot.addWidget(bb)
        self.l_estado = QLabel(); bot.addWidget(self.l_estado, 1)
        salir = QPushButton('Salir'); salir.clicked.connect(self.reject); bot.addWidget(salir)
        lay.addLayout(bot)
        self._mostrar_carpeta()
        if self.carpeta():
            self.escanear()
        else:
            self.l_estado.setText('Elige la carpeta donde están los Holter.')

    def sizeHint(self):
        return QSize(1180, 680)

    # ---- carpeta y escaneo
    def carpeta(self):
        return self.cfg.value('carpeta_holter', '') or ''

    def _mostrar_carpeta(self):
        self.l_carpeta.setText(self.carpeta() or '(sin elegir)')

    def elegir_carpeta(self):
        d = QFileDialog.getExistingDirectory(self, 'Elige la carpeta donde están los Holter', self.carpeta() or os.path.expanduser('~'))
        if d:
            self.cfg.setValue('carpeta_holter', d); self._mostrar_carpeta(); self.escanear()

    def escanear(self):
        raiz = self.carpeta()
        if not raiz:
            return self.elegir_carpeta()
        if not os.path.isdir(raiz):
            self.l_estado.setText('No encuentro la carpeta ' + raiz + '. ¿Está conectada la unidad?'); return
        if self.hilo is not None:
            return
        self.b_esc.setEnabled(False); self.l_estado.setText('Buscando exámenes…')
        self.hilo = _Escaneo(raiz, self)
        self.hilo.avance.connect(lambda n, t: self.l_estado.setText(f'Leyendo exámenes: {n} de {t}…'))
        self.hilo.listo.connect(self._escaneado)
        self.hilo.start()

    def _escaneado(self, items):
        self.hilo.wait(); self.hilo = None; self.b_esc.setEnabled(True)
        self.items = completar(items)
        self._pinta()

    def done(self, r):
        if self.hilo is not None:
            self.hilo.parar(); self.hilo.wait(5000)
        super().done(r)

    # ---- tabla
    def _visibles(self):
        lista = filtrar(self.items, self.q.text())
        e = self.f_estado.currentText()
        if self.f_estado.currentIndex():
            lista = [it for it in lista if it.get('estado') == e]
        c = self.f_cuando.currentIndex()
        if c:
            dias = {1: 0, 2: 6, 3: 30}[c]
            desde = _clave_orden(list(time.localtime(time.time() - dias * 86400)[:3]) + [0, 0, 0])
            lista = [it for it in lista if _clave_orden(it.get('inicio')) >= desde]
        return lista

    def _pinta(self):
        lista = self._visibles()
        col, orden = self.t.horizontalHeader().sortIndicatorSection(), self.t.horizontalHeader().sortIndicatorOrder()
        self.t.setSortingEnabled(False); self.t.setRowCount(len(lista))
        for r, it in enumerate(lista):
            nac = it.get('nacimiento', '')
            m = re.match(r'(\d{1,2})-(\d{1,2})-(\d{4})', nac)
            valores = [
                (it.get('id', ''), _norm_id(it.get('id', '')).rjust(12, '0')),
                (it.get('apellidos', ''), it.get('apellidos', '').lower()),
                (it.get('nombre', ''), it.get('nombre', '').lower()),
                (it.get('estado', ''), None),
                (_fmt_fecha(it.get('inicio')) + (' ~' if it.get('inicio_aprox') else ''), _clave_orden(it.get('inicio'))),
                (nac, (m.group(3) + m.group(2).zfill(2) + m.group(1).zfill(2)) if m else ''),
                (it.get('grupo', ''), None)]
            for j, (txt, clave) in enumerate(valores):
                c = _Item(txt)
                if clave is not None:
                    c.setData(Qt.UserRole + 1, clave)
                c.setData(Qt.UserRole, it['clave'])
                if j == 3 and self.COLORES.get(txt):
                    c.setForeground(QColor(self.COLORES[txt]))
                if j == 4 and it.get('inicio_aprox'):
                    c.setToolTip('Fecha aproximada: la de los archivos, porque el examen no trae la hora de inicio.')
                if j == 0:
                    c.setToolTip(it.get('ruta', ''))
                self.t.setItem(r, j, c)
        self.t.setSortingEnabled(True); self.t.sortByColumn(col, orden)
        sin_nombre = sum(1 for it in lista if not (it.get('apellidos') or it.get('nombre')))
        msg = f'{len(lista)} de {len(self.items)} exámenes'
        if sin_nombre:
            msg += (f' · {sin_nombre} sin nombre (HScribe guarda el nombre en su base de datos; se puede escribir con '
                    '«Datos del paciente…» y queda para todos los exámenes con ese RUT)')
        self.l_estado.setText(msg)

    def _por_clave(self, k):
        return next((it for it in self.items if it['clave'] == k), None)

    def _abrir_unico(self):
        if self.t.rowCount() == 1:
            self.t.selectRow(0); self._ok('editar')

    def _ok(self, accion):
        r = self.t.currentRow()
        if r < 0:
            return
        it = self._por_clave(self.t.item(r, 0).data(Qt.UserRole))
        if it is None:
            return
        if not os.path.isdir(it['ruta']):
            QMessageBox.warning(self, 'Lista de exámenes', 'El examen ya no está en ' + it['ruta']); return
        if accion == 'paciente':
            return self._editar_paciente(it)
        self.elegido = it; self.accion = accion; self.accept()

    def _editar_paciente(self, it):
        from .dialogos import DialogoPaciente
        ses = Sesion(it['ruta'], it.get('carpeta', ''))
        p = dict(ses.d.get('paciente') or {})
        for k_p, k_it in (('apellido', 'apellidos'), ('nombre', 'nombre'), ('id', 'id'), ('nacimiento', 'nacimiento')):
            p[k_p] = p.get(k_p) or it.get(k_it, '')
        d = DialogoPaciente(p, self)
        if not d.exec():
            return
        ses.d['paciente'] = d.valores()
        if not ses.d.get('origen'):
            ses.d['origen'] = {'tipo': 'hscribe', 'rutas': [it['ruta']]}
        ses.guardar()
        self.items = completar(self.items, Registro().items)
        self._pinta()


def paciente_desde_lista(it):
    """Datos para la Información del paciente de un examen abierto desde la lista."""
    out = {}
    for k_p, k_it in (('apellido', 'apellidos'), ('nombre', 'nombre'), ('id', 'id'), ('nacimiento', 'nacimiento')):
        if it.get(k_it):
            out[k_p] = it[k_it]
    if it.get('inicio') and not it.get('inicio_aprox'):
        out['fecha_registro'] = _fmt_fecha(it['inicio'])
    return out
