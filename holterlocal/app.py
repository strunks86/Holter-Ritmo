"""HolterLocal: ventana principal de la aplicación de escritorio."""
import argparse
import csv
import json
import os
import sys
import time
from xml.sax.saxutils import escape

import numpy as np
from PySide6.QtCore import QProcess, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QIcon, QKeySequence
from PySide6.QtWidgets import (QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox, QProgressBar, QProgressDialog,
                               QSplitter, QTabWidget)

from . import __version__
from .dialogos import (DialogoAjustes, DialogoArtefactos, DialogoBuscar, DialogoDiario, DialogoEtiquetas, DialogoFinalizar,
                       DialogoInforme, DialogoIrHora, DialogoPaciente, DialogoReescanear, DialogoST, DialogoTiempos, presets)
from .estilo import QSS
from .motor import CLS_NAME, CLS_SYM, ORDEN_ETIQUETAS, Examen, Motor, recurso
from .perfil import VistaPerfil
from .prospectiva import VistaProspectiva, VistaSuperposicion
from .resumen import VistaResumen, grupos_finales, html_resumen
from .resumen_ha import html_holterai
from .sesion import Sesion
from .tira import DURACIONES, GANANCIAS, PanelECG
from .tiras import POR_DEFECTO, DialogoTira, DialogoTirasAuto, VistaTiras, anotacion_en, generar_auto, nueva_tira
from .vistas import VistaCriterios, VistaEventos, VistaHistogramas, VistaPlantillas, VistaTendencias, VistaValidacion


def es_hscribe(p):
    d = p if os.path.isdir(p) else os.path.dirname(p)
    if not os.path.isfile(os.path.join(d, 'Samples.dat')):
        return False
    return os.path.isdir(p) or not (p.lower().endswith('.hea') or os.path.isfile(os.path.splitext(p)[0] + '.hea'))


FILTRO = ('Exámenes Holter (*.hea *.dat *.edf *.bdf *.ecg *.ish *.isn *.txt *.csv);;'
          'WFDB de PhysioNet (*.hea);;EDF o BDF (*.edf *.bdf);;ISHNE (*.ecg *.ish *.isn);;Todos los archivos (*)')

# pestañas: (atributo, nombre, atajo)
PESTANAS = [('v_pf', 'Perfil', 'Alt+1'), ('v_pr', 'Prospectiva', 'Alt+2'), ('v_tr', 'Tendencias', 'Alt+3'), ('v_su', 'Superposición', 'Alt+4'),
            ('v_pl', 'Plantillas', 'Alt+5'), ('v_hi', 'Histogramas', 'Alt+7'), ('v_ti', 'Tiras', 'Alt+8'), ('v_res', 'Resumen', 'Alt+9'),
            ('v_ev', 'Eventos', 'Alt+0'), ('v_cr', 'Criterios', ''), ('v_va', 'Validación', '')]
OCULTABLES = ('Perfil', 'Prospectiva', 'Tendencias', 'Superposición', 'Plantillas', 'Histogramas', 'Eventos', 'Validación')


class Principal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('HolterLocal')
        self.resize(1440, 920)
        self.ex = None; self.ses = None; self.clave = None; self.rutas = None; self.origen = {}
        self.cfg = QSettings('HolterLocal', 'HolterLocal')
        self.motor = Motor(self)
        self.motor.progreso.connect(self._progreso)
        self.motor.fallo.connect(lambda m: QMessageBox.critical(self, 'HolterLocal', m))
        self.dlg = None
        self.nav = {'titulo': '', 'items': [], 'i': -1}

        self.panel = PanelECG()
        self.tabs = QTabWidget()
        self.v_pf = VistaPerfil(); self.v_pr = VistaProspectiva(); self.v_tr = VistaTendencias(); self.v_su = VistaSuperposicion()
        self.v_pl = VistaPlantillas(); self.v_hi = VistaHistogramas(); self.v_ti = VistaTiras(); self.v_res = VistaResumen()
        self.v_ev = VistaEventos(); self.v_cr = VistaCriterios(); self.v_va = VistaValidacion()
        for at, n, _ in PESTANAS:
            self.tabs.addTab(getattr(self, at), n)
        self.v_pr.set_panel(self.panel)
        self.split = QSplitter(Qt.Vertical); self.split.addWidget(self.tabs); self.split.addWidget(self.panel); self.split.setSizes([470, 450])
        self.split.setChildrenCollapsible(True)
        self.setCentralWidget(self.split)
        # señales de las vistas
        for v in (self.v_ev, self.v_pl, self.v_tr, self.v_hi, self.v_ti, self.v_res, self.v_su):
            v.ir.connect(self.ir_a)
        for v in (self.v_pf, self.v_hi):
            v.nav.connect(self.set_nav)
        for v in (self.v_pf, self.v_res, self.v_ti):
            v.cambio.connect(self._cambio_sesion)
        self.v_pl.reetiquetar_plantillas.connect(self.reetiquetar_plantillas)
        self.v_pl.combinar.connect(self.combinar_plantillas)
        self.v_pl.borrar_plantillas.connect(self.borrar_plantillas)
        self.v_cr.aplicar.connect(self.aplicar_criterios); self.v_cr.reescanear.connect(self.reescanear)
        self.v_ti.excluir.connect(self.excluir); self.v_ti.pedir_auto.connect(self.tiras_auto)
        self.v_pr.estado.connect(lambda m: self.estado.setText(m))
        t = self.panel.tira
        t.reetiquetar.connect(self.reetiquetar); t.excluir.connect(self.excluir); t.latido_elegido.connect(self._info_latido)
        t.insertar.connect(self.insertar); t.borrar.connect(self.borrar_latidos); t.aprender.connect(lambda tid, k: self.reetiquetar_plantillas([tid], k))
        t.mostrar_plantilla.connect(self.ver_plantilla); t.pedir_tira.connect(self.pedir_tira)
        t.agregar_suceso.connect(self.agregar_suceso); t.borrar_suceso.connect(self.borrar_suceso); t.editar_suceso.connect(self.editar_suceso)
        t.detalles_st.connect(self.detalles_st); t.navegar.connect(self.nav_paso)
        t.tiempo_cambiado.connect(self._tiempo)
        self.tabs.currentChanged.connect(self._pestana)
        self._menus()
        self.estado = QLabel('Listo.'); self.statusBar().addWidget(self.estado, 1)
        self.barra = QProgressBar(); self.barra.setMaximumWidth(260); self.barra.setVisible(False); self.statusBar().addPermanentWidget(self.barra)
        self.ha_proc = None; self.ha_timer = QTimer(self); self.ha_timer.timeout.connect(self._ha_poll)
        self.t_guardar = QTimer(self); self.t_guardar.setSingleShot(True); self.t_guardar.timeout.connect(self.guardar)
        self.motor.iniciar()

    # =================================================================== menús
    def _acc(self, menu, txt, fn, sc=None, check=None):
        a = QAction(txt, self); a.triggered.connect(fn)
        if sc:
            a.setShortcut(QKeySequence(sc) if isinstance(sc, str) else sc)
        if check is not None:
            a.setCheckable(True); a.setChecked(check)
        menu.addAction(a); return a

    def _menus(self):
        mb = self.menuBar(); P = self.panel; T = P.tira
        m = mb.addMenu('E&xamen')
        self._acc(m, 'Abrir examen…', self.abrir, QKeySequence.Open)
        self._acc(m, 'Abrir carpeta de examen HScribe…', self.abrir_hscribe)
        self._acc(m, 'Abrir horas crudas de Mortara (H12+)…', self.abrir_horas)
        self._acc(m, 'Abrir el Holter de ejemplo (21 h, PhysioNet)', self.abrir_ejemplo)
        self._acc(m, 'Buscar examen…', self.buscar, 'Ctrl+F')
        m.addSeparator()
        self._acc(m, 'Información del paciente…', self.paciente)
        self._acc(m, 'Volver a escanear…', self.reescanear)
        m.addSeparator()
        self._acc(m, 'Imprimir informe…', self.informe, 'Ctrl+P')
        self._acc(m, 'Imprimir pantalla…', self.imprimir_pantalla, 'Ctrl+Shift+P')
        self._acc(m, 'Exportar informe en PDF…', lambda: self.exportar_pdf())
        self._acc(m, 'Exportar latidos en CSV…', lambda: self.exportar_csv())
        self._acc(m, 'Exportar estadísticas en XML…', lambda: self.exportar_xml())
        m.addSeparator()
        self._acc(m, 'Guardar el trabajo del examen', self.guardar)
        self._acc(m, 'Finalizar examen (estado, revisión y firma)…', self.finalizar)
        self._acc(m, 'Salir', self.close, QKeySequence.Quit)
        e = mb.addMenu('&Editar')
        self._acc(e, 'Lista del diario…', self.diario)
        self._acc(e, 'Criterios de escaneo…', lambda: self.tabs.setCurrentWidget(self.v_cr))
        self._acc(e, 'Ajustes del QTc…', lambda: self.tabs.setCurrentWidget(self.v_cr))
        self._acc(e, 'Editar etiquetas de sucesos…', self.etiquetas_usuario)
        self._acc(e, 'Deshacer artefacto…', self.deshacer_artefacto)
        e.addSeparator()
        self._acc(e, 'Deshacer todas mis correcciones', self.deshacer_todo)
        e.addSeparator()
        self._acc(e, 'Ajustes del informe (institución, logotipo, narrativo)…', self.ajustes_informe)
        n = mb.addMenu('&Navegar')
        self._acc(n, 'Página siguiente\tAvPág', lambda: T.pagina(1)); self._acc(n, 'Página anterior\tRePág', lambda: T.pagina(-1))
        self._acc(n, 'Línea siguiente', lambda: T.linea_sig(1), 'Ctrl+Down'); self._acc(n, 'Línea anterior', lambda: T.linea_sig(-1), 'Ctrl+Up')
        self._acc(n, 'Segundo siguiente', lambda: P.mover(1), 'Alt+Right'); self._acc(n, 'Segundo anterior', lambda: P.mover(-1), 'Alt+Left')
        self._acc(n, 'Una hora adelante', lambda: P.mover(3600), 'Ctrl+PgDown'); self._acc(n, 'Una hora atrás', lambda: P.mover(-3600), 'Ctrl+PgUp')
        self._acc(n, 'Primera página', lambda: T.ir_a(0, centrar=False), 'Ctrl+Home')
        self._acc(n, 'Última página', lambda: T.ir_a(self.ex.dur if self.ex else 0, centrar=False), 'Ctrl+End')
        self._acc(n, 'Ir a una hora…', self.ir_hora, 'Ctrl+H')
        n.addSeparator()
        self._acc(n, 'Suceso siguiente\tTab', lambda: self.nav_paso(1)); self._acc(n, 'Suceso anterior\tMayús+Tab', lambda: self.nav_paso(-1))
        et = mb.addMenu('E&tiqueta')
        for k in ORDEN_ETIQUETAS + [4]:
            self._acc(et, f'{CLS_NAME[k]}\t{CLS_SYM[k]}', lambda _=False, k=k: self.reetiquetar(T.seleccion(), k) if self.ex and T.seleccion() else None)
        et.addSeparator()
        self._acc(et, 'Borrar latido(s)\tSupr', lambda: self.borrar_latidos(T.seleccion()) if self.ex and T.seleccion() else None)
        f = mb.addMenu('&Formato')
        g = f.addMenu('Ganancia')
        for i, (txt, v) in enumerate(GANANCIAS):
            self._acc(g, txt, lambda _=False, i=i: P.gan.setCurrentIndex(i))
        self._acc(f, 'Acercar (página más corta)', lambda: self._zoom(-1), 'Ctrl++')
        self._acc(f, 'Alejar (página más larga)', lambda: self._zoom(1), 'Ctrl+-')
        self.a_rej = self._acc(f, 'Cuadrícula', self._formato, 'Ctrl+G', True)
        self.a_eti = self._acc(f, 'Etiquetas de latidos', self._formato, 'Ctrl+T', True)
        self.a_osc = self._acc(f, 'Fondo oscuro', self._formato, 'Ctrl+D', False)
        self._acc(f, 'Filtro de pantalla 0,5–40 Hz', lambda: P.fil.setChecked(not P.fil.isChecked()))
        v = mb.addMenu('&Vista')
        self._acc(v, 'Perfil', lambda: self.tabs.setCurrentWidget(self.v_pf), 'Ctrl+1')
        for at, nombre, sc in PESTANAS:
            if sc:
                self._acc(v, nombre, lambda _=False, at=at: self._ver(at), sc)
        self._acc(v, 'Solo ECG (pantalla completa)', self._solo_ecg, 'Alt+6')
        v.addSeparator()
        self._acc(v, 'Ventana dividida (ECG a la derecha)', lambda: self._dividir(Qt.Horizontal), 'Ctrl+S')
        self._acc(v, 'Ventana dividida (ECG abajo)', lambda: self._dividir(Qt.Vertical), 'Ctrl+Shift+S')
        self.a_ctx = self._acc(v, 'Contexto', self._contexto, 'Alt+C', False)
        self.m_ctx = v.addMenu('Canal del contexto')
        pm = mb.addMenu('&Pestañas'); self.a_tabs = {}
        for nombre in OCULTABLES:
            self.a_tabs[nombre] = self._acc(pm, nombre, self._pestanas_visibles, None, True)
        h = mb.addMenu('&Herramientas')
        for k, txt in (('latidos', 'Latidos'), ('calibrador', 'Calibrador'), ('tira', 'Tira'), ('suceso', 'Suceso')):
            self._acc(h, txt, lambda _=False, k=k: P.set_herramienta(k))
        mo = mb.addMenu('&Motor')
        gr = QActionGroup(self); gr.setExclusive(True)
        actual = self.cfg.value('motor', 'mixto')
        for clave, txt in (('mixto', 'Mixto (recomendado): latidos, V y FA de la estación; S, PR, AV, ramas y marcapasos de holterai'),
                           ('holterai', 'Solo holterai (latidos, clases y FA de holterai)'),
                           ('estacion', 'Estación (rápido, solo el motor de la estación)')):
            a = QAction(txt, self, checkable=True); a.setChecked(actual == clave); a.setData(clave); gr.addAction(a); mo.addAction(a)
        gr.triggered.connect(lambda a: self.cfg.setValue('motor', a.data()))
        ay = mb.addMenu('A&yuda')
        self._acc(ay, 'Atajos de teclado', self.atajos)
        self._acc(ay, 'Acerca de HolterLocal', self.acerca)
        for sc, fn in (('F7', self._f7), ('F8', self._f8)):
            a = QAction(self); a.setShortcut(QKeySequence(sc)); a.triggered.connect(fn); self.addAction(a)

    def _ver(self, at):
        if self.split.sizes()[0] < 50:
            self.split.setSizes([470, 450])
        self.tabs.setCurrentWidget(getattr(self, at))

    def _solo_ecg(self):
        tot = sum(self.split.sizes()); self.split.setSizes([0, tot]); self.panel.tira.setFocus()

    def _dividir(self, o):
        self.split.setOrientation(o); self.split.setSizes([600, 800] if o == Qt.Horizontal else [470, 450])

    def _contexto(self):
        self.panel.contexto.setVisible(self.a_ctx.isChecked())

    def _formato(self):
        T = self.panel.tira; T.rejilla = self.a_rej.isChecked(); T.etiquetas = self.a_eti.isChecked(); T.oscuro = self.a_osc.isChecked()
        self.v_pr.sup.oscuro = T.oscuro; T.update(); self.panel._sync(T.t0)

    def _zoom(self, d):
        c = self.panel.dur; c.setCurrentIndex(int(np.clip(c.currentIndex() + d, 0, c.count() - 1)))

    def _pestanas_visibles(self):
        ocultas = [n for n, a in self.a_tabs.items() if not a.isChecked()]
        for i in range(self.tabs.count()):
            self.tabs.setTabVisible(i, self.tabs.tabText(i) not in ocultas)
        if self.ses:
            self.ses.d['pestanas_ocultas'] = ocultas; self._cambio_sesion()

    def _f7(self):
        w = self.tabs.currentWidget()
        if w is self.v_su:
            self.v_su.alternar()
        else:
            if w is not self.v_pr:
                self.tabs.setCurrentWidget(self.v_pr)
            self.v_pr.alternar()

    def _f8(self):
        self.v_pr.parar(); self.v_su.parar()

    # =================================================================== abrir
    def abrir(self):
        d = self.cfg.value('carpeta', os.path.expanduser('~'))
        p, _ = QFileDialog.getOpenFileName(self, 'Abrir examen Holter', d, FILTRO)
        if p:
            self.cfg.setValue('carpeta', os.path.dirname(p))
            if es_hscribe(p):
                self.importar_hscribe(os.path.dirname(p)); return
            self.cargar([p])

    def abrir_hscribe(self):
        d = QFileDialog.getExistingDirectory(self, 'Elige la carpeta del examen (la que tiene Samples.dat)', self.cfg.value('carpeta', os.path.expanduser('~')))
        if not d:
            return
        if not es_hscribe(d):
            QMessageBox.information(self, 'HolterLocal', 'En esa carpeta no está Samples.dat de HScribe.'); return
        self.cfg.setValue('carpeta', d); self.importar_hscribe(d)

    def _dll_hscribe(self, d):
        cands = []
        a = os.path.abspath(d)
        while True:
            if os.path.basename(a).lower() == 'modalitymgr':
                cands.append(os.path.join(a, 'Modalities', 'HScribe'))
            cands.append(os.path.join(a, 'ModalityMgr', 'Modalities', 'HScribe'))
            b = os.path.dirname(a)
            if b == a:
                break
            a = b
        guardada = self.cfg.value('dll_hscribe', '')
        if guardada:
            cands.insert(0, guardada)
        for pf in (os.environ.get('ProgramFiles(x86)', ''), os.environ.get('ProgramFiles', '')):
            if pf:
                cands.append(os.path.join(pf, 'Mortara Instrument Inc', 'ModalityMgr', 'Modalities', 'HScribe'))
        for c in cands:
            if c and os.path.isfile(os.path.join(c, 'HW32ExamAPI.dll')):
                return c
        c = QFileDialog.getExistingDirectory(self, 'Elige la carpeta de HScribe que tiene HW32ExamAPI.dll', d)
        if c and os.path.isfile(os.path.join(c, 'HW32ExamAPI.dll')):
            self.cfg.setValue('dll_hscribe', c); return c
        return None

    def importar_hscribe(self, d):
        exe = recurso('hscribe', 'export180.exe')
        if sys.platform != 'win32' or not os.path.isfile(exe):
            QMessageBox.warning(self, 'HolterLocal', 'Los exámenes de HScribe solo se pueden leer en Windows.'); return
        dll = self._dll_hscribe(d)
        if not dll:
            QMessageBox.warning(self, 'HolterLocal', 'No encontré HW32ExamAPI.dll de HScribe, que hace falta para leer la señal.'); return
        nombre = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in os.path.basename(os.path.normpath(d))) or 'examen'
        out = os.path.join(self.motor.tmp, 'hscribe_' + nombre); os.makedirs(out, exist_ok=True)
        base = os.path.join(out, nombre)
        self.dlg = QProgressDialog('Leyendo la señal de HScribe (puede tardar unos minutos)…', None, 0, 0, self)
        self.dlg.setWindowTitle('Importando'); self.dlg.setWindowModality(Qt.WindowModal); self.dlg.setMinimumDuration(0); self.dlg.show()
        self.hs_proc = QProcess(self); self.hs_proc.setWorkingDirectory(dll)
        self.hs_proc.setProcessChannelMode(QProcess.MergedChannels)
        self.hs_proc.finished.connect(lambda *a: self._hscribe_fin(base, d))
        self.hs_proc.start(exe, [d, base, '0', dll, '0'])

    def _hscribe_fin(self, base, d):
        txt = bytes(self.hs_proc.readAll()).decode('utf-8', 'replace'); rc = self.hs_proc.exitCode(); self.hs_proc = None
        if self.dlg:
            self.dlg.close(); self.dlg = None
        hea = base + '.hea'
        if rc != 0 or not os.path.isfile(hea):
            QMessageBox.critical(self, 'HolterLocal', 'No se pudo leer el examen de HScribe:\n\n' + txt[-1500:]); return
        self.cargar([hea], clave=d, origen={'tipo': 'hscribe', 'rutas': [d]})

    def abrir_horas(self):
        d = self.cfg.value('carpeta', os.path.expanduser('~'))
        ps, _ = QFileDialog.getOpenFileNames(self, 'Elige todos los archivos horarios', d, 'Todos los archivos (*)')
        if ps:
            self.cfg.setValue('carpeta', os.path.dirname(ps[0])); self.cargar(sorted(ps))

    def abrir_ejemplo(self):
        p = recurso('ejemplo', '00.hea')
        if not os.path.isfile(p):
            QMessageBox.information(self, 'HolterLocal', 'Esta copia no trae el Holter de ejemplo.'); return
        self.cargar([p])

    def buscar(self):
        d = DialogoBuscar(self)
        if not d.exec() or not d.elegido:
            return
        it = d.elegido; o = it.get('origen') or {}
        rutas = o.get('rutas') or [it.get('ruta')]
        if not rutas or not os.path.exists(rutas[0]):
            QMessageBox.warning(self, 'Buscar examen', 'El examen ya no está en ' + str(rutas[0])); return
        self._accion_tras_abrir = d.accion
        if o.get('tipo') == 'hscribe':
            self.importar_hscribe(rutas[0])
        else:
            self.cargar(rutas)

    def cargar(self, rutas, clave=None, origen=None):
        if self.ex is not None and self.ses is not None:
            self.guardar()
        self.t_ini = time.time(); self.rutas = list(rutas); self.clave = clave or rutas[0]
        self.origen = origen or {'tipo': 'archivos', 'rutas': list(rutas)}
        previa = Sesion(self.clave)
        crit = previa.d.get('criterios') or self._criterios()
        re_ = previa.d.get('reescaneo') or {}
        analysis = {}
        if re_.get('excluir'):
            analysis['channels'] = [c for c in range(16) if c not in re_['excluir']]
        self.dlg = QProgressDialog('Leyendo el examen…', None, 0, 100, self)
        self.dlg.setWindowTitle('Analizando'); self.dlg.setWindowModality(Qt.WindowModal); self.dlg.setMinimumDuration(0)
        self.dlg.setAutoClose(False); self.dlg.setValue(1); self.dlg.show()
        self.motor.n_open += 1
        out = os.path.join(self.motor.tmp, 'ex%d' % self.motor.n_open)
        self.motor.llamar('open', lambda r: self._abierto(r, rutas), self._error, paths=list(rutas), out=out, settings=crit or {},
                          opt={}, analysis=analysis, maxSec=re_.get('maxSec') or 0)

    def _criterios(self):
        v = self.cfg.value('criterios')
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except ValueError:
                v = None
        return v if isinstance(v, dict) else {}

    def _progreso(self, msg, p):
        if self.dlg:
            self.dlg.setLabelText(msg + '…'); self.dlg.setValue(int(p * 100))
        self.estado.setText(msg)

    def _error(self, msg):
        if self.dlg:
            self.dlg.close(); self.dlg = None
        QMessageBox.critical(self, 'HolterLocal', 'No se pudo analizar el examen:\n\n' + msg)

    def _abierto(self, r, rutas):
        try:
            ex = Examen(r, rutas)
        except Exception as e:  # noqa: BLE001
            self._error(str(e)); return
        if self.dlg:
            self.dlg.close(); self.dlg = None
        self.ses = Sesion(self.clave, ex.nombre); self.ses.d['origen'] = self.origen
        ex.sesion = self.ses; ex.sumar_sesion(); self.ex = ex
        self._etiquetar_motor()
        self.setWindowTitle(f'HolterLocal — {ex.nombre}')
        self.panel.set_examen(ex)
        self.v_cr.cargar(ex.A.get('settings') or {})
        self._menu_contexto_canal()
        ocultas = self.ses.d.get('pestanas_ocultas', [])
        for n, a in self.a_tabs.items():
            a.blockSignals(True); a.setChecked(n not in ocultas); a.blockSignals(False)
        for i in range(self.tabs.count()):
            self.tabs.setTabVisible(i, self.tabs.tabText(i) not in ocultas)
        self.set_nav('', [])
        if self.ses.hay_ediciones():
            self.motor.llamar('replay', self._replay_listo, self._error, **self.ses.peticion_replay())
        else:
            self.refrescar()
        self.tabs.setCurrentWidget(self.v_pf)
        self.estado.setText(f'{ex.nombre}: {len(ex.t):,} latidos analizados en {time.time() - self.t_ini:.0f} s'.replace(',', '.'))
        if self.ses.nueva:
            self.guardar()
        accion = getattr(self, '_accion_tras_abrir', '')
        self._accion_tras_abrir = ''
        if accion == 'informe':
            QTimer.singleShot(200, self.informe)
        if self.cfg.value('motor', 'mixto') in ('mixto', 'holterai'):
            self.lanzar_holterai(r)

    def _replay_listo(self, r):
        self._actualizado(r)
        rp = r['A'].get('replay') or {}
        self.estado.setText('Se recuperó el trabajo guardado: %d correcciones, %d latidos insertados, %d borrados.' % (rp.get('nRel', 0), rp.get('nIns', 0), rp.get('nDel', 0)))

    # ---- motor holterai en un proceso aparte
    def lanzar_holterai(self, r):
        rec = r['rec']; out = os.path.dirname(rec['sig'][0]['file'])
        for f in ('progreso.jsonl', 'ha_fin.json'):
            try:
                os.remove(os.path.join(out, f))
            except OSError:
                pass
        pet = {'out': out, 'sig': rec['sig'], 'gain': rec['gain'], 'baseline': rec['baseline'], 'names': rec['names'],
               'fs': rec['fs'], 'chans': r['A']['chans'], 'modelo': 'actual'}
        ruta = os.path.join(out, 'peticion_holterai.json')
        with open(ruta, 'w', encoding='utf-8') as f:
            json.dump(pet, f)
        self.ha_out = out; self.ha_t0 = time.time(); self.ha_pos = 0
        self.ha_proc = QProcess(self)
        self.ha_proc.setProcessChannelMode(QProcess.ForwardedChannels)
        if getattr(sys, 'frozen', False):
            self.ha_proc.start(sys.executable, ['--motor-holterai', ruta])
        else:
            self.ha_proc.start(sys.executable, [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'HolterLocal.py'), '--motor-holterai', ruta])
        self.barra.setVisible(True); self.barra.setValue(0); self.barra.setFormat('holterai %p %')
        self.estado.setText('Lo que ves es el análisis rápido. El motor holterai está trabajando y lo reemplazará al terminar…')
        self.ha_timer.start(500)

    def _ha_poll(self):
        out = self.ha_out
        try:
            with open(os.path.join(out, 'progreso.jsonl'), encoding='utf-8') as f:
                f.seek(self.ha_pos); data = f.read(); self.ha_pos = f.tell()
            for line in data.splitlines():
                m = json.loads(line); self.barra.setValue(int(100 * m['p']))
                self.estado.setText(f'holterai: {m["msg"]}… ({time.time() - self.ha_t0:.0f} s)')
        except (OSError, ValueError):
            pass
        fin = os.path.join(out, 'ha_fin.json')
        if not os.path.isfile(fin):
            if self.ha_proc and self.ha_proc.state() == QProcess.NotRunning and time.time() - self.ha_t0 > 5:
                self.ha_timer.stop(); self.barra.setVisible(False); self.ha_proc = None
                self.estado.setText('El motor holterai se cerró sin terminar; se muestra el análisis rápido.')
            return
        self.ha_timer.stop(); self.barra.setVisible(False); self.ha_proc = None
        try:
            with open(fin, encoding='utf-8') as f:
                r = json.load(f)
            if not r.get('ok'):
                raise RuntimeError(r.get('error', 'error desconocido'))
            with open(os.path.join(out, 'ha.json'), encoding='utf-8') as f:
                ha = json.load(f)
        except Exception as e:  # noqa: BLE001
            self.estado.setText('holterai no pudo analizar este examen; se muestra el análisis rápido.')
            QMessageBox.warning(self, 'HolterLocal', 'El motor holterai falló:\n\n' + str(e)); return
        af = [[e['start_s'], e['end_s']] for e in ha['events'] if e['kind'] == 'afib']
        orden = 'applyMix' if self.cfg.value('motor', 'mixto') == 'mixto' else 'applyExt'
        self.ex.modo = 'mixto' if orden == 'applyMix' else 'holterai'
        self.motor.llamar(orden, lambda res: self._ha_listo(res, ha), self._error, t=os.path.join(out, 'ha_t.bin'),
                          cls=os.path.join(out, 'ha_cls.bin'), af=af, motor='holterai')

    def _ha_listo(self, res, ha):
        self.ex.ha = ha
        self._actualizado(res)
        self.panel.tira.sel = -1
        self.estado.setText(f'Análisis de holterai listo: {ha["n"]:,} latidos en {ha["elapsed"]:.0f} s'.replace(',', '.'))
        if self.ses and self.ses.hay_ediciones():
            self.motor.llamar('replay', self._replay_listo, self._error, **self.ses.peticion_replay())

    def _etiquetar_motor(self):
        ha = getattr(self.ex, 'ha', None)
        if ha:
            if getattr(self.ex, 'modo', '') == 'mixto':
                self.ex.A['motor'] = 'mixto: estación %s (latidos, V, FA) + holterai %s (S, PR, AV, ramas, marcapasos)' % (self.motor.version or '', ha.get('modelo', 'actual'))
            else:
                self.ex.A['motor'] = 'holterai (%s) con morfología y calidad de la estación %s' % (ha.get('modelo', 'actual'), self.motor.version or '')
            self.ex.A['extra_html'] = html_holterai(ha, self.ex)
        else:
            self.ex.A['motor'] = 'estación ' + (self.motor.version or '')

    def refrescar(self):
        ex = self.ex
        for v in (self.v_pf, self.v_pr, self.v_tr, self.v_su, self.v_pl, self.v_hi, self.v_ti, self.v_res, self.v_ev, self.v_va):
            try:
                v.mostrar(ex)
            except Exception as e:  # noqa: BLE001  una vista con problemas no debe tumbar la ventana
                sys.stderr.write(f'vista {type(v).__name__}: {e!r}\n')
        self.panel.contexto.tiras = self.ses.d['tiras'] if self.ses else []
        self.panel.tira.update(); self.panel._sync(self.panel.tira.t0)

    def _menu_contexto_canal(self):
        self.m_ctx.clear(); g = QActionGroup(self.m_ctx)
        for i, c in enumerate(self.ex.chans):
            a = QAction(self.ex.names[c] if c < len(self.ex.names) else f'Canal {i + 1}', self, checkable=True); a.setChecked(i == self.panel.contexto.canal)
            a.triggered.connect(lambda _=False, i=i: self._canal_ctx(i)); g.addAction(a); self.m_ctx.addAction(a)

    def _canal_ctx(self, i):
        self.panel.contexto.canal = i; self.panel.contexto.update()

    # =================================================================== navegación
    def ir_a(self, t, b=-1):
        if self.split.sizes()[1] < 50:
            self.split.setSizes([470, 450])
        self.panel.ir_a(t, b)

    def _tiempo(self, t0):
        self.v_tr.set_tiempo(t0 + self.panel.tira.ancho_seg() / 2)

    def _pestana(self, i):
        if self.tabs.widget(i) is self.v_su and self.ex:
            self.v_su.desde(self.panel.tira.t0)

    def set_nav(self, titulo, items):
        self.nav = {'titulo': titulo, 'items': list(items), 'i': -1}
        if items:
            self.nav_paso(1)
        else:
            self.panel.tira.titulo_nav = ''; self.panel.nav_lbl.setText(''); self.panel.tira.update()

    def nav_paso(self, d):
        if not self.ex:
            return
        n = self.nav
        if not n['items']:
            t = self.panel.tira.t0 + self.panel.tira.ancho_seg() * 0.4
            evs = [e for e in self.ex.eventos if e['type'] not in ('diary',)]
            cand = [e for e in evs if e['t0'] > t + 0.5] if d > 0 else [e for e in evs if e['t0'] < t - 0.5]
            if cand:
                e = cand[0] if d > 0 else cand[-1]
                self.panel.ir_a(e['t0'], int(e['b0']) if e.get('n') and e.get('b0', -1) >= 0 else -1)
                self.panel.tira.titulo_nav = (e.get('label') or e['type']); self.panel.tira.update()
            return
        n['i'] = int(np.clip(n['i'] + d, 0, len(n['items']) - 1))
        t, b = n['items'][n['i']]
        self.panel.ir_a(t, b)
        txt = f'{n["titulo"]} · {n["i"] + 1} de {len(n["items"])}'
        self.panel.tira.titulo_nav = txt; self.panel.nav_lbl.setText(txt); self.panel.tira.update()

    def ir_hora(self):
        if not self.ex:
            return
        d = DialogoIrHora(self.ex, self)
        if d.exec():
            t = d.tiempo()
            if t is None:
                QMessageBox.information(self, 'Ir a una hora', 'No entendí la hora. Usa HH:MM:SS o HH:MM:SS (2).'); return
            self.panel.tira.ir_a(t)

    def _info_latido(self, b):
        ex = self.ex
        if not ex or b < 0 or b >= len(ex.t):
            return
        k = int(ex.cls[b]); rr = ex.rr[b]
        txt = f'Latido {b + 1:,} · {ex.reloj(ex.t[b])} · {CLS_NAME[k]}'.replace(',', '.')
        if ex.edited[b]:
            txt += ' (corregido)'
        if rr == rr:
            txt += f' · RR {rr * 1000:.0f} ms ({60 / rr:.0f} lpm)'
        if ex.prem[b] == ex.prem[b]:
            txt += f' · prematuridad {ex.prem[b] * 100:.0f} %'
        n = len(self.panel.tira.seleccion())
        txt += f' · morfología {int(ex.tid[b]) + 1} · prob. V {ex.pV[b]:.2f}' + (f' · {n} latidos elegidos' if n > 1 else '')
        self.estado.setText(txt)

    def ver_plantilla(self, tid):
        self._ver('v_pl'); self.v_pl.elegir(tid)

    # =================================================================== correcciones
    def _llamar(self, cmd, **kw):
        if self.ex:
            self.motor.llamar(cmd, self._actualizado, self._error, **kw)

    def _actualizado(self, r):
        tira = self.panel.tira; sel = tira.sel
        self.ex.actualizar(r['A']); self._etiquetar_motor()
        if tira.sels:
            tira.sels = {b for b in tira.sels if b < len(self.ex.t)}
        self.refrescar()
        if 0 <= sel < len(self.ex.t):
            tira.sel = sel; self._info_latido(sel)
        self._cambio_sesion()

    def _cambio_sesion(self):
        if self.ses:
            if self.ses.d.get('estado') == 'Adquirido':
                self.ses.d['estado'] = 'Editado'
            self.t_guardar.start(1500)
        self.panel.contexto.tiras = self.ses.d['tiras'] if self.ses else []
        self.panel.contexto.update()

    def guardar(self):
        if self.ses and self.ex:
            try:
                self.ses.guardar(self.ex)
            except OSError as e:
                self.estado.setText('No se pudo guardar el trabajo del examen: ' + str(e))

    def reetiquetar(self, beats, k):
        if self.ex and beats:
            self.estado.setText(f'Etiquetando {len(beats)} latido(s) como {CLS_NAME[k].lower()}…')
            self._llamar('relabel', beats=[int(b) for b in beats], cls=int(k))

    def reetiquetar_plantillas(self, tids, k):
        if self.ex and tids:
            self._llamar('relabelTemplates', tids=[int(t) for t in tids], cls=int(k))

    def combinar_plantillas(self, tids):
        if self.ex and len(tids) > 1:
            self.ses.d['combinadas'].append([int(t) for t in tids]); self.ses.d['firma'] = self.ex.A.get('firma', '')
            self._llamar('mergeTemplates', tids=[int(t) for t in tids])

    def borrar_plantillas(self, tids):
        if not self.ex:
            return
        idx = np.nonzero(np.isin(self.ex.tid, tids))[0]
        if QMessageBox.question(self, 'Plantillas', f'¿Borrar las etiquetas de {len(idx):,} latidos?'.replace(',', '.')) != QMessageBox.Yes:
            return
        self.ses.d['borrados'] += [round(float(self.ex.t[b]), 4) for b in idx]
        self._llamar('deleteTemplateBeats', tids=[int(t) for t in tids])

    def insertar(self, t, k):
        if self.ex:
            self._llamar('insertBeat', t=float(t), cls=int(k))

    def borrar_latidos(self, beats):
        if self.ex and beats:
            self.ses.d['borrados'] += [round(float(self.ex.t[b]), 4) for b in beats if 0 <= b < len(self.ex.t)]
            self.panel.tira.sels = set(); self.panel.tira.sel = -1
            self._llamar('deleteBeats', beats=[int(b) for b in beats])

    def excluir(self, t0, t1):
        if self.ex:
            lista = list(self.ex.A.get('exclusions') or []) + [[float(t0), float(t1)]]
            self.panel.tira.rango = None
            self.motor.llamar('setExclusions', self._tras_artefacto, self._error, list=lista)

    def _tras_artefacto(self, r):
        self._actualizado(r)
        # las tiras automáticas de mínimos y máximos se reemplazan por el siguiente suceso
        L = self.ses.d['tiras']
        if any(t.get('auto') and str(t.get('clave', '')).startswith('mm:') for t in L):
            op = self.ses.d.get('informe', {}).get('auto') or POR_DEFECTO
            nuevas = [t for t in generar_auto(self.ex, op) if str(t.get('clave', '')).startswith('mm:')]
            self.ses.d['tiras'] = [t for t in L if not (t.get('auto') and str(t.get('clave', '')).startswith('mm:'))] + nuevas
            self.v_ti.mostrar(self.ex)

    def aplicar_criterios(self, d):
        self.cfg.setValue('criterios', json.dumps(d))
        if self.ex:
            self.ses.d['criterios'] = d
            self._llamar('settings', settings=d)

    def deshacer_todo(self):
        if self.ex and QMessageBox.question(self, 'HolterLocal', '¿Deshacer todas las correcciones de latidos, artefactos y FA de este examen?\n'
                                            '(Los latidos borrados solo vuelven con Volver a escanear.)') == QMessageBox.Yes:
            for k in ('ediciones', 'exclusiones', 'inclusiones', 'af_usuario', 'combinadas'):
                self.ses.d[k] = []
            self._llamar('resetEdits')

    def deshacer_artefacto(self):
        if not self.ex:
            return
        d = DialogoArtefactos(self.ex, self.ex.A.get('exclusions') or [], self)
        if d.exec():
            self.motor.llamar('setExclusions', self._tras_artefacto, self._error, list=d.lista)

    # ---- sucesos
    def agregar_suceso(self, k, a, b):
        if not self.ex:
            return
        if k == 'artifact':
            self.excluir(a, b)
        elif k in ('af', 'noaf'):
            lista = list(self.ex.A.get('afUser') or []) + [[float(a), float(b), 1 if k == 'af' else 0]]
            self._llamar('setAf', list=lista)
        elif k in ('u1', 'u2', 'u3'):
            self.ses.d['sucesos_usuario'].append({'k': int(k[1]), 't0': float(a), 't1': float(b)})
            self.ex.sumar_sesion(); self.refrescar(); self._cambio_sesion()

    def borrar_suceso(self, ev):
        ex = self.ex; ty = ev['type']
        if ty in ('u1', 'u2', 'u3'):
            self.ses.d['sucesos_usuario'] = [u for u in self.ses.d['sucesos_usuario'] if not (abs(u['t0'] - ev['t0']) < 0.05 and 'u%d' % u['k'] == ty)]
            ex.sumar_sesion(); self.refrescar(); self._cambio_sesion()
        elif ty == 'af':
            self._llamar('setAf', list=list(ex.A.get('afUser') or []) + [[float(ev['t0']) - 0.5, float(ev['t1']) + 0.5, 0]])
        elif ty in ('artifact', 'nosig'):
            excl = [e for e in (ex.A.get('exclusions') or []) if not (e[0] < ev['t1'] and e[1] > ev['t0'])]
            incl = list(ex.A.get('inclusions') or [])
            if len(excl) == len(ex.A.get('exclusions') or []):
                incl.append([float(ev['t0']), float(ev['t1'])])   # artefacto automático: vuelve al análisis
            self.motor.llamar('setExclusions', self._tras_artefacto, self._error, list=excl, incl=incl)
        elif ty in ('stDep', 'stEle'):
            self.ses.d['ocultos'].append([ty, float(ev['t0'])]); ex.sumar_sesion(); self.refrescar(); self._cambio_sesion()

    def editar_suceso(self, ev):
        ex = self.ex; ty = ev['type']
        if ty in ('stDep', 'stEle'):
            self.detalles_st(ev); return
        d = DialogoTiempos(ex, ev['t0'], ev['t1'], 'Editar tiempos del suceso', self)
        if not d.exec() or not d.valores():
            return
        a, b = d.valores()
        if ty in ('u1', 'u2', 'u3'):
            for u in self.ses.d['sucesos_usuario']:
                if abs(u['t0'] - ev['t0']) < 0.05 and 'u%d' % u['k'] == ty:
                    u['t0'], u['t1'] = a, b
            ex.sumar_sesion(); self.refrescar(); self._cambio_sesion()
        elif ty == 'af':
            self._llamar('setAf', list=list(ex.A.get('afUser') or []) + [[float(ev['t0']), float(ev['t1']), 0], [a, b, 1]])
        elif ty in ('artifact', 'nosig'):
            excl = [e for e in (ex.A.get('exclusions') or []) if not (e[0] < ev['t1'] and e[1] > ev['t0'])] + [[a, b]]
            self.motor.llamar('setExclusions', self._tras_artefacto, self._error, list=excl)

    def detalles_st(self, ev):
        d = DialogoST(self.ex, ev, self)
        if d.exec():
            self.ses.d['st_edit']['%s@%.1f' % (ev['type'], ev['t0'])] = d.valores()
            self.ex.sumar_sesion(); self.refrescar(); self._cambio_sesion()

    def etiquetas_usuario(self):
        if not self.ses:
            return
        d = DialogoEtiquetas(self.ses.d.get('etiquetas_usuario', ['', '', '']), self)
        if d.exec():
            self.ses.d['etiquetas_usuario'] = d.valores(); self.ex.sumar_sesion(); self.refrescar(); self._cambio_sesion()

    def diario(self):
        if not self.ex:
            return
        d = DialogoDiario(self.ex, self.ses.d.get('diario', []), self)
        if d.exec():
            self.ses.d['diario'] = d.valores(); self.ex.sumar_sesion(); self.refrescar(); self._cambio_sesion()

    def paciente(self):
        if not self.ses:
            QMessageBox.information(self, 'HolterLocal', 'Primero abre un examen.'); return
        d = DialogoPaciente(self.ses.d.get('paciente') or {}, self)
        if d.exec():
            self.ses.d['paciente'] = d.valores(); self.refrescar(); self._cambio_sesion()

    # ---- tiras
    def pedir_tira(self, t, dur, minutos):
        if not self.ex:
            return
        L, nl = self.panel.tira.lineas()
        chans = list(self.panel.tira.canales())
        t0 = t if minutos else t
        tira = nueva_tira(t0, dur, anotacion_en(self.ex, t0, dur), chans, pagina=minutos)
        d = DialogoTira(self.ex, tira, self)
        if d.exec():
            self.ses.d['tiras'].append(d.valor()); self.v_ti.mostrar(self.ex); self._cambio_sesion()
            self.estado.setText('Tira agregada al informe (%d en total).' % len(self.ses.d['tiras']))

    def tiras_auto(self):
        if not self.ex:
            return
        op = self.ses.d.get('informe', {}).get('auto') or json.loads(self.cfg.value('tiras_auto', '') or 'null') or POR_DEFECTO
        d = DialogoTirasAuto(self.ex, op, self)
        if d.exec():
            op = d.opciones(); self.ses.d.setdefault('informe', {})['auto'] = op; self.cfg.setValue('tiras_auto', json.dumps(op))
            self.aplicar_tiras_auto(op)

    def aplicar_tiras_auto(self, op):
        nuevas = generar_auto(self.ex, op)
        self.ses.d['tiras'] = [t for t in self.ses.d['tiras'] if not t.get('auto')] + nuevas
        self.v_ti.mostrar(self.ex); self._cambio_sesion()
        self.estado.setText(f'{len(nuevas)} tiras automáticas agregadas.')

    # ---- volver a escanear y finalizar
    def reescanear(self):
        if not self.ex:
            return
        d = DialogoReescanear(self.ex, self.ses.d.get('reescaneo') or {}, self)
        if not d.exec():
            return
        self.ses.olvidar_ediciones(); self.ses.d['reescaneo'] = d.valores(); self.guardar()
        self.ses = None; self.ex.sesion = None
        self.cargar(self.rutas, clave=self.clave, origen=self.origen)

    def finalizar(self):
        if not self.ses:
            return
        d = DialogoFinalizar(self.ses, list(presets()), self)
        if not d.exec():
            return
        v = d.valores(); s = self.ses.d
        s['estado'] = v['estado']; s['revisado_por'] = v['revisado_por']; s['firmado_por'] = v['firmado_por']
        if v['estado'] == 'Firmado' and not s.get('fecha_firma'):
            s['fecha_firma'] = time.strftime('%d-%m-%Y %H:%M')
        if v['estado'] != 'Firmado':
            s['fecha_firma'] = ''
        s.setdefault('informe', {})['tipo'] = v['informe']
        self.guardar(); self.v_res.mostrar(self.ex)
        self.estado.setText(f'Examen actualizado a «{v["estado"]}».')
        if d.vista_previa:
            self.informe()
        elif v['imprimir'] == 1 or (v['imprimir'] == 2 and v['estado'] == 'Firmado'):
            self.exportar_pdf()

    # =================================================================== informe e impresión
    def _opciones_informe(self):
        o = dict(presets().get((self.ses.d.get('informe') or {}).get('tipo', 'Estándar'), presets()['Estándar']))
        o.update({k: v for k, v in (self.ses.d.get('informe') or {}).items() if k not in ('auto',)})
        return o

    def informe(self):
        if not self.ex:
            return
        d = DialogoInforme(self.ex, self._opciones_informe(), self)
        if not d.exec():
            return
        op = d.valores(); auto = self.ses.d.get('informe', {}).get('auto')
        self.ses.d['informe'] = dict(op, **({'auto': auto} if auto else {})); self._cambio_sesion()
        from .informe import Informe, impresora
        if d.accion == 'pdf':
            self.exportar_pdf(op=op); return
        from PySide6.QtPrintSupport import QPrintDialog, QPrintPreviewDialog
        pr = impresora()
        if d.accion == 'imprimir':
            dl = QPrintDialog(pr, self)
            if dl.exec():
                Informe(self.ex, op).pintar(pr)
            return
        pv = QPrintPreviewDialog(pr, self); pv.setWindowTitle('Vista previa del informe final'); pv.resize(1100, 900)
        pv.paintRequested.connect(lambda p: Informe(self.ex, op).pintar(p)); pv.exec()

    def exportar_pdf(self, ruta=None, op=None):
        if not self.ex:
            return
        if not ruta:
            ruta, _ = QFileDialog.getSaveFileName(self, 'Guardar informe', self.ex.nombre + '_informe.pdf', 'PDF (*.pdf)')
        if not ruta:
            return
        from .informe import Informe, impresora
        Informe(self.ex, op or self._opciones_informe()).pintar(impresora(ruta))
        self.estado.setText('Informe guardado en ' + ruta)

    def imprimir_pantalla(self):
        if not self.ex:
            return
        from PySide6.QtPrintSupport import QPrintDialog
        from .informe import impresora, imprimir_pantalla
        pr = impresora(); pr.setPageOrientation(pr.pageLayout().Orientation.Landscape)
        if QPrintDialog(pr, self).exec():
            imprimir_pantalla(self.ex, self.panel.tira, pr)

    def exportar_csv(self, ruta=None):
        if not self.ex:
            return
        if not ruta:
            ruta, _ = QFileDialog.getSaveFileName(self, 'Guardar latidos', self.ex.nombre + '_latidos.csv', 'CSV (*.csv)')
        if not ruta:
            return
        ex = self.ex
        with open(ruta, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f, delimiter=';')
            w.writerow(['latido', 'tiempo_s', 'hora', 'clase', 'corregido', 'rr_ms', 'prematuridad_pct', 'morfologia', 'p_ventricular', 'p_artefacto', 'fa'])
            for b in range(len(ex.t)):
                rr = ex.rr[b]; pm = ex.prem[b]
                w.writerow([b + 1, f'{ex.t[b]:.3f}', ex.reloj(ex.t[b]), CLS_SYM[int(ex.cls[b])], int(ex.edited[b]),
                            '' if rr != rr else round(rr * 1000), '' if pm != pm else f'{pm * 100:.1f}', int(ex.tid[b]) + 1,
                            f'{ex.pV[b]:.2f}', f'{ex.pX[b]:.2f}', int(ex.afBeat[b])])
        self.estado.setText('Latidos guardados en ' + ruta)

    def exportar_xml(self, ruta=None):
        if not self.ex:
            return
        if not ruta:
            ruta, _ = QFileDialog.getSaveFileName(self, 'Guardar estadísticas', self.ex.nombre + '_estadisticas.xml', 'XML (*.xml)')
        if not ruta:
            return
        ex = self.ex; pa = self.ses.d.get('paciente') or {}
        x = ['<?xml version="1.0" encoding="UTF-8"?>', f'<holterlocal version="{__version__}">',
             '  <paciente ' + ' '.join(f'{k}="{escape(str(v), {chr(34): "&quot;"})}"' for k, v in pa.items() if v) + '/>',
             f'  <registro nombre="{escape(ex.nombre)}" inicio="{ex.reloj(0)}" duracion_s="{ex.dur:.0f}" canales="{len(ex.chans)}" fs="{ex.fs:g}" estado="{self.ses.d.get("estado", "")}"/>']
        for g, tit, fs, on in grupos_finales(ex):
            x.append(f'  <grupo clave="{g}" titulo="{escape(tit)}" incluido="{int(on)}">')
            for k, lab, v, t, nota in fs:
                x.append(f'    <valor clave="{k}" etiqueta="{escape(lab)}" hora="{ex.reloj(t) if t is not None else ""}" nota="{escape(str(nota or ""))}">{escape(str(v))}</valor>')
            x.append('  </grupo>')
        for t in self.ses.d.get('tiras', []):
            x.append(f'  <tira hora="{ex.reloj(t["t0"])}" duracion_s="{t["dur"]:g}" anotacion="{escape(t.get("ann", ""))}" automatica="{int(bool(t.get("auto")))}"/>')
        x.append(f'  <conclusiones>{escape(self.ses.d.get("conclusiones", ""))}</conclusiones>')
        x.append('</holterlocal>')
        with open(ruta, 'w', encoding='utf-8') as f:
            f.write('\n'.join(x))
        self.estado.setText('Estadísticas guardadas en ' + ruta)

    def ajustes_informe(self):
        d = DialogoAjustes(self)
        if d.exec():
            d.guardar()

    # =================================================================== ayuda
    def atajos(self):
        QMessageBox.information(self, 'Atajos de teclado', (
            'Tira de ECG (herramienta Latidos):\n'
            '  clic: elegir · arrastrar, Mayús+clic o Ctrl+clic: varios · doble clic: ver su plantilla\n'
            '  N S B T V R I E C P D F U: etiquetar · A o X: artefacto · Mayús+letra: insertar latido · Supr: borrar\n'
            '  ← →: latido anterior / siguiente · AvPág / RePág / espacio: página · Inicio / Fin: principio / final\n'
            '  Alt+clic: centrar · Tab / Mayús+Tab: suceso siguiente / anterior · rueda: avanzar · Ctrl+rueda: ganancia\n\n'
            'Calibrador: arrastrar para medir; Ctrl o Mayús + flechas mueven 1 o 10 píxeles; Ctrl + / −: distancia\n\n'
            'Formato: Ctrl+G cuadrícula · Ctrl+T etiquetas · Ctrl+D fondo oscuro · Ctrl++ / Ctrl+− página\n'
            'Vista: Ctrl+1 / Alt+1 Perfil · Alt+2 Prospectiva · Alt+3 Tendencias · Alt+4 Superposición · Alt+5 Plantillas\n'
            '  Alt+6 solo ECG · Alt+7 Histogramas · Alt+8 Tiras · Alt+9 Resumen · Ctrl+S / Ctrl+Mayús+S ventana dividida · Alt+C contexto\n'
            'Prospectiva y superposición: F7 inicio / parar · F8 parar\n'
            'Navegar: Ctrl+H ir a una hora · Ctrl+AvPág / Ctrl+RePág una hora · Alt+→ / Alt+← un segundo\n'
            'Examen: Ctrl+O abrir · Ctrl+F buscar · Ctrl+P informe · Ctrl+Mayús+P imprimir pantalla'))

    def acerca(self):
        QMessageBox.about(self, 'HolterLocal', f'<b>HolterLocal {__version__}</b><br>Análisis local de Holter de ritmo.<br>'
                          f'Motor {self.motor.version}.<br><br>Herramienta de apoyo: toda conclusión debe revisarla un médico.')

    def closeEvent(self, e):
        self.guardar(); self.motor.cerrar(); super().closeEvent(e)


def autoprueba(win, carpeta, rutas):
    """Abre un examen, recorre las pestañas, corrige, inserta y borra latidos, arma tiras e informe, vuelve a abrir el examen
    para comprobar que el trabajo se recupera, guarda capturas y sale con código 0 si todo funcionó."""
    os.makedirs(carpeta, exist_ok=True)
    estado = {'paso': 0, 't': time.time(), 'ok': False}
    log = open(os.path.join(carpeta, 'autoprueba.txt'), 'w', encoding='utf-8')
    n_tabs = win.tabs.count()

    def escribir(m):
        log.write(m + '\n'); log.flush(); print(m, flush=True)

    # en la autoprueba ningún aviso puede quedar esperando un clic: se anota y se sigue
    def aviso(titulo):
        def f(*a, **k):
            escribir('aviso (%s): %s' % (titulo, ' '.join(str(x) for x in a[1:3]).replace('\n', ' ')))
            return QMessageBox.Yes if titulo == 'pregunta' else QMessageBox.Ok
        return staticmethod(f)
    QMessageBox.critical = aviso('error'); QMessageBox.warning = aviso('advertencia')
    QMessageBox.information = aviso('información'); QMessageBox.question = aviso('pregunta')

    def vigilante():
        time.sleep(2600)
        escribir('FALLO: la autoprueba quedó detenida en el paso %d' % estado['paso']); salir(5)
    import threading
    threading.Thread(target=vigilante, daemon=True).start()

    def salir(codigo):
        # salida directa: en Windows un proceso hijo (holterai) o un hilo del motor no debe dejar la app abierta
        try:
            log.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if win.ha_proc is not None:
                win.ha_proc.kill()
            win.motor.cerrar()
        except Exception:  # noqa: BLE001
            pass
        sys.stdout.flush(); os._exit(codigo)

    def esperar():
        estado['paso'] -= 1

    def tick():
        if time.time() - estado['t'] > 1500:
            escribir('FALLO: el análisis no terminó a tiempo (paso %d)' % estado['paso']); salir(2); return
        if win.ex is None or win.ha_proc is not None or (win.cfg.value('motor', 'mixto') in ('mixto', 'holterai') and not getattr(win.ex, 'ha', None)):
            return
        p = estado['paso']; estado['paso'] += 1
        ex = win.ex
        try:
            if p == 0:
                escribir(f'motor: {ex.A.get("motor")}')
                escribir(f'examen {ex.nombre}: {ex.dur / 3600:.2f} h, {ex.nsig} canales, {len(ex.t)} latidos, {len(ex.eventos)} eventos, '
                         f'{len(ex.A.get("templates", []))} morfologías, abierto en {time.time() - estado["t"]:.1f} s')
                c = ex.A['counts']; escribir('clases ' + ' '.join(f'{CLS_SYM[k]}={c[k]}' for k in range(len(c)) if c[k]))
                b = int(np.nonzero(ex.cls == 2)[0][0]) if (ex.cls == 2).any() else len(ex.t) // 2
                win.ir_a(float(ex.t[b]), b)
            elif p <= n_tabs:
                win.tabs.setCurrentIndex(p - 1)
                QApplication.processEvents()
                win.grab().save(os.path.join(carpeta, f'pantalla_{p:02d}_{win.tabs.tabText(p - 1)}.png'))
                if win.tabs.currentWidget() is win.v_va and ex.ref is not None:
                    escribir('validación: ' + win.v_va.toPlainText().replace('\n', ' | ')[:600])
                if win.tabs.currentWidget() is win.v_pr:
                    win.v_pr.parar() if hasattr(win.v_pr, 'parar') else None
            elif p == n_tabs + 1:
                estado['n_v'] = int(ex.A['counts'][2]); b = win.panel.tira.sel
                if b < 0:
                    b = len(ex.t) // 2
                estado['b'] = b; estado['tb'] = float(ex.t[b]); estado['k0'] = int(ex.cls[b])
                win.reetiquetar([b], 0 if estado['k0'] != 0 else 2)
            elif p == n_tabs + 2:
                b = estado['b']
                if ex.edited[b] != 1:
                    return esperar()
                escribir(f'corrección: latido {b + 1} de {CLS_SYM[estado["k0"]]} a {CLS_SYM[int(ex.cls[b])]}; V {estado["n_v"]} → {ex.A["counts"][2]}')
                estado['n'] = len(ex.t); b2 = b + 20
                estado['t_ins'] = round(float(ex.t[b2] + ex.t[b2 + 1]) / 2, 3)
                win.insertar(estado['t_ins'], 1)
            elif p == n_tabs + 3:
                if len(ex.t) != estado['n'] + 1:
                    return esperar()
                i = int(np.argmin(np.abs(ex.t - estado['t_ins'])))
                escribir(f'insertar: latido {CLS_SYM[int(ex.cls[i])]} en {estado["t_ins"]:.2f} s; {estado["n"]} → {len(ex.t)} latidos')
                estado['t_del'] = float(ex.t[estado['b'] + 40])
                win.borrar_latidos([estado['b'] + 40])
            elif p == n_tabs + 4:
                if len(ex.t) != estado['n']:
                    return esperar()
                escribir(f'borrar: latido en {estado["t_del"]:.2f} s; ahora {len(ex.t)} latidos')
                ex.sesion.d['paciente'].update({'apellido': 'Prueba', 'nombre': 'Autoprueba', 'id': '0'})
                ex.sesion.d['diario'].append({'t': float(ex.t[100]), 'texto': 'Palpitaciones'})
                win.agregar_suceso(1, float(ex.t[200]), float(ex.t[200]) + 20)
                win.aplicar_tiras_auto(POR_DEFECTO)
                escribir(f'tiras automáticas: {len(ex.sesion.d["tiras"])}')
                win.tabs.setCurrentWidget(win.v_ti); QApplication.processEvents()
                win.grab().save(os.path.join(carpeta, 'pantalla_tiras_auto.png'))
                op = dict(win._opciones_informe())
                op['secciones'] = {k: True for k in op['secciones']}; op['completa_horas'] = [1]
                win.exportar_pdf(os.path.join(carpeta, 'informe.pdf'), op)
                win.exportar_csv(os.path.join(carpeta, 'latidos.csv')); win.exportar_xml(os.path.join(carpeta, 'resumen.xml'))
                escribir('pdf %d bytes, csv %d bytes, xml %d bytes' % tuple(os.path.getsize(os.path.join(carpeta, f))
                                                                          for f in ('informe.pdf', 'latidos.csv', 'resumen.xml')))
                win.guardar()
                estado['ed'] = len(ex.sesion.d['ediciones']); estado['n_fin'] = len(ex.t); estado['k_fin'] = int(ex.cls[estado['b']])
                estado['ex_viejo'] = ex; estado['t'] = time.time()
                win.cargar(rutas)
            elif p == n_tabs + 5:
                if ex is estado['ex_viejo'] or int((ex.edited == 1).sum()) < estado['ed']:
                    return esperar()
                b = int(np.argmin(np.abs(ex.t - estado['tb'])))
                ok = len(ex.t) == estado['n_fin'] and int(ex.cls[b]) == estado['k_fin'] and len(ex.sesion.d['tiras']) > 0
                escribir(f'volver a abrir: {len(ex.t)} latidos, latido corregido = {CLS_SYM[int(ex.cls[b])]}, '
                         f'{int((ex.edited == 1).sum())} ediciones, {len(ex.sesion.d["tiras"])} tiras, paciente {ex.sesion.d["paciente"].get("apellido", "")}')
                if not ok:
                    escribir('FALLO: el trabajo guardado no se recuperó igual'); salir(3); return
                win.tabs.setCurrentWidget(win.v_res); QApplication.processEvents()
                win.grab().save(os.path.join(carpeta, 'pantalla_reabierto.png'))
            else:
                escribir('OK'); salir(0)
        except Exception:  # noqa: BLE001
            import traceback
            escribir('FALLO en el paso %d:\n%s' % (p, traceback.format_exc())); salir(4)

    win.cargar(rutas)
    tm = QTimer(win); tm.timeout.connect(tick); tm.start(700)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='HolterLocal')
    ap.add_argument('examen', nargs='*')
    ap.add_argument('--autoprueba', metavar='CARPETA')
    ap.add_argument('--motor', choices=['mixto', 'holterai', 'estacion'])
    a = ap.parse_args(argv)
    app = QApplication(sys.argv[:1])
    app.setApplicationName('HolterLocal'); app.setOrganizationName('HolterLocal')
    app.setStyle('Fusion'); app.setStyleSheet(QSS)
    if os.path.isfile(recurso('holterlocal.ico')):
        app.setWindowIcon(QIcon(recurso('holterlocal.ico')))
    if a.motor:
        QSettings('HolterLocal', 'HolterLocal').setValue('motor', a.motor)
    win = Principal()
    app.aboutToQuit.connect(win.motor.cerrar)
    win.show()
    if a.autoprueba:
        rutas = a.examen or [recurso('ejemplo', '00.hea')]
        autoprueba(win, a.autoprueba, rutas)
    elif a.examen:
        QTimer.singleShot(300, lambda: win.cargar(a.examen))
    return app.exec()

