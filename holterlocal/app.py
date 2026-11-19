"""HolterLocal: ventana principal de la aplicación de escritorio."""
import argparse
import csv
import os
import sys
import time

import numpy as np
import json

from PySide6.QtCore import QProcess, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QIcon, QKeySequence, QTextDocument
from PySide6.QtWidgets import (QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox, QProgressBar, QProgressDialog,
                               QSplitter, QTabWidget, QWidget)

from . import __version__
from .estilo import QSS
from .motor import CLS_NAME, CLS_SYM, Examen, Motor, recurso
from .resumen_ha import html_holterai
from .tira import PanelECG
from .vistas import (VistaCriterios, VistaEventos, VistaHistogramas, VistaPlantillas, VistaResumen, VistaTendencias,
                     VistaValidacion)

def es_hscribe(p):
    d = p if os.path.isdir(p) else os.path.dirname(p)
    if not os.path.isfile(os.path.join(d, 'Samples.dat')):
        return False
    return os.path.isdir(p) or not (p.lower().endswith('.hea') or os.path.isfile(os.path.splitext(p)[0] + '.hea'))


FILTRO = ('Exámenes Holter (*.hea *.dat *.edf *.bdf *.ecg *.ish *.isn *.txt *.csv);;'
          'WFDB de PhysioNet (*.hea);;EDF o BDF (*.edf *.bdf);;ISHNE (*.ecg *.ish *.isn);;Todos los archivos (*)')


class Principal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('HolterLocal')
        self.resize(1400, 900)
        self.ex = None
        self.cfg = QSettings('HolterLocal', 'HolterLocal')
        self.motor = Motor(self)
        self.motor.progreso.connect(self._progreso)
        self.motor.fallo.connect(lambda m: QMessageBox.critical(self, 'HolterLocal', m))
        self.dlg = None

        self.panel = PanelECG()
        self.tabs = QTabWidget()
        self.v_res = VistaResumen(); self.v_ev = VistaEventos(); self.v_pl = VistaPlantillas(); self.v_tr = VistaTendencias()
        self.v_hi = VistaHistogramas(); self.v_cr = VistaCriterios(); self.v_va = VistaValidacion()
        for w, n in ((self.v_res, 'Resumen'), (self.v_ev, 'Eventos'), (self.v_pl, 'Plantillas'), (self.v_tr, 'Tendencias'),
                     (self.v_hi, 'Histogramas'), (self.v_cr, 'Criterios'), (self.v_va, 'Validación')):
            self.tabs.addTab(w, n)
        sp = QSplitter(Qt.Vertical); sp.addWidget(self.tabs); sp.addWidget(self.panel); sp.setSizes([480, 420])
        sp.setChildrenCollapsible(False)
        self.setCentralWidget(sp)
        for v in (self.v_ev, self.v_pl, self.v_tr, self.v_hi):
            v.ir.connect(self.ir_a)
        self.v_pl.reetiquetar_plantilla.connect(self.reetiquetar_plantilla)
        self.v_cr.aplicar.connect(self.aplicar_criterios)
        self.panel.tira.reetiquetar.connect(self.reetiquetar)
        self.panel.tira.excluir.connect(self.excluir)
        self.panel.tira.latido_elegido.connect(self._info_latido)
        self._menus()
        self.estado = QLabel('Listo.'); self.statusBar().addWidget(self.estado, 1)
        self.barra = QProgressBar(); self.barra.setMaximumWidth(260); self.barra.setVisible(False); self.statusBar().addPermanentWidget(self.barra)
        self.ha_proc = None; self.ha_timer = QTimer(self); self.ha_timer.timeout.connect(self._ha_poll)
        self.motor.iniciar()

    # ---- menús
    def _menus(self):
        mb = self.menuBar()
        a = mb.addMenu('&Archivo')
        self._acc(a, 'Abrir examen…', self.abrir, QKeySequence.Open)
        self._acc(a, 'Abrir carpeta de examen HScribe…', self.abrir_hscribe)
        self._acc(a, 'Abrir horas crudas de Mortara (H12+)…', self.abrir_horas)
        self._acc(a, 'Abrir el Holter de ejemplo (21 h, PhysioNet)', self.abrir_ejemplo)
        a.addSeparator()
        self._acc(a, 'Exportar resumen en PDF…', self.exportar_pdf)
        self._acc(a, 'Exportar latidos en CSV…', self.exportar_csv)
        a.addSeparator()
        self._acc(a, 'Salir', self.close, QKeySequence.Quit)
        m = mb.addMenu('&Motor')
        g = QActionGroup(self); g.setExclusive(True)
        actual = self.cfg.value('motor', 'mixto')
        for clave, txt in (('mixto', 'Mixto (recomendado): latidos, V y FA de la estación; S, PR, AV, ramas y marcapasos de holterai'),
                           ('holterai', 'Solo holterai (latidos, clases y FA de holterai)'),
                           ('estacion', 'Estación (rápido, solo el motor de la estación)')):
            a = QAction(txt, self, checkable=True); a.setChecked(actual == clave); a.setData(clave); g.addAction(a); m.addAction(a)
        g.triggered.connect(lambda a: self.cfg.setValue('motor', a.data()))
        e = mb.addMenu('&Edición')
        self._acc(e, 'Deshacer todas mis correcciones', self.deshacer_todo)
        h = mb.addMenu('A&yuda')
        self._acc(h, 'Atajos de teclado', self.atajos)
        self._acc(h, 'Acerca de HolterLocal', self.acerca)

    def _acc(self, menu, txt, fn, sc=None):
        a = QAction(txt, self); a.triggered.connect(fn)
        if sc is not None:
            a.setShortcut(sc)
        menu.addAction(a); return a

    # ---- abrir
    def abrir(self):
        d = self.cfg.value('carpeta', os.path.expanduser('~'))
        p, _ = QFileDialog.getOpenFileName(self, 'Abrir examen Holter', d, FILTRO)
        if p:
            self.cfg.setValue('carpeta', os.path.dirname(p))
            if es_hscribe(p):
                self.importar_hscribe(os.path.dirname(p)); return
            self.cargar([p])

    # ---- examen nativo de HScribe (carpeta con Samples.dat): se exporta con la DLL del propio HScribe
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
        self.hs_proc.finished.connect(lambda *a: self._hscribe_fin(base))
        self.hs_proc.start(exe, [d, base, '0', dll, '0'])

    def _hscribe_fin(self, base):
        txt = bytes(self.hs_proc.readAll()).decode('utf-8', 'replace'); rc = self.hs_proc.exitCode(); self.hs_proc = None
        if self.dlg:
            self.dlg.close(); self.dlg = None
        hea = base + '.hea'
        if rc != 0 or not os.path.isfile(hea):
            QMessageBox.critical(self, 'HolterLocal', 'No se pudo leer el examen de HScribe:\n\n' + txt[-1500:]); return
        self.cargar([hea])

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

    def cargar(self, rutas):
        self.t_ini = time.time()
        self.dlg = QProgressDialog('Leyendo el examen…', None, 0, 100, self)
        self.dlg.setWindowTitle('Analizando'); self.dlg.setWindowModality(Qt.WindowModal); self.dlg.setMinimumDuration(0)
        self.dlg.setAutoClose(False); self.dlg.setValue(1); self.dlg.show()
        self.motor.abrir(rutas, lambda r: self._abierto(r, rutas), self._error, settings=self._criterios())

    def _criterios(self):
        v = self.cfg.value('criterios')
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
            self.ex = Examen(r, rutas)
        except Exception as e:  # noqa: BLE001
            self._error(str(e)); return
        if self.dlg:
            self.dlg.close(); self.dlg = None
        self._etiquetar_motor()
        self.setWindowTitle(f'HolterLocal — {self.ex.nombre}')
        self.panel.set_examen(self.ex)
        self.v_cr.cargar(self.ex.A.get('settings') or {})
        self.refrescar()
        self.tabs.setCurrentIndex(0)
        self.estado.setText(f'{self.ex.nombre}: {len(self.ex.t):,} latidos analizados en {time.time() - self.t_ini:.0f} s'.replace(',', '.'))
        if self.cfg.value('motor', 'mixto') in ('mixto', 'holterai'):
            self.lanzar_holterai(r)

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
        for v in (self.v_res, self.v_ev, self.v_pl, self.v_tr, self.v_hi, self.v_va):
            v.mostrar(ex)
        self.panel.tira.update()

    # ---- navegación
    def ir_a(self, t, b=-1):
        self.panel.ir_a(t, b)

    def _info_latido(self, b):
        ex = self.ex
        if not ex or b < 0:
            return
        k = int(ex.cls[b]); rr = ex.rr[b]
        txt = f'Latido {b + 1:,} · {ex.reloj(ex.t[b])} · {CLS_NAME[k]}'.replace(',', '.')
        if ex.edited[b]:
            txt += ' (corregido)'
        if rr == rr:
            txt += f' · RR {rr * 1000:.0f} ms ({60 / rr:.0f} lpm)'
        if ex.prem[b] == ex.prem[b]:
            txt += f' · prematuridad {ex.prem[b] * 100:.0f} %'
        txt += f' · morfología {int(ex.tid[b]) + 1} · prob. V {ex.pV[b]:.2f}'
        self.estado.setText(txt)

    # ---- correcciones
    def _actualizado(self, r):
        sel = self.panel.tira.sel
        self.ex.actualizar(r['A']); self._etiquetar_motor()
        self.refrescar()
        if sel >= 0:
            self.panel.tira.sel = sel; self._info_latido(sel)

    def reetiquetar(self, beats, k):
        if self.ex:
            self.estado.setText(f'Reetiquetando {len(beats)} latido(s) como {CLS_NAME[k].lower()}…')
            self.motor.llamar('relabel', self._actualizado, self._error, beats=[int(b) for b in beats], cls=int(k))

    def reetiquetar_plantilla(self, tid, k):
        if self.ex:
            self.motor.llamar('relabelTemplate', self._actualizado, self._error, tid=int(tid), cls=int(k))

    def excluir(self, t0, t1):
        if self.ex:
            self.motor.llamar('exclude', self._actualizado, self._error, t0=float(t0), t1=float(t1))

    def aplicar_criterios(self, d):
        self.cfg.setValue('criterios', d)
        if self.ex:
            self.motor.llamar('settings', self._actualizado, self._error, settings=d)

    def deshacer_todo(self):
        if self.ex and QMessageBox.question(self, 'HolterLocal', '¿Deshacer todas las correcciones de este examen?') == QMessageBox.Yes:
            self.motor.llamar('resetEdits', self._actualizado, self._error)

    # ---- exportar
    def exportar_pdf(self, ruta=None):
        if not self.ex:
            return
        if not ruta:
            ruta, _ = QFileDialog.getSaveFileName(self, 'Guardar resumen', self.ex.nombre + '_resumen.pdf', 'PDF (*.pdf)')
        if not ruta:
            return
        from PySide6.QtGui import QPageLayout, QPageSize
        from PySide6.QtPrintSupport import QPrinter
        doc = QTextDocument(); doc.setDefaultStyleSheet(self.v_res.document().defaultStyleSheet()); doc.setHtml(self.v_res.html(self.ex))
        pr = QPrinter(QPrinter.HighResolution); pr.setOutputFormat(QPrinter.PdfFormat); pr.setOutputFileName(ruta)
        pr.setPageSize(QPageSize(QPageSize.A4)); pr.setPageOrientation(QPageLayout.Portrait)
        doc.print_(pr)
        self.estado.setText('Resumen guardado en ' + ruta)

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

    # ---- ayuda
    def atajos(self):
        QMessageBox.information(self, 'Atajos de teclado', (
            'En la tira de ECG:\n\n'
            'Clic sobre un latido: elegirlo\n← y →: latido anterior y siguiente\n'
            'N, S, V, F, B, A, P, X: reetiquetar el latido elegido\n'
            'Arrastrar: calibrador (ms, lpm y mV)\nMayúscula + arrastrar: elegir un tramo para etiquetarlo o excluirlo\n'
            'Rueda: avanzar o retroceder\nCtrl + rueda: ganancia\nAvPág / RePág o barra espaciadora: página siguiente o anterior\nEsc: borrar el calibrador'))

    def acerca(self):
        QMessageBox.about(self, 'HolterLocal', f'<b>HolterLocal {__version__}</b><br>Análisis local de Holter de ritmo.<br>'
                          f'Motor {self.motor.version}.<br><br>Herramienta de apoyo: toda conclusión debe revisarla un médico.')

    def closeEvent(self, e):
        self.motor.cerrar(); super().closeEvent(e)


def autoprueba(win, carpeta, rutas):
    """Abre un examen, recorre las pestañas, guarda capturas y sale con código 0 si todo funcionó."""
    os.makedirs(carpeta, exist_ok=True)
    estado = {'paso': 0, 't': time.time(), 'ok': False}
    log = open(os.path.join(carpeta, 'autoprueba.txt'), 'w', encoding='utf-8')

    def escribir(m):
        log.write(m + '\n'); log.flush(); print(m, flush=True)

    def tick():
        app = QApplication.instance()
        if win.ex is None or win.ha_proc is not None or (win.cfg.value('motor', 'mixto') in ('mixto', 'holterai') and not getattr(win.ex, 'ha', None) and time.time() - estado['t'] < 1500):
            if time.time() - estado['t'] > 1500:
                escribir('FALLO: el análisis no terminó a tiempo'); app.exit(2)
            return
        p = estado['paso']; estado['paso'] += 1
        if p == 0:
            ex = win.ex
            escribir(f'motor: {ex.A.get("motor")}')
            escribir(f'examen {ex.nombre}: {ex.dur / 3600:.2f} h, {ex.nsig} canales, {len(ex.t)} latidos, {len(ex.eventos)} eventos, '
                     f'{len(ex.A.get("templates", []))} morfologías, abierto en {time.time() - estado["t"]:.1f} s')
            c = ex.A['counts']; escribir('clases ' + ' '.join(f'{CLS_SYM[k]}={c[k]}' for k in range(10) if c[k]))
            b = int(np.nonzero(ex.cls == 2)[0][0]) if (ex.cls == 2).any() else len(ex.t) // 2
            win.ir_a(float(ex.t[b]), b)
        elif p <= 7:
            win.tabs.setCurrentIndex(p - 1)
            QApplication.processEvents()
            win.grab().save(os.path.join(carpeta, f'pantalla_{p}_{win.tabs.tabText(p - 1)}.png'))
            if p == 7 and win.ex.ref is not None:
                escribir('validación: ' + win.v_va.toPlainText().replace('\n', ' | ')[:600])
        elif p == 8:
            estado['n_v'] = int(win.ex.A['counts'][2]); b = win.panel.tira.sel
            estado['b'] = b; estado['k0'] = int(win.ex.cls[b])
            win.reetiquetar([b], 0 if estado['k0'] != 0 else 2)
        elif p == 9:
            b = estado['b']
            if win.ex.edited[b] != 1:
                estado['paso'] = 9; return
            escribir(f'corrección: latido {b + 1} de {CLS_SYM[estado["k0"]]} a {CLS_SYM[int(win.ex.cls[b])]}; V {estado["n_v"]} → {win.ex.A["counts"][2]}')
            win.exportar_pdf(os.path.join(carpeta, 'resumen.pdf')); win.exportar_csv(os.path.join(carpeta, 'latidos.csv'))
            escribir('pdf %d bytes, csv %d bytes' % (os.path.getsize(os.path.join(carpeta, 'resumen.pdf')), os.path.getsize(os.path.join(carpeta, 'latidos.csv'))))
        else:
            escribir('OK'); log.close(); app.exit(0)

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
