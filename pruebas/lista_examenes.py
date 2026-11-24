"""Prueba de la lista de exámenes con exámenes inventados (sin datos de pacientes reales).

uso: python pruebas/lista_examenes.py [carpeta_para_captura]
Arma una carpeta con exámenes falsos de HScribe (con CFV.XML, solo con ExamInfo.dat y en subcarpetas slot_N),
la escanea y comprueba lo leído. Si se da una carpeta, guarda una captura de la ventana."""
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

CFV = '''<?xml version="1.0" encoding="utf-8"?>
<CustomFormatValues CustomFormatDefName="CorScribe CFD" CustomFormatDefVersion="1">
  <patient>
    <DataField DataType="DTString" FieldID="LastName" FieldKey="14"><FieldValue AutoCalc="false">{ap}</FieldValue></DataField>
    <DataField DataType="DTString" FieldID="FirstName" FieldKey="9"><FieldValue AutoCalc="false">{no}</FieldValue></DataField>
    <DataField DataType="DTString" FieldID="PatientID" FieldKey="16"><FieldValue AutoCalc="false">{id}</FieldValue></DataField>
    <DataField DataType="DTDate" FieldID="DOB" FieldKey="6"><FieldValue AutoCalc="false">{nac}</FieldValue></DataField>
  </patient>
  <exam>
    <DataField DataType="DTDateTime" FieldID="AcquisitionTimestamp" FieldKey="55"><FieldValue AutoCalc="false">{ini}</FieldValue></DataField>
  </exam>
  <report>
    <DataField DataType="DTBoolean" FieldID="LegallySignedID" FieldKey="84"><FieldValue AutoCalc="false">{fir}</FieldValue></DataField>
  </report>
</CustomFormatValues>
'''


def examinfo(ruta, fecha, pid, desfase=328, tam=592):
    """ExamInfo.dat inventado con la misma forma que los de muestra: basura, la fecha en 6 enteros y el ID en UTF-16."""
    b = bytearray(tam); b[0:4] = b'\x00\n\x00\x01'
    b[8:8 + 16] = 'Archive'.encode('utf-16-le') + b'\0\0'
    b[100:120] = b'Abnormal ECG (794.3)'
    struct.pack_into('<6i', b, desfase, *fecha)
    t = pid.encode('utf-16-le') + b'\0\0'
    b[desfase + 24:desfase + 24 + len(t)] = t
    with open(ruta, 'wb') as f:
        f.write(bytes(b))


def armar(raiz):
    def ex(rel, archivos):
        d = os.path.join(raiz, rel); os.makedirs(d, exist_ok=True)
        for nombre, contenido in archivos.items():
            p = os.path.join(d, nombre); os.makedirs(os.path.dirname(p), exist_ok=True)
            if callable(contenido):
                contenido(p)
            else:
                with open(p, 'w' if isinstance(contenido, str) else 'wb') as f:
                    f.write(contenido)
    # 1) examen archivado de HScribe, con CFV.XML
    ex('Exam000', {'Samples.dat': b'x', 'Beats.dat': b'x', 'ExamInfo.dat': lambda p: examinfo(p, (2026, 11, 19, 11, 20, 11), '11111111-1'),
                   'CFV.XML': CFV.format(ap='Prueba Uno', no='Ana', id='11111111-1', nac='1971-07-01', ini='2026-11-19T11:20:11-03:00', fir='0')})
    # 2) repositorio de HScribe: Exam_N con slot_N y sin nombre (está en la base de datos de HScribe)
    ex('FDrepo/Exam_12', {'slot_100/Samples.dat': b'x', 'slot_100/Samples2.dat': b'x',
                          'slot_101/ExamInfo.dat': lambda p: examinfo(p, (2026, 11, 18, 10, 39, 53), '22.222.222-2', desfase=336),
                          'slot_101/Beats.dat': b'x'})
    # 3) mismo paciente que el 1 en otro examen sin nombre: debe tomar el nombre del otro
    ex('FDrepo/Exam_13', {'slot_7/Samples.dat': b'x', 'slot_8/ExamInfo.dat': lambda p: examinfo(p, (2026, 10, 3, 9, 47, 46), '11111111-1')})
    # 4) firmado según CFV.XML
    ex('Otros/Exam002', {'Samples.dat': b'x', 'CFV.XML': CFV.format(ap='Prueba Tres', no='Luis', id='33333333-3', nac='1952-07-18',
                                                                     ini='2026-11-17T10:27:17-03:00', fir='1')})
    # 5) carpeta que no es examen
    ex('Varios', {'notas.txt': 'nada'})


def main():
    from holterlocal import lista_examenes as L
    raiz = tempfile.mkdtemp(prefix='hl_lista_')
    armar(raiz)
    items = L.escanear(raiz, {'version': L.VERSION_CACHE, 'examenes': {}})
    items = L.completar(items, registro={})
    por = {os.path.relpath(it['ruta'], raiz).replace('\\', '/'): it for it in items}
    fallos = []

    def ver(cond, msg):
        print(('ok   ' if cond else 'FALLO ') + msg)
        if not cond:
            fallos.append(msg)
    ver(sorted(por) == ['Exam000', 'FDrepo/Exam_12', 'FDrepo/Exam_13', 'Otros/Exam002'], 'encuentra los 4 exámenes: %s' % sorted(por))
    a = por.get('Exam000', {})
    ver(a.get('apellidos') == 'Prueba Uno' and a.get('nombre') == 'Ana' and a.get('nacimiento') == '01-07-1971', 'CFV.XML: nombre y fecha de nacimiento')
    ver(L._fmt_fecha(a.get('inicio')) == '19-11-2026 11:20:11', 'CFV.XML: hora de inicio')
    b = por.get('FDrepo/Exam_12', {})
    ver(b.get('id') == '22.222.222-2' and L._fmt_fecha(b.get('inicio')) == '18-11-2026 10:39:53', 'ExamInfo.dat en slot_N: ID y hora')
    ver(b.get('grupo') == 'FDrepo' and b.get('analizado'), 'grupo = subcarpeta, analizado por Beats.dat')
    c = por.get('FDrepo/Exam_13', {})
    ver(c.get('apellidos') == 'Prueba Uno', 'nombre compartido por RUT entre exámenes')
    d = por.get('Otros/Exam002', {})
    ver(d.get('estado') == 'Firmado' and a.get('estado') == 'Adquirido', 'estado Firmado / Adquirido')
    ver([it['carpeta'] for it in L.filtrar(items, '222222222')] == ['Exam_12'], 'buscar RUT sin puntos ni guion')
    ver(len(L.filtrar(items, 'prueba ana')) == 2, 'buscar por varias palabras')
    ver(len(L.filtrar(items, 'luis')) == 1, 'buscar por nombre')
    from holterlocal.app import es_hscribe
    ver(es_hscribe(os.path.join(raiz, 'FDrepo', 'Exam_12')) and not es_hscribe(os.path.join(raiz, 'Varios')), 'abre la carpeta Exam_N con slot_N')
    # caché: no vuelve a leer lo que no cambió
    cache = {'version': L.VERSION_CACHE, 'examenes': {}}
    L.escanear(raiz, cache)
    k = next(iter(cache['examenes'])); cache['examenes'][k]['apellidos'] = 'DE LA CACHE'
    ver(any(it['apellidos'] == 'DE LA CACHE' for it in L.escanear(raiz, cache)), 'usa la caché si nada cambió')

    if len(sys.argv) > 1:
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])
        from holterlocal.estilo import QSS
        app.setStyle('Fusion'); app.setStyleSheet(QSS)
        cfg = QSettings(os.path.join(raiz, 'cfg.ini'), QSettings.IniFormat); cfg.setValue('carpeta_holter', raiz)
        dlg = L.DialogoListaExamenes(None, cfg); dlg.show()
        while dlg.hilo is not None:
            app.processEvents()
        app.processEvents()
        ver(dlg.t.rowCount() == 4, 'la ventana muestra 4 filas')
        ver(dlg.t.item(0, 4).text() == '19-11-2026 11:20:11', 'la más reciente arriba')
        os.makedirs(sys.argv[1], exist_ok=True)
        dlg.grab().save(os.path.join(sys.argv[1], 'lista_examenes.png'))
        dlg.q.setText('2222'); app.processEvents()
        ver(dlg.t.rowCount() == 1, 'el buscador filtra la tabla')
        dlg.done(0)
    print('FALLARON %d' % len(fallos) if fallos else 'OK')
    return 1 if fallos else 0


if __name__ == '__main__':
    sys.exit(main())
