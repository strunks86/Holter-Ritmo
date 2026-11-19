# PyInstaller: genera dist/HolterLocal/HolterLocal.exe con el motor, Node y el Holter de ejemplo dentro.
import os
raiz = os.path.abspath(os.path.join(SPECPATH, '..'))
datos = [(os.path.join(raiz, 'holterlocal', 'motor_js'), 'motor_js'), (os.path.join(raiz, 'holterlocal', 'holterlocal.ico'), '.')]
for carpeta in ('node', 'ejemplo'):
    p = os.path.join(raiz, 'holterlocal', carpeta)
    if os.path.isdir(p):
        datos.append((p, carpeta))
fuera = ['PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets', 'PySide6.QtWebEngineQuick', 'PySide6.Qt3DCore',
         'PySide6.Qt3DRender', 'PySide6.QtMultimedia', 'PySide6.QtQuick', 'PySide6.QtQml', 'PySide6.QtCharts',
         'PySide6.QtDataVisualization', 'PySide6.QtPdf', 'PySide6.QtBluetooth', 'PySide6.QtSensors', 'tkinter',
         'matplotlib', 'IPython', 'pyqtgraph.opengl', 'OpenGL']
a = Analysis([os.path.join(raiz, 'HolterLocal.py')], pathex=[raiz], datas=datos, excludes=fuera,
             hiddenimports=['scipy.signal', 'PySide6.QtPrintSupport'])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='HolterLocal', console=False,
          icon=os.path.join(raiz, 'holterlocal', 'holterlocal.ico'))
coll = COLLECT(exe, a.binaries, a.datas, name='HolterLocal')
