# PyInstaller: genera dist/HolterLocal/HolterLocal.exe con el motor, Node y el Holter de ejemplo dentro.
import os
from PyInstaller.utils.hooks import collect_submodules
raiz = os.path.abspath(os.path.join(SPECPATH, '..'))
datos = [(os.path.join(raiz, 'holterlocal', 'motor_js'), 'motor_js'), (os.path.join(raiz, 'holterlocal', 'holterlocal.ico'), '.')]
datos.append((os.path.join(raiz, 'motor_holterai'), 'motor_holterai'))
for carpeta in ('node', 'ejemplo'):
    p = os.path.join(raiz, 'holterlocal', carpeta)
    if os.path.isdir(p):
        datos.append((p, carpeta))
fuera = ['PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets', 'PySide6.QtWebEngineQuick', 'PySide6.Qt3DCore',
         'PySide6.Qt3DRender', 'PySide6.QtMultimedia', 'PySide6.QtQuick', 'PySide6.QtQml', 'PySide6.QtCharts',
         'PySide6.QtDataVisualization', 'PySide6.QtPdf', 'PySide6.QtBluetooth', 'PySide6.QtSensors', 'tkinter',
         'matplotlib', 'IPython', 'torch', 'scipy.tests', 'numpy.tests', 'pyqtgraph.opengl', 'OpenGL']
a = Analysis([os.path.join(raiz, 'HolterLocal.py')], pathex=[raiz], datas=datos, excludes=fuera,
             hiddenimports=['PySide6.QtPrintSupport'] + collect_submodules('scipy') + collect_submodules('omegaconf')
             + collect_submodules('pydantic') + collect_submodules('yaml') + collect_submodules('annotated_types')
             + collect_submodules('attr') + collect_submodules('typing_inspection')
             + [m.strip() for m in open(os.path.join(SPECPATH, 'stdlib_holterai.txt')) if m.strip()])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='HolterLocal', console=False,
          icon=os.path.join(raiz, 'holterlocal', 'holterlocal.ico'))
coll = COLLECT(exe, a.binaries, a.datas, name='HolterLocal')
