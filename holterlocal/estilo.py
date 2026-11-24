"""Colores y hoja de estilo de la aplicación (aspecto de programa clínico de escritorio)."""

C = {
    'fondo': '#f3f4f6', 'panel': '#ffffff', 'borde': '#d5d9e0', 'texto': '#1f2933', 'texto2': '#5b6573',
    'acento': '#1f5fbf', 'acento2': '#e8f0fc', 'papel': '#fffdfb', 'rejilla1': '#f6d9d9', 'rejilla2': '#e9a9a9',
    'trazo': '#111111', 'calibrador': '#0b57d0', 'fa': '#b45309', 'ref': '#0f766e',
}

# 0 N, 1 S, 2 V, 3 F, 4 artefacto, 5 desconocido, 6 bloqueo de rama, 7 aberrante, 8 escape ventricular,
# 9 ventricular estimulado, 10 R sobre T, 11 interpolado, 12 auricular estimulado, 13 estimulado doble
COLOR_CLASE = ['#374151', '#1d4ed8', '#dc2626', '#92400e', '#9ca3af', '#c2410c', '#0d9488', '#0891b2', '#ea580c',
               '#65a30d', '#e8735e', '#d97706', '#db2777', '#b8860b']

# colores del papel claro y del fondo oscuro de la tira
PAPEL = {
    False: {'papel': '#fffdfb', 'rejilla1': '#f6d9d9', 'rejilla2': '#e9a9a9', 'trazo': '#111111', 'texto': '#1f2933', 'texto2': '#5b6573', 'normal': '#374151'},
    True: {'papel': '#0d1117', 'rejilla1': '#26191b', 'rejilla2': '#4a2a2e', 'trazo': '#e5e7eb', 'texto': '#e5e7eb', 'texto2': '#9ca3af', 'normal': '#e5e7eb'},
}

# barras de sucesos: (tipo, nombre, color, prioridad: 1 es la más alta)
SUCESOS_BARRA = [
    ('artifact', 'Artefacto', '#d946ef', 1), ('af', 'Fibrilación auricular', '#2dd4bf', 2), ('pause', 'Pausa', '#facc15', 3),
    ('sTrig', 'Trigeminismo supraventricular', '#808000', 4), ('sBig', 'Bigeminismo supraventricular', '#06b6d4', 5),
    ('svt', 'Taquicardia supraventricular', '#16a34a', 6), ('vTrig', 'Trigeminismo ventricular', '#fdba74', 7),
    ('vBig', 'Bigeminismo ventricular', '#b07a7a', 8), ('vt', 'Taquicardia ventricular', '#a78bfa', 9),
    ('u3', 'Definido por el usuario 3', '#fb7185', 10), ('u2', 'Definido por el usuario 2', '#c2410c', 11),
    ('u1', 'Definido por el usuario 1', '#b08968', 12), ('tachy', 'Taquicardia', '#d2b48c', 13), ('brady', 'Bradicardia', '#86efac', 14),
    ('stDep', 'Descenso del ST', '#0d9488', 15), ('stEle', 'Elevación del ST', '#be123c', 16),
]
# tipos del motor que se dibujan con la barra de otro
ALIAS_BARRA = {'nosig': 'artifact', 'sRun': 'svt', 'aivr': 'vt', 'ivr': 'vt'}

QSS = f"""
QMainWindow, QWidget {{ background: {C['fondo']}; color: {C['texto']}; font-size: 10pt; }}
QTabWidget::pane {{ border: 1px solid {C['borde']}; background: {C['panel']}; top: -1px; }}
QTabBar::tab {{ background: #e6e9ee; border: 1px solid {C['borde']}; padding: 6px 14px; margin-right: 2px;
               border-top-left-radius: 4px; border-top-right-radius: 4px; color: {C['texto2']}; }}
QTabBar::tab:selected {{ background: {C['panel']}; color: {C['texto']}; font-weight: 600; border-bottom-color: {C['panel']}; }}
QToolBar {{ background: {C['panel']}; border-bottom: 1px solid {C['borde']}; spacing: 4px; padding: 3px; }}
QToolButton {{ padding: 4px 8px; border-radius: 4px; }}
QToolButton:hover {{ background: {C['acento2']}; }}
QPushButton {{ background: {C['panel']}; border: 1px solid {C['borde']}; border-radius: 4px; padding: 5px 12px; }}
QPushButton:hover {{ border-color: {C['acento']}; }}
QPushButton#primario {{ background: {C['acento']}; color: white; border-color: {C['acento']}; font-weight: 600; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTimeEdit {{ background: {C['panel']}; border: 1px solid {C['borde']}; border-radius: 4px; padding: 3px 6px; }}
QTreeWidget, QTableWidget, QListWidget, QTextBrowser {{ background: {C['panel']}; border: 1px solid {C['borde']};
               alternate-background-color: #f8f9fb; selection-background-color: {C['acento2']}; selection-color: {C['texto']}; }}
QHeaderView::section {{ background: #eef0f4; border: none; border-right: 1px solid {C['borde']}; border-bottom: 1px solid {C['borde']}; padding: 4px 6px; font-weight: 600; }}
QStatusBar {{ background: {C['panel']}; border-top: 1px solid {C['borde']}; color: {C['texto2']}; }}
QLabel#reloj {{ font-weight: 600; font-family: Consolas, 'DejaVu Sans Mono', monospace; font-size: 10.5pt; }}
QLabel#ayuda {{ color: {C['texto2']}; font-size: 8.5pt; padding: 0 6px 3px 6px; }}
QLabel#titulo {{ font-size: 12pt; font-weight: 600; }}
QSplitter::handle {{ background: {C['borde']}; }}
QGroupBox {{ border: 1px solid {C['borde']}; border-radius: 4px; margin-top: 14px; padding-top: 8px; background: {C['panel']}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; font-weight: 600; }}
"""

RESUMEN_CSS = """
body { font-family: 'Segoe UI', 'DejaVu Sans', sans-serif; font-size: 10pt; color: #1f2933; }
h2 { font-size: 13pt; margin: 4px 0 2px 0; } h3 { font-size: 11pt; margin: 14px 0 4px 0; color: #1f5fbf; }
table { border-collapse: collapse; } td, th { padding: 3px 8px; border-bottom: 1px solid #e3e6eb; }
th { background: #eef0f4; text-align: left; } td.n { text-align: right; } .m { color: #5b6573; }
"""
