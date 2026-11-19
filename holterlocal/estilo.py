"""Colores y hoja de estilo de la aplicación (aspecto de programa clínico de escritorio)."""

C = {
    'fondo': '#f3f4f6', 'panel': '#ffffff', 'borde': '#d5d9e0', 'texto': '#1f2933', 'texto2': '#5b6573',
    'acento': '#1f5fbf', 'acento2': '#e8f0fc', 'papel': '#fffdfb', 'rejilla1': '#f6d9d9', 'rejilla2': '#e9a9a9',
    'trazo': '#111111', 'calibrador': '#0b57d0', 'fa': '#b45309', 'ref': '#0f766e',
}

COLOR_CLASE = ['#374151', '#1d4ed8', '#c81e1e', '#c2410c', '#6b7280', '#6b7280', '#7c3aed', '#0e7490', '#9f1239', '#15803d']

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
