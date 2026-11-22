"""Ajustes tomados de medilog Darwin2: periodo de sueño (día y noche), criterios de arritmia de Darwin, exportar e importar
criterios, atajos de teclado configurables y el panel de flujo de trabajo (manual, capítulos 3 y 7)."""
import json

from PySide6.QtCore import QTime, Qt, Signal
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QHeaderView, QInputDialog, QKeySequenceEdit, QLabel,
                               QListWidget, QListWidgetItem, QPushButton, QTableWidget, QTableWidgetItem, QTimeEdit, QVBoxLayout, QWidget)

from .darwin_comun import SUENO_DEF

# Árbol de decisión de la configuración de arritmias de Darwin2 (valores de fábrica del manual)
PRESET_DARWIN = {'bradyBpm': 46, 'tachyBpm': 131, 'svtBpm': 110, 'svtBeats': 3, 'vtBpm': 111, 'vtBeats': 3, 'pauseSec': 1.88}
NOTA_PRESET = ('Bradicardia: 45 lpm o menos · Taquicardia: más de 130 lpm · TSVP: 110 lpm o más y 3 latidos o más · '
               'TV: 111 lpm o más (si no, ritmo idioventricular) · Pausa: 1880 ms. TV sostenida: 30 s o más.')


class DialogoSueno(QDialog):
    """Periodo de sueño del diario: define la noche del resumen tabular, la VFC diaria y el día/noche de Darwin."""

    def __init__(self, actual, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Periodo de sueño')
        f = QFormLayout(self); self.w = {}
        for k, lab in (('luces', 'Luces apagadas'), ('dormir', 'Se durmió'), ('despierto', 'Despertó')):
            h, m = (actual.get(k) or SUENO_DEF[k]).split(':')[:2]
            t = QTimeEdit(QTime(int(h), int(m))); t.setDisplayFormat('HH:mm'); self.w[k] = t; f.addRow(lab, t)
        f.addRow(QLabel('La noche va desde «Se durmió» hasta «Despertó».'))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        f.addRow(bb)

    def valores(self):
        return {k: t.time().toString('HH:mm') for k, t in self.w.items()}


def exportar_criterios(d, ruta):
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump({'holterlocal_criterios': 1, 'criterios': d}, f, ensure_ascii=False, indent=1)


def importar_criterios(ruta):
    with open(ruta, encoding='utf-8') as f:
        d = json.load(f)
    if isinstance(d, dict) and 'criterios' in d:
        d = d['criterios']
    if not isinstance(d, dict):
        raise ValueError('el archivo no tiene criterios de HolterLocal')
    return d


class DialogoAtajos(QDialog):
    """Atajos de teclado configurables: cada acción de los menús con su combinación de teclas."""

    def __init__(self, acciones, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Atajos de teclado'); self.resize(640, 640)
        self.acciones = acciones
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel('Haz clic en la columna Atajo y presiona la combinación nueva. Supr deja la acción sin atajo.'))
        self.tabla = QTableWidget(len(acciones), 2); self.tabla.setHorizontalHeaderLabels(['Acción', 'Atajo'])
        self.tabla.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch); self.tabla.verticalHeader().setVisible(False)
        self.editores = []
        for i, (clave, a) in enumerate(acciones):
            it = QTableWidgetItem(clave.replace('&', '')); it.setFlags(it.flags() & ~Qt.ItemIsEditable); self.tabla.setItem(i, 0, it)
            ed = QKeySequenceEdit(a.shortcut()); self.tabla.setCellWidget(i, 1, ed); self.editores.append(ed)
        lay.addWidget(self.tabla, 1)
        fila = QHBoxLayout(); b = QPushButton('Restaurar los de fábrica'); b.clicked.connect(self._fabrica); fila.addWidget(b); fila.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); bb.accepted.connect(self._ok); bb.rejected.connect(self.reject)
        fila.addWidget(bb); lay.addLayout(fila)
        self.fabrica = False

    def _fabrica(self):
        self.fabrica = True; self.accept()

    def _ok(self):
        usados = {}
        for (clave, _), ed in zip(self.acciones, self.editores):
            s = ed.keySequence().toString()
            if s and s in usados:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, 'Atajos', f'«{s}» está en «{usados[s]}» y en «{clave.replace("&", "")}».'); return
            if s:
                usados[s] = clave.replace('&', '')
        self.accept()

    def valores(self):
        return {clave: ed.keySequence().toString() for (clave, _), ed in zip(self.acciones, self.editores)}


# Flujo de trabajo: pasos ordenados como en Darwin2 (cada paso abre una pestaña)
FLUJO_DEF = ['Ruido', 'Mín./Máx.', 'Plantillas', 'Cascada', 'Arritmias', 'Bandas', 'Auricular', 'QT y ST', 'VFC', 'Tiras', 'Resumen']


class PanelFlujo(QWidget):
    """Panel lateral con los pasos del análisis. Doble clic abre el paso; las flechas mueven al siguiente."""
    abrir = Signal(str)
    editado = Signal(list)

    def __init__(self, pasos, todos, parent=None):
        super().__init__(parent)
        self.todos = list(todos)
        lay = QVBoxLayout(self); lay.setContentsMargins(4, 4, 4, 4)
        self.lista = QListWidget(); lay.addWidget(self.lista, 1)
        self.lista.itemDoubleClicked.connect(lambda it: self.abrir.emit(it.data(Qt.UserRole)))
        self.lista.itemClicked.connect(lambda it: self.abrir.emit(it.data(Qt.UserRole)))
        fila = QHBoxLayout()
        for txt, fn in (('◀ Anterior', lambda: self.paso(-1)), ('Siguiente ▶', lambda: self.paso(1))):
            b = QPushButton(txt); b.clicked.connect(fn); fila.addWidget(b)
        lay.addLayout(fila)
        b = QPushButton('Editar el flujo…'); b.clicked.connect(self._editar); lay.addWidget(b)
        self.poner(pasos)

    def poner(self, pasos):
        self.pasos = [p for p in pasos if p in self.todos] or list(FLUJO_DEF)
        self.lista.clear()
        for i, p in enumerate(self.pasos):
            it = QListWidgetItem(f'{i + 1}. {p}'); it.setData(Qt.UserRole, p); self.lista.addItem(it)

    def marcar(self, nombre):
        for i in range(self.lista.count()):
            it = self.lista.item(i); f = it.font(); f.setBold(it.data(Qt.UserRole) == nombre); it.setFont(f)
            if it.data(Qt.UserRole) == nombre:
                self.lista.setCurrentRow(i)

    def paso(self, d):
        i = self.lista.currentRow() + d
        if 0 <= i < self.lista.count():
            self.lista.setCurrentRow(i); self.abrir.emit(self.lista.item(i).data(Qt.UserRole))

    def _editar(self):
        txt, ok = QInputDialog.getMultiLineText(self, 'Flujo de trabajo', 'Un paso por línea, en orden. Pasos posibles:\n' + ', '.join(self.todos),
                                                '\n'.join(self.pasos))
        if ok:
            self.poner([x.strip() for x in txt.splitlines() if x.strip()]); self.editado.emit(self.pasos)
