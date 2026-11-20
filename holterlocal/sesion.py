"""Lo que el usuario trabaja en cada examen (correcciones, tiras, diario, paciente, conclusiones y estado) y la
lista local de exámenes. Se guarda en la carpeta de datos del usuario, nunca junto al examen original."""
import copy
import hashlib
import json
import os
import time

from PySide6.QtCore import QStandardPaths

ESTADOS = ['Adquirido', 'Editado', 'Revisado', 'Firmado']

VACIA = {
    'version': 1,
    'paciente': {},
    'diario': [],               # {t, texto}
    'tiras': [],                # {id, t0, dur, ann, auto, clave, chans, pagina, canal}
    'sucesos_usuario': [],      # {k: 1..3, t0, t1}
    'etiquetas_usuario': ['', '', ''],
    'ocultos': [],              # [tipo, t0] de sucesos borrados (ST)
    'st_edit': {},              # 'tipo@t0': {level, ch, ...}
    'borrados': [],             # tiempos de latidos borrados
    'ediciones': [],            # [t, clase] de latidos corregidos o insertados
    'combinadas': [], 'firma': '',
    'exclusiones': [],          # [t0, t1] marcados como artefacto
    'af_usuario': [],           # [t0, t1, 1 agrega | 0 quita]
    'criterios': None,
    'reescaneo': {},            # canales excluidos y duración acortada
    'resumen_override': {},     # clave: texto
    'resumen_grupos_off': [],
    'perfil_override': {},      # 'col@hora': texto
    'perfil_cols_off': [],
    'diagnostico': '', 'notas': '', 'conclusiones': '',
    'estado': 'Adquirido', 'revisado_por': '', 'firmado_por': '', 'fecha_firma': '',
    'informe': {},              # secciones y opciones elegidas para este examen
    'pestanas_ocultas': [],
}


def carpeta_datos():
    d = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation) or os.path.join(os.path.expanduser('~'), '.holterlocal')
    os.makedirs(os.path.join(d, 'examenes'), exist_ok=True)
    return d


def clave_examen(ruta):
    r = os.path.normcase(os.path.abspath(ruta))
    return hashlib.sha1(r.encode('utf-8')).hexdigest()[:20]


class Sesion:
    def __init__(self, ruta, nombre=''):
        self.ruta = ruta
        self.clave = clave_examen(ruta)
        self.archivo = os.path.join(carpeta_datos(), 'examenes', self.clave + '.json')
        self.d = copy.deepcopy(VACIA)
        self.nueva = True
        try:
            with open(self.archivo, encoding='utf-8') as f:
                guardado = json.load(f)
            self.d.update(guardado); self.nueva = False
        except (OSError, ValueError):
            pass
        self.nombre = nombre

    def etiqueta_usuario(self, k):
        t = (self.d.get('etiquetas_usuario') or ['', '', ''])[k - 1]
        return t or f'Definido por el usuario {k}'

    def guardar(self, ex=None):
        if ex is not None:
            self.capturar(ex)
        self.d['guardado'] = time.strftime('%Y-%m-%d %H:%M:%S')
        tmp = self.archivo + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.d, f, ensure_ascii=False)
        os.replace(tmp, self.archivo)
        Registro().anotar(self, ex)

    def capturar(self, ex):
        """Toma del examen el estado de las correcciones de latidos (para reaplicarlas al volver a abrirlo)."""
        import numpy as np
        idx = np.nonzero(ex.edited == 1)[0]
        self.d['ediciones'] = [[round(float(ex.t[b]), 4), int(ex.cls[b])] for b in idx]
        self.d['exclusiones'] = [list(map(float, e)) for e in (ex.A.get('exclusions') or [])]
        self.d['af_usuario'] = [list(map(float, e)) for e in (ex.A.get('afUser') or [])]

    def hay_ediciones(self):
        d = self.d
        return bool(d['ediciones'] or d['borrados'] or d['exclusiones'] or d['af_usuario'] or d['combinadas'])

    def peticion_replay(self):
        d = self.d
        return {'edits': d['ediciones'], 'del': d['borrados'], 'merges': d['combinadas'], 'firma': d.get('firma', ''),
                'exclusions': d['exclusiones'], 'afUser': d['af_usuario']}

    def olvidar_ediciones(self):
        for k in ('ediciones', 'borrados', 'combinadas', 'exclusiones', 'af_usuario', 'resumen_override', 'perfil_override',
                  'perfil_cols_off', 'ocultos', 'st_edit'):
            self.d[k] = copy.deepcopy(VACIA[k])
        self.d['tiras'] = [t for t in self.d['tiras'] if not t.get('auto')]


class Registro:
    """Lista de exámenes abiertos en este computador (para Buscar examen)."""

    def __init__(self):
        self.archivo = os.path.join(carpeta_datos(), 'registro.json')
        try:
            with open(self.archivo, encoding='utf-8') as f:
                self.items = json.load(f)
        except (OSError, ValueError):
            self.items = {}

    def anotar(self, ses, ex=None):
        p = ses.d.get('paciente') or {}
        it = self.items.get(ses.clave, {})
        it.update({'ruta': ses.ruta, 'nombre': ses.nombre or it.get('nombre', ''), 'apellido': p.get('apellido', ''),
                   'nombres': p.get('nombre', ''), 'id': p.get('id', ''), 'estado': ses.d.get('estado', 'Adquirido'),
                   'modificado': time.strftime('%Y-%m-%d %H:%M')})
        if ex is not None:
            it['inicio'] = ex.reloj(0); it['duracion_h'] = round(ex.dur / 3600, 1)
            it['fecha'] = p.get('fecha_registro', '') or it.get('fecha', '')
        self.items[ses.clave] = it
        try:
            tmp = self.archivo + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(self.items, f, ensure_ascii=False)
            os.replace(tmp, self.archivo)
        except OSError:
            pass

    def quitar(self, clave):
        self.items.pop(clave, None)
        try:
            os.remove(os.path.join(carpeta_datos(), 'examenes', clave + '.json'))
        except OSError:
            pass
        with open(self.archivo, 'w', encoding='utf-8') as f:
            json.dump(self.items, f, ensure_ascii=False)
