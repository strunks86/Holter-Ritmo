"""Trabajador del motor holterai. Corre en un proceso aparte (el mismo .exe con --motor-holterai)
para que la ventana no se congele. Lee la señal que ya dejó el lector, analiza y escribe resultados.

Comunicación por archivos en la carpeta del examen: progreso.jsonl (avance), ha_*.bin y ha.json (resultado)
y ha_fin.json (fin, con error si lo hubo)."""
import json
import os
import sys
import time
import traceback

import numpy as np

CLS = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}
TARGET_EXCL = ('hrv_windows', 'minute_p_qrs', 'hourly')


def raiz_motor():
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, 'motor_holterai')
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'motor_holterai')


def preparar_entorno(datos):
    root = raiz_motor()
    os.environ.setdefault('HOLTERIA_ROOT', root)
    os.environ.setdefault('HOLTERIA_DATA', datos)
    t158 = os.path.join(root, 'models', 'ecgfounder', 't158_full_s0.pt')
    if os.path.isfile(t158):
        os.environ.setdefault('HOLTERIA_T158_PT', t158)
    src = os.path.join(root, 'src')
    if src not in sys.path:
        sys.path.insert(0, src)


def jsonable(o, depth=0):
    if depth > 8:
        return None
    if isinstance(o, dict):
        return {str(k): jsonable(v, depth + 1) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        if len(o) > 2000:
            return None
        return [jsonable(v, depth + 1) for v in o]
    if isinstance(o, np.ndarray):
        return None if o.size > 2000 else jsonable(o.tolist(), depth + 1)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if v == v and abs(v) != float('inf') else None
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def analizar(peticion):
    out = peticion['out']
    prog = open(os.path.join(out, 'progreso.jsonl'), 'a', encoding='utf-8')

    def avisar(msg, p):
        prog.write(json.dumps({'msg': msg, 'p': float(p), 't': time.time()}) + '\n'); prog.flush()

    preparar_entorno(os.path.join(out, 'holterai_datos'))
    avisar('Cargando el motor holterai', 0.0)
    from holterai.legacy.app import pipeline as P
    from holterai.legacy.app.core.models import Recording

    sigs = []
    for s, g, b in zip(peticion['sig'], peticion['gain'], peticion['baseline']):
        x = np.fromfile(s['file'], dtype=np.int16 if s['dtype'] == 'Int16Array' else np.float32, count=s['n'])
        sigs.append(((x.astype(np.float32) - b) / (g or 1.0)).astype(np.float32))
    chans = peticion.get('chans') or list(range(len(sigs)))
    samples = np.stack([sigs[c] for c in chans], axis=1)
    names = [peticion['names'][c] if c < len(peticion['names']) else f'Canal {c + 1}' for c in chans]
    rec = Recording(samples=samples, fs=float(peticion['fs']), leads=names, source='HolterLocal', fmt='int16')
    del sigs

    opts = P.Options(use_ecgfounder=False, ecgf_model=peticion.get('modelo', 'actual'))
    for k, v in (peticion.get('opciones') or {}).items():
        if hasattr(opts, k):
            setattr(opts, k, v)
    t0 = time.time()
    A = P.analyze(rec, opts, progress=lambda m, f: avisar(m, f))
    avisar('Guardando resultados', 0.99)
    n = len(A.beats)
    t = np.fromiter((b.t for b in A.beats), float, n)
    cls = np.fromiter((CLS.get(b.cls, 4) for b in A.beats), np.uint8, n)
    afp = np.fromiter((float(b.features.get('af_prob', 0.0) or 0.0) for b in A.beats), np.float32, n)
    width = np.fromiter((float(b.features.get('width_ms', np.nan) or np.nan) for b in A.beats), np.float32, n)
    t.tofile(os.path.join(out, 'ha_t.bin')); cls.tofile(os.path.join(out, 'ha_cls.bin'))
    afp.tofile(os.path.join(out, 'ha_af.bin')); width.tofile(os.path.join(out, 'ha_ancho.bin'))
    # PR por latido (ms) para el análisis auricular; NaN donde no hay onda P medible
    def _pr(b):
        try:
            v = b.features.get('pr_ms')
            return float(v) if v is not None and b.features.get('p_present', True) else np.nan
        except (TypeError, ValueError):
            return np.nan
    np.fromiter((_pr(b) for b in A.beats), np.float32, n).tofile(os.path.join(out, 'ha_pr.bin'))
    S = {k: v for k, v in A.summary.items() if k not in TARGET_EXCL}
    res = {
        'n': n, 'elapsed': time.time() - t0, 'modelo': opts.ecgf_model,
        'events': [{'kind': e.kind, 'start_s': float(e.start_s), 'end_s': float(e.end_s), 'severity': e.severity,
                    'detail': e.detail, 'n': len(e.beats)} for e in A.events],
        'rhythm': [jsonable(r.to_dict()) for r in A.rhythm][:5000],
        'summary': jsonable(S), 'quality': jsonable(A.quality), 'model_info': jsonable(A.model_info),
    }
    with open(os.path.join(out, 'ha.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False)
    avisar('Listo', 1.0)


def main(arg):
    with open(arg, encoding='utf-8') as f:
        peticion = json.load(f)
    fin = os.path.join(peticion['out'], 'ha_fin.json')
    try:
        analizar(peticion)
        r = {'ok': True}
    except Exception as e:  # noqa: BLE001
        r = {'ok': False, 'error': f'{type(e).__name__}: {e}', 'traza': traceback.format_exc()}
    with open(fin, 'w', encoding='utf-8') as f:
        json.dump(r, f, ensure_ascii=False)
    return 0 if r['ok'] else 1
