"""VFC completa (dominio del tiempo y de la frecuencia, espectrograma, histogramas, TINN, índice circadiano), tabla de QT por
hora con varias fórmulas de corrección y tabla de ST (manual de medilog Darwin2, 4.2.25 y 4.2.32 a 4.2.36)."""
import csv

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QScrollArea,
                               QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)
from scipy.signal import welch

from .darwin_comun import MiniTira, es_noche, fc_evento, fmt

BANDAS = [('ULF', 0.0, 0.0033, '#94a3b8'), ('VLF', 0.0033, 0.04, '#a78bfa'), ('LF', 0.04, 0.15, '#f59e0b'), ('HF', 0.15, 0.4, '#10b981')]
VFC_DEF = {'solo_normales': True, 'excluir_sve': True, 'filtro': True, 'min_ms': 300, 'max_ms': 2000, 'cambio_pct': 20,
           'fc_min': 30, 'fc_max': 200, 'ventana_tabla': 300, 'metodo': 'welch', 'orden': 16, 'tendencia': 'lineal',
           'ventana_f': 300, 'solape': True, 'ancho_fft': 256, 'media': 'fn_media'}


class Perezosa:
    """Las vistas pesadas calculan solo cuando se ven (las correcciones refrescan todas las pestañas)."""

    def mostrar(self, ex):
        self.ex = ex; self.sucio = True
        if self.isVisible():
            self._calcular_seguro()

    def showEvent(self, e):
        super().showEvent(e)
        if getattr(self, 'sucio', False) and getattr(self, 'ex', None) is not None:
            self._calcular_seguro()

    def _calcular_seguro(self):
        self.sucio = False
        try:
            self.calcular()
        except Exception as e:  # noqa: BLE001  una vista con problemas no debe tumbar la ventana
            import sys
            sys.stderr.write(f'{type(self).__name__}: {e!r}\n')


# --------------------------------------------------------------------------- cálculo de VFC
def cfg_vfc(ex):
    c = dict(VFC_DEF)
    if getattr(ex, 'sesion', None):
        c.update(ex.sesion.d.get('vfc_cfg') or {})
    return c


def serie_nn(ex, c):
    """Tiempos (s), intervalos NN (ms) e índice de latido de los intervalos que entran al cálculo."""
    n = len(ex.t)
    if n < 10:
        return np.zeros(0), np.zeros(0), np.zeros(0, int)
    rr = np.asarray(ex.rr, float) * 1000.0; cls = np.asarray(ex.cls)
    sec = np.clip(ex.t.astype(int), 0, max(0, len(ex.excl) - 1))
    ok = np.isfinite(rr) & (cls != 4) & (np.asarray(ex.excl)[sec] == 0)
    ven = np.isin(cls, (2, 3, 8, 10, 11, 9, 12, 13, 5))
    if c['solo_normales']:
        bueno = np.isin(cls, (0, 6))
    else:
        bueno = ~ven & ((cls != 1) if c['excluir_sve'] else True)
    ok &= bueno & np.r_[False, bueno[:-1]] & np.r_[False, ok[:-1] | True]
    ok[0] = False
    if c['filtro']:
        ok &= (rr >= c['min_ms']) & (rr <= c['max_ms']) & (60000.0 / np.maximum(rr, 1) >= c['fc_min']) & (60000.0 / np.maximum(rr, 1) <= c['fc_max'])
        idx = np.nonzero(ok)[0]
        if len(idx) > 2:
            ref = np.convolve(rr[idx], np.ones(5) / 5, mode='same')
            salto = np.abs(rr[idx] - ref) / np.maximum(ref, 1) * 100 > c['cambio_pct']
            ok[idx[salto]] = False
    idx = np.nonzero(ok)[0]
    return ex.t[idx].astype(float), rr[idx], idx


def _rmssd_pnn(nn, idx):
    if len(nn) < 3:
        return np.nan, np.nan
    d = np.diff(nn)[np.diff(idx) == 1]
    if not len(d):
        return np.nan, np.nan
    return float(np.sqrt(np.mean(d ** 2))), float(100.0 * np.mean(np.abs(d) > 50))


def tinn_triangular(nn, bw=7.8125):
    if len(nn) < 50:
        return np.nan, np.nan, None
    edges = np.arange(nn.min() - bw, nn.max() + 2 * bw, bw)
    D, _ = np.histogram(nn, bins=edges); X = int(np.argmax(D)); H = D[X]
    tri = len(nn) / max(1, H)
    D2 = D.astype(float) ** 2; cs = np.r_[0, np.cumsum(D2)]
    mejorL, NL = np.inf, 0
    for N in range(0, X):
        q = np.linspace(0, H, X - N + 1)
        err = cs[N] + np.sum((D[N:X + 1] - q) ** 2)
        if err < mejorL:
            mejorL, NL = err, N
    mejorR, MR = np.inf, len(D) - 1
    for M in range(X + 1, len(D)):
        q = np.linspace(H, 0, M - X + 1)
        err = (cs[-1] - cs[M + 1]) + np.sum((D[X:M + 1] - q) ** 2)
        if err < mejorR:
            mejorR, MR = err, M
    cent = (edges[:-1] + edges[1:]) / 2
    return float(tri), float((MR - NL) * bw), (cent[NL], cent[X], cent[MR], H)


def burg(x, orden):
    """Coeficientes AR (convención 1 + a1 z^-1 + …) y varianza del ruido por el método de Burg."""
    x = np.asarray(x, float); N = len(x); a = np.array([1.0]); E = np.dot(x, x) / N
    f = x[1:].copy(); b = x[:-1].copy()
    for m in range(orden):
        den = np.dot(f, f) + np.dot(b, b)
        if den <= 0:
            break
        k = -2.0 * np.dot(f, b) / den
        a = np.r_[a, 0.0] + k * np.r_[a, 0.0][::-1]
        E *= (1 - k * k)
        f, b = f[1:] + k * b[1:], b[:-1] + k * f[:-1]
        if len(f) < 2:
            break
    return a, E


def espectro(x, fs, c):
    x = np.asarray(x, float)
    if c['tendencia'] == 'lineal':
        tt = np.arange(len(x)); x = x - np.polyval(np.polyfit(tt, x, 1), tt)
    elif c['tendencia'] == 'media':
        x = x - x.mean()
    if c['metodo'] == 'burg':
        a, E = burg(x, int(c['orden']))
        f = np.linspace(0, 0.5, 257)
        z = np.exp(-2j * np.pi * np.outer(f, np.arange(len(a))) / fs)
        P = 2 * E / fs / np.abs(z @ a) ** 2
        return f, P
    nper = int(min(len(x), max(32, c['ancho_fft'])))
    f, P = welch(x, fs=fs, nperseg=nper, noverlap=nper // 2, detrend=False)
    return f, P


def potencia(f, P, a, b):
    m = (f >= a) & (f < b)
    return float(np.trapezoid(P[m], f[m])) if m.sum() > 1 else 0.0


def calcular_vfc(ex):
    """Todos los valores de VFC del examen con la configuración guardada."""
    c = cfg_vfc(ex)
    tt, nn, idx = serie_nn(ex, c)
    R = {'cfg': c, 'tt': tt, 'nn': nn, 'idx': idx, 'n': len(nn), 'validos_pct': 100.0 * len(nn) / max(1, len(ex.t))}
    if len(nn) < 100:
        return R
    R['sdnn'] = float(np.std(nn)); R['rmssd'], R['pnn50'] = _rmssd_pnn(nn, idx)
    R['media_nn'] = float(np.mean(nn))
    # ventanas de la tabla
    W = int(c['ventana_tabla']); g = (tt // W).astype(int); filas = []
    for k in range(int(ex.dur // W) + 1):
        m = g == k
        b0, b1 = np.searchsorted(ex.t, [k * W, (k + 1) * W])
        if m.sum() < 10:
            filas.append({'t0': k * W, 'qrs': int(b1 - b0), 'valido': 100.0 * m.sum() / max(1, b1 - b0)}); continue
        x = nn[m]; rm, pn = _rmssd_pnn(x, idx[m])
        filas.append({'t0': k * W, 'qrs': int(b1 - b0), 'fcmin': 60000 / x.max(), 'fcmed': 60000 / x.mean(), 'fcmax': 60000 / x.min(),
                      'media': float(x.mean()), 'sdnn': float(np.std(x)), 'rmssd': rm, 'pnn50': pn, 'valido': 100.0 * m.sum() / max(1, b1 - b0)})
    R['tabla'] = filas
    w5 = (tt // 300).astype(int); medias, sds = [], []
    for k in np.unique(w5):
        x = nn[w5 == k]
        if len(x) >= 30:
            medias.append(x.mean()); sds.append(np.std(x))
    R['sdann'] = float(np.std(medias)) if len(medias) > 2 else np.nan
    R['sdnni'] = float(np.mean(sds)) if sds else np.nan
    R['tri'], R['tinn'], R['tri_geo'] = tinn_triangular(nn)
    noche = es_noche(ex, tt)
    R['fc_dia'] = 60000 / nn[~noche].mean() if (~noche).sum() > 50 else np.nan
    R['fc_noche'] = 60000 / nn[noche].mean() if noche.sum() > 50 else np.nan
    R['circadiano'] = R['fc_dia'] / R['fc_noche'] if R['fc_noche'] == R['fc_noche'] and R['fc_noche'] else np.nan
    # dominio de la frecuencia: ventanas cortas a 4 Hz
    fs = 4.0; Wf = float(c['ventana_f']); paso = Wf / 2 if c['solape'] else Wf
    filas_f = []; spec = []; ts = []
    t = 0.0
    while t + Wf <= ex.dur + 1:
        m = (tt >= t) & (tt < t + Wf)
        if m.sum() >= 0.6 * Wf / max(0.3, np.mean(nn) / 1000):
            ti = np.arange(t, t + Wf, 1 / fs); x = np.interp(ti, tt[m], nn[m])
            f, P = espectro(x, fs, c)
            d = {'t0': t, 'VLF': potencia(f, P, 0.0033, 0.04), 'LF': potencia(f, P, 0.04, 0.15), 'HF': potencia(f, P, 0.15, 0.4),
                 'Total': potencia(f, P, 0.0, 0.4), 'noche': bool(es_noche(ex, [t + Wf / 2])[0])}
            d['lfhf'] = d['LF'] / d['HF'] if d['HF'] > 0 else np.nan
            filas_f.append(d); fm = f <= 0.4
            spec.append(P[fm]); ts.append(t); R['f_spec'] = f[fm]
        t += paso
    R['frec'] = filas_f; R['spec'] = np.array(spec) if spec else None; R['spec_t'] = np.array(ts)
    # ULF del registro completo (1 Hz)
    if ex.dur > 3 * 3600 and len(tt) > 1000:
        ti = np.arange(tt[0], tt[-1], 1.0); x = np.interp(ti, tt, nn); x = x - np.polyval(np.polyfit(ti, x, 1), ti)
        f, P = welch(x, fs=1.0, nperseg=len(x), window='hann', detrend=False)
        R['ULF'] = potencia(f, P, 0.0, 0.0033)
    else:
        R['ULF'] = np.nan
    return R


def resumen_frecuencia(R):
    """Filas (parámetro, media, mediana, despierto, dormir) de la pestaña Energía."""
    F = R.get('frec') or []
    if not F:
        return []
    out = []
    ulf = R.get('ULF', np.nan)
    for k in ('VLF', 'LF', 'HF', 'Total'):
        v = np.array([d[k] for d in F]); n = np.array([d['noche'] for d in F])
        out.append((k + ' (ms²)', v.mean(), np.median(v), v[~n].mean() if (~n).any() else np.nan, v[n].mean() if n.any() else np.nan))
    lr = np.array([np.log10(d['lfhf']) for d in F if d['lfhf'] == d['lfhf'] and d['lfhf'] > 0])
    nl = np.array([d['noche'] for d in F if d['lfhf'] == d['lfhf'] and d['lfhf'] > 0])
    if len(lr):
        out.append(('log LF/HF', lr.mean(), np.median(lr), lr[~nl].mean() if (~nl).any() else np.nan, lr[nl].mean() if nl.any() else np.nan))
    tot = np.array([d['VLF'] + d['LF'] + d['HF'] for d in F]).mean() + (ulf if ulf == ulf else 0)
    out.append(('ULF (ms², registro completo)', ulf, np.nan, np.nan, np.nan))
    for k in ('ULF', 'VLF', 'LF', 'HF'):
        v = ulf if k == 'ULF' else np.mean([d[k] for d in F])
        out.append((k + ' %', 100 * v / tot if tot and v == v else np.nan, np.nan, np.nan, np.nan))
    return out


def html_vfc(ex):
    R = calcular_vfc(ex)
    if R['n'] < 100:
        return '<p class="m">No hay intervalos NN suficientes para la VFC.</p>'
    c = R['cfg']
    h = ['<h3>Dominio del tiempo</h3><table>']
    for lab, v, d, u in (('Intervalos NN válidos', R['n'], 0, ''), ('Válidos', R['validos_pct'], 1, '%'), ('NN medio', R['media_nn'], 0, 'ms'),
                         ('SDNN', R['sdnn'], 0, 'ms'), ('SDANN-i', R['sdann'], 0, 'ms'), ('SDNN-i', R['sdnni'], 0, 'ms'),
                         ('rMSSD', R['rmssd'], 0, 'ms'), ('pNN50', R['pnn50'], 1, '%'), ('Índice triangular', R['tri'], 1, ''),
                         ('TINN', R['tinn'], 0, 'ms'), ('FC media día / noche', None, 0, ''), ('Índice circadiano (FC día / FC noche)', R['circadiano'], 2, '')):
        if lab.startswith('FC media'):
            h.append(f'<tr><td>{lab}</td><td class="n"><b>{fmt(R["fc_dia"])} / {fmt(R["fc_noche"])}</b></td><td>lpm</td></tr>'); continue
        h.append(f'<tr><td>{lab}</td><td class="n"><b>{fmt(v, d)}</b></td><td>{u}</td></tr>')
    h.append('</table>')
    rf = resumen_frecuencia(R)
    if rf:
        h.append(f'<h3>Dominio de la frecuencia ({"AR Burg" if c["metodo"] == "burg" else "FFT Welch"}, ventanas de {int(c["ventana_f"])} s)</h3>'
                 '<table><tr><th>Parámetro</th><th>Media</th><th>Mediana</th><th>Despierto</th><th>Dormir</th></tr>')
        for lab, *vs in rf:
            dd = 2 if 'log' in lab else (1 if '%' in lab else 0)
            h.append(f'<tr><td>{lab}</td>' + ''.join(f'<td class="n">{fmt(v, dd)}</td>' for v in vs) + '</tr>')
        h.append('</table>')
    return ''.join(h)


# --------------------------------------------------------------------------- QT
FORMULAS = [('bazett', 'Bazett'), ('fridericia', 'Fridericia'), ('framingham', 'Lineal (Framingham)'), ('hodges', 'Hodges'),
            ('pfeufer', 'Pfeufer (edad y sexo)'), ('sagie', 'Sagie (QTLC)')]
QT_DEF = {'formula': 'bazett', 'umbral': 500, 'min_dur': 5, 'fc_ignorar': 120, 'suavizar': 1, 'corte_max': None, 'corte_min': None}


def cfg_qt(ex):
    c = dict(QT_DEF)
    if getattr(ex, 'sesion', None):
        c.update(ex.sesion.d.get('qt_cfg') or {})
    return c


def edad_sexo(ex):
    pa = (ex.sesion.d.get('paciente') or {}) if getattr(ex, 'sesion', None) else {}
    try:
        edad = float(str(pa.get('edad', '')).replace(',', '.'))
    except ValueError:
        edad = np.nan
    if edad != edad and pa.get('nacimiento'):
        import re
        m = re.search(r'(\d{4})', pa['nacimiento'])
        if m:
            import time
            edad = time.localtime().tm_year - int(m.group(1))
    sx = str(pa.get('sexo', '')).strip().lower()
    if sx.startswith('f') or 'mujer' in sx:
        return edad, 'f'
    return edad, ('m' if sx else '')


def qtc_minutos(ex, formula=None):
    """QT (ms), RR (s), QTc (ms) y FC por minuto con la fórmula elegida."""
    c = cfg_qt(ex); formula = formula or c['formula']
    M = ex.A.get('minute') or {}
    qt = np.array([np.nan if v is None else v for v in M.get('qt', [])], float)
    hr = np.array([np.nan if v is None else v for v in M.get('hr', [])], float)
    n = min(len(qt), len(hr)); qt, hr = qt[:n], hr[:n]
    rr = 60.0 / hr
    if formula == 'bazett':
        q = qt / np.sqrt(rr)
    elif formula == 'fridericia':
        q = qt / np.cbrt(rr)
    elif formula in ('framingham', 'sagie'):
        q = qt + 154.0 * (1 - rr)
    elif formula == 'hodges':
        q = qt + 1.75 * (hr - 60)
    else:
        edad, sexo = edad_sexo(ex); edad = 60.0 if edad != edad else edad
        if sexo == 'f':
            q = qt - 0.154 * (rr * 1000 - 1000) - 0.207 * (edad - 60) - 4.58
        else:
            q = qt - 0.152 * (rr * 1000 - 1000) - 0.318 * (edad - 60)
    q[hr > c['fc_ignorar']] = np.nan
    if c['suavizar'] and int(c['suavizar']) > 1:
        k = int(c['suavizar']); ok = np.isfinite(q)
        s = np.convolve(np.where(ok, q, 0), np.ones(k), 'same'); w = np.convolve(ok.astype(float), np.ones(k), 'same')
        q = np.where(ok, s / np.maximum(w, 1), np.nan)
    if c.get('corte_max') is not None:
        q[q > c['corte_max']] = np.nan
    if c.get('corte_min') is not None:
        q[q < c['corte_min']] = np.nan
    return qt, rr, q, hr


def eventos_qt_largo(ex):
    """Episodios de QT largo (minutos seguidos con QTc sobre el umbral) para el directorio de bandas y las tendencias."""
    clave = (id(ex.A), str(ex.sesion.d.get('qt_cfg') if getattr(ex, 'sesion', None) else ''), str((ex.sesion.d.get('paciente') or {}) if getattr(ex, 'sesion', None) else ''))
    if getattr(ex, '_qtl', (None,))[0] == clave:
        return ex._qtl[1]
    c = cfg_qt(ex); out = []
    try:
        _, _, q, hr = qtc_minutos(ex)
        alto = np.isfinite(q) & (q > c['umbral'])
        d = np.diff(np.r_[0, alto.astype(np.int8), 0]); a = np.nonzero(d == 1)[0]; b = np.nonzero(d == -1)[0]
        for x, y in zip(a, b):
            if y - x >= int(c['min_dur']):
                t0, t1 = x * 60.0, y * 60.0
                out.append({'type': 'qtLong', 'label': 'QT largo', 't0': t0, 't1': t1, 'dur': t1 - t0, 'n': 0, 'b0': -1,
                            'rate': float(np.nanmean(hr[x:y])), 'level': float(np.nanmax(q[x:y])),
                            'detail': 'QTc máx. %s ms' % fmt(np.nanmax(q[x:y]))})
    except Exception:  # noqa: BLE001
        out = []
    ex._qtl = (clave, out)
    return out


def filas_qt(ex):
    qt, rr, q, hr = qtc_minutos(ex)
    n = len(q); tm = np.arange(n) * 60 + 30
    noche = es_noche(ex, tm) if n else np.zeros(0, bool)
    filas = []
    for h in range(int(np.ceil(ex.dur / 3600))):
        filas.append((ex.reloj(h * 3600, False), (tm >= h * 3600) & (tm < (h + 1) * 3600), (h * 3600, (h + 1) * 3600)))
    filas += [('Registro completo', np.ones(n, bool), (0, ex.dur)), ('Día', ~noche, None), ('Noche', noche, None)]
    out = []
    for lab, m, rg in filas:
        ok = m & np.isfinite(q)
        if rg:
            b0, b1 = np.searchsorted(ex.t, rg); lat = int((ex.cls[b0:b1] != 4).sum())
        else:
            nb = es_noche(ex, ex.t); lat = int(((nb if lab == 'Noche' else ~nb) & (ex.cls != 4)).sum())
        if ok.any():
            i_mx = np.nonzero(ok)[0][np.nanargmax(q[ok])]; i_mn = np.nonzero(ok)[0][np.nanargmin(q[ok])]
            out.append((lab, lat, int(ok.sum()), np.nanmean(qt[ok]), q[i_mn], q[i_mx], np.nanmean(q[ok]), np.nanmean(hr[ok]), i_mn, i_mx))
        else:
            out.append((lab, lat, 0, np.nan, np.nan, np.nan, np.nan, np.nan, -1, -1))
    return out


def html_qt(ex):
    c = cfg_qt(ex)
    h = [f'<h3>QT por hora (QTc de {dict(FORMULAS).get(c["formula"], c["formula"])}; minutos con FC &gt; {fmt(c["fc_ignorar"])} lpm ignorados)</h3>'
         '<table><tr><th>Hora</th><th>Latidos</th><th>Min. con QT</th><th>QT medio</th><th>QTc mín</th><th>QTc máx</th><th>QTc media</th><th>FC media</th></tr>']
    for lab, lat, nm, qtm, mn, mx, med, fc, *_ in filas_qt(ex):
        b = lab in ('Registro completo', 'Día', 'Noche')
        h.append(('<tr style="font-weight:600">' if b else '<tr>') + f'<td>{lab}</td><td class="n">{fmt(lat)}</td><td class="n">{fmt(nm)}</td>'
                 f'<td class="n">{fmt(qtm)}</td><td class="n">{fmt(mn)}</td><td class="n">{fmt(mx)}</td><td class="n">{fmt(med)}</td><td class="n">{fmt(fc)}</td></tr>')
    h.append('</table>')
    ql = eventos_qt_largo(ex)
    h.append(f'<p>Episodios de QT largo (QTc &gt; {fmt(c["umbral"])} ms durante {fmt(c["min_dur"])} min o más): <b>{len(ql)}</b></p>')
    return ''.join(h)


def html_st(ex):
    st = [e for e in ex.eventos if e['type'] in ('stDep', 'stEle')]
    if not st:
        return '<p class="m">Sin episodios del ST.</p>'
    h = ['<table><tr><th>Inicio</th><th>Duración</th><th>Nivel (mV)</th><th>Canal</th><th>Tipo</th><th>FC mín/media/máx</th></tr>']
    for e in st[:200]:
        ch = e.get('ch', 0); nc = ex.names[ex.chans[ch]] if ch < len(ex.chans) and ex.chans[ch] < len(ex.names) else str(ch + 1)
        a, m, b = fc_evento(ex, e)
        h.append(f'<tr><td>{ex.reloj(e["t0"])}</td><td>{fmt(e["dur"] / 60, 1)} min</td><td class="n">{fmt((e.get("level") or 0) / 1000, 2)}</td>'
                 f'<td>{nc}</td><td>{"descenso" if e["type"] == "stDep" else "elevación"}</td><td>{fmt(a)}/{fmt(m)}/{fmt(b)}</td></tr>')
    return ''.join(h) + '</table>'


# --------------------------------------------------------------------------- vistas
def _tabla(cab):
    t = QTableWidget(); t.setColumnCount(len(cab)); t.setHorizontalHeaderLabels(cab); t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setAlternatingRowColors(True); t.verticalHeader().setVisible(False); t.setSelectionBehavior(QAbstractItemView.SelectRows)
    return t


def _llenar(t, filas, negrita=()):
    t.setRowCount(len(filas))
    for r, fila in enumerate(filas):
        for c, v in enumerate(fila):
            it = QTableWidgetItem(v if isinstance(v, str) else fmt(v, 1 if isinstance(v, float) and abs(v) < 100 else 0))
            it.setTextAlignment((Qt.AlignRight if c else Qt.AlignLeft) | Qt.AlignVCenter)
            if r in negrita:
                f = it.font(); f.setBold(True); it.setFont(f)
            t.setItem(r, c, it)
    t.resizeColumnsToContents()


def mapa_fuego():
    return pg.ColorMap([0.0, 0.35, 0.6, 0.85, 1.0], [(70, 70, 70), (160, 160, 160), (250, 204, 21), (37, 99, 235), (255, 255, 255)])


class VistaVFC(Perezosa, QWidget):
    ir = Signal(float, int)
    nav = Signal(str, list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.R = None; self.sucio = False
        lay = QHBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        sp = QSplitter(Qt.Horizontal)
        # configuración (Configuración de VFC de Darwin)
        cfgw = QWidget(); cl = QVBoxLayout(cfgw); cl.setContentsMargins(0, 0, 0, 0)
        g1 = QGroupBox('Dominio del tiempo (prefiltrado de datos)'); f1 = QFormLayout(g1); self.w = {}
        self.w['filtro'] = QCheckBox('Filtro activado'); f1.addRow(self.w['filtro'])
        for k, n, lo, hi in (('min_ms', 'Intervalo mín. (ms)', 200, 1000), ('max_ms', 'Intervalo máx. (ms)', 800, 4000), ('fc_min', 'FC mín. (lpm)', 20, 80),
                             ('fc_max', 'FC máx. (lpm)', 100, 260), ('cambio_pct', 'Cambio máx. del intervalo (%)', 5, 60)):
            s = QSpinBox(); s.setRange(lo, hi); self.w[k] = s; f1.addRow(n, s)
        self.w['solo_normales'] = QCheckBox('Solo latidos normales (N a N)'); f1.addRow(self.w['solo_normales'])
        self.w['excluir_sve'] = QCheckBox('Excluir latidos supraventriculares'); f1.addRow(self.w['excluir_sve'])
        self.w['ventana_tabla'] = QComboBox(); [self.w['ventana_tabla'].addItem(n, s) for n, s in (('1 min', 60), ('2 min', 120), ('5 min', 300), ('10 min', 600), ('15 min', 900), ('30 min', 1800), ('1 h', 3600))]
        f1.addRow('Ancho de ventana de la tabla', self.w['ventana_tabla'])
        self.w['media'] = QComboBox(); self.w['media'].addItem('fn(Media(x)): sobre todos los intervalos', 'fn_media'); self.w['media'].addItem('Media(fn(x)): media de las ventanas', 'media_fn')
        f1.addRow('Valor del registro completo', self.w['media'])
        cl.addWidget(g1)
        g2 = QGroupBox('Dominio de la frecuencia'); f2 = QFormLayout(g2)
        self.w['metodo'] = QComboBox(); self.w['metodo'].addItem('FFT (Welch)', 'welch'); self.w['metodo'].addItem('AR (Burg)', 'burg'); f2.addRow('Método', self.w['metodo'])
        self.w['orden'] = QSpinBox(); self.w['orden'].setRange(4, 40); f2.addRow('Orden (AR)', self.w['orden'])
        self.w['tendencia'] = QComboBox(); [self.w['tendencia'].addItem(n, k) for k, n in (('lineal', 'Lineal'), ('media', 'Media'), ('ninguna', 'Ninguna'))]
        f2.addRow('Eliminar tendencia', self.w['tendencia'])
        self.w['ventana_f'] = QComboBox(); [self.w['ventana_f'].addItem(f'{s} s', s) for s in (120, 180, 300, 600)]; f2.addRow('Tamaño de ventana', self.w['ventana_f'])
        self.w['solape'] = QCheckBox('Solapamiento del 50 %'); f2.addRow(self.w['solape'])
        self.w['ancho_fft'] = QComboBox(); [self.w['ancho_fft'].addItem(str(s), s) for s in (64, 128, 256, 512, 1024)]; f2.addRow('Ancho de FFT (puntos)', self.w['ancho_fft'])
        cl.addWidget(g2)
        bf = QHBoxLayout(); b = QPushButton('Volver a calcular'); b.setObjectName('primario'); b.clicked.connect(self._aplicar); bf.addWidget(b)
        bd = QPushButton('Predeterminados'); bd.clicked.connect(self._defecto); bf.addWidget(bd); cl.addLayout(bf)
        be = QPushButton('Exportar a CSV…'); be.clicked.connect(self.exportar); cl.addWidget(be)
        self.glob = QLabel(''); self.glob.setTextFormat(Qt.RichText); self.glob.setWordWrap(True); cl.addWidget(self.glob); cl.addStretch(1)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(cfgw); sc.setMinimumWidth(330)
        sp.addWidget(sc)
        self.tabs = QTabWidget()
        self.t_tabla = _tabla(['Inicio', 'FC mín', 'FC media', 'FC máx', 'QRS', 'NN medio', 'SDNN', 'rMSSD', 'pNN50 %', 'Válido %'])
        self.t_tabla.cellClicked.connect(lambda r, c: self._ir_fila(r)); self.tabs.addTab(self.t_tabla, 'Resumen tabular')
        wf = QWidget(); fl = QVBoxLayout(wf); fl.setContentsMargins(0, 0, 0, 0)
        self.t_frec = _tabla(['Parámetro', 'Media', 'Mediana', 'Despierto', 'Dormir']); fl.addWidget(self.t_frec, 1)
        self.g_frec = pg.PlotWidget(); self.g_frec.showGrid(x=True, y=True, alpha=0.25); self.g_frec.setLabel('bottom', 'Hz'); self.g_frec.setLabel('left', 'ms²/Hz')
        fl.addWidget(self.g_frec, 1); self.tabs.addTab(wf, 'Dominio de la frecuencia')
        ws = QWidget(); sl = QVBoxLayout(ws); sl.setContentsMargins(0, 0, 0, 0)
        self.g_spec = pg.PlotWidget(); self.g_spec.setLabel('left', 'Hz'); self.g_spec.setLabel('bottom', 'horas desde el inicio')
        self.img = pg.ImageItem(); self.g_spec.addItem(self.img); self.g_spec.scene().sigMouseClicked.connect(self._clic_spec)
        self.cur_spec = pg.InfiniteLine(angle=90, pen=pg.mkPen('#dc2626', width=1.5)); self.g_spec.addItem(self.cur_spec)
        sl.addWidget(self.g_spec, 2)
        self.g_uno = pg.PlotWidget(); self.g_uno.showGrid(x=True, y=True, alpha=0.25); self.g_uno.setLabel('bottom', 'Hz'); sl.addWidget(self.g_uno, 1)
        self.tabs.addTab(ws, 'Espectrograma (Fire-of-Life)')
        wh = QWidget(); hl = QVBoxLayout(wh); hl.setContentsMargins(0, 0, 0, 0)
        oh = QHBoxLayout(); self.c_log = QCheckBox('Eje Y logarítmico'); self.c_tinn = QCheckBox('Mostrar TINN'); self.c_tinn.setChecked(True)
        self.res = QComboBox(); [self.res.addItem(f'{v} ms', v) for v in (7.8125, 10, 20, 50)]
        self.prom = QComboBox(); [self.prom.addItem(f'Valor medio sobre {k} intervalo(s)', k) for k in (1, 2, 3, 4)]
        for w in (self.c_log, self.c_tinn):
            w.toggled.connect(self._hist); oh.addWidget(w)
        for w in (self.res, self.prom):
            w.currentIndexChanged.connect(self._hist); oh.addWidget(w)
        oh.addStretch(1); hl.addLayout(oh)
        fh = QHBoxLayout(); self.g_rr = pg.PlotWidget(); self.g_dif = pg.PlotWidget()
        for g, tit in ((self.g_rr, 'Histograma de VFC (NN, ms)'), (self.g_dif, 'Histograma diferencial de VFC (NN n+1 − NN n, ms)')):
            g.setTitle(tit); g.showGrid(x=True, y=True, alpha=0.25); fh.addWidget(g, 1)
            g.scene().sigMouseClicked.connect(lambda ev, g=g: self._clic_hist(g, ev))
        hl.addLayout(fh, 1); self.tabs.addTab(wh, 'Histogramas')
        self.t_per = _tabla(['Periodo', 'Inicio', 'Duración', 'FC media', 'NN medio', 'SDNN', 'rMSSD', 'pNN50 %', 'Intervalos'])
        self.t_per.cellClicked.connect(lambda r, c: self._ir_per(r)); self.tabs.addTab(self.t_per, 'Resumen diario (diario y sueño)')
        sp.addWidget(self.tabs); sp.setSizes([340, 1000]); lay.addWidget(sp)

    # ---- configuración
    def _cargar_cfg(self, c):
        for k, w in self.w.items():
            if isinstance(w, QCheckBox):
                w.setChecked(bool(c[k]))
            elif isinstance(w, QSpinBox):
                w.setValue(int(c[k]))
            else:
                w.setCurrentIndex(max(0, w.findData(c[k])))

    def _leer_cfg(self):
        out = {}
        for k, w in self.w.items():
            out[k] = w.isChecked() if isinstance(w, QCheckBox) else w.value() if isinstance(w, QSpinBox) else w.currentData()
        return out

    def _aplicar(self):
        if self.ex and getattr(self.ex, 'sesion', None):
            self.ex.sesion.d['vfc_cfg'] = self._leer_cfg(); self._calcular_seguro()

    def _defecto(self):
        if self.ex and getattr(self.ex, 'sesion', None):
            self.ex.sesion.d['vfc_cfg'] = dict(VFC_DEF); self._calcular_seguro()

    # ---- cálculo y pantalla
    def calcular(self):
        ex = self.ex; self._cargar_cfg(cfg_vfc(ex))
        self.R = R = calcular_vfc(ex)
        if R['n'] < 100:
            self.glob.setText('No hay intervalos NN suficientes.'); return
        media_fn = R['cfg']['media'] == 'media_fn'
        T = [f for f in R['tabla'] if 'sdnn' in f]
        g = lambda k: np.nanmean([f[k] for f in T]) if (media_fn and T) else R[k]
        self.glob.setText(
            f'<b>Registro completo</b> ({fmt(R["n"])} NN, {fmt(R["validos_pct"], 1)} % válidos)<br>'
            f'SDNN {fmt(g("sdnn"))} ms · SDANN-i {fmt(R["sdann"])} ms · SDNN-i {fmt(R["sdnni"])} ms<br>'
            f'rMSSD {fmt(g("rmssd"))} ms · pNN50 {fmt(g("pnn50"), 1)} %<br>'
            f'Índice triangular {fmt(R["tri"], 1)} · TINN {fmt(R["tinn"])} ms<br>'
            f'FC media día / noche {fmt(R["fc_dia"])} / {fmt(R["fc_noche"])} lpm · índice circadiano {fmt(R["circadiano"], 2)}')
        filas = [(ex.reloj(f['t0'], False), f.get('fcmin', np.nan), f.get('fcmed', np.nan), f.get('fcmax', np.nan), f['qrs'], f.get('media', np.nan),
                  f.get('sdnn', np.nan), f.get('rmssd', np.nan), f.get('pnn50', np.nan), f['valido']) for f in R['tabla']]
        _llenar(self.t_tabla, filas)
        rf = resumen_frecuencia(R)
        _llenar(self.t_frec, [(lab, *vs) for lab, *vs in rf])
        self.g_frec.clear()
        if R.get('spec') is not None and len(R['spec']):
            f = R['f_spec']; P = np.mean(R['spec'], 0)
            for nm, a, b, col in BANDAS[1:]:
                m = (f >= a) & (f <= b)
                if m.sum() > 1:
                    c = QColor(col); c.setAlpha(110)
                    self.g_frec.addItem(pg.PlotDataItem(f[m], P[m], fillLevel=0, brush=c, pen=pg.mkPen(col)))
            self.g_frec.plot(f, P, pen=pg.mkPen('#1f2933', width=1.3))
            S = np.log10(np.maximum(R['spec'], 1e-3))
            lo, hi = np.percentile(S, 2), np.percentile(S, 99.5)
            self.img.setImage(S, levels=(lo, hi)); self.img.setLookupTable(mapa_fuego().getLookupTable(nPts=256))
            ts = R['spec_t'] / 3600.0; paso = (ts[1] - ts[0]) if len(ts) > 1 else 1 / 12
            from PySide6.QtCore import QRectF
            self.img.setRect(QRectF(ts[0], 0, paso * len(ts), f[-1]))
        self._hist(); self._periodos()

    def _hist(self, *_):
        R = self.R
        if not R or R['n'] < 100:
            return
        k = self.prom.currentData() or 1; nn = R['nn']
        if k > 1:
            nn = np.convolve(nn, np.ones(k) / k, mode='valid')
        bw = self.res.currentData() or 7.8125
        self.g_rr.clear(); self.g_dif.clear()
        e = np.arange(np.floor(nn.min() / bw) * bw, nn.max() + bw, bw); y, _ = np.histogram(nn, bins=e)
        self.h_rr = (e, nn)
        yy = np.log10(y + 1) if self.c_log.isChecked() else y
        self.g_rr.addItem(pg.BarGraphItem(x=(e[:-1] + e[1:]) / 2, height=yy, width=bw * 0.9, brush='#1f5fbf', pen=None))
        if self.c_tinn.isChecked() and R.get('tri_geo') and bw == 7.8125 and k == 1 and not self.c_log.isChecked():
            N, X, M, H = R['tri_geo']; self.g_rr.plot([N, X, M], [0, H, 0], pen=pg.mkPen('#dc2626', width=2))
        self.g_rr.setLabel('bottom', f'ms · índice triangular {fmt(R["tri"], 1)} · TINN {fmt(R["tinn"])} ms')
        d = np.diff(nn); lim = np.percentile(np.abs(d), 99.5) if len(d) else 100
        e2 = np.arange(-np.ceil(lim / bw) * bw, lim + bw, bw); y2, _ = np.histogram(np.clip(d, e2[0], e2[-1] - 1e-6), bins=e2)
        self.h_dif = (e2, d)
        yy2 = np.log10(y2 + 1) if self.c_log.isChecked() else y2
        self.g_dif.addItem(pg.BarGraphItem(x=(e2[:-1] + e2[1:]) / 2, height=yy2, width=bw * 0.9, brush='#2563eb', pen=None))
        self.g_dif.setLabel('bottom', 'ms (cuanto más angosta el área azul, menor la VFC)')

    def _clic_hist(self, g, ev):
        R = self.R
        if not R or not g.sceneBoundingRect().contains(ev.scenePos()):
            return
        x = g.getPlotItem().vb.mapSceneToView(ev.scenePos()).x()
        if g is self.g_rr and hasattr(self, 'h_rr'):
            e, _ = self.h_rr; j = int(np.searchsorted(e, x) - 1)
            if 0 <= j < len(e) - 1:
                m = (R['nn'] >= e[j]) & (R['nn'] < e[j + 1])
                self.nav.emit(f'NN {fmt(e[j])}–{fmt(e[j + 1])} ms', [(float(t), int(b)) for t, b in zip(R['tt'][m], R['idx'][m])][:20000])
        elif g is self.g_dif and hasattr(self, 'h_dif'):
            e, d = self.h_dif; j = int(np.searchsorted(e, x) - 1)
            if 0 <= j < len(e) - 1 and len(d) == len(R['nn']) - 1:
                m = np.r_[False, (d >= e[j]) & (d < e[j + 1])]
                self.nav.emit(f'Diferencia NN {fmt(e[j])}–{fmt(e[j + 1])} ms', [(float(t), int(b)) for t, b in zip(R['tt'][m], R['idx'][m])][:20000])

    def _clic_spec(self, ev):
        R = self.R
        if not R or R.get('spec') is None or not self.g_spec.sceneBoundingRect().contains(ev.scenePos()):
            return
        x = self.g_spec.getPlotItem().vb.mapSceneToView(ev.scenePos()).x() * 3600
        i = int(np.clip(np.searchsorted(R['spec_t'], x) - 1, 0, len(R['spec_t']) - 1))
        self.cur_spec.setValue(R['spec_t'][i] / 3600)
        self.g_uno.clear(); self.g_uno.plot(R['f_spec'], R['spec'][i], pen=pg.mkPen('#1f5fbf', width=1.5))
        F = R['frec'][i]
        self.g_uno.setTitle(f'{self.ex.reloj(R["spec_t"][i])} · LF {fmt(F["LF"])} · HF {fmt(F["HF"])} ms² · LF/HF {fmt(F["lfhf"], 2)}')
        self.ir.emit(float(R['spec_t'][i]), -1)

    def _ir_fila(self, r):
        if self.R and r < len(self.R.get('tabla', [])):
            self.ir.emit(float(self.R['tabla'][r]['t0']), -1)

    def periodos(self):
        """Periodos para el resumen diario: despierto y dormido (periodo de sueño) y cada anotación del diario hasta la siguiente."""
        ex = self.ex; out = []
        from .darwin_comun import hm, sueno
        s = sueno(ex); a = hm(s['dormir']); b = hm(s['despierto']); ini = ex.seg_inicio()
        t = 0.0
        while t < ex.dur:
            tod = (ini + t) % 86400
            noche = bool(es_noche(ex, [t])[0])
            fin_tod = b if noche else a
            dt = (fin_tod - tod) % 86400 or 86400
            t1 = min(ex.dur, t + dt)
            out.append(('Dormir' if noche else 'Despierto', t, t1)); t = t1
        dia = sorted((ex.sesion.d.get('diario') or []) if getattr(ex, 'sesion', None) else [], key=lambda d: d['t'])
        for i, d in enumerate(dia):
            t1 = dia[i + 1]['t'] if i + 1 < len(dia) else min(ex.dur, d['t'] + 3600)
            out.append(('Diario: ' + d.get('texto', ''), float(d['t']), float(t1)))
        return out

    def _periodos(self):
        R = self.R; filas = []; self._per = []
        for lab, a, b in self.periodos():
            m = (R['tt'] >= a) & (R['tt'] < b); x = R['nn'][m]
            rm, pn = _rmssd_pnn(x, R['idx'][m]) if len(x) > 2 else (np.nan, np.nan)
            filas.append((lab, self.ex.reloj(a), f'{fmt((b - a) / 60)} min', 60000 / x.mean() if len(x) else np.nan, x.mean() if len(x) else np.nan,
                          np.std(x) if len(x) > 2 else np.nan, rm, pn, len(x)))
            self._per.append(a)
        _llenar(self.t_per, filas)

    def _ir_per(self, r):
        if r < len(getattr(self, '_per', [])):
            self.ir.emit(float(self._per[r]), -1)

    def exportar(self):
        if not self.R or self.R['n'] < 100:
            return
        ruta, _ = QFileDialog.getSaveFileName(self, 'Exportar VFC', self.ex.nombre + '_vfc.csv', 'CSV (*.csv)')
        if not ruta:
            return
        R = self.R
        with open(ruta, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f, delimiter=';')
            w.writerow(['inicio', 'fc_min', 'fc_media', 'fc_max', 'qrs', 'nn_medio_ms', 'sdnn_ms', 'rmssd_ms', 'pnn50_pct', 'valido_pct'])
            for t in R['tabla']:
                w.writerow([self.ex.reloj(t['t0'])] + [('' if t.get(k) is None or t.get(k) != t.get(k) else f'{t[k]:.2f}'.replace('.', ','))
                                                     for k in ('fcmin', 'fcmed', 'fcmax', 'qrs', 'media', 'sdnn', 'rmssd', 'pnn50', 'valido')])
            w.writerow([]); w.writerow(['inicio_ventana', 'VLF_ms2', 'LF_ms2', 'HF_ms2', 'Total_ms2', 'LF_HF', 'noche'])
            for d in R.get('frec', []):
                w.writerow([self.ex.reloj(d['t0'])] + [f'{d[k]:.2f}'.replace('.', ',') for k in ('VLF', 'LF', 'HF', 'Total', 'lfhf')] + [int(d['noche'])])


class DialogoQTc(QDialog):
    """Elegir el primer minuto con el QT bien medido; los de QTc más extremo se descartan (Selec. interv. QTc correcto)."""

    def __init__(self, ex, extremo, parent=None):
        super().__init__(parent)
        self.ex = ex; self.extremo = extremo; self.setWindowTitle('Elegir el QTc ' + ('máximo' if extremo == 'max' else 'mínimo') + ' correcto'); self.resize(1000, 620)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel('Los minutos están ordenados del QTc ' + ('más largo' if extremo == 'max' else 'más corto') + ' al más normal. Elige el primero '
                             'en que el final de la T se ve bien medido: los de arriba se descartan y los valores se recalculan.'))
        sp = QSplitter(Qt.Horizontal); self.lista = QListWidget(); sp.addWidget(self.lista)
        self.tira = MiniTira(alto=320); sp.addWidget(self.tira); sp.setSizes([260, 740]); lay.addWidget(sp, 1)
        qt, rr, q, hr = qtc_minutos(ex)
        ok = np.nonzero(np.isfinite(q))[0]; orden = ok[np.argsort(q[ok])]
        if extremo == 'max':
            orden = orden[::-1]
        for i in orden[:400]:
            it = QListWidgetItem(f'{ex.reloj(i * 60, False)} · QTc {fmt(q[i])} ms · QT {fmt(qt[i])} · FC {fmt(hr[i])}'); it.setData(Qt.UserRole, (int(i), float(q[i])))
            self.lista.addItem(it)
        self.lista.currentItemChanged.connect(self._ver)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        if self.lista.count():
            self.lista.setCurrentRow(0)

    def _ver(self, it):
        if it:
            i, v = it.data(Qt.UserRole); self.tira.mostrar(self.ex, i * 60 + 28, 4.0, titulo=f'QTc {fmt(v)} ms')

    def valor(self):
        it = self.lista.currentItem()
        return it.data(Qt.UserRole)[1] if it else None


class VistaQTST(Perezosa, QWidget):
    ir = Signal(float, int)
    borrar_suceso = Signal(dict)
    cambio = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ex = None; self.sucio = False; self.filas = []; self.st = []
        lay = QHBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6)
        sp = QSplitter(Qt.Horizontal)
        izq = QWidget(); il = QVBoxLayout(izq); il.setContentsMargins(0, 0, 0, 0)
        tl = QLabel('Tabla resumen de QT'); tl.setObjectName('titulo'); il.addWidget(tl)
        op = QHBoxLayout(); op.addWidget(QLabel('Fórmula'))
        self.formula = QComboBox(); [self.formula.addItem(n, k) for k, n in FORMULAS]; op.addWidget(self.formula)
        op.addWidget(QLabel('Ignorar FC sobre')); self.fcig = QSpinBox(); self.fcig.setRange(60, 220); op.addWidget(self.fcig)
        op.addWidget(QLabel('Suavizar (min)')); self.suav = QSpinBox(); self.suav.setRange(1, 30); op.addWidget(self.suav)
        il.addLayout(op)
        op2 = QHBoxLayout(); op2.addWidget(QLabel('QT largo: QTc sobre (ms)')); self.umbral = QSpinBox(); self.umbral.setRange(400, 650); op2.addWidget(self.umbral)
        op2.addWidget(QLabel('durante (min)')); self.mindur = QSpinBox(); self.mindur.setRange(1, 120); op2.addWidget(self.mindur)
        b = QPushButton('Aplicar'); b.setObjectName('primario'); b.clicked.connect(self._aplicar); op2.addWidget(b)
        bq = QPushButton('Quitar descartes'); bq.clicked.connect(self._sin_cortes); op2.addWidget(bq); op2.addStretch(1)
        il.addLayout(op2)
        self.t_qt = _tabla(['Hora', 'Latidos', 'Min. con QT', 'QT medio', 'QTc mín', 'QTc máx', 'QTc media', 'FC media'])
        self.t_qt.cellClicked.connect(self._clic_qt); self.t_qt.cellDoubleClicked.connect(self._doble_qt); il.addWidget(self.t_qt, 1)
        self.l_qt = QLabel(''); self.l_qt.setWordWrap(True); self.l_qt.setObjectName('ayuda'); il.addWidget(self.l_qt)
        sp.addWidget(izq)
        der = QWidget(); dl = QVBoxLayout(der); dl.setContentsMargins(0, 0, 0, 0)
        tl2 = QLabel('Tabla ST'); tl2.setObjectName('titulo'); dl.addWidget(tl2)
        self.t_st = _tabla(['Inicio', 'Duración', 'Nivel (mV)', 'Canal', 'Episodio', 'FC mín', 'FC media', 'FC máx'])
        self.t_st.cellClicked.connect(lambda r, c: self._ir_st(r)); dl.addWidget(self.t_st, 1)
        QShortcut(QKeySequence.Delete, self.t_st, self._borrar_st)
        self.l_st = QLabel('Descensos en azul y elevaciones en naranja. Clic: va al ECG · Supr: borra el episodio. Los criterios del ST están en la pestaña Criterios.')
        self.l_st.setWordWrap(True); self.l_st.setObjectName('ayuda'); dl.addWidget(self.l_st)
        sp.addWidget(der); sp.setSizes([700, 560]); lay.addWidget(sp)

    def _cfg(self):
        return cfg_qt(self.ex)

    def _guardar(self, c):
        self.ex.sesion.d['qt_cfg'] = c; self.cambio.emit(); self._calcular_seguro()

    def _aplicar(self):
        c = self._cfg(); c.update(formula=self.formula.currentData(), fc_ignorar=self.fcig.value(), suavizar=self.suav.value(),
                                  umbral=self.umbral.value(), min_dur=self.mindur.value())
        self._guardar(c)

    def _sin_cortes(self):
        c = self._cfg(); c['corte_max'] = None; c['corte_min'] = None; self._guardar(c)

    def calcular(self):
        ex = self.ex; c = self._cfg()
        self.formula.setCurrentIndex(max(0, self.formula.findData(c['formula']))); self.fcig.setValue(int(c['fc_ignorar'])); self.suav.setValue(int(c['suavizar']))
        self.umbral.setValue(int(c['umbral'])); self.mindur.setValue(int(c['min_dur']))
        self.filas = filas_qt(ex); n = len(self.filas)
        _llenar(self.t_qt, [f[:8] for f in self.filas], negrita=(n - 3, n - 2, n - 1))
        edad, sexo = edad_sexo(ex)
        ql = eventos_qt_largo(ex)
        txt = f'Episodios de QT largo: {len(ql)} (aparecen en el directorio de bandas y en las tendencias de arritmia). Doble clic en QTc máx o mín: elegir el valor bien medido.'
        if c['formula'] == 'pfeufer':
            txt += f' Pfeufer usa edad {fmt(edad) if edad == edad else "60 (no está en los datos del paciente)"} y sexo {"femenino" if sexo == "f" else "masculino" if sexo else "no indicado (se usa la fórmula de hombres)"}.'
        if c.get('corte_max') is not None or c.get('corte_min') is not None:
            txt += f' Descartados: QTc sobre {fmt(c.get("corte_max"))} y bajo {fmt(c.get("corte_min"))} ms.'
        self.l_qt.setText(txt)
        self.st = [e for e in ex.eventos if e['type'] in ('stDep', 'stEle')]
        filas = []
        for e in self.st:
            ch = e.get('ch', 0); nc = ex.names[ex.chans[ch]] if ch < len(ex.chans) and ex.chans[ch] < len(ex.names) else str(ch + 1)
            a, m, b = fc_evento(ex, e)
            filas.append((ex.reloj(e['t0']), f'{fmt(e["dur"] / 60, 1)} min', fmt((e.get('level') or 0) / 1000, 2), nc,
                          'Descenso' if e['type'] == 'stDep' else 'Elevación', a, m, b))
        _llenar(self.t_st, filas)
        for r, e in enumerate(self.st):
            col = QColor('#1d4ed8' if e['type'] == 'stDep' else '#ea580c')
            for c_ in range(self.t_st.columnCount()):
                it = self.t_st.item(r, c_)
                if it:
                    it.setForeground(col)

    def _clic_qt(self, r, c):
        f = self.filas[r] if r < len(self.filas) else None
        if not f:
            return
        i = f[9] if c == 5 else f[8] if c == 4 else -1
        if i >= 0:
            self.ir.emit(i * 60.0 + 30, -1)
        elif ':' in f[0]:
            self.ir.emit(r * 3600.0, -1)

    def _doble_qt(self, r, c):
        if c not in (4, 5):
            return
        d = DialogoQTc(self.ex, 'max' if c == 5 else 'min', self)
        if d.exec() and d.valor() is not None:
            cfg = self._cfg(); cfg['corte_max' if c == 5 else 'corte_min'] = d.valor(); self._guardar(cfg)

    def _ir_st(self, r):
        if r < len(self.st):
            self.ir.emit(float(self.st[r]['t0']), -1)

    def _borrar_st(self):
        r = self.t_st.currentRow()
        if 0 <= r < len(self.st):
            self.borrar_suceso.emit(self.st[r])
