"""Informe final: portada con datos del paciente, estadísticas, resumen narrativo, perfiles, tendencias, plantillas,
tiras de ECG con índice y contexto, información completa y firma. Se dibuja sobre una impresora o un PDF."""
import numpy as np
from PySide6.QtCore import QMarginsF, QPointF, QRectF, QSettings, QSizeF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPageLayout, QPageSize, QPainter, QPen, QPolygonF, QTextDocument

from . import __version__
from .estilo import COLOR_CLASE, RESUMEN_CSS
from .motor import CLS_NAME, CLS_SYM, GRUPO_CLASE
from .perfil import COL, calcular
from .resumen import grupos_finales, hms, narrativo
from .tira import dibujar_ecg, linea, senal
from .vistas import series_tendencia

CSS = RESUMEN_CSS + 'body { font-size: 8.5pt; } h3 { font-size: 9.5pt; margin: 6px 0 2px 0; } td, th { padding: 1px 5px; } h2 { font-size: 11pt; }'

PERFILES = [('Ritmo general', ['min', 'lat', 'diario', 'fcmin', 'fcmed', 'fcmax', 'rrmax', 'pausas', 'stdep', 'stele', 'u1', 'u2', 'u3']),
            ('Ritmo supraventricular', ['diario', 'fcmin', 'fcmed', 'fcmax', 's1', 's2', 's3', 'stot', 'tsv', 'sbig', 'strig', 'aber', 'brb', 'fa']),
            ('Ritmo ventricular', ['diario', 'fcmin', 'fcmed', 'fcmax', 'v1', 'v2', 'v3', 'vtot', 'tv', 'vbig', 'vtrig', 'ront', 'fus', 'interp', 'esc', 'desc']),
            ('RR y QT', ['diario', 'fcmin', 'fcmed', 'fcmax', 'pnn50', 'rmssd', 'sdnn', 'qtc'])]


def _f(v, d=0):
    if v is None or (isinstance(v, float) and v != v):
        return ''
    return f'{v:,.{d}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')


class Informe:
    def __init__(self, ex, op):
        self.ex = ex; self.op = op; self.ses = ex.sesion.d if getattr(ex, 'sesion', None) else {}
        self.cfg = QSettings('HolterLocal', 'HolterLocal')

    # ---- página
    def _empezar(self, printer):
        self.pr = printer; self.p = QPainter(printer); self.p.setRenderHint(QPainter.Antialiasing, True)
        self.res = printer.resolution(); self.mm = self.res / 25.4
        self.area = printer.pageLayout().paintRectPixels(self.res)
        self.W = self.area.width(); self.H = self.area.height(); self.pag = 1; self.primera = True

    def nueva(self):
        if not self.primera:
            self.pr.newPage(); self.pag += 1
        self.primera = False
        self._marco()
        return 16 * self.mm   # y donde empieza el contenido

    def _fuente(self, pt, bold=False):
        f = QFont('Segoe UI'); f.setPointSizeF(pt); f.setBold(bold)
        f.setPixelSize(int(pt * self.res / 72)); self.p.setFont(f); return f

    def _marco(self):
        p, mm = self.p, self.mm; ex = self.ex; pa = self.ses.get('paciente') or {}
        p.setPen(QColor('#1f2933')); self._fuente(8, True)
        nombre = ' '.join(x for x in (pa.get('nombre', ''), pa.get('apellido', '')) if x) or ex.nombre
        p.drawText(QRectF(0, 0, self.W * 0.6, 5 * mm), Qt.AlignLeft | Qt.AlignVCenter, nombre)
        self._fuente(7)
        datos = ' · '.join(x for x in (('ID ' + pa['id']) if pa.get('id') else '', pa.get('sexo', ''), (pa.get('edad', '') + ' años') if pa.get('edad') else '',
                                       ('nac. ' + pa['nacimiento']) if pa.get('nacimiento') else '', 'inicio ' + ex.reloj(0)) if x)
        p.drawText(QRectF(0, 5 * mm, self.W * 0.7, 4 * mm), Qt.AlignLeft | Qt.AlignVCenter, datos)
        p.drawText(QRectF(self.W * 0.5, 0, self.W * 0.5, 5 * mm), Qt.AlignRight | Qt.AlignVCenter, self.cfg.value('institucion', '') or 'Informe de Holter de ritmo')
        p.setPen(QPen(QColor('#9ca3af'), 0.3 * mm)); p.drawLine(QPointF(0, 10 * mm), QPointF(self.W, 10 * mm))
        self._fuente(6.5); p.setPen(QColor('#6b7280'))
        p.drawText(QRectF(0, self.H - 4 * mm, self.W, 4 * mm), Qt.AlignLeft | Qt.AlignVCenter,
                   f'HolterLocal {__version__} · herramienta de apoyo: toda conclusión la revisa un médico')
        p.drawText(QRectF(0, self.H - 4 * mm, self.W, 4 * mm), Qt.AlignRight | Qt.AlignVCenter, f'Página {self.pag}')

    def _html(self, html, y):
        """Dibuja HTML desde y; pasa de página si no cabe. Devuelve la y final."""
        doc = QTextDocument(); doc.setDefaultStyleSheet(CSS); doc.setDocumentMargin(0)
        f = QFont('Segoe UI'); f.setPixelSize(int(8.5 * self.res / 72)); doc.setDefaultFont(f)
        doc.setHtml(html); doc.setTextWidth(self.W)
        alto = doc.size().height(); hecho = 0.0; lim = self.H - 6 * self.mm
        while hecho < alto - 1:
            disp = lim - y
            if disp < 20 * self.mm:
                y = self.nueva(); disp = lim - y
            self.p.save(); self.p.translate(0, y - hecho)
            doc.drawContents(self.p, QRectF(0, hecho, self.W, disp)); self.p.restore()
            hecho += disp
            if hecho < alto - 1:
                y = self.nueva()
            else:
                y = y + disp - (hecho - alto)
        return y + 2 * self.mm

    # ---- secciones
    def portada(self):
        p, mm, ex, d = self.p, self.mm, self.ex, self.ses
        y = self.nueva(); pa = d.get('paciente') or {}
        logo = self.cfg.value('logo', '')
        if logo:
            img = QImage(logo)
            if not img.isNull():
                h = 16 * mm; w = img.width() * h / max(1, img.height()); p.drawImage(QRectF(self.W - w, y, w, h), img)
        self._fuente(14, True); p.setPen(QColor('#1f5fbf')); p.drawText(QPointF(0, y + 6 * mm), 'Informe de Holter de ritmo')
        self._fuente(8); p.setPen(QColor('#1f2933'))
        cont = (self.cfg.value('contacto', '') or '').replace('\n', ' · ')
        if cont:
            p.drawText(QPointF(0, y + 11 * mm), cont)
        y += 18 * mm
        filas = [('Paciente', ' '.join(x for x in (pa.get('nombre', ''), pa.get('apellido', '')) if x)), ('ID', pa.get('id', '')), ('ID secundaria', pa.get('id2', '')),
                 ('Admisión', pa.get('admision', '')), ('Nacimiento', pa.get('nacimiento', '')), ('Edad', pa.get('edad', '')), ('Sexo', pa.get('sexo', '')),
                 ('Etnia', pa.get('raza', '')), ('Dirección', pa.get('direccion', '')), ('Teléfono', pa.get('telefono', '')), ('Correo', pa.get('email', '')),
                 ('Indicaciones', pa.get('indicaciones', '')), ('Medicamentos', pa.get('medicamentos', '')), ('Médico que deriva', pa.get('derivador', '')),
                 ('Procedimiento', pa.get('procedimiento', '')), ('Ubicación', pa.get('ubicacion', '')), ('Fecha del registro', pa.get('fecha_registro', '')),
                 ('Inicio', ex.reloj(0)), ('Duración', hms(ex.dur)), ('Técnico', pa.get('tecnico', '')), ('Analista', pa.get('analista', '')),
                 ('Grabadora', ' '.join(x for x in (pa.get('grabadora_tipo', ''), pa.get('grabadora_num', '')) if x)),
                 ('Canales', ', '.join(ex.names[c] for c in ex.chans if c < len(ex.names)))]
        filas = [f for f in filas if f[1]]
        mitad = (len(filas) + 1) // 2
        t = '<table width="100%"><tr><td valign="top" width="50%"><table>' + ''.join(f'<tr><td class="m">{a}</td><td><b>{b}</b></td></tr>' for a, b in filas[:mitad])
        t += '</table></td><td valign="top"><table>' + ''.join(f'<tr><td class="m">{a}</td><td><b>{b}</b></td></tr>' for a, b in filas[mitad:]) + '</table></td></tr></table>'
        y = self._html(t, y)
        if self.op.get('condensado'):
            claves = {'latidos', 'analizado', 'fc_media', 'fc_min', 'fc_max', 'v_tot', 'v_par', 'v_rachas', 'v_tv', 's_tot', 's_par', 's_rachas', 's_tsv',
                      'fa_pct', 'fa_ep', 'pausas', 'rr_max', 'pa', 'pv', 'pd'}
            h = '<h3>Estadísticas resumidas</h3><table>'
            for g, tit, fs, on in grupos_finales(ex):
                for k, lab, v, tt, nota in fs:
                    if on and k in claves:
                        h += f'<tr><td>{lab}</td><td class="n"><b>{v}</b></td><td class="m">{ex.reloj(tt) if tt is not None else ""}</td></tr>'
            y = self._html(h + '</table>', y)
        for k, tit in (('diagnostico', 'Diagnóstico'), ('notas', 'Notas'), ('conclusiones', 'Conclusiones')):
            if d.get(k):
                y = self._html(f'<h3>{tit}</h3><p>' + d[k].replace('&', '&amp;').replace('<', '&lt;').replace('\n', '<br>') + '</p>', y)
        y = max(y, self.H - 34 * mm)
        estado = d.get('estado', 'Adquirido')
        firma = d.get('firmado_por', '')
        h = (f'<h3>Estado del examen: {estado}</h3><table width="100%"><tr><td>Revisado por: <b>{d.get("revisado_por", "")}</b></td>'
             f'<td>{"Firmado electrónicamente por" if estado == "Firmado" and firma else "Médico que firma"}: <b>{firma}</b> {d.get("fecha_firma", "")}</td></tr></table>')
        self._html(h, y)

    def criterios(self):
        S = self.ex.A.get('settings') or {}
        return ('<p class="m">Criterios: prematuridad SV %s %% · pausa %s s%s · ST desc./elev. %s/%s µV · taquicardia > %s lpm · bradicardia < %s lpm '
                '(mín. %s s) · TV ≥ %s latidos a ≥ %s lpm · TSV ≥ %s latidos a ≥ %s lpm · QTc %s · FA automática: %s</p>') % (
            _f(100 * S.get('svPrematurity', 0.25)), _f(S.get('pauseSec', 2), 1), ' (solo N–N)' if S.get('pauseNNOnly') else '',
            _f(S.get('stDepUv') or S.get('stThreshUv')), _f(S.get('stEleUv') or S.get('stThreshUv')), _f(S.get('tachyBpm')), _f(S.get('bradyBpm')),
            _f(S.get('minRateDurSec')), _f(S.get('vtBeats')), _f(S.get('vtBpm')), _f(S.get('svtBeats')), _f(S.get('svtBpm')), S.get('qtcFormula', 'bazett'),
            'sí' if S.get('afDetect', True) else 'no')

    def estadisticas(self):
        y = self.nueva(); ex = self.ex
        h = '<h2>Estadísticas resumidas</h2>' + self.criterios()
        for g, tit, fs, on in grupos_finales(ex):
            if not on:
                continue
            h += f'<h3>{tit}</h3><table>' + ''.join(
                f'<tr><td>{lab}</td><td class="n"><b>{v}</b></td><td class="m">{(ex.reloj(t) + " ") if t is not None else ""}{nota or ""}</td></tr>' for k, lab, v, t, nota in fs) + '</table>'
        st = [e for e in ex.eventos if e['type'] in ('stDep', 'stEle')]
        if st:
            h += '<h3>Episodios del ST</h3><table><tr><th>Inicio</th><th>Duración</th><th>µV máx.</th><th>Canal</th><th>Tipo</th></tr>'
            for e in st[:60]:
                ch = e.get('ch', 0); nc = ex.names[ex.chans[ch]] if ch < len(ex.chans) and ex.chans[ch] < len(ex.names) else str(ch + 1)
                h += f'<tr><td>{ex.reloj(e["t0"])}</td><td>{hms(e["dur"])}</td><td class="n">{_f(e.get("level"))}</td><td>{nc}</td><td>{"descenso" if e["type"] == "stDep" else "elevación"}</td></tr>'
            h += '</table>'
        self._html(h, y)

    def holterai(self):
        extra = self.ex.A.get('extra_html')
        if extra:
            y = self.nueva(); self._html('<h2>Hallazgos de conducción y morfología</h2>' + extra, y)

    def narrativo(self):
        y = self.nueva()
        txt = narrativo(self.ex, self.cfg.value('narrativa', '') or None)
        self._html('<h2>Resumen narrativo</h2><p>' + txt.replace('&', '&amp;').replace('<', '&lt;').replace('\n', '<br>') + '</p>', y)

    def perfiles(self):
        ex = self.ex; P, filas, tot = calcular(ex)
        ov = self.ses.get('perfil_override', {}); off = set(self.ses.get('perfil_cols_off', []))
        for tit, cols in PERFILES:
            y = self.nueva()
            h = f'<h2>Perfil: {tit}</h2><table><tr><th>Hora</th>' + ''.join(f'<th>{COL[c][1]}</th>' for c in cols) + '</tr>'
            for i, r in enumerate(filas + [tot]):
                lab = ex.reloj(r['t0'], False) if i < len(filas) else '<b>Total</b>'
                fila = []
                for c in cols:
                    key = '%s@%d' % (c, i if i < len(filas) else -1)
                    v = '—' if c in off else ov.get(key, _f(r.get(c, np.nan), COL[c][4]))
                    fila.append(f'<td class="n">{v}</td>')
                h += f'<tr><td>{lab}</td>' + ''.join(fila) + '</tr>'
            self._html(h + '</table>', y)

    def _grafico(self, rect, x, ys, titulo, tipo='line', colores=('#1f5fbf',), banda=None, x0=0, x1=None):
        p, mm = self.p, self.mm
        p.setPen(QPen(QColor('#d1d5db'), 0.2 * mm)); p.drawRect(rect)
        self._fuente(7, True); p.setPen(QColor('#1f2933')); p.drawText(QPointF(rect.left() + 1 * mm, rect.top() - 1 * mm), titulo)
        allv = np.concatenate([np.asarray(y, float)[np.isfinite(y)] for y in ys if len(y)] + ([np.asarray(banda[1])[np.isfinite(banda[1])]] if banda else []) + [np.zeros(1)])
        lo = 0 if tipo == 'bar' else float(np.nanmin(allv)) if len(allv) else 0
        hi = float(np.nanmax(allv)) if len(allv) else 1
        if banda:
            lo = float(np.nanmin(np.asarray(banda[0])[np.isfinite(banda[0])])) if np.isfinite(banda[0]).any() else lo
        if hi <= lo:
            hi = lo + 1
        x1 = x1 or (x[-1] if len(x) else 1)
        X = lambda v: rect.left() + (v - x0) / max(1e-6, x1 - x0) * rect.width()
        Y = lambda v: rect.bottom() - (v - lo) / (hi - lo) * rect.height()
        self._fuente(6); p.setPen(QColor('#6b7280'))
        p.drawText(QRectF(rect.left() - 12 * mm, rect.top() - 2 * mm, 11 * mm, 4 * mm), Qt.AlignRight, _f(hi))
        p.drawText(QRectF(rect.left() - 12 * mm, rect.bottom() - 2 * mm, 11 * mm, 4 * mm), Qt.AlignRight, _f(lo))
        if banda:
            p.setPen(QPen(QColor('#9fb7e3'), 0.25 * mm))
            for xi, a, b in zip(x, banda[0], banda[1]):
                if np.isfinite(a) and np.isfinite(b) and x0 <= xi <= x1:
                    p.drawLine(QPointF(X(xi), Y(a)), QPointF(X(xi), Y(b)))
        for y, col in zip(ys, colores):
            y = np.asarray(y, float)
            if tipo == 'bar':
                p.setPen(QPen(QColor(col), 0.3 * mm))
                for xi, v in zip(x, y):
                    if np.isfinite(v) and v > 0 and x0 <= xi <= x1:
                        p.drawLine(QPointF(X(xi), Y(0)), QPointF(X(xi), Y(v)))
            else:
                p.setPen(QPen(QColor(col), 0.3 * mm))
                pts = [QPointF(X(xi), Y(v)) for xi, v in zip(x, y) if np.isfinite(v) and x0 <= xi <= x1]
                if tipo == 'dots':
                    for q in pts:
                        p.drawPoint(q)
                elif len(pts) > 1:
                    p.drawPolyline(QPolygonF(pts))
        # horas abajo, cada 2 h
        self._fuente(5.5); p.setPen(QColor('#6b7280'))
        h = 0.0
        while h * 60 <= x1:
            if h * 60 >= x0:
                p.drawText(QPointF(X(h * 60) - 2 * mm, rect.bottom() + 3 * mm), self.ex.reloj(h * 3600, False))
            h += 2

    def tendencias(self):
        ex = self.ex; S = series_tendencia(ex); mm = self.mm
        n = len(S['hr'])
        if not n:
            return
        x = np.arange(n) * 5 + 2.5
        sel = self.op.get('tendencias', ['fc', 'ectopia', 'st', 'qt', 'vfc'])
        graf = []
        if 'fc' in sel:
            graf.append(('Frecuencia cardíaca (lpm): mín., media y máx. cada 5 min', 'line', [S['hr']], ('#1f5fbf',), (S['hrmin'], S['hrmax'])))
        if 'ectopia' in sel:
            graf += [('Latidos ventriculares cada 5 min', 'bar', [S['v']], ('#dc2626',), None), ('Latidos supraventriculares cada 5 min', 'bar', [S['s']], ('#1d4ed8',), None),
                     ('Fibrilación auricular (%)', 'bar', [S['af']], ('#0d9488',), None)]
        if 'qt' in sel and np.isfinite(S['qtc']).any():
            graf.append(('QT y QTc (ms)', 'dots', [S['qt'], S['qtc']], ('#374151', '#7c3aed'), None))
        if 'vfc' in sel:
            graf.append(('RMSSD y SDNN (ms)', 'line', [S['rmssd'], S['sdnn']], ('#0f766e', '#7c3aed'), None))
        if 'st' in sel and S['st']:
            graf.append(('ST por canal (µV)', 'line', S['st'], ('#111111', '#0f766e', '#7c3aed'), None))
        dia = 24 * 60
        for d0 in range(0, int(np.ceil(ex.dur / 60 / dia)) * dia, dia):
            y = self.nueva() + 4 * mm
            alto = (self.H - y - 12 * mm) / max(1, len(graf)) - 8 * mm
            for tit, tipo, ys, cols, banda in graf:
                r = QRectF(14 * mm, y, self.W - 16 * mm, alto)
                self._grafico(r, x, ys, tit + (f' · día {d0 // dia + 1}' if ex.dur > 86400 else ''), tipo, cols, banda, d0, d0 + dia)
                y += alto + 8 * mm

    def plantillas(self):
        ex = self.ex; T = ex.A.get('templates') or []; tcls = ex.A.get('tcls') or []; mm = self.mm
        nch = ex.A['nch']; M = ex.A.get('morphM', {}).get('M', 56)
        grupos = {}
        for i, Tt in enumerate(T):
            n = int(sum(tcls[i])) if i < len(tcls) else Tt['count']
            if n == 0:
                continue
            k = int(np.argmax(tcls[i])) if i < len(tcls) and sum(tcls[i]) else 0
            g = GRUPO_CLASE.get(k, 'U')
            if g == 'X':
                continue
            grupos.setdefault(g, []).append((n, i, k))
        nombres = {'N': 'Normal', 'S': 'Supraventricular', 'V': 'Ventricular', 'P': 'Estimulado', 'U': 'Desconocido'}
        for g in ('N', 'S', 'V', 'P', 'U'):
            if g not in grupos:
                continue
            y = self.nueva(); self._fuente(10, True); self.p.setPen(QColor('#1f2933')); self.p.drawText(QPointF(0, y + 4 * mm), 'Plantillas: ' + nombres[g])
            y += 8 * mm; cw, chh = 45 * mm, 38 * mm; cols = int(self.W // cw)
            for j, (n, i, k) in enumerate(sorted(grupos[g], reverse=True)[:cols * int((self.H - y - 10 * mm) // chh)]):
                r = QRectF((j % cols) * cw, y + (j // cols) * chh, cw - 2 * mm, chh - 2 * mm)
                self.p.setPen(QPen(QColor('#e5e7eb'), 0.2 * mm)); self.p.drawRect(r)
                w = np.asarray(T[i]['wave'], float); h = (r.height() - 6 * mm) / max(1, nch)
                self.p.setPen(QPen(QColor(COLOR_CLASE[k]), 0.35 * mm))
                for c in range(nch):
                    yy = w[c * M:(c + 1) * M]
                    if len(yy):
                        rng = max(1e-6, np.nanmax(np.abs(yy)))
                        xs = np.linspace(r.left() + 2 * mm, r.right() - 2 * mm, len(yy)); ys = r.top() + 5 * mm + h * (c + 0.55) - yy / rng * h * 0.42
                        self.p.drawPolyline(QPolygonF([QPointF(a, b) for a, b in zip(xs, ys)]))
                self._fuente(6.5); self.p.setPen(QColor('#1f2933'))
                self.p.drawText(QRectF(r.left() + 1 * mm, r.top() + 0.5 * mm, r.width(), 4 * mm), Qt.AlignLeft, f'#{i + 1} · {CLS_NAME[k]} · {_f(n)} latidos')

    def tiras(self):
        ex = self.ex; L = [t for t in self.ses.get('tiras', [])][:400]; mm = self.mm
        if not L:
            return
        por_pag = 3
        # índice con página de cada tira
        pag0 = self.pag + 1 + int(np.ceil(len(L) / 40))
        paginas = []; k = pag0; cnt = 0
        for t in L:
            if t.get('pagina'):
                if cnt:
                    k += 1; cnt = 0
                paginas.append(k); k += 1
            else:
                paginas.append(k); cnt += 1
                if cnt == por_pag:
                    k += 1; cnt = 0
        for a in range(0, len(L), 40):
            y = self.nueva()
            h = '<h2>Índice de tiras de ECG</h2><table><tr><th>Hora</th><th>Duración</th><th>Canales</th><th>Anotación</th><th>Página</th></tr>'
            for t, pg in zip(L[a:a + 40], paginas[a:a + 40]):
                can = ex.names[ex.chans[t.get('canal', 0)]] if t.get('pagina') else ', '.join(ex.names[ex.chans[c]] for c in t['chans'] if c < len(ex.chans))
                h += f'<tr><td>{ex.reloj(t["t0"])}</td><td>{("%d min" % t["pagina"]) if t.get("pagina") else hms(t["dur"])}</td><td>{can}</td><td>{t.get("ann", "")}</td><td class="n">{pg}</td></tr>'
            self._html(h + '</table>', y)
        gan = float(self.op.get('ganancia', 10)); rej = self.op.get('rejilla', True)
        j = 0; y = None
        for t in L:
            if t.get('pagina'):
                self._tira_pagina(t); j = 0; y = None; continue
            if j % por_pag == 0:
                y = self.nueva()
            dur = min(t['dur'], 7.5 * int(self.W / (7.5 * 25 * mm)) or 7.5)
            w = min(self.W, dur * 25 * mm)
            chs = [ex.chans[c] for c in t['chans'] if c < len(ex.chans)] or [ex.chans[0]]
            alto = max(22 * mm, min(14 * mm * len(chs) * gan / 10, 60 * mm))
            self._fuente(7.5, True); self.p.setPen(QColor('#1f2933'))
            hr = self._fc(t['t0'], t['t0'] + dur)
            self.p.drawText(QPointF(0, y + 3 * mm), f'{ex.reloj(t["t0"])} · {t.get("ann", "")} · {hr} · 25 mm/s · {_f(gan)} mm/mV')
            dibujar_ecg(self.p, QRectF(0, y + 4.5 * mm, w, alto), ex, t['t0'], dur, chs, mm_mv=gan, rejilla=rej, grosor=self.res / 150)
            # contexto de 22,5 s alrededor
            c0 = max(0.0, t['t0'] + dur / 2 - 11.25)
            dibujar_ecg(self.p, QRectF(0, y + 5 * mm + alto, self.W, 12 * mm), ex, c0, 22.5, [chs[0]], mm_mv=gan / 2, etiquetas=False,
                        rejilla=False, titulo_canal=False, colorear=False, grosor=self.res / 200)
            a = (t['t0'] - c0) / 22.5 * self.W
            self.p.setPen(QPen(QColor('#dc2626'), 0.3 * mm)); self.p.setBrush(Qt.NoBrush)
            self.p.drawRect(QRectF(a, y + 5 * mm + alto, dur / 22.5 * self.W, 12 * mm))
            y += alto + 22 * mm; j += 1

    def _fc(self, a, b):
        ex = self.ex; b0, b1 = np.searchsorted(ex.t, [a, b]); rr = ex.rr[b0:b1]; rr = rr[np.isfinite(rr)]
        return ('FC %d lpm' % round(60 / np.mean(rr))) if len(rr) else ''

    def _miniatura(self, t0, minutos, canal, titulo, linea_s=60.0):
        """Página de ECG en miniatura: una línea por minuto (o por 30 s) con la FC a la izquierda."""
        ex = self.ex; mm = self.mm; p = self.p
        y = self.nueva(); self._fuente(9, True); p.setPen(QColor('#1f2933')); p.drawText(QPointF(0, y + 3 * mm), titulo)
        self._fuente(6.5); p.drawText(QPointF(self.W - 60 * mm, y + 3 * mm), f'{_f(self.W / mm / linea_s, 1)} mm/s · 2,5 mm/mV')
        y += 6 * mm
        n = int(np.ceil(minutos * 60 / linea_s)); lh = (self.H - y - 8 * mm) / max(1, n); x0 = 13 * mm; w = self.W - x0
        c = ex.chans[min(canal, len(ex.chans) - 1)]
        pxs = w / linea_s; pxmv = 2.5 * mm
        p.setPen(QPen(QColor('#111111'), max(1.0, self.res / 400)))
        for i in range(n):
            ta = t0 + i * linea_s
            if ta >= ex.dur:
                break
            yy = senal(ex, c, int(ta * ex.fs), int((ta + linea_s) * ex.fs))
            mid = y + lh * (i + 0.6)
            p.setPen(QPen(QColor('#111111'), max(1.0, self.res / 400)))
            p.drawPolyline(linea(yy, ex.fs, ta, ta, x0, pxs, mid, pxmv, y + lh * i, y + lh * (i + 1)))
            self._fuente(5.5); p.setPen(QColor('#6b7280'))
            p.drawText(QPointF(0, mid), ex.reloj(ta, False) if (int(ta) // 60) % 5 == 0 and linea_s >= 60 else '')
            p.drawText(QPointF(0, mid + 2.2 * mm), self._fc(ta, ta + linea_s))
            b0, b1 = np.searchsorted(ex.t, [ta, ta + linea_s])
            for b in range(int(b0), int(b1)):
                k = int(ex.cls[b])
                if k:
                    p.fillRect(QRectF(x0 + (ex.t[b] - ta) * pxs - 0.3 * mm, y + lh * i + 0.3 * mm, 0.6 * mm, 1.2 * mm), QColor(COLOR_CLASE[k]))

    def _tira_pagina(self, t):
        ex = self.ex; m = int(t.get('pagina') or 5)
        nombre = ex.names[ex.chans[t.get('canal', 0)]] if t.get('canal', 0) < len(ex.chans) else ''
        self._miniatura(t['t0'], m, t.get('canal', 0), f'Tira de página: {ex.reloj(t["t0"])} · {m} min · {nombre} · {t.get("ann", "")}', 60.0 if m >= 20 else 30.0)

    def completa(self):
        ex = self.ex; horas = self.op.get('completa_horas') or []; chs = self.op.get('completa_canales') or [0]
        k = 0
        for h in horas:
            for c in chs:
                if k >= 50:
                    return
                nombre = ex.names[ex.chans[c]] if c < len(ex.chans) and ex.chans[c] < len(ex.names) else str(c + 1)
                self._miniatura(h * 3600.0, 60, c, f'Información completa · {ex.reloj(h * 3600)} · {nombre}'); k += 1

    # ---- todo
    def pintar(self, printer):
        self._empezar(printer)
        s = self.op.get('secciones', {})
        try:
            for k in ('portada', 'estadisticas', 'narrativo', 'holterai', 'perfiles', 'tendencias', 'plantillas', 'tiras', 'completa'):
                if s.get(k, k in ('portada', 'estadisticas')):
                    getattr(self, k)()
            if self.primera:
                self.nueva()
        finally:
            self.p.end()


def impresora(pdf=None):
    from PySide6.QtPrintSupport import QPrinter
    pr = QPrinter(QPrinter.HighResolution); pr.setResolution(300)
    pr.setPageLayout(QPageLayout(QPageSize(QPageSize.A4), QPageLayout.Portrait, QMarginsF(10, 10, 10, 10), QPageLayout.Millimeter))
    if pdf:
        pr.setOutputFormat(QPrinter.PdfFormat); pr.setOutputFileName(pdf)
    return pr


def imprimir_pantalla(ex, tira, printer):
    """Página con lo que muestra la tira: hora, paciente, ID y FC arriba."""
    p = QPainter(printer); res = printer.resolution(); mm = res / 25.4
    area = printer.pageLayout().paintRectPixels(res); W = area.width()
    pa = (ex.sesion.d.get('paciente') or {}) if getattr(ex, 'sesion', None) else {}
    f = QFont('Segoe UI'); f.setPixelSize(int(9 * res / 72)); f.setBold(True); p.setFont(f)
    t0 = tira.t0; dur = tira.ancho_seg()
    b0, b1 = np.searchsorted(ex.t, [t0, t0 + dur]); rr = ex.rr[b0:b1]; rr = rr[np.isfinite(rr)]
    fc = ('FC %d lpm' % round(60 / np.mean(rr))) if len(rr) else ''
    nombre = ' '.join(x for x in (pa.get('nombre', ''), pa.get('apellido', '')) if x) or ex.nombre
    p.drawText(QPointF(0, 5 * mm), f'{ex.reloj(t0)} – {ex.reloj(t0 + dur)} · {nombre} · ID {pa.get("id", "—")} · {fc}')
    chs = [ex.chans[c] for c in tira.canales()]
    L, nl = tira.lineas()
    alto = (area.height() - 12 * mm) / max(1, nl)
    for i in range(nl):
        dibujar_ecg(p, QRectF(0, 8 * mm + i * alto, W, alto - 2 * mm), ex, t0 + i * L, L, chs, mm_mv=tira.gan if nl == 1 else tira.gan / 2,
                    etiquetas=tira.etiquetas, rejilla=tira.rejilla, filtro=tira.filtro, grosor=res / 150)
    p.end()
