"""Sección del resumen con lo que mide holterai además del recuento de latidos."""


def _f(v, d=0, suf=''):
    if v is None or v == '':
        return '—'
    try:
        s = f'{float(v):,.{d}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')
    except (TypeError, ValueError):
        return str(v)
    return s + suf


def _filas(rows):
    return '<table>' + ''.join(f'<tr><td>{a}</td><td class="n"><b>{b}</b></td><td class="m">{c or ""}</td></tr>' for a, b, c in rows) + '</table>'


def html_holterai(ha, ex):
    S = ha.get('summary') or {}
    h = []
    pw = S.get('pwave') or {}; av = S.get('av_conduction') or {}
    bloques = av.get('blocks') or {}
    nombres = {'av_block_1': 'primer grado', 'av_block_mobitz1': 'Mobitz I', 'av_block_mobitz2': 'Mobitz II', 'av_block_2to1': '2:1',
               'av_block_advanced': 'avanzado', 'av_block_complete': 'completo'}
    txt_bloq = ', '.join(f'{nombres.get(k, k)}: {v}' for k, v in bloques.items()) or 'ninguno'
    h.append('<h3>Onda P y conducción AV (holterai)</h3>' + _filas([
        ('Minutos con onda P presente', _f(pw.get('p_present_pct'), 1, ' %'), f'evaluables {_f(pw.get("evaluable_pct"), 1, " %")}'),
        ('PR mediano', _f(pw.get('pr_median_ms') or av.get('pr_median_ms'), 0, ' ms'),
         f'mín {_f(pw.get("pr_min_ms") or av.get("pr_min_ms"), 0)} · máx {_f(pw.get("pr_max_ms") or av.get("pr_max_ms"), 0)} ms'),
        ('Latidos con PR > 200 ms', _f(av.get('first_degree_pct'), 1, ' %'), 'en los tramos medidos'),
        ('Bloqueos AV', txt_bloq, 'a revisión' if av.get('review') else ''),
    ]))
    qm = S.get('qrs_morph') or {}; ramas = S.get('ramas') or {}
    h.append('<h3>QRS y bloqueos de rama (holterai)</h3>' + _filas([
        ('Ancho mediano del QRS', _f(qm.get('median_width_ms'), 0, ' ms'), f'p95 {_f(qm.get("p95_width_ms"), 0)} ms'),
        ('Tiempo con QRS ≥ 120 ms', _f(qm.get('wide_pct_time'), 1, ' %'), ''),
        ('Morfología', ramas.get('pattern_name') or qm.get('pattern_name') or '—', (ramas.get('kind') or '').replace('_', ' ')),
    ]))
    rb = S.get('ritmo_base') or {}
    tramos = rb.get('tramos') or []
    tipos = {}
    for t in tramos:
        tipos[t.get('tipo', '?')] = tipos.get(t.get('tipo', '?'), 0) + (t.get('end_s', 0) - t.get('start_s', 0))
    carga = S.get('carga') or {}; lown = S.get('lown') or {}; pac = S.get('pacing') or {}
    filas = [('Pausas / asistolias', f'{_f(S.get("n_pauses"))} / {_f(S.get("n_asystole"))}', ''),
             ('Ectopia ventricular estimada en 24 h', _f(carga.get('ev_24h')), carga.get('categoria_ev', '')),
             ('Ectopia supraventricular estimada en 24 h', _f(carga.get('esv_24h')), carga.get('categoria_esv', '')),
             ('Clase de Lown', lown.get('label', '—'), lown.get('detail', '')),
             ('Marcapasos', _f(pac.get('pct_paced'), 1, ' % estimulado'), pac.get('note', ''))]
    for k, v in sorted(tipos.items(), key=lambda x: -x[1])[:4]:
        filas.append((f'Ritmo de base: {k.replace("_", " ")}', f'{v / 60:.0f} min', ''))
    h.append('<h3>Ritmo y carga (holterai)</h3>' + _filas(filas))
    return ''.join(h)
