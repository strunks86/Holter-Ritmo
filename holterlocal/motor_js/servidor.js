/* HolterLocal: servidor del motor. Lo lanza la aplicación de escritorio y conversa por stdin/stdout
   con una orden JSON por línea. Los arreglos grandes viajan como archivos binarios en una carpeta temporal. */
'use strict';
const fs = require('fs'); const path = require('path'); const readline = require('readline');
const E = require('./engine.js'); global.HolterEngine = E;
eval(fs.readFileSync(path.join(__dirname, 'model_beats.js'), 'utf8'));
eval(fs.readFileSync(path.join(__dirname, 'model_af.js'), 'utf8'));

let REC = null, A = null, SET = {}, OUT = null;
const send = o => process.stdout.write(JSON.stringify(o) + '\n');
const ab = b => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
const dump = (name, arr) => { const p = path.join(OUT, name); fs.writeFileSync(p, Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength)); return { file: p, dtype: arr.constructor.name, n: arr.length }; };

function readRecord(paths, o) {
  o = o || {};
  const p0 = paths[0]; const ext = path.extname(p0).toLowerCase(); const base = path.basename(p0, path.extname(p0));
  if (ext === '.hea' || ext === '.dat') {
    const dir = path.dirname(p0); const hea = fs.readFileSync(path.join(dir, base + '.hea'), 'latin1');
    const h = E.parseWfdbHeader(hea); const datName = (h.sigs[0] && h.sigs[0].file) || (base + '.dat');
    const rec = E.readWfdb(hea, ab(fs.readFileSync(path.join(dir, datName)))); rec.name = base;
    let ref = null; for (const ax of ['atr', 'qrs', 'ecg']) { const ap = path.join(dir, base + '.' + ax); if (fs.existsSync(ap)) { try { ref = E.readMitAnn(ab(fs.readFileSync(ap))); ref.fs = rec.fs; ref.file = path.basename(ap); } catch (e) { ref = null; } break; } }
    return { rec, ref };
  }
  const buf = ab(fs.readFileSync(p0)); const u8 = new Uint8Array(buf); const magic = Buffer.from(u8.slice(0, 8)).toString('latin1');
  if (magic === 'ISHNE1.0') { const rec = E.readIshne(buf); rec.name = base; return { rec, ref: null }; }
  if (ext === '.edf' || ext === '.bdf' || /^0 {7}/.test(magic) || (u8[0] === 255 && magic.slice(1) === 'BIOSEMI')) { const rec = E.readEdf(buf); rec.name = base; return { rec, ref: null }; }
  if (ext === '.txt' || ext === '.csv') { const rec = E.readText(fs.readFileSync(p0, 'latin1'), o.fs || 0, o.uv || 1); rec.name = base; return { rec, ref: null }; }
  if (paths.length > 1 || /^\d+$/.test(base)) { const files = paths.map(p => ({ name: path.basename(p), buf: ab(fs.readFileSync(p)) })); const rec = E.readMortaraHours(files, o); rec.name = path.basename(path.dirname(p0)); return { rec, ref: null }; }
  throw new Error('Formato no reconocido: ' + path.basename(p0));
}

function summaryOut() {
  const R = E.summarize(A, SET); const B = A.beats;
  const files = {
    t: dump('t.bin', B.t), cls: dump('cls.bin', B.cls), auto: dump('auto.bin', B.auto), edited: dump('edited.bin', B.edited), tid: dump('tid.bin', B.tid),
    pV: dump('pV.bin', B.pV), pX: dump('pX.bin', B.pX), conf: dump('conf.bin', B.conf), rr: dump('rr.bin', R.rr), prem: dump('prem.bin', R.prem),
    hr8: dump('hr8.bin', R.hr8), afBeat: dump('afBeat.bin', R.afBeat), best: dump('best.bin', A.best), excl: dump('excl.bin', R.excl)
  };
  A.qual.forEach((q, i) => { files['qual' + i] = dump('qual' + i + '.bin', q); });
  const M = R.minute; const minute = {}; for (const k in M) { if (ArrayBuffer.isView(M[k])) minute[k] = Array.from(M[k], v => (v === v ? v : null)); }
  minute.qt = A.qt ? Array.from(A.qt.qt, v => (v === v ? v : null)) : [];
  minute.st = M.st ? M.st.map(a => Array.from(a, v => (v === v && v !== -32768 ? v : null))) : [];
  const templates = A.templates.map(T => { const n = T.wave.length; return { id: T.id, count: T.count, width: T.width, pV: T.pV, pX: T.pX, wave: Array.from(T.wave) }; });
  const tcls = A.templates.map(() => new Array(E.NCLS).fill(0)); for (let b = 0; b < B.n; b++) { const k = B.tid[b]; if (k >= 0 && k < tcls.length) tcls[k][B.cls[b]]++; }
  const strip = o => { const r = {}; for (const k in o) { const v = o[k]; if (ArrayBuffer.isView(v)) continue; r[k] = v; } return r; };
  const hrv = Object.assign({}, R.hrv); delete hrv.specRows; delete hrv.seg; delete hrv.spec;
  return {
    files, minute, exclusions: A.exclusions || [], afUser: A.afUser || [], nDeleted: A.nDeleted || 0, templates, tcls, morphM: E.MORPH, nch: A.nch, chans: A.chans, chanNames: A.chanNames, fs: A.fs, nsec: A.nsec, dur: A.dur, start: A.start, name: A.name, source: A.source,
    events: R.events, counts: Array.from(R.counts), total: R.total, vTot: R.vTot, sTot: R.sTot, cnt: R.cnt, rOnT: R.rOnT, interp: R.interp, vBurden: R.vBurden, sBurden: R.sBurden,
    analyzedSec: R.analyzedSec, exSec: R.exSec, exPct: R.exPct, hr: R.hr, rrMax: R.rrMax, rrMaxB: R.rrMaxB, rrMin: R.rrMin, pauses: R.pauses, af: R.af, qt: R.qt, stRef: R.stRef, stEpisodes: R.stEpisodes,
    hours: R.hours.map(strip), hrv, settings: R.settings, qtc: Array.from(R.qtc, v => (v === v ? v : null)),
    clsSym: E.CLS_SYM, clsName: E.CLS_NAME
  };
}


/* ---- utilidades de edición: los arreglos por latido se reconstruyen al insertar o borrar */
function lowerIdx(t) { return E.lowerBound(A.beats.t, t); }
function nearest(t, tol) { const T = A.beats.t; const i = E.lowerBound(T, t); let best = -1, bd = tol; for (const j of [i - 1, i]) if (j >= 0 && j < T.length && Math.abs(T[j] - t) <= bd) { best = j; bd = Math.abs(T[j] - t); } return best; }
function firma() { return A.beats.n + ':' + A.templates.length + ':' + (A.extMotor || 'estacion'); }
function perBeatKeys(B) { const ks = []; for (const k in B) { const v = B[k]; if (ArrayBuffer.isView(v) && v.length === B.n && k !== 'X') ks.push(k); } return ks; }
function insertBeat(t, c) {
  const B = A.beats; const i = E.lowerBound(B.t, t); const n = B.n;
  const ins = (arr, v) => { const o = new arr.constructor(n + 1); o.set(arr.subarray(0, i), 0); o[i] = v; o.set(arr.subarray(i), i + 1); return o; };
  const defs = { t, samp: Math.round(t * A.fs), cls: c, auto: c, conf: 1, pV: 0, pS: 0, pX: 0, tid: -1, tcorr: 0, edited: 1 };
  for (const k of perBeatKeys(B)) B[k] = ins(B[k], k in defs ? defs[k] : 0);
  B.st = B.st.map(a => ins(a, -32768)); B.n = n + 1;
}
function deleteBeats(list) {
  const B = A.beats; const n = B.n; const kill = new Uint8Array(n); let nk = 0; for (const b of list) if (b >= 0 && b < n && !kill[b]) { kill[b] = 1; nk++; }
  if (!nk) return; const m = n - nk;
  const del = arr => { const o = new arr.constructor(m); let j = 0; for (let b = 0; b < n; b++) if (!kill[b]) o[j++] = arr[b]; return o; };
  for (const k of perBeatKeys(B)) B[k] = del(B[k]);
  B.st = B.st.map(del); B.n = m; A.nDeleted = (A.nDeleted || 0) + nk;
}
function mergeTemplates(tids) {
  if (!tids || tids.length < 2) return; const dst = tids[0]; const S = new Set(tids.slice(1)); const B = A.beats; let moved = 0;
  for (let b = 0; b < B.n; b++) if (S.has(B.tid[b])) { B.tid[b] = dst; moved++; }
  for (const k of S) if (A.templates[k]) A.templates[k].count = 0; if (A.templates[dst]) A.templates[dst].count += moved;
}

const H = {
  open(m) {
    OUT = m.out; fs.mkdirSync(OUT, { recursive: true });
    const { rec, ref } = readRecord(m.paths, m.opt);
    if (m.maxSec && m.maxSec > 60 && m.maxSec * rec.fs < rec.n) { rec.n = Math.floor(m.maxSec * rec.fs); rec.sig = rec.sig.map(x => x.subarray(0, rec.n)); }
    REC = rec; SET = m.settings || {};
    A = E.analyze(rec, m.analysis || {}, (msg, p) => send({ type: 'progress', msg, p }));
    send({ type: 'progress', msg: 'Calculando el resumen', p: 0.93 });
    const sig = rec.sig.map((s, i) => dump('sig' + i + '.bin', s));
    let refOut = null; if (ref) { const keep = []; const sym = []; for (let i = 0; i < ref.sample.length; i++) if (E.isBeatSym(ref.sym[i])) { keep.push(ref.sample[i] / rec.fs); sym.push(ref.sym[i]); } refOut = { file: ref.file, t: keep, sym: sym.join('') }; }
    return { rec: { name: rec.name, source: rec.source, fs: rec.fs, n: rec.n, nsig: rec.nsig, gain: rec.gain, baseline: rec.baseline, names: rec.names, start: rec.start, sig }, ref: refOut, A: summaryOut() };
  },
  /* Latidos y FA de otro motor (holterai): reemplazan a los propios; morfología, calidad, ST y QT
     se heredan del latido propio más cercano (±80 ms). */
  applyExt(m) {
    const tb = fs.readFileSync(m.t), cb = fs.readFileSync(m.cls);
    const tExt = new Float64Array(tb.buffer.slice(tb.byteOffset, tb.byteOffset + tb.byteLength));
    const cExt = new Uint8Array(cb.buffer.slice(cb.byteOffset, cb.byteOffset + cb.byteLength));
    const O = A.beats, nb = tExt.length, nch = O.st.length;
    const nB = { n: nb, t: new Float64Array(nb), samp: new Int32Array(nb), cls: new Uint8Array(nb), auto: new Uint8Array(nb), conf: new Float32Array(nb),
      pV: new Float32Array(nb), pS: new Float32Array(nb), pX: new Float32Array(nb), tid: new Int32Array(nb).fill(-1), tcorr: new Float32Array(nb),
      X: O.X, st: O.st.map(() => new Int16Array(nb).fill(-32768)), edited: new Uint8Array(nb) };
    let j = 0, matched = 0;
    for (let b = 0; b < nb; b++) {
      const t = tExt[b]; nB.t[b] = t; nB.samp[b] = Math.round(t * A.fs); nB.cls[b] = cExt[b]; nB.auto[b] = cExt[b]; nB.conf[b] = 1;
      while (j + 1 < O.n && Math.abs(O.t[j + 1] - t) <= Math.abs(O.t[j] - t)) j++;
      if (O.n && Math.abs(O.t[j] - t) <= 0.08) {
        matched++; nB.pV[b] = O.pV[j]; nB.pS[b] = O.pS[j]; nB.pX[b] = O.pX[j]; nB.tid[b] = O.tid[j]; nB.tcorr[b] = O.tcorr[j];
        for (let c = 0; c < nch; c++) nB.st[c][b] = O.st[c][j];
      }
    }
    A.beats = nB; A.extMotor = m.motor || 'externo'; A.extClasses = true;
    if (m.af) A.afManual = [[0, A.dur + 1, 0]].concat(m.af.map(e => [e[0], e[1], 1]));
    const R = summaryOut(); R.extMatched = matched; return { A: R };
  },
  /* Modo mixto: latidos, V y FA de la estación; holterai decide solo entre N y S en los latidos que la
     estación llama normales o supraventriculares (±80 ms). */
  applyMix(m) {
    const tb = fs.readFileSync(m.t), cb = fs.readFileSync(m.cls);
    const tExt = new Float64Array(tb.buffer.slice(tb.byteOffset, tb.byteOffset + tb.byteLength));
    const cExt = new Uint8Array(cb.buffer.slice(cb.byteOffset, cb.byteOffset + cb.byteLength));
    const O = A.beats; let j = 0, matched = 0, toS = 0, toN = 0;
    for (let b = 0; b < O.n; b++) {
      if (O.cls[b] > 1) continue;
      const t = O.t[b];
      while (j + 1 < tExt.length && Math.abs(tExt[j + 1] - t) <= Math.abs(tExt[j] - t)) j++;
      if (!tExt.length || Math.abs(tExt[j] - t) > 0.08) continue;
      matched++; const c = cExt[j] === 1 ? 1 : 0;
      if (c !== O.auto[b]) { if (c) toS++; else toN++; }
      O.auto[b] = c; if (!O.edited[b]) O.cls[b] = c;
    }
    A.extMotor = 'mixto'; A.extClasses = true;
    const R = summaryOut(); R.extMatched = matched; R.mix = { toS, toN }; return { A: R };
  },
  relabel(m) { const B = A.beats; for (const b of m.beats) { if (b >= 0 && b < B.n) { B.cls[b] = m.cls; B.edited[b] = 1; } } return { A: summaryOut() }; },
  relabelTemplate(m) { const B = A.beats; for (let b = 0; b < B.n; b++) if (B.tid[b] === m.tid) { B.cls[b] = m.cls; B.edited[b] = 1; } return { A: summaryOut() }; },

  /* ---- edición de latidos, plantillas, artefactos y FA */
  insertBeat(m) { insertBeat(m.t, m.cls); return { A: summaryOut(), beat: lowerIdx(m.t) }; },
  deleteBeats(m) { deleteBeats(m.beats); return { A: summaryOut() }; },
  relabelTemplates(m) { const S = new Set(m.tids); const B = A.beats; for (let b = 0; b < B.n; b++) if (S.has(B.tid[b])) { B.cls[b] = m.cls; B.edited[b] = 1; } return { A: summaryOut() }; },
  mergeTemplates(m) { mergeTemplates(m.tids); return { A: summaryOut() }; },
  deleteTemplateBeats(m) { const S = new Set(m.tids); const B = A.beats; const del = []; for (let b = 0; b < B.n; b++) if (S.has(B.tid[b])) del.push(b); deleteBeats(del); return { A: summaryOut() }; },
  setExclusions(m) { A.exclusions = m.list || []; return { A: summaryOut() }; },
  setAf(m) { A.afUser = m.list || []; return { A: summaryOut() }; },
  /* Reaplica lo guardado de un examen: borrados, etiquetas manuales (o latidos insertados), combinaciones, artefactos y FA. */
  replay(m) {
    let nDel = 0, nRel = 0, nIns = 0;
    if (m.del && m.del.length) { const B = A.beats; const idx = []; for (const t of m.del) { const b = nearest(t, 0.05); if (b >= 0) idx.push(b); } nDel = idx.length; deleteBeats(idx); }
    if (m.edits) for (const [t, c] of m.edits) { const b = nearest(t, 0.05); if (b >= 0) { A.beats.cls[b] = c; A.beats.edited[b] = 1; nRel++; } else { insertBeat(t, c); nIns++; } }
    if (m.merges && m.firma === firma()) for (const g of m.merges) mergeTemplates(g);
    if (m.exclusions) A.exclusions = m.exclusions;
    if (m.afUser) A.afUser = m.afUser;
    const R = summaryOut(); R.replay = { nDel, nRel, nIns }; return { A: R };
  },
  firma() { return { firma: firma() }; },
  resetEdits() { const B = A.beats; B.edited.fill(0); A.exclusions = []; A.afUser = []; return { A: summaryOut() }; },
  exclude(m) { A.exclusions = (A.exclusions || []).concat([[m.t0, m.t1]]); return { A: summaryOut() }; },
  settings(m) { SET = Object.assign({}, SET, m.settings || {}); return { A: summaryOut() }; },
  defaults() { return { defaults: E.DEFAULTS, version: E.VERSION }; },
  ping() { return { ok: true, version: E.VERSION }; }
};

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', line => {
  let m; try { m = JSON.parse(line); } catch (e) { return; }
  try { const r = H[m.cmd] ? H[m.cmd](m) : { error: 'orden desconocida ' + m.cmd }; send(Object.assign({ type: 'result', id: m.id }, r)); }
  catch (e) { send({ type: 'result', id: m.id, error: (e && e.message) || String(e) }); }
});
send({ type: 'ready', version: E.VERSION });
