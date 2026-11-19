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
  minute.st = M.st ? M.st.map(a => Array.from(a, v => (v === v && v !== -32768 ? v : null))) : [];
  const templates = A.templates.map(T => { const n = T.wave.length; return { id: T.id, count: T.count, width: T.width, pV: T.pV, pX: T.pX, wave: Array.from(T.wave) }; });
  const tcls = A.templates.map(() => new Array(10).fill(0)); for (let b = 0; b < B.n; b++) { const k = B.tid[b]; if (k >= 0 && k < tcls.length) tcls[k][B.cls[b]]++; }
  const strip = o => { const r = {}; for (const k in o) { const v = o[k]; if (ArrayBuffer.isView(v)) continue; r[k] = v; } return r; };
  const hrv = Object.assign({}, R.hrv); delete hrv.specRows; delete hrv.seg; delete hrv.spec;
  return {
    files, minute, templates, tcls, morphM: E.MORPH, nch: A.nch, chans: A.chans, chanNames: A.chanNames, fs: A.fs, nsec: A.nsec, dur: A.dur, start: A.start, name: A.name, source: A.source,
    events: R.events, counts: Array.from(R.counts), total: R.total, vTot: R.vTot, sTot: R.sTot, cnt: R.cnt, rOnT: R.rOnT, interp: R.interp, vBurden: R.vBurden, sBurden: R.sBurden,
    analyzedSec: R.analyzedSec, exSec: R.exSec, exPct: R.exPct, hr: R.hr, rrMax: R.rrMax, rrMaxB: R.rrMaxB, rrMin: R.rrMin, pauses: R.pauses, af: R.af, qt: R.qt, stRef: R.stRef, stEpisodes: R.stEpisodes,
    hours: R.hours.map(strip), hrv, settings: R.settings, qtc: Array.from(R.qtc, v => (v === v ? v : null)),
    clsSym: E.CLS_SYM, clsName: E.CLS_NAME
  };
}

const H = {
  open(m) {
    OUT = m.out; fs.mkdirSync(OUT, { recursive: true });
    const { rec, ref } = readRecord(m.paths, m.opt);
    REC = rec; SET = m.settings || {};
    A = E.analyze(rec, m.analysis || {}, (msg, p) => send({ type: 'progress', msg, p }));
    send({ type: 'progress', msg: 'Calculando el resumen', p: 0.93 });
    const sig = rec.sig.map((s, i) => dump('sig' + i + '.bin', s));
    let refOut = null; if (ref) { const keep = []; const sym = []; for (let i = 0; i < ref.sample.length; i++) if (E.isBeatSym(ref.sym[i])) { keep.push(ref.sample[i] / rec.fs); sym.push(ref.sym[i]); } refOut = { file: ref.file, t: keep, sym: sym.join('') }; }
    return { rec: { name: rec.name, source: rec.source, fs: rec.fs, n: rec.n, nsig: rec.nsig, gain: rec.gain, baseline: rec.baseline, names: rec.names, start: rec.start, sig }, ref: refOut, A: summaryOut() };
  },
  relabel(m) { const B = A.beats; for (const b of m.beats) { if (b >= 0 && b < B.n) { B.cls[b] = m.cls; B.edited[b] = 1; } } return { A: summaryOut() }; },
  relabelTemplate(m) { const B = A.beats; for (let b = 0; b < B.n; b++) if (B.tid[b] === m.tid) { B.cls[b] = m.cls; B.edited[b] = 1; } return { A: summaryOut() }; },
  resetEdits() { const B = A.beats; B.edited.fill(0); return { A: summaryOut() }; },
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
