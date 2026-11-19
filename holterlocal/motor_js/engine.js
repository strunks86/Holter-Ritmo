/* HolterLocal: motor de interpretación (lectores, filtros, detección, calidad, plantillas, clasificación).
   Funciona igual en el navegador (hilo principal o Web Worker) y en Node (pruebas). Sin dependencias. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.HolterEngine = factory();
})(typeof self !== 'undefined' ? self : this, function () {
'use strict';
const E = {};
E.VERSION = '0.1.0';

/* ------------------------------------------------------------------ utilidades */
function median(arr) {
  const a = Array.prototype.slice.call(arr).sort((x, y) => x - y);
  const n = a.length; if (!n) return 0;
  return n & 1 ? a[(n - 1) >> 1] : 0.5 * (a[n / 2 - 1] + a[n / 2]);
}
function quantile(arr, q) {
  const a = Array.prototype.slice.call(arr).sort((x, y) => x - y);
  if (!a.length) return 0;
  const p = (a.length - 1) * q, i = Math.floor(p), f = p - i;
  return i + 1 < a.length ? a[i] * (1 - f) + a[i + 1] * f : a[i];
}
function clamp(x, lo, hi) { return x < lo ? lo : x > hi ? hi : x; }
E.median = median; E.quantile = quantile;

/* ------------------------------------------------------------------ lectores */
const dec = (typeof TextDecoder !== 'undefined') ? new TextDecoder('latin1') : null;
function bytesToStr(u8, a, b) {
  if (dec) return dec.decode(u8.subarray(a, b));
  let s = ''; for (let i = a; i < b; i++) s += String.fromCharCode(u8[i]); return s;
}
function cstr(u8, a, len) {
  let e = a; while (e < a + len && u8[e] !== 0) e++;
  return bytesToStr(u8, a, e).trim();
}

/* Encabezado WFDB (.hea) */
E.parseWfdbHeader = function (text) {
  const lines = text.split(/\r?\n/);
  const comments = []; const sigs = []; let rec = null;
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;
    if (line[0] === '#') { comments.push(line.replace(/^#\s?/, '')); continue; }
    const t = line.split(/\s+/);
    if (!rec) {
      const fsPart = (t[2] || '250').split('/')[0];
      rec = { name: t[0].split('/')[0], nsig: parseInt(t[1], 10), fs: parseFloat(fsPart), nsamp: parseInt(t[3] || '0', 10) || 0, baseTime: t[4] || null, baseDate: t[5] || null };
      continue;
    }
    if (sigs.length >= rec.nsig) continue;
    const fmtM = /^(\d+)(?:x(\d+))?(?::(\d+))?(?:\+(\d+))?$/.exec(t[1]) || [null, t[1]];
    const s = { file: t[0], fmt: parseInt(fmtM[1], 10), spf: parseInt(fmtM[2] || '1', 10), skew: parseInt(fmtM[3] || '0', 10), offset: parseInt(fmtM[4] || '0', 10), gain: 200, baseline: null, units: 'mV', adcres: 12, adczero: 0, name: '' };
    if (t[2]) {
      const g = /^([-\d.eE+]+)(?:\(([-\d]+)\))?(?:\/(\S+))?$/.exec(t[2]);
      if (g) { s.gain = parseFloat(g[1]) || 200; if (g[2] !== undefined) s.baseline = parseInt(g[2], 10); if (g[3]) s.units = g[3]; }
    }
    if (t[3] !== undefined) s.adcres = parseInt(t[3], 10) || 12;
    if (t[4] !== undefined) s.adczero = parseInt(t[4], 10) || 0;
    if (s.baseline === null) s.baseline = s.adczero;
    s.name = t.slice(8).join(' ') || ('Canal ' + (sigs.length + 1));
    sigs.push(s);
  }
  if (!rec) throw new Error('Encabezado WFDB vacío');
  rec.sigs = sigs; rec.comments = comments;
  return rec;
};

/* Señal WFDB: todas las señales en un mismo archivo .dat */
E.readWfdb = function (heaText, datBuf) {
  const h = E.parseWfdbHeader(heaText);
  const u8 = new Uint8Array(datBuf);
  const ns = h.sigs.length; const fmt = h.sigs[0].fmt;
  for (const s of h.sigs) { if (s.fmt !== fmt) throw new Error('Formatos mezclados en un registro WFDB: no soportado'); if (s.spf !== 1) throw new Error('Registro WFDB con varias muestras por trama: no soportado'); }
  const off = h.sigs[0].offset || 0;
  let total;
  const bytes = u8.length - off;
  if (fmt === 212) total = Math.floor(bytes / 3) * 2;
  else if (fmt === 16 || fmt === 61 || fmt === 160) total = Math.floor(bytes / 2);
  else if (fmt === 80 || fmt === 8) total = bytes;
  else if (fmt === 310 || fmt === 311) total = Math.floor(bytes / 4) * 3;
  else if (fmt === 24) total = Math.floor(bytes / 3);
  else if (fmt === 32) total = Math.floor(bytes / 4);
  else throw new Error('Formato WFDB ' + fmt + ' no soportado');
  let n = Math.floor(total / ns); if (h.nsamp && h.nsamp < n) n = h.nsamp;
  const sig = []; for (let c = 0; c < ns; c++) sig.push(new Int16Array(n));
  const put = (k, v) => { const i = (k / ns) | 0; if (i < n) sig[k % ns][i] = v > 32767 ? 32767 : v < -32768 ? -32768 : v; };
  const N = n * ns;
  if (fmt === 212) {
    for (let k = 0, p = off; k + 1 < N + 1 && p + 2 < u8.length + 1; k += 2, p += 3) {
      const b0 = u8[p], b1 = u8[p + 1], b2 = u8[p + 2];
      let s0 = b0 | ((b1 & 0x0f) << 8); if (s0 & 0x800) s0 -= 4096;
      let s1 = b2 | ((b1 & 0xf0) << 4); if (s1 & 0x800) s1 -= 4096;
      put(k, s0); if (k + 1 < N) put(k + 1, s1);
    }
  } else if (fmt === 16) {
    const dv = new DataView(datBuf, off);
    for (let k = 0; k < N; k++) put(k, dv.getInt16(k * 2, true));
  } else if (fmt === 61) {
    const dv = new DataView(datBuf, off);
    for (let k = 0; k < N; k++) put(k, dv.getInt16(k * 2, false));
  } else if (fmt === 160) {
    const dv = new DataView(datBuf, off);
    for (let k = 0; k < N; k++) put(k, dv.getUint16(k * 2, true) - 32768);
  } else if (fmt === 80) {
    for (let k = 0; k < N; k++) put(k, u8[off + k] - 128);
  } else if (fmt === 8) {
    const acc = new Int32Array(ns); for (let c = 0; c < ns; c++) acc[c] = 0;
    for (let k = 0; k < N; k++) { const d = (u8[off + k] << 24) >> 24; acc[k % ns] += d; put(k, acc[k % ns]); }
  } else if (fmt === 310 || fmt === 311) {
    for (let k = 0, p = off; k < N; k += 3, p += 4) {
      const w0 = u8[p] | (u8[p + 1] << 8), w1 = u8[p + 2] | (u8[p + 3] << 8);
      let a, b, c;
      if (fmt === 310) { a = (w0 >> 1) & 0x3ff; b = (w1 >> 1) & 0x3ff; c = ((w0 >> 11) & 0x1f) | (((w1 >> 11) & 0x1f) << 5); }
      else { const w = (w0 | (w1 << 16)) >>> 0; a = w & 0x3ff; b = (w >>> 10) & 0x3ff; c = (w >>> 20) & 0x3ff; }
      const sx = v => (v & 0x200) ? v - 1024 : v;
      put(k, sx(a)); if (k + 1 < N) put(k + 1, sx(b)); if (k + 2 < N) put(k + 2, sx(c));
    }
  } else if (fmt === 24) {
    for (let k = 0, p = off; k < N; k++, p += 3) { let v = u8[p] | (u8[p + 1] << 8) | (u8[p + 2] << 16); if (v & 0x800000) v -= 0x1000000; put(k, v); }
  } else if (fmt === 32) {
    const dv = new DataView(datBuf, off);
    for (let k = 0; k < N; k++) put(k, dv.getInt32(k * 4, true));
  }
  let start = null;
  if (h.baseTime) {
    const tm = /^(\d+):(\d+):(\d+(?:\.\d+)?)$/.exec(h.baseTime);
    if (tm) {
      let y = 2000, mo = 0, d = 1;
      const dm = h.baseDate && /^(\d+)\/(\d+)\/(\d+)$/.exec(h.baseDate);
      if (dm) { d = +dm[1]; mo = +dm[2] - 1; y = +dm[3]; }
      start = { y, mo, d, h: +tm[1], mi: +tm[2], s: Math.floor(+tm[3]), hasDate: !!dm };
    }
  }
  return {
    source: 'WFDB', name: h.name, fs: h.fs, n, nsig: ns, sig,
    gain: h.sigs.map(s => s.gain * (/^uV$/i.test(s.units) ? 1000 : /^V$/.test(s.units) ? 0.001 : 1)), // adu por mV
    baseline: h.sigs.map(s => s.baseline), names: h.sigs.map(s => s.name), start, comments: h.comments
  };
};

/* Anotaciones MIT (.atr, .qrs, .ari) */
const ANN_CODES = ['', 'N', 'L', 'R', 'a', 'V', 'F', 'J', 'A', 'S', 'E', 'j', '/', 'Q', '~', '', '|', '', 's', 'T', '*', 'D', '"', '=', 'p', 'B', '^', 't', '+', 'u', '?', '!', '[', ']', 'e', 'n', '@', 'x', 'f', '(', ')', 'r'];
const BEAT_SYMS = { N: 1, L: 1, R: 1, a: 1, V: 1, F: 1, J: 1, A: 1, S: 1, E: 1, j: 1, '/': 1, Q: 1, e: 1, n: 1, f: 1, B: 1, r: 1, '?': 1 };
E.isBeatSym = s => BEAT_SYMS[s] === 1;
E.readMitAnn = function (buf) {
  const u8 = new Uint8Array(buf); const out = { sample: [], sym: [], aux: [], sub: [] };
  let t = 0, p = 0; let pendingSub = 0;
  while (p + 1 < u8.length) {
    const w = u8[p] | (u8[p + 1] << 8); p += 2;
    const A = w >> 10, I = w & 0x3ff;
    if (A === 0 && I === 0) break;
    if (A === 59) { // SKIP
      const hi = u8[p] | (u8[p + 1] << 8), lo = u8[p + 2] | (u8[p + 3] << 8); p += 4;
      let v = (hi << 16) | lo; t += v; continue;
    }
    if (A === 60 || A === 62) continue; // NUM, CHN
    if (A === 61) { if (out.sub.length) out.sub[out.sub.length - 1] = I; continue; }
    if (A === 63) { const len = I; const s = cstr(u8, p, len); p += len + (len & 1); if (out.aux.length) out.aux[out.aux.length - 1] = s; continue; }
    t += I;
    out.sample.push(t); out.sym.push(ANN_CODES[A] || '?'); out.aux.push(''); out.sub.push(0);
  }
  return out;
};
/* Clase AAMI (N, S, V, F, Q) a partir del símbolo MIT */
E.aamiClass = function (sym) {
  switch (sym) {
    case 'N': case 'L': case 'R': case 'e': case 'j': case 'B': case 'n': return 'N';
    case 'A': case 'a': case 'J': case 'S': return 'S';
    case 'V': case 'E': case 'r': return 'V';
    case 'F': return 'F';
    default: return 'Q';
  }
};

/* ISHNE Holter Standard (.ecg) */
E.readIshne = function (buf) {
  const u8 = new Uint8Array(buf); const dv = new DataView(buf);
  const magic = bytesToStr(u8, 0, 8);
  if (magic !== 'ISHNE1.0') throw new Error('No es un archivo ISHNE (falta la firma ISHNE1.0)');
  const o = 10;
  const nSamp = dv.getInt32(o + 4, true), offEcg = dv.getInt32(o + 12, true);
  const nLeads = dv.getInt16(o + 146, true);
  const leadSpec = [], res = [];
  for (let i = 0; i < 12; i++) { leadSpec.push(dv.getInt16(o + 148 + 2 * i, true)); res.push(dv.getInt16(o + 196 + 2 * i, true)); }
  const fs = dv.getInt16(o + 262, true);
  const rd = [dv.getInt16(o + 128, true), dv.getInt16(o + 130, true), dv.getInt16(o + 132, true)];
  const st = [dv.getInt16(o + 140, true), dv.getInt16(o + 142, true), dv.getInt16(o + 144, true)];
  const LEADS = ['Desconocida', 'Bipolar', 'X', 'Y', 'Z', 'I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'ES', 'AS', 'AI'];
  const avail = Math.floor((u8.length - offEcg) / 2 / nLeads);
  const n = Math.min(nSamp > 0 ? nSamp : avail, avail);
  const sig = []; for (let c = 0; c < nLeads; c++) sig.push(new Int16Array(n));
  const d16 = new DataView(buf, offEcg);
  for (let i = 0, k = 0; i < n; i++) for (let c = 0; c < nLeads; c++, k += 2) sig[c][i] = d16.getInt16(k, true);
  return {
    source: 'ISHNE', name: (cstr(u8, o + 98, 20) || 'ISHNE'), fs, n, nsig: nLeads, sig,
    gain: res.slice(0, nLeads).map(r => 1e6 / (r > 0 ? r : 1000)), baseline: new Array(nLeads).fill(0),
    names: leadSpec.slice(0, nLeads).map((s, i) => LEADS[s] && s > 1 ? LEADS[s] : 'Canal ' + (i + 1)),
    start: rd[2] > 0 ? { y: rd[2], mo: rd[1] - 1, d: rd[0], h: st[0], mi: st[1], s: st[2], hasDate: true } : { y: 2000, mo: 0, d: 1, h: st[0], mi: st[1], s: st[2], hasDate: false },
    comments: ['Grabadora: ' + cstr(u8, o + 222, 40)]
  };
};

/* EDF, EDF+ y BDF */
E.readEdf = function (buf) {
  const u8 = new Uint8Array(buf);
  const isBdf = u8[0] === 255;
  const f = (a, len) => bytesToStr(u8, a, a + len).trim();
  const hdrBytes = parseInt(f(184, 8), 10), nrec = parseInt(f(236, 8), 10), dur = parseFloat(f(244, 8)), ns = parseInt(f(252, 4), 10);
  if (!(ns > 0) || !(hdrBytes >= 256)) throw new Error('Encabezado EDF no válido');
  let p = 256; const arr = (len, conv) => { const o = []; for (let i = 0; i < ns; i++) o.push(conv(f(p + i * len, len))); p += ns * len; return o; };
  const label = arr(16, s => s), _tr = arr(80, s => s), dim = arr(8, s => s), pmin = arr(8, parseFloat), pmax = arr(8, parseFloat), dmin = arr(8, parseFloat), dmax = arr(8, parseFloat), _pf = arr(80, s => s), spr = arr(8, s => parseInt(s, 10));
  const bps = isBdf ? 3 : 2;
  const recSamples = spr.reduce((a, b) => a + b, 0);
  let nrecEff = nrec > 0 ? nrec : Math.floor((u8.length - hdrBytes) / (recSamples * bps));
  nrecEff = Math.min(nrecEff, Math.floor((u8.length - hdrBytes) / (recSamples * bps)));
  // señales de ECG: se descartan anotaciones y señales con otra frecuencia que la mayor
  const cand = []; let maxSpr = 0;
  for (let i = 0; i < ns; i++) { if (/annotation/i.test(label[i])) continue; if (/SpO2|resp|accel|temp|event|marker|status/i.test(label[i])) continue; cand.push(i); if (spr[i] > maxSpr) maxSpr = spr[i]; }
  const use = cand.filter(i => spr[i] === maxSpr);
  if (!use.length) throw new Error('El EDF no contiene señales de ECG reconocibles');
  const n = nrecEff * maxSpr; const fs = maxSpr / (dur > 0 ? dur : 1);
  const sig = use.map(() => new Int16Array(n)); const gain = [], baseline = [];
  // escala: unidades físicas a µV y luego a enteros de 16 bits
  const scale = use.map(i => {
    const unit = dim[i].toLowerCase(); const toUv = unit === 'mv' ? 1000 : unit === 'v' ? 1e6 : 1;
    const g = (pmax[i] - pmin[i]) / (dmax[i] - dmin[i]) * toUv; // µV por unidad digital
    return { g: g || 1, off: (pmin[i] * toUv) - dmin[i] * (g || 1) };
  });
  const offs = []; { let a = 0; for (let i = 0; i < ns; i++) { offs.push(a); a += spr[i]; } }
  const dv = new DataView(buf);
  for (let r = 0; r < nrecEff; r++) {
    const base = hdrBytes + r * recSamples * bps;
    for (let u = 0; u < use.length; u++) {
      const i = use[u]; const sc = scale[u]; const dst = sig[u]; let q = base + offs[i] * bps; const o0 = r * maxSpr;
      // se guarda en unidades de 1 µV si cabe, o de 2,5 µV si la escala es mayor
      for (let k = 0; k < maxSpr; k++, q += bps) {
        let v; if (isBdf) { v = u8[q] | (u8[q + 1] << 8) | (u8[q + 2] << 16); if (v & 0x800000) v -= 0x1000000; } else v = dv.getInt16(q, true);
        const uv = v * sc.g + sc.off; dst[o0 + k] = clamp(Math.round(uv / 2.5), -32768, 32767);
      }
    }
  }
  for (let u = 0; u < use.length; u++) { gain.push(400); baseline.push(0); }
  const dm = /^(\d\d)\.(\d\d)\.(\d\d)$/.exec(f(168, 8)), tm = /^(\d\d)\.(\d\d)\.(\d\d)$/.exec(f(176, 8));
  let start = null; if (tm) { const yy = dm ? +dm[3] : 0; start = { y: dm ? (yy >= 85 ? 1900 + yy : 2000 + yy) : 2000, mo: dm ? +dm[2] - 1 : 0, d: dm ? +dm[1] : 1, h: +tm[1], mi: +tm[2], s: +tm[3], hasDate: !!dm }; }
  return { source: isBdf ? 'BDF' : 'EDF', name: 'EDF', fs, n, nsig: use.length, sig, gain, baseline, names: use.map(i => label[i].replace(/^ECG\s*/i, '') || label[i]), start, comments: [] };
};

/* Binario crudo con parámetros dados por el usuario (para explorar formatos sin documentar, como la H3+).
   opt: {fs, nch, fmt: 'i16le'|'i16be'|'u16le'|'i8'|'u8'|'p12a'|'p12b'|'i24le', offset, uvPerLsb} */
E.readRaw = function (buf, opt) {
  const u8 = new Uint8Array(buf); const off = opt.offset || 0; const nch = opt.nch; const dv = new DataView(buf);
  let total; const bytes = u8.length - off;
  switch (opt.fmt) {
    case 'i16le': case 'i16be': case 'u16le': total = Math.floor(bytes / 2); break;
    case 'i8': case 'u8': total = bytes; break;
    case 'p12a': case 'p12b': total = Math.floor(bytes / 3) * 2; break;
    case 'i24le': total = Math.floor(bytes / 3); break;
    default: throw new Error('Formato crudo desconocido');
  }
  const maxN = opt.maxSamples ? Math.min(opt.maxSamples, Math.floor(total / nch)) : Math.floor(total / nch);
  const n = maxN; const sig = []; for (let c = 0; c < nch; c++) sig.push(new Int16Array(n));
  const N = n * nch;
  const put = (k, v) => { sig[k % nch][(k / nch) | 0] = v > 32767 ? 32767 : v < -32768 ? -32768 : v; };
  if (opt.fmt === 'i16le') for (let k = 0; k < N; k++) put(k, dv.getInt16(off + 2 * k, true));
  else if (opt.fmt === 'i16be') for (let k = 0; k < N; k++) put(k, dv.getInt16(off + 2 * k, false));
  else if (opt.fmt === 'u16le') for (let k = 0; k < N; k++) put(k, dv.getUint16(off + 2 * k, true) - 32768);
  else if (opt.fmt === 'i8') for (let k = 0; k < N; k++) put(k, (u8[off + k] << 24) >> 24);
  else if (opt.fmt === 'u8') for (let k = 0; k < N; k++) put(k, u8[off + k] - 128);
  else if (opt.fmt === 'i24le') for (let k = 0, p = off; k < N; k++, p += 3) { let v = u8[p] | (u8[p + 1] << 8) | (u8[p + 2] << 16); if (v & 0x800000) v -= 0x1000000; put(k, v >> 4); }
  else for (let k = 0, p = off; k < N; k += 2, p += 3) {
    const b0 = u8[p], b1 = u8[p + 1], b2 = u8[p + 2]; let s0, s1;
    if (opt.fmt === 'p12a') { s0 = b0 | ((b1 & 0x0f) << 8); s1 = b2 | ((b1 & 0xf0) << 4); }
    else { s0 = (b0 << 4) | (b1 >> 4); s1 = ((b1 & 0x0f) << 8) | b2; }
    if (s0 & 0x800) s0 -= 4096; if (s1 & 0x800) s1 -= 4096;
    put(k, s0); if (k + 1 < N) put(k + 1, s1);
  }
  const g = 1000 / (opt.uvPerLsb || 6.25);
  return { source: 'Binario crudo', name: opt.name || 'registro', fs: opt.fs, n, nsig: nch, sig, gain: new Array(nch).fill(g), baseline: new Array(nch).fill(0), names: Array.from({ length: nch }, (_, i) => 'Canal ' + (i + 1)), start: null, comments: [] };
};
/* Puntúa qué tan "parecida a un ECG" es una interpretación de un binario: señal suave con picos esporádicos. */
E.scoreRawLayout = function (buf, opt) {
  let r; try { r = E.readRaw(buf, Object.assign({}, opt, { maxSamples: Math.min(20000, 40 * opt.fs) })); } catch (e) { return 0; }
  let score = 0;
  for (let c = 0; c < r.nsig; c++) {
    const x = r.sig[c]; const n = x.length; if (n < 200) return 0;
    let sd = 0, sv = 0, m = 0; for (let i = 0; i < n; i++) m += x[i]; m /= n;
    for (let i = 1; i < n; i++) { sd += Math.abs(x[i] - x[i - 1]); } for (let i = 0; i < n; i++) sv += Math.abs(x[i] - m);
    if (sv === 0) return 0;
    score += 1 - Math.min(1, (sd / (n - 1)) / (2 * sv / n)); // 1 = muy suave, 0 = ruido blanco
  }
  return score / r.nsig;
};

/* Texto: columnas numéricas (CSV o separadas por espacios o tabuladores) */
E.readText = function (text, fs, uvPerUnit) {
  const lines = text.split(/\r?\n/); const rows = [];
  for (const ln of lines) {
    const t = ln.trim(); if (!t) continue; const parts = t.split(/[\s,;]+/).map(Number);
    if (parts.some(v => !isFinite(v))) continue; rows.push(parts);
  }
  if (rows.length < 10) throw new Error('El archivo de texto no tiene columnas numéricas suficientes');
  let nc = rows[0].length; let first = 0;
  // si la primera columna es monótona creciente se toma como tiempo y se descarta
  let mono = true; for (let i = 1; i < Math.min(rows.length, 500); i++) if (!(rows[i][0] > rows[i - 1][0])) { mono = false; break; }
  if (mono && nc > 1) { first = 1; if (!fs) { const dt = (rows[Math.min(rows.length, 500) - 1][0] - rows[0][0]) / (Math.min(rows.length, 500) - 1); if (dt > 0) fs = dt < 0.1 ? Math.round(1 / dt) : Math.round(1000 / dt); } }
  const n = rows.length; const ns = nc - first; const sig = [];
  let amax = 0; for (let i = 0; i < n; i++) for (let c = first; c < nc; c++) { const v = Math.abs(rows[i][c]); if (v > amax) amax = v; }
  const k = uvPerUnit || (amax < 20 ? 1000 : 1); // si los valores parecen mV se pasan a µV
  for (let c = 0; c < ns; c++) { const a = new Int16Array(n); for (let i = 0; i < n; i++) a[i] = clamp(Math.round(rows[i][c + first] * k / 2.5), -32768, 32767); sig.push(a); }
  return { source: 'Texto', name: 'texto', fs: fs || 250, n, nsig: ns, sig, gain: new Array(ns).fill(400), baseline: new Array(ns).fill(0), names: Array.from({ length: ns }, (_, i) => 'Canal ' + (i + 1)), start: null, comments: [] };
};

/* Mortara H12+ ya extraída: archivos HourNRawData.bin (int16, 8 derivaciones intercaladas: I, II, V1 a V6; 1000 Hz).
   files: [{name, buf}] en cualquier orden. Supuesto tomado de ecg-kit; ganancia por confirmar con archivos reales. */
E.readMortaraHours = function (files, opt) {
  opt = opt || {}; const nch = opt.nch || 8, fs = opt.fs || 1000, uv = opt.uvPerLsb || 2.5;
  const sorted = files.slice().sort((a, b) => (parseInt((/(\d+)/.exec(a.name) || [0, 0])[1], 10)) - (parseInt((/(\d+)/.exec(b.name) || [0, 0])[1], 10)));
  let n = 0; for (const f of sorted) n += Math.floor(f.buf.byteLength / 2 / nch);
  const sig = []; for (let c = 0; c < nch; c++) sig.push(new Int16Array(n));
  let o = 0;
  for (const f of sorted) { const dv = new DataView(f.buf); const m = Math.floor(f.buf.byteLength / 2 / nch); for (let i = 0, k = 0; i < m; i++) for (let c = 0; c < nch; c++, k += 2) sig[c][o + i] = dv.getInt16(k, true); o += m; }
  const names = nch === 8 ? ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] : Array.from({ length: nch }, (_, i) => 'Canal ' + (i + 1));
  return { source: 'Mortara H12+ (horas crudas)', name: 'H12+', fs, n, nsig: nch, sig, gain: new Array(nch).fill(1000 / uv), baseline: new Array(nch).fill(0), names, start: null, comments: [] };
};

/* ------------------------------------------------------------------ filtros */
function biquad(type, fc, fs, Q) {
  const w0 = 2 * Math.PI * fc / fs, c = Math.cos(w0), s = Math.sin(w0), al = s / (2 * (Q || Math.SQRT1_2));
  let b0, b1, b2; const a0 = 1 + al, a1 = -2 * c, a2 = 1 - al;
  if (type === 'lp') { b0 = (1 - c) / 2; b1 = 1 - c; b2 = (1 - c) / 2; }
  else if (type === 'hp') { b0 = (1 + c) / 2; b1 = -(1 + c); b2 = (1 + c) / 2; }
  else { b0 = 1; b1 = -2 * c; b2 = 1; } // notch
  return [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0];
}
function runBiquad(x, k, reverse) {
  const b0 = k[0], b1 = k[1], b2 = k[2], a1 = k[3], a2 = k[4]; const n = x.length; if (!n) return;
  const G = (b0 + b1 + b2) / (1 + a1 + a2);
  if (!reverse) {
    const x0 = x[0], y0 = G * x0; let z1 = y0 - b0 * x0, z2 = b2 * x0 - a2 * y0;
    for (let i = 0; i < n; i++) { const xi = x[i]; const y = b0 * xi + z1; z1 = b1 * xi - a1 * y + z2; z2 = b2 * xi - a2 * y; x[i] = y; }
  } else {
    const x0 = x[n - 1], y0 = G * x0; let z1 = y0 - b0 * x0, z2 = b2 * x0 - a2 * y0;
    for (let i = n - 1; i >= 0; i--) { const xi = x[i]; const y = b0 * xi + z1; z1 = b1 * xi - a1 * y + z2; z2 = b2 * xi - a2 * y; x[i] = y; }
  }
}
function filtfilt(x, k) { runBiquad(x, k, false); runBiquad(x, k, true); }
E.biquad = biquad; E.filtfilt = filtfilt;

/* media móvil centrada, en sitio, con ventana de w muestras */
function movAvg(x, w, tmp) {
  const n = x.length; const h = w >> 1; let acc = 0; const out = tmp || new Float32Array(n);
  for (let i = 0; i < n + h; i++) {
    if (i < n) acc += x[i];
    if (i - w >= 0) acc -= x[i - w];
    const c = i - h; if (c >= 0 && c < n) out[c] = acc / w;
  }
  x.set(out.subarray(0, n));
}

/* ------------------------------------------------------------------ preparación de un canal
   Devuelve la señal de análisis (µV, 0,5 a 40 Hz, a frecuencia fsA) y métricas de calidad por segundo. */
E.analysisRate = function (fs) {
  const dec = fs >= 300 ? Math.max(1, Math.round(fs / 200)) : 1;
  return { dec, fsA: fs / dec };
};
function prepChannel(raw, gain, baseline, fs, decim) {
  const n0 = raw.length; const n = Math.floor(n0 / decim); const fsA = fs / decim;
  const x = new Float32Array(n); const k = 1000 / gain;
  // rieles del conversor (para detectar saturación)
  let rmin = 32767, rmax = -32768; for (let i = 0; i < n0; i += 7) { const v = raw[i]; if (v < rmin) rmin = v; if (v > rmax) rmax = v; }
  if (decim === 1) for (let i = 0; i < n; i++) x[i] = (raw[i] - baseline) * k;
  else for (let i = 0, j = 0; i < n; i++) { let a = 0; for (let d = 0; d < decim; d++) a += raw[j++]; x[i] = (a / decim - baseline) * k; }
  const nsec = Math.ceil(n / fsA);
  const q = { flat: new Uint8Array(nsec), sat: new Uint8Array(nsec), hf: new Float32Array(nsec), bw: new Float32Array(nsec), amp: new Float32Array(nsec) };
  // saturación y línea plana sobre la señal sin filtrar
  const satLo = (rmin - baseline) * k + 1e-3, satHi = (rmax - baseline) * k - 1e-3;
  for (let s = 0; s < nsec; s++) {
    const a = Math.round(s * fsA), b = Math.min(n, Math.round((s + 1) * fsA)); let mn = 1e9, mx = -1e9, ns = 0;
    for (let i = a; i < b; i++) { const v = x[i]; if (v < mn) mn = v; if (v > mx) mx = v; if (v <= satLo || v >= satHi) ns++; }
    if (mx - mn < 25) q.flat[s] = 1;
    if (ns > 0.04 * (b - a) && (rmax - rmin) > 500) q.sat[s] = 1;
  }
  // deriva de la línea de base: paso alto de 0,5 Hz sin desfase
  const lowp = new Float32Array(x); // copia
  const hp = biquad('hp', 0.5, fsA); filtfilt(x, hp);
  for (let s = 0; s < nsec; s++) {
    const a = Math.round(s * fsA), b = Math.min(n, Math.round((s + 1) * fsA)); let mn = 1e9, mx = -1e9;
    for (let i = a; i < b; i++) { const v = lowp[i] - x[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
    q.bw[s] = mx - mn;
  }
  // paso bajo de 40 Hz sin desfase; lo que queda por encima es ruido de alta frecuencia
  lowp.set(x);
  const fc = Math.min(40, fsA * 0.4); const lp = biquad('lp', fc, fsA); filtfilt(x, lp);
  for (let s = 0; s < nsec; s++) {
    const a = Math.round(s * fsA), b = Math.min(n, Math.round((s + 1) * fsA)); let e = 0, mx = 0;
    for (let i = a; i < b; i++) { const d = lowp[i] - x[i]; e += d * d; const v = Math.abs(x[i]); if (v > mx) mx = v; }
    q.hf[s] = Math.sqrt(e / Math.max(1, b - a)); q.amp[s] = mx;
  }
  // función de detección: 5 a 20 Hz, derivada al cuadrado, integración de 100 ms y raíz
  const f = lowp; f.set(x);
  filtfilt(f, biquad('hp', 5, fsA)); filtfilt(f, biquad('lp', Math.min(20, fsA * 0.35), fsA));
  let prev = f[0];
  for (let i = 0; i < n; i++) { const d = f[i] - prev; prev = f[i]; f[i] = d * d; }
  const w = Math.max(3, Math.round(0.1 * fsA)); movAvg(f, w);
  for (let i = 0; i < n; i++) f[i] = Math.sqrt(f[i] > 0 ? f[i] : 0);
  const ana = new Int16Array(n); for (let i = 0; i < n; i++) { const v = x[i]; ana[i] = v > 32767 ? 32767 : v < -32768 ? -32768 : Math.round(v); }
  return { ana, feat: f, q, fsA, n };
}

/* escala local de la función de detección: mediana móvil de los máximos por bloques de 2 s */
function localScale(feat, fsA) {
  const blk = Math.round(2 * fsA); const nb = Math.ceil(feat.length / blk); const mx = new Float32Array(nb);
  for (let b = 0; b < nb; b++) { let m = 0; const e = Math.min(feat.length, (b + 1) * blk); for (let i = b * blk; i < e; i++) if (feat[i] > m) m = feat[i]; mx[b] = m; }
  const sc = new Float32Array(nb); const H = 8; const buf = [];
  for (let b = 0; b < nb; b++) { buf.length = 0; for (let j = Math.max(0, b - H); j <= Math.min(nb - 1, b + H); j++) buf.push(mx[j]); sc[b] = median(buf); }
  return { sc, blk, mx };
}

/* calidad por segundo de un canal: 0 buena, 1 aceptable (sirve para ritmo), 2 inservible.
   Combina línea plana, saturación, ruido de alta frecuencia, deriva, amplitud anómala y ruido dentro de la
   banda del QRS (cruces y ocupación de la función de detección, que en un ECG limpio es casi cero entre latidos). */
function channelQuality(q, feat, ls, fsA) {
  const nsec = q.hf.length; const out = new Uint8Array(nsec); const n = feat.length;
  const amps = []; for (let s = 0; s < nsec; s += 3) if (!q.flat[s]) amps.push(q.amp[s]);
  const A = Math.max(150, median(amps));
  const sc = ls.sc, blk = ls.blk; const scAll = median(sc) || 1;
  const cross = new Uint8Array(nsec), minor = new Uint8Array(nsec), busy = new Float32Array(nsec);
  let up = false, pk = 0;
  for (let s = 0; s < nsec; s++) {
    const a = Math.round(s * fsA), b = Math.min(n, Math.round((s + 1) * fsA)); let nb = 0, nc = 0, nm = 0;
    for (let i = a; i < b; i++) {
      const v = feat[i] / Math.max(sc[(i / blk) | 0], 0.25 * scAll);
      if (v > 0.3) nb++;
      if (!up && v > 0.3) { up = true; pk = v; }
      else if (up) { if (v > pk) pk = v; if (v < 0.2) { up = false; nc++; if (pk < 0.6) nm++; } }
    }
    cross[s] = nc; minor[s] = nm; busy[s] = nb / Math.max(1, b - a);
  }
  // señal ausente: amplitud muy baja durante 7 s seguidos (electrodo suelto sin ruido, o asistolia: se informa aparte para revisión)
  const low = new Uint8Array(nsec);
  for (let s = 0; s < nsec; s++) { let m = 0; for (let j = Math.max(0, s - 3); j <= Math.min(nsec - 1, s + 3); j++) if (q.amp[j] > m) m = q.amp[j]; if (m < 0.15 * A) low[s] = 1; }
  for (let s = 0; s < nsec; s++) {
    let c = 0;
    if (q.flat[s] || q.sat[s] || low[s]) c = 2;
    else {
      const hf = q.hf[s] / A, bw = q.bw[s] / A, am = q.amp[s] / A;
      // ruido en banda evaluado con el segundo vecino para no castigar un latido ancho aislado
      const cr = cross[s] + (s + 1 < nsec ? cross[s + 1] : cross[s]), mi = minor[s] + (s + 1 < nsec ? minor[s + 1] : minor[s]);
      if (hf > 0.20 || bw > 6 || am > 6 || cr >= 13 || busy[s] > 0.62) c = 2;
      else if (hf > 0.08 || bw > 2.5 || am > 3.5 || cr >= 10 || mi >= 3 || busy[s] > 0.5) c = 1;
    }
    out[s] = c;
  }
  for (let s = 0; s < nsec; s++) if (q.flat[s]) low[s] = 1;
  return { cls: out, A, flat: q.flat, sat: q.sat, low };
}

/* ------------------------------------------------------------------ detección de QRS sobre la función fusionada */
function detectQrs(F, fsA, P, segCls) {
  P = P || {};
  const TH = P.th || 0.30; const n = F.length;
  const refr = Math.round(0.20 * fsA), tw = Math.round(0.36 * fsA), minDist = Math.round(0.12 * fsA);
  const qrs = []; const qh = [];
  // candidatos: máximos locales separados al menos minDist
  const cand = [], candH = [];
  let lastI = -1e9, lastH = 0;
  for (let i = 1; i < n - 1; i++) {
    const v = F[i];
    if (v > F[i - 1] && v >= F[i + 1] && v > 0.08) {
      if (i - lastI < minDist) { if (v > lastH) { cand[cand.length - 1] = i; candH[candH.length - 1] = v; lastI = i; lastH = v; } }
      else { cand.push(i); candH.push(v); lastI = i; lastH = v; }
    }
  }
  const Q = [], Nz = [], RR = [];
  const push8 = (a, v) => { a.push(v); if (a.length > 8) a.shift(); };
  // arranque: mediana de los máximos por segundo de los primeros 8 s
  { const init = []; for (let s = 0; s < 8; s++) { let m = 0; const e = Math.min(n, Math.round((s + 1) * fsA)); for (let i = Math.round(s * fsA); i < e; i++) if (F[i] > m) m = F[i]; init.push(m); } const m0 = median(init) || 1; for (let i = 0; i < 8; i++) { Q.push(m0); Nz.push(m0 * 0.1); } }
  let qm = median(Q), nm = median(Nz), rrm = fsA; let lastQ = -1e9, lastQh = 0;
  let pend = []; // candidatos bajo umbral desde el último QRS
  const accept = (i, h) => { if (lastQ > -1e8) { const rr = i - lastQ; if (rr < 2.5 * fsA) { push8(RR, rr); rrm = median(RR); } } qrs.push(i); qh.push(h); lastQ = i; lastQh = h; push8(Q, Math.min(h, 3 * qm)); qm = Math.max(0.25, median(Q)); pend = []; };
  for (let c = 0; c < cand.length; c++) {
    const i = cand[c], h = candH[c];
    // búsqueda hacia atrás si pasó demasiado tiempo sin QRS
    if (lastQ > -1e8 && i - lastQ > 1.5 * rrm && pend.length) {
      let best = -1, bh = 0; const thr2 = nm + 0.5 * TH * (qm - nm);
      for (let k = 0; k < pend.length; k++) { const pi = pend[k][0], ph = pend[k][1]; if (pi - lastQ > tw && ph > thr2 && ph > bh) { bh = ph; best = k; } }
      if (best >= 0) { const pi = pend[best][0]; const rest = pend.slice(best + 1); accept(pi, bh); for (const r of rest) if (r[0] - pi > refr) pend.push(r); }
    }
    if (i - lastQ < refr) { if (h > lastQh * 1.5 && h > qm) { /* pico mucho mayor dentro del refractario: se reemplaza */ qrs[qrs.length - 1] = i; qh[qh.length - 1] = h; lastQ = i; lastQh = h; } continue; }
    const sc_ = segCls ? segCls[Math.min(segCls.length - 1, (i / fsA) | 0)] : 0;
    const thr = nm + (sc_ === 2 ? 0.62 : sc_ === 1 ? 0.45 : TH) * (qm - nm);
    if (h > thr) {
      if (i - lastQ < tw && h < 0.5 * lastQh) { push8(Nz, h); nm = median(Nz); pend.push([i, h]); continue; } // probable onda T
      accept(i, h);
    } else { push8(Nz, h); nm = median(Nz); pend.push([i, h]); if (pend.length > 40) pend.shift(); }
  }
  return { idx: qrs, h: qh };
}

/* ------------------------------------------------------------------ paso 1: señales -> latidos */
E.DEFAULTS = {
  th: 0.30,
  svPrematurity: 0.25, pauseSec: 2.0, tachyBpm: 100, bradyBpm: 50, vtBpm: 100, vtBeats: 3, svtBpm: 100, svtBeats: 3,
  minRateDurSec: 30, stThreshUv: 100, stPointMs: 60, qtcFormula: 'bazett', afDetect: true, hrvNormalOnly: true
};

E.pass1 = function (rec, opt, progress) {
  opt = Object.assign({}, E.DEFAULTS, opt || {}); progress = progress || function () { };
  const { dec, fsA } = E.analysisRate(rec.fs);
  const chans = (opt.channels && opt.channels.length) ? opt.channels : (rec.nsig <= 3 ? rec.sig.map((_, i) => i) : pickChannels(rec));
  const nch = chans.length; const ana = [], qual = [], amps = [];
  let F = null, W = null, n = 0, nsec = 0, nosig = null;
  for (let ci = 0; ci < nch; ci++) {
    progress('Filtrando el canal ' + (ci + 1) + ' de ' + nch, 0.05 + 0.35 * ci / nch);
    const c = chans[ci]; const p = prepChannel(rec.sig[c], rec.gain[c], rec.baseline[c], rec.fs, dec);
    n = p.n; nsec = p.q.hf.length;
    const ls = localScale(p.feat, fsA); const cq = channelQuality(p.q, p.feat, ls, fsA);
    if (!F) { F = new Float32Array(n); W = new Float32Array(n); }
    const feat = p.feat; const blk = ls.blk; const sc = ls.sc;
    // escala mínima para no amplificar ruido puro en canales sin señal
    const scAll = median(sc) || 1;
    for (let s = 0; s < nsec; s++) {
      const w = (cq.flat[s] || cq.sat[s] || cq.low[s]) ? 0 : cq.cls[s] === 2 ? 0.12 : cq.cls[s] === 1 ? 0.4 : 1; if (!w) continue;
      const a = Math.round(s * fsA), b = Math.min(n, Math.round((s + 1) * fsA));
      for (let i = a; i < b; i++) { const sv = Math.max(sc[(i / blk) | 0], 0.25 * scAll); F[i] += w * feat[i] / sv; W[i] += w; }
    }
    ana.push(p.ana); qual.push(cq.cls); amps.push(cq.A);
    if (!nosig) nosig = Uint8Array.from(cq.low); else for (let s = 0; s < nsec; s++) nosig[s] &= cq.low[s];
    p.feat = null;
  }
  for (let i = 0; i < n; i++) F[i] = W[i] > 0 ? F[i] / W[i] : 0;
  W = null;
  progress('Detectando latidos', 0.42);
  // calidad del mejor canal en cada segundo: gobierna el umbral de detección
  const best = new Uint8Array(nsec); for (let s = 0; s < nsec; s++) { let m = 2; for (let ci = 0; ci < nch; ci++) if (qual[ci][s] < m) m = qual[ci][s]; best[s] = m; }
  const det = detectQrs(F, fsA, opt, best);
  // punto fiducial: máximo de la desviación absoluta combinada en ±70 ms
  const nb = det.idx.length; const fid = new Int32Array(nb); const h = Math.round(0.07 * fsA), hb = Math.round(0.15 * fsA);
  for (let b = 0; b < nb; b++) {
    const p = det.idx[b]; const sec = Math.min(nsec - 1, Math.floor(p / fsA));
    const a = Math.max(0, p - h), e = Math.min(n - 1, p + h); let best = p, bv = -1;
    const base = [];
    for (let ci = 0; ci < nch; ci++) { let s = 0, cnt = 0; const x = ana[ci]; for (let i = Math.max(0, p - hb); i <= Math.min(n - 1, p + hb); i += 2) { s += x[i]; cnt++; } base.push(s / cnt); }
    for (let i = a; i <= e; i++) { let v = 0; for (let ci = 0; ci < nch; ci++) { if (qual[ci][sec] === 2) continue; v += Math.abs(ana[ci][i] - base[ci]) / amps[ci]; } if (v > bv) { bv = v; best = i; } }
    fid[b] = best;
  }
  return { fsA, dec, chans, n, nsec, ana, qual, best, nosig, amps, F, fid, peakH: Float32Array.from(det.h), opt };
};
/* ------------------------------------------------------------------ morfología, plantillas y rasgos por latido */
const MG = 56, MSTEP = 0.005, MPRE = 0.110; // rejilla de 56 puntos cada 5 ms: de -110 a +165 ms respecto del fiducial
E.MORPH = { M: MG, step: MSTEP, pre: MPRE };

function beatVectors(p1) {
  const { ana, fid, fsA, n, amps, qual } = p1; const nch = ana.length, nb = fid.length;
  const V = new Float32Array(nb * nch * MG); const mask = new Uint8Array(nb);
  for (let b = 0; b < nb; b++) {
    const f = fid[b]; const sec = Math.min(p1.nsec - 1, Math.floor(f / fsA)); let m = 0;
    for (let c = 0; c < nch; c++) {
      if (qual[c][sec] < 2) m |= (1 << c);
      const x = ana[c]; const o = (b * nch + c) * MG; const inv = 1 / amps[c];
      let base = 0;
      for (let k = 0; k < MG; k++) {
        const t = f + (k * MSTEP - MPRE) * fsA; let i0 = Math.floor(t); const fr = t - i0;
        if (i0 < 0) i0 = 0; if (i0 > n - 2) i0 = n - 2;
        const v = (x[i0] * (1 - fr) + x[i0 + 1] * fr) * inv; V[o + k] = v; if (k < 4) base += v;
      }
      base /= 4; for (let k = 0; k < MG; k++) V[o + k] -= base;
    }
    if (!m) m = (1 << nch) - 1; // si todos los canales son malos se usan todos igual
    mask[b] = m;
  }
  return { V, mask };
}

/* correlación de un latido con una plantilla, con desplazamiento de hasta ±3 puntos (±15 ms), solo canales válidos */
function corrShift(V, ob, T, ot, nch, mask, out) {
  let best = -2, bs = 0, br = 1;
  for (let si = 0; si < 7; si++) {
    const sh = si === 0 ? 0 : (si & 1 ? (si + 1) >> 1 : -(si >> 1));
    let sxy = 0, sxx = 0, syy = 0;
    for (let c = 0; c < nch; c++) {
      if (!(mask & (1 << c))) continue;
      const a = ob + c * MG, t = ot + c * MG; const k0 = sh > 0 ? sh : 0, k1 = sh < 0 ? MG + sh : MG;
      for (let k = k0; k < k1; k++) { const x = V[a + k], y = T[t + k - sh]; sxy += x * y; sxx += x * x; syy += y * y; }
    }
    const c = sxy / (Math.sqrt(sxx * syy) + 1e-9);
    if (c > best) { best = c; bs = sh; br = Math.sqrt((sxx + 1e-9) / (syy + 1e-9)); }
    if (si === 0 && c < 0.3) break; // muy distinto: no vale la pena desplazar
    if (c > 0.985) break;
  }
  out[0] = best; out[1] = bs; out[2] = br;
}

function clusterBeats(bv, nch, nb, P) {
  const V = bv.V, mask = bv.mask; const D = nch * MG;
  const thr = (P && P.tmplCorr) || 0.93; const MAXT = 600;
  const T = []; const cnt = []; const order = []; // order: plantillas por uso reciente
  const tid = new Int32Array(nb); const tcorr = new Float32Array(nb); const tshift = new Int8Array(nb);
  const o3 = [0, 0, 0];
  for (let b = 0; b < nb; b++) {
    const ob = b * D; let best = -1, bc = -2, bsh = 0, bpos = -1;
    const lim = Math.min(order.length, 80);
    for (let oi = 0; oi < lim; oi++) {
      const t = order[oi]; corrShift(V, ob, T[t], 0, nch, mask[b], o3);
      const ok = o3[2] > 0.55 && o3[2] < 1.8;
      const c = ok ? o3[0] : o3[0] - 0.2;
      if (c > bc) { bc = c; best = t; bsh = o3[1]; bpos = oi; }
      if (c >= 0.975) break;
    }
    if (best >= 0 && (bc >= thr || T.length >= MAXT)) {
      const t = best; const tm = T[t]; cnt[t]++; const al = Math.max(1 / cnt[t], 1 / 48);
      if (bc >= thr) for (let c = 0; c < nch; c++) { if (!(mask[b] & (1 << c))) continue; const a = ob + c * MG, tt = c * MG; const k0 = bsh > 0 ? bsh : 0, k1 = bsh < 0 ? MG + bsh : MG; for (let k = k0; k < k1; k++) tm[tt + k - bsh] += al * (V[a + k] - tm[tt + k - bsh]); }
      tid[b] = t; tcorr[b] = bc; tshift[b] = bsh;
      if (bpos > 0) { order.splice(bpos, 1); order.unshift(t); }
    } else {
      const t = T.length; T.push(Float32Array.from(V.subarray(ob, ob + D))); cnt.push(1); order.unshift(t);
      tid[b] = t; tcorr[b] = 1; tshift[b] = 0;
    }
  }
  // fusión de plantillas casi idénticas (de mayor a menor población)
  const nt = T.length; const parent = new Int32Array(nt); for (let i = 0; i < nt; i++) parent[i] = i;
  const idx = Array.from({ length: nt }, (_, i) => i).sort((a, b) => cnt[b] - cnt[a]);
  const full = (1 << nch) - 1; const lim2 = Math.min(nt, 250);
  for (let i = 1; i < lim2; i++) {
    const a = idx[i];
    for (let j = 0; j < i; j++) {
      const b = idx[j]; if (parent[b] !== b) continue;
      corrShift(T[a], 0, T[b], 0, nch, full, o3);
      if (o3[0] >= 0.965 && o3[2] > 0.7 && o3[2] < 1.45) { parent[a] = b; break; }
    }
  }
  const remap = new Int32Array(nt).fill(-1); const TT = [], CC = []; 
  for (const i of idx) { if (parent[i] === i) { remap[i] = TT.length; TT.push(T[i]); CC.push(0); } }
  for (let i = 0; i < nt; i++) if (parent[i] !== i) remap[i] = remap[parent[i]];
  for (let b = 0; b < nb; b++) { tid[b] = remap[tid[b]]; CC[tid[b]]++; }
  return { tid, tcorr, tshift, T: TT, count: CC };
}

/* ancho del QRS (ms) sobre un vector de morfología multicanal: envolvente de pendiente combinada */
function vecWidth(V, o, nch, mask) {
  const env = new Float32Array(MG); let pk = 0, pi = 0;
  for (let k = 1; k < MG - 1; k++) { let e = 0; for (let c = 0; c < nch; c++) { if (!(mask & (1 << c))) continue; const a = o + c * MG; e += Math.abs(V[a + k + 1] - V[a + k - 1]); } env[k] = e; }
  for (let k = 2; k < MG - 2; k++) { const v = (env[k - 1] + env[k] + env[k + 1]) / 3; if (v > pk) { pk = v; pi = k; } }
  if (pk <= 0) return 0;
  const th = 0.14 * pk; let a = pi, b = pi, run = 0;
  for (let k = pi; k >= 1; k--) { const v = (env[k - 1] + env[k] + env[k + 1]) / 3; if (v < th) { run++; if (run >= 3) break; } else { run = 0; a = k; } }
  run = 0;
  for (let k = pi; k < MG - 1; k++) { const v = (env[k - 1] + env[k] + env[k + 1]) / 3; if (v < th) { run++; if (run >= 3) break; } else { run = 0; b = k; } }
  return (b - a + 1) * MSTEP * 1000;
}
/* índice de polaridad (área neta / área absoluta) y pendiente máxima relativa de un vector */
function vecShape(V, o, nch, mask, out) {
  let pol = 0, slew = 0, m = 0, nrm = 0;
  for (let c = 0; c < nch; c++) {
    if (!(mask & (1 << c))) continue; const a = o + c * MG; let s = 0, sa = 0, mx = -1e9, mn = 1e9, ms = 0;
    for (let k = 4; k < MG - 4; k++) { const v = V[a + k]; s += v; sa += Math.abs(v); if (v > mx) mx = v; if (v < mn) mn = v; const d = Math.abs(V[a + k + 1] - v); if (d > ms) ms = d; nrm += v * v; }
    pol += s / (sa + 1e-9); slew += ms / (mx - mn + 1e-9); m++;
  }
  out[0] = m ? pol / m : 0; out[1] = m ? slew / m : 0; out[2] = Math.sqrt(nrm);
}

E.NFEAT = 30;
E.FEAT_NAMES = ['rrPrev', 'rrPrev/rrLoc', 'rrNext/rrLoc', 'rrNext/rrPrev', 'irregularidad', 'ancho', 'ancho/dom', 'anchoPlant', 'anchoPlant/dom', 'corrDom', 'corrPlantDom', 'amp/dom', 'fracPlant', 'prematPlant', 'corrPlant', 'discordT', 'difPolaridad', 'calidad', 'alturaPico', 'anchoDom', 'fcLocal', 'corrDomPrev', 'corrDomSig', 'pendRel', 'pendRel/dom', 'difDiscordT', 'rrPrev/rrAmplio', 'rrNext/rrAmplio', 'rrPrev/rrGlobal', 'compensadora'];

function beatFeatures(p1, bv, cl) {
  const { fid, fsA, ana, amps, n, best, peakH } = p1; const nch = ana.length, nb = fid.length, D = nch * MG; const V = bv.V, mask = bv.mask;
  const NF = E.NFEAT; const X = new Float32Array(nb * NF);
  const rr = new Float32Array(nb); for (let b = 1; b < nb; b++) rr[b] = (fid[b] - fid[b - 1]) / fsA; rr[0] = nb > 1 ? rr[1] : 1;
  // RR local: mediana de los 9 intervalos previos
  const rrLoc = new Float32Array(nb); const w = [];
  for (let b = 0; b < nb; b++) { w.length = 0; for (let j = Math.max(1, b - 9); j < b; j++) w.push(rr[j]); rrLoc[b] = w.length >= 3 ? median(w) : (rr[b] || 1); }
  const irr = new Float32Array(nb);
  for (let b = 0; b < nb; b++) { w.length = 0; for (let j = Math.max(2, b - 8); j <= Math.min(nb - 1, b + 8); j++) w.push(Math.abs(rr[j] - rr[j - 1])); irr[b] = w.length ? median(w) / (rrLoc[b] || 1) : 0; }
  // RR de contexto amplio (mediana de ±40 latidos) y RR global del registro
  const rrWide = new Float32Array(nb); { const step = 8; let last = 1; for (let b = 0; b < nb; b++) { if (b % step === 0) { w.length = 0; for (let j = Math.max(1, b - 40); j <= Math.min(nb - 1, b + 40); j += 2) w.push(rr[j]); last = w.length ? median(w) : 1; } rrWide[b] = last; } }
  let rrGlobal = 1; { w.length = 0; const st = Math.max(1, Math.floor(nb / 4000)); for (let j = 1; j < nb; j += st) w.push(rr[j]); rrGlobal = w.length ? median(w) : 1; }
  // plantillas: ancho, forma, prematuridad media
  const nt = cl.T.length; const full = (1 << nch) - 1; const tW = new Float32Array(nt), tPol = new Float32Array(nt), tSlew = new Float32Array(nt), tPrem = new Float32Array(nt), tDisc = new Float32Array(nt);
  const sh = [0, 0, 0];
  for (let t = 0; t < nt; t++) { tW[t] = vecWidth(cl.T[t], 0, nch, full); vecShape(cl.T[t], 0, nch, full, sh); tPol[t] = sh[0]; tSlew[t] = sh[1]; }
  // discordancia de la repolarización por latido: media de +130 a +330 ms respecto de la línea de base, con el signo del QRS
  const disc = new Float32Array(nb); const i1 = Math.round(0.13 * fsA), i2 = Math.round(0.33 * fsA), ib = Math.round(0.11 * fsA);
  for (let b = 0; b < nb; b++) {
    const f = fid[b]; let acc = 0, m = 0;
    for (let c = 0; c < nch; c++) {
      if (!(mask[b] & (1 << c))) continue; const x = ana[c]; if (f + i2 >= n || f - ib < 0) continue;
      const base = x[f - ib]; let s = 0; for (let i = f + i1; i < f + i2; i += 2) s += x[i] - base; s /= ((i2 - i1) / 2) * amps[c];
      const o = (b * nch + c) * MG; let q = 0; for (let k = 12; k < 36; k++) q += V[o + k];
      acc += (q >= 0 ? 1 : -1) * s; m++;
    }
    disc[b] = m ? acc / m : 0;
  }
  { const cc = new Float32Array(nt); for (let b = 0; b < nb; b++) { const t = cl.tid[b]; tPrem[t] += 1 - rr[b] / (rrLoc[b] || 1); tDisc[t] += disc[b]; cc[t]++; } for (let t = 0; t < nt; t++) if (cc[t]) { tPrem[t] /= cc[t]; tDisc[t] /= cc[t]; } }
  // plantilla dominante en una ventana deslizante de ±150 latidos (se cuentan los latidos no prematuros)
  const dom = new Int32Array(nb); const H = 150; const counts = new Map(); const countsAll = new Map();
  const reg = b => { const r = rr[b] / (rrLoc[b] || 1); return r > 0.85 && r < 1.2 && cl.tcorr[b] >= 0.9; };
  const add = (m, t, d) => { const v = (m.get(t) || 0) + d; if (v <= 0) m.delete(t); else m.set(t, v); };
  let lo = 0, hi = -1, curDom = 0;
  for (let b = 0; b < nb; b++) {
    const wantLo = Math.max(0, b - H), wantHi = Math.min(nb - 1, b + H);
    while (hi < wantHi) { hi++; add(countsAll, cl.tid[hi], 1); if (reg(hi)) add(counts, cl.tid[hi], 1); }
    while (lo < wantLo) { add(countsAll, cl.tid[lo], -1); if (reg(lo)) add(counts, cl.tid[lo], -1); lo++; }
    if ((b & 15) === 0 || b === 0) {
      let bt = -1, bcn = 0; let tot = 0; counts.forEach((v, t) => { tot += v; if (v > bcn) { bcn = v; bt = t; } });
      if (tot < 20) { bt = -1; bcn = 0; countsAll.forEach((v, t) => { if (v > bcn) { bcn = v; bt = t; } }); }
      // entre candidatas de población parecida se prefiere la más estrecha (un ritmo bigeminado no debe elegir la ventricular)
      const src = tot < 20 ? countsAll : counts; src.forEach((v, t) => { if (t !== bt && v >= 0.5 * bcn && tW[t] < tW[bt] - 15) { bt = t; } });
      curDom = bt < 0 ? cl.tid[b] : bt;
    }
    dom[b] = curDom;
  }
  // rasgos
  const o3 = [0, 0, 0]; const corrDom = new Float32Array(nb); const tdCorr = new Map();
  const bw = new Float32Array(nb), bsl = new Float32Array(nb), bpol = new Float32Array(nb), bnorm = new Float32Array(nb);
  for (let b = 0; b < nb; b++) {
    const d = dom[b]; corrShift(V, b * D, cl.T[d], 0, nch, mask[b], o3); corrDom[b] = o3[0]; const ampR = o3[2];
    bw[b] = vecWidth(V, b * D, nch, mask[b]); vecShape(V, b * D, nch, mask[b], sh); bpol[b] = sh[0]; bsl[b] = sh[1]; bnorm[b] = ampR;
  }
  for (let b = 0; b < nb; b++) {
    const t = cl.tid[b], d = dom[b]; const key = t * 100003 + d; let ctd = tdCorr.get(key);
    if (ctd === undefined) { corrShift(cl.T[t], 0, cl.T[d], 0, nch, full, o3); ctd = o3[0]; tdCorr.set(key, ctd); }
    const o = b * NF; const rl = rrLoc[b] || 1; const rn = b + 1 < nb ? rr[b + 1] : rl; const wd = Math.max(40, tW[d]);
    X[o] = rr[b]; X[o + 1] = rr[b] / rl; X[o + 2] = rn / rl; X[o + 3] = rn / (rr[b] || 1); X[o + 4] = irr[b];
    X[o + 5] = bw[b]; X[o + 6] = bw[b] / wd; X[o + 7] = tW[t]; X[o + 8] = tW[t] / wd; X[o + 9] = corrDom[b]; X[o + 10] = ctd;
    X[o + 11] = bnorm[b]; X[o + 12] = cl.count[t] / nb; X[o + 13] = tPrem[t]; X[o + 14] = cl.tcorr[b]; X[o + 15] = disc[b];
    X[o + 16] = Math.abs(bpol[b] - tPol[d]); X[o + 17] = best[Math.min(p1.nsec - 1, Math.floor(fid[b] / fsA))]; X[o + 18] = peakH[b];
    X[o + 19] = wd; X[o + 20] = 60 / rl; X[o + 21] = b > 0 ? corrDom[b - 1] : corrDom[b]; X[o + 22] = b + 1 < nb ? corrDom[b + 1] : corrDom[b];
    X[o + 23] = bsl[b]; X[o + 24] = bsl[b] / (tSlew[d] + 1e-6); X[o + 25] = disc[b] - tDisc[d];
    X[o + 26] = rr[b] / rrWide[b]; X[o + 27] = rn / rrWide[b]; X[o + 28] = rr[b] / rrGlobal; X[o + 29] = (rr[b] + rn) / (2 * rrWide[b]);
  }
  return { X, rr, rrLoc, rrWide, dom, tW, tPrem, corrDom, bw, disc };
}

E.features = function (p1, progress) {
  progress = progress || function () { };
  progress('Agrupando morfologías', 0.5);
  const bv = beatVectors(p1); const nch = p1.ana.length, nb = p1.fid.length;
  const cl = clusterBeats(bv, nch, nb, p1.opt);
  progress('Midiendo cada latido', 0.62);
  const ft = beatFeatures(p1, bv, cl);
  return { bv, cl, ft };
};

function pickChannels(rec) {
  // con 12 derivaciones se usan II, V1 y V5 si existen; si no, las tres primeras
  const want = ['II', 'V1', 'V5']; const idx = [];
  for (const w of want) { const i = rec.names.findIndex(nm => nm.toUpperCase().replace(/^ML/, '') === w); if (i >= 0) idx.push(i); }
  return idx.length === 3 ? idx : [0, 1, 2].filter(i => i < rec.nsig);
}
E.pickChannels = pickChannels;

/* ------------------------------------------------------------------ clasificador de latidos (árboles potenciados entrenados con datos públicos) */
E.CLS = { N: 0, S: 1, V: 2, F: 3, X: 4, Q: 5, B: 6, A: 7, E: 8, P: 9 };
E.CLS_SYM = ['N', 'S', 'V', 'F', 'X', 'Q', 'B', 'A', 'E', 'P'];
E.CLS_NAME = ['Normal', 'Supraventricular', 'Ventricular', 'Fusión', 'Artefacto', 'No clasificable', 'Bloqueo de rama', 'Aberrante', 'Escape ventricular', 'Estimulado'];
/* clases que cuentan como latido de origen supraventricular con conducción propia (para ritmo y VFC) */
E.isVentricular = c => c === 2 || c === 3 || c === 8;
E.isBeat = c => c !== 4;

function compileGbm(m) {
  const K = m.classes.length, IT = m.trees.length; let total = 0; for (const it of m.trees) for (const t of it) total += t.f.length;
  const f = new Int16Array(total), th = new Float32Array(total), l = new Int32Array(total), r = new Int32Array(total), v = new Float32Array(total), leaf = new Uint8Array(total), root = new Int32Array(IT * K);
  let o = 0;
  for (let i = 0; i < IT; i++) for (let k = 0; k < K; k++) {
    const t = m.trees[i][k]; root[i * K + k] = o;
    for (let j = 0; j < t.f.length; j++) { f[o + j] = t.f[j]; th[o + j] = t.t[j]; l[o + j] = o + t.l[j]; r[o + j] = o + t.r[j]; v[o + j] = t.v[j]; leaf[o + j] = t.leaf[j]; }
    o += t.f.length;
  }
  return { K, IT, f, th, l, r, v, leaf, root, base: Float32Array.from(m.base), classes: m.classes, nfeat: m.feat.length };
}
function gbmProba(g, X, o, out) {
  const K = g.K; let mx = -1e9;
  for (let k = 0; k < K; k++) {
    let a = g.base[k];
    for (let i = 0; i < g.IT; i++) { let nd = g.root[i * K + k]; while (!g.leaf[nd]) nd = X[o + g.f[nd]] <= g.th[nd] ? g.l[nd] : g.r[nd]; a += g.v[nd]; }
    out[k] = a; if (a > mx) mx = a;
  }
  let sum = 0; for (let k = 0; k < K; k++) { out[k] = Math.exp(out[k] - mx); sum += out[k]; }
  for (let k = 0; k < K; k++) out[k] /= sum;
}
E.setBeatModel = function (m) { E._beatModel = compileGbm(m); };
E.setAfModel = function (m) { E._afModel = m; };

function classifyBeats(p1, f, P) {
  const nb = p1.fid.length, NF = E.NFEAT, X = f.ft.X, cl = f.cl; const g = E._beatModel;
  const prob = new Float32Array(nb * 5); const o5 = new Float32Array(5);
  if (g) { for (let b = 0; b < nb; b++) { gbmProba(g, X, b * NF, o5); prob.set(o5, b * 5); } }
  else { // sin modelo: regla simple de respaldo (ancho, correlación y prematuridad)
    for (let b = 0; b < nb; b++) { const o = b * NF; const wide = X[o + 6] > 1.3 && X[o + 5] > 110, diff = X[o + 9] < 0.75; const v = (wide && diff) ? 0.9 : (diff && X[o + 1] < 0.85) ? 0.7 : 0.05; prob[b * 5 + 2] = v; prob[b * 5] = 1 - v; }
  }
  // suavizado por plantilla: los latidos de una misma morfología comparten origen
  const nt = cl.T.length; const tV = new Float32Array(nt), tX = new Float32Array(nt), tN = new Float32Array(nt);
  for (let b = 0; b < nb; b++) { const t = cl.tid[b]; tV[t] += prob[b * 5 + 2] + prob[b * 5 + 3]; tX[t] += prob[b * 5 + 4]; tN[t]++; }
  for (let t = 0; t < nt; t++) { tV[t] /= tN[t]; tX[t] /= tN[t]; }
  const cls = new Uint8Array(nb), conf = new Float32Array(nb), pV = new Float32Array(nb), pS = new Float32Array(nb), pX = new Float32Array(nb);
  for (let b = 0; b < nb; b++) {
    const o = b * 5, t = cl.tid[b]; let v = prob[o + 2] + prob[o + 3], x = prob[o + 4];
    if (tN[t] >= 6) { v = 0.5 * v + 0.5 * tV[t]; x = 0.6 * x + 0.4 * tX[t]; }
    const sec = Math.min(p1.nsec - 1, Math.floor(p1.fid[b] / p1.fsA));
    // en un segundo inservible en todos los canales solo se acepta un latido que calce con una morfología conocida
    if (p1.best[sec] === 2 && (cl.tcorr[b] < 0.9 || tN[t] < 5)) x = Math.max(x, 0.6);
    pV[b] = v; pX[b] = x; pS[b] = prob[o + 1];
    let c, cf;
    if (x >= 0.5 && x >= v) { c = 4; cf = x; }
    else if (v >= 0.5) { c = (prob[o + 3] > prob[o + 2] && prob[o + 3] > 0.5) ? 3 : 2; cf = v; }
    else { c = 0; cf = 1 - v - x; }
    cls[b] = c; conf[b] = cf;
  }
  return { cls, conf, pV, pS, pX, tV, tX };
}

/* ------------------------------------------------------------------ ST por latido y QT por minuto */
function templateBounds(T, nch) {
  // inicio y fin del QRS (ms respecto del fiducial) medidos sobre la plantilla
  const full = (1 << nch) - 1; const env = new Float32Array(MG); let pk = 0, pi = 0;
  for (let k = 1; k < MG - 1; k++) { let e = 0; for (let c = 0; c < nch; c++) { const a = c * MG; e += Math.abs(T[a + k + 1] - T[a + k - 1]); } env[k] = e; }
  for (let k = 2; k < MG - 2; k++) { const v = (env[k - 1] + env[k] + env[k + 1]) / 3; if (v > pk) { pk = v; pi = k; } }
  const th = 0.14 * pk; let a = pi, b = pi, run = 0;
  for (let k = pi; k >= 1; k--) { const v = (env[k - 1] + env[k] + env[k + 1]) / 3; if (v < th) { run++; if (run >= 3) break; } else { run = 0; a = k; } }
  run = 0; for (let k = pi; k < MG - 1; k++) { const v = (env[k - 1] + env[k] + env[k + 1]) / 3; if (v < th) { run++; if (run >= 3) break; } else { run = 0; b = k; } }
  return { on: (a * MSTEP - MPRE) * 1000, off: (b * MSTEP - MPRE) * 1000 };
}
function stPerBeat(p1, f, cls, P) {
  const { ana, fid, fsA, n } = p1; const nch = ana.length, nb = fid.length; const st = []; for (let c = 0; c < nch; c++) st.push(new Int16Array(nb).fill(-32768));
  const bounds = f.cl.T.map(T => templateBounds(T, nch));
  const d = ms => Math.round(ms / 1000 * fsA);
  for (let b = 0; b < nb; b++) {
    if (cls[b] !== 0) continue; const bd = bounds[f.cl.tid[b]]; const fi = fid[b];
    const hr = 60 / (f.ft.rrLoc[b] || 1); const pt = P.stPointMs || (hr > 120 ? 60 : 80);
    const i0 = fi + d(bd.on - 30), i1 = fi + d(bd.on - 8), j = fi + d(bd.off + pt);
    if (i0 < 0 || j + 2 >= n) continue;
    for (let c = 0; c < nch; c++) {
      if (p1.qual[c][Math.min(p1.nsec - 1, Math.floor(fi / fsA))] === 2) continue; const x = ana[c];
      let iso = 0, m = 0; for (let i = i0; i <= i1; i++) { iso += x[i]; m++; } iso /= Math.max(1, m);
      const v = (x[j - 1] + x[j] + x[j + 1]) / 3 - iso; st[c][b] = clamp(Math.round(v), -3000, 3000);
    }
  }
  return { st, bounds };
}
/* QT por minuto sobre el latido promedio de los latidos normales del minuto (método de la tangente) */
E._qt = { onTh: 0.06, onRun: 0.012, tpkMin: 0.16, tpkMax: 0.52, teMax: 0.30, w: 0.128, signed: true, latest: 0 };
function qtPerMinute(p1, f, cls) {
  const { ana, fid, fsA, n } = p1; const nch = ana.length, nb = fid.length; const nmin = Math.ceil(p1.nsec / 60);
  const qt = new Float32Array(nmin).fill(NaN), rrm = new Float32Array(nmin).fill(NaN), qch = new Int8Array(nmin).fill(-1), tpk = new Float32Array(nmin).fill(NaN), qon = new Float32Array(nmin).fill(NaN);
  const pre = Math.round(0.30 * fsA), post = Math.round(0.62 * fsA), L = pre + post; const avg = []; for (let c = 0; c < nch; c++) avg.push(new Float32Array(L));
  const bounds = f.cl.T.map(T => templateBounds(T, nch));
  let b = 0;
  for (let mi = 0; mi < nmin; mi++) {
    const tEnd = (mi + 1) * 60 * fsA; for (let c = 0; c < nch; c++) avg[c].fill(0); let cnt = 0, rrs = 0; const tcount = new Map();
    const b0 = b; while (b < nb && fid[b] < tEnd) b++;
    // plantilla más frecuente del minuto entre los latidos normales
    for (let k = b0; k < b; k++) if (cls[k] === 0) tcount.set(f.cl.tid[k], (tcount.get(f.cl.tid[k]) || 0) + 1);
    let bt = -1, bc = 0; tcount.forEach((v, t) => { if (v > bc) { bc = v; bt = t; } }); if (bt < 0 || bc < 8) continue;
    for (let k = b0; k < b; k++) {
      if (cls[k] !== 0 || f.cl.tid[k] !== bt || f.cl.tcorr[k] < 0.9) continue; const fi = fid[k]; if (fi - pre < 0 || fi + post >= n) continue;
      const r1 = f.ft.rr[k], r0 = f.ft.rrLoc[k]; if (!(r1 > 0.85 * r0 && r1 < 1.15 * r0)) continue; if (k + 1 < nb && cls[k + 1] !== 0) continue;
      if (p1.best[Math.min(p1.nsec - 1, Math.floor(fi / fsA))] !== 0) continue;
      for (let c = 0; c < nch; c++) { const x = ana[c], a = avg[c]; for (let i = 0; i < L; i++) a[i] += x[fi - pre + i]; }
      cnt++; rrs += r1;
    }
    if (cnt < 8) continue; const rr = rrs / cnt; rrm[mi] = rr; const bd = bounds[bt]; const Q = E._qt;
    for (let c = 0; c < nch; c++) { const a = avg[c]; for (let i = 0; i < L; i++) a[i] /= cnt; }
    // inicio del QRS sobre el latido promedio: última muestra, yendo hacia atrás desde la pendiente máxima, en que la pendiente combinada supera el umbral
    const der = new Float32Array(L); const w60 = Math.round(0.06 * fsA);
    for (let c = 0; c < nch; c++) { const a = avg[c]; let mn = 1e9, mx = -1e9; for (let i = pre - w60; i <= pre + w60; i++) { if (a[i] < mn) mn = a[i]; if (a[i] > mx) mx = a[i]; } const sc = 1 / Math.max(50, mx - mn); for (let i = 1; i < L - 1; i++) der[i] += Math.abs(a[i + 1] - a[i - 1]) * sc; }
    let pk = 0, pi = pre; for (let i = pre - w60; i <= pre + w60; i++) { const v = (der[i - 1] + der[i] + der[i + 1]) / 3; if (v > pk) { pk = v; pi = i; } }
    const nrun = Math.max(2, Math.round(Q.onRun * fsA)); let on = pi, run = 0; const lo = Math.max(2, pre - Math.round(0.13 * fsA));
    for (let k = pi; k >= lo; k--) { const v = (der[k - 1] + der[k] + der[k + 1]) / 3; if (v < Q.onTh * pk) { run++; if (run >= nrun) break; } else { run = 0; on = k; } }
    on -= 1; const off = pre + Math.round(bd.off / 1000 * fsA);
    const sq = Math.sqrt(rr); const sEnd = Math.min(L - 2, on + Math.round(Q.tpkMax * sq * fsA)); const s0 = Math.max(off + Math.round(0.06 * fsA), on + Math.round(Q.tpkMin * sq * fsA)); const eLim = Math.min(L - 3, on + Math.round(0.66 * sq * fsA), pre + Math.round((rr - 0.10) * fsA));
    const wA = Math.max(4, Math.round(Q.w * fsA)), pS = Math.max(1, Math.round(0.016 * fsA));
    const cand = [];
    for (let c = 0; c < nch; c++) {
      const a = avg[c];
      let iso = 0, m = 0; for (let i = on - Math.round(0.03 * fsA); i <= on - Math.round(0.008 * fsA); i++) { if (i >= 0) { iso += a[i]; m++; } } iso /= Math.max(1, m);
      // suavizado de 28 ms
      const w = Math.max(1, Math.round(0.014 * fsA)); const sm = new Float32Array(L); for (let i = 0; i < L; i++) { let s = 0, q = 0; for (let j = Math.max(0, i - w); j <= Math.min(L - 1, i + w); j++) { s += a[j]; q++; } sm[i] = s / q - iso; }
      let tp = -1, ta = 0; for (let i = s0; i < sEnd; i++) { const v = Math.abs(sm[i]); if (v > ta) { ta = v; tp = i; } }
      if (tp < 0 || ta < 50 || tp >= sEnd - 2) continue;
      // final de T por el método del área (Zhang 2006): máximo del área entre la señal y su valor local en una ventana deslizante
      const kb = Math.min(eLim, tp + Math.round(Q.teMax * fsA)); let bestA = 0, te = -1; const sgn = sm[tp] >= 0 ? 1 : -1;
      const cum = new Float64Array(L + 1); for (let i = 0; i < L; i++) cum[i + 1] = cum[i] + sm[i];
      for (let t = tp + Math.round(0.02 * fsA); t <= kb; t++) { if (t - wA < 0 || t + pS >= L) continue; const sbar = (cum[t + pS + 1] - cum[t - pS]) / (2 * pS + 1); const ar = (cum[t + 1] - cum[t - wA + 1]) - wA * sbar; const v = Q.signed ? sgn * ar : Math.abs(ar); if (v > bestA) { bestA = v; te = t; } }
      if (te < 0 || te >= kb) continue; const q = (te - on) / fsA * 1000;
      if (q < 220 || q > 720) continue;
      cand.push({ c, ta, q, tp: (tp - on) / fsA * 1000 });
    }
    if (!cand.length) continue; cand.sort((x, y) => y.ta - x.ta); let pick = cand[0];
    if (Q.latest) for (const k of cand) if (k.ta >= Q.latest * cand[0].ta && k.q > pick.q) pick = k;
    qt[mi] = pick.q; qch[mi] = pick.c; tpk[mi] = pick.tp; qon[mi] = (on - pre) / fsA * 1000;
  }
  return { qt, rr: rrm, ch: qch, tpk, on: qon };
}

/* ------------------------------------------------------------------ análisis completo (paso 1) */
E.analyze = function (rec, opt, progress) {
  progress = progress || function () { };
  const p1 = E.pass1(rec, opt, progress); const f = E.features(p1, progress);
  progress('Clasificando latidos', 0.72);
  const c = classifyBeats(p1, f, p1.opt);
  progress('Midiendo ST y QT', 0.82);
  const st = stPerBeat(p1, f, c.cls, p1.opt); const qt = qtPerMinute(p1, f, c.cls);
  const nb = p1.fid.length, nch = p1.ana.length; const t = new Float64Array(nb), samp = new Int32Array(nb);
  for (let b = 0; b < nb; b++) { t[b] = p1.fid[b] / p1.fsA; samp[b] = Math.round(p1.fid[b] * p1.dec); }
  const templates = f.cl.T.map((T, i) => ({ id: i, count: f.cl.count[i], wave: T, width: f.ft.tW[i], pV: c.tV[i], pX: c.tX[i], on: st.bounds[i].on, off: st.bounds[i].off }));
  progress('Listo', 0.9);
  return {
    version: E.VERSION, name: rec.name, source: rec.source, fs: rec.fs, fsA: p1.fsA, dec: p1.dec, nsec: p1.nsec, dur: rec.n / rec.fs, start: rec.start,
    chans: p1.chans, chanNames: p1.chans.map(i => rec.names[i]), amps: p1.amps, qual: p1.qual, best: p1.best, nosig: p1.nosig,
    beats: { n: nb, t, samp, cls: c.cls, auto: Uint8Array.from(c.cls), conf: c.conf, pV: c.pV, pS: c.pS, pX: c.pX, tid: f.cl.tid, tcorr: f.cl.tcorr, X: f.ft.X, st: st.st, edited: new Uint8Array(nb) },
    templates, qt, nch, opt: p1.opt
  };
};

/* ------------------------------------------------------------------ fibrilación auricular: rasgos de irregularidad en ventanas de 64 intervalos */
E.AF_NF = 9;
function afWindowFeatures(x, m, out) { // x: intervalos RR (s) consecutivos válidos, m: cuántos
  if (m < 40) return false;
  let mean = 0; for (let i = 0; i < m; i++) mean += x[i]; mean /= m;
  const d = []; for (let i = 1; i < m; i++) d.push(x[i] - x[i - 1]);
  const ad = d.map(Math.abs).sort((a, b) => a - b); const keep = Math.floor(ad.length * 0.9); let s2 = 0; for (let i = 0; i < keep; i++) s2 += ad[i] * ad[i];
  out[0] = Math.sqrt(s2 / Math.max(1, keep)) / mean;
  const xs = Array.prototype.slice.call(x, 0, m).sort((a, b) => a - b); const tr = Math.floor(m / 16); const lo = xs[tr], hi = xs[m - 1 - tr]; const nbins = 16; const h = new Float32Array(nbins); let cnt = 0;
  if (hi > lo) for (let i = 0; i < m; i++) { const v = x[i]; if (v < lo || v > hi) continue; let k = Math.floor((v - lo) / (hi - lo) * nbins); if (k >= nbins) k = nbins - 1; h[k]++; cnt++; }
  let ent = 0; if (cnt) for (let k = 0; k < nbins; k++) if (h[k]) { const p = h[k] / cnt; ent -= p * Math.log(p); }
  out[1] = ent / Math.log(nbins);
  let tp = 0; for (let i = 1; i < m - 1; i++) if ((x[i] > x[i - 1] && x[i] > x[i + 1]) || (x[i] < x[i - 1] && x[i] < x[i + 1])) tp++;
  out[2] = tp / (m - 2);
  let c5 = 0; for (let i = 0; i < d.length; i++) if (Math.abs(d[i]) > 0.05 * mean) c5++; out[3] = c5 / d.length;
  const cells = new Set(); let orig = 0; for (let i = 1; i < d.length; i++) { const a = Math.round(d[i] / 0.025), b = Math.round(d[i - 1] / 0.025); cells.add(a * 1000 + b); if (a === 0 && b === 0) orig++; }
  out[4] = cells.size / (d.length - 1); out[5] = orig / (d.length - 1);
  out[6] = ad[Math.floor(ad.length / 2)] / mean; out[7] = mean;
  // entropía muestral (m = 1, r = 30 ms)
  let A = 0, B = 0; const r = 0.03; for (let i = 0; i < m - 1; i++) for (let j = i + 1; j < m - 1; j++) { if (Math.abs(x[i] - x[j]) < r) { B++; if (Math.abs(x[i + 1] - x[j + 1]) < r) A++; } }
  out[8] = -Math.log((A + 0.5) / (B + 0.5));
  return true;
}
/* devuelve rasgos por ventana centrada cada 'step' latidos; rr[b] = intervalo que termina en b (NaN si no es válido) */
E.afFeatures = function (rr, use, nb, step) {
  step = step || 8; const W = 64; const nw = Math.floor(nb / step) + 1; const X = new Float32Array(nw * E.AF_NF); const okw = new Uint8Array(nw); const buf = new Float32Array(W); const o = new Float32Array(E.AF_NF);
  for (let w = 0; w < nw; w++) {
    const c = w * step; let m = 0;
    for (let b = Math.max(1, c - W / 2); b < Math.min(nb, c + W / 2); b++) if (use[b] && rr[b] === rr[b] && rr[b] < 3 && rr[b] > 0.2) buf[m++] = rr[b];
    if (afWindowFeatures(buf, m, o)) { X.set(o, w * E.AF_NF); okw[w] = 1; }
  }
  return { X, ok: okw, step, nw };
};
function afProbability(feat) {
  const m = E._afModel; const p = new Float32Array(feat.nw);
  for (let w = 0; w < feat.nw; w++) {
    if (!feat.ok[w]) { p[w] = NaN; continue; } const o = w * E.AF_NF;
    if (m && m.gbm) { const g = m.gbm; let a = g.base[0]; for (let i = 0; i < g.IT; i++) { let nd = g.root[i]; while (!g.leaf[nd]) nd = feat.X[o + g.f[nd]] <= g.th[nd] ? g.l[nd] : g.r[nd]; a += g.v[nd]; } p[w] = 1 / (1 + Math.exp(-a)); }
    else if (m && m.coef) { let z = m.intercept; for (let k = 0; k < E.AF_NF; k++) z += m.coef[k] * (feat.X[o + k] - m.mean[k]) / m.scale[k]; p[w] = 1 / (1 + Math.exp(-z)); }
    else { // respaldo sin modelo: irregularidad alta, entropía alta y pocos pares en el origen
      const z = 30 * (feat.X[o] - 0.085) + 6 * (feat.X[o + 1] - 0.7) - 6 * (feat.X[o + 5] - 0.1); p[w] = 1 / (1 + Math.exp(-z)); }
  }
  return p;
}
E.compileAfModel = function (m) { if (m.trees) { const g = compileGbm({ classes: ['af'], trees: m.trees.map(t => [t]), base: m.base, feat: m.feat }); return { gbm: g }; } return m; };

/* ------------------------------------------------------------------ FFT (radix 2) */
function fft(re, im) {
  const n = re.length; for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
  for (let len = 2; len <= n; len <<= 1) { const ang = -2 * Math.PI / len; const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) { let cr = 1, ci = 0; for (let k = 0; k < len / 2; k++) { const a = i + k, b = a + len / 2; const xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr; re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi; const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t; } } }
}
E.fft = fft;

/* ------------------------------------------------------------------ paso 2: latidos -> ritmo, eventos, estadísticas (rápido; se repite tras cada edición) */
function clockOf(A, t) { // segundos desde medianoche del día 0
  const s = A.start ? A.start.h * 3600 + A.start.mi * 60 + A.start.s : 0; return s + t;
}
E.clockOf = clockOf;
E.fmtClock = function (A, t, withDay) {
  const c = clockOf(A, t); const day = Math.floor(c / 86400); const r = ((c % 86400) + 86400) % 86400; const h = Math.floor(r / 3600), m = Math.floor((r % 3600) / 60), s = Math.floor(r % 60);
  const p = v => (v < 10 ? '0' : '') + v; return (withDay && day > 0 ? 'D' + (day + 1) + ' ' : '') + p(h) + ':' + p(m) + ':' + p(s);
};

E.summarize = function (A, S0) {
  const S = Object.assign({}, E.DEFAULTS, S0 || {}); const B = A.beats, nb = B.n, t = B.t; const cls = B.cls;
  const nsec = A.nsec; const nmin = Math.ceil(nsec / 60), nhour = Math.ceil(nsec / 3600);
  // segundos excluidos: inservibles en todos los canales o marcados por el usuario
  const excl = new Uint8Array(nsec); for (let s = 0; s < nsec; s++) if (A.best[s] === 2) excl[s] = 1;
  if (A.exclusions) for (const e of A.exclusions) for (let s = Math.max(0, Math.floor(e[0])); s < Math.min(nsec, Math.ceil(e[1])); s++) excl[s] = 2;
  if (A.inclusions) for (const e of A.inclusions) for (let s = Math.max(0, Math.floor(e[0])); s < Math.min(nsec, Math.ceil(e[1])); s++) excl[s] = 0;
  const exPre = new Int32Array(nsec + 1); for (let s = 0; s < nsec; s++) exPre[s + 1] = exPre[s] + (excl[s] ? 1 : 0);
  // etiquetas: se parte de la clase morfológica automática salvo latidos editados
  for (let b = 0; b < nb; b++) if (!B.edited[b]) { cls[b] = B.auto[b]; if (excl[Math.min(nsec - 1, Math.floor(t[b]))] === 2) cls[b] = 4; }
  const rr = new Float32Array(nb).fill(NaN), prevIdx = new Int32Array(nb).fill(-1);
  let lastB = -1, sawX = false;
  for (let b = 0; b < nb; b++) {
    if (cls[b] === 4) { sawX = true; continue; }
    if (lastB >= 0) {
      const s0 = Math.floor(t[lastB]), s1 = Math.min(nsec - 1, Math.floor(t[b])); const ex = exPre[s1 + 1] - exPre[s0];
      prevIdx[b] = lastB; if (!ex && !sawX) rr[b] = t[b] - t[lastB];
    }
    lastB = b; sawX = false;
  }
  const supra = c => c === 0 || c === 1 || c === 6 || c === 7; const vent = E.isVentricular;
  // RR de referencia: mediana de hasta 8 intervalos válidos previos entre latidos de conducción normal
  const rrRef = new Float32Array(nb).fill(NaN); { const q = []; for (let b = 0; b < nb; b++) { if (q.length >= 3) rrRef[b] = median(q); if (rr[b] === rr[b] && supra(cls[b]) && prevIdx[b] >= 0 && supra(cls[prevIdx[b]]) && rr[b] < 2.5) { q.push(rr[b]); if (q.length > 8) q.shift(); } } }
  // fibrilación auricular
  const afBeat = new Uint8Array(nb); const afP = new Float32Array(nb).fill(NaN); let afEpisodes = [];
  if (S.afDetect) {
    const use = new Uint8Array(nb); for (let b = 1; b < nb; b++) use[b] = (rr[b] === rr[b] && !vent(cls[b]) && prevIdx[b] >= 0 && !vent(cls[prevIdx[b]])) ? 1 : 0;
    const ft = E.afFeatures(rr, use, nb, 8); const pw = afProbability(ft);
    for (let b = 0; b < nb; b++) { const w = b / 8; const w0 = Math.floor(w), w1 = Math.min(ft.nw - 1, w0 + 1); const a = pw[w0], c = pw[w1]; afP[b] = a === a ? (c === c ? a + (c - a) * (w - w0) : a) : c; }
    // histéresis (entra a 0,6 y sale a 0,4) y duración mínima de 30 s
    let on = false, st = -1; const thOn = S.afOn || 0.6, thOff = S.afOff || 0.4;
    for (let b = 0; b <= nb; b++) {
      const p = b < nb ? afP[b] : 0; const gap = b < nb && b > 0 && (t[b] - t[b - 1] > 10);
      if (!on && p >= thOn && !gap) { on = true; st = b; }
      else if (on && (!(p >= thOff) || gap || b === nb)) { on = false; const e = b - 1; if (t[e] - t[st] >= 30) afEpisodes.push([st, e]); }
    }
    for (const ep of afEpisodes) for (let b = ep[0]; b <= ep[1]; b++) afBeat[b] = 1;
  }
  if (A.afManual) for (const e of A.afManual) { let b0 = lowerBound(t, e[0]), b1 = lowerBound(t, e[1]) - 1; if (b1 > b0) { if (e[2] === 0) { for (let b = b0; b <= b1; b++) afBeat[b] = 0; } else { for (let b = b0; b <= b1; b++) afBeat[b] = 1; } } }
  if (A.afManual && A.afManual.length) { afEpisodes = []; let st = -1; for (let b = 0; b <= nb; b++) { const v = b < nb && afBeat[b]; if (v && st < 0) st = b; else if (!v && st >= 0) { afEpisodes.push([st, b - 1]); st = -1; } } }
  // extrasístoles supraventriculares: latido de morfología normal adelantado, fuera de fibrilación auricular
  const prem = new Float32Array(nb).fill(NaN); let runRef = NaN;
  for (let b = 0; b < nb; b++) {
    if (rr[b] === rr[b] && rrRef[b] === rrRef[b]) prem[b] = 1 - rr[b] / rrRef[b];
    if (A.extClasses || B.edited[b] || cls[b] !== 0) { if (cls[b] !== 1) runRef = NaN; continue; } // con otro motor sus S se respetan
    if (afBeat[b]) { runRef = NaN; continue; }
    const pb = prevIdx[b]; if (pb < 0 || vent(cls[pb])) { runRef = NaN; continue; }
    if (prem[b] >= S.svPrematurity) { cls[b] = 1; if (!(runRef === runRef)) runRef = rrRef[b]; }
    // continuación de una racha supraventricular: se compara con el RR previo al inicio, no con la referencia móvil
    else if (runRef === runRef && cls[pb] === 1 && rr[b] === rr[b] && rr[b] <= (1 - 0.6 * S.svPrematurity) * runRef && rr[b] <= 60 / S.svtBpm) cls[b] = 1;
    else runRef = NaN;
  }
  // ---- eventos
  const ev = []; const add = (type, b0, b1, extra) => { const e = Object.assign({ type, b0, b1, t0: t[b0], t1: t[b1], n: b1 - b0 + 1 }, extra || {}); ev.push(e); return e; };
  const seq = []; for (let b = 0; b < nb; b++) if (cls[b] !== 4) seq.push(b);
  const ns = seq.length; const kind = new Uint8Array(ns); // 0 normal, 1 SV, 2 V
  for (let i = 0; i < ns; i++) { const c = cls[seq[i]]; kind[i] = vent(c) ? 2 : c === 1 ? 1 : 0; }
  const cnt = { vIso: 0, vPair: 0, vRun: 0, vRunBeats: 0, sIso: 0, sPair: 0, sRun: 0, sRunBeats: 0, vBig: 0, vTrig: 0, sBig: 0, sTrig: 0 };
  const role = new Uint8Array(nb); // 1 aislada, 2 par, 3 racha
  for (let i = 0; i < ns;) {
    const k = kind[i]; if (!k) { i++; continue; }
    let j = i; while (j + 1 < ns && kind[j + 1] === k && rr[seq[j + 1]] === rr[seq[j + 1]]) j++;
    const len = j - i + 1; const b0 = seq[i], b1 = seq[j];
    for (let q = i; q <= j; q++) role[seq[q]] = len === 1 ? 1 : len === 2 ? 2 : 3;
    if (len === 1) { if (k === 2) cnt.vIso++; else cnt.sIso++; }
    else if (len === 2) { if (k === 2) { cnt.vPair++; add('vPair', b0, b1); } else { cnt.sPair++; add('sPair', b0, b1); } }
    else {
      const dur = t[b1] - t[b0]; const rate = dur > 0 ? 60 * (len - 1) / dur : 0;
      if (k === 2) { cnt.vRun++; cnt.vRunBeats += len; const ty = (rate >= S.vtBpm && len >= S.vtBeats) ? 'vt' : rate < 50 ? 'ivr' : 'aivr'; add(ty, b0, b1, { rate, dur, sustained: dur >= 30 }); }
      else { cnt.sRun++; cnt.sRunBeats += len; add((rate >= S.svtBpm && len >= S.svtBeats) ? 'svt' : 'sRun', b0, b1, { rate, dur }); }
    }
    i = j + 1;
  }
  // bigeminismo y trigeminismo (al menos 3 ciclos)
  const pattern = (k, period, name, key) => {
    let i = 0; while (i < ns) {
      if (kind[i] !== k) { i++; continue; }
      let cyc = 1, j = i; 
      for (;;) { const nx = j + period; if (nx >= ns || kind[nx] !== k) break; let okp = true; for (let q = j + 1; q < nx; q++) if (kind[q] !== 0) { okp = false; break; } if (!okp) break; cyc++; j = nx; }
      let pre = true; for (let q = i - period + 1; q < i; q++) if (q < 0 || kind[q] !== 0) pre = false;
      if (cyc >= 3 && period > 1) { const b0 = seq[Math.max(0, i - (pre ? period - 1 : 0))], b1 = seq[j]; add(name, b0, b1, { cycles: cyc }); cnt[key] += cyc; i = j + 1; } else i++;
    }
  };
  pattern(2, 2, 'vBig', 'vBig'); pattern(2, 3, 'vTrig', 'vTrig'); pattern(1, 2, 'sBig', 'sBig'); pattern(1, 3, 'sTrig', 'sTrig');
  // R sobre T e interpoladas
  let rOnT = 0, interp = 0;
  for (let i = 1; i < ns - 1; i++) {
    if (kind[i] !== 2 || kind[i - 1] === 2) continue; const b = seq[i]; const r = rr[b], ref = rrRef[b]; if (!(r === r) || !(ref === ref)) continue;
    const qtEst = 0.39 * Math.sqrt(ref); // QT esperado (Bazett con QTc de 390 ms)
    if (r < 0.85 * qtEst + 0.04) { rOnT++; add('rOnT', b, b, { coupling: r }); }
    const rn = rr[seq[i + 1]]; if (rn === rn && kind[i + 1] === 0 && Math.abs((r + rn) - ref) < 0.15 * ref) { interp++; add('interp', b, b); }
  }
  // pausas y RR más largo
  let rrMax = 0, rrMaxB = -1, rrMin = 99, rrMinB = -1; const pauses = [];
  for (let b = 0; b < nb; b++) {
    const r = rr[b]; if (!(r === r)) continue;
    if (r > rrMax) { rrMax = r; rrMaxB = b; } if (r < rrMin && supra(cls[b])) { rrMin = r; rrMinB = b; }
    if (r >= S.pauseSec) { const pb = prevIdx[b]; const after = vent(cls[pb]) ? 'posextrasistólica ventricular' : cls[pb] === 1 ? 'posextrasistólica supraventricular' : afBeat[b] ? 'en fibrilación auricular' : ''; const ref = rrRef[pb]; const dbl = ref === ref && r > 1.8 * ref && r < 2.25 * ref; pauses.push(add('pause', pb, b, { rr: r, after, dbl })); }
  }
  // frecuencia cardíaca: promedio de 8 intervalos; mínimos, máximos, bradicardia y taquicardia
  const hr8 = new Float32Array(nb).fill(NaN); { const q = []; let sum = 0; for (let b = 0; b < nb; b++) { const r = rr[b]; if (cls[b] === 4) continue; if (!(r === r)) { q.length = 0; sum = 0; continue; } q.push(r); sum += r; if (q.length > 8) sum -= q.shift(); if (q.length === 8) hr8[b] = 60 * 8 / sum; } }
  let hrMin = 999, hrMinB = -1, hrMax = 0, hrMaxB = -1; for (let b = 0; b < nb; b++) { const h = hr8[b]; if (!(h === h)) continue; if (h < hrMin) { hrMin = h; hrMinB = b; } if (h > hrMax) { hrMax = h; hrMaxB = b; } }
  const rateEpisodes = (test, name) => { let st = -1, ext = 0; for (let b = 0; b <= nb; b++) { const h = b < nb ? hr8[b] : NaN; if (b < nb && cls[b] === 4) continue; const inside = h === h && test(h);
      if (inside) { if (st < 0) { st = b; ext = h; } else if (name === 'tachy' ? h > ext : h < ext) ext = h; }
      else if (st >= 0) { let e = b - 1; while (e > st && cls[e] === 4) e--; if (t[e] - t[st] >= S.minRateDurSec) add(name, st, e, { rate: ext, dur: t[e] - t[st] }); st = -1; } } };
  rateEpisodes(h => h > S.tachyBpm, 'tachy'); rateEpisodes(h => h < S.bradyBpm, 'brady');
  for (const ep of afEpisodes) { const dur = t[ep[1]] - t[ep[0]]; let s = 0, m = 0; for (let b = ep[0] + 1; b <= ep[1]; b++) if (rr[b] === rr[b]) { s += rr[b]; m++; } add('af', ep[0], ep[1], { dur, rate: m ? 60 * m / s : NaN }); }
  // segmentos excluidos
  { let st = -1, ns2 = 0; for (let s = 0; s <= nsec; s++) { const e = s < nsec && excl[s]; if (e && st < 0) { st = s; ns2 = 0; } if (e && A.nosig && A.nosig[s]) ns2++; if (!e && st >= 0) { ev.push({ type: ns2 >= 0.6 * (s - st) ? 'nosig' : 'artifact', t0: st, t1: s, b0: lowerBound(t, st), b1: Math.max(0, lowerBound(t, s) - 1), n: 0, dur: s - st }); st = -1; } } }
  // ---- por minuto y por hora
  const M = { hr: new Float32Array(nmin).fill(NaN), hrMin: new Float32Array(nmin).fill(NaN), hrMax: new Float32Array(nmin).fill(NaN), beats: new Uint16Array(nmin), v: new Uint16Array(nmin), s: new Uint16Array(nmin), af: new Float32Array(nmin), excl: new Float32Array(nmin), q1: new Float32Array(nmin), st: [], rmssd: new Float32Array(nmin).fill(NaN), sdnn: new Float32Array(nmin).fill(NaN) };
  const mRR = new Float64Array(nmin), mN = new Uint16Array(nmin), mAf = new Uint16Array(nmin);
  for (let b = 0; b < nb; b++) {
    if (cls[b] === 4) continue; const mi = Math.min(nmin - 1, Math.floor(t[b] / 60)); M.beats[mi]++; if (vent(cls[b])) M.v[mi]++; else if (cls[b] === 1) M.s[mi]++; if (afBeat[b]) mAf[mi]++;
    const r = rr[b]; if (r === r) { mRR[mi] += r; mN[mi]++; } const h = hr8[b]; if (h === h) { if (!(M.hrMin[mi] <= h)) M.hrMin[mi] = h; if (!(M.hrMax[mi] >= h)) M.hrMax[mi] = h; }
  }
  for (let mi = 0; mi < nmin; mi++) { if (mN[mi] >= 5) M.hr[mi] = 60 * mN[mi] / mRR[mi]; M.af[mi] = M.beats[mi] ? mAf[mi] / M.beats[mi] : 0; let e = 0, q = 0, c = 0; for (let s = mi * 60; s < Math.min(nsec, mi * 60 + 60); s++) { if (excl[s]) e++; if (A.best[s] === 1) q++; c++; } M.excl[mi] = c ? e / c : 0; M.q1[mi] = c ? q / c : 0; }
  // ST por minuto: mediana de los latidos normales con buena señal, y referencia por canal (mediana del registro)
  const stRef = []; const stEpisodes = [];
  for (let c = 0; c < A.nch; c++) {
    const arr = new Float32Array(nmin).fill(NaN); const bucket = []; let mi = 0; const src = B.st[c];
    for (let b = 0; b <= nb; b++) {
      const m2 = b < nb ? Math.min(nmin - 1, Math.floor(t[b] / 60)) : nmin;
      if (m2 !== mi) { if (bucket.length >= 10) arr[mi] = median(bucket); bucket.length = 0; mi = m2; }
      if (b < nb && cls[b] === 0 && src[b] > -30000) bucket.push(src[b]);
    }
    M.st.push(arr); const valid = []; for (let i = 0; i < nmin; i++) if (arr[i] === arr[i]) valid.push(arr[i]); const ref = valid.length ? median(valid) : 0; stRef.push(ref);
    let st0 = -1, ext = 0; for (let i = 0; i <= nmin; i++) { const v = i < nmin ? arr[i] - ref : NaN; const dev = v === v && Math.abs(v) >= S.stThreshUv;
      if (dev) { if (st0 < 0) { st0 = i; ext = v; } else if (Math.abs(v) > Math.abs(ext)) ext = v; }
      else if (st0 >= 0 && !(i < nmin && !(v === v) && i - st0 < 3)) { if (i - st0 >= 1) { const e = { type: ext < 0 ? 'stDep' : 'stEle', ch: c, t0: st0 * 60, t1: i * 60, b0: lowerBound(t, st0 * 60), b1: Math.max(0, lowerBound(t, i * 60) - 1), n: 0, dur: (i - st0) * 60, level: ext }; ev.push(e); stEpisodes.push(e); } st0 = -1; } }
  }
  // VFC: intervalos NN
  const nnOk = new Uint8Array(nb); const hrvCls = S.hrvNormalOnly ? (c => c === 0 || c === 6) : supra;
  for (let b = 0; b < nb; b++) if (rr[b] === rr[b] && hrvCls(cls[b]) && prevIdx[b] >= 0 && hrvCls(cls[prevIdx[b]]) && !afBeat[b] && rr[b] > 0.3 && rr[b] < 2.5) nnOk[b] = 1;
  const hrv = E.hrv(A, t, rr, nnOk, cls, prevIdx, M);
  // perfil por hora
  const hours = []; const hIdx = b => Math.min(nhour - 1, Math.floor(t[b] / 3600));
  for (let h = 0; h < nhour; h++) hours.push({ h, t0: h * 3600, min: 0, beats: 0, hrMin: NaN, hrMean: NaN, hrMax: NaN, rrMax: 0, pauses: 0, vTot: 0, vIso: 0, vPair: 0, vRun: 0, vBig: 0, vTrig: 0, sTot: 0, sIso: 0, sPair: 0, sRun: 0, afPct: 0, exclPct: 0, sdnn: NaN, rmssd: NaN, pnn50: NaN, stMin: [], stMax: [], qtc: NaN, fus: 0, x: 0 });
  { const sr = new Float64Array(nhour), nr = new Uint32Array(nhour), afc = new Uint32Array(nhour);
    for (let b = 0; b < nb; b++) { const H = hours[hIdx(b)]; if (cls[b] === 4) { H.x++; continue; } H.beats++; const c = cls[b];
      if (vent(c)) { H.vTot++; if (role[b] === 1) H.vIso++; if (c === 3) H.fus++; } else if (c === 1) { H.sTot++; if (role[b] === 1) H.sIso++; }
      if (afBeat[b]) afc[hIdx(b)]++; const r = rr[b]; if (r === r) { sr[hIdx(b)] += r; nr[hIdx(b)]++; if (r > H.rrMax) H.rrMax = r; } const hh = hr8[b]; if (hh === hh) { if (!(H.hrMin <= hh)) H.hrMin = hh; if (!(H.hrMax >= hh)) H.hrMax = hh; } }
    for (let h = 0; h < nhour; h++) { const H = hours[h]; if (nr[h] > 10) H.hrMean = 60 * nr[h] / sr[h]; H.afPct = H.beats ? 100 * afc[h] / H.beats : 0; let e = 0, c = 0; for (let s = h * 3600; s < Math.min(nsec, h * 3600 + 3600); s++) { if (excl[s]) e++; c++; } H.exclPct = c ? 100 * e / c : 0; H.min = (c - e) / 60;
      for (let ch = 0; ch < A.nch; ch++) { let mn = NaN, mx = NaN; for (let mi = h * 60; mi < Math.min(nmin, h * 60 + 60); mi++) { const v = M.st[ch][mi]; if (v === v) { if (!(mn <= v)) mn = v; if (!(mx >= v)) mx = v; } } H.stMin.push(mn); H.stMax.push(mx); }
      const hh = hrv.hourly[h]; if (hh) { H.sdnn = hh.sdnn; H.rmssd = hh.rmssd; H.pnn50 = hh.pnn50; } }
    for (const e of ev) { const H = hours[Math.min(nhour - 1, Math.floor(e.t0 / 3600))]; if (e.type === 'pause') H.pauses++; else if (e.type === 'vPair') H.vPair++; else if (e.type === 'vt' || e.type === 'aivr' || e.type === 'ivr') H.vRun++; else if (e.type === 'sPair') H.sPair++; else if (e.type === 'svt' || e.type === 'sRun') H.sRun++; else if (e.type === 'vBig') H.vBig += e.cycles; else if (e.type === 'vTrig') H.vTrig += e.cycles; } }
  // QT
  const qtc = new Float32Array(nmin).fill(NaN); const qf = S.qtcFormula; let qn = 0, qs = 0, qmin = 1e9, qmax = 0, qmaxM = -1, q450 = 0, q480 = 0, q500 = 0, qtS = 0, qskip = 0;
  if (A.qt) for (let mi = 0; mi < nmin; mi++) { const q = A.qt.qt[mi], r = A.qt.rr[mi]; if (!(q === q) || !(r > 0)) continue; if (M.af[mi] >= 0.5 || r < 0.5) { qskip++; continue; } let c; // en FA o sobre 120 lpm la corrección no es fiable
    if (qf === 'fridericia') c = q / Math.cbrt(r); else if (qf === 'framingham') c = q + 154 * (1 - r); else if (qf === 'hodges') c = q + 1.75 * (60 / r - 60); else c = q / Math.sqrt(r);
    qtc[mi] = c; qn++; qs += c; qtS += q; if (c < qmin) qmin = c; if (c > qmax) { qmax = c; qmaxM = mi; } if (c > 450) q450++; if (c > 480) q480++; if (c > 500) q500++; }
  for (let h = 0; h < nhour; h++) { const a = []; for (let mi = h * 60; mi < Math.min(nmin, h * 60 + 60); mi++) if (qtc[mi] === qtc[mi]) a.push(qtc[mi]); if (a.length >= 5) hours[h].qtc = median(a); }
  // totales
  const counts = new Uint32Array(10); for (let b = 0; b < nb; b++) counts[cls[b]]++;
  const total = nb - counts[4]; const vTot = counts[2] + counts[3] + counts[8], sTot = counts[1];
  let exSec = 0; for (let s = 0; s < nsec; s++) if (excl[s]) exSec++;
  let sumRR = 0, nRR = 0; for (let b = 0; b < nb; b++) if (rr[b] === rr[b]) { sumRR += rr[b]; nRR++; }
  let afSec = 0, afLong = 0; for (const e of ev) if (e.type === 'af') { afSec += e.dur; if (e.dur > afLong) afLong = e.dur; }
  // día y noche (23:00 a 07:00) si se conoce la hora de inicio
  let dayS = 0, dayN = 0, nightS = 0, nightN = 0; if (A.start) for (let mi = 0; mi < nmin; mi++) { const h = M.hr[mi]; if (!(h === h)) continue; const hod = Math.floor((clockOf(A, mi * 60) % 86400) / 3600); if (hod >= 23 || hod < 7) { nightS += h; nightN++; } else { dayS += h; dayN++; } }
  ev.sort((a, b) => a.t0 - b.t0);
  const analyzedSec = nsec - exSec;
  return {
    settings: S, rr, prem, rrRef, prevIdx, afBeat, afP, hr8, role, excl, minute: M, qtc, hours, events: ev, hrv, stRef, stEpisodes,
    counts, total, vTot, sTot, cnt, rOnT, interp,
    vBurden: total ? 100 * vTot / total : 0, sBurden: total ? 100 * sTot / total : 0,
    analyzedSec, exSec, exPct: nsec ? 100 * exSec / nsec : 0,
    hr: { mean: nRR ? 60 * nRR / sumRR : NaN, min: hrMinB >= 0 ? hrMin : NaN, minB: hrMinB, max: hrMaxB >= 0 ? hrMax : NaN, maxB: hrMaxB, day: dayN ? dayS / dayN : NaN, night: nightN ? nightS / nightN : NaN },
    rrMax, rrMaxB, rrMin: rrMinB >= 0 ? rrMin : NaN, rrMinB, pauses: pauses.length,
    af: { sec: afSec, pct: analyzedSec ? 100 * afSec / analyzedSec : 0, episodes: afEpisodes.length, longest: afLong },
    qt: { n: qn, mean: qn ? qs / qn : NaN, min: qn ? qmin : NaN, max: qn ? qmax : NaN, maxMin: qmaxM, over450: qn ? 100 * q450 / qn : 0, over480: qn ? 100 * q480 / qn : 0, over500: qn ? 100 * q500 / qn : 0, qtMean: qn ? qtS / qn : NaN, skipped: qskip }
  };
};
function lowerBound(arr, v) { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < v) lo = m + 1; else hi = m; } return lo; }
E.lowerBound = lowerBound;

/* ------------------------------------------------------------------ variabilidad de la frecuencia cardíaca */
E.hrv = function (A, t, rr, nnOk, cls, prevIdx, M) {
  const nb = t.length; const nsec = A.nsec; const n5 = Math.ceil(nsec / 300), nhour = Math.ceil(nsec / 3600);
  let n = 0, s = 0, s2 = 0; for (let b = 0; b < nb; b++) if (nnOk[b]) { n++; s += rr[b]; s2 += rr[b] * rr[b]; }
  const out = { n, hourly: [], seg: [], spec: null };
  if (n < 50) { out.insufficient = true; for (let h = 0; h < nhour; h++) out.hourly.push(null); return out; }
  const mean = s / n; out.meanNN = mean * 1000; out.sdnn = Math.sqrt(Math.max(0, s2 / n - mean * mean)) * 1000;
  // diferencias sucesivas entre NN contiguos
  let nd = 0, sd2 = 0, n50 = 0; const hd = new Float64Array(nhour), hn = new Uint32Array(nhour), h50 = new Uint32Array(nhour), hs = new Float64Array(nhour), hs2 = new Float64Array(nhour), hc = new Uint32Array(nhour);
  const md = new Float64Array(M.rmssd.length), mn = new Uint16Array(M.rmssd.length), ms = new Float64Array(M.rmssd.length), ms2 = new Float64Array(M.rmssd.length), mc = new Uint16Array(M.rmssd.length);
  for (let b = 1; b < nb; b++) {
    if (!nnOk[b]) continue; const h = Math.min(nhour - 1, Math.floor(t[b] / 3600)), mi = Math.min(mn.length - 1, Math.floor(t[b] / 60)); hs[h] += rr[b]; hs2[h] += rr[b] * rr[b]; hc[h]++; ms[mi] += rr[b]; ms2[mi] += rr[b] * rr[b]; mc[mi]++;
    const p = prevIdx[b]; if (p < 0 || !nnOk[p]) continue; const d = rr[b] - rr[p]; nd++; sd2 += d * d; hd[h] += d * d; hn[h]++; md[mi] += d * d; mn[mi]++; if (Math.abs(d) > 0.05) { n50++; h50[h]++; }
  }
  out.rmssd = nd ? Math.sqrt(sd2 / nd) * 1000 : NaN; out.pnn50 = nd ? 100 * n50 / nd : NaN;
  for (let h = 0; h < nhour; h++) out.hourly.push(hc[h] >= 100 ? { sdnn: Math.sqrt(Math.max(0, hs2[h] / hc[h] - Math.pow(hs[h] / hc[h], 2))) * 1000, rmssd: hn[h] ? Math.sqrt(hd[h] / hn[h]) * 1000 : NaN, pnn50: hn[h] ? 100 * h50[h] / hn[h] : NaN } : null);
  for (let mi = 0; mi < mn.length; mi++) { if (mn[mi] >= 20) M.rmssd[mi] = Math.sqrt(md[mi] / mn[mi]) * 1000; if (mc[mi] >= 20) M.sdnn[mi] = Math.sqrt(Math.max(0, ms2[mi] / mc[mi] - Math.pow(ms[mi] / mc[mi], 2))) * 1000; }
  // segmentos de 5 min: SDANN, índice SDNN y espectro
  const segMean = [], segSd = []; const NFFT = 1024, FS = 4; const re = new Float64Array(NFFT), im = new Float64Array(NFFT); const nbin = 80; // 0 a 0,3125 Hz... se guardan 0 a 0,5 Hz en 128 puntos
  const specRows = []; let vlf = 0, lf = 0, hf = 0, nspec = 0; const avgSpec = new Float64Array(NFFT / 2);
  let b = 0;
  for (let g = 0; g < n5; g++) {
    const t0 = g * 300, t1 = t0 + 300; const xs = [], ts = []; while (b < nb && t[b] < t1) { if (nnOk[b]) { xs.push(rr[b]); ts.push(t[b]); } b++; }
    let row = null; const seg = { t0, n: xs.length, mean: NaN, sd: NaN, lf: NaN, hf: NaN, vlf: NaN };
    if (xs.length >= 150) {
      let m = 0; for (const v of xs) m += v; m /= xs.length; let v2 = 0; for (const v of xs) v2 += (v - m) * (v - m); const sd = Math.sqrt(v2 / xs.length); segMean.push(m); segSd.push(sd); seg.mean = m * 1000; seg.sd = sd * 1000;
      let cover = 0; for (const v of xs) cover += v;
      if (cover >= 0.8 * 300) {
        // remuestreo a 4 Hz por interpolación lineal, sin tendencia, ventana de Hann
        let j = 0; const y = new Float64Array(NFFT); const L = Math.min(NFFT, 256 * FS); let sm = 0;
        for (let i = 0; i < L; i++) { const tt = t0 + 22 + i / FS; while (j + 1 < ts.length && ts[j + 1] < tt) j++; const a = ts[j], c = ts[Math.min(ts.length - 1, j + 1)]; const fr = c > a ? clamp((tt - a) / (c - a), 0, 1) : 0; y[i] = (xs[j] * (1 - fr) + xs[Math.min(xs.length - 1, j + 1)] * fr) * 1000; sm += y[i]; }
        sm /= L; let sxy = 0, sxx = 0; for (let i = 0; i < L; i++) { const xx = i - (L - 1) / 2; sxy += xx * (y[i] - sm); sxx += xx * xx; } const slope = sxy / sxx; let wn = 0;
        for (let i = 0; i < NFFT; i++) { if (i < L) { const w = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (L - 1)); re[i] = (y[i] - sm - slope * (i - (L - 1) / 2)) * w; wn += w * w; } else re[i] = 0; im[i] = 0; }
        fft(re, im); const df = FS / NFFT; let a = 0, bb = 0, c = 0; row = new Float32Array(64);
        for (let k = 1; k < NFFT / 2; k++) { const p = 2 * (re[k] * re[k] + im[k] * im[k]) / (FS * wn); const f = k * df; avgSpec[k] += p; if (f >= 0.003 && f < 0.04) a += p * df; else if (f < 0.15) bb += p * df; else if (f < 0.4) c += p * df; if (f < 0.5) row[Math.min(63, Math.floor(f / 0.5 * 64))] += p * df; }
        seg.vlf = a; seg.lf = bb; seg.hf = c; vlf += a; lf += bb; hf += c; nspec++;
      }
    }
    out.seg.push(seg); specRows.push(row);
  }
  if (segMean.length >= 2) { let m = 0; for (const v of segMean) m += v; m /= segMean.length; let v2 = 0; for (const v of segMean) v2 += (v - m) * (v - m); out.sdann = Math.sqrt(v2 / segMean.length) * 1000; let q = 0; for (const v of segSd) q += v; out.sdnnIndex = q / segSd.length * 1000; }
  if (nspec) { out.vlf = vlf / nspec; out.lf = lf / nspec; out.hf = hf / nspec; out.lfhf = out.hf > 0 ? out.lf / out.hf : NaN; out.lfnu = 100 * out.lf / (out.lf + out.hf); out.hfnu = 100 * out.hf / (out.lf + out.hf); out.nspec = nspec; const sp = new Float32Array(NFFT / 2); for (let k = 0; k < NFFT / 2; k++) sp[k] = avgSpec[k] / nspec; out.spec = { psd: sp, df: FS / NFFT }; }
  out.specRows = specRows;
  // índice triangular y TINN (clases de 1/128 s)
  { const bw = 1 / 128; const hist = new Map(); let mx = 0, mk = 0; for (let b2 = 0; b2 < nb; b2++) if (nnOk[b2]) { const k = Math.floor(rr[b2] / bw); const v = (hist.get(k) || 0) + 1; hist.set(k, v); if (v > mx) { mx = v; mk = k; } }
    out.tri = mx ? n / mx : NaN; let kmin = mk, kmax = mk; hist.forEach((v, k) => { if (v >= 0.02 * mx) { if (k < kmin) kmin = k; if (k > kmax) kmax = k; } }); out.tinn = (kmax - kmin + 1) * bw * 1000; }
  // Poincaré
  { const sdsd2 = nd ? (sd2 / nd) : 0; out.sd1 = Math.sqrt(0.5 * sdsd2) * 1000; const v = 2 * Math.pow(out.sdnn / 1000, 2) - 0.5 * sdsd2; out.sd2 = v > 0 ? Math.sqrt(v) * 1000 : NaN; }
  // capacidad de desaceleración y aceleración (PRSA, Bauer 2006)
  { let d0 = 0, d1 = 0, dm1 = 0, dm2 = 0, dn = 0, a0 = 0, a1 = 0, am1 = 0, am2 = 0, an = 0;
    for (let b2 = 2; b2 < nb - 1; b2++) { if (!nnOk[b2] || !nnOk[b2 + 1]) continue; const p = prevIdx[b2]; if (p < 0 || !nnOk[p]) continue; const pp = prevIdx[p]; if (pp < 0 || !nnOk[pp] || prevIdx[b2 + 1] !== b2) continue;
      const r0 = rr[b2], rm1 = rr[p]; const ch = r0 / rm1; if (ch > 1 && ch <= 1.05) { d0 += r0; d1 += rr[b2 + 1]; dm1 += rm1; dm2 += rr[pp]; dn++; } else if (ch < 1 && ch >= 0.95) { a0 += r0; a1 += rr[b2 + 1]; am1 += rm1; am2 += rr[pp]; an++; } }
    out.dc = dn > 100 ? (d0 + d1 - dm1 - dm2) / dn / 4 * 1000 : NaN; out.ac = an > 100 ? (a0 + a1 - am1 - am2) / an / 4 * 1000 : NaN; }
  // DFA alfa 1 (4 a 16 latidos) y alfa 2 (16 a 64) sobre hasta 30 000 NN
  { const x = []; for (let b2 = 0; b2 < nb && x.length < 30000; b2++) if (nnOk[b2]) x.push(rr[b2]); const N = x.length;
    if (N >= 500) { let m = 0; for (const v of x) m += v; m /= N; const y = new Float64Array(N); let acc = 0; for (let i = 0; i < N; i++) { acc += x[i] - m; y[i] = acc; }
      const F = nbox => { let tot = 0, c = 0; for (let st = 0; st + nbox <= N; st += nbox) { let sx = 0, sy = 0, sxx = 0, sxy = 0; for (let i = 0; i < nbox; i++) { sx += i; sy += y[st + i]; sxx += i * i; sxy += i * y[st + i]; } const sl = (nbox * sxy - sx * sy) / (nbox * sxx - sx * sx), ic = (sy - sl * sx) / nbox; for (let i = 0; i < nbox; i++) { const e = y[st + i] - (sl * i + ic); tot += e * e; } c += nbox; } return Math.sqrt(tot / c); };
      const fit = sizes => { let sx = 0, sy = 0, sxx = 0, sxy = 0; for (const z of sizes) { const lx = Math.log(z), ly = Math.log(F(z)); sx += lx; sy += ly; sxx += lx * lx; sxy += lx * ly; } const k = sizes.length; return (k * sxy - sx * sy) / (k * sxx - sx * sx); };
      out.alpha1 = fit([4, 5, 6, 7, 8, 10, 12, 14, 16]); out.alpha2 = fit([16, 20, 24, 32, 40, 48, 64]); } }
  // turbulencia de la frecuencia cardíaca (Schmidt 1999): extrasístoles ventriculares aisladas con 2 NN antes y 15 después
  { const pre = 2, post = 15; const acc = new Float64Array(pre + post + 2); let k = 0, toSum = 0;
    for (let b2 = 3; b2 < nb - post - 2; b2++) { if (!E.isVentricular(cls[b2])) continue; let ok = true; for (let i = 1; i <= pre; i++) if (!nnOk[b2 - i]) ok = false; if (!nnOk[b2 - 1 - pre + 1]) ok = false;
      for (let i = 2; i <= post + 1 && ok; i++) if (!nnOk[b2 + i]) ok = false; if (!ok || cls[b2 + 1] === 4) continue; const cp = rr[b2], comp = rr[b2 + 1], ref = (rr[b2 - 1] + rr[b2 - 2]) / 2; if (!(cp === cp) || !(comp === comp)) continue; if (!(cp < 0.8 * ref && comp > 1.2 * ref)) continue;
      const to = ((rr[b2 + 2] + rr[b2 + 3]) - (rr[b2 - 2] + rr[b2 - 1])) / (rr[b2 - 2] + rr[b2 - 1]) * 100; toSum += to; k++; acc[0] += rr[b2 - 2]; acc[1] += rr[b2 - 1]; acc[2] += cp; acc[3] += comp; for (let i = 0; i < post; i++) acc[4 + i] += rr[b2 + 2 + i]; }
    if (k >= 5) { const av = Array.from(acc, v => v / k * 1000); let ts = -1e9; for (let i = 0; i + 5 <= post; i++) { let sx = 0, sy = 0, sxx = 0, sxy = 0; for (let j = 0; j < 5; j++) { sx += j; sy += av[4 + i + j]; sxx += j * j; sxy += j * av[4 + i + j]; } const sl = (5 * sxy - sx * sy) / (5 * sxx - sx * sx); if (sl > ts) ts = sl; } out.hrt = { n: k, to: toSum / k, ts, tach: av, cat: (toSum / k >= 0 ? 1 : 0) + (ts <= 2.5 ? 1 : 0) }; } else out.hrt = { n: k }; }
  return out;
};

return E;
});
