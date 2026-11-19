/* Prueba del motor: lee un registro WFDB, lo analiza y comprueba latidos contra sus anotaciones. */
const path = require('path'); const { spawn } = require('child_process');
const hea = path.resolve(process.argv[2]); const out = path.join(require('os').tmpdir(), 'hl_prueba');
const p = spawn(process.execPath, [path.join(__dirname, '..', 'holterlocal', 'motor_js', 'servidor.js')]);
let buf = ''; const t0 = Date.now();
p.stdout.on('data', d => { buf += d; let i; while ((i = buf.indexOf('\n')) >= 0) { const line = buf.slice(0, i); buf = buf.slice(i + 1); const m = JSON.parse(line);
  if (m.type === 'ready') p.stdin.write(JSON.stringify({ cmd: 'open', id: 1, paths: [hea], out }) + '\n');
  if (m.type === 'result') { if (m.error) { console.error('ERROR', m.error); process.exit(1); }
    const A = m.A; console.log(`${A.total} latidos, V ${A.vTot}, S ${A.sTot}, FA ${A.af.pct.toFixed(1)} %, ${A.events.length} eventos en ${((Date.now() - t0) / 1000).toFixed(1)} s`);
    if (!(A.total > 1000) || !m.ref || Math.abs(A.total - m.ref.t.length) / m.ref.t.length > 0.05) { console.error('Conteo de latidos fuera de lo esperado'); process.exit(1); }
    p.kill(); process.exit(0); } } });
p.stderr.on('data', d => process.stderr.write(d));
setTimeout(() => { console.error('tiempo agotado'); process.exit(1); }, 300000);
