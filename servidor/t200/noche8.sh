#!/bin/bash
# T207: T206 + tanda 2 (MGH/MF, Normal Sinus, CHF, ST Change, excluidos, IRIDIA-AF).
export HOLTERAI=/root/holter/holterai
P=/root/venv/bin/python
cd /root/holter/t200
L=/root/holter/t200/logs
log() { echo "$(date +%T) $*" | tee -a $L/orquesta.log; }
# etiquetas dudosas (diag_m.log): TSV por salvas de S en mghdb/chfdb y S de chfdb (anotacion automatica) -> sin etiqueta
$P - <<'PY'
import numpy as np
for t in ("n_mghdb", "n_chfdb"):
    q = np.load(f"data/{t}_tq.npz")["TQ"]; n = int((q[:, 1] == 1).sum()); q[q[:, 1] == 1, 1] = -1
    np.savez(f"data/{t}_tq.npz", TQ=q); print(t, "TSV+ enmascaradas", n)
m = dict(np.load("data/n_chfdb_meta.npz")); y = m["Y"]; n = int((y[:, 3] == 1).sum()); y[y[:, 3] == 1, 3] = -1
np.savez("data/n_chfdb_meta.npz", **m); print("chfdb PAC+ enmascaradas", n)
PY
TAGS=holter,icentia,taqui,icentia2,n_leipzig,n_vf,n_butqdb,n_nhdb,n_cpsc20,n_mghdb,n_nsrdb,n_chfdb,n_stdb,n_excluded,n_iridia
timeout 900 $P t202_train.py --name smoke207 --init runs/t206/best.pt --tags $TAGS --rob 0.5 --minutes 1 --every 0.5 --val_cap 1500 --workers 2 > $L/smoke207.log 2>&1
grep -q "FIN pasos" $L/smoke207.log || { log "T207: prueba corta fallo; sigue T206"; exit 1; }
pkill -f "t200/noche7.sh"; sleep 1; pkill -f "name t206"; sleep 3
cp runs/t206/best.pt runs/t206/best_final.pt
log "T206 detenido (mejor = min 30, best_final.pt). inicio T207 (tanda 2) desde T206"
( cd runs; last=""; sleep 120; while pgrep -f "name t207" >/dev/null; do m=$(stat -c %Y t207/last.pt 2>/dev/null); if [ -n "$m" ] && [ "$m" != "$last" ]; then sleep 20; n=$(python3 -c "import json;print(int(json.load(open('t207/hist.json'))[-1]['min']))"); cp t207/last.pt t207/min$n.pt; last=$m; fi; sleep 30; done ) &
$P t202_train.py --name t207 --init /root/holter/t200/runs/t206/best_final.pt --tags $TAGS --val_tags holter,icentia,taqui,n_leipzig,n_vf,n_butqdb,n_nhdb,n_cpsc20 --rob 0.5 --lr_bb 5e-6 --lr_head 2.5e-5 --minutes 600 --every 30 --paciencia 5 --workers 6 > $L/train_t207.log 2>&1
log "T207 entrenamiento rc=$?"
