#!/bin/bash
# T209: bajar S falsas (pedido de Jorge 19-nov 19:09). Desde T208, con mas peso a "no hay S" (x2, x4 si hay V),
# mas muestreo de ventanas con V sin S, y Long-Term ST (sin sus S positivas: dudosas en test) como negativos reales.
export HOLTERAI=/root/holter/holterai
P=/root/venv/bin/python
cd /root/holter/t200
L=/root/holter/t200/logs
log() { echo "$(date +%T) $*" | tee -a $L/orquesta.log; }
$P - <<'PY'
import numpy as np
m = dict(np.load("data/n_ltstdb_meta.npz")); y = m["Y"]; n = int((y[:, 3] == 1).sum()); y[y[:, 3] == 1, 3] = -1
np.savez("data/n_ltstdb_meta.npz", **m); print("ltstdb PAC+ enmascaradas", n)
q = np.load("data/n_ltstdb_tq.npz")["TQ"]; n = int((q[:, 1] == 1).sum()); q[q[:, 1] == 1, 1] = -1; np.savez("data/n_ltstdb_tq.npz", TQ=q); print("ltstdb TSV+ enmascaradas", n)
PY
TAGS=holter,icentia,taqui,icentia2,n_leipzig,n_vf,n_butqdb,n_nhdb,n_cpsc20,n_mghdb,n_nsrdb,n_chfdb,n_stdb,n_excluded,n_iridia,n_ltstdb
EXTRA="--s_neg_w 2 --s_negv_w 2 --s_samp 2"
timeout 900 $P t202_train.py --name smoke209 --init runs/t208/best.pt --tags $TAGS --rob 0.5 $EXTRA --minutes 1 --every 0.5 --val_cap 1500 --workers 2 > $L/smoke209.log 2>&1
grep -q "FIN pasos" $L/smoke209.log || { log "T209: prueba corta fallo; sigue T208"; exit 1; }
pkill -f "t200/noche9.sh"; sleep 1; pkill -f "name t208"; sleep 3
cp runs/t208/best.pt runs/t208/best_final.pt
log "T208 detenido. inicio T209 (menos S falsas) desde T208"
( cd runs; last=""; sleep 120; while pgrep -f "name t209" >/dev/null; do m=$(stat -c %Y t209/last.pt 2>/dev/null); if [ -n "$m" ] && [ "$m" != "$last" ]; then sleep 20; n=$(python3 -c "import json;print(int(json.load(open('t209/hist.json'))[-1]['min']))"); cp t209/last.pt t209/min$n.pt; last=$m; fi; sleep 30; done ) &
$P t202_train.py --name t209 --init /root/holter/t200/runs/t208/best_final.pt --tags $TAGS --val_tags holter,icentia,taqui,n_leipzig,n_vf,n_butqdb,n_nhdb,n_cpsc20 --rob 0.5 $EXTRA --lr_bb 5e-6 --lr_head 2.5e-5 --minutes 600 --every 30 --paciencia 6 --workers 6 > $L/train_t209.log 2>&1
log "T209 entrenamiento rc=$?"
