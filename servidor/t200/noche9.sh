#!/bin/bash
# T208: como T207 pero sin las S de MGH/MF (diag_t207.log: el promedio las ve con AUROC 0,90 y T207 las memoriza;
# en la prueba antigua las S falsas sobre V subieron 51 -> 60 por mil). Parte otra vez de T206.
export HOLTERAI=/root/holter/holterai
P=/root/venv/bin/python
cd /root/holter/t200
L=/root/holter/t200/logs
log() { echo "$(date +%T) $*" | tee -a $L/orquesta.log; }
$P - <<'PY'
import numpy as np
m = dict(np.load("data/n_mghdb_meta.npz")); y = m["Y"]; n = int((y[:, 3] == 1).sum()); y[y[:, 3] == 1, 3] = -1
np.savez("data/n_mghdb_meta.npz", **m); print("mghdb PAC+ enmascaradas", n)
PY
TAGS=holter,icentia,taqui,icentia2,n_leipzig,n_vf,n_butqdb,n_nhdb,n_cpsc20,n_mghdb,n_nsrdb,n_chfdb,n_stdb,n_excluded,n_iridia
pkill -f "t200/noche8.sh"; sleep 1; pkill -f "name t207"; sleep 3
log "T207 detenido (mejor = min 120). inicio T208 desde T206 sin S de MGH/MF"
( cd runs; last=""; sleep 120; while pgrep -f "name t208" >/dev/null; do m=$(stat -c %Y t208/last.pt 2>/dev/null); if [ -n "$m" ] && [ "$m" != "$last" ]; then sleep 20; n=$(python3 -c "import json;print(int(json.load(open('t208/hist.json'))[-1]['min']))"); cp t208/last.pt t208/min$n.pt; last=$m; fi; sleep 30; done ) &
$P t202_train.py --name t208 --init /root/holter/t200/runs/t206/best_final.pt --tags $TAGS --val_tags holter,icentia,taqui,n_leipzig,n_vf,n_butqdb,n_nhdb,n_cpsc20 --rob 0.5 --lr_bb 5e-6 --lr_head 2.5e-5 --minutes 600 --every 30 --paciencia 5 --workers 6 > $L/train_t208.log 2>&1
log "T208 entrenamiento rc=$?"
