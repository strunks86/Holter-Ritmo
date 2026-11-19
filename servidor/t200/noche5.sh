#!/bin/bash
# T205: T204 + bases nuevas (Leipzig, vfdb/cudb, BUT QDB, Noise Holter DB, CPSC2020, Chapman).
export HOLTERAI=/root/holter/holterai
P=/root/venv/bin/python
cd /root/holter/t200
L=/root/holter/t200/logs
TAGS=holter,icentia,taqui,icentia2,n_leipzig,n_vf,n_butqdb,n_nhdb,n_cpsc20,n_chapman
log() { echo "$(date +%T) $*" | tee -a $L/orquesta.log; }
for i in $(seq 1 240); do grep -q "LISTO n_chapman" $L/build_n3.log && break; pgrep -f t205_build > /dev/null || break; sleep 15; done
for t in n_leipzig n_vf n_butqdb n_nhdb n_cpsc20 n_chapman; do [ -f data/${t}_tq.npz ] || log "T205: falta $t (se entrena sin el)"; done
timeout 900 $P t202_train.py --name smoke205 --init runs/t204/best.pt --tags $TAGS --rob 0.5 --minutes 1 --every 0.5 --val_cap 1500 --workers 2 > $L/smoke205.log 2>&1
grep -q "FIN pasos" $L/smoke205.log || { log "T205: prueba corta fallo; sigue T204"; exit 1; }
pkill -f "t200/noche4.sh"; sleep 1; pkill -f "name t204"; sleep 5
cp runs/t204/best.pt runs/t204/best_final.pt
log "T204 detenido (mejor guardado en best_final.pt). inicio T205 (bases nuevas) desde T204"
$P t202_train.py --name t205 --init /root/holter/t200/runs/t204/best_final.pt --tags $TAGS --rob 0.5 --lr_bb 1e-5 --lr_head 5e-5 --minutes 600 --every 30 --paciencia 6 --workers 6 > $L/train_t205.log 2>&1
log "T205 entrenamiento rc=$?"
