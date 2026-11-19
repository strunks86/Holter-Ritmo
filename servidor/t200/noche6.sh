#!/bin/bash
# T205b: como T205 pero sin Chapman (flutter dudoso + atajo de dominio) ni cudb; vfdb corregido.
export HOLTERAI=/root/holter/holterai
P=/root/venv/bin/python
cd /root/holter/t200
L=/root/holter/t200/logs
TAGS=holter,icentia,taqui,icentia2,n_leipzig,n_vf,n_butqdb,n_nhdb,n_cpsc20
log() { echo "$(date +%T) $*" | tee -a $L/orquesta.log; }
pkill -f "t200/noche5.sh"; sleep 1; pkill -f "name t205"; sleep 3
log "T205 detenido: empeoro la validacion Holter (0,856->0,816; TV, flutter). Diagnostico en logs/diag205.log"
nice -n 5 $P t205_build.py --tag n_vf --workers 4 > $L/build_n4.log 2>&1
$P t202_train.py --name t205b --init /root/holter/t200/runs/t204/best_final.pt --tags $TAGS --rob 0.5 --lr_bb 1e-5 --lr_head 5e-5 --minutes 600 --every 30 --paciencia 6 --workers 6 > $L/train_t205b.log 2>&1
log "T205b entrenamiento rc=$?"
